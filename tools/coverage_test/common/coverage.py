#!/usr/bin/env python3
"""Incremental coverage analysis for git patches and commits.

LCOV and unified-diff parsing (``PathMapper``, ``parse_lcov_file``,
``merge_lcov_data``, ``load_and_merge_lcov``, ``parse_unified_diff``) and the
incremental line classification (``analyze_incremental_coverage``) are ported
from ``tools/incremental_coverage/analysis.py`` in the ``dev_2.15.0_gtest``
branch, so path normalization, merge semantics and the coverage denominator
stay consistent with that implementation.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

HUNK_RE = re.compile(
    r"^@@ -(?P<old>\d+)(?:,(?P<old_count>\d+))? \+(?P<new>\d+)(?:,(?P<new_count>\d+))? @@"
)

# Only these file kinds are meaningful for C/C++ coverage. Everything else in a
# git diff (BUILD files, *.py wrappers, *.pbtxt api_def, docs, *.lcov output,
# .gitignore, etc.) must be ignored.
CPP_EXTENSIONS = {
    ".c", ".cc", ".cp", ".cpp", ".cxx",
    ".h", ".hh", ".hpp", ".hxx", ".inc",
}

# Ported from tools/incremental_coverage/config/excludes.txt. These rules only
# affect the coverage report; test binaries still build and run.
DEFAULT_EXCLUDES = (
    "*test.cc", "*_test.cc", "*_test.cpp", "*_test.c", "*_test.h", "*_test.hpp",
    "*_benchmark.cc",
    "test/*", "*/test/*", "tests/*", "*/tests/*", "testing/*", "*/testing/*",
)


class CoverageToolError(RuntimeError):
    """A user-actionable failure in the coverage workflow."""


def is_cpp_source(path: str) -> bool:
    """Return True if ``path`` is a C/C++ source or header file."""
    return PurePosixPath(path).suffix.lower() in CPP_EXTENSIONS


def is_excluded(path: str, patterns: Iterable[str]) -> tuple[bool, str | None]:
    """Apply the basename/full-relative-path glob semantics of the original tool.

    A pattern without ``/`` matches the basename; a pattern with ``/`` matches
    the full repository-relative path.
    """
    normalized = path.replace("\\", "/").lstrip("./")
    basename = PurePosixPath(normalized).name
    for pattern in patterns:
        pattern = str(pattern).strip()
        if not pattern:
            continue
        candidate = normalized if "/" in pattern else basename
        if fnmatch.fnmatchcase(candidate, pattern):
            return True, pattern
    return False, None


def _decode_diff_path(value: str) -> str:
    """Decode Git's optional C-style quoted path and strip a/ or b/ prefix."""
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = ast.literal_eval(value)
            if isinstance(decoded, str):
                value = decoded
        except (SyntaxError, ValueError):
            pass
    if value.startswith(("a/", "b/")):
        value = value[2:]
    value = value.replace("\\", "/")
    return value[2:] if value.startswith("./") else value


def _header_path(value: str) -> str | None:
    value = value.strip()
    if value == "/dev/null":
        return None
    return _decode_diff_path(value.split("\t", 1)[0])


def parse_unified_diff(diff: str | Path) -> dict:
    """Parse final-side changed line numbers from a unified diff.

    Ported from ``analysis.parse_unified_diff``. Added and modified lines both
    appear as ``+`` lines; deleted lines are retained as metadata but have no
    executable final location.
    """
    if isinstance(diff, Path):
        content = diff.read_text(encoding="utf-8", errors="replace")
    else:
        content = diff

    entries: dict[str, dict] = {}
    diagnostics: list[str] = []
    old_path: str | None = None
    new_path: str | None = None
    current: dict | None = None
    new_line: int | None = None
    old_line: int | None = None
    pending_binary_paths: tuple[str | None, str | None] = (None, None)

    def ensure_entry(path: str) -> dict:
        entry = entries.get(path)
        if entry is None:
            entry = {
                "path": path,
                "old_path": old_path,
                "added_lines": set(),
                "deleted_lines": set(),
                "binary": False,
            }
            entries[path] = entry
        elif entry.get("old_path") is None and old_path is not None:
            entry["old_path"] = old_path
        return entry

    for raw in content.splitlines():
        if raw.startswith("diff --git "):
            pending_binary_paths = (None, None)
            old_path = None
            new_path = None
            current = None
            new_line = None
            old_line = None
            fields = raw.split(" ", 3)
            if len(fields) == 4:
                try:
                    pair = ast.literal_eval("[" + fields[2].replace(" ", ", ", 1) + "]")
                    if isinstance(pair, list) and len(pair) == 2:
                        pending_binary_paths = (_decode_diff_path(pair[0]), _decode_diff_path(pair[1]))
                except (SyntaxError, ValueError):
                    pass
            continue
        if raw.startswith("--- "):
            old_path = _header_path(raw[4:])
            current = None
            new_line = None
            old_line = None
            continue
        if raw.startswith("+++ ") and new_line is None:
            new_path = _header_path(raw[4:])
            current = ensure_entry(new_path) if new_path else None
            new_line = None
            old_line = None
            continue
        if raw.startswith("Binary files ") or raw.startswith("GIT binary patch"):
            binary_path = new_path or pending_binary_paths[1] or old_path or pending_binary_paths[0]
            if binary_path:
                entry = ensure_entry(binary_path)
                entry["binary"] = True
            continue
        match = HUNK_RE.match(raw)
        if match:
            if current is None and new_path:
                current = ensure_entry(new_path)
            new_line = int(match.group("new"))
            old_line = int(match.group("old"))
            continue
        if current is None or new_line is None:
            continue
        if raw.startswith("+"):
            current["added_lines"].add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            if old_line is not None:
                current["deleted_lines"].add(old_line)
                old_line += 1
        elif raw.startswith("\\ No newline at end of file"):
            continue
        else:
            new_line += 1
            if old_line is not None:
                old_line += 1

    files: list[dict] = []
    for path, item in sorted(entries.items()):
        if not path:
            diagnostics.append("ignored diff entry without a final path")
            continue
        files.append({
            "path": path,
            "old_path": item["old_path"],
            "added_lines": sorted(item["added_lines"]),
            "deleted_lines": sorted(line for line in item["deleted_lines"] if line),
            "binary": bool(item["binary"]),
        })
    return {"schema_version": 1, "files": files, "diagnostics": diagnostics}


class PathMapper:
    """Map LCOV ``SF:`` values to validated repository-relative paths.

    Ported from ``analysis.PathMapper``. A path is accepted only when it
    resolves to an existing source file in the repository, avoiding unsafe
    string truncation of arbitrary paths containing ``/tensorflow/``.
    """

    def __init__(self, source_root: Path, *, repository_root: Path | None = None):
        self.source_root = source_root.resolve()
        self.repository_root = repository_root.resolve() if repository_root else None

    def _relative_if_source(self, candidate: Path) -> str | None:
        try:
            relative = candidate.resolve().relative_to(self.source_root)
        except ValueError:
            return None
        if candidate.is_file():
            return relative.as_posix()
        return None

    def normalize(self, source: str) -> str | None:
        raw = source.strip().replace("\\", "/")
        if not raw:
            return None
        source_path = Path(raw)
        if source_path.is_absolute():
            mapped = self._relative_if_source(source_path)
            if mapped:
                return mapped
            if self.repository_root:
                try:
                    relative = source_path.resolve().relative_to(self.repository_root)
                    candidate = self.source_root / relative
                    if candidate.is_file():
                        return relative.as_posix()
                except ValueError:
                    pass
            pieces = source_path.parts
            if "execroot" in pieces:
                index = pieces.index("execroot")
                if len(pieces) > index + 2:
                    relative = Path(*pieces[index + 2:])
                    candidate = self.source_root / relative
                    if candidate.is_file():
                        return relative.as_posix()
            for marker, root in (
                ("/external/local_tsl/tsl/", "third_party/xla/third_party/tsl/tsl"),
                ("/external/local_xla/", "third_party/xla"),
            ):
                if marker in raw:
                    relative = Path(root) / raw.split(marker, 1)[1]
                    mapped = self._relative_if_source(self.source_root / relative)
                    if mapped:
                        return mapped
            return None

        relative = Path(raw.lstrip("./"))
        candidate = self.source_root / relative
        if candidate.is_file():
            return relative.as_posix()
        for prefix, root in (
            ("external/local_tsl/", "third_party/xla/third_party/tsl"),
            ("external/local_xla/", "third_party/xla"),
        ):
            if raw.startswith(prefix):
                mapped = self._relative_if_source(self.source_root / root / raw.removeprefix(prefix))
                if mapped:
                    return mapped
        return None


def parse_lcov_file(path: Path, *, mapper: PathMapper) -> dict:
    """Read an LCOV file and merge duplicate source records by line hit count.

    Ported from ``analysis.parse_lcov_file``.
    """
    records: dict[str, dict[int, int]] = {}
    diagnostics: list[dict] = []
    source: str | None = None
    counts: dict[int, int] = {}
    source_line = 0

    def flush() -> None:
        nonlocal source, counts
        if source is None:
            counts = {}
            return
        normalized = mapper.normalize(source)
        if normalized is None:
            diagnostics.append({"kind": "unmapped_source", "source": source, "report": str(path)})
        else:
            target = records.setdefault(normalized, {})
            for line, hits in counts.items():
                target[line] = target.get(line, 0) + hits
        source = None
        counts = {}

    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError as error:
        raise CoverageToolError(f"cannot read LCOV file {path}: {error}") from error
    with stream:
        for source_line, raw in enumerate(stream, 1):
            line = raw.rstrip("\n")
            if line.startswith("SF:"):
                flush()
                source = line[3:]
                counts = {}
            elif line.startswith("DA:"):
                if source is None:
                    diagnostics.append({"kind": "orphan_da", "line": source_line, "report": str(path)})
                    continue
                pieces = line[3:].split(",", 2)
                if len(pieces) < 2:
                    diagnostics.append({"kind": "invalid_da", "line": source_line, "report": str(path)})
                    continue
                try:
                    number = int(pieces[0])
                    hits = int(pieces[1])
                    if number <= 0 or hits < 0:
                        raise ValueError
                except ValueError:
                    diagnostics.append({"kind": "invalid_da", "line": source_line, "report": str(path)})
                    continue
                counts[number] = counts.get(number, 0) + hits
            elif line == "end_of_record":
                flush()
    flush()
    return {
        "schema_version": 1,
        "records": {key: dict(sorted(value.items())) for key, value in sorted(records.items())},
        "diagnostics": diagnostics,
        "inputs": [str(path)],
    }


def merge_lcov_data(reports: Iterable[Mapping]) -> dict:
    """Merge several parsed LCOV reports, summing hit counts per line."""
    records: dict[str, dict[int, int]] = {}
    diagnostics: list[dict] = []
    inputs: list[str] = []
    for report in reports:
        inputs.extend(str(item) for item in report.get("inputs", []))
        diagnostics.extend(dict(item) for item in report.get("diagnostics", []))
        for path, counts in report.get("records", {}).items():
            target = records.setdefault(str(path), {})
            for number, hits in counts.items():
                line = int(number)
                target[line] = target.get(line, 0) + int(hits)
    return {
        "schema_version": 1,
        "records": {path: dict(sorted(counts.items())) for path, counts in sorted(records.items())},
        "diagnostics": diagnostics,
        "inputs": inputs,
    }


def load_and_merge_lcov(
    paths: Iterable[Path], *, source_root: Path, repository_root: Path | None = None
) -> dict:
    mapper = PathMapper(source_root, repository_root=repository_root)
    return merge_lcov_data(parse_lcov_file(path, mapper=mapper) for path in paths)


def _non_executable_lines(lines: Sequence[str]) -> dict[int, str]:
    """Return only lines that are definitely blank/comment/preprocessor lines.

    Anything remotely ambiguous stays absent so missing LCOV data becomes
    ``unknown`` instead of being silently removed from the denominator.
    """
    result: dict[int, str] = {}
    in_block_comment = False
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            result[number] = "blank"
            continue
        if not in_block_comment and line.lstrip().startswith("#"):
            result[number] = "preprocessor"
            continue

        visible: list[str] = []
        saw_comment = False
        index = 0
        string_delimiter: str | None = None
        while index < len(line):
            if in_block_comment:
                saw_comment = True
                close = line.find("*/", index)
                if close < 0:
                    index = len(line)
                    break
                in_block_comment = False
                index = close + 2
                continue
            if string_delimiter:
                visible.append("x")
                break
            if line.startswith("//", index):
                saw_comment = True
                break
            if line.startswith("/*", index):
                in_block_comment = True
                saw_comment = True
                index += 2
                continue
            char = line[index]
            if char in {'"', "'"}:
                string_delimiter = char
                visible.append("x")
                break
            if not char.isspace():
                visible.append(char)
            index += 1
        if not visible and saw_comment:
            result[number] = "comment"
    return result


def _state_counts(lines: Iterable[Mapping]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in lines:
        counts[str(item["state"])] += 1
    return dict(sorted(counts.items()))


def _rate(covered: int, executable: int) -> float | None:
    return None if not executable else round(covered * 100.0 / executable, 4)


def _directory_summaries(files: Sequence[Mapping]) -> list[dict]:
    """Roll up filtered files once per ancestor, including repository root (.).

    Ported from ``analysis._directory_summaries``. A directory total includes
    all files in its subdirectories; parent and child rows overlap and must
    not be added together.
    """
    directories: dict[str, dict] = {}
    for file_data in files:
        for directory in PurePosixPath(str(file_data["path"])).parents:
            path = directory.as_posix()
            item = directories.setdefault(path, {
                "path": path,
                "parent": None if path == "." else directory.parent.as_posix(),
                "changed_files": 0,
                "changed_lines": 0,
                "counts": {},
            })
            item["changed_files"] += 1
            item["changed_lines"] += file_data["added_lines"]
            for state, count in file_data["counts"].items():
                item["counts"][state] = item["counts"].get(state, 0) + count

    results = []
    for path in sorted(directories, key=lambda value: PurePosixPath(value).parts):
        item = directories[path]
        counts = dict(sorted(item["counts"].items()))
        covered = counts.get("covered", 0)
        uncovered = counts.get("uncovered", 0)
        unknown = counts.get("unknown", 0)
        executable = covered + uncovered
        item.update({
            "counts": counts,
            "covered_lines": covered,
            "uncovered_lines": uncovered,
            "executable_lines": executable,
            "unmeasured_lines": unknown,
            "coverage_complete": unknown == 0,
            "coverage_rate": _rate(covered, executable),
            "denominator_reason": (
                None if executable else
                "no LCOV line data for the changed lines in this directory" if unknown else
                "no executable changed lines in this directory for the selected configuration"
            ),
        })
        results.append(item)
    return results


def analyze_incremental_coverage(
    changes: Mapping,
    lcov: Mapping,
    *,
    source_root: Path,
    exclude_patterns: Sequence[str] = (),
) -> dict:
    """Classify every final-side changed line using explicit LCOV semantics.

    Ported from ``analysis.analyze_incremental_coverage``. Coverage is
    ``covered / (covered + uncovered)``, i.e. over changed lines that have an
    LCOV ``DA`` record. Blank/comment/preprocessor lines are ``non_executable``
    and lines whose executable status cannot be confirmed are ``unknown``;
    neither is silently folded into the denominator.
    """
    source_root = source_root.resolve()
    records = {
        str(path): {int(number): int(hits) for number, hits in counts.items()}
        for path, counts in lcov.get("records", {}).items()
    }
    output_files: list[dict] = []
    excluded_files: list[dict] = []

    for change in sorted(changes.get("files", []), key=lambda item: str(item["path"])):
        path = str(change["path"]).replace("\\", "/").lstrip("./")
        excluded, matched_rule = is_excluded(path, exclude_patterns)
        if excluded or not is_cpp_source(path) or bool(change.get("binary")):
            reason = (
                f"matched exclusion rule {matched_rule!r}" if excluded else
                "outside C/C++ coverage scope" if not is_cpp_source(path) else
                "binary file"
            )
            excluded_files.append({"path": path, "reason": reason})
            continue

        added = sorted({int(number) for number in change.get("added_lines", [])})
        source_path = source_root / path
        source_lines: list[str] | None = None
        source_error: str | None = None
        if source_path.is_file():
            try:
                source_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as error:
                source_error = str(error)
        else:
            source_error = "source file is unavailable in the repository"

        non_executable = _non_executable_lines(source_lines or [])
        coverage_lines = records.get(path, {})
        classified: list[dict] = []
        for number in added:
            item: dict = {"line": number, "hits": None}
            if number in coverage_lines:
                hits = coverage_lines[number]
                item.update({"state": "covered" if hits > 0 else "uncovered", "hits": hits})
            elif number in non_executable:
                item.update({"state": "non_executable", "reason": non_executable[number]})
            else:
                reason = source_error or (
                    "source file has no LCOV record" if path not in records else "line has no LCOV DA record"
                )
                item.update({"state": "unknown", "reason": reason})
            classified.append(item)

        counts = _state_counts(classified)
        executable = counts.get("covered", 0) + counts.get("uncovered", 0)
        output_files.append({
            "path": path,
            "added_lines": len(added),
            "line_results": classified,
            "counts": counts,
            "executable_lines": executable,
            "covered_lines": counts.get("covered", 0),
            "uncovered_lines": counts.get("uncovered", 0),
            "unmeasured_lines": counts.get("unknown", 0),
            "coverage_complete": counts.get("unknown", 0) == 0,
            "coverage_rate": _rate(counts.get("covered", 0), executable),
        })

    all_lines = [line for file_data in output_files for line in file_data["line_results"]]
    counts = _state_counts(all_lines)
    executable = counts.get("covered", 0) + counts.get("uncovered", 0)
    summary = {
        "changed_files": len(output_files),
        "changed_lines": len(all_lines),
        "counts": counts,
        "executable_lines": executable,
        "covered_lines": counts.get("covered", 0),
        "uncovered_lines": counts.get("uncovered", 0),
        "unmeasured_lines": counts.get("unknown", 0),
        "coverage_rate": _rate(counts.get("covered", 0), executable),
        "coverage_complete": counts.get("unknown", 0) == 0,
    }
    return {
        "schema_version": 1,
        "summary": summary,
        "files": output_files,
        "directories": _directory_summaries(output_files),
        "exclude_patterns": list(exclude_patterns),
        "excluded_files": excluded_files,
        "lcov_diagnostics": list(lcov.get("diagnostics", [])),
        "lcov_inputs": list(lcov.get("inputs", [])),
    }


class IncrementalCoverageAnalyzer:
    """Analyze incremental coverage for git changes."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def _git_diff(self, args: Sequence[str]) -> dict:
        try:
            result = subprocess.run(
                list(args), cwd=self.repo_root, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get diff: {e.stderr}")
        return parse_unified_diff(result.stdout)

    def analyze_patch_coverage(
        self, patch_file: Path, lcov_files: Sequence[Path]
    ) -> dict:
        """Analyze coverage for a patch file."""
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_file}")
        return self._analyze(parse_unified_diff(patch_file), lcov_files)

    def diff_files(self, base_ref: str, target_ref: str) -> list[dict]:
        """Return the changed-file list for ``base_ref..target_ref``."""
        return self._git_diff(["git", "diff", base_ref, target_ref, "--unified=0"])["files"]

    def commit_files(self, commit_ref: str) -> list[dict]:
        """Return the changed-file list introduced by ``commit_ref``."""
        return self._git_diff(["git", "show", commit_ref, "--unified=0"])["files"]

    def analyze_commit_coverage(
        self, commit_ref: str, lcov_files: Sequence[Path]
    ) -> dict:
        """Analyze coverage for a git commit."""
        diff = self._git_diff(["git", "show", commit_ref, "--unified=0"])
        return self._analyze(diff, lcov_files)

    def analyze_diff_coverage(
        self, base_ref: str, target_ref: str, lcov_files: Sequence[Path]
    ) -> dict:
        """Analyze coverage for a git diff."""
        diff = self._git_diff(["git", "diff", base_ref, target_ref, "--unified=0"])
        return self._analyze(diff, lcov_files)

    def _analyze(self, diff: Mapping, lcov_files: Sequence[Path]) -> dict:
        lcov = load_and_merge_lcov(list(lcov_files), source_root=self.repo_root)
        return analyze_incremental_coverage(
            diff,
            lcov,
            source_root=self.repo_root,
            exclude_patterns=list(DEFAULT_EXCLUDES),
        )
