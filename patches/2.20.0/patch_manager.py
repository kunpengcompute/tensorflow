#!/usr/bin/env python3
"""Audit, generate and verify the TensorFlow feature patch series.

The development branch remains a full-source view.  This tool compares one
committed source ref with the exact upstream TensorFlow base and assigns each
changed path to one and only one patch group from manifest.json.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest.json"
DEFAULT_PATCH_DIR = SCRIPT_DIR / "feature"


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Group:
    name: str
    output: str
    description: str
    patterns: tuple[str, ...]
    feature_copts: tuple["FeatureCopt", ...] = ()


@dataclass(frozen=True)
class FeatureCopt:
    load: str
    symbol: str


@dataclass(frozen=True)
class GeneratedFile:
    path: str
    owner_group: str
    kind: str
    function: str


@dataclass(frozen=True)
class Profile:
    name: str
    groups: tuple[str, ...]
    full_source: bool = False


@dataclass(frozen=True)
class Manifest:
    base_ref: str
    groups: tuple[Group, ...]
    profiles: tuple[Profile, ...]
    tooling_patterns: tuple[str, ...]
    generated_files: tuple[GeneratedFile, ...]


def run_git(
    args: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> str:
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )
    if result.returncode != 0:
        detail_parts = []
        if capture and result.stdout:
            detail_parts.append(result.stdout.strip())
        if capture and result.stderr:
            detail_parts.append(result.stderr.strip())
        detail = "\n".join(filter(None, detail_parts))
        raise PatchError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result.stdout if capture else ""


def worktree_source(manifest: Manifest) -> str:
    """Write the current worktree to a temporary Git tree without changing the index.

    Untracked files are included only when they belong to a declared patch group
    or to the release tooling. This deliberately leaves unrelated local files
    out of generated artifacts.
    """
    descriptor, index_name = tempfile.mkstemp(prefix="tensorflow-patch-index-")
    os.close(descriptor)
    index_path = Path(index_name)
    index_path.unlink()
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        run_git(["read-tree", "HEAD"], env=environment)
        run_git(["add", "-u", "--", "."], env=environment)
        untracked = run_git(
            ["ls-files", "--others", "--exclude-standard"], env=environment
        ).splitlines()
        eligible = [
            path
            for path in untracked
            if matches(path, manifest.tooling_patterns)
            or path in generated_paths(manifest)
            or any(matches(path, group.patterns) for group in manifest.groups)
        ]
        if eligible:
            run_git(["add", "--", *eligible], env=environment)
        return run_git(["write-tree"], env=environment).strip()
    finally:
        index_path.unlink(missing_ok=True)


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise PatchError(f"unsupported manifest schema in {path}")
    groups = tuple(
        Group(
            name=item["name"],
            output=item["output"],
            description=item.get("description", ""),
            patterns=tuple(item["patterns"]),
            feature_copts=tuple(
                FeatureCopt(load=entry["load"], symbol=entry["symbol"])
                for entry in item.get("feature_copts", ())
            ),
        )
        for item in data["groups"]
    )
    names = [group.name for group in groups]
    outputs = [group.output for group in groups]
    if len(names) != len(set(names)):
        raise PatchError("patch group names must be unique")
    if len(outputs) != len(set(outputs)):
        raise PatchError("patch output names must be unique")
    profiles = tuple(
        Profile(
            name=item["name"],
            groups=tuple(item["groups"]),
            full_source=item.get("full_source", False),
        )
        for item in data.get("profiles", ())
    )
    known_groups = set(names)
    for profile in profiles:
        unknown = set(profile.groups) - known_groups
        if unknown:
            raise PatchError(
                f"profile {profile.name} contains unknown groups: {sorted(unknown)}"
            )
    generated_files = tuple(
        GeneratedFile(
            path=item["path"],
            owner_group=item["owner_group"],
            kind=item["kind"],
            function=item["function"],
        )
        for item in data.get("generated_files", ())
    )
    unknown_generated_owners = {
        generated.owner_group
        for generated in generated_files
        if generated.owner_group not in known_groups
    }
    if unknown_generated_owners:
        raise PatchError(
            "generated files reference unknown owner groups: "
            f"{sorted(unknown_generated_owners)}"
        )
    return Manifest(
        base_ref=data["base_ref"],
        groups=groups,
        profiles=profiles,
        tooling_patterns=tuple(data.get("tooling_patterns", ())),
        generated_files=generated_files,
    )


def matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def generated_paths(manifest: Manifest) -> set[str]:
    return {generated.path for generated in manifest.generated_files}


def changed_paths(base_ref: str, source_ref: str) -> list[str]:
    output = run_git(
        [
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            base_ref,
            source_ref,
            "--",
        ]
    )
    return sorted(filter(None, output.splitlines()))


def classify(
    manifest: Manifest, paths: Iterable[str]
) -> tuple[dict[str, list[str]], list[str], dict[str, list[str]]]:
    owned = {group.name: [] for group in manifest.groups}
    unowned: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    generated = generated_paths(manifest)
    for path in paths:
        if matches(path, manifest.tooling_patterns) or path in generated:
            continue
        owners = [group.name for group in manifest.groups if matches(path, group.patterns)]
        if not owners:
            unowned.append(path)
        elif len(owners) > 1:
            ambiguous[path] = owners
        else:
            owned[owners[0]].append(path)
    return owned, unowned, ambiguous


def audit(manifest: Manifest, source_ref: str, *, quiet: bool = False) -> dict[str, list[str]]:
    paths = changed_paths(manifest.base_ref, source_ref)
    owned, unowned, ambiguous = classify(manifest, paths)
    if not quiet:
        print(f"base:   {manifest.base_ref}")
        print(f"source: {source_ref}")
        print(f"changed paths: {len(paths)}")
        for group in manifest.groups:
            print(f"  {group.name:12s} {len(owned[group.name]):4d}")
    if unowned:
        print("\nunowned changed paths:", file=sys.stderr)
        for path in unowned:
            print(f"  {path}", file=sys.stderr)
    if ambiguous:
        print("\npaths owned by multiple groups:", file=sys.stderr)
        for path, owners in ambiguous.items():
            print(f"  {path}: {', '.join(owners)}", file=sys.stderr)
    if unowned or ambiguous:
        raise PatchError("patch ownership audit failed")
    return owned


def generate_patch(base_ref: str, source_ref: str, paths: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        output.write_text("", encoding="utf-8")
        return
    command = [
        "git",
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-renames",
        base_ref,
        source_ref,
        "--",
        *paths,
    ]
    with output.open("wb") as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, stdout=stream)
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise PatchError(f"failed to generate {output.name}")


def render_feature_copts(function: str, entries: Sequence[FeatureCopt]) -> str:
    lines = [
        "# Generated by patches/patch_manager.py from patches/manifest.json.",
        "# Do not edit this file directly.",
        "",
    ]
    for entry in entries:
        lines.append(f'load("{entry.load}", "{entry.symbol}")')
    if entries:
        lines.append("")
    lines.extend(
        [
            f"def {function}():",
            "    return (",
        ]
    )
    for entry in entries:
        lines.append(f"        {entry.symbol}() +")
    lines.extend(
        [
            "        []",
            "    )",
            "",
        ]
    )
    return "\n".join(lines)


def render_generated_file(generated: GeneratedFile, entries: Sequence[FeatureCopt]) -> str:
    if generated.kind != "feature_copts":
        raise PatchError(f"unsupported generated file kind: {generated.kind}")
    return render_feature_copts(generated.function, entries)


def patch_new_text_file(relative: str, content: str) -> str:
    old_line = "--- /dev/null"
    new_line = f"+++ b/{relative}"
    body = "".join(f"+{line}\n" for line in content.splitlines())
    line_count = len(content.splitlines())
    return (
        f"diff --git a/{relative} b/{relative}\n"
        "new file mode 100644\n"
        f"{old_line}\n"
        f"{new_line}\n"
        f"@@ -0,0 +1,{line_count} @@\n"
        f"{body}"
    )


def append_generated_files(manifest: Manifest, output_dir: Path) -> None:
    groups_by_name = {group.name: group for group in manifest.groups}
    for generated in manifest.generated_files:
        owner = groups_by_name[generated.owner_group]
        output = output_dir / owner.output
        content = render_generated_file(generated, ())
        with output.open("a", encoding="utf-8") as stream:
            if output.stat().st_size:
                stream.write("\n")
            stream.write(patch_new_text_file(generated.path, content))


def generate(manifest: Manifest, source_ref: str, output_dir: Path) -> None:
    owned = audit(manifest, source_ref)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {group.output for group in manifest.groups}
    for stale in output_dir.glob("*.patch"):
        if stale.name not in expected:
            stale.unlink()
    for group in manifest.groups:
        output = output_dir / group.output
        generate_patch(manifest.base_ref, source_ref, owned[group.name], output)
        print(f"generated {output} ({len(owned[group.name])} paths)")
    append_generated_files(manifest, output_dir)


def verify_file_contents(
    source_ref: str, owned: dict[str, list[str]], checkout: Path
) -> None:
    for paths in owned.values():
        for relative in paths:
            expected_exists = subprocess.run(
                ["git", "cat-file", "-e", f"{source_ref}:{relative}"],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            actual = checkout / relative
            if not expected_exists:
                if actual.exists():
                    raise PatchError(f"deleted path still exists after patching: {relative}")
                continue
            expected = subprocess.run(
                ["git", "show", f"{source_ref}:{relative}"],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            if not actual.is_file() or actual.read_bytes() != expected:
                raise PatchError(f"patched content differs from {source_ref}: {relative}")


def feature_copts_for_profile(
    manifest: Manifest, profile: Profile
) -> tuple[FeatureCopt, ...]:
    groups_by_name = {group.name: group for group in manifest.groups}
    entries: list[FeatureCopt] = []
    seen_symbols: set[str] = set()
    for group_name in profile.groups:
        for entry in groups_by_name[group_name].feature_copts:
            if entry.symbol in seen_symbols:
                raise PatchError(f"duplicate feature_copts symbol: {entry.symbol}")
            seen_symbols.add(entry.symbol)
            entries.append(entry)
    return tuple(entries)


def write_generated_files(
    manifest: Manifest, profile: Profile, checkout: Path
) -> None:
    entries = feature_copts_for_profile(manifest, profile)
    for generated in manifest.generated_files:
        target = checkout / generated.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_generated_file(generated, entries), encoding="utf-8")


def verify_generated_file_contents(
    manifest: Manifest, profile: Profile, source_ref: str, checkout: Path
) -> None:
    entries = feature_copts_for_profile(manifest, profile)
    for generated in manifest.generated_files:
        expected = render_generated_file(generated, entries).encode()
        actual = checkout / generated.path
        if not actual.is_file() or actual.read_bytes() != expected:
            raise PatchError(f"generated file content is wrong: {generated.path}")
        source = subprocess.run(
            ["git", "show", f"{source_ref}:{generated.path}"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if source != expected:
            raise PatchError(
                f"{source_ref}:{generated.path} does not match generated "
                f"profile {profile.name}"
            )


def verify_profile(
    manifest: Manifest,
    profile: Profile,
    source_ref: str,
    patch_dir: Path,
    owned: dict[str, list[str]],
) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix=f"tensorflow-{profile.name}-"))
    checkout = temp_root / "checkout"
    try:
        run_git(["worktree", "add", "--detach", str(checkout), manifest.base_ref])
        apply_profile_patches(manifest, profile, patch_dir, checkout)
        write_generated_files(manifest, profile, checkout)
        if profile.full_source:
            verify_file_contents(source_ref, owned, checkout)
            verify_generated_file_contents(manifest, profile, source_ref, checkout)
        print(f"verified profile {profile.name}: {', '.join(profile.groups)}")
    finally:
        if checkout.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shutil.rmtree(temp_root, ignore_errors=True)


def profile_by_name(manifest: Manifest, profile_name: str) -> Profile:
    for profile in manifest.profiles:
        if profile.name == profile_name:
            return profile
    raise PatchError(f"unknown profile: {profile_name}")


def apply_profile_patches(
    manifest: Manifest, profile: Profile, patch_dir: Path, checkout: Path
) -> None:
    groups_by_name = {group.name: group for group in manifest.groups}
    for group_name in profile.groups:
        group = groups_by_name[group_name]
        patch = patch_dir / group.output
        if not patch.is_file():
            raise PatchError(f"missing generated patch: {patch}")
        if patch.stat().st_size == 0:
            continue
        run_git(["apply", "--check", str(patch)], cwd=checkout)
        run_git(["apply", str(patch)], cwd=checkout)


def verify(manifest: Manifest, source_ref: str, patch_dir: Path) -> None:
    owned = audit(manifest, source_ref)
    profiles = manifest.profiles or (
        Profile(
            name="full",
            groups=tuple(group.name for group in manifest.groups),
            full_source=True,
        ),
    )
    if sum(profile.full_source for profile in profiles) != 1:
        raise PatchError("exactly one verification profile must set full_source=true")
    for profile in profiles:
        verify_profile(manifest, profile, source_ref, patch_dir, owned)
    print("all profiles apply cleanly; the full profile matches the source ref")


def materialize_profile(
    manifest: Manifest,
    profile_name: str,
    patch_dir: Path,
    checkout: Path,
    *,
    force: bool = False,
) -> None:
    profile = profile_by_name(manifest, profile_name)
    if checkout.exists():
        if not force:
            raise PatchError(f"checkout already exists: {checkout}")
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(checkout)],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(checkout, ignore_errors=True)
    run_git(["worktree", "add", "--detach", str(checkout), manifest.base_ref])
    try:
        apply_profile_patches(manifest, profile, patch_dir, checkout)
        write_generated_files(manifest, profile, checkout)
    except Exception:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(checkout)],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(checkout, ignore_errors=True)
        raise
    print(
        f"created TensorFlow source tree: {checkout} "
        f"(feature set: {profile.name})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source",
        default="HEAD",
        help="committed source ref to publish, or WORKTREE for local validation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="check that every changed path has one owner")
    generate_parser = subparsers.add_parser("generate", help="generate the patch series")
    generate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATCH_DIR)
    verify_parser = subparsers.add_parser("verify", help="apply and compare generated patches")
    verify_parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    materialize_parser = subparsers.add_parser(
        "materialize",
        help="create a profile checkout and write its generated files",
    )
    materialize_parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    materialize_parser.add_argument(
        "--feature-set",
        "--profile",
        dest="profile",
        metavar="NAME",
        required=True,
        help="feature set to apply (for example: full-default)",
    )
    materialize_parser.add_argument(
        "--output-dir",
        "--checkout",
        dest="checkout",
        metavar="PATH",
        type=Path,
        required=True,
        help="directory where the complete TensorFlow source tree is created",
    )
    materialize_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing checkout path",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        source_ref = worktree_source(manifest) if args.source == "WORKTREE" else args.source
        if args.command == "audit":
            audit(manifest, source_ref)
        elif args.command == "generate":
            generate(manifest, source_ref, args.output_dir.resolve())
        elif args.command == "verify":
            verify(manifest, source_ref, args.patch_dir.resolve())
        elif args.command == "materialize":
            materialize_profile(
                manifest,
                args.profile,
                args.patch_dir.resolve(),
                args.checkout.resolve(),
                force=args.force,
            )
        else:
            raise PatchError(f"unknown command: {args.command}")
    except (OSError, PatchError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
