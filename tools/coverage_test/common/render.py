#!/usr/bin/env python3
"""Generate CSV and fully offline HTML incremental coverage reports.

Ported from ``tools/incremental_coverage/report.py`` in the
``dev_2.15.0_gtest`` branch, adapted to this tool's data model: the coverage
input is the analysis result of ``common.coverage.analyze_incremental_coverage``
(which already carries per-line states and directory rollups) and the
execution table comes from the serialized ``TestResult`` dicts instead of the
branch's per-scenario execution archives.

Outputs, relative to the feature result directory:
``html/index.html`` (repository coverage tree + test executions + diagnostics),
``html/directories/<...>.html`` (per-directory rollups), ``html/files/<...>.html``
(per-file changed-line detail), and the CSVs ``incremental_coverage.csv`` /
``file_coverage.csv`` / ``directory_coverage.csv``.
"""

from __future__ import annotations

import csv
import html
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return value or "item"


def _link(from_path: Path, target: Path) -> str:
    return os.path.relpath(target, from_path.parent).replace(os.sep, "/")


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _state_css(state: str | None) -> str:
    return {
        "covered": "covered",
        "uncovered": "uncovered",
        "unknown": "unknown",
        "non_executable": "neutral",
        "excluded": "neutral",
    }.get(state or "", "")


CSS = """
:root { color-scheme: light; --ink:#1f2933; --muted:#52616b; --line:#d5dde3; --panel:#fff; --page:#f4f7f8; --good:#d8f3dc; --bad:#ffe0df; --warn:#fff1bc; --neutral:#e7edf0; }
* { box-sizing:border-box; } body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { background:#18323f; color:#fff; padding:22px max(20px, calc((100vw - 1460px) / 2)); } header h1 { margin:0 0 8px; font-size:24px; font-weight:650; letter-spacing:0; } header p { margin:0; color:#d9e5e9; }
main { max-width:1460px; margin:0 auto; padding:20px; } h2 { font-size:17px; margin:26px 0 10px; } h3 { font-size:14px; margin:0 0 8px; } a { color:#075f7a; } .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:16px 0; }
.metric,.panel { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px; } .metric b { font-size:20px; display:block; } .metric span { color:var(--muted); font-size:12px; }.panel { margin:10px 0; overflow:auto; }
table { width:100%; border-collapse:collapse; } th,td { padding:7px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; } th { background:#edf2f4; font-size:12px; } td.path { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
.status { display:inline-block; padding:1px 6px; border-radius:3px; font-size:12px; font-weight:600; background:var(--neutral); } .covered { background:var(--good); } .uncovered { background:var(--bad); } .unknown { background:var(--warn); } .neutral { background:var(--neutral); }
.note { color:var(--muted); } .warning { border-left:4px solid #d79500; padding:9px 12px; background:#fff7dc; }
.source { font:12px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace; table-layout:fixed; } .source .number { width:64px; text-align:right; color:var(--muted); } .source .marker { width:26px; text-align:center; font-weight:bold; } .source .hits { width:70px; text-align:right; } .source .state { width:145px; font:11px system-ui,sans-serif; font-weight:600; } .source pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; }
.source tr.covered td { background:var(--good); }.source tr.uncovered td { background:var(--bad); }.source tr.unknown td { background:var(--warn); }.source tr.neutral td { background:var(--neutral); }
code { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; } .foot { margin:28px 0 6px; color:var(--muted); font-size:12px; }
.coverage-tree { min-width:940px; } .tree-row { display:grid; grid-template-columns:minmax(310px,1fr) 115px 155px 85px 85px 85px 65px; border-bottom:1px solid var(--line); align-items:start; }
.tree-row > span { padding:7px 9px; } .tree-header { background:#edf2f4; font-size:12px; font-weight:600; } .tree-row .tree-name { padding-left:calc(9px + var(--depth, 0) * 18px); font-family:ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
summary.tree-row { list-style:none; cursor:pointer; background:#f6f9fa; } summary.tree-row::-webkit-details-marker { display:none; } .tree-name::before { content:""; display:inline-block; width:18px; } summary.tree-row .tree-name::before { content:"\\25b6"; } details[open] > summary.tree-row .tree-name::before { content:"\\25bc"; }
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _page(title: str, body: str, *, subtitle: str = "") -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>"
        f"<main>{body}<p class=\"foot\">Generated {_utc_now()}. This report is self-contained and can be opened directly from disk.</p></main>"
        "</body></html>"
    )


def _write_line_csv(path: Path, coverage: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "line", "state", "hits", "reason"])
        writer.writeheader()
        for file_data in coverage.get("files", []):
            for line in file_data.get("line_results", []):
                writer.writerow({
                    "path": file_data["path"],
                    "line": line["line"],
                    "state": line["state"],
                    "hits": "" if line.get("hits") is None else line["hits"],
                    "reason": line.get("reason", ""),
                })


def _write_file_summary_csv(path: Path, coverage: Mapping[str, Any]) -> None:
    fields = [
        "path", "added_lines", "covered_lines", "uncovered_lines", "executable_lines",
        "coverage_rate", "unmeasured_lines", "coverage_complete",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in coverage.get("files", []):
            writer.writerow({field: item.get(field) for field in fields})


def _write_directory_summary_csv(path: Path, coverage: Mapping[str, Any]) -> None:
    fields = [
        "path", "parent", "changed_files", "changed_lines", "covered_lines", "uncovered_lines",
        "executable_lines", "coverage_rate", "unmeasured_lines", "coverage_complete", "denominator_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in coverage.get("directories", []):
            writer.writerow({field: item.get(field) for field in fields})


def _coverage_note(data: Mapping[str, Any]) -> str:
    if data.get("denominator_reason"):
        return str(data["denominator_reason"])
    unknown = data.get("counts", {}).get("unknown", 0)
    if unknown:
        return f"Partial data: {unknown} changed lines have no coverage record."
    return "All changed lines have a confirmed state."


def _coverage_pages(
    html_dir: Path, category: str, items: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    return {
        str(item["path"]): html_dir / category / f"{index:04d}-{_safe_component(str(item['path']))}.html"
        for index, item in enumerate(items)
    }


def _breadcrumbs(
    destination: Path, index_path: Path, directory: str, directory_pages: Mapping[str, Path],
) -> str:
    links = [f'<a href="{html.escape(_link(destination, index_path))}">Back to overview</a>']
    current = PurePosixPath(directory)
    for parent in [*reversed(current.parents), current]:
        path = parent.as_posix()
        label = "Repository root" if path == "." else parent.name
        links.append(
            f'<a href="{html.escape(_link(destination, directory_pages[path]))}">{html.escape(label)}</a>'
        )
    return '<nav aria-label="Source directories">' + " / ".join(links) + "</nav>"


def _file_page(
    *,
    destination: Path,
    index_path: Path,
    source_root: Path,
    file_data: Mapping[str, Any],
    directory_pages: Mapping[str, Path],
) -> None:
    path = str(file_data["path"])
    source_path = source_root / path
    try:
        source_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        source_lines = []
    line_results = {int(item["line"]): item for item in file_data.get("line_results", [])}
    rows: list[str] = []
    if not source_lines:
        rows.append("<tr><td colspan=5>Source is unavailable in this checkout.</td></tr>")
    for number, source in enumerate(source_lines, 1):
        result = line_results.get(number)
        state = str(result.get("state")) if result else ""
        marker = "+" if result else ""
        status = state.replace("_", " ") if result else ""
        reason = str(result.get("reason", "")) if result else ""
        hits = "" if not result or result.get("hits") is None else str(result["hits"])
        label = html.escape(status + (f": {reason}" if reason else ""))
        rows.append(
            f"<tr id=\"L{number}\" class=\"{_state_css(state)}\"><td class=\"number\">{number}</td>"
            f"<td class=\"marker\">{marker}</td><td class=\"state\">{label}</td><td class=\"hits\">{hits}</td>"
            f"<td><pre>{html.escape(source)}</pre></td></tr>"
        )
    counts = file_data.get("counts", {})
    body = (
        _breadcrumbs(destination, index_path, PurePosixPath(path).parent.as_posix(), directory_pages) +
        "<div class=\"metrics\">"
        f"<div class=\"metric\"><b>{_format_rate(file_data.get('coverage_rate'))}</b><span>File incremental line coverage</span></div>"
        f"<div class=\"metric\"><b>{file_data.get('covered_lines', 0)} / {file_data.get('executable_lines', 0)}</b><span>Covered / executable changed lines</span></div>"
        f"<div class=\"metric\"><b>{counts.get('uncovered', 0)}</b><span>Uncovered changed lines</span></div>"
        f"<div class=\"metric\"><b>{counts.get('unknown', 0)}</b><span>Unknown changed lines</span></div></div>"
        f"<p class=\"note\">{html.escape(_coverage_note(file_data))}</p>"
        "<p class=\"note\">Coverage = covered executable changed lines / all confirmed executable changed lines. "
        "Only changed lines count toward this percentage; unknown lines are reported separately.</p>"
        "<p><span class=\"status covered\">+ Covered</span> <span class=\"status uncovered\">+ Uncovered</span> "
        "<span class=\"status unknown\">+ Unknown</span> <span class=\"status neutral\">+ Non-executable</span></p>"
        f"<div class=\"panel\"><h2>{html.escape(path)}</h2>"
        "<table class=\"source\"><thead><tr><th>Line</th><th>New</th><th>Status</th><th>Hits</th><th>Source</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_page(f"Incremental Coverage: {path}", body, subtitle="Changed source line detail"), encoding="utf-8")


def _coverage_table(
    files: Sequence[Mapping[str, Any]], page_path: Path, file_pages: Mapping[str, Path],
) -> str:
    rows: list[str] = []
    for file_data in files:
        path = str(file_data["path"])
        href = html.escape(_link(page_path, file_pages[path]))
        state_counts = file_data.get("counts", {})
        rows.append(
            "<tr>"
            f"<td class=\"path\"><a href=\"{href}\">{html.escape(path)}</a></td>"
            f"<td><b>{_format_rate(file_data.get('coverage_rate'))}</b>"
            f"<div class=\"note\">{html.escape(_coverage_note(file_data))}</div></td>"
            f"<td>{file_data.get('covered_lines', 0)} / {file_data.get('executable_lines', 0)}</td>"
            f"<td>{state_counts.get('uncovered', 0)}</td><td>{state_counts.get('unknown', 0)}</td>"
            f"<td>{file_data.get('added_lines', 0)}</td></tr>"
        )
    return (
        "<div class=\"panel\"><table id=\"file-coverage\"><thead><tr><th>C/C++ file</th><th>Incremental coverage</th>"
        "<th>Covered / executable changed lines</th><th>Uncovered</th><th>Unknown</th><th>Changed</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=6>No C/C++ files in this list.</td></tr>'}</tbody></table></div>"
    )


def _directory_table(
    directories: Sequence[Mapping[str, Any]], page_path: Path, directory_pages: Mapping[str, Path],
) -> str:
    rows = []
    for item in directories:
        path = str(item["path"])
        label = "./ (repository total)" if path == "." else path + "/"
        href = html.escape(_link(page_path, directory_pages[path]))
        rows.append(
            "<tr>"
            f'<td class="path"><a href="{href}">{html.escape(label)}</a></td>'
            f"<td>{item['changed_files']}</td><td><b>{_format_rate(item['coverage_rate'])}</b>"
            f'<div class="note">{html.escape(_coverage_note(item))}</div></td>'
            f"<td>{item['covered_lines']} / {item['executable_lines']}</td>"
            f"<td>{item['uncovered_lines']}</td><td>{item['unmeasured_lines']}</td>"
            f"<td>{item['changed_lines']}</td></tr>"
        )
    return (
        '<div class="panel"><table id="directory-coverage"><thead><tr><th>Directory</th><th>Files</th>'
        "<th>Incremental coverage</th><th>Covered / executable changed lines</th>"
        "<th>Uncovered</th><th>Unknown</th><th>Changed</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=7>No directories in this list.</td></tr>'}</tbody></table></div>"
    )


def _coverage_tree(
    coverage: Mapping[str, Any], page_path: Path, file_pages: Mapping[str, Path],
    directory_pages: Mapping[str, Path], *, root_label: str,
) -> str:
    directories = {item["path"]: item for item in coverage["directories"]}
    if not directories:
        return '<div class="panel">No C/C++ files remain after filtering.</div>'
    children = defaultdict(list)
    files = defaultdict(list)
    for item in directories.values():
        children[item["parent"]].append(item)
    for item in coverage["files"]:
        files[PurePosixPath(item["path"]).parent.as_posix()].append(item)

    def row(item: Mapping[str, Any], depth: int, *, is_directory: bool) -> str:
        path = str(item["path"])
        name = (root_label if path == "." else PurePosixPath(path).name) + ("/" if is_directory else "")
        pages = directory_pages if is_directory else file_pages
        href = html.escape(_link(page_path, pages[path]))
        tag = "summary" if is_directory else "div"
        changed = item["changed_lines"] if is_directory else item["added_lines"]
        file_count = item["changed_files"] if is_directory else 1
        partial = '<br><small class="note">Partial data</small>' if not item["coverage_complete"] else ""
        return (
            f'<{tag} class="tree-row" style="--depth:{depth}">'
            f'<span class="tree-name"><a href="{href}">{html.escape(name)}</a></span>'
            f'<span title="{html.escape(_coverage_note(item))}"><b>{_format_rate(item["coverage_rate"])}</b>{partial}</span>'
            f'<span>{item["covered_lines"]} / {item["executable_lines"]}</span>'
            f'<span>{item["uncovered_lines"]}</span><span>{item["unmeasured_lines"]}</span>'
            f'<span>{changed}</span><span>{file_count}</span></{tag}>'
        )

    def directory_node(item: Mapping[str, Any], depth: int) -> str:
        path = str(item["path"])
        content = "".join(directory_node(child, depth + 1) for child in children[path])
        content += "".join(row(child, depth + 1, is_directory=False) for child in files[path])
        return '<details open>' + row(item, depth, is_directory=True) + content + '</details>'

    return (
        '<div class="panel"><div class="coverage-tree" id="coverage-tree">'
        '<div class="tree-row tree-header"><span>Directory / file</span><span>Incremental coverage</span>'
        '<span>Covered / executable</span><span>Uncovered</span><span>Unknown</span><span>Changed</span><span>Files</span></div>'
        + directory_node(directories["."], 0) + '</div></div>'
    )


def _directory_page(
    *,
    destination: Path,
    index_path: Path,
    directory: Mapping[str, Any],
    subdirectories: Sequence[Mapping[str, Any]],
    files: Sequence[Mapping[str, Any]],
    directory_pages: Mapping[str, Path],
    file_pages: Mapping[str, Path],
) -> None:
    path = str(directory["path"])
    body = (
        _breadcrumbs(destination, index_path, path, directory_pages) +
        '<div class="metrics">'
        f'<div class="metric"><b>{_format_rate(directory["coverage_rate"])}</b><span>Directory incremental line coverage</span></div>'
        f'<div class="metric"><b>{directory["covered_lines"]} / {directory["executable_lines"]}</b><span>Covered / executable changed lines</span></div>'
        f'<div class="metric"><b>{directory["changed_files"]}</b><span>C/C++ files, including subdirectories</span></div>'
        f'<div class="metric"><b>{directory["uncovered_lines"]}</b><span>Uncovered changed lines</span></div>'
        f'<div class="metric"><b>{directory["unmeasured_lines"]}</b><span>Unknown changed lines</span></div></div>'
        f'<p class="note">{html.escape(_coverage_note(directory))}</p>'
        '<p class="note">Totals include all nested subdirectories after filtering. '
        "Coverage = total covered executable changed lines / total confirmed executable changed lines. "
        "Unknown lines are listed separately; file percentages are not averaged.</p>"
        "<h2>Subdirectories</h2>" + _directory_table(subdirectories, destination, directory_pages) +
        "<h2>Files in this directory</h2>" + _coverage_table(files, destination, file_pages)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _page(f"Directory Coverage: {path}/", body, subtitle="Incremental totals including all subdirectories"),
        encoding="utf-8",
    )


def _execution_table(results: Sequence[Mapping[str, Any]], html_dir: Path) -> str:
    rows: list[str] = []
    for item in results:
        status = "passed" if item.get("passed") else "failed"
        coverage_file = item.get("coverage_file")
        coverage_link = ""
        if coverage_file and Path(coverage_file).is_file():
            href = os.path.relpath(coverage_file, html_dir).replace(os.sep, "/")
            coverage_link = f'<a href="{html.escape(href)}">coverage.info</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('target_id', '')))}</td>"
            f"<td class=\"path\">{html.escape(str(item.get('target', '')))}</td>"
            f"<td><span class=\"status {_state_css('covered' if item.get('passed') else 'uncovered')}\">{status}</span></td>"
            f"<td>{item.get('cases_run', '')}</td><td>{item.get('cases_failed', '')}</td>"
            f"<td>{item.get('duration', 0):.2f}</td>"
            f"<td>{coverage_link}</td></tr>"
        )
    return (
        "<div class=\"panel\"><table><thead><tr><th>Target</th><th>Bazel target</th><th>Status</th>"
        "<th>Cases</th><th>Failed</th><th>Seconds</th><th>Coverage data</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=7>No test execution results.</td></tr>'}</tbody></table></div>"
    )


def generate_report(
    *,
    analysis: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    source_root: Path,
    exclude_patterns: Sequence[str] = (),
    title: str = "TensorFlow Incremental Coverage",
    subtitle: str = "Offline report for changed C/C++ production lines",
) -> dict[str, Any]:
    """Create the HTML tree, per-directory/per-file pages and CSVs."""

    output_dir = output_dir.resolve()
    source_root = source_root.resolve()
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    # html/files/ and html/directories/ are owned by this generator; remove
    # stale pages from a previous report so renames/filters do not leave
    # orphaned source pages behind.
    for category in ("files", "directories"):
        category_dir = html_dir / category
        if category_dir.is_dir():
            for stale in category_dir.glob("*.html"):
                stale.unlink()

    _write_line_csv(output_dir / "incremental_coverage.csv", analysis)
    _write_file_summary_csv(output_dir / "file_coverage.csv", analysis)
    _write_directory_summary_csv(output_dir / "directory_coverage.csv", analysis)

    index_path = html_dir / "index.html"
    file_pages = _coverage_pages(html_dir, "files", analysis["files"])
    directory_pages = _coverage_pages(html_dir, "directories", analysis["directories"])
    files_by_directory = defaultdict(list)
    directories_by_parent = defaultdict(list)
    for file_data in analysis["files"]:
        files_by_directory[PurePosixPath(file_data["path"]).parent.as_posix()].append(file_data)
        _file_page(
            destination=file_pages[file_data["path"]],
            index_path=index_path,
            source_root=source_root,
            file_data=file_data,
            directory_pages=directory_pages,
        )
    for directory in analysis["directories"]:
        directories_by_parent[directory["parent"]].append(directory)
    for directory in analysis["directories"]:
        _directory_page(
            destination=directory_pages[directory["path"]],
            index_path=index_path,
            directory=directory,
            subdirectories=directories_by_parent[directory["path"]],
            files=files_by_directory[directory["path"]],
            directory_pages=directory_pages,
            file_pages=file_pages,
        )

    coverage_summary = analysis["summary"]
    state_counts = coverage_summary.get("counts", {})
    metrics = (
        "<div class=\"metrics\">"
        f"<div class=\"metric\"><b>{_format_rate(coverage_summary.get('coverage_rate'))}</b><span>incremental line coverage</span></div>"
        f"<div class=\"metric\"><b>{coverage_summary.get('covered_lines', 0)} / {coverage_summary.get('executable_lines', 0)}</b><span>covered / confirmed executable</span></div>"
        f"<div class=\"metric\"><b>{state_counts.get('unknown', 0)}</b><span>unknown changed lines</span></div>"
        f"<div class=\"metric\"><b>{len(analysis.get('files', []))}</b><span>C/C++ files in coverage scope</span></div></div>"
    )
    completeness = (
        "<div class=\"panel warning\">Coverage data is partial: one or more changed lines have no reliable LCOV DA record. "
        "They are displayed as UNKNOWN and are not silently treated as non-executable.</div>"
        if not coverage_summary.get("coverage_complete") else
        "<div class=\"panel\">All included changed lines have a confirmed state for this report.</div>"
    )
    filtered = "".join(
        f"<li><code>{html.escape(str(item['path']))}</code>: {html.escape(str(item['reason']))}</li>"
        for item in analysis.get("excluded_files", [])
    )
    diagnostic_items = "".join(
        f"<li>{html.escape(repr(item))}</li>"
        for item in analysis.get("lcov_diagnostics", [])
    )
    body = (
        f"{metrics}{completeness}"
        "<h2>Repository Coverage Tree</h2><p class=\"note\">Expand or collapse directories with the arrows. "
        "Click a directory name to browse it, or a file name to see its source coverage. "
        "Each directory includes all files in its subdirectories, "
        "after C/C++ and configured exclusions. Coverage = total covered / total confirmed executable changed lines. "
        "Parent and child rows overlap; do not add them together.</p>"
        + _coverage_tree(analysis, index_path, file_pages, directory_pages, root_label=source_root.name) +
        "<details><summary>Coverage by File — flat list</summary><p class=\"note\">Only C/C++ sources and headers are included. "
        "Script, build and configured test files are excluded. Click a file to see changed lines and coverage hits.</p>"
        + _coverage_table(analysis["files"], index_path, file_pages) +
        "</details><h2>Test Executions</h2>" + _execution_table(results, html_dir) +
        "<h2>Filtering and Diagnostics</h2>"
        f"<div class=\"panel\"><h3>Exclude patterns</h3><p>{html.escape(', '.join(exclude_patterns) or '(none)')}</p>"
        f"<details><summary>Excluded files ({len(analysis.get('excluded_files', []))})</summary><ul>{filtered or '<li>None</li>'}</ul></details>"
        f"<h3>LCOV diagnostics</h3><ul>{diagnostic_items or '<li>None</li>'}</ul></div>"
        "<h2>Artifacts</h2><div class=\"panel\"><ul>"
        "<li><a href=\"../report.json\">report.json</a></li>"
        "<li><a href=\"../incremental_coverage.csv\">incremental_coverage.csv — per changed line</a></li>"
        "<li><a href=\"../file_coverage.csv\">file_coverage.csv — coverage by file</a></li>"
        "<li><a href=\"../directory_coverage.csv\">directory_coverage.csv — coverage by directory</a></li>"
        "</ul></div>"
    )
    index_path.write_text(_page(title, body, subtitle=subtitle), encoding="utf-8")
    return {
        "html_index": str(index_path),
        "file_pages": len(file_pages),
        "directory_pages": len(directory_pages),
    }
