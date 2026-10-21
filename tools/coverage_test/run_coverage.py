#!/usr/bin/env python3
"""Main entry point for TensorFlow incremental coverage testing framework.

This script provides a unified interface to run coverage tests for different
TensorFlow features with incremental coverage analysis support.

Usage:
    # Run tests for a feature (no coverage)
    python3 tools/coverage_test/run_coverage.py --feature kembedding

    # Run tests with incremental coverage from patch file
    python3 tools/coverage_test/run_coverage.py --feature kembedding \
        --patch path/to/feature.patch \
        --enable-coverage

    # Run tests with incremental coverage from commit
    python3 tools/coverage_test/run_coverage.py --feature kembedding \
        --commit 716b3149563 \
        --enable-coverage

    # Run tests with incremental coverage from git diff
    python3 tools/coverage_test/run_coverage.py --feature kembedding \
        --base-ref HEAD~1 --target-ref HEAD \
        --enable-coverage

    # Run tests with incremental coverage since a commit (shorthand)
    python3 tools/coverage_test/run_coverage.py --feature kembedding \
        --since 716b3149563 \
        --enable-coverage

    # Run all features and report ONE aggregate incremental coverage number
    python3 tools/coverage_test/run_coverage.py --all \
        --since 086c065e75843cddaa17188cd84943ad2d9d74f1 --enable-coverage
"""

import argparse
import json
import sys
from pathlib import Path

# Add common directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.coverage import DEFAULT_EXCLUDES, IncrementalCoverageAnalyzer
from common.render import generate_report
from common.config import load_feature_config
from common.runner import TestRunner
from datetime import datetime


FEATURES_DIR = SCRIPT_DIR / "features"


def discover_features():
    """Auto-discover feature directories under FEATURES_DIR.

    Each subdirectory (excluding hidden/private dirs like ``__pycache__``)
    is treated as one feature. Returns a sorted list of feature names.
    """
    if not FEATURES_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in FEATURES_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
        and not p.name.startswith("__")
    )


AVAILABLE_FEATURES = discover_features()


def list_features():
    """List all available features."""
    print("Available features:")
    for feature in AVAILABLE_FEATURES:
        feature_dir = FEATURES_DIR / feature
        config_file = feature_dir / "config.json"
        if config_file.exists():
            config = load_feature_config(feature_dir)
            print(f"  ✓ {feature:<15} - {config.description}")
        else:
            print(f"  ✗ {feature:<15} (config missing)")


def _print_incremental_report(report: dict) -> None:
    """Print the incremental coverage report (summary + per-file)."""
    summary = report["summary"]
    rate = summary.get("coverage_rate")
    rate_str = f"{rate:.2f}%" if rate is not None else "n/a"

    print(f"\n📊 Incremental Coverage Summary")
    print(f"  Changed files: {summary['changed_files']}")
    print(f"  Changed lines: {summary['changed_lines']}")
    print(f"  Executable lines: {summary['executable_lines']}")
    print(f"  Covered lines: {summary['covered_lines']}")
    print(f"  Uncovered lines: {summary['uncovered_lines']}")
    print(f"  Unmeasured (unknown) lines: {summary['unmeasured_lines']}")
    print(f"  Coverage: {rate_str}")
    print(f"  Coverage complete: {summary['coverage_complete']}")

    print(f"\n📋 Per-File Coverage:")
    for file_data in report["files"]:
        fr = file_data.get("coverage_rate")
        fr_str = f"{fr:.2f}%" if fr is not None else "n/a"
        print(f"  {file_data['path']}")
        print(
            f"    Added: {file_data['added_lines']}, "
            f"Executable: {file_data['executable_lines']}, "
            f"Covered: {file_data['covered_lines']}, "
            f"Uncovered: {file_data['uncovered_lines']}, "
            f"Unmeasured: {file_data['unmeasured_lines']}"
        )
        print(f"    Coverage: {fr_str}")

    excluded = report.get("excluded_files", [])
    if excluded:
        print(f"\n📌 Excluded files ({len(excluded)}):")
        for item in excluded:
            print(f"  - {item['path']} ({item['reason']})")


def _mode_note(
    patch_file: Path | None, commit_ref: str | None,
    base_ref: str | None, target_ref: str | None, since_ref: str | None,
) -> str:
    if patch_file:
        return f"patch {patch_file}"
    if commit_ref:
        return f"commit {commit_ref}"
    if since_ref:
        return f"since {since_ref} (inclusive) to HEAD"
    if base_ref and target_ref:
        return f"diff {base_ref}..{target_ref}"
    return "no incremental range selected"


def _generate_incremental_html(
    incremental_report: dict | None,
    results: list,
    output_dir: Path,
    mode_note: str,
) -> None:
    """Render the incremental coverage HTML tree + CSVs.

    Ported from tools/incremental_coverage/report.py in the dev_2.15.0_gtest
    branch: html/index.html (repository coverage tree, test executions,
    diagnostics), html/directories/ (per-directory rollups), html/files/
    (per-file changed-line detail). Unlike a genhtml dump of the merged LCOV
    (total coverage of every executed file, third-party dependencies
    included), only changed C++ files and their changed lines are reported.
    """
    if not incremental_report:
        return
    repo_root = SCRIPT_DIR.parent.parent
    try:
        info = generate_report(
            analysis=incremental_report,
            results=results,
            output_dir=output_dir,
            source_root=repo_root,
            exclude_patterns=list(DEFAULT_EXCLUDES),
            subtitle=f"Offline incremental coverage report — {mode_note}",
        )
        print(f"Incremental report: {info['html_index']}")
    except Exception as e:
        print(f"⚠️  Failed to render incremental HTML: {e}")
        import traceback
        traceback.print_exc()


def run_feature_tests(
    feature: str,
    enable_coverage: bool,
):
    """Build and run all gtest targets for one feature.

    Returns ``(config, results, coverage_files, failed_count)``.
    """
    feature_dir = FEATURES_DIR / feature
    config = load_feature_config(feature_dir)
    repo_root = SCRIPT_DIR.parent.parent

    print(f"\n{'=' * 70}")
    print(f"Feature: {config.name}")
    print(f"Description: {config.description}")
    print(f"Coverage enabled: {enable_coverage}")
    print(f"{'=' * 70}\n")

    runner = TestRunner(repository_root=repo_root, enable_coverage=enable_coverage)

    results = []
    coverage_files = []

    for test_config in config.test_targets:
        if not test_config.enabled:
            print(f"⏭️  Skipping disabled target: {test_config.id}")
            continue

        if test_config.kind != "gtest":
            # Benchmarks are out of scope: skip without recording a result so
            # they don't drag the pass/fail count down.
            print(f"⏭️  Skipping benchmark target: {test_config.id}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Target: {test_config.id}")
        print(f"Bazel: {test_config.target}")
        print(f"{'=' * 70}\n")

        # Merge bazel args: feature-level + target-level build flags
        merged_bazel_args = config.bazel_args.copy()

        for flag in config.bazel_build_flags:
            merged_bazel_args.append(f"--copt={flag}")
        for flag in test_config.bazel_build_flags:
            merged_bazel_args.append(f"--copt={flag}")

        success, build_output = runner.build_target(
            test_config.target,
            bazel_args=merged_bazel_args,
        )
        if not success:
            print(f"❌ Build failed")
            print(build_output[-1000:])
            results.append({
                "target_id": test_config.id,
                "passed": False,
                "error": "Build failed",
            })
            continue

        print(f"✅ Build succeeded\n")

        binary_path = runner.find_test_binary(test_config.target)
        if not binary_path:
            print(f"❌ Test binary not found")
            results.append({
                "target_id": test_config.id,
                "passed": False,
                "error": "Binary not found",
            })
            continue

        print(f"Binary: {binary_path}\n")

        result = runner.run_gtest(
            test_config.id,
            test_config.target,
            binary_path,
            gtest_filter=test_config.gtest_filter,
            env=test_config.env,
        )

        print(f"\nCompleted in {result.duration:.2f}s")
        print(f"Cases: {result.cases_run}, Failed: {result.cases_failed}")

        if result.passed:
            print(f"✅ PASSED")
        else:
            print(f"❌ FAILED: {result.error}")

        if result.coverage_file:
            coverage_files.append(result.coverage_file)
            print(f"📊 Coverage: {result.coverage_file}")

        results.append({
            "target_id": result.target_id,
            "target": result.target,
            "passed": result.passed,
            "duration": result.duration,
            "cases_run": result.cases_run,
            "cases_failed": result.cases_failed,
            "coverage_file": str(result.coverage_file) if result.coverage_file else None,
        })

    failed_count = sum(1 for r in results if not r.get("passed"))
    return config, results, coverage_files, failed_count


def run_feature_with_coverage(
    feature: str,
    patch_file: Path | None = None,
    commit_ref: str | None = None,
    base_ref: str | None = None,
    target_ref: str | None = None,
    since_ref: str | None = None,
    enable_coverage: bool = False,
) -> int:
    """Run tests for a single feature with per-feature incremental analysis."""

    repo_root = SCRIPT_DIR.parent.parent
    _, results, coverage_files, failed_count = run_feature_tests(
        feature, enable_coverage
    )

    # Analyze incremental coverage over the whole diff
    incremental_report = None
    if enable_coverage and coverage_files:
        print(f"\n{'=' * 70}")
        print("Incremental Coverage Analysis")
        print(f"{'=' * 70}\n")

        analyzer = IncrementalCoverageAnalyzer(repo_root)
        try:
            if patch_file:
                print(f"Analyzing patch: {patch_file}")
                incremental_report = analyzer.analyze_patch_coverage(
                    patch_file, coverage_files
                )
            elif commit_ref:
                print(f"Analyzing commit: {commit_ref}")
                incremental_report = analyzer.analyze_commit_coverage(
                    commit_ref, coverage_files
                )
            elif since_ref:
                # --since is inclusive: diff from the parent of the given
                # commit so the commit's own changes are included.
                inclusive_base = f"{since_ref}^"
                print(f"Analyzing changes since: {since_ref}..HEAD (inclusive)")
                incremental_report = analyzer.analyze_diff_coverage(
                    inclusive_base, "HEAD", coverage_files
                )
            elif base_ref and target_ref:
                print(f"Analyzing diff: {base_ref}..{target_ref}")
                incremental_report = analyzer.analyze_diff_coverage(
                    base_ref, target_ref, coverage_files
                )

            if incremental_report:
                _print_incremental_report(incremental_report)
        except Exception as e:
            print(f"⚠️  Coverage analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate report
    output_dir = repo_root / "coverage_test_results" / feature
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "feature": feature,
        "timestamp": datetime.now().isoformat(),
        "coverage_enabled": enable_coverage,
        "patch_file": str(patch_file) if patch_file else None,
        "commit_ref": commit_ref,
        "since_ref": since_ref,
        "base_ref": base_ref,
        "target_ref": target_ref,
        "total_targets": len(results),
        "passed_targets": len(results) - failed_count,
        "failed_targets": failed_count,
        "results": results,
        "incremental_coverage": incremental_report,
    }

    report_file = output_dir / "report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    if enable_coverage and coverage_files:
        _generate_incremental_html(
            incremental_report, results, output_dir,
            _mode_note(patch_file, commit_ref, base_ref, target_ref, since_ref),
        )

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"Total targets: {len(results)}")
    print(f"Passed: {len(results) - failed_count}")
    print(f"Failed: {failed_count}")
    print(f"Report: {report_file}")
    print(f"{'=' * 70}\n")

    return 0 if failed_count == 0 else 1


def run_all_with_coverage(
    patch_file: Path | None = None,
    commit_ref: str | None = None,
    base_ref: str | None = None,
    target_ref: str | None = None,
    since_ref: str | None = None,
    enable_coverage: bool = False,
) -> int:
    """Run every feature's tests, then report ONE aggregate incremental
    coverage total computed over the whole diff with all collected lcov data.
    """
    repo_root = SCRIPT_DIR.parent.parent
    analyzer = IncrementalCoverageAnalyzer(repo_root)

    all_results = []
    all_coverage_files = []
    failed_features = []
    ran_features = []

    for feature in AVAILABLE_FEATURES:
        ran_features.append(feature)
        _, results, coverage_files, failed_count = run_feature_tests(
            feature, enable_coverage
        )
        all_results.extend(results)
        all_coverage_files.extend(coverage_files)
        if failed_count:
            failed_features.append(feature)

    # Single aggregate incremental coverage across all features.
    incremental_report = None
    if enable_coverage and all_coverage_files:
        print(f"\n{'=' * 70}")
        print("Aggregate Incremental Coverage Analysis (all features)")
        print(f"{'=' * 70}\n")
        try:
            if patch_file:
                print(f"Analyzing patch: {patch_file}")
                incremental_report = analyzer.analyze_patch_coverage(
                    patch_file, all_coverage_files
                )
            elif commit_ref:
                print(f"Analyzing commit: {commit_ref}")
                incremental_report = analyzer.analyze_commit_coverage(
                    commit_ref, all_coverage_files
                )
            elif since_ref:
                print(f"Analyzing changes since: {since_ref}..HEAD (inclusive)")
                incremental_report = analyzer.analyze_diff_coverage(
                    f"{since_ref}^", "HEAD", all_coverage_files
                )
            elif base_ref and target_ref:
                print(f"Analyzing diff: {base_ref}..{target_ref}")
                incremental_report = analyzer.analyze_diff_coverage(
                    base_ref, target_ref, all_coverage_files
                )

            if incremental_report:
                _print_incremental_report(incremental_report)
        except Exception as e:
            print(f"⚠️  Coverage analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate aggregate report
    output_dir = repo_root / "coverage_test_results" / "all"
    output_dir.mkdir(parents=True, exist_ok=True)

    passed_count = sum(1 for r in all_results if r.get("passed"))
    failed_count = len(all_results) - passed_count

    report_data = {
        "feature": "all",
        "timestamp": datetime.now().isoformat(),
        "coverage_enabled": enable_coverage,
        "patch_file": str(patch_file) if patch_file else None,
        "commit_ref": commit_ref,
        "since_ref": since_ref,
        "base_ref": base_ref,
        "target_ref": target_ref,
        "features_ran": ran_features,
        "total_targets": len(all_results),
        "passed_targets": passed_count,
        "failed_targets": failed_count,
        "results": all_results,
        "incremental_coverage": incremental_report,
    }

    report_file = output_dir / "report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    if enable_coverage and all_coverage_files:
        _generate_incremental_html(
            incremental_report, all_results, output_dir,
            _mode_note(patch_file, commit_ref, base_ref, target_ref, since_ref),
        )

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"Features run: {', '.join(ran_features) if ran_features else '(none)'}")
    print(f"Total targets: {len(all_results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Report: {report_file}")
    print(f"{'=' * 70}\n")

    return 0 if failed_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="TensorFlow incremental coverage testing framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--feature",
        choices=AVAILABLE_FEATURES,
        help="Feature to test",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all features",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available features",
    )

    # Coverage options
    parser.add_argument(
        "--enable-coverage",
        action="store_true",
        help="Enable coverage collection (requires lcov)",
    )

    parser.add_argument(
        "--patch",
        type=Path,
        help="Patch file for incremental coverage analysis",
    )

    parser.add_argument(
        "--commit",
        help="Git commit reference for incremental coverage",
    )

    parser.add_argument(
        "--base-ref",
        help="Base git reference for diff-based coverage",
    )

    parser.add_argument(
        "--target-ref",
        help="Target git reference for diff-based coverage",
    )

    parser.add_argument(
        "--since",
        help="Git reference to compare from, INCLUSIVE of that commit "
             "(diff from <ref>^ to HEAD)",
    )

    args = parser.parse_args()

    if args.list:
        list_features()
        return 0

    # Validate coverage options
    coverage_sources = sum([
        args.patch is not None,
        args.commit is not None,
        args.since is not None,
        args.base_ref is not None or args.target_ref is not None,
    ])

    if coverage_sources > 1:
        print("Error: Specify only one of --patch, --commit, --since, or --base-ref/--target-ref")
        return 1

    if (args.base_ref is None) != (args.target_ref is None):
        print("Error: Both --base-ref and --target-ref must be specified together")
        return 1

    if args.enable_coverage and coverage_sources == 0:
        print("Warning: --enable-coverage specified without --patch, --commit, --since, or --base-ref/--target-ref")
        print("Coverage will be collected but no incremental analysis will be performed")

    if args.all:
        print("Running all features...")
        return run_all_with_coverage(
            patch_file=args.patch,
            commit_ref=args.commit,
            base_ref=args.base_ref,
            target_ref=args.target_ref,
            since_ref=args.since,
            enable_coverage=args.enable_coverage,
        )

    if args.feature:
        return run_feature_with_coverage(
            args.feature,
            patch_file=args.patch,
            commit_ref=args.commit,
            base_ref=args.base_ref,
            target_ref=args.target_ref,
            since_ref=args.since,
            enable_coverage=args.enable_coverage,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
