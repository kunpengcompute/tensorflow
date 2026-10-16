#!/usr/bin/env python3
"""Test runner for gtest and benchmark targets with coverage support."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of running a single test target."""
    target_id: str
    target: str
    kind: str
    success: bool
    duration: float
    output: str
    error: str = ""
    cases_run: int = 0
    cases_failed: int = 0
    coverage_file: Path | None = None  # Path to lcov file if coverage enabled

    @property
    def passed(self) -> bool:
        return self.success and self.cases_failed == 0


class TestRunner:
    """Execute test targets and collect results with optional coverage."""

    def __init__(
        self,
        repository_root: Path,
        bazel_bin: str = "bazel",
        enable_coverage: bool = False,
    ):
        self.repository_root = repository_root
        self.bazel_bin = bazel_bin
        self.output_base = repository_root / "output"
        self.enable_coverage = enable_coverage
        self.coverage_output_dir = repository_root / "coverage_data"

        if self.enable_coverage:
            self.coverage_output_dir.mkdir(parents=True, exist_ok=True)

    def build_target(
        self,
        target: str,
        bazel_args: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Build a single Bazel target with optional coverage instrumentation."""
        args = bazel_args or []

        # Add coverage flags if enabled
        if self.enable_coverage:
            coverage_args = [
                "--collect_code_coverage",
                "--instrumentation_filter=//tensorflow/...,//third_party/...",
                "--combined_report=lcov",
            ]
            args = coverage_args + args

        cmd = [
            self.bazel_bin,
            "--batch",
            f"--output_base={self.output_base}",
            "build" if not self.enable_coverage else "coverage",
        ] + args + [target]

        print(f"Building: {target}")
        if self.enable_coverage:
            print("Coverage collection enabled")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Build timeout (30 minutes)"
        except Exception as e:
            return False, f"Build error: {e}"

    def find_test_binary(self, target: str) -> Path | None:
        """Find the compiled test binary."""
        # Extract binary name from target
        # //tensorflow/core/kernels:test_name -> test_name
        target_name = target.split(":")[-1]

        # Search in common Bazel output paths
        search_paths = [
            self.output_base / "execroot" / "*" / "bazel-out" / "*" / "bin",
        ]

        import glob
        for pattern in search_paths:
            for base_path in glob.glob(str(pattern)):
                # Recursively search for the binary
                for root, _, files in os.walk(base_path):
                    if target_name in files:
                        binary_path = Path(root) / target_name
                        if binary_path.is_file() and os.access(binary_path, os.X_OK):
                            return binary_path

        return None

    def run_gtest(
        self,
        target_id: str,
        target: str,
        binary_path: Path,
        gtest_filter: str = "*",
        env: dict[str, str] | None = None,
    ) -> TestResult:
        """Run a gtest binary and optionally collect coverage."""
        start_time = time.time()

        test_env = os.environ.copy()
        if env:
            test_env.update(env)

        # Add required test directories
        test_tmpdir = self.repository_root / "test_tmp"
        test_tmpdir.mkdir(exist_ok=True)
        test_env["TEST_TMPDIR"] = str(test_tmpdir)
        test_env["TMPDIR"] = str(test_tmpdir)

        cmd = [str(binary_path), f"--gtest_filter={gtest_filter}"]

        print(f"Running gtest: {target_id}")
        print(f"Command: {' '.join(cmd)}")

        coverage_file = None

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                env=test_env,
                timeout=600,  # 10 minutes
            )

            duration = time.time() - start_time
            output = result.stdout

            # Collect coverage data if enabled
            if self.enable_coverage:
                coverage_file = self._collect_coverage_data(target_id, target)

            # Parse gtest output
            cases_run = 0
            cases_failed = 0
            for line in output.split("\n"):
                if " tests from " in line or " test from " in line:
                    # [==========] Running 10 tests from 2 test suites.
                    parts = line.split()
                    if "Running" in parts:
                        idx = parts.index("Running")
                        if idx + 1 < len(parts):
                            try:
                                cases_run = int(parts[idx + 1])
                            except ValueError:
                                pass
                elif "[  FAILED  ]" in line and " tests" in line:
                    # [  FAILED  ] 2 tests
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "FAILED" and i + 2 < len(parts):
                            try:
                                cases_failed = int(parts[i + 2])
                            except ValueError:
                                pass

            return TestResult(
                target_id=target_id,
                target=target,
                kind="gtest",
                success=result.returncode == 0,
                duration=duration,
                output=output,
                error=result.stderr,
                cases_run=cases_run,
                cases_failed=cases_failed,
                coverage_file=coverage_file,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                target_id=target_id,
                target=target,
                kind="gtest",
                success=False,
                duration=duration,
                output="",
                error="Test timeout (10 minutes)",
                coverage_file=coverage_file,
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                target_id=target_id,
                target=target,
                kind="gtest",
                success=False,
                duration=duration,
                output="",
                error=f"Test execution error: {e}",
                coverage_file=coverage_file,
            )

    def _collect_coverage_data(self, target_id: str, target: str) -> Path | None:
        """Collect coverage data after test execution."""
        try:
            # Find the coverage.dat file generated by bazel
            # Bazel puts it in execroot/org_tensorflow/bazel-out/_coverage/_coverage_report.dat
            coverage_dat = (
                self.output_base
                / "execroot"
                / "org_tensorflow"
                / "bazel-out"
                / "_coverage"
                / "_coverage_report.dat"
            )

            if not coverage_dat.exists():
                print(f"⚠️  Coverage data not found at: {coverage_dat}")
                # Try to find it
                import glob

                for root, dirs, files in os.walk(self.output_base / "execroot"):
                    if "_coverage_report.dat" in files:
                        coverage_dat = Path(root) / "_coverage_report.dat"
                        print(f"   Found at: {coverage_dat}")
                        break

            if not coverage_dat.exists():
                print(f"⚠️  Coverage data not found for {target_id}")
                return None

            # Copy coverage data (it's already in lcov-compatible format)
            output_file = self.coverage_output_dir / f"{target_id}.lcov"
            subprocess.run(
                ["cp", str(coverage_dat), str(output_file)],
                check=True,
            )

            print(f"✅ Coverage data saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"⚠️  Failed to collect coverage data: {e}")
            import traceback

            traceback.print_exc()
            return None

