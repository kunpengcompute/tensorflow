#!/usr/bin/env python3
"""Test runner for gtest and benchmark targets with coverage support."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
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
    asan_error_count: int = 0  # Number of ASAN findings in this target's run
    asan_report_files: list[Path] = field(default_factory=list)
    failed_cases: list[str] = field(default_factory=list)  # Failed gtest case names

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
        enable_asan: bool = False,
    ):
        self.repository_root = repository_root
        self.bazel_bin = bazel_bin
        self.output_base = repository_root / "output"
        self.enable_coverage = enable_coverage
        self.coverage_output_dir = repository_root / "coverage_data"
        self.enable_asan = enable_asan
        # ASAN 报告目录（ASAN_OPTIONS 的 log_path 前缀），ASAN_LOG_DIR 可覆盖。
        self.asan_log_dir = Path(
            os.environ.get("ASAN_LOG_DIR") or repository_root / "asan_log"
        )
        # libasan 运行库：测试二进制动态链接（DT_NEEDED），loader 从默认
        # 路径解析，无需 LD_PRELOAD；仅作预检 fail-fast，找不到时为 None。
        self.libasan_path: Path | None = None

        if self.enable_coverage:
            self.coverage_output_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_asan:
            self.asan_log_dir.mkdir(parents=True, exist_ok=True)
            self.libasan_path = self._find_libasan()
            if self.libasan_path is None:
                raise RuntimeError(
                    "gcc 未找到可 LD_PRELOAD 的 libasan.so.8，"
                    "安装配套运行库后重试：yum install libasan。"
                )
            print(f"ASAN enabled: log dir {self.asan_log_dir}")
            print(f"ASAN: libasan runtime {self.libasan_path}")

    def _find_libasan(self) -> Path | None:
        """预检 libasan 运行库是否就位。

        测试二进制经 -fsanitize=address 动态链接，DT_NEEDED=libasan.so.8 由
        loader 从默认路径解析（gcc 探测即定位到系统路径 /usr/lib64）。
        gcc 找不到时返回原名（非绝对路径），据此判定缺失并给出安装提示。
        """
        try:
            result = subprocess.run(
                ["gcc", "-print-file-name=libasan.so.8"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            candidate = Path(result.stdout.strip())
            if candidate.is_absolute() and candidate.exists():
                return candidate.resolve()
        except Exception:
            pass
        return None

    def _asan_options(self, prefix: str) -> str:
        """ASAN_OPTIONS：recover 模式（halt_on_error=0），日志按前缀归因。

        log_path 是前缀，文件名 = <prefix>.<pid>。
        """
        return f"halt_on_error=0:log_path={self.asan_log_dir}/{prefix}"

    def _collect_asan_reports(self, target_id: str) -> tuple[list[Path], int]:
        """汇总 asan_<target_id>.* 报告文件及其中 ERROR 行数。

        超时/异常路径同样调用：recover 模式下报告随执行落盘，
        进程被杀前的发现不能丢。
        """
        report_files = sorted(self.asan_log_dir.glob(f"asan_{target_id}.*"))
        error_count = 0
        for report_file in report_files:
            try:
                error_count += report_file.read_text(
                    encoding="utf-8", errors="ignore"
                ).count("ERROR: AddressSanitizer")
            except OSError:
                pass
        return report_files, error_count

    def build_target(
        self,
        target: str,
        bazel_args: list[str] | None = None,
        asan_prefix: str = "asan",
    ) -> tuple[bool, str]:
        """Build a single Bazel target with optional coverage instrumentation.

        asan_prefix: ASAN 日志前缀（按 target 归因），仅 bazel coverage
        这一趟使用；run_gtest 直接执行用 asan_<target_id> 前缀，两趟前缀
        分开，避免同一发现被计入两次。
        """
        args = bazel_args or []

        # Add coverage flags if enabled
        if self.enable_coverage:
            coverage_args = [
                "--collect_code_coverage",
                "--instrumentation_filter=//tensorflow/...,//third_party/...",
                "--combined_report=lcov",
            ]
            args = coverage_args + args

        if self.enable_asan:
            # 清掉上一轮同名前缀的陈旧报告（本前缀仅 bazel coverage 一趟）。
            for stale in self.asan_log_dir.glob(f"{asan_prefix}.*"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            # asan.bazelrc（--config=asan）：-O0 / -fsanitize=address 全套。
            # --test_env 只作用于 bazel coverage 阶段实际执行的测试动作；
            # 沙箱默认拦工作区外的绝对路径写，须显式放开日志目录。
            args = args + [
                "--config=asan",
                f"--test_env=ASAN_OPTIONS={self._asan_options(asan_prefix)}",
                f"--sandbox_writable_path={self.asan_log_dir}",
            ]

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

        # ASAN -O0 全量重编远慢于常规构建（动作缓存键全变），放宽超时。
        timeout = 7200 if self.enable_asan else 1800

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Build timeout ({timeout // 60} minutes)"
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

        if self.enable_asan:
            # 直接执行二进制（无 bazel 沙箱），日志用 asan_<target_id> 前缀，
            # 与 build_target 的 bazel coverage 一趟（独立前缀）分开，同一
            # 发现只计一次。recover 模式下报告落盘、进程继续、用例不会因
            # ASAN 发现而失败——结论以日志为准。二进制动态链接 libasan.so.8，
            # 无需 LD_PRELOAD。
            for stale in self.asan_log_dir.glob(f"asan_{target_id}.*"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            test_env["ASAN_OPTIONS"] = self._asan_options(f"asan_{target_id}")

        cmd = [str(binary_path), f"--gtest_filter={gtest_filter}"]

        print(f"Running gtest: {target_id}")
        print(f"Command: {' '.join(cmd)}")

        coverage_file = None
        asan_error_count = 0
        asan_report_files: list[Path] = []

        # ASAN -O0 插桩后执行显著变慢，放宽超时。
        timeout = 1800 if self.enable_asan else 600

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                env=test_env,
                timeout=timeout,
            )

            duration = time.time() - start_time
            output = result.stdout

            # Collect coverage data if enabled
            if self.enable_coverage:
                coverage_file = self._collect_coverage_data(target_id, target)

            # 汇总本 target 的 ASAN 报告（日志文件里数 ERROR 行）。
            if self.enable_asan:
                asan_report_files, asan_error_count = (
                    self._collect_asan_reports(target_id)
                )

            # Parse gtest output
            cases_run = 0
            cases_failed = 0
            failed_cases: list[str] = []
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
                elif line.startswith("[  FAILED  ]"):
                    # 计数行（失败数）："[  FAILED  ] 1 test, listed below:"
                    # 单数不带 s，用 " tests" 子串匹配会漏。
                    m = re.match(r"\[\s*FAILED\s*\]\s+(\d+)\s+tests?\b", line)
                    if m:
                        cases_failed = int(m.group(1))
                    else:
                        # 具名失败行：[  FAILED  ] Suite.Test
                        m = re.match(r"\[\s*FAILED\s*\]\s+(\S+)\s*$", line)
                        if m:
                            failed_cases.append(m.group(1))

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
                failed_cases=failed_cases,
                coverage_file=coverage_file,
                asan_error_count=asan_error_count,
                asan_report_files=asan_report_files,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            if self.enable_asan:
                asan_report_files, asan_error_count = (
                    self._collect_asan_reports(target_id)
                )
            return TestResult(
                target_id=target_id,
                target=target,
                kind="gtest",
                success=False,
                duration=duration,
                output="",
                error=f"Test timeout ({timeout // 60} minutes)",
                coverage_file=coverage_file,
                asan_error_count=asan_error_count,
                asan_report_files=asan_report_files,
            )
        except Exception as e:
            duration = time.time() - start_time
            if self.enable_asan:
                asan_report_files, asan_error_count = (
                    self._collect_asan_reports(target_id)
                )
            return TestResult(
                target_id=target_id,
                target=target,
                kind="gtest",
                success=False,
                duration=duration,
                output="",
                error=f"Test execution error: {e}",
                coverage_file=coverage_file,
                asan_error_count=asan_error_count,
                asan_report_files=asan_report_files,
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

