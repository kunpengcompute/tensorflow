#!/usr/bin/env python3
"""解析 embedding_table_lookup benchmark 结果并生成 CSV 报告。"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TF_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = TF_ROOT / "experiment_results_by_op"
RUN_SPLIT_RE = re.compile(r"={10} 第 (\d+) 次运行 ={10}")
BENCHMARK_LINE_RE = re.compile(r"^(BM_[A-Za-z0-9_]+)\s+(\d+)\s+ns\s+(\d+)\s+ns\s+(\d+)")


@dataclass(frozen=True)
class BenchmarkPair:
    native_name: str
    optimized_name: str
    display_name: str


BENCHMARK_PAIRS = [
    BenchmarkPair(
        "BM_EmbeddingTableLookup_SmallSingleThread_true",
        "BM_EmbeddingTableLookup_SmallSingleThread_false",
        "4KKeys_Dim16_4Values_1Thread_Hit100",
    ),
    BenchmarkPair(
        "BM_EmbeddingTableLookup_MediumParallel_true",
        "BM_EmbeddingTableLookup_MediumParallel_false",
        "16KKeys_Dim32_8Values_8Threads_Hit100",
    ),
    BenchmarkPair(
        "BM_EmbeddingTableLookup_MediumParallelMiss_true",
        "BM_EmbeddingTableLookup_MediumParallelMiss_false",
        "16KKeys_Dim32_8Values_8Threads_Hit75",
    ),
    BenchmarkPair(
        "BM_EmbeddingTableLookup_LargeParallel_true",
        "BM_EmbeddingTableLookup_LargeParallel_false",
        "64KKeys_Dim64_16Values_16Threads_Hit100",
    ),
]


def parse_benchmark_output(file_path: Path, expected_names: set[str]) -> dict[str, list[int]]:
    content = file_path.read_text(encoding="utf-8")
    runs = RUN_SPLIT_RE.split(content)
    results: dict[str, list[int]] = defaultdict(list)

    for index in range(1, len(runs), 2):
        run_content = runs[index + 1]
        seen_in_run: set[str] = set()

        for raw_line in run_content.splitlines():
            match = BENCHMARK_LINE_RE.match(raw_line.strip())
            if not match:
                continue
            benchmark_name = match.group(1)
            if benchmark_name not in expected_names or benchmark_name in seen_in_run:
                continue
            results[benchmark_name].append(int(match.group(2)))
            seen_in_run.add(benchmark_name)

    return results


def calculate_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "stdev": 0, "count": 0}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "count": len(values),
    }


def generate_csv(
    native_results: dict[str, list[int]],
    arm_results: dict[str, list[int]],
    output_file: Path,
) -> None:
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Benchmark Comparison Report"])
        writer.writerow(["Generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])
        writer.writerow([
            "Benchmark Name", "Native Mean (ns)", "Native Min (ns)",
            "Native Max (ns)", "Native Stdev (ns)", "Native Runs",
            "ARM Opt Mean (ns)", "ARM Opt Min (ns)", "ARM Opt Max (ns)",
            "ARM Opt Stdev (ns)", "ARM Opt Runs",
            "Speedup Ratio", "Improvement %",
        ])

        for pair in BENCHMARK_PAIRS:
            native_vals = native_results.get(pair.native_name, [])
            arm_vals = arm_results.get(pair.optimized_name, [])
            native_stats = calculate_stats(native_vals)
            arm_stats = calculate_stats(arm_vals)
            speedup = native_stats["mean"] / arm_stats["mean"] if arm_stats["mean"] > 0 else 0
            improvement = (
                (1 - arm_stats["mean"] / native_stats["mean"]) * 100
                if native_stats["mean"] > 0 else 0
            )
            writer.writerow([
                pair.display_name,
                f"{native_stats['mean']:.0f}", f"{native_stats['min']:.0f}",
                f"{native_stats['max']:.0f}", f"{native_stats['stdev']:.0f}",
                native_stats["count"],
                f"{arm_stats['mean']:.0f}", f"{arm_stats['min']:.0f}",
                f"{arm_stats['max']:.0f}", f"{arm_stats['stdev']:.0f}",
                arm_stats["count"],
                f"{speedup:.2f}x", f"{improvement:.2f}%",
            ])

        writer.writerow([])
        writer.writerow(["=== Raw Data Details ==="])
        writer.writerow([])
        for pair in BENCHMARK_PAIRS:
            writer.writerow([f"Benchmark: {pair.display_name}"])
            writer.writerow(["Run #", f"Native ({pair.native_name})", f"ARM Opt ({pair.optimized_name})"])
            native_vals = native_results.get(pair.native_name, [])
            arm_vals = arm_results.get(pair.optimized_name, [])
            for run_index in range(max(len(native_vals), len(arm_vals))):
                writer.writerow([
                    run_index + 1,
                    native_vals[run_index] if run_index < len(native_vals) else "",
                    arm_vals[run_index] if run_index < len(arm_vals) else "",
                ])
            writer.writerow([])


def resolve_result_dir(result_arg: str | None) -> Path:
    op_root = RESULTS_ROOT / "embedding_lookup"
    if result_arg is None:
        latest_dir = op_root / "latest"
        if latest_dir.exists():
            return latest_dir.resolve()
        raise FileNotFoundError(
            "未找到 embedding_lookup 的 latest 结果目录，请显式传入 run_tag 或目录路径"
        )
    candidate = Path(result_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return (op_root / result_arg).resolve()


def main() -> None:
    if len(sys.argv) not in {1, 2}:
        print("用法: python3 04_parse_results.py [run_tag|result_dir]")
        print("      不带参数时使用 latest 符号链接")
        sys.exit(1)

    result_arg = sys.argv[1] if len(sys.argv) == 2 else None
    try:
        result_dir = resolve_result_dir(result_arg)
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        sys.exit(1)

    native_file = result_dir / "native_results.txt"
    arm_file = result_dir / "arm_optimized_results.txt"
    if not native_file.is_file() or not arm_file.is_file():
        print(f"错误：结果目录不完整: {result_dir}")
        print(f"  期望文件: {native_file}")
        print(f"  期望文件: {arm_file}")
        sys.exit(1)

    expected_names = {p.native_name for p in BENCHMARK_PAIRS} | {p.optimized_name for p in BENCHMARK_PAIRS}

    print(f"解析原生版本结果: {native_file}")
    native_results = parse_benchmark_output(native_file, expected_names)
    print(f"解析 ARM 优化版本结果: {arm_file}")
    arm_results = parse_benchmark_output(arm_file, expected_names)

    output_csv = result_dir / "benchmark_comparison.csv"
    print(f"生成 CSV 报告: {output_csv}")
    generate_csv(native_results, arm_results, output_csv)

    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)
    for pair in BENCHMARK_PAIRS:
        native_vals = native_results.get(pair.native_name, [])
        arm_vals = arm_results.get(pair.optimized_name, [])
        if not native_vals or not arm_vals:
            print(f"\n{pair.display_name}: 数据不完整，无法计算")
            continue
        native_mean = statistics.mean(native_vals)
        arm_mean = statistics.mean(arm_vals)
        print(f"\n{pair.display_name}:")
        print(f"  原生: {native_mean:.0f} ns  |  ARM优化: {arm_mean:.0f} ns")
        print(f"  加速比: {native_mean / arm_mean:.2f}x  |  提升: {(1 - arm_mean / native_mean) * 100:.2f}%")
    print(f"\n详细结果: {output_csv}")


if __name__ == "__main__":
    main()
