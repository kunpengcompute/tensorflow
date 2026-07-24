#!/usr/bin/env bash
# 运行 embedding_table_lookup benchmark 并监控 CPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

VERSION_NAME="${1:-native}"
RUN_COUNT="${2:-10}"
CORE_RANGE="${3:-0-15}"
RUN_TAG="${4:-manual_$(date '+%Y%m%d_%H%M%S')}"

if [ -z "$VERSION_NAME" ]; then
    echo "Usage: $0 <native|arm_optimized> [run_count] [core_range] [run_tag]"
    echo "Example: $0 native 10 0-15"
    echo "Example: $0 arm_optimized 10 0-15 embedding_lookup_20260419_120000"
    exit 1
fi

BENCHMARK_BIN="$(find_benchmark_binary)"
if [[ -z "${BENCHMARK_BIN}" || ! -f "${BENCHMARK_BIN}" ]]; then
    echo "错误：找不到 benchmark 可执行文件 ${TARGET_NAME}" >&2
    echo "请先运行: ${SCRIPT_DIR}/02_compile_benchmark.sh" >&2
    exit 1
fi

BENCHMARK_FILTER="$(benchmark_filter_for_version "${VERSION_NAME}")"

RUN_DIR="${RESULTS_OP_ROOT}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

export TF_NUM_INTEROP_THREADS=16
export TF_NUM_INTRAOP_THREADS=16

RESULT_FILE="$RUN_DIR/${VERSION_NAME}_results.txt"
CPU_LOG_FILE="$RUN_DIR/${VERSION_NAME}_cpu_log.txt"
: > "$RESULT_FILE"
: > "$CPU_LOG_FILE"

{
    echo "=========================================="
    echo "算子: embedding_table_lookup"
    echo "版本: $VERSION_NAME"
    echo "实验批次: $RUN_TAG"
    echo "运行次数: $RUN_COUNT"
    echo "核心范围: $CORE_RANGE"
    echo "Benchmark 过滤器: $BENCHMARK_FILTER"
    echo "Benchmark 二进制: $BENCHMARK_BIN"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
} | tee "$RESULT_FILE"

# 后台 CPU 监控
echo "启动 CPU 监控..."
(
    echo "# CPU 监控数据 - $VERSION_NAME" >> "$CPU_LOG_FILE"
    echo "# 格式: 时间戳,CPU使用率%,负载1,负载5,负载15" >> "$CPU_LOG_FILE"
    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        load_avg=$(uptime | awk -F'load average:' '{print $2}' | tr -d ' ')
        echo "$timestamp,$cpu_usage,$load_avg" >> "$CPU_LOG_FILE"
        sleep 1
    done
) &
CPU_MONITOR_PID=$!

trap "kill $CPU_MONITOR_PID 2>/dev/null; exit" INT TERM EXIT

BENCHMARK_TMPDIR="${TEST_TMPDIR:-${OUTPUT_USER_ROOT}/tmp/${TARGET_NAME}}"
mkdir -p "${BENCHMARK_TMPDIR}"
echo "临时目录: ${BENCHMARK_TMPDIR}" | tee -a "$RESULT_FILE"

for i in $(seq 1 "$RUN_COUNT"); do
    echo "" | tee -a "$RESULT_FILE"
    echo "========== 第 $i 次运行 ==========" | tee -a "$RESULT_FILE"
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"

    TEST_TMPDIR="${BENCHMARK_TMPDIR}" \
    TMPDIR="${BENCHMARK_TMPDIR}" \
    taskset -c "$CORE_RANGE" "$BENCHMARK_BIN" \
        --benchmark_filter="$BENCHMARK_FILTER" 2>&1 | tee -a "$RESULT_FILE"
done

kill $CPU_MONITOR_PID 2>/dev/null
trap - INT TERM EXIT

echo "" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"
echo "全部运行完成" | tee -a "$RESULT_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RESULT_FILE"
echo "结果文件: $RESULT_FILE" | tee -a "$RESULT_FILE"
echo "CPU 日志: $CPU_LOG_FILE" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"

echo ""
echo "✓ 测试完成，结果已保存到:"
echo "  - 输出结果: $RESULT_FILE"
echo "  - CPU 日志: $CPU_LOG_FILE"
echo "  - 实验目录: $RUN_DIR"
