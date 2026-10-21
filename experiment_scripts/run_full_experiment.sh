#!/usr/bin/env bash
# embedding_table_lookup 一键 benchmark 对比实验
# 对比原生实现 vs ARM 优化实现

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/experiment_scripts"
source "$EXPERIMENT_DIR/common.sh"

usage() {
    cat <<'EOF'
Usage: run_full_experiment.sh [run_count] [core_range]

Examples:
  ./run_full_experiment.sh            # 默认 10 次, 核心 0-15
  ./run_full_experiment.sh 5          # 5 次, 核心 0-15
  ./run_full_experiment.sh 10 8-15    # 10 次, 核心 8-15
EOF
}

RUN_COUNT="${1:-10}"
CORE_RANGE="${2:-0-15}"

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

chmod +x "$EXPERIMENT_DIR"/*.sh

RUN_TAG="${OP_KIND}_$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${RESULTS_OP_ROOT}/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

echo "=========================================="
echo "EmbeddingTableLookup Benchmark 性能对比"
echo "=========================================="
echo "运行次数: ${RUN_COUNT}"
echo "核心范围: ${CORE_RANGE}"
echo "实验批次: ${RUN_TAG}"
echo "结果目录: ${RUN_DIR}"
echo ""

# 步骤 1: 编译原生版本 (force_ref_impl=true 的 benchmark 过滤)
echo ""
echo "=========================================="
echo "步骤 1/4: 编译 benchmark"
echo "=========================================="
"${EXPERIMENT_DIR}/01_modify_ref_impl.sh" true
"${EXPERIMENT_DIR}/02_compile_benchmark.sh"

# 步骤 2: 运行原生版本
echo ""
echo "=========================================="
echo "步骤 2/4: 运行原生版本 (force_ref_impl=true)"
echo "=========================================="
"${EXPERIMENT_DIR}/03_run_benchmark_with_monitoring.sh" "native" "${RUN_COUNT}" "${CORE_RANGE}" "${RUN_TAG}"

# 步骤 3: 编译优化版本（共用同一份 benchmark 二进制，但过滤不同的测试用例）
echo ""
echo "=========================================="
echo "步骤 3/4: 切换为 ARM 优化版本 (force_ref_impl=false)"
echo "=========================================="
"${EXPERIMENT_DIR}/01_modify_ref_impl.sh" false

# 步骤 4: 运行 ARM 优化版本
echo ""
echo "=========================================="
echo "步骤 4/4: 运行 ARM 优化版本 (force_ref_impl=false)"
echo "=========================================="
"${EXPERIMENT_DIR}/03_run_benchmark_with_monitoring.sh" "arm_optimized" "${RUN_COUNT}" "${CORE_RANGE}" "${RUN_TAG}"

# 生成对比报告
echo ""
echo "=========================================="
echo "生成对比报告"
echo "=========================================="
python3 "${EXPERIMENT_DIR}/04_parse_results.py" "${RUN_TAG}"

ln -sfn "${RUN_DIR}" "${RESULTS_OP_ROOT}/latest"

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  原生版本:  ${RUN_DIR}/native_results.txt"
echo "  ARM优化:   ${RUN_DIR}/arm_optimized_results.txt"
echo "  原生CPU:   ${RUN_DIR}/native_cpu_log.txt"
echo "  ARM CPU:   ${RUN_DIR}/arm_optimized_cpu_log.txt"
echo "  CSV报告:   ${RUN_DIR}/benchmark_comparison.csv"
echo "  latest ->  ${RUN_DIR}"
echo ""
echo "CSV 报告预览:"
head -20 "${RUN_DIR}/benchmark_comparison.csv"
echo ""
