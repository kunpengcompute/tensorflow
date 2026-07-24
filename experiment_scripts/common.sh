#!/usr/bin/env bash
# Common variables and helpers for embedding_table_lookup benchmark experiments.

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"
RESULTS_ROOT="${TF_ROOT}/experiment_results_by_op"
OUTPUT_USER_ROOT="${TF_ROOT}/output_bench"

OP_KIND="embedding_lookup"
TARGET="//third_party/kembedding:embedding_table_lookup_benchmark"
TARGET_NAME="embedding_table_lookup_benchmark"
BENCHMARK_PREFIX="BM_EmbeddingTableLookup_"
NATIVE_BENCHMARK_FILTER="BM_EmbeddingTableLookup_.*_true$"
OPTIMIZED_BENCHMARK_FILTER="BM_EmbeddingTableLookup_.*_false$"
BAZEL_OUTPUT_ROOT="${TF_ROOT}/output_bench_${OP_KIND}"
RESULTS_OP_ROOT="${RESULTS_ROOT}/${OP_KIND}"

benchmark_filter_for_version() {
    case "${1:-}" in
        native)   echo "${NATIVE_BENCHMARK_FILTER}" ;;
        arm_optimized) echo "${OPTIMIZED_BENCHMARK_FILTER}" ;;
        *)        echo "${BENCHMARK_PREFIX}.*" ;;
    esac
}

find_benchmark_binary() {
    find "${BAZEL_OUTPUT_ROOT}" \
        -path "*/execroot/**/bin/third_party/kembedding/${TARGET_NAME}" \
        -type f -print -quit
}
