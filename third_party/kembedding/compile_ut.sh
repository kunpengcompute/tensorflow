#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: third_party/kembedding/compile_ut.sh [options]

  Compile and run the EmbeddingTableLookup operator unit test.

Options:
  --dist=<path>   Path to bazel --distdir (offline deps).
                  Defaults to <tensorflow-root>/distdir.
  -h, --help      Show this help.

Examples:
  ./third_party/kembedding/compile_ut.sh
  ./third_party/kembedding/compile_ut.sh --dist $DIST_DIR
  DIST_DIR=/path/to/distdir ./third_party/kembedding/compile_ut.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_BASE="${BUILD_CACHE_DIR:-${TF_ROOT}/output}"
NUMA_NODE="${NUMA_NODE:-2}"
echo "**************NUMA_NODE=$NUMA_NODE**************"
HTTP_PROXY_ADDR="${HTTP_PROXY_ADDR:-}"
HTTPS_PROXY_ADDR="${HTTPS_PROXY_ADDR:-}"
DIST_DIR="${DIST_DIR:-${TF_ROOT}/distdir}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dist=*)
      DIST_DIR="${1#*=}"
      shift
      ;;
    --dist)
      shift
      if [[ $# -eq 0 ]]; then
        echo "错误：--dist 需要一个参数" >&2
        usage
        exit 1
      fi
      DIST_DIR="$1"
      shift
      ;;
    *)
      echo "错误：未知参数 $1" >&2
      usage
      exit 1
      ;;
  esac
done

BAZEL_ENV=()
if [[ -n "${HTTP_PROXY_ADDR}" ]]; then
  BAZEL_ENV+=("http_proxy=${HTTP_PROXY_ADDR}")
fi
if [[ -n "${HTTPS_PROXY_ADDR}" ]]; then
  BAZEL_ENV+=("https_proxy=${HTTPS_PROXY_ADDR}")
fi

TARGET="//third_party/kembedding:embedding_table_lookup_op_test"
TARGET_NAME="embedding_table_lookup_op_test"

cd "${TF_ROOT}"
mkdir -p "${OUTPUT_BASE}" "${DIST_DIR}"

echo "运行单测: embedding_lookup"
echo "目标: ${TARGET}"

numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  env "${BAZEL_ENV[@]}" \
  bazel --batch \
  --output_base="${OUTPUT_BASE}" \
  test \
  --experimental_repo_remote_exec \
  --distdir="${DIST_DIR}" \
  --cache_test_results=no \
  --test_output=all \
  "${TARGET}"

TEST_LOG="$(find "${OUTPUT_BASE}" \
  -path "*/testlogs/third_party/kembedding/${TARGET_NAME}/test.log" \
  -print -quit)"

if [[ -n "${TEST_LOG}" ]]; then
  cat "${TEST_LOG}"
fi
