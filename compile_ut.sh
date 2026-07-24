#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: compile_ut.sh [options]

  Compile and run the EmbeddingTableLookup operator unit test.

Options:
  --dist=<path>   Path to bazel --distdir (offline deps).
                  If omitted, --distdir is NOT passed to bazel.
  -h, --help      Show this help.

Examples:
  ./compile_ut.sh
  ./compile_ut.sh --dist $DIST_DIR
  DIST_DIR=/home/wzx/download ./compile_ut.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_USER_ROOT="${SCRIPT_DIR}/output_ut"
NUMA_NODE="${NUMA_NODE:-2}"
echo "**************NUMA_NODE=$NUMA_NODE**************"
HTTP_PROXY_ADDR="${HTTP_PROXY_ADDR:-}"
HTTPS_PROXY_ADDR="${HTTPS_PROXY_ADDR:-}"
DIST_DIR="${DIST_DIR:-}"

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

DISTDIR_ARGS=()
if [[ -n "${DIST_DIR}" ]]; then
  DISTDIR_ARGS+=(--distdir="${DIST_DIR}")
fi

TARGET="//third_party/kembedding:embedding_table_lookup_op_test"
TARGET_NAME="embedding_table_lookup_op_test"

cd "${SCRIPT_DIR}"

echo "运行单测: embedding_lookup"
echo "目标: ${TARGET}"

numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  env "${BAZEL_ENV[@]}" \
  bazel --batch \
  --output_user_root="${OUTPUT_USER_ROOT}" \
  test \
  --experimental_repo_remote_exec \
  "${DISTDIR_ARGS[@]}" \
  --cache_test_results=no \
  --test_output=all \
  "${TARGET}"

TEST_LOG="$(find "${OUTPUT_USER_ROOT}" \
  -path "*/testlogs/third_party/kembedding/${TARGET_NAME}/test.log" \
  -print -quit)"

if [[ -n "${TEST_LOG}" ]]; then
  cat "${TEST_LOG}"
fi
