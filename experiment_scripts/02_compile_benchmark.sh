#!/usr/bin/env bash
# 编译 embedding_table_lookup benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

cd "${TF_ROOT}"

echo "=========================================="
echo "开始编译 ${TARGET_NAME}"
echo "目标: ${TARGET}"
echo "=========================================="

# Bazel 配置（可通过环境变量覆盖）
BAZEL_BIN="${BAZEL_BIN:-bazel}"
BAZEL_DISTDIR="${BAZEL_DISTDIR:-}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"

DISTDIR_ARGS=()
if [[ -n "${BAZEL_DISTDIR}" ]]; then
    DISTDIR_ARGS+=(--distdir="${BAZEL_DISTDIR}")
fi

export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"

"${BAZEL_BIN}" --batch --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --experimental_repo_remote_exec \
    "${DISTDIR_ARGS[@]}" \
    "$TARGET"

echo ""
echo "=========================================="
echo "编译完成"
echo "=========================================="

BENCHMARK_BIN="$(find_benchmark_binary)"

if [ -n "${BENCHMARK_BIN}" ] && [ -f "$BENCHMARK_BIN" ]; then
    echo "✓ 可执行文件已生成: $BENCHMARK_BIN"
else
    echo "✗ 可执行文件未找到: $TARGET_NAME"
    exit 1
fi
