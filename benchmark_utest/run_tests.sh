#!/bin/bash
# Run predictor framework unit tests independently of artifact builds.
set -euo pipefail

TF_DIR="${TF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${TF_DIR}/output}"
REPO_CACHE="${REPO_CACHE:-/workspace/repo_cache}"
DISTDIR="${DISTDIR:-/workspace/download}"
BUILD_LOG="${TEST_LOG:-${TF_DIR}/unit_test_$(date +%Y%m%d_%H%M%S).log}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
CXXOPT="-std=c++17"
COPT="-march=armv8.5-a"
BAZEL_JOBS="${BAZEL_JOBS:-64}"
TEST_TARGETS=(
    "//benchmark_utest:brpc_client_utils_test"
    "//benchmark_utest:dummy_tf_func_test"
)

log_info() { echo "[INFO] $*" | tee -a "${BUILD_LOG}"; }

cd "${TF_DIR}"
mkdir -p "${OUTPUT_BASE}" "${REPO_CACHE}" "${DISTDIR}"
log_info "开始运行单元测试: ${TEST_TARGETS[*]}"

bazel --batch --output_user_root="${OUTPUT_BASE}" test \
    --jobs="${BAZEL_JOBS}" \
    --experimental_repo_remote_exec \
    --repository_cache="${REPO_CACHE}" \
    --distdir="${DISTDIR}" \
    --cxxopt="${CXXOPT}" \
    --host_cxxopt="${CXXOPT}" \
    --copt="${COPT}" \
    --host_copt="${COPT}" \
    --linkopt=-Wl,--stub-group-size=0x2000000 \
    --linkopt=-lssl \
    --linkopt=-lcrypto \
    --linkopt=-lsnappy \
    --action_env=http_proxy="${HTTP_PROXY}" \
    --action_env=https_proxy="${HTTPS_PROXY}" \
    --action_env=no_proxy="${NO_PROXY}" \
    --action_env=TF_SYSTEM_LIBS="boringssl,snappy" \
    --define=use_system_libs=openssl \
    --repo_env=PIP_INDEX_URL="${PIP_INDEX_URL}" \
    --repo_env=PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" \
    --check_direct_dependencies=off \
    --test_output=errors \
    -c opt \
    "${TEST_TARGETS[@]}" 2>&1 | tee -a "${BUILD_LOG}"


log_info "单元测试通过！"
