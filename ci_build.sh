#!/bin/bash
# TensorFlow CI Build Script for Kunpeng Platform

set -e

TF_DIR="/workspace/tensorflow"
OUTPUT_BASE="${TF_DIR}/output"
REPO_CACHE="/workspace/repo_cache"
DISTDIR="/workspace/download"
BUILD_LOG="${TF_DIR}/build_$(date +%Y%m%d_%H%M%S).log"

PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
PIP_TRUSTED_HOST="mirrors.aliyun.com"
NO_PROXY="localhost,127.0.0.1"

# 移除 -O3，使用 bazelrc 中的默认设置
CXXOPT="-std=c++17"
COPT="-march=armv8.5-a"
BAZEL_JOBS="${BAZEL_JOBS:-64}"
BAZEL_TARGETS=(
    "//tensorflow/tools/pip_package:wheel"
    "//:predictor_server"
    "//:brpc_client"
)

log_info() { echo -e "\033[32m[INFO]\033[0m $1" | tee -a "${BUILD_LOG}"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1" | tee -a "${BUILD_LOG}"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1" | tee -a "${BUILD_LOG}"; }

log_info "创建构建目录..."
mkdir -p "${OUTPUT_BASE}" "${REPO_CACHE}" "${DISTDIR}" "${TF_DIR}/dist"

if [ "${CLEAN_CACHE:-false}" = "true" ]; then
    log_warn "清理旧缓存..."
    rm -rf "${OUTPUT_BASE:?}/*"
    log_info "缓存清理完成"
fi

cd "${TF_DIR}"

log_info "开始构建 TensorFlow wheel 和 predictor_server 测试框架..."
log_info "构建日志: ${BUILD_LOG}"
log_info "构建目标: ${BAZEL_TARGETS[*]}"

bazel --output_user_root="${OUTPUT_BASE}" build \
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
    -c opt \
    "${BAZEL_TARGETS[@]}" 2>&1 | tee -a "${BUILD_LOG}"

BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -eq 0 ]; then
    log_info "构建成功！"
    WHEEL_FILE=$(find "${OUTPUT_BASE}" -name "tensorflow-*.whl" -type f 2>/dev/null | head -1)
    if [ -n "${WHEEL_FILE}" ]; then
        log_info "Wheel 包位置: ${WHEEL_FILE}"
        cp "${WHEEL_FILE}" "${TF_DIR}/dist/"
        log_info "已复制到: ${TF_DIR}/dist/"
    fi
    cp -L \
        "${TF_DIR}/bazel-bin/predictor_server" \
        "${TF_DIR}/bazel-bin/brpc_client" \
        "${TF_DIR}/dist/"
    log_info "已保存 predictor_server 和 brpc_client 构建产物"
    ls -lh "${TF_DIR}/dist/"
    exit 0
else
    log_error "构建失败，状态码: $BUILD_STATUS"
    exit 1
fi
