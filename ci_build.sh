#!/bin/bash
# TensorFlow CI Build Script for Kunpeng Platform

set -e

TF_DIR="/workspace/tensorflow"
OUTPUT_BASE="${TF_DIR}/output"
REPO_CACHE="${TF_DIR}/repo_cache"
DISTDIR="/workspace/download"
BUILD_LOG="${TF_DIR}/build_$(date +%Y%m%d_%H%M%S).log"

PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
PIP_TRUSTED_HOST="mirrors.aliyun.com"
HTTP_PROXY="http://127.0.0.1:11082"
HTTPS_PROXY="http://127.0.0.1:11082"
NO_PROXY="localhost,127.0.0.1"

# 移除 -O3，使用 bazelrc 中的默认设置
CXXOPT="-std=c++17"
COPT="-march=armv8.5-a"
BAZEL_TARGET="//tensorflow/tools/pip_package:wheel"

log_info() { echo -e "\033[32m[INFO]\033[0m $1" | tee -a "${BUILD_LOG}"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1" | tee -a "${BUILD_LOG}"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1" | tee -a "${BUILD_LOG}"; }

log_info "创建构建目录..."
mkdir -p "${OUTPUT_BASE}" "${REPO_CACHE}" "${DISTDIR}" "${TF_DIR}/dist"

if [ "${CLEAN_CACHE:-false}" = "true" ]; then
    log_warn "清理旧缓存..."
    rm -rf "${OUTPUT_BASE:?}/*"
    rm -rf "${REPO_CACHE:?}/*"
    log_info "缓存清理完成"
fi

cd "${TF_DIR}"

log_info "开始构建 TensorFlow..."
log_info "构建日志: ${BUILD_LOG}"

bazel --output_user_root="${OUTPUT_BASE}" build \
    --experimental_repo_remote_exec \
    --repository_cache="${REPO_CACHE}" \
    --distdir="${DISTDIR}" \
    --cxxopt="${CXXOPT}" \
    --host_cxxopt="${CXXOPT}" \
    --copt="${COPT}" \
    --host_copt="${COPT}" \
    --linkopt=-Wl,--stub-group-size=0x2000000 \
    --action_env=http_proxy="${HTTP_PROXY}" \
    --action_env=https_proxy="${HTTPS_PROXY}" \
    --action_env=no_proxy="${NO_PROXY}" \
    --repo_env=PIP_INDEX_URL="${PIP_INDEX_URL}" \
    --repo_env=PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" \
    --check_direct_dependencies=off \
    -c opt \
    "${BAZEL_TARGET}" 2>&1 | tee -a "${BUILD_LOG}"

BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -eq 0 ]; then
    log_info "构建成功！"
    WHEEL_FILE=$(find "${OUTPUT_BASE}" -name "tensorflow-*.whl" -type f 2>/dev/null | head -1)
    if [ -n "${WHEEL_FILE}" ]; then
        log_info "Wheel 包位置: ${WHEEL_FILE}"
        cp "${WHEEL_FILE}" "${TF_DIR}/dist/"
        log_info "已复制到: ${TF_DIR}/dist/"
        ls -lh "${TF_DIR}/dist/"
    fi
    exit 0
else
    log_error "构建失败，状态码: $BUILD_STATUS"
    exit 1
fi
