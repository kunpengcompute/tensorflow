#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${BAZEL_TARGET:-//tensorflow/core/kernels:sparse_tensor_dense_matmul_op_test}"
FILTER="${BENCHMARK_FILTER:-BM_SparseTensorDenseMatmul_.*_cpu_t[0-9]+}"
# Strip trailing $: Google Benchmark appends suffixes like /threads:N,
# so the line never ends at the user pattern.
FILTER="${FILTER%\$}"
MIN_TIME="${BENCHMARK_MIN_TIME:-0.2s}"
# If MIN_TIME is a bare number (no suffix like s/ms/us/ns), append "s".
if [[ "${MIN_TIME}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  MIN_TIME="${MIN_TIME}s"
fi
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_results}"
BAZEL_OUTPUT_ROOT="${BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.bazel_output_user_root}"
TEST_TMPDIR="${TEST_TMPDIR:-${ROOT_DIR}/.bazel_test_tmpdir}"
CLEAR_PROXY="${CLEAR_PROXY:-0}"
BAZEL_EXTRA_ARGS="${BAZEL_EXTRA_ARGS:---experimental_repo_remote_exec}"

if [[ -n "${BAZEL_BIN:-}" ]]; then
  BAZEL_CMD="${BAZEL_BIN}"
else
  BAZEL_CMD="bazel"
fi

mkdir -p "${RESULT_DIR}" "${BAZEL_OUTPUT_ROOT}" "${TEST_TMPDIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULT_DIR}/sparse_tensor_dense_matmul_threaded_${TIMESTAMP}.log"

PROXY_ENV=()
if [[ "${CLEAR_PROXY}" == "1" ]]; then
  PROXY_ENV=(
    env
    -u http_proxy
    -u https_proxy
    -u HTTP_PROXY
    -u HTTPS_PROXY
    -u all_proxy
    -u ALL_PROXY
    -u no_proxy
    -u NO_PROXY
  )
fi

BUILD_CMD=(
  "${BAZEL_CMD}"
  --batch
  --output_user_root="${BAZEL_OUTPUT_ROOT}"
  build
)

if [[ -n "${BAZEL_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS_ARRAY <<< "${BAZEL_EXTRA_ARGS}"
  BUILD_CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

BUILD_CMD+=("${TARGET}")

printf 'Workspace: %s\n' "${ROOT_DIR}" | tee "${LOG_FILE}"
printf 'Bazel: %s\n' "${BAZEL_CMD}" | tee -a "${LOG_FILE}"
printf 'Target: %s\n' "${TARGET}" | tee -a "${LOG_FILE}"
printf 'Filter: %s\n' "${FILTER}" | tee -a "${LOG_FILE}"
printf 'Benchmark min time: %s\n' "${MIN_TIME}" | tee -a "${LOG_FILE}"
printf 'Clear proxy: %s\n' "${CLEAR_PROXY}" | tee -a "${LOG_FILE}"
printf 'http_proxy: %s\n' "${http_proxy:-<unset>}" | tee -a "${LOG_FILE}"
printf 'https_proxy: %s\n' "${https_proxy:-<unset>}" | tee -a "${LOG_FILE}"
printf 'Bazel extra args: %s\n' "${BAZEL_EXTRA_ARGS}" | tee -a "${LOG_FILE}"
printf 'Log: %s\n\n' "${LOG_FILE}" | tee -a "${LOG_FILE}"

(
  cd "${ROOT_DIR}"
  "${PROXY_ENV[@]}" "${BUILD_CMD[@]}"
) 2>&1 | tee -a "${LOG_FILE}"

BIN_PATH="${ROOT_DIR}/bazel-bin/tensorflow/core/kernels/sparse_tensor_dense_matmul_op_test_cpu"
if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Benchmark binary not found: ${BIN_PATH}" | tee -a "${LOG_FILE}"
  exit 1
fi

RUN_CMD=(
  "${BIN_PATH}"
  "--benchmark_filter=${FILTER}"
  "--benchmark_min_time=${MIN_TIME}"
)

(
  cd "${ROOT_DIR}"
  TEST_TMPDIR="${TEST_TMPDIR}" "${PROXY_ENV[@]}" "${RUN_CMD[@]}"
) 2>&1 | tee -a "${LOG_FILE}"

printf '\nSaved benchmark log to %s\n' "${LOG_FILE}"
