#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TF_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

MODEL_NAME=${MODEL_NAME:-wd_dcn}
MACHINE_LABEL=${MACHINE_LABEL:-unknown}
OUT_DIR=${OUT_DIR:-/workspace/tmp/${MODEL_NAME}_benchmark_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8890}
THREAD_NUM=${THREAD_NUM:-16}
CLIENT_THREAD_NUM=${CLIENT_THREAD_NUM:-1}
WARMUP=${WARMUP:-5}
DURATION=${DURATION:-30}
PREWARM_QPS=${PREWARM_QPS:-}
PREWARM_DURATION=${PREWARM_DURATION:-0}
MAX_INFLIGHT=${MAX_INFLIGHT:-100}
QPS_LIST=${QPS_LIST:-"100 350 600 850"}
RUN_MODES=${RUN_MODES:-"baseline dnn opt"}
REQUEST_MIN_BATCH_SIZE=${REQUEST_MIN_BATCH_SIZE:-1}
REQUEST_MAX_BATCH_SIZE=${REQUEST_MAX_BATCH_SIZE:-100}
TIMEOUT_MS=${TIMEOUT_MS:-30000}
TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-16}
TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-16}
MODEL=${MODEL:-/workspace/${MODEL_NAME}/1}
DATA=${DATA:-/workspace/benchmark_dataset/${MODEL_NAME}.tsv}
BIN_SERVER=${BIN_SERVER:-/usr/local/bin/predictor_server}
BIN_CLIENT=${BIN_CLIENT:-/usr/local/bin/brpc_client}
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-120}
HZ=$(getconf CLK_TCK)

detect_cpu_quota_cores() {
  local quota period

  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [[ "${quota}" != "max" && -n "${period}" && "${period}" != "0" ]]; then
      awk -v q="${quota}" -v p="${period}" 'BEGIN {printf "%.6f", q / p}'
      return
    fi
  fi

  if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [[ "${quota}" != "-1" && "${quota}" -gt 0 && "${period}" -gt 0 ]]; then
      awk -v q="${quota}" -v p="${period}" 'BEGIN {printf "%.6f", q / p}'
      return
    fi
  fi

  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif getconf _NPROCESSORS_ONLN >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    echo 1
  fi
}

CPU_QUOTA_CORES=${CPU_QUOTA_CORES:-$(detect_cpu_quota_cores)}

usage() {
  cat <<'USAGE'
Usage:
  run_serving_benchmark.sh [--help]

Runs predictor_server/brpc_client benchmark sweeps for baseline, DNN, and opt modes.
Configuration is provided with environment variables.

Common variables:
  MODEL_NAME                 Model name used for default MODEL/DATA paths.
                             Default: wd_dcn
  MODEL                      SavedModel path. Default: /workspace/${MODEL_NAME}/1
  DATA                       TSV input data path. Default: /workspace/benchmark_dataset/${MODEL_NAME}.tsv
  OUT_DIR                    Output directory. Default: /workspace/tmp/${MODEL_NAME}_benchmark_<timestamp>
  PORT                       Server port. Default: 8890
  THREAD_NUM                 predictor_server thread count. Default: 16
  CLIENT_THREAD_NUM          brpc_client thread count. Default: 1
  WARMUP                     Client warmup seconds. Default: 5
  DURATION                   Client test seconds. Default: 30
  PREWARM_QPS                Optional low-pressure server prewarm QPS before
                             measured CPU/client run. Default: disabled
  PREWARM_DURATION           Optional low-pressure prewarm seconds. Default: 0
  QPS_LIST                   Space-separated target QPS list. Default: "100 350 600 850"
  MAX_INFLIGHT               Client max in-flight RPCs. Default: 100
  REQUEST_MIN_BATCH_SIZE     Min samples per RPC. Default: 1
  REQUEST_MAX_BATCH_SIZE     Max samples per RPC. Default: 100
  TIMEOUT_MS                 Client timeout in ms. Default: 30000
  BIN_SERVER                 predictor_server binary.
                             Default: /usr/local/bin/predictor_server
  BIN_CLIENT                 brpc_client binary.
                             Default: /usr/local/bin/brpc_client
  CPU_QUOTA_CORES            CPU capacity used to normalize server CPU percent.
                             Default: detected from cgroup quota, else nproc.

Mode variables:
  RUN_MODES                  Space-separated modes: baseline, dnn, or opt.
                             Default: "baseline dnn opt"

Server mode behavior:
  baseline                   Starts server with TF_ENABLE_ONEDNN_OPTS=0 and --enable_kdnn=false.
  dnn                        Starts server with TF_ENABLE_ONEDNN_OPTS=1 and --enable_kdnn=true.
  opt                        Same as dnn, plus --annc=true,
                             --annc_cf_matmul_batchnorm=2, and --annc_fused_matmul=true.
USAGE
}

proc_ticks() {
  local pid=$1
  local fields
  read -r -a fields < "/proc/${pid}/stat"
  echo $((fields[13] + fields[14]))
}

monitor_cpu() {
  local pid=$1
  local out=$2
  local hz=$3
  local prev_ticks prev_ns now_ticks now_ns dt_ticks dt_ns cpu
  prev_ticks=$(proc_ticks "${pid}")
  prev_ns=$(date +%s%N)
  echo "timestamp_ns,cpu_pct" > "${out}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 1
    if [[ ! -r "/proc/${pid}/stat" ]]; then
      break
    fi
    now_ticks=$(proc_ticks "${pid}")
    now_ns=$(date +%s%N)
    dt_ticks=$((now_ticks - prev_ticks))
    dt_ns=$((now_ns - prev_ns))
    cpu=$(awk -v ticks="${dt_ticks}" -v ns="${dt_ns}" -v hz="${hz}" \
      'BEGIN { if (ns > 0) printf "%.2f", (ticks / hz) / (ns / 1000000000.0) * 100.0; else printf "0.00" }')
    echo "${now_ns},${cpu}" >> "${out}"
    prev_ticks=${now_ticks}
    prev_ns=${now_ns}
  done
}

summary_field() {
  local field=$1
  local file=$2
  awk -F: -v key="${field}" '
    /Load test finished/ {
      in_summary=1
      next
    }
    !in_summary {
      next
    }
    {
      lhs=$1
      gsub(/^[ \t]+|[ \t]+$/, "", lhs)
    }
    lhs == key {
      gsub(/^[ \t]+|[ \t]+$/, "", $2);
      value=$2
    }
    END { print value }
  ' "${file}"
}

progress_field() {
  local field=$1
  local file=$2
  awk -F: -v key="${field}" '
    $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
      gsub(/^[ \t]+|[ \t]+$/, "", $2);
      value=$2
    }
    END { print value }
  ' "${file}"
}

normalize_mode() {
  local mode=$1
  case "${mode}" in
    baseline) echo "baseline" ;;
    dnn) echo "dnn" ;;
    opt) echo "opt" ;;
    *)
      echo "unsupported run mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

mode_gflags() {
  local mode=$1
  case "${mode}" in
    baseline)
      echo "--enable_kdnn=false"
      ;;
    dnn)
      echo "--enable_kdnn=true"
      ;;
    opt)
      echo "--enable_kdnn=true --annc=true --annc_cf_matmul_batchnorm=2 --annc_fused_matmul=true"
      ;;
  esac
}

mode_env() {
  local mode=$1
  case "${mode}" in
    baseline)
      echo "TF_ENABLE_ONEDNN_OPTS=0"
      ;;
    dnn|opt)
      echo "TF_ENABLE_ONEDNN_OPTS=1"
      ;;
  esac
}

ensure_inputs() {
  if [[ ! -x "${BIN_SERVER}" ]]; then
    echo "predictor_server not executable: ${BIN_SERVER}" >&2
    exit 1
  fi
  if [[ ! -x "${BIN_CLIENT}" ]]; then
    echo "brpc_client not executable: ${BIN_CLIENT}" >&2
    exit 1
  fi
  if [[ ! -d "${MODEL}" ]]; then
    echo "model directory not found: ${MODEL}" >&2
    exit 1
  fi
  if [[ ! -f "${DATA}" ]]; then
    echo "input data file not found: ${DATA}" >&2
    exit 1
  fi
}

stop_port_server() {
  pkill -f "predictor_server.*--port=${PORT}" 2>/dev/null || true
}

wait_server_ready() {
  local pid=$1
  local log=$2
  local waited=0
  while (( waited < SERVER_STARTUP_TIMEOUT )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server exited before becoming ready" >&2
      tail -100 "${log}" >&2
      exit 1
    fi
    if grep -q "is serving on port=${PORT}" "${log}"; then
      return
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "server did not become ready within ${SERVER_STARTUP_TIMEOUT}s" >&2
  tail -100 "${log}" >&2
  exit 1
}

write_metadata() {
  {
    echo "model_name=${MODEL_NAME}"
    echo "machine_label=${MACHINE_LABEL}"
    echo "out_dir=${OUT_DIR}"
    echo "model=${MODEL}"
    echo "data=${DATA}"
    echo "port=${PORT}"
    echo "thread_num=${THREAD_NUM}"
    echo "client_thread_num=${CLIENT_THREAD_NUM}"
    echo "tf_num_intraop_threads=${TF_NUM_INTRAOP_THREADS}"
    echo "tf_num_interop_threads=${TF_NUM_INTEROP_THREADS}"
    echo "cpu_quota_cores=${CPU_QUOTA_CORES}"
    echo "warmup_s=${WARMUP}"
    echo "duration_s=${DURATION}"
    echo "prewarm_qps=${PREWARM_QPS}"
    echo "prewarm_duration_s=${PREWARM_DURATION}"
    echo "qps_list=${QPS_LIST}"
    echo "run_modes=${RUN_MODES}"
    echo "request_batch_size=${REQUEST_MIN_BATCH_SIZE}-${REQUEST_MAX_BATCH_SIZE}"
    echo "max_inflight=${MAX_INFLIGHT}"
    echo "timeout_ms=${TIMEOUT_MS}"
    echo "server_startup_timeout_s=${SERVER_STARTUP_TIMEOUT}"
    echo "bin_server=${BIN_SERVER}"
    echo "bin_client=${BIN_CLIENT}"
    echo "container_hostname=$(cat /proc/sys/kernel/hostname 2>/dev/null || true)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$(LC_ALL=C lscpu 2>/dev/null | awk -F: '/Model name|Vendor ID|Architecture|CPU[(]s[)]|Socket[(]s[)]|Core[(]s[)] per socket|Thread[(]s[)] per core|CPU max MHz|CPU min MHz/ {gsub(/^[ \t]+/, "", $2); print $1 "=" $2}' | paste -sd ';' -)"
  } > "${OUT_DIR}/metadata.txt"
}

run_mode() {
  local mode=$1
  local dnn=$2
  local onednn_opts=$3
  shift 3
  local extra_server_args=("$@")
  local mode_dir="${OUT_DIR}/${mode}"
  mkdir -p "${mode_dir}"

  stop_port_server

  TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS}" \
  TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS}" \
  TF_ENABLE_ONEDNN_OPTS="${onednn_opts}" \
  "${BIN_SERVER}" \
    --enable_kdnn="${dnn}" \
    --model_path="${MODEL}" \
    --thread_num="${THREAD_NUM}" \
    --port="${PORT}" \
    "${extra_server_args[@]}" \
    > "${mode_dir}/server.log" 2>&1 &
  local spid=$!

  echo "SERVER mode=${mode} gflags=\"$(mode_gflags "${mode}")\" env=\"$(mode_env "${mode}")\" pid=${spid}" | tee -a "${OUT_DIR}/run.log"
  wait_server_ready "${spid}" "${mode_dir}/server.log"

  for qps in ${QPS_LIST}; do
    local prefix="${mode_dir}/qps_${qps}"
    echo "RUN mode=${mode} qps=${qps}" | tee -a "${OUT_DIR}/run.log"

    if [[ -n "${PREWARM_QPS}" && "${PREWARM_DURATION}" -gt 0 ]]; then
      echo "PREWARM mode=${mode} qps=${qps} prewarm_qps=${PREWARM_QPS} duration=${PREWARM_DURATION}" | tee -a "${OUT_DIR}/run.log"
      "${BIN_CLIENT}" \
        --server=127.0.0.1:${PORT} \
        --input_data_path="${DATA}" \
        --thread_num="${CLIENT_THREAD_NUM}" \
        --warmup_duration_s=0 \
        --test_duration_s="${PREWARM_DURATION}" \
        --max_qps="${PREWARM_QPS}" \
        --max_inflight="${MAX_INFLIGHT}" \
        --request_min_batch_size="${REQUEST_MIN_BATCH_SIZE}" \
        --request_max_batch_size="${REQUEST_MAX_BATCH_SIZE}" \
        --timeout_ms="${TIMEOUT_MS}" \
        > "${prefix}.prewarm.log" 2>&1
    fi

    monitor_cpu "${spid}" "${prefix}.cpu.csv" "${HZ}" &
    local mpid=$!

    "${BIN_CLIENT}" \
      --server=127.0.0.1:${PORT} \
      --input_data_path="${DATA}" \
      --thread_num="${CLIENT_THREAD_NUM}" \
      --warmup_duration_s="${WARMUP}" \
      --test_duration_s="${DURATION}" \
      --max_qps="${qps}" \
      --max_inflight="${MAX_INFLIGHT}" \
      --request_min_batch_size="${REQUEST_MIN_BATCH_SIZE}" \
      --request_max_batch_size="${REQUEST_MAX_BATCH_SIZE}" \
      --timeout_ms="${TIMEOUT_MS}" \
      > "${prefix}.client.log" 2>&1

    kill "${mpid}" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true

    local avg_cpu_raw max_cpu_raw core_avg core_max avg_cpu max_cpu
    avg_cpu_raw=$(awk -F, 'NR>1 {sum+=$2; n++} END {if(n) printf "%.2f", sum/n; else printf "0.00"}' "${prefix}.cpu.csv")
    max_cpu_raw=$(awk -F, 'NR>1 {if($2>max) max=$2} END {printf "%.2f", max}' "${prefix}.cpu.csv")
    core_avg=$(awk -v c="${avg_cpu_raw}" 'BEGIN {printf "%.2f", c/100.0}')
    core_max=$(awk -v c="${max_cpu_raw}" 'BEGIN {printf "%.2f", c/100.0}')
    avg_cpu=$(awk -v c="${avg_cpu_raw}" -v cores="${CPU_QUOTA_CORES}" 'BEGIN {if (cores > 0) printf "%.2f", c / cores; else printf "0.00"}')
    max_cpu=$(awk -v c="${max_cpu_raw}" -v cores="${CPU_QUOTA_CORES}" 'BEGIN {if (cores > 0) printf "%.2f", c / cores; else printf "0.00"}')

    local actual_qps total success failure dropped avg_lat p99
    actual_qps=$(summary_field actual_success_qps "${prefix}.client.log")
    if [[ -z "${actual_qps}" ]]; then
      actual_qps=$(summary_field qps "${prefix}.client.log")
    fi
    if [[ -z "${actual_qps}" ]]; then
      actual_qps=$(progress_field QPS "${prefix}.client.log")
    fi
    total=$(summary_field total "${prefix}.client.log")
    success=$(summary_field success "${prefix}.client.log")
    failure=$(summary_field failure "${prefix}.client.log")
    dropped=$(summary_field dropped "${prefix}.client.log")
    avg_lat=$(summary_field avg_latency_us "${prefix}.client.log")
    p99=$(summary_field p99_latency_us "${prefix}.client.log")

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${mode}" "${dnn}" "${qps}" "${actual_qps}" "${total}" "${success}" \
      "${failure}" "${dropped}" "${avg_lat}" "${p99}" "${avg_cpu}" \
      "${max_cpu}" "${core_avg}" "${core_max}" | tee -a "${OUT_DIR}/results.csv"
  done

  kill "${spid}" 2>/dev/null || true
  wait "${spid}" 2>/dev/null || true
}

write_markdown() {
  awk -F, '
    NR == 1 { next }
    {
      key=$3
      mode=$1
      rows[mode SUBSEP key]=$0
      if (!(key in seen)) {
        order[++n]=key
        seen[key]=1
      }
    }
    function pct_reduction(base, opt) {
      if (base == "" || base == 0 || opt == "") return "-"
      return sprintf("%.2f%%", (base - opt) / base * 100.0)
    }
    BEGIN {
      print "| Mode | DNN | Target QPS | Actual QPS | Success | Failure | Dropped | P99 latency (us) | Server CPU avg (% quota) | Server CPU max (% quota) | Server CPU avg cores | Server CPU max cores | P99 reduction vs baseline | Avg CPU reduction vs baseline |"
      print "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    }
    END {
      for (i = 1; i <= n; i++) {
        qps=order[i]
        split(rows["baseline", qps], b, ",")
        if (b[1] != "") {
          print "| " b[1] " | " b[2] " | " b[3] " | " b[4] " | " b[6] " | " b[7] " | " b[8] " | " b[10] " | " b[11] " | " b[12] " | " b[13] " | " b[14] " | - | - |"
        }
      }
      for (i = 1; i <= n; i++) {
        qps=order[i]
        split(rows["baseline", qps], b, ",")
        split(rows["dnn", qps], k, ",")
        if (k[1] != "") {
          print "| " k[1] " | " k[2] " | " k[3] " | " k[4] " | " k[6] " | " k[7] " | " k[8] " | " k[10] " | " k[11] " | " k[12] " | " k[13] " | " k[14] " | " pct_reduction(b[10], k[10]) " | " pct_reduction(b[11], k[11]) " |"
        }
      }
      for (i = 1; i <= n; i++) {
        qps=order[i]
        split(rows["baseline", qps], b, ",")
        split(rows["opt", qps], k, ",")
        if (k[1] != "") {
          print "| " k[1] " | " k[2] " | " k[3] " | " k[4] " | " k[6] " | " k[7] " | " k[8] " | " k[10] " | " k[11] " | " k[12] " | " k[13] " | " k[14] " | " pct_reduction(b[10], k[10]) " | " pct_reduction(b[11], k[11]) " |"
        }
      }
    }
  ' "${OUT_DIR}/results.csv" > "${OUT_DIR}/results.md"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  if (( $# > 0 )); then
    echo "unsupported argument: $1" >&2
    usage >&2
    exit 1
  fi

  cd "${TF_DIR}"
  mkdir -p "${OUT_DIR}"
  ensure_inputs
  write_metadata

  for mode in ${RUN_MODES}; do
    normalized_mode=$(normalize_mode "${mode}")
    echo "MODE ${normalized_mode} gflags=\"$(mode_gflags "${normalized_mode}")\" env=\"$(mode_env "${normalized_mode}")\"" | tee -a "${OUT_DIR}/run.log"
  done

  echo "mode,dnn,target_qps,actual_qps,total,success,failure,dropped,avg_latency_us,p99_latency_us,server_cpu_pct_avg,server_cpu_pct_max,server_cpu_cores_avg,server_cpu_cores_max" > "${OUT_DIR}/results.csv"

  for mode in ${RUN_MODES}; do
    normalized_mode=$(normalize_mode "${mode}")
    case "${normalized_mode}" in
      baseline)
        run_mode baseline false 0
        ;;
      dnn)
        run_mode dnn true 1
        ;;
      opt)
        run_mode opt true 1 \
          --annc=true \
          --annc_cf_matmul_batchnorm=2 \
          --annc_fused_matmul=true
        ;;
    esac
  done
  write_markdown

  echo "results: ${OUT_DIR}"
  echo "markdown: ${OUT_DIR}/results.md"
}

main "$@"
