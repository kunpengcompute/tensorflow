#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

env_file=".env"
if [[ ! -f "${env_file}" ]]; then
  env_file=".env.tmp"
fi

set -a
source "${env_file}"
set +a

CPU_CORES="${CPU_CORES:-16}"
CPU_QUOTA="$((CPU_CORES * 100000))"
BENCHMARK_INFER_IMAGE="${BENCHMARK_INFER_IMAGE:-benchmark-infer:v1.0.0}"
BENCHMARK_INFER_CONTAINER="${BENCHMARK_INFER_CONTAINER:-benchmark_infer_workspace}"
CPUSET_CPUS="${CPUSET_CPUS:-0-79}"
CPUSET_MEMS="${CPUSET_MEMS:-0}"
HOST_WORKSPACE="${HOST_WORKSPACE:-/home/c00913906}"
export CPU_QUOTA
export BENCHMARK_INFER_IMAGE
export BENCHMARK_INFER_CONTAINER
export CPUSET_CPUS
export CPUSET_MEMS
export HOST_WORKSPACE

exec docker compose -f docker-compose.infer.yml "$@"
