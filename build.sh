#!/bin/bash
set -euo pipefail

ENABLE_GCC12=false
ENABLE_KDNN=false
ENABLE_BZLMOD=true

BAZEL_BIN="/usr/local/bin/bazel-7.4.1"

usage() {
    cat <<EOF
Usage: $0 [--features <feature1,feature2>]

Example:
  $0 --features gcc12,kdnn

Supported features:
  gcc12   Use gcc-toolset-12 on openEuler 22.03
  kdnn    Add --define=enable_kdnn=true to Bazel build
EOF
    exit 1
}

log() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

err() {
    echo "[ERROR] $*" >&2
    exit 1
}

detect_os_and_toolchain() {
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        source /etc/os-release
    else
        err "/etc/os-release not found"
    fi

    log "Current OS: ${NAME:-unknown} ${VERSION_ID:-unknown}"

    case "${VERSION_ID:-}" in
        "22.03")
            log "Detected openEuler 22.03"
            GCC12_PATH="/opt/openEuler/gcc-toolset-12/root/usr/bin"
            GCC12_LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64"
            ;;
        "24.03")
            log "Detected openEuler 24.03, using default gcc"
            GCC12_PATH=""
            GCC12_LD_LIBRARY_PATH=""
            ;;
        *)
            err "Unsupported OS version: ${VERSION_ID:-unknown}"
            ;;
    esac
}

ensure_aarch64_gcc_symlink() {
    local target_link="/usr/bin/aarch64-linux-gnu-gcc"

    if [ -f "$target_link" ] || [ -L "$target_link" ]; then
        log "Detected $target_link, skip creating symlink"
        return
    fi

    local gcc_path
    gcc_path="$(command -v gcc || true)"

    if [ -z "$gcc_path" ]; then
        err "gcc not found, please install gcc first"
    fi

    log "Creating symlink: $target_link -> $gcc_path"
    ln -s "$gcc_path" "$target_link"
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --features)
                [ -n "${2:-}" ] || err "--features requires a value"
                IFS=',' read -ra features_array <<< "$2"
                for feature in "${features_array[@]}"; do
                    case "$feature" in
                        gcc12)
                            ENABLE_GCC12=true
                            ;;
                        kdnn)
                            ENABLE_KDNN=true
                            ;;
                        *)
                            warn "Unknown feature '$feature', ignoring"
                            ;;
                    esac
                done
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                err "Unknown parameter: $1"
                ;;
        esac
    done
}

prepare_env() {
    TENSORFLOW_ROOT="$(pwd)"
    DIST_DIR_DEFAULT="/home/rrx/code/jd-tf220/proxy/" #"$TENSORFLOW_ROOT/download"

    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:$PATH"
    export DIST_DIR="${DISTDIR:-$DIST_DIR_DEFAULT}"
    export BAZEL_COMPILE_CACHE="${BUILD_CACHE_DIR:-$TENSORFLOW_ROOT/output}"

    [ -x "$BAZEL_BIN" ] || err "Bazel binary not found or not executable: $BAZEL_BIN"

    if [ "$ENABLE_GCC12" = true ]; then
        [ -n "${GCC12_PATH:-}" ] || err "gcc12 path is empty for current OS"
        export PATH="$GCC12_PATH:$PATH"
        export LD_LIBRARY_PATH="${GCC12_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"

        local gcc_version
        gcc_version="$(gcc -dumpversion | cut -d. -f1)"
        if [ "$gcc_version" != "12" ]; then
            err "GCC version is $gcc_version, but gcc12 feature requires GCC 12"
        fi
    fi

    KDNN_OPTIONS=()
    if [ "$ENABLE_KDNN" = true ]; then
        KDNN_OPTIONS+=(--define=enable_kdnn=true)
    fi

    BAZEL_MOD_OPTIONS=()
    if [ "$ENABLE_BZLMOD" = true ]; then
        BAZEL_MOD_OPTIONS+=(--enable_bzlmod=true)
    else
        BAZEL_MOD_OPTIONS+=(--noenable_bzlmod)
    fi

    log "Using BAZEL_BIN=$BAZEL_BIN"
    log "Using DIST_DIR=$DIST_DIR"
    log "Using BAZEL_COMPILE_CACHE=$BAZEL_COMPILE_CACHE"

    "$BAZEL_BIN" version
    gcc --version
}

build_target() {
    local target="$1"
    local tf_system_libs="$2"

    log "Building $target with TF_SYSTEM_LIBS=$tf_system_libs"

    "$BAZEL_BIN" --output_user_root="$BAZEL_COMPILE_CACHE" build \
        "${BAZEL_MOD_OPTIONS[@]}" \
        --experimental_repo_remote_exec \
        --cxxopt=-std=c++17 \
        --host_cxxopt=-std=c++17 \
        --copt=-march=armv8.5-a \
	    --copt=-I/usr/local/include \
        --host_copt=-I/usr/local/include \
        --linkopt=-L/usr/local/lib64 \
        --host_linkopt=-L/usr/local/lib64 \
        --linkopt=-Wl,-rpath,/usr/local/lib64 \
        --host_linkopt=-Wl,-rpath,/usr/local/lib64 \
        --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} \
        --action_env=TF_SYSTEM_LIBS="$tf_system_libs" \
        --define=use_system_libs=openssl \
        --distdir="$DIST_DIR" \
        --check_direct_dependencies=off \
        "${KDNN_OPTIONS[@]}" \
        "$target"
}

main() {
    parse_args "$@"
    detect_os_and_toolchain
    ensure_aarch64_gcc_symlink
    prepare_env

    cd "$TENSORFLOW_ROOT"

    build_target "//:dummy_server" "boringssl,snappy"
    build_target "//:dummy_client" "boringssl"

    log "Build finished successfully"
}

main "$@"
