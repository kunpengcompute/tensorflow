#/bin/bash
set -x

TENSORFLOW_DIR=""
ENABLE_GCC12=false
ENABLE_KDNN=false
KDNN_OPTIONS=""

usage() {
    echo "Usage: $0 [--features <feature1,feature2>]"
    echo "Example: $0 --features gcc12,kdnn"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --features)
            if [[ -z "$2" ]]; then
                echo "Error: --features requires a value"
                usage
            fi
            IFS=',' read -ra features_array <<< "$2"
            for feature in "${features_array[@]}"; do
                case "$feature" in
                    "gcc12")
                        ENABLE_GCC12=true
                        ;;
                    "kdnn")
                        ENABLE_KDNN=true
                        ;;
                    *) 
                        echo "Warning: Unknown feature '$feature', ignoring"
                        ;;
                esac
            done
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

TENSORFLOW_ROOT=$(pwd)
DIST_DIR=$TENSORFLOW_ROOT/download
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

export PATH=$BAZEL_PATH:$PATH
DIST_DIR="${DISTDIR:-$DIST_DIR}"
BAZEL_COMPILE_CACHE="${BUILD_CACHE_DIR:-$TENSORFLOW_ROOT/output}"

if ! command -v bazel &> /dev/null; then
    echo "Error: Bazel is not installed. Please install Bazel and try again."
    exit 1
fi

bazel version

if [ "$ENABLE_GCC12" == true ]; then
    PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
    LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$GCC_VERSION" != "12" ]]; then
        echo "Error: GCC version is $GCC_VERSION. Please install GCC 12. Consider use command: yum install gcc-toolset-12-gcc*"
        exit 1
    fi

    mv /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6.bak
    ln -s /opt/openEuler/gcc-toolset-12/root/usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6
    NEED_RESTORE_GCC=true
fi

if [ "$ENABLE_KDNN" == true ]; then
    KDNN_OPTIONS="--define=enable_kdnn=true"
fi

gcc --version
cd $TENSORFLOW_ROOT && \
PATH=$PATH \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
bazel --output_user_root=$BAZEL_COMPILE_CACHE build --distdir=$DIST_DIR \
--host_copt=-march=armv8.3-a --copt=-march=armv8.3-a --define with_default_optimizations=true \
--copt=-Wno-sign-compare --config=v2 --config=noaws \
$KDNN_OPTIONS \
//tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release

if [ "$NEED_RESTORE_GCC" == true ]; then
    rm /usr/lib64/libstdc++.so.6
    mv /usr/lib64/libstdc++.so.6.bak /usr/lib64/libstdc++.so.6
fi