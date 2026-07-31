#/bin/bash
set -ex

TENSORFLOW_DIR=""
ENABLE_GCC12=false
ENABLE_KDNN=false
KDNN_OPTIONS=""

usage() {
    echo "Usage: $0 [--features <feature1,feature2>]"
    echo "Example: $0 --features gcc12,kdnn"
    echo "Notes: --features gcc12 is only suitable for openeuler 22.03 to set gcc12 insdead of default gcc10"
    exit 1
}

if [ -f /etc/os-release ]; then
    source /etc/os-release
fi

echo "current os: $NAME $VERSION_ID"

case "$VERSION_ID" in
    "22.03")
        echo "config gcc12 path"
        GCC12_PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/
        GCC12_LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64
        ;;
    "24.03")
        echo "use default gcc"
        ;;
    *)
        echo "unsupported os version: $VERSION_ID"
        exit 1
        ;;
esac

# 定义目标软链接路径
TARGET_LINK="/usr/bin/aarch64-linux-gnu-gcc"

# 检查文件或链接是否已经存在
if [ ! -f "$TARGET_LINK" ] && [ ! -L "$TARGET_LINK" ]; then
    echo "未检测到 $TARGET_LINK，正在创建软链接..."
    
    # 获取系统自带 gcc 的路径
    GCC_PATH=$(which gcc)
    
    if [ -n "$GCC_PATH" ]; then
        # 使用 sudo 权限创建链接（如果是 root 用户可去掉 sudo）
        ln -s "$GCC_PATH" "$TARGET_LINK"
        echo "软链接创建成功: $TARGET_LINK -> $GCC_PATH"
    else
        echo "错误: 系统未安装 gcc，请先运行 yum install gcc"
        exit 1
    fi
else
    echo "检测到 $TARGET_LINK 已存在，跳过创建步骤。"
fi

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
DIST_DIR=$TENSORFLOW_ROOT/distdir
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

export PATH=$BAZEL_PATH:$PATH
DIST_DIR="${DISTDIR:-$DIST_DIR}"
BAZEL_COMPILE_CACHE="${BUILD_CACHE_DIR:-$TENSORFLOW_ROOT/output}"
mkdir -p "$DIST_DIR" "$BAZEL_COMPILE_CACHE" "$TENSORFLOW_ROOT/output-release"

if ! command -v bazel &> /dev/null; then
    echo "Error: Bazel is not installed. Please install Bazel and try again."
    exit 1
fi

bazel version

if [ "$ENABLE_GCC12" == true ]; then
    export PATH=$GCC12_PATH:$PATH
    export LD_LIBRARY_PATH=$GCC12_LD_LIBRARY_PATH
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$GCC_VERSION" != "12" ]]; then
        echo "Error: GCC version is $GCC_VERSION. Please install GCC 12. Consider use command: yum install gcc-toolset-12-gcc*"
        exit 1
    fi
fi

if [ "$ENABLE_KDNN" == true ]; then
    KDNN_OPTIONS="--define=enable_kdnn=true"
fi

gcc --version
cd $TENSORFLOW_ROOT && \
PATH=$PATH \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
bazel --output_base=$BAZEL_COMPILE_CACHE build --distdir=$DIST_DIR \
--host_copt=-march=armv8.3-a --copt=-march=armv8.3-a --define with_default_optimizations=true \
--copt=-Wno-sign-compare --config=v2 --config=noaws \
$KDNN_OPTIONS \
//tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
