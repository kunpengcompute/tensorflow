export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export TF_NEED_CLANG=0
export CC_OPT_FLAGS='-march=armv8.3-a+crc'

TENSORFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAZEL_OUTPUT_BASE="${BUILD_CACHE_DIR:-$TENSORFLOW_ROOT/output}"
DIST_DIR="${DISTDIR:-$TENSORFLOW_ROOT/distdir}"
mkdir -p "$BAZEL_OUTPUT_BASE" "$DIST_DIR"

export PYTHON_BIN_PATH=$(which python)
yes "" | $PYTHON_BIN_PATH configure.py

bazel --output_base="$BAZEL_OUTPUT_BASE" test --distdir="$DIST_DIR" --test_tag_filters=-no_oss,-oss_excluded,-gpu,-tpu,-benchmark-test --test_lang_filters=cc,java -k --test_timeout 300,450,1200,3600 --config=opt --test_output=errors --test_size_filters=small,medium,large --build_tests_only -- //tensorflow/core/... //tensorflow/compiler/jit/... -//tensorflow/core/tpu/...
