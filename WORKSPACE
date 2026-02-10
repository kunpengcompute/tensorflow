# buildifier: disable=load-on-top

workspace(name = "org_tensorflow")

# buildifier: disable=load-on-top

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["//:WORKSPACE"],
    requirements = {
        "3.9": "//:requirements_lock_3_9.txt",
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
        "3.13": "//:requirements_lock_3_13.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_configure.bzl",
    "nvshmem_configure",
)

nvshmem_configure(name = "local_config_nvshmem")

load(
    "@rules_ml_toolchain//cc_toolchain/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc_toolchain:lx64_lx64")

register_toolchains("@rules_ml_toolchain//cc_toolchain:lx64_lx64_cuda")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "bazel_skylib",
    sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz"],
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

http_archive(
    name = "rules_python",
    sha256 = "a644da969b6824cc87f8fe03b58cb92418f3d0b18b027ec2f56a0a4bc9efc7d6",
    strip_prefix = "rules_python-0.20.0",
    urls = ["https://github.com/bazelbuild/rules_python/releases/download/0.20.0/rules_python-0.20.0.tar.gz"],
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

http_archive(
    name = "hedron_compile_commands",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/3dddf205a1f5cde20faf2444c1757abe0564ff4c.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-3dddf205a1f5cde20faf2444c1757abe0564ff4c",
    sha256 = "3cd0e49f0f4a6d406c1d74b53b7616f5e24f5fd319eafc1bf8eee6e14124d115",
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")


BAZEL_IO_VERSION = "4.2.2"  # 2021-12-03T09:26:35Z

BAZEL_IO_SHA256 = "4c179ce66bbfff6ac5d81b8895518096e7f750866d08da2d4a574d1b8029e914"

BAZEL_SKYLIB_VERSION = "1.1.1"  # 2021-09-27T17:33:49Z

BAZEL_SKYLIB_SHA256 = "c6966ec828da198c5d9adbaa94c05e3a1c7f21bd012a0b29ba8ddbccb2c93b0d"

BAZEL_PLATFORMS_VERSION = "0.0.4"  # 2021-02-26

BAZEL_PLATFORMS_SHA256 = "079945598e4b6cc075846f7fd6a9d0857c33a7afc0de868c2ccb96405225135d"

RULES_PROTO_TAG = "4.0.0"  # 2021-09-15T14:13:21Z

RULES_PROTO_SHA256 = "66bfdf8782796239d3875d37e7de19b1d94301e8972b3cbd2446b332429b4df1"

RULES_CC_COMMIT_ID = "0913abc3be0edff60af681c0473518f51fb9eeef"  # 2021-08-12T14:14:28Z

RULES_CC_SHA256 = "04d22a8c6f0caab1466ff9ae8577dbd12a0c7d0bc468425b75de094ec68ab4f9"


http_archive(
    name = "io_bazel",
    sha256 = BAZEL_IO_SHA256,
    strip_prefix = "bazel-" + BAZEL_IO_VERSION,
    url = "https://github.com/bazelbuild/bazel/archive/" + BAZEL_IO_VERSION + ".zip",
)

http_archive(
    name = "bazel_skylib",
    sha256 = BAZEL_SKYLIB_SHA256,
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = BAZEL_SKYLIB_VERSION),
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = BAZEL_SKYLIB_VERSION),
    ],
)

http_archive(
    name = "platforms",
    sha256 = BAZEL_PLATFORMS_SHA256,
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/{version}/platforms-{version}.tar.gz".format(version = BAZEL_PLATFORMS_VERSION),
        "https://github.com/bazelbuild/platforms/releases/download/{version}/platforms-{version}.tar.gz".format(version = BAZEL_PLATFORMS_VERSION),
    ],
)

http_archive(
    name = "rules_proto",
    sha256 = RULES_PROTO_SHA256,
    strip_prefix = "rules_proto-{version}".format(version = RULES_PROTO_TAG),
    urls = ["https://github.com/bazelbuild/rules_proto/archive/refs/tags/{version}.tar.gz".format(version = RULES_PROTO_TAG)],
)

http_archive(
    name = "rules_cc",
    sha256 = RULES_CC_SHA256,
    strip_prefix = "rules_cc-{commit_id}".format(commit_id = RULES_CC_COMMIT_ID),
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/{commit_id}.tar.gz".format(commit_id = RULES_CC_COMMIT_ID),
    ],
)

http_archive(
    name = "rules_perl",  # 2021-09-23T03:21:58Z
    sha256 = "4128dc22e438e81b7d9491841d7209d7169635294693a4ed8f760e5a720a88ad",
    strip_prefix = "rules_perl-0.5.0",
    urls = [
        "https://github.com/bazel-contrib/rules_perl/archive/refs/tags/0.5.0.tar.gz",
    ],
)

load("@rules_perl//perl:deps.bzl", "perl_register_toolchains")
perl_register_toolchains()

http_archive(
    name = "rules_foreign_cc",  # 2021-12-03T17:15:40Z
    sha256 = "1df78c7d7eed2dc21b8b325a2853c31933a81e7b780f9a59a5d078be9008b13a",
    strip_prefix = "rules_foreign_cc-0.7.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.7.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies(register_preinstalled_tools = False)

# ---------- brpc ----------
http_archive(
    name = "brpc",
    sha256 = "f674b753af71dc313d9d2dcf34f574f0a3438c9f9bb9e7e6ca500a3b0ca7ddfb",
    strip_prefix = "brpc-1.15.0",
    urls = ["https://github.com/apache/brpc/archive/refs/tags/1.15.0.tar.gz"],
    patches = ["//third_party/brpc:brpc_pb33.patch"],
    patch_args = ["-p1"],
    patch_cmds = [
        "find bazel/third_party -type f -name '*.BUILD' -exec sed -i 's|@//bazel/|@brpc//bazel/|g' {} +",
    ],
)

# ---------- gperftools ----------
http_archive(
    name = "com_github_google_gperftools",
    sha256 = "885dbbf1f25a922de0cdc78b0703c3ab93c43850e1d2f7c889e41be7c824c53d",
    strip_prefix = "gperftools-gperftools-2.17.2",
    urls = [
        "https://github.com/gperftools/gperftools/archive/refs/tags/gperftools-2.17.2.tar.gz",
    ],
)

http_archive(
    name = "boost",  # 2021-08-05T01:30:05Z
    build_file = "@com_github_nelhage_rules_boost//:BUILD.boost",
    patch_cmds = ["rm -f doc/pdf/BUILD"],
    patch_cmds_win = ["Remove-Item -Force doc/pdf/BUILD"],
    sha256 = "5347464af5b14ac54bb945dc68f1dd7c56f0dad7262816b956138fc53bcc0131",
    strip_prefix = "boost_1_77_0",
    urls = [
        "https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz",
    ],
)

http_archive(
    name = "com_github_gflags_gflags",  # 2018-11-11T21:30:10Z
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_crc32c",  # 2021-10-05T19:47:30Z
    build_file = "@brpc//bazel/third_party/crc32c:crc32c.BUILD",
    sha256 = "ac07840513072b7fcebda6e821068aa04889018f24e10e46181068fb214d7e56",
    strip_prefix = "crc32c-1.1.2",
    urls = ["https://github.com/google/crc32c/archive/1.1.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",  # 2021-05-07T23:06:39Z
    patch_args = ["-p1"],
    patches = [
	    "@brpc//bazel/third_party/glog:0001-mark-override-resolve-warning.patch",
    ],
    sha256 = "21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a",
    strip_prefix = "glog-0.5.0",
    urls = ["https://github.com/google/glog/archive/v0.5.0.zip"],
)

http_archive(
    name = "com_github_google_leveldb",  # 2021-02-23T21:51:12Z
    build_file = "@brpc//bazel/third_party/leveldb:leveldb.BUILD",
    sha256 = "9a37f8a6174f09bd622bc723b55881dc541cd50747cbd08831c2a82d620f6d76",
    strip_prefix = "leveldb-1.23",
    urls = [
        "https://github.com/google/leveldb/archive/refs/tags/1.23.tar.gz",
    ],
)

http_archive(
    name = "com_github_google_snappy",  # 2017-08-25
    build_file = "@brpc//bazel/third_party/snappy:snappy.BUILD",
    sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
    strip_prefix = "snappy-1.1.7",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/snappy/archive/1.1.7.tar.gz",
        "https://github.com/google/snappy/archive/1.1.7.tar.gz",
    ],
)

http_archive(
    name = "com_github_libevent_libevent",  # 2020-07-05T13:33:03Z
    build_file = "@brpc//bazel/third_party/event:event.BUILD",
    sha256 = "92e6de1be9ec176428fd2367677e61ceffc2ee1cb119035037a27d346b0403bb",
    strip_prefix = "libevent-2.1.12-stable",
    urls = [
        "https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz",
    ],
)

http_archive(
    name = "com_github_madler_zlib",  # 2017-01-15T17:57:23Z
    build_file = "@brpc//bazel/third_party/zlib:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://downloads.sourceforge.net/project/libpng/zlib/1.2.11/zlib-1.2.11.tar.gz",
        "https://zlib.net/fossils/zlib-1.2.11.tar.gz",
    ],
)

http_archive(
    name = "com_github_nelhage_rules_boost",  # 2021-08-27T15:46:06Z
    patch_cmds = ["sed -i 's/net_zlib_zlib/com_github_madler_zlib/g' BUILD.boost"],
    patch_cmds_win = [
        """$content = (Get-Content 'BUILD.boost') -replace "net_zlib_zlib", "com_github_madler_zlib"
Set-Content BUILD.boost -Value $content -Encoding UTF8
""",
    ],
    sha256 = "2d0b2eef7137730dbbb180397fe9c3d601f8f25950c43222cb3ee85256a21869",
    strip_prefix = "rules_boost-fce83babe3f6287bccb45d2df013a309fa3194b8",
    urls = [
        "https://github.com/nelhage/rules_boost/archive/fce83babe3f6287bccb45d2df013a309fa3194b8.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",  # 2021-09-27T18:06:52Z
    sha256 = "2f0d9c7bc770f32bda06a9548f537b63602987d5a173791485151aba28a90099",
    strip_prefix = "abseil-cpp-7143e49e74857a009e16c51f6076eb197b6ccb49",
    urls = ["https://github.com/abseil/abseil-cpp/archive/7143e49e74857a009e16c51f6076eb197b6ccb49.zip"],
)

http_archive(
    name = "com_google_googletest",  # 2021-07-09T13:28:13Z
    sha256 = "12ef65654dc01ab40f6f33f9d02c04f2097d2cd9fbe48dc6001b29543583b0ad",
    strip_prefix = "googletest-8d51ffdfab10b3fba636ae69bc03da4b54f8c235",
    urls = ["https://github.com/google/googletest/archive/8d51ffdfab10b3fba636ae69bc03da4b54f8c235.zip"],
)


http_archive(
    name = "openssl",  # 2021-12-14T15:45:01Z
    build_file = "@brpc//bazel/third_party/openssl:openssl.BUILD",
    sha256 = "f89199be8b23ca45fc7cb9f1d8d3ee67312318286ad030f5316aca6462db6c96",
    strip_prefix = "openssl-1.1.1m",
    urls = [
        "https://www.openssl.org/source/openssl-1.1.1m.tar.gz",
        "https://github.com/openssl/openssl/archive/OpenSSL_1_1_1m.tar.gz",
    ],
)

git_repository(
    name = "boringssl", # 2021-05-01T12:26:01Z
    commit = "0e6b86549db4c888666512295c3ebd4fa2a402f5", # fips-20210429
    remote = "https://github.com/google/boringssl",
)

http_archive(
    name = "org_apache_thrift",  # 2021-09-11T11:54:01Z
    build_file = "@brpc//bazel/third_party/thrift:thrift.BUILD",
    sha256 = "d5883566d161f8f6ddd4e21f3a9e3e6b8272799d054820f1c25b11e86718f86b",
    strip_prefix = "thrift-0.15.0",
    urls = ["https://archive.apache.org/dist/thrift/0.15.0/thrift-0.15.0.tar.gz"],
)
