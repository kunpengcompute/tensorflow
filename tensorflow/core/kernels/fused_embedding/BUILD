load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//tensorflow:internal"],
)

DYNAMIC_DEPS = [
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "fused_embedding_ops",
    srcs = [
        "embedding_lookup_hash_op.cc",
        "lookup_embedding_by_hash.h",
    ],
    deps = [
        "@eigen_archive//:eigen3",
        "@farmhash_archive//:farmhash",
    ] + DYNAMIC_DEPS
)

tf_cc_test(
    name = "fused_embedding_ops_test",
    srcs = ["fused_embedding_with_hash_bucket_ops_test.cc"],
    deps = [
        ":fused_embedding_ops",
        "//tensorflow/core/ops:fused_embedding_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)