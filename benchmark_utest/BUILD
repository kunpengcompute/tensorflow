load("@rules_cc//cc:defs.bzl", "cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_test(
    name = "brpc_client_utils_test",
    srcs = ["brpc_client_utils_test.cc"],
    deps = [
        "//:brpc_client_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "dummy_tf_func_test",
    srcs = ["dummy_tf_func_test.cc"],
    deps = [
        "//:dummy_tf_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
