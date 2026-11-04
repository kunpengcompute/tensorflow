load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "flash_attn_graph_opt",
    srcs = ["flash_attn_graph_opt.cc"],
    hdrs = ["flash_attn_graph_opt.h"],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    copts = select({
        # KDNN 功能开关 gflag 分支（predictor_server 路径）：读点走 FLAGS_*；
        # 默认（so/wheel）走环境变量分支。
        "//:gflags_switch_on": ["-DKDNN_SWITCH_GFLAGS"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
    ] + select({
        "//:gflags_switch_on": ["@com_github_gflags_gflags//:gflags"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "annc_graph_opt",
    srcs = ["graph_opt.cc"],
    hdrs = ["graph_opt.h"],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    copts = select({
        # KDNN 功能开关 gflag 分支（predictor_server 路径）：读点走 FLAGS_*；
        # 默认（so/wheel）走环境变量分支。
        "//:gflags_switch_on": ["-DKDNN_SWITCH_GFLAGS"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/costs:graph_properties",
    ] + select({
        "//:gflags_switch_on": ["@com_github_gflags_gflags//:gflags"],
        "//conditions:default": [],
    }),
)

tf_cc_test(
    name = "graph_opt_test",
    srcs = ["graph_opt_test.cc"],
    tags = ["no_rocm"],
    deps = [
        ":flash_attn_graph_opt",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)
