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
    deps = [
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
    ],
)

cc_library(
    name = "annc_graph_opt",
    srcs = ["graph_opt.cc"],
    hdrs = ["graph_opt.h"],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/costs:graph_properties",
    ],
)
