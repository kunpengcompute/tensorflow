licenses(["notice"])

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":build_defs.bzl",  "generate_reorder_instances")

config_setting(
    name = "enable_kdnn",
    define_values = {
        "enable_kdnn": "true",
    },
    visibility = ["//visibility:public"],
)

# 条件配置：当 --define=kdnn_platform=kp920f 时激活
config_setting(
    name = "platform_kp920f",
    values = {"define": "kdnn_platform=kp920f"},
)

# === 1. 排除规则 ===
BASE_EXCLUDE_PATTERNS = [
    "src/kernels/**/*.cpp",
    "src/kernels/**/*.hpp",
]

# === 2. MatMul JIT ===
MATMUL_JIT_NON_SME_SRCS = glob(
    ["src/kernels/matmul_jit/**/*.cpp"],
    exclude = ["src/kernels/matmul_jit/sme/**/*.cpp"],
)
MATMUL_JIT_SME_SRCS = glob(["src/kernels/matmul_jit/sme/**/*.cpp"])

# === 3. Reorder ===
REORDER_NON_SME_SRCS = glob(
    ["src/kernels/reorder/**/*.cpp"],
    exclude = [
        "src/kernels/reorder/sme/**/*.cpp",
        "src/kernels/reorder/generic/default/kdnn_reorder_instances.cpp",
        "src/kernels/reorder/sve/same_layouts/kdnn_fast_reorder_instances.cpp",
    ],
)
REORDER_SME_SRCS = glob(["src/kernels/reorder/sme/**/*.cpp"])

# === 4. 其他源文件 ===
EXTRA_EXCLUDES = [
    "src/kdnn_batch_normalization_bwd.cpp",
    "src/kdnn_batch_normalization_fwd.cpp",
    "src/kdnn_binary.cpp",
    "src/kdnn_concat.cpp",
    "src/kdnn_convolution.cpp",
    "src/kdnn_deconvolution.cpp",
    "src/kdnn_eltwise.cpp",
    "src/kdnn_group_normalization_bwd.cpp",
    "src/kdnn_group_normalization_fwd.cpp",
    "src/kdnn_layer_normalization.cpp",
    "src/kdnn_lrn_bwd.cpp",
    "src/kdnn_lrn_fwd.cpp",
    "src/kdnn_pooling_bwd.cpp",
    "src/kdnn_pooling_fwd.cpp",
    "src/kdnn_prelu.cpp",
    "src/kdnn_reduction.cpp",
    "src/kdnn_resampling.cpp",
    "src/kdnn_rnn_bwd.cpp",
    "src/kdnn_rnn_fwd.cpp",
    "src/kdnn_shuffle.cpp",
    "src/kdnn_softmax_bwd.cpp",
    "src/kdnn_softmax_fwd.cpp",
    "src/kdnn_sum.cpp",
]

NON_SME_CPP_SRCS = glob(
    ["src/**/*.cpp"],
    exclude = BASE_EXCLUDE_PATTERNS + ["src/**/sme/**/*.cpp"] + EXTRA_EXCLUDES,
)
C_SRCS = glob(["src/**/*.c"])
ASM_SRCS = glob(["src/**/*.S"])

# === 5. 头文件 ===
HDRS = glob([
    "include/**/*.h",
    "include/**/*.hpp",
]) + glob([
    "src/threading_impl.hpp",
    "src/vector_wrappers.hpp",
    "src/vector_apis/*.hpp",
    "src/kdnn_wrapper_helpers.hpp",
    "src/kdnn_jit.hpp",
    "src/kdnn_jit_attributes.hpp",
    "src/internal_funcs.hpp",
    "src/kernels/kernel_funcs.h",
    "src/kernels/matmul_jit/**/*.hpp",
    "src/kernels/reorder/**/*.hpp",
    "src/kernels/vml/**/*.h",
    "src/kernels/vml/**/*.hpp",
    "src/aarch64_codegen/**/*.hpp",
    "src/brgemm/*.hpp",
    "src/sparse_gemm/**/*.h",
    "src/sparse_gemm/kernel/*.S",
    "src/sparse_gemm/kernel/gemm_csr_k1x1.c",
])

# === 6.1 只包含 C++ 源文件（不含 C / ASM）===
ALL_CPP_SRCS = select({
    ":platform_kp920f": (
        NON_SME_CPP_SRCS +
        MATMUL_JIT_NON_SME_SRCS + MATMUL_JIT_SME_SRCS +
        REORDER_NON_SME_SRCS + REORDER_SME_SRCS
    ),
    "//conditions:default": (
        NON_SME_CPP_SRCS +
        MATMUL_JIT_NON_SME_SRCS +
        REORDER_NON_SME_SRCS
    ),
})

# === 6.2 只包含 C / ASM 源文件 ===
ALL_C_ASM_SRCS = select({
    ":platform_kp920f": (C_SRCS + ASM_SRCS),
    "//conditions:default": (C_SRCS + ASM_SRCS),
})

REORDER_ALL_INSTANCES = generate_reorder_instances(HDRS)

# === 7. 库定义 ===
cc_library(
    name = "kdnn_cpp",
    srcs = ALL_CPP_SRCS,
    hdrs = HDRS,
    deps = REORDER_ALL_INSTANCES,
    includes = [
        "include",
        "src",
        "src/aarch64_codegen",
        "src/brgemm",
        "src/sparse_gemm",
    ],
    linkopts = ["-lgomp"],
    copts = select({
        ":platform_kp920f": [
            "-march=armv9.2-a+fp+simd+bf16+sve+sve2+sme",
            "-include", "cstdint",
        ],
        "//conditions:default": [
            "-march=armv8.5-a+fp+simd+bf16+sve",
            "-include", "cstdint",
        ],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "kdnn_c_asm",
    srcs = ALL_C_ASM_SRCS,
    hdrs = HDRS,
    includes = [
        "include",
        "src",
        "src/aarch64_codegen",
        "src/brgemm",
        "src/sparse_gemm",
    ],
    copts = select({
        ":platform_kp920f": ["-march=armv9.2-a+fp+simd+bf16+sve+sve2+sme"],
        "//conditions:default": ["-march=armv8.5-a+fp+simd+bf16+sve"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "kdnn",
    hdrs = HDRS,
    deps = [
        ":kdnn_cpp",
        ":kdnn_c_asm",
    ],
    includes = [
        "include",
        "src", 
        "src/aarch64_codegen",
        "src/brgemm",
        "src/sparse_gemm",
    ],
    linkopts = ["-lgomp"],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kdnn_adapter",
    hdrs = ["kdnn_adapter.h", "kdnn_threadpool.h"],
    includes = ["include"],
    strip_include_prefix = "/third_party/KDNN",
    visibility = ["//visibility:public"],
    deps = [":kdnn"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)