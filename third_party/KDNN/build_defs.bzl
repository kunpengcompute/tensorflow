def if_enable_kdnn(if_true, if_false = []):
    """Shorthand to select() if we are building with KDNN and KDNN is enabled.

    This is only effective when built with KDNN.

    Args:
        if_true: expression to evaluate if building with KDNN and KDNN is enabled      
        if_false: expression to evaluate if building without KDNN or KDNN is not enabled.

    Returns:
        A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@org_tensorflow//tensorflow:linux_aarch64": if_true,
        "//conditions:default": if_false,
    })

def kdnn_deps():
    """Shorthand for select() to pull in the correct set of KDNN library deps.
    """
    return select({
        "@org_tensorflow//tensorflow:linux_aarch64": ["//third_party/KDNN:kdnn_adapter"],
        "//conditions:default": [],
    })


def reorder_instance(name, in_type, out_type, headers):
    """Generates a cc_library that instantiates ReorderDefault<in_type, out_type>.

    This macro compiles the generic reorder instance source file with specific
    IN_TYPE and OUT_TYPE macro definitions to produce a concrete template
    instantiation. It is used internally by the KDNN library to support multiple
    data type combinations.

    Args:
        name: Name of the generated cc_library target.
        in_type: C++ type name for input (e.g., "float", "__fp16").
        out_type: C++ type name for output (e.g., "int8_t", "float").
    """
    native.cc_library(
        name = name,
        srcs = ["src/kernels/reorder/generic/default/kdnn_reorder_instances.cpp"],
        hdrs = headers,
        copts = select({
            "//third_party/KDNN:platform_kp920f": [
                "-DIN_TYPE={}".format(in_type),
                "-DOUT_TYPE={}".format(out_type),
                "-march=armv9.2-a+fp+simd+bf16+sve+sve2+sme",
            ],
            "//conditions:default": [
                "-DIN_TYPE={}".format(in_type),
                "-DOUT_TYPE={}".format(out_type),
                "-march=armv8.5-a+fp+simd+bf16+sve",
            ],
        }),
        includes = [
            "src",
            "include",
        ],
        visibility = ["//visibility:private"],
    )

def fast_reorder_sve_same_layouts_instance(name, in_type, out_type, headers):
    """Generates a cc_library that instantiates the SVE fast-path reorder kernel
    for same-layout tensor transformations.

    This macro compiles the SVE-specific fast reorder source file with concrete
    IN_TYPE and OUT_TYPE macro definitions. The resulting library provides a
    vectorized SVE implementation optimized for cases where input and output
    tensors share the same logical layout (e.g., NCHW → NCHW with type cast).

    It is intended for use on ARM platforms supporting SVE (ARMv8.5-A or newer).
    On KP920F (ARMv9.2-A), additional features like SVE2 and SME are enabled via
    the build configuration, but the core SVE path remains valid across all
    SVE-capable targets.

    Args:
        name: Name of the generated cc_library target.
        in_type: C++ type name for input (e.g., "float", "__fp16", "__bf16").
        out_type: C++ type name for output (e.g., "float", "int8_t").
    """
    native.cc_library(
        name = name,
        srcs = ["src/kernels/reorder/sve/same_layouts/kdnn_fast_reorder_instances.cpp"],
        hdrs = headers,
        copts = [
            "-DIN_TYPE={}".format(in_type),
            "-DOUT_TYPE={}".format(out_type),
            "-march=armv8.5-a+fp+simd+bf16+sve",
        ],
        includes = ["src", "include"],
        visibility = ["//visibility:private"],
    )


def generate_reorder_instances(headers):
    """Generates cc_library targets for all supported reorder type combinations.

    This function iterates over all pairs of input and output data types and
    creates both generic and SVE-optimized reorder instantiation libraries.
    It serves as a workaround for Bazel's restriction that BUILD files cannot
    contain loops, by encapsulating the logic in a Starlark macro.

    The generated targets cover all combinations of the following precisions:
    float (f32), __fp16 (f16), __bf16 (bf16), int8_t (s8), uint8_t (u8),
    and int32_t (s32).

    Returns:
        A list of labels (strings) referencing all generated reorder instance
        cc_library targets, suitable for use in a cc_library's deps attribute.
    """
    
    PRECISIONS = {
        "float": "f32",
        "__fp16": "f16",
        "__bf16": "bf16",
        "int8_t": "s8",
        "uint8_t": "u8",
        "int32_t": "s32",
    }

    all_instance_targets = []

    for in_type, in_short in PRECISIONS.items():
        for out_type, out_short in PRECISIONS.items():
            # 1. Generate generic reorder instance
            generic_name = "reorder_{}_{}".format(in_short, out_short)
            reorder_instance(
                name = generic_name,
                in_type = in_type,
                out_type = out_type,
                headers = headers,
            )
            all_instance_targets.append(":" + generic_name)

            # 2. Generate SVE fast-path reorder instance
            sve_name = "reorder_sve_{}_{}".format(in_short, out_short)
            fast_reorder_sve_same_layouts_instance(
                name = sve_name,
                in_type = in_type,
                out_type = out_type,
                headers = headers,
            )
            all_instance_targets.append(":" + sve_name)
    
    return all_instance_targets