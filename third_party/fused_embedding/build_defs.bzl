"""Starlark macros for fused_embedding.
"""

def if_fused_embedding(if_true, if_false = []):
    return select({
        "@org_tensorflow//third_party/fused_embedding:build_with_fused_embedding": if_true,
        "//conditions:default": if_false,
    })
