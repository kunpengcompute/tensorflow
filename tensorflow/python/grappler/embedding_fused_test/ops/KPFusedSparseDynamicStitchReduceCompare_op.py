"""Performance comparison for the two sparse stitch fusion strategies."""

import tensorflow.compat.v1 as tf

from framework.runner import TestCase

from ops.KPFusedSparseDynamicStitchReduce_op import (
    TABLE_CONFIGS,
    _build_graph_core,
    _build_input,
)

COMPARE_CASES = (
    ("12_20G", "Sum"),
    ("8_1G", "Sum"),
    ("8_500M", "Mean"),
)


def _build_compare_graph(inputs, meta):
    """Build the graph required by both stitch-reduce and split rewriters."""
    return _build_graph_core(inputs, meta, include_shape_output=True)


def get_test_cases():
    table_shapes = {
        label: (num_tables, rows, embedding_width)
        for label, num_tables, rows, embedding_width in TABLE_CONFIGS
    }
    cases = []
    for label, op_suffix in COMPARE_CASES:
        num_tables, rows, embedding_width = table_shapes[label]

        if num_tables <= 1:
            continue
        reduce_mode = op_suffix.lower()
        num_ids = 1024
        compare_meta = {
            "num_tables": num_tables,
            "table_rows": rows,
            "embedding_width": embedding_width,
            "num_ids": num_ids,
            "num_segments": 16,
            "reduce_mode": reduce_mode,
        }
        fused_name = (
            "KPFusedSparseDynamicStitchFast" + op_suffix
            if num_tables & (num_tables - 1) == 0
            else "KPFusedSparseDynamicStitch" + op_suffix
        )
        cases.append(
            TestCase(
                name="KPFusedSparseDynamicStitchReduce_compare_%s_%s_x%d"
                % (label, reduce_mode, num_ids),
                op_fn=_build_compare_graph,
                input_fn=_build_input,
                fused_op_name=fused_name,
                is_fused=True,
                num_iters=100,
                optimize_percent=-10000,
                meta=compare_meta,

                fused_env={
                    "ANNC_FUSED_ALL": "0",
                    "ANNC_FUSED_SPS_STITCH_REDUCE": "1",
                    "ANNC_FUSED_SPS_STITCH": "0",
                    "ANNC_FUSED_SPS_REDUCE": "0",
                },

                unfused_env={
                    "ANNC_FUSED_ALL": "0",
                    "ANNC_FUSED_SPS_STITCH_REDUCE": "0",
                    "ANNC_FUSED_SPS_STITCH": "1",
                    "ANNC_FUSED_SPS_REDUCE": "1",
                },
                unfused_op_names=[
                    "KPFusedSparseDynamicStitch",
                    "KPFusedSparseSegmentReduce",
                ],
                start_op_name="Unique",
                end_op_name="StridedSlice_1/kp_fused",
                fused_start_op_name="SparseSegment%s/kp_fused" % op_suffix,
                fused_end_op_name="StridedSlice_1",
                init_fn=lambda: tf.global_variables_initializer(),
            )
        )
    return cases
