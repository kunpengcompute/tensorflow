"""Tests for the Unique -> DynamicStitch -> SparseSegment fusion pattern."""

import numpy as np
import tensorflow as tf

from framework.runner import TestCase


TABLE_CONFIGS = (
    ("12_20G", 12, 44739242, 10),
    ("8_1G", 8, 8388609, 4),
    ("8_500M", 8, 4194304, 4),
)


def _build_graph_core(inputs, meta, include_shape_output):
    num_tables = meta["num_tables"]
    x = tf.compat.v1.placeholder(tf.int64, shape=inputs["x"].shape, name="x")
    slice_input = tf.compat.v1.placeholder(
        tf.int64, shape=inputs["slice_input"].shape, name="slice_input"
    )
    begin = tf.constant(inputs["begin"], dtype=tf.int32, name="begin")
    end = tf.constant(inputs["end"], dtype=tf.int32, name="end")
    strides = tf.constant(inputs["strides"], dtype=tf.int32, name="strides")
    emb_tables = []
    feed_dict = {x: inputs["x"], slice_input: inputs["slice_input"]}
    for index in range(num_tables):
        table = tf.compat.v1.get_variable(
            "emb_table_%d" % index,
            shape=(meta["table_rows"], meta["embedding_width"]),
            dtype=tf.float32,
            initializer=tf.compat.v1.zeros_initializer(),
            trainable=False,
        )
        emb_tables.append(table.read_value())

    unique_values, unique_indices = tf.unique(x)
    group_ids = tf.cast(tf.math.floormod(unique_values, num_tables), tf.int32)
    chunk_indices = tf.math.floordiv(unique_values, num_tables)
    original_indices = tf.range(0, tf.size(unique_values), 1)
    stitch_indices = tf.dynamic_partition(
        original_indices, group_ids, num_partitions=num_tables
    )
    table_indices = tf.dynamic_partition(
        chunk_indices, group_ids, num_partitions=num_tables
    )
    gathered = [
        tf.gather(emb_tables[index], table_indices[index], axis=0)
        for index in range(num_tables)
    ]
    stitched = tf.raw_ops.ParallelDynamicStitch(
        indices=stitch_indices, data=gathered
    )

    segment_ids = tf.strided_slice(
        slice_input,
        begin,
        end,
        strides,
        begin_mask=1,
        end_mask=1,
        shrink_axis_mask=2,
    )
    segment_ids = tf.cast(segment_ids, tf.int32)
    reduce_mode = meta.get("reduce_mode", "sum")
    if reduce_mode == "mean":
        reduced = tf.sparse.segment_mean(stitched, unique_indices, segment_ids)
    elif reduce_mode == "sqrt_n":
        reduced = tf.sparse.segment_sqrt_n(stitched, unique_indices, segment_ids)
    else:
        reduced = tf.sparse.segment_sum(stitched, unique_indices, segment_ids)
    reduced_output = tf.identity(reduced)
    if not include_shape_output:
        return (reduced_output,), feed_dict

    reduced_shape = tf.shape(reduced)
    shape_output = tf.strided_slice(
        reduced_shape,
        [0],
        [1],
        [1],
        shrink_axis_mask=1,
    )
    return (reduced_output, tf.identity(shape_output)), feed_dict


def _build_graph(inputs, meta):
    """Build the minimal graph needed by the pooling rewriter."""
    return _build_graph_core(inputs, meta, include_shape_output=False)


def _build_input(meta):
    rng = np.random.RandomState(meta.get("seed", 2026))
    num_tables = meta["num_tables"]
    rows = meta["table_rows"]
    num_ids = meta["num_ids"]
    num_segments = int(meta["num_segments"])
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if num_segments > num_ids:
        raise ValueError("num_segments cannot exceed num_ids")

    total_ids = num_tables * rows
    request_x = rng.randint(0, total_ids, size=num_ids).astype(np.int64)
    request_groups = rng.randint(0, num_segments, size=num_ids).astype(
        np.int64
    )
    missing_groups = np.setdiff1d(
        np.arange(num_segments, dtype=np.int64),
        np.unique(request_groups),
        assume_unique=True,
    )
    if missing_groups.size:
        positions = rng.choice(
            num_ids, size=missing_groups.size, replace=False
        )
        request_groups[positions] = missing_groups
    order = np.argsort(request_groups, kind="stable")
    return {
        "x": request_x[order],
        "slice_input": np.stack(
            [np.zeros(num_ids, dtype=np.int64), request_groups[order]],
            axis=1,
        ),
        "begin": np.array([0, 1], dtype=np.int32),
        "end": np.array([0, 2], dtype=np.int32),
        "strides": np.array([1, 2], dtype=np.int32),
    }


def get_test_cases():
    REDUCE_MODES = (
        ("sum", "Sum", "SparseSegmentSum"),
        ("mean", "Mean", "SparseSegmentMean"),
        ("sqrt_n", "SqrtN", "SparseSegmentSqrtN"),
    )
    cases = []
    for label, num_tables, rows, embedding_width in TABLE_CONFIGS:
        for num_ids in (128, 1024):
            for reduce_mode, op_suffix, tf_segment_op in REDUCE_MODES:
                fused_op_name = (
                    "KPFusedSparseDynamicStitchFast" + op_suffix
                    if num_tables & (num_tables - 1) == 0
                    else "KPFusedSparseDynamicStitch" + op_suffix
                )
                meta = {
                    "num_tables": num_tables,
                    "table_rows": rows,
                    "embedding_width": embedding_width,
                    "num_ids": num_ids,
                    "num_segments": 16,
                    "reduce_mode": reduce_mode,
                }
                cases.append(
                    TestCase(
                        name="KPFusedSparseDynamicStitchReduce_%s_%s_x%d"
                        % (label, reduce_mode, num_ids),
                        op_fn=_build_graph,
                        input_fn=_build_input,
                        fused_op_name=fused_op_name,
                        is_fused=True,
                        num_iters=200,
                        optimize_percent=-10000,
                        meta=meta,
                        fused_env={
                            "ANNC_FUSED_ALL": "0",
                            "ANNC_FUSED_SPS_STITCH_REDUCE": "1",
                            "ANNC_FUSED_SPS_STITCH": "0",
                            "ANNC_FUSED_SPS_REDUCE": "0",
                        },
                        unfused_env={
                            "ANNC_FUSED_ALL": "0",
                            "ANNC_FUSED_SPS_STITCH_REDUCE": "0",
                            "ANNC_FUSED_SPS_STITCH": "0",
                            "ANNC_FUSED_SPS_REDUCE": "0",
                        },
                        start_op_name="Unique",
                        end_op_name=tf_segment_op,
                        init_fn=lambda: tf.compat.v1.global_variables_initializer(),
                    )
                )
    return cases
