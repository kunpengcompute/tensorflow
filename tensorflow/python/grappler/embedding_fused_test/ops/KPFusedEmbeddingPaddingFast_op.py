import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedEmbeddingPaddingFastOp(input0, input1, input2, input3, pack, dims):
        cast = tf.cast(input0, tf.int32)
        begin = tf.constant([0], dtype=tf.int32)
        end = tf.constant([1], dtype=tf.int32)
        strides = tf.constant([1], dtype=tf.int32)
        hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
        sub_out = hash_rows - input2
        if dims == 3:
            fill_shape = tf.stack([sub_out, pack, 5], axis=0)
            fill = tf.fill(fill_shape, tf.constant(0, dtype=tf.float32))
        else:
            pack_op = tf.stack([sub_out, pack], axis=0)
            fill = tf.fill(pack_op, tf.constant(0, dtype=tf.float32))
        concat = tf.concat([input1, fill], 0)
        reshape = tf.reshape(concat, input3)
        shape_tensor = tf.shape(reshape)
        output = tf.strided_slice(shape_tensor, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
        return output

    def KPFusedEmbeddingPaddingFast_graph(input, meta):
        # 根据核心函数入参名创建placeholder
        input0 = tf.compat.v1.placeholder(tf.int64, shape=(2,), name="input0")
        input1 = tf.compat.v1.placeholder(tf.float32, shape=input['input1'].shape, name="input1")
        input2 = tf.compat.v1.placeholder(tf.int32, shape=(), name="input2")
        input3 = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="input3")
        pack = tf.constant(input['pack'], dtype=tf.int32)

        feed_dict = {
            input0: input['input0'],
            input1: input['input1'],
            input2: input['input2'],
            input3: input['input3'],
            pack: input['pack']
        }

        result = KPFusedEmbeddingPaddingFastOp(input0, input1, input2, input3, pack, len(input['input1'].shape))
        return [result], feed_dict

    def build_input_func(input_shape, pooling_shape, reshape):
        input = {}
        input["input0"] = np.array(input_shape).astype(np.int64)
        input["input1"] = np.random.rand(*pooling_shape).astype(np.float32)
        input["input2"] = pooling_shape[0]
        input["input3"] = np.array(reshape).astype(np.int32)
        input["pack"] = pooling_shape[1]
        return input

    def build_input_2D_case_1(meta):
        return build_input_func((151 * 1, 10), (151 * 1, 10), (-1, 1510))

    def build_input_2D_case_2(meta):
        return build_input_func((151 * 1000, 10), (151 * 10, 10), (-1, 1510))

    def build_input_2D_case_3(meta):
        return build_input_func((2 * 1, 12), (2 * 1, 12), (-1, 24))

    def build_input_2D_case_4(meta):
        return build_input_func((2 * 1000, 12), (2 * 10, 12), (-1, 24))

    def build_input_3D_case_1(meta):
        return build_input_func((2 * 1000, 12), (2 * 10, 12, 5), (-1, 24))

    return [
        TestCase(
            name="KPFusedEmbeddingPaddingFast_case_1",
            op_fn=KPFusedEmbeddingPaddingFast_graph,
            input_fn=build_input_2D_case_1,
            fused_op_name="KPFusedEmbeddingPaddingFast",
            start_op_name="PartitionedCall_1/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast",
            end_op_name="PartitionedCall_1/StridedSlice_1",
            is_fused=True,
            num_iters=500,
            optimize_percent=600,
        ),
        TestCase(
            name="KPFusedEmbeddingPaddingFast_case_2",
            op_fn=KPFusedEmbeddingPaddingFast_graph,
            input_fn=build_input_2D_case_2,
            fused_op_name="KPFusedEmbeddingPaddingFast",
            start_op_name="PartitionedCall_3/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast",
            end_op_name="PartitionedCall_3/StridedSlice_1",
            is_fused=True,
            num_iters=500,
            optimize_percent=7000,
        ),
        TestCase(
            name="KPFusedEmbeddingPaddingFast_case_3",
            op_fn=KPFusedEmbeddingPaddingFast_graph,
            input_fn=build_input_2D_case_3,
            fused_op_name="KPFusedEmbeddingPaddingFast",
            start_op_name="PartitionedCall_5/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast",
            end_op_name="PartitionedCall_5/StridedSlice_1",
            is_fused=True,
            num_iters=500,
            optimize_percent=600,
        ),
        TestCase(
            name="KPFusedEmbeddingPaddingFast_case_4",
            op_fn=KPFusedEmbeddingPaddingFast_graph,
            input_fn=build_input_2D_case_4,
            fused_op_name="KPFusedEmbeddingPaddingFast",
            start_op_name="PartitionedCall_7/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast",
            end_op_name="PartitionedCall_7/StridedSlice_1",
            is_fused=True,
            num_iters=500,
            optimize_percent=800,
        ),
        TestCase(
            name="KPFusedEmbeddingPadding_3D",
            op_fn=KPFusedEmbeddingPaddingFast_graph,
            input_fn=build_input_3D_case_1,
            fused_op_name="KPFusedEmbeddingPaddingFast",
            is_fused=False,
        ),
    ]