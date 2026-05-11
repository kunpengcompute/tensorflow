import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedGatherOp(data, slice_input, begin, dims):
        if dims == 3:
            end=[slice_input.shape.as_list()[0], begin[1] + 2, begin[2] + 2]
            strides=[1, 1, 1]
        else:
            end=[slice_input.shape.as_list()[0], begin[1] + 2]
            strides=[1, 1]
        slice_out = tf.strided_slice(
            slice_input,
            begin=begin,
            end=end,
            strides=strides,
            begin_mask=1,
            end_mask=1,
            shrink_axis_mask=2
        )

        if dims == 3:
            value, indices = tf.unique(slice_out[0])
        else:
            value, indices = tf.unique(slice_out)
        value_1, indices_1 = tf.unique(value)
        gather1 = tf.gather(data, value_1)
        gather2 = tf.gather(gather1, indices_1)
        return value, indices, gather2

    def KPFusedGather_graph(input, meta):
        data = tf.compat.v1.placeholder(tf.float32, shape=input['data'].shape, name="data")
        slice_input = tf.compat.v1.placeholder(tf.int64, shape=input['slice_input'].shape, name="slice_input")
        begin = tf.compat.v1.placeholder(tf.int32, shape=input['begin'].shape, name="begin")

        feed = {
            data: input['data'],
            slice_input: input['slice_input'],
            begin: input['begin']
        }
        shape, indices, data = KPFusedGatherOp(data, slice_input, begin, len(input['slice_input'].shape))
        
        return [shape, indices, data], feed

    def KPFusedGather_check_fn(input_a, input_b, meta):
        np.testing.assert_array_equal(input_a[0], input_b[0])
        np.testing.assert_array_equal(input_a[1], input_b[1])
        return np.allclose(input_a[2], input_b[2], rtol=1e-3, atol=1e-5)

    def build_input_2D_case_1(meta):
        input = {}
        input["data"] = np.random.rand(50, 12).astype(np.float32)
        input["slice_input"] = np.array([[10, 7], [20, 7], [30, 7]], dtype=np.int64)
        input["begin"] = np.array([0, 1], dtype=np.int32)
        return input

    def build_input_3D_case_1(meta):
        input = {}
        input["data"] = np.random.rand(50, 12, 5).astype(np.float32)
        input["slice_input"] = np.array([[10, 7], [20, 7], [30, 7]], dtype=np.int64)
        input["begin"] = np.array([0, 1], dtype=np.int32)
        return input

    def build_input_3D_case_2(meta):
        input = {}
        input["data"] = np.random.rand(50, 12).astype(np.float32)
        input["slice_input"] = np.array([[[10, 7], [20, 7], [30, 7]], [[10, 2], [20, 3], [30, 4]]], dtype=np.int64)
        input["begin"] = np.array([0, 1, 1], dtype=np.int32)
        return input

    return [
        TestCase(
            name="KPFusedGather_input_2D_case_1",
            op_fn=KPFusedGather_graph,
            input_fn=build_input_2D_case_1,
            check_fn = KPFusedGather_check_fn,
            fused_op_name = "KPFusedGather",
            start_op_name = "PartitionedCall_1/strided_slice",
            end_op_name = "PartitionedCall_1/GatherV2",
            is_fused = True,
            num_iters=500,
            optimize_percent = 400,
        ),
        TestCase(
            name="KPFusedGather_input_3D_case_1",
            op_fn=KPFusedGather_graph,
            input_fn=build_input_3D_case_1,
            check_fn = KPFusedGather_check_fn,
            fused_op_name = "KPFusedGather",
            is_fused = False,
        ),
        TestCase(
            name="KPFusedGather_input_3D_case_2",
            op_fn=KPFusedGather_graph,
            input_fn=build_input_3D_case_2,
            check_fn = KPFusedGather_check_fn,
            fused_op_name = "KPFusedGather",
            is_fused = False,
        ),
    ]