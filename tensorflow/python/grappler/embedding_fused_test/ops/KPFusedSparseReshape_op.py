import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedSparseReshapeOp(slice_input, begin, newshape, pack_const, dims):
        if dims == 3:
            end = [0, 0, 2]
            strides = [1, 1, 1]
        else:
            end = [0, 2]
            strides = [1, 1]

        slice67_out = tf.strided_slice(
                slice_input,
                begin=begin,
                end=end,
                strides=strides,
                begin_mask=1,
                end_mask=1,
                shrink_axis_mask=2
            )

        slice67_out = tf.reshape(slice67_out, [-1, 1])
        shape_out = tf.shape(slice_input)
        slice57_out = tf.strided_slice(
            shape_out, 
            begin=[0],
            end=[1],
            strides=[1],
            shrink_axis_mask=1
        )
        
        input_shape = tf.stack([slice57_out, pack_const])
        input_shape = tf.cast(input_shape, tf.int64)

        range_out = tf.range(0, slice57_out, 1)
        range_out = tf.reshape(range_out, [-1, 1])
        range_out_64 = tf.cast(range_out, dtype=tf.int64)
        concat_out = tf.concat([range_out_64, slice67_out], axis=-1)
        
        values = np.arange(slice_input.shape[0], dtype=np.float32)
        
        sparse_tensor = tf.SparseTensor(
            indices=concat_out,
            values=values,
            dense_shape=input_shape
        )
        sparse_tensor_out = tf.sparse.reshape(sparse_tensor, newshape)
        return sparse_tensor_out.indices, sparse_tensor_out.dense_shape, concat_out

    def KPFusedSparseReshape_graph(input, meta):
        # 根据核心函数入参名创建placeholder
        slice_shape = input['slice_input'].shape
        lst = list(slice_shape)    # 1. 转 list
        lst[-1] = None
        slice_shape = tuple(lst)
        slice_input = tf.compat.v1.placeholder(tf.int64, shape=slice_shape, name="slice_input")
        begin = tf.compat.v1.placeholder(tf.int32, shape=input['begin'].shape, name="begin")
        newshape = tf.compat.v1.placeholder(tf.int64, shape=input['newshape'].shape, name="newshape")
        pack_const = tf.compat.v1.placeholder(tf.int32, shape=(), name="pack_const")

        feed_dict = {
            slice_input: input['slice_input'],
            begin: input['begin'],
            newshape: input['newshape'],
            pack_const: input['pack_const']
        }
        
        result = KPFusedSparseReshapeOp(slice_input, begin, newshape, pack_const, len(slice_shape))
        return result, feed_dict

    def build_input_case_1(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        input["slice_input"] = np.array([[0, 0], [0, 1], [1, 2], [3, 4]]).astype(np.int64)
        input["begin"] = np.array([0, 1]).astype(np.int32)
        input["newshape"] = np.array([2, 4]).astype(np.int64)
        input["pack_const"] = 2
        return input

    def build_input_case_2(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        input["slice_input"] = np.array([[0, 1]]).astype(np.int64)
        input["begin"] = np.array([0, 1]).astype(np.int32)
        input["newshape"] = np.array([-1, 1]).astype(np.int64)
        input["pack_const"] = 1
        return input

    def build_input_3D_case_1(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        input["slice_input"] = np.array([[[0, 0], [0, 1], [1, 2], [3, 4]], [[1, 2], [2, 3], [3, 4], [4, 4]]]).astype(np.int64)
        input["begin"] = np.array([0, 0, 1]).astype(np.int32)
        input["newshape"] = np.array([2, 2]).astype(np.int64)
        input["pack_const"] = 2
        return input

    return [
        TestCase(
            name="KPFusedSparseReshape_case_1",
            op_fn=KPFusedSparseReshape_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseReshape",
            start_op_name="PartitionedCall_1/StridedSlice",
            end_op_name="PartitionedCall_1/SparseReshape",
            is_fused=True,
            num_iters=1000,
            optimize_percent=400,
        ),
        TestCase(
            name="KPFusedSparseReshape_case_2",
            op_fn=KPFusedSparseReshape_graph,
            input_fn=build_input_case_2,
            fused_op_name="KPFusedSparseReshape",
            start_op_name="PartitionedCall_3/StridedSlice",
            end_op_name="PartitionedCall_3/SparseReshape",
            is_fused=True,
            num_iters=1000,
            optimize_percent=800,
        ),
        TestCase(
            name="KPFusedSparseReshape_3D_case_1",
            op_fn=KPFusedSparseReshape_graph,
            input_fn=build_input_3D_case_1,
            fused_op_name="KPFusedSparseReshape",
            is_fused=False,
        ),
    ]