import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedSparseSegmentReduceOp(data, indices, slice_input, begin, end, strides, is_mean):
        shrink_axis_mask = 2 ** len(begin) - 2 #slice_out需要仅保留一维
        slice_out = tf.strided_slice(
                slice_input,
                begin=begin,
                end=end,
                strides=strides,
                begin_mask=1,
                end_mask=1,
                shrink_axis_mask=shrink_axis_mask
            )

        segment_ids = tf.cast(slice_out, dtype=tf.int32)
        if is_mean:
            output = tf.sparse.segment_mean(
                data=data,
                indices=indices,
                segment_ids=segment_ids
            )
        else:
            output = tf.sparse.segment_sum(
                data=data,
                indices=indices,
                segment_ids=segment_ids
            )
        
        output_shape = tf.shape(output)
        slice_out = tf.strided_slice(output_shape, begin=[0], end=[1], strides=[1], shrink_axis_mask=1)
        
        return output, slice_out

    def KPFusedSparseSegmentReduce_graph(input, meta):
        # 根据核心函数入参名创建placeholder
        indices_type = np.int32
        if "indices_type" in meta.keys():
            indices_type = meta["indices_type"]
        data = tf.compat.v1.placeholder(tf.float32, shape=input['data'].shape, name="data")
        indices = tf.compat.v1.placeholder(indices_type, shape=input['indices'].shape, name="indices")
        slice_input = tf.compat.v1.placeholder(tf.int64, shape=input['slice_input'].shape, name="slice_input")
        begin = tf.constant(input['begin'], dtype=tf.int32)
        end = tf.constant(input['end'], dtype=tf.int32)
        strides = tf.compat.v1.placeholder(tf.int32, shape=input['strides'].shape, name="strides")
        is_mean = meta['is_mean']

        feed_dict = {
            data: input['data'],
            indices: input['indices'],
            slice_input: input['slice_input'],
            begin: input['begin'],
            end: input['end'],
            strides: input['strides']
        }
        
        result = KPFusedSparseSegmentReduceOp(data, indices, slice_input, begin, end, strides, is_mean)
        return result, feed_dict

    def build_input_case_1(meta):
        input = {}
        indices_type = np.int32
        if "indices_type" in meta.keys():
            indices_type = meta["indices_type"]
        # 根据核心函数入参名生成输入数据
        input["data"] = np.random.rand(4, 3).astype(np.float32)
        input["indices"] = np.array([0, 1, 2]).astype(indices_type)
        input["slice_input"] = np.array([[0, 0], [0, 2], [1, 2]], dtype=np.int64) 
        input["begin"] = np.array([0, 1]).astype(np.int32)
        input["end"] = np.array([0, 2]).astype(np.int32)
        input["strides"] = np.array([1, 2]).astype(np.int32)
        return input

    def build_input_3D_case_1(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        input["data"] = np.random.rand(4, 3, 2).astype(np.float32)
        input["indices"] = np.array([0, 1, 2]).astype(np.int32)
        input["slice_input"] = np.array([[0, 0], [0, 2], [1, 2]], dtype=np.int64)
        input["begin"] = np.array([0, 1]).astype(np.int32)
        input["end"] = np.array([0, 2]).astype(np.int32)
        input["strides"] = np.array([1, 2]).astype(np.int32)
        return input

    def build_input_3D_case_2(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        input["data"] = np.random.rand(4, 3).astype(np.float32)
        input["indices"] = np.array([0, 1]).astype(np.int32)
        input["slice_input"] = np.array([[[0, 0], [0, 2], [1, 2]], [[0, 1], [1, 2], [2, 2]]], dtype=np.int64)
        input["begin"] = np.array([0, 1, 1]).astype(np.int32)
        input["end"] = np.array([0, 2, 1]).astype(np.int32)
        input["strides"] = np.array([1, 2, 1]).astype(np.int32)
        return input

    return [
        TestCase(
            name="KPFusedSparseSegmentReduce_sum_case_1",
            op_fn=KPFusedSparseSegmentReduce_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduce",
            start_op_name="PartitionedCall_1/StridedSlice",
            end_op_name="PartitionedCall_1/StridedSlice_1",
            is_fused=True,
            num_iters=1000,
            optimize_percent=200,
            meta = {"is_mean": False}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduce_mean_case_1",
            op_fn=KPFusedSparseSegmentReduce_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduce",
            start_op_name="PartitionedCall_3/StridedSlice",
            end_op_name="PartitionedCall_3/StridedSlice_1",
            is_fused=True,
            num_iters=1000,
            optimize_percent=200,
            meta = {"is_mean": True}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduce_sum_case_1_int64",
            op_fn=KPFusedSparseSegmentReduce_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduce",
            start_op_name="PartitionedCall_5/StridedSlice",
            end_op_name="PartitionedCall_5/StridedSlice_1",
            is_fused=True,
            num_iters=1000,
            optimize_percent=200,
            meta = {"is_mean": False, "indices_type": np.int64}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduce_mean_case_1_int64",
            op_fn=KPFusedSparseSegmentReduce_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduce",
            start_op_name="PartitionedCall_7/StridedSlice",
            end_op_name="PartitionedCall_7/StridedSlice_1",
            is_fused=True,
            num_iters=1000,
            optimize_percent=200,
            meta = {"is_mean": True, "indices_type": np.int64}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduce_sum_3D_case_1",
            op_fn=KPFusedSparseSegmentReduce_graph,
            input_fn=build_input_3D_case_1,
            fused_op_name="KPFusedSparseSegmentReduce",
            is_fused=False,
            meta = {"is_mean": False}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduce_sum_3D_case_2",
            op_fn=KPFusedSparseSegmentReduce_graph,
            input_fn=build_input_3D_case_2,
            fused_op_name="KPFusedSparseSegmentReduce",
            is_fused=False,
            meta = {"is_mean": True}
        ),
    ]