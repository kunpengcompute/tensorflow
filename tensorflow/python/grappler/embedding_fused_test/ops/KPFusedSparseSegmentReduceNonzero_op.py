import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedSparseSegmentReduceNonzeroOp(data, indices, slice_input, begin, end, strides, is_mean):
        shrink_axis_mask = 2 ** len(begin) - 2 #slice_out需要仅保留一维
        slice_out = tf.strided_slice(
                    slice_input,
                    begin= begin,
                    end= end,
                    strides= strides,
                    begin_mask=1,
                    end_mask=1,
                    shrink_axis_mask=shrink_axis_mask
                )

        segment_ids = tf.cast(slice_out, dtype=tf.int32)

        if is_mean:
            sparseseg_out = tf.sparse.segment_mean(
                    data = data,
                    indices = indices,
                    segment_ids= segment_ids
                )
        else:
            sparseseg_out = tf.sparse.segment_sum(
                    data = data,
                    indices = indices,
                    segment_ids= segment_ids
                )
        zero = tf.zeros_like(sparseseg_out)
        notequal = tf.not_equal(x=sparseseg_out, y = zero)
        where_out = tf.where(notequal)
        output_shape = tf.cast(where_out, dtype=tf.int32)
        output_data = tf.gather_nd(params=sparseseg_out, indices=where_out)
        shape = tf.shape(sparseseg_out, out_type=tf.int64)
        output_ids = tf.cast(shape, dtype=tf.int32)

        return output_shape, output_ids, output_data

    def KPFusedSparseSegmentReduceNonzero_graph(input, meta):
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
        
        result = KPFusedSparseSegmentReduceNonzeroOp(data, indices, slice_input, begin, end, strides, is_mean)
        return result, feed_dict

    def build_input_case_1(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        indices_type = np.int32
        if "indices_type" in meta.keys():
            indices_type = meta["indices_type"]
        data = np.random.rand(1449).astype(np.float32) * 10
        zero_prob = 0.3
        mask = np.random.rand(1449) > zero_prob
        data[~mask] = 0
        input["data"] = data
        input["indices"] = np.random.randint(0, 1449, size=5742, dtype=indices_type)
                
        start_points = np.sort(np.random.choice(np.arange(0, 15660), size=5742, replace=False))
        end_points = start_points + np.random.randint(1, 100, size=5742)
        end_points = np.minimum(end_points, 15661)
        slice_input = np.column_stack((start_points, end_points))
        slice_input[:, 1] = slice_input[:, 0]
        input["slice_input"] = slice_input

        input["begin"] = np.array([0, 1]).astype(np.int32)
        input["end"] = np.array([0, 2]).astype(np.int32)
        input["strides"] = np.array([1, 2]).astype(np.int32)
        return input

    def build_input_3D_case_1(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        data = np.random.rand(1449, 2, 3).astype(np.float32) * 10
        zero_prob = 0.3
        mask = np.random.rand(1449, 2, 3) > zero_prob
        data[~mask] = 0
        input["data"] = data
        input["indices"] = np.random.randint(0, 1449, size=5742, dtype=np.int32)
                
        start_points = np.sort(np.random.choice(np.arange(0, 15660), size=5742, replace=False))
        end_points = start_points + np.random.randint(1, 100, size=5742)
        end_points = np.minimum(end_points, 15661)
        slice_input = np.column_stack((start_points, end_points))
        slice_input[:, 1] = slice_input[:, 0]
        input["slice_input"] = slice_input

        input["begin"] = np.array([0, 1]).astype(np.int32)
        input["end"] = np.array([0, 2]).astype(np.int32)
        input["strides"] = np.array([1, 2]).astype(np.int32)
        return input

    def build_input_3D_case_2(meta):
        input = {}
        # 根据核心函数入参名生成输入数据
        data = np.random.rand(1449).astype(np.float32) * 10
        zero_prob = 0.3
        mask = np.random.rand(1449) > zero_prob
        data[~mask] = 0
        input["data"] = data
        input["indices"] = np.random.randint(0, 1449, size=5742, dtype=np.int32)

        start_points = np.sort(np.random.choice(np.arange(0, 15660), size=5742, replace=False))
        end_points = start_points + np.random.randint(1, 100, size=5742)
        end_points = np.minimum(end_points, 15661)
        slice_input = np.column_stack((start_points, end_points))
        slice_input[:, 1] = slice_input[:, 0]
        input["slice_input"] = np.stack([slice_input, slice_input], axis=2) #3D数据

        input["begin"] = np.array([0, 1, 1]).astype(np.int32)
        input["end"] = np.array([0, 2, 1]).astype(np.int32)
        input["strides"] = np.array([1, 2, 1]).astype(np.int32)
        return input

    return [
        TestCase(
            name="KPFusedSparseSegmentReduceNonzero_sum_case_1",
            op_fn=KPFusedSparseSegmentReduceNonzero_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduceNonzero",
            start_op_name="PartitionedCall_1/StridedSlice",
            end_op_name="PartitionedCall_1/GatherNd",
            is_fused=True,
            num_iters=1000,
            optimize_percent=30,
            meta = {"is_mean": False}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduceNonzero_mean_case_1",
            op_fn=KPFusedSparseSegmentReduceNonzero_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduceNonzero",
            start_op_name="PartitionedCall_3/StridedSlice",
            end_op_name="PartitionedCall_3/GatherNd",
            is_fused=True,
            num_iters=1000,
            optimize_percent=30,
            meta = {"is_mean": True}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduceNonzero_sum_case_1_int64",
            op_fn=KPFusedSparseSegmentReduceNonzero_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduceNonzero",
            start_op_name="PartitionedCall_5/StridedSlice",
            end_op_name="PartitionedCall_5/GatherNd",
            is_fused=True,
            num_iters=1000,
            optimize_percent=30,
            meta = {"is_mean": False, "indices_type": np.int64}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduceNonzero_mean_case_1_int64",
            op_fn=KPFusedSparseSegmentReduceNonzero_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSegmentReduceNonzero",
            start_op_name="PartitionedCall_7/StridedSlice",
            end_op_name="PartitionedCall_7/GatherNd",
            is_fused=True,
            num_iters=1000,
            optimize_percent=30,
            meta = {"is_mean": True, "indices_type": np.int64}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduceNonzero_sum_3D_case_1",
            op_fn=KPFusedSparseSegmentReduceNonzero_graph,
            input_fn=build_input_3D_case_1,
            fused_op_name="KPFusedSparseSegmentReduceNonzero",
            is_fused=False,
            meta = {"is_mean": False}
        ),
        TestCase(
            name="KPFusedSparseSegmentReduceNonzero_sum_3D_case_2",
            op_fn=KPFusedSparseSegmentReduceNonzero_graph,
            input_fn=build_input_3D_case_2,
            fused_op_name="KPFusedSparseSegmentReduceNonzero",
            is_fused=False,
            meta = {"is_mean": True}
        ),
    ]