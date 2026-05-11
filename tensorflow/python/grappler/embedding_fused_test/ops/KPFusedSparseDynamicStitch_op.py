import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedSparseDynamicStitchOp(x, emb_tables, num_tables):
        x_1 = tf.reshape(x, shape=[-1])  # 将输入 x 展平成一维向量 x_1
        group_ids = tf.math.floormod(x_1, num_tables)
        group_ids = tf.cast(group_ids, dtype=np.int32)
        chunk_indices = tf.math.floordiv(x_1, num_tables)
        original_indices = tf.range(0, tf.size(x_1), 1)
        a = tf.dynamic_partition(original_indices, group_ids, num_partitions=num_tables)
        b = tf.dynamic_partition(chunk_indices, group_ids, num_partitions=num_tables)
        c = [tf.gather(emb_tables[i], b[i]) for i in range(num_tables)]
        d = tf.raw_ops.ParallelDynamicStitch(indices=a, data=c)
        return d

    def KPFusedSparseDynamicStitch_graph(input, meta):
        # 根据核心函数入参名创建placeholder
        x_shape = input['x'].shape
        lst = list(x_shape)    # 1. 转 list
        lst[1] = None
        x_shape = tuple(lst)
        num_tables = meta["num_tables"]
        x = tf.compat.v1.placeholder(tf.int64, shape=x_shape, name="x")
        
        emb_tables = []
        feed_dict = {x: input['x']}
        
        for i in range(num_tables):
            placeholder = tf.compat.v1.placeholder(tf.float32, shape=input[f'emb_table_{i}'].shape, name=f'emb_table_{i}')
            emb_tables.append(placeholder)
            feed_dict[placeholder] = input[f'emb_table_{i}']
        
        result = KPFusedSparseDynamicStitchOp(x, emb_tables, num_tables)
        return result, feed_dict

    def build_input_case_1(meta):
        input = {}
        num_tables = meta["num_tables"]
        emb_dim = 10
        max_val = float('inf')

        for i in range(num_tables):
            N = np.random.randint(1000000, 44739244)
            max_val = min(N, max_val)
            input[f'emb_table_{i}'] = np.random.rand(N, emb_dim).astype(np.float32)

        input["x"] = np.random.randint(0, num_tables * max_val, size=(1000, num_tables)).astype(np.int32)  # 12个元素，每个元素范围0-143
        return input

    def build_input_case_2(meta):
        input = {}
        num_tables = meta["num_tables"]
        emb_dim = 10
        max_val = float('inf')

        for i in range(num_tables):
            N = np.random.randint(1000000, 44739244)
            max_val = min(N, max_val)
            input[f'emb_table_{i}'] = np.random.rand(N, emb_dim).astype(np.float32)

        input["x"] = np.random.randint(0, num_tables * max_val, size=(10, 100, num_tables)).astype(np.int32)  # 12个元素，每个元素范围0-143
        return input

    def build_input_3D_case_1(meta):
        input = {}
        num_tables = meta["num_tables"]
        emb_dim = 10
        max_val = float('inf')

        for i in range(num_tables):
            N = np.random.randint(1000000, 44739244)
            max_val = min(N, max_val)
            input[f'emb_table_{i}'] = np.random.rand(N, emb_dim, 1).astype(np.float32)

        input["x"] = np.random.randint(0, num_tables * max_val, size=(1000, num_tables)).astype(np.int32)  # 12个元素，每个元素范围0-143
        return input

    return [
        TestCase(
            name="KPFusedSparseDynamicStitch_case_1",
            op_fn=KPFusedSparseDynamicStitch_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseDynamicStitch",
            start_op_name="PartitionedCall_1/Reshape",
            end_op_name="PartitionedCall_1/ParallelDynamicStitch",
            is_fused=True,
            num_iters=200,
            optimize_percent=50,
            meta={"num_tables": 12},
        ),
        TestCase(
            name="KPFusedSparseDynamicStitch_case_2",
            op_fn=KPFusedSparseDynamicStitch_graph,
            input_fn=build_input_case_2,
            fused_op_name="KPFusedSparseDynamicStitch",
            is_fused=True,
            meta={"num_tables": 12},
        ),
        TestCase(
            name="KPFusedSparseDynamicStitch_3D_case_1",
            op_fn=KPFusedSparseDynamicStitch_graph,
            input_fn=build_input_3D_case_1,
            fused_op_name="KPFusedSparseDynamicStitch",
            is_fused=False,
            meta={"num_tables": 12},
        ),
    ]