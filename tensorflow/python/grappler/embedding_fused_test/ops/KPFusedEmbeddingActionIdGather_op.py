import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedEmbeddingActionIdGatherOp(input0, input1, input2, input3, pack):
        gather1 = tf.gather(input1, input0, axis=0)
        gather2 = tf.gather(gather1, input2, axis=0)
        pack1 = tf.stack([input3, pack], axis=0)
        pack2 = tf.stack([input3, -1], axis=0)
        reshape = tf.reshape(gather2, pack2)
        fill = tf.fill(pack1, tf.constant(0, dtype=tf.float32))
        output = tf.concat([reshape, fill], axis=-1)
        return output

    def KPFusedEmbeddingActionIdGather_graph(input, meta):
        # 根据核心函数入参名创建placeholder
        input0_type = np.int32
        if "input0_type" in meta.keys():
            input0_type = meta["input0_type"]
        input2_type = np.int32
        if "input2_type" in meta.keys():
            input2_type = meta["input2_type"]
        input0 = tf.compat.v1.placeholder(input0_type, shape=input['input0'].shape, name="input0")
        input1 = tf.compat.v1.placeholder(tf.float32, shape=input['input1'].shape, name="input1")
        input2 = tf.compat.v1.placeholder(input2_type, shape=input['input2'].shape, name="input2")
        input3 = tf.compat.v1.placeholder(tf.int32, shape=(), name="input3")
        pack = tf.compat.v1.placeholder(tf.int32, shape=(), name="pack")

        feed_dict = {
            input0: input['input0'],
            input1: input['input1'],
            input2: input['input2'],
            input3: input['input3'],
            pack: input['pack']
        }
        
        result = KPFusedEmbeddingActionIdGatherOp(input0, input1, input2, input3, pack)
        return result, feed_dict

    def build_input_case_1(meta):
        indices1_shape = (8, 10)
        indices2_shape = (5, 6)
        params_shape = (80, 300)
        input = {}
        input0_type = np.int32
        if "input0_type" in meta.keys():
            input0_type = meta["input0_type"]
        input2_type = np.int32
        if "input2_type" in meta.keys():
            input2_type = meta["input2_type"]
        # 根据核心函数入参名生成输入数据
        input["input0"] = np.random.randint(0, params_shape[0], indices1_shape).astype(input0_type)
        input["input1"] = np.random.random(params_shape).astype(np.float32)
        input["input2"] = np.random.randint(0, indices1_shape[0], indices2_shape).astype(input2_type)
        input["input3"] = params_shape[0]
        input["pack"] = 1680
        return input

    def build_input_3D_case_1(meta):
        indices1_shape = (8, 10, 2)
        indices2_shape = (5, 6)
        params_shape = (160, 300)
        input = {}
        # 根据核心函数入参名生成输入数据
        input["input0"] = np.random.randint(0, params_shape[0], indices1_shape).astype(np.int32)
        input["input1"] = np.random.random(params_shape).astype(np.float32)
        input["input2"] = np.random.randint(0, indices1_shape[0], indices2_shape).astype(np.int32)
        input["input3"] = params_shape[0]
        input["pack"] = 1680
        return input

    def build_input_3D_case_2(meta):
        indices1_shape = (8, 10)
        indices2_shape = (5, 6, 2)
        params_shape = (160, 300)
        input = {}
        # 根据核心函数入参名生成输入数据
        input["input0"] = np.random.randint(0, params_shape[0], indices1_shape).astype(np.int32)
        input["input1"] = np.random.random(params_shape).astype(np.float32)
        input["input2"] = np.random.randint(0, indices1_shape[0], indices2_shape).astype(np.int32)
        input["input3"] = params_shape[0]
        input["pack"] = 1680
        return input

    return [
        TestCase(
            name="KPFusedEmbeddingActionIdGather_case_1_int32_int32",
            op_fn = KPFusedEmbeddingActionIdGather_graph,
            input_fn = build_input_case_1,
            fused_op_name = "KPFusedEmbeddingActionIdGather",
            start_op_name = "PartitionedCall_1/stack_1",
            end_op_name = "PartitionedCall_1/concat",
            is_fused=True,
            num_iters=500,
            optimize_percent=50,
        ),
        TestCase(
            name="KPFusedEmbeddingActionIdGather_case_1_int64_int32",
            op_fn = KPFusedEmbeddingActionIdGather_graph,
            input_fn = build_input_case_1,
            fused_op_name = "KPFusedEmbeddingActionIdGather",
            start_op_name = "PartitionedCall_3/stack_1",
            end_op_name = "PartitionedCall_3/concat",
            is_fused=True,
            num_iters=500,
            optimize_percent=50,
            meta = {"input0_type": np.int64},
        ),
        TestCase(
            name="KPFusedEmbeddingActionIdGather_case_1_int32_int64",
            op_fn = KPFusedEmbeddingActionIdGather_graph,
            input_fn = build_input_case_1,
            fused_op_name = "KPFusedEmbeddingActionIdGather",
            start_op_name = "PartitionedCall_5/stack_1",
            end_op_name = "PartitionedCall_5/concat",
            is_fused=True,
            num_iters=500,
            optimize_percent=50,
            meta = {"input2_type": np.int64},
        ),
        TestCase(
            name="KPFusedEmbeddingActionIdGather_case_1_int64_int64",
            op_fn = KPFusedEmbeddingActionIdGather_graph,
            input_fn = build_input_case_1,
            fused_op_name = "KPFusedEmbeddingActionIdGather",
            start_op_name = "PartitionedCall_7/stack_1",
            end_op_name = "PartitionedCall_7/concat",
            is_fused=True,
            num_iters=500,
            optimize_percent=50,
            meta = {"input0_type": np.int64, "input2_type": np.int64},
        ),
        TestCase(
            name="KPFusedEmbeddingActionIdGather_3D_case_1",
            op_fn = KPFusedEmbeddingActionIdGather_graph,
            input_fn = build_input_3D_case_1,
            fused_op_name = "KPFusedEmbeddingActionIdGather",
            is_fused=False,
        ),
        TestCase(
            name="KPFusedEmbeddingActionIdGather_3D_case_2",
            op_fn = KPFusedEmbeddingActionIdGather_graph,
            input_fn = build_input_3D_case_2,
            fused_op_name = "KPFusedEmbeddingActionIdGather",
            is_fused=False,
        ),
    ]