import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    @tf.function
    def KPFusedSparseSelectOp(input_a, input_b, input_c, greater, equal1, equal2, equal3):
        a = tf.reshape(input_a, [-1, 1])
        b = tf.reshape(input_b, [-1, 1])
        c = tf.reshape(input_c, [-1, 1])
        output_x = a

        greater_a = tf.greater(a, greater)
        shape_reshape_a1 = tf.shape(a)
        fill_a1 = tf.fill(shape_reshape_a1, tf.constant(1, dtype=tf.float32))
        realdiv = tf.realdiv(fill_a1, tf.constant(1, dtype=tf.float32))
        cast_a = tf.cast(greater_a, tf.float32)
        shape_a = tf.shape(cast_a)
        fill_a = tf.fill(shape_a, tf.constant(1, dtype=tf.float32))
        equal_4563 = tf.equal(b, equal1)
        equal_10831 = tf.equal(b, equal2)
        equal_3 = tf.equal(c, equal3)
        select_1 = tf.where(equal_4563, fill_a, cast_a)
        select_2 = tf.where(equal_10831, fill_a, select_1)
        output_y = select_2
        select_3 = tf.where(equal_3, realdiv, fill_a1)
        output_z = tf.concat([select_2, select_3], axis=-1)
        return output_x, output_y, output_z

    def KPFusedSparseSelect_graph(input, meta):
        # 根据核心函数入参名创建placeholder
        input_a = tf.compat.v1.placeholder(tf.int32, shape=input['input_a'].shape, name="input_a")
        input_b = tf.compat.v1.placeholder(tf.int32, shape=input['input_b'].shape, name="input_b")
        input_c = tf.compat.v1.placeholder(tf.int32, shape=input['input_c'].shape, name="input_c")
        greater = tf.constant(input['greater'], dtype=tf.int32)
        equal1 = tf.constant(input['equal1'], dtype=tf.int32)
        equal2 = tf.constant(input['equal2'], dtype=tf.int32)
        equal3 = tf.constant(input['equal3'], dtype=tf.int32)

        feed_dict = {
            input_a: input['input_a'],
            input_b: input['input_b'],
            input_c: input['input_c'],
            greater: input['greater'],
            equal1: input['equal1'],
            equal2: input['equal2'],
            equal3: input['equal3']
        }
        
        result = KPFusedSparseSelectOp(input_a, input_b, input_c, greater, equal1, equal2, equal3)
        return result, feed_dict

    def build_input_case(a_shape, b_shape, c_shape):
        input = {}
        # 根据核心函数入参名生成输入数据
        input["input_a"] = np.random.randint(0, 100, size=a_shape).astype(np.int32)
        input["input_b"] = np.random.randint(0, 100, size=b_shape).astype(np.int32)
        input["input_c"] = np.random.randint(0, 100, size=c_shape).astype(np.int32)
        input["greater"] = np.array(0, dtype=np.int32)
        input["equal1"] = np.array(4563, dtype=np.int32)
        input["equal2"] = np.array(10831, dtype=np.int32)
        input["equal3"] = np.array(3, dtype=np.int32)
        return input

    def build_input_case_1(meta):
        return build_input_case((100, 10), (10, 100), (20, 50))

    def build_input_case_2(meta):
        return build_input_case((50, 50, 50), (50, 50, 50), (50, 50, 50))

    return [
        TestCase(
            name="KPFusedSparseSelect_case_1",
            op_fn=KPFusedSparseSelect_graph,
            input_fn=build_input_case_1,
            fused_op_name="KPFusedSparseSelect",
            start_op_name="PartitionedCall_1/Reshape",
            end_op_name="PartitionedCall_1/concat",
            is_fused=True,
            num_iters=1000,
            optimize_percent=400,
        ),
        TestCase(
            name="KPFusedSparseSelect_case_2",
            op_fn=KPFusedSparseSelect_graph,
            input_fn=build_input_case_2,
            fused_op_name="KPFusedSparseSelect",
            start_op_name="PartitionedCall_3/Reshape",
            end_op_name="PartitionedCall_3/concat",
            is_fused=True,
            num_iters=1000,
            optimize_percent=600,
        ),
    ]