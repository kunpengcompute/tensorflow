import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    
    def sigmoid_op(inputs, meta):
        return tf.nn.sigmoid(inputs)

    def build_input_1D_float():
        shape = [1024]
        return [np.random.uniform(-10, 10, shape).astype(np.float32)]

    def build_input_3D_float():
        shape = [128, 64, 128]
        return [np.random.uniform(-10, 10, shape).astype(np.float32)]

    def build_input_empty():
        return [np.array([]).astype(np.float32)]


    return [
        TestCase(
            name="sigmoid_input_1D_float",
            op_fn=sigmoid_op,
            num_iters=0,
            input_fn=build_input_1D_float
        ),
        TestCase(
            name="sigmoid_input_3D_float",
            op_fn=sigmoid_op,
            num_iters=1000,
            input_fn=build_input_3D_float,
            optimize_percent = 50,
            operator_name = "Sigmoid:Sigmoid"
        ),
        TestCase(
            name="sigmoid_input_empty",
            op_fn=sigmoid_op,
            num_iters=0,
            input_fn=build_input_empty
        ),
    ]