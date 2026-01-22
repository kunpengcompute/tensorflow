import tensorflow as tf
import numpy as np
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""

    def softmax_op(inputs, meta):
        return tf.nn.softmax(inputs, axis=meta["axis"])

    def logsoftmax_op(inputs, meta):
        return tf.nn.log_softmax(inputs, axis=meta["axis"])

    def build_input_2D_float():
        shape = [1024, 128]
        return [np.random.uniform(-100, 100, shape).astype(np.float32)]

    def build_input_3D_float():
        shape = [64, 64, 128]
        return [np.random.uniform(-10, 10, shape).astype(np.float32)]

    return [
        TestCase(
            name="softmax_input_2D_float",
            op_fn=softmax_op,
            input_fn=build_input_2D_float,
            num_iters=0,
            meta={'axis': 2,}
        ),
        TestCase(
            name="softmax_input_3D_float_axis_1",
            op_fn=softmax_op,
            input_fn=build_input_3D_float,
            num_iters=0,
            meta={'axis': 1,}
        ),
        TestCase(
            name="softmax_input_3D_float_axis_2",
            op_fn=softmax_op,
            input_fn=build_input_3D_float,
            num_iters=1000,
            optimize_percent = 200,
            operator_name = "Softmax:Softmax",
            meta={'axis': 2,}
        ),
        TestCase(
            name="logsoftmax_input_2D_float",
            op_fn=logsoftmax_op,
            input_fn=build_input_2D_float,
            num_iters=0,
            meta={'axis': -1,}
        ),
    ]