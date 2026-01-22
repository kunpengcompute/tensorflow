import tensorflow as tf
import numpy as np
from framework.runner import TestCase
from framework.runner import CheckFuncClass

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    
    def reduce_floormod_op(inputs, meta):
        return tf.math.floormod(inputs[0], inputs[1])

    def build_input_1D_float():
        shape = [500000]
        return [np.random.uniform(-1, 1, shape).astype(np.float32) for i in range(2)]

    def build_input_3D_float():
        shape = [64, 64, 128]
        return [np.random.uniform(0, 1, shape).astype(np.float32) for i in range(2)]

    def build_input_1D_int64():
        shape = [1000000]
        return [np.random.randint(-2**63, 2**63-1, size = shape, dtype=np.int64) for i in range(2)]

    def build_input_3D_int64():
        shape = [128, 64, 128]
        return [np.random.randint(0, 2**63-1, size = shape, dtype=np.int64) for i in range(2)]

    return [
        TestCase(
            name="floormod_input_1D_float",
            op_fn=reduce_floormod_op,
            input_fn=build_input_3D_float,
            num_iters=1000,
            optimize_percent = 150,
            operator_name = "FloorMod:FloorMod",
            meta={}
        ),
        TestCase(
            name="floormod_input_3D_float",
            op_fn=reduce_floormod_op,
            input_fn=build_input_3D_float,
            num_iters=1000,
            optimize_percent = 150,
            operator_name = "FloorMod:FloorMod",
            meta={}
        ),
        TestCase(
            name="floormod_input_1D_int64",
            op_fn=reduce_floormod_op,
            input_fn=build_input_1D_int64,
            check_fn=CheckFuncClass.check_fn_equal,
            num_iters=1000,
            optimize_percent = 25,
            operator_name = "FloorMod:FloorMod",
            meta={}
        ),
        TestCase(
            name="floormod_input_3D_int64",
            op_fn=reduce_floormod_op,
            input_fn=build_input_3D_int64,
            check_fn=CheckFuncClass.check_fn_equal,
            num_iters=1000,
            optimize_percent = 25,
            operator_name = "FloorMod:FloorMod",
            meta={}
        ),
    ]