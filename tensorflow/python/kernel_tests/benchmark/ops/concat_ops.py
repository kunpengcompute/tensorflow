import tensorflow as tf
import numpy as np
from functools import partial
from framework.runner import TestCase, CheckFuncClass

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    def concat_op(inputs, meta):
        return tf.concat(inputs, meta["axis"])

    def build_18input_2D(dtype=np.float32):
        batch = 150
        shapes = [
            (batch, 960),
            (batch, 24),
            (batch, 4),
            (batch, 4),
            (batch, 4),
            (batch, 64),
            (batch, 4),
            (batch, 4),
            (batch, 32),
            (batch, 4),
            (batch, 4),
            (batch, 24),
            (batch, 4),
            (batch, 4),
            (batch, 24),
            (batch, 4),
            (batch, 4),
            (batch, 16),
        ]
        return [np.random.uniform(0, 1, shape).astype(dtype) for shape in shapes]

    def build_12input_2D(dtype=np.float32):
        batch = 150
        shapes = [
            (batch, 192),
            (batch, 288),
            (batch, 64),
            (batch, 48),
            (batch, 48),
            (batch, 8),
            (batch, 48),
            (batch, 48),
            (batch, 8),
            (batch, 256),
            (batch, 128),
            (batch, 128),
        ]
        return [np.random.uniform(0, 1, shape).astype(dtype) for shape in shapes]

    def build_8input_3D(dtype=np.float32):
        batch = 16
        seq = 64
        shapes = [
            (batch, seq, 32),
            (batch, seq, 64),
            (batch, seq, 16),
            (batch, seq, 16),
            (batch, seq, 16),
            (batch, seq, 16),
            (batch, seq, 16),
            (batch, seq, 16),
        ]
        return [np.random.uniform(0, 1, shape).astype(dtype) for shape in shapes]

    def build_4input_2D(dtype=np.float32):
        batch = 256
        shapes = [
            (batch, 192),
            (batch, 288),
            (batch, 64),
            (batch, 48),
        ]
        return [np.random.uniform(0, 1, shape).astype(dtype) for shape in shapes]
    
    def build_3input_2D(dtype=np.float32):
        shapes = [
            (128, 1510),
            (128, 36),
            (128, 24),
        ]
        return [np.random.uniform(0, 1, shape).astype(dtype) for shape in shapes]
    
    def build_2input_2D(batch, input1_dim, input2_dim, dtype=np.float32):
        shapes = [
            (batch, input1_dim),
            (batch, input2_dim),
        ]
        return [np.random.uniform(0, 1, shape).astype(dtype) for shape in shapes]

    def build_dynamic_input(num_inputs, dtype, shape_template, concat_axis):
        """
        通用输入构造器
        :param num_inputs: 输入 Tensor 的个数 (1, 2, 8, 12, 18)
        :param dtype: 数据类型 (np.float32, np.int32 等)
        :param shape_template: 基础 3D 形状，例如 (batch, seq, feature)
        :param concat_axis: 拼接的轴，用于微调形状确保可以 concat
        """
        inputs = []
        for i in range(num_inputs):
            curr_shape = list(shape_template)
            # 为了模拟真实情况，我们可以让拼接轴上的长度略有不同
            curr_shape[concat_axis] = np.random.randint(1, 10) # 如果需要动态长度可开启
            
            if np.issubdtype(dtype, np.integer):
                data = np.random.randint(0, 100, curr_shape).astype(dtype)
            else:
                data = np.random.uniform(0, 1, curr_shape).astype(dtype)
            inputs.append(data)
        return inputs

    class ConcatTestCaseFactory:
        @staticmethod
        def create_func_test():
            all_cases = []
            input_counts = [1, 2, 8, 12, 18]
            dtypes = {
                "fp32": np.float32, 
                "bf16": np.uint16, # 工业界常用 uint16 模拟 bf16 存储
                "int32": np.int32, 
                "int8": np.int8, 
                "uint8": np.uint8
            }
            axes = [0, 1, 2, -1, -2, -3]
            base_3d_shape = (32, 64, 128) # 示例 3D 基础形状
            for num in input_counts:
                for dtype_name, dtype_val in dtypes.items():
                    for axis in axes:
                        case_name = f"concat_3D_{num}in_{dtype_name}_axis{axis}"
                        
                        # 使用 partial 预绑定所有参数
                        input_fn = partial(
                            build_dynamic_input, 
                            num_inputs=num, 
                            dtype=dtype_val, 
                            shape_template=base_3d_shape, 
                            concat_axis=axis
                        )
                        
                        case = TestCase(
                            name=case_name,
                            op_fn=concat_op,
                            input_fn=input_fn,
                            num_iters=0,
                            check_fn=CheckFuncClass.check_fn_equal,
                            meta={'axis': axis}
                        )
                        all_cases.append(case)
            return all_cases
    
    test_case = []
    test_case.extend(ConcatTestCaseFactory.create_func_test())
    # perf test case
    test_case.extend([
        TestCase(
            name="concat_18input_2D_float",
            op_fn=concat_op,
            input_fn=partial(build_18input_2D, dtype=np.float32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_12input_2D_float",
            op_fn=concat_op,
            input_fn=partial(build_12input_2D, dtype=np.float32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_8input_3D_float",
            op_fn=concat_op,
            input_fn=partial(build_8input_3D, dtype=np.float32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_4input_2D_float",
            op_fn=concat_op,
            input_fn=partial(build_4input_2D, dtype=np.float32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_3input_2D_float",
            op_fn=concat_op,
            input_fn=partial(build_3input_2D, dtype=np.float32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_2input_2D_small_float",
            op_fn=concat_op,
            input_fn=partial(build_2input_2D, batch=128, input1_dim=400, input2_dim=400, dtype=np.float32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_18input_2D_int32",
            op_fn=concat_op,
            input_fn=partial(build_18input_2D, dtype=np.int32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_12input_2D_int32",
            op_fn=concat_op,
            input_fn=partial(build_12input_2D, dtype=np.int32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_8input_3D_int32",
            op_fn=concat_op,
            input_fn=partial(build_8input_3D, dtype=np.int32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_4input_2D_int32",
            op_fn=concat_op,
            input_fn=partial(build_4input_2D, dtype=np.int32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_3input_2D_int32",
            op_fn=concat_op,
            input_fn=partial(build_3input_2D, dtype=np.int32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
        TestCase(
            name="concat_2input_2D_small_int32",
            op_fn=concat_op,
            input_fn=partial(build_2input_2D, batch=128, input1_dim=400, input2_dim=400, dtype=np.int32),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "ConcatV2",
            check_fn=CheckFuncClass.check_fn_equal,
            meta={'axis': -1}
        ),
    ])
    return test_case