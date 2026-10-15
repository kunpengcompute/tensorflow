import tensorflow as tf
import numpy as np
from functools import partial
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    
    def matmul_op(inputs, meta):
        return tf.matmul(inputs[0], inputs[1], transpose_a=meta.get('trans_a', False), transpose_b=meta.get('trans_b', False))

    def input_2D_float(m, k, n):
        return [
            np.random.uniform(0, 1, (m, k)).astype(np.float32), 
            np.random.uniform(0, 1, (k, n)).astype(np.float32),
        ]
    
    def input_3D_no_broadcast_float(b1, m, k, n):
        return [
            np.random.uniform(0, 1, (b1, m, k)).astype(np.float32), 
            np.random.uniform(0, 1, (b1, k, n)).astype(np.float32),
        ]
    
    def input_3D_broadcast_float(b1, m, k, n):
        return [
            np.random.uniform(0, 1, (b1, m, k)).astype(np.float32), 
            np.random.uniform(0, 1, (1, k, n)).astype(np.float32),
        ]

    def build_batch_matmul_input(shape_a, shape_b, trans_a, trans_b, dtype):
        """
        构造满足矩阵乘法条件的两个输入 Tensor
        """
        # 如果转置，需要交换最后两个维度来生成原始数据
        final_shape_a = list(shape_a)
        if trans_a:
            final_shape_a[-1], final_shape_a[-2] = final_shape_a[-2], final_shape_a[-1]
            
        final_shape_b = list(shape_b)
        if trans_b:
            final_shape_b[-1], final_shape_b[-2] = final_shape_b[-2], final_shape_b[-1]

        a = np.random.uniform(-1, 1, final_shape_a).astype(dtype)
        b = np.random.uniform(-1, 1, final_shape_b).astype(dtype)
        return [a, b]

    class BatchMatMulTestCaseFactory:
        @staticmethod
        def create_func_test():
            # 基础组合维度定义
            dims_to_test = [2, 3, 4, 5]
            trans_options = [True, False]
            broadcast_options = [True, False]
            dtypes = {"fp32": np.float32}
            
            # 矩阵乘法的核心维度 M, K, N
            M, K, N = 64, 32, 48
            
            all_cases = []
            
            for dim in dims_to_test:
                for is_bc in broadcast_options:
                    # 2D 场景不存在广播概念，跳过重复
                    if dim == 2 and is_bc: continue 
                    
                    for ta in trans_options:
                        for tb in trans_options:
                            for dt_name, dt_val in dtypes.items():
                                
                                # 构造形状逻辑
                                # A: (..., M, K), B: (..., K, N)
                                # 如果转置参数为真，op内部会处理，但输入生成需匹配
                                batch_dims_a = [2, 3, 1][:dim-2] if dim > 2 else []
                                if is_bc:
                                    # 构造广播场景：A的batch维有1，B的batch维正常
                                    batch_dims_a = [1] * (dim - 2)
                                    batch_dims_b = [2] * (dim - 2)
                                else:
                                    batch_dims_b = batch_dims_a
                                    
                                shape_a = tuple(batch_dims_a + [M, K])
                                shape_b = tuple(batch_dims_b + [K, N])
                                
                                case_name = f"batch_matmul_{dim}D_{dt_name}_bc{is_bc}_ta{ta}_tb{tb}"
                                
                                # 预绑定输入构造函数
                                input_fn = partial(
                                    build_batch_matmul_input,
                                    shape_a=shape_a,
                                    shape_b=shape_b,
                                    trans_a=ta,
                                    trans_b=tb,
                                    dtype=dt_val
                                )
                                
                                all_cases.append(TestCase(
                                    name=case_name,
                                    op_fn=matmul_op,
                                    input_fn=input_fn,
                                    num_iters=0,
                                    meta={
                                        'trans_a': ta, # 对应算子属性：是否转置第一个输入
                                        'trans_b': tb  # 对应算子属性：是否转置第二个输入
                                    }
                                ))
            return all_cases
    test_case = []
    test_case.extend(BatchMatMulTestCaseFactory.create_func_test())
    # perf test case
    test_case.extend([
        TestCase(
            name="MatMul_2D_79_1570_256_False_False",
            op_fn=matmul_op,
            input_fn=partial(input_2D_float, m=79, k=1570, n=256),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        TestCase(
            name="MatMul_2D_79_1570_128_False_False",
            op_fn=matmul_op,
            input_fn=partial(input_2D_float, m=79, k=1570, n=128),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        TestCase(
            name="MatMul_2D_4480_32_16_False_False",
            op_fn=matmul_op,
            input_fn=partial(input_2D_float, m=4480, k=32, n=16),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        TestCase(
            name="MatMul_2D_64_256_128_False_False",
            op_fn=matmul_op,
            input_fn=partial(input_2D_float, m=64, k=256, n=128),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        TestCase(
            name="MatMul_2D_128_592_128_False_False",
            op_fn=matmul_op,
            input_fn=partial(input_2D_float, m=128, k=592, n=128),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        TestCase(
            name="MatMul_3D_No_Broadcast_64_64_256_256_False",
            op_fn=matmul_op,
            input_fn=partial(input_3D_no_broadcast_float, b1=64, m=64, k=256, n=256),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        TestCase(
            name="MatMul_3D_No_Broadcast_64_32_64_64_False",
            op_fn=matmul_op,
            input_fn=partial(input_3D_no_broadcast_float, b1=64, m=32, k=64, n=64),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "MatMul",
            meta={'trans_a': False, 'trans_b': False},
        ),
        
    ])
    return test_case