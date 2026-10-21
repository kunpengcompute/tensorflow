import tensorflow as tf
import numpy as np
from functools import partial
from framework.runner import TestCase

def get_test_cases():
    """每个算子文件都统一实现这个接口"""
    
    def einsum_op(inputs, meta):
        return tf.einsum(meta["label"], inputs[0], inputs[1])

    def input_3D_bh_bsh_bs_float(B, H, S):
        return [
            np.random.uniform(0, 1, (B, H)).astype(np.float32),
            np.random.uniform(0, 1, (B, S, H)).astype(np.float32),
        ]
    
    def input_3D_bs_bsh_bh_float(B, H, S):
        return [
            np.random.uniform(0, 1, (B, S)).astype(np.float32),
            np.random.uniform(0, 1, (B, S, H)).astype(np.float32),
        ]
    
    def build_einsum_input(shapes, dtype):
        """
        根据 einsum 表达式和形状列表构造输入
        """
        inputs = []
        for shape in shapes:
            if np.issubdtype(dtype, np.integer):
                data = np.random.randint(0, 10, shape).astype(dtype)
            else:
                data = np.random.uniform(-1, 1, shape).astype(dtype)
            inputs.append(data)
        return inputs

    class EinsumTestCaseFactory:
        @staticmethod
        def create_func_test():
            all_cases = []
            # 定义 B, H, S 的测试点
            B_list = [1, 79, 256]
            H_list = [1, 79, 128]
            S_list = [1, 79, 512]
            
            # 1. Scoring 场景: bh, bsh -> bs
            for b in B_list:
                for h in H_list:
                    for s in S_list:
                        eq = "bh,bsh->bs"
                        shapes = [(b, h), (b, s, h)]
                        all_cases.append(EinsumTestCaseFactory._create_case(
                            "Scoring", eq, shapes, np.float32, b, h, s
                        ))

            # 2. Pooling 场景: bs, bsh -> bh
            for b in B_list:
                for h in H_list:
                    for s in S_list:
                        eq = "bs,bsh->bh"
                        shapes = [(b, s), (b, s, h)]
                        all_cases.append(EinsumTestCaseFactory._create_case(
                            "Pooling", eq, shapes, np.float32, b, h, s
                        ))

            # 3. 批量矩阵乘法场景: bij, bjk -> bik
            ikj_variants = [
                (64, 128, 64), (79, 128, 64), (1, 128, 64), 
                (128, 1, 64), (128, 128, 1)
            ]
            for b in B_list:
                for i, k, j in ikj_variants:
                    eq = "bij,bjk->bik"
                    shapes = [(b, i, j), (b, j, k)]
                    all_cases.append(EinsumTestCaseFactory._create_case(
                        "BatchMatMul", eq, shapes, np.float32, b, i, j, k
                    ))
            
            return all_cases

        @staticmethod
        def _create_case(scene, eq, shapes, dtype, *args):
            dims_str = "_".join([str(a) for a in args])
            case_name = f"einsum_{scene}_{eq.replace(',', '_').replace('->', '_')}_params_{dims_str}"
            
            return TestCase(
                name=case_name,
                op_fn=einsum_op,
                input_fn=partial(build_einsum_input, shapes=shapes, dtype=dtype),
                num_iters=0,
                operator_name="Einsum",
                meta={'label': eq}
            )

    test_case = []
    test_case.extend(EinsumTestCaseFactory.create_func_test())
    # perf test case
    test_case.extend([
        TestCase(
            name="Einsum_3D_float_bh_bsh_bs_128_256_512",
            op_fn=einsum_op,
            input_fn=partial(input_3D_bh_bsh_bs_float, B=128, H=256, S=512),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "Einsum",
            meta={'label': "bh,bsh->bs"}
        ),
        TestCase(
            name="Einsum_3D_float_bs_bsh_bh_128_128_512",
            op_fn=einsum_op,
            input_fn=partial(input_3D_bs_bsh_bh_float, B=128, H=128, S=512),
            num_iters=1000,
            optimize_percent = 5,
            operator_name = "Einsum",
            meta={'label': "bs,bsh->bh"}
        ),
    ])
    return test_case