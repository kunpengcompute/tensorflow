import tensorflow as tf
import numpy as np
from functools import partial
from framework.runner import TestCase
from tensorflow.python.framework import sparse_tensor


def get_test_cases():
    """sparse_tensor_dense_matmul 算子测试用例

    测试 shape (M x K x N) & 稀疏度 sparsity 的组合:
        - 128 x 1668 x 400, 0.7
        - 256 x 1668 x 400, 0.7
        - 128 x  492 x 400, 0.0
        - 256 x  492 x 400, 0.0 
    - sp_a: SparseTensor, shape (M, K)
    - b:    DenseTensor,  shape (K, N)
    - out:  DenseTensor,  shape (M, N)
    """

    SHAPES_SPARSITY = [
        (128, 1668, 400, 0.7),
        (256, 1668, 400, 0.7),
        (128,  492, 400, 0.0),
        (256,  492, 400, 0.0),
    ]

    def build_sparse_input(M, K, N, sparsity, dtype=np.float32):
        """构造稀疏矩阵输入

        Returns:
            [indices, values, dense_shape, b_dense] — 四个独立的 numpy 数组，
            以适配 benchmark framework 的 graph-mode placeholder 机制。
        """
        np.random.seed(42)

        a_dense = np.random.randn(M, K).astype(dtype)

        if sparsity > 0.0:
            threshold = np.quantile(np.abs(a_dense), sparsity)
            a_dense[np.abs(a_dense) <= threshold] = 0.0

        nonzero_mask = a_dense != 0
        indices = np.vstack(np.where(nonzero_mask)).astype(np.int64).T
        values = a_dense[nonzero_mask]
        dense_shape = np.array(a_dense.shape).astype(np.int64)

        b_dense = np.random.randn(K, N).astype(dtype)

        # 返回 4 个独立数组，framework 会逐个包装为 placeholder
        return [indices, values, dense_shape, b_dense]

    def sparse_op_fn(inputs, meta):
        """算子函数：从 4 个 placeholder 构造 SparseTensor 并执行 sparse_dense_matmul

        inputs[0]: indices   placeholder, shape (nnz, 2), dtype int64
        inputs[1]: values    placeholder, shape (nnz,),  dtype float32
        inputs[2]: dense_shape placeholder, shape (2,),   dtype int64
        inputs[3]: b         placeholder, shape (K, N),   dtype float32
        """
        indices, values, dense_shape, b = inputs[0], inputs[1], inputs[2], inputs[3]
        sp_a = sparse_tensor.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=dense_shape
        )
        adjoint_a = meta.get('adjoint_a', False)
        adjoint_b = meta.get('adjoint_b', False)
        return tf.sparse.sparse_dense_matmul(sp_a, b,
                                             adjoint_a=adjoint_a,
                                             adjoint_b=adjoint_b)

    def make_name(M, K, N, sparsity, suffix=''):
        """生成统一的测试用例名称"""
        sp_str = f'{sparsity}'.replace('.', '_')
        name = f'sparse_tensor_dense_matmul_{M}_{K}_{N}_sparsity_{sp_str}'
        return name + suffix if suffix else name

    test_cases = []

    for M, K, N, sparsity in SHAPES_SPARSITY:
        sp_str = f'{sparsity}'.replace('.', '_')

        # 绑定 shape 参数
        input_fn = partial(build_sparse_input, M=M, K=K, N=N, sparsity=sparsity)

        test_cases.append(TestCase(
            name=make_name(M, K, N, sp_str, suffix='_perf'),
            op_fn=sparse_op_fn,
            input_fn=input_fn,
            num_iters=1000,
            optimize_percent=5,
            operator_name='SparseTensorDenseMatMul',
            meta={}
        ))

    return test_cases
