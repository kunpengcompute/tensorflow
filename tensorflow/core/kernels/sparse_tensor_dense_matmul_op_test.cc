/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Fully naive reference for all adjoint combinations
static void ReferenceSparseMatmulFn(
    const int nnz,
    const int64_t* a_indices_data,
    const float* a_values_data,
    const float* b_data,
    int lhs_left, int lhs_right, int rhs_right,
    bool adjoint_a, bool adjoint_b,
    float* out_data) {
  // Initialize output to zero
  for (int i = 0; i < lhs_left * rhs_right; ++i) out_data[i] = 0.0f;

  const int lhs_index_a = adjoint_a ? 1 : 0;
  const int rhs_index_a = adjoint_a ? 0 : 1;

  for (int i = 0; i < nnz; ++i) {
    int row = a_indices_data[i * 2 + lhs_index_a];
    int col = a_indices_data[i * 2 + rhs_index_a];
    float a_val = a_values_data[i];
    if (adjoint_b) {
      for (int n = 0; n < rhs_right; ++n) {
        out_data[row * rhs_right + n] += a_val * b_data[col * rhs_right + n];
      }
    } else {
      for (int n = 0; n < rhs_right; ++n) {
        out_data[row * rhs_right + n] += a_val * b_data[col * rhs_right + n];
      }
    }
  }
}

// ===========================================================================
// SparseMatMul accuracy tests (Section 10.1.1 of design doc)
// ===========================================================================

class SparseMatmulAccuracyTest : public OpsTestBase {
 protected:
  void MakeOp(bool adjoint_a, bool adjoint_b) {
    TF_CHECK_OK(NodeDefBuilder("sparse_matmul", "SparseTensorDenseMatMul")
                     .Input(FakeInput(DT_INT64))    // a_indices
                     .Input(FakeInput(DT_FLOAT))    // a_values
                     .Input(FakeInput(DT_INT64))    // a_shape
                     .Input(FakeInput(DT_FLOAT))    // b
                     .Attr("T", DT_FLOAT)
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
  }

  // Set up and run the SparseTensorDenseMatMul op.
  Tensor RunOp(const Tensor& a_indices, const Tensor& a_values,
               const Tensor& a_shape, const Tensor& b,
               bool adjoint_a, bool adjoint_b) {
    MakeOp(adjoint_a, adjoint_b);

    *AddInput(DT_INT64, a_indices.shape()) = a_indices;
    *AddInput(DT_FLOAT, a_values.shape()) = a_values;
    *AddInput(DT_INT64, a_shape.shape()) = a_shape;
    *AddInput(DT_FLOAT, b.shape()) = b;

    TF_CHECK_OK(RunOpKernel());

    return *GetOutput(0);
  }

  // Compare two matrices with a tolerance
  void ExpectClose(const Tensor& actual, const Tensor& expected,
                   float rtol = 1e-5, float atol = 1e-5) {
    ASSERT_EQ(actual.shape(), expected.shape());
    auto actual_m = actual.matrix<float>();
    auto expected_m = expected.matrix<float>();
    int rows = actual.dim_size(0);
    int cols = actual.dim_size(1);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        float diff = std::abs(actual_m(i, j) - expected_m(i, j));
        float tol = rtol * std::max(std::abs(expected_m(i, j)), 1.0f) + atol;
        EXPECT_LE(diff, tol)
            << "(" << i << "," << j << "): actual=" << actual_m(i, j)
            << " expected=" << expected_m(i, j);
      }
    }
  }

  // Helper: create a deterministic sparse matrix with given sparsity
  void CreateSparseMatrix(int rows, int cols, int nnz, int seed,
                          Tensor* a_indices, Tensor* a_values,
                          Tensor* a_shape) {
    *a_indices = Tensor(DT_INT64, TensorShape({nnz, 2}));
    *a_values = Tensor(DT_FLOAT, TensorShape({nnz}));
    *a_shape = Tensor(DT_INT64, TensorShape({2}));
    a_shape->vec<int64_t>()(0) = rows;
    a_shape->vec<int64_t>()(1) = cols;

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> row_dist(0, rows - 1);
    std::uniform_int_distribution<> col_dist(0, cols - 1);
    std::uniform_real_distribution<float> val_dist(-1.0, 1.0);

    auto idx_m = a_indices->matrix<int64_t>();
    auto val_v = a_values->vec<float>();
    for (int i = 0; i < nnz; ++i) {
      idx_m(i, 0) = row_dist(gen);
      idx_m(i, 1) = col_dist(gen);
      val_v(i) = val_dist(gen);
    }
  }
};

// ---------------------------------------------------------------------------
// 1. Basic functionality: different sparsity levels
// ---------------------------------------------------------------------------
TEST_F(SparseMatmulAccuracyTest, BasicSparsity) {
  const int m = 8, k = 16, n = 4;
  const int nnz = 20;

  Tensor a_indices, a_values, a_shape;
  CreateSparseMatrix(m, k, nnz, /*seed=*/42, &a_indices, &a_values, &a_shape);

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  b.flat<float>().setRandom();

  // Compute reference
  Tensor ref(DT_FLOAT, TensorShape({m, n}));
  ReferenceSparseMatmulFn(
      nnz, a_indices.matrix<int64_t>().data(), a_values.vec<float>().data(),
      b.matrix<float>().data(), m, k, n,
      /*adjoint_a=*/false, /*adjoint_b=*/false,
      ref.matrix<float>().data());

  // Compute via TF op (KDNN path when ENABLE_KDNN is on)
  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/false, /*adjoint_b=*/false);

  ExpectClose(result, ref);
}

// ---------------------------------------------------------------------------
// 2. Boundary conditions
// ---------------------------------------------------------------------------
TEST_F(SparseMatmulAccuracyTest, ZeroNonZeroEntries) {
  const int m = 4, k = 8, n = 2;
  const int nnz = 0;

  Tensor a_indices(DT_INT64, TensorShape({0, 2}));
  Tensor a_values(DT_FLOAT, TensorShape({0}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  a_shape.vec<int64_t>()(0) = m;
  a_shape.vec<int64_t>()(1) = k;

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  b.flat<float>().setRandom();

  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/false, /*adjoint_b=*/false);

  // All zeros
  auto result_m = result.matrix<float>();
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_EQ(result_m(i, j), 0.0f);
}

TEST_F(SparseMatmulAccuracyTest, IdentityMatrix) {
  const int m = 4, k = 4, n = 4;
  // A = I (identity sparse), A * B = B
  Tensor a_indices(DT_INT64, TensorShape({4, 2}));
  Tensor a_values(DT_FLOAT, TensorShape({4}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  a_shape.vec<int64_t>()(0) = m;
  a_shape.vec<int64_t>()(1) = k;

  auto idx = a_indices.matrix<int64_t>();
  auto val = a_values.vec<float>();
  for (int i = 0; i < 4; ++i) {
    idx(i, 0) = i;
    idx(i, 1) = i;
    val(i) = 1.0f;
  }

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  b.flat<float>().setRandom();

  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/false, /*adjoint_b=*/false);

  ExpectClose(result, b);
}

TEST_F(SparseMatmulAccuracyTest, AllZeroMatrix) {
  const int m = 4, k = 8, n = 4;
  // A = zeros, so result = zeros
  Tensor a_indices(DT_INT64, TensorShape({0, 2}));
  Tensor a_values(DT_FLOAT, TensorShape({0}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  a_shape.vec<int64_t>()(0) = m;
  a_shape.vec<int64_t>()(1) = k;

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  b.flat<float>().setRandom();

  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/false, /*adjoint_b=*/false);

  auto result_m = result.matrix<float>();
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_EQ(result_m(i, j), 0.0f);
}

// ---------------------------------------------------------------------------
// 3. Transpose semantics
// ---------------------------------------------------------------------------
TEST_F(SparseMatmulAccuracyTest, AdjointA) {
  const int m = 8, k = 16, n = 4;
  const int nnz = 30;

  Tensor a_indices, a_values, a_shape;
  CreateSparseMatrix(k, m, nnz, /*seed=*/123, &a_indices, &a_values, &a_shape);
  // adjoint_a=true: A is k×m but treated as A^T which is m×k
  // So the sparse matrix shape is (k, m) with indices in (k, m) space.
  // The op with adjoint_a treats it as: A^T (m×k) * B (k×n) → (m×n)

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  b.flat<float>().setRandom();

  // Reference: manually transpose
  Tensor ref(DT_FLOAT, TensorShape({m, n}));
  auto a_shape_v = a_shape.vec<int64_t>();
  int rows_a = a_shape_v(0);  // k
  int cols_a = a_shape_v(1);  // m
  auto a_idx = a_indices.matrix<int64_t>();
  auto a_val = a_values.vec<float>();
  auto b_m = b.matrix<float>();
  auto ref_m = ref.matrix<float>();
  ref_m.setZero();
  for (int i = 0; i < nnz; ++i) {
    // With adjoint_a=true: A_orig is k×m, index(i,0)=row_in_orig(=k_dim), index(i,1)=col_in_orig(=m_dim)
    // A^T has dimension m×k: A^T(col_in_orig, row_in_orig) = A_orig(row_in_orig, col_in_orig)
    int orig_row = a_idx(i, 0);  // k dimension
    int orig_col = a_idx(i, 1);  // m dimension
    // In A^T, result row = orig_col, A^T's k-index = orig_row
    for (int j = 0; j < n; ++j) {
      ref_m(orig_col, j) += a_val(i) * b_m(orig_row, j);
    }
  }

  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/true, /*adjoint_b=*/false);
  ExpectClose(result, ref);
}

// ---------------------------------------------------------------------------
// 4. Large scale data (stability test)
// ---------------------------------------------------------------------------
TEST_F(SparseMatmulAccuracyTest, LargeScale) {
  const int m = 64, k = 128, n = 16;
  const int nnz = 200;

  Tensor a_indices(DT_INT64, TensorShape({nnz, 2}));
  Tensor a_values(DT_FLOAT, TensorShape({nnz}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  a_shape.vec<int64_t>()(0) = m;
  a_shape.vec<int64_t>()(1) = k;

  auto idx = a_indices.matrix<int64_t>();
  auto val = a_values.vec<float>();
  for (int i = 0; i < nnz; ++i) {
    idx(i, 0) = (i * 7) % m;
    idx(i, 1) = (i * 13) % k;
    val(i) = static_cast<float>((i % 100) - 50) / 50.0f;
  }

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  auto b_m = b.matrix<float>();
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < n; ++j)
      b_m(i, j) = static_cast<float>((i * 3 + j * 7) % 20 - 10) / 10.0f;

  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/false, /*adjoint_b=*/false);
  EXPECT_EQ(result.shape(), TensorShape({m, n}));
}

// ---------------------------------------------------------------------------
// 5. Multi-thread path consistency (env var set before binary start)
// ---------------------------------------------------------------------------
// Note: TF_ENABLE_KDNN_SPARSE_MATMUL_PARALLEL controls parallel vs serial
// in the KDNN SparseGemm path (see kdnn_adapter.h). It is read once at
// first invocation. To test with parallel=0, run:
//   TF_ENABLE_KDNN_SPARSE_MATMUL_PARALLEL=0 ... --benchmark_filter=...
TEST_F(SparseMatmulAccuracyTest, KdnnPathMatchesReference) {
  const int m = 32, k = 64, n = 8;
  const int nnz = 80;

  Tensor a_indices, a_values, a_shape;
  CreateSparseMatrix(m, k, nnz, /*seed=*/99, &a_indices, &a_values, &a_shape);

  Tensor b(DT_FLOAT, TensorShape({k, n}));
  b.flat<float>().setRandom();

  Tensor ref(DT_FLOAT, TensorShape({m, n}));
  ReferenceSparseMatmulFn(
      nnz, a_indices.matrix<int64_t>().data(), a_values.vec<float>().data(),
      b.matrix<float>().data(), m, k, n,
      false, false, ref.matrix<float>().data());

  Tensor result = RunOp(a_indices, a_values, a_shape, b,
                        /*adjoint_a=*/false, /*adjoint_b=*/false);

  ExpectClose(result, ref);
}


Node* SparseTensorDenseMatMulNode(Graph* g, Node* a_indices, Node* a_values,
                                  Node* a_shape, Node* b, bool adjoint_a,
                                  bool adjoint_b) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseTensorDenseMatMul")
                  .Input(a_indices)
                  .Input(a_values)
                  .Input(a_shape)
                  .Input(b)
                  .Attr("T", DT_FLOAT)
                  .Attr("adjoint_a", adjoint_a)
                  .Attr("adjoint_b", adjoint_b)
                  .Finalize(g, &ret));
  return ret;
}

static Graph* SparseTensorDenseMatmul(int nnz, int m, int k, int n,
                                      bool adjoint_a, bool adjoint_b) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor a_values(DT_FLOAT, TensorShape({nnz}));
  Tensor a_indices(DT_INT64, TensorShape({nnz, 2}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  auto a_shape_t = a_shape.vec<int64_t>();
  a_shape_t(0) = adjoint_a ? k : m;
  a_shape_t(1) = adjoint_a ? m : k;
  a_values.flat<float>().setRandom();
  auto a_indices_t = a_indices.matrix<int64_t>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> a_lhs_dist(0, a_shape_t(0) - 1);
  std::uniform_int_distribution<> a_rhs_dist(0, a_shape_t(1) - 1);
  for (int32_t i = 0; i < nnz; ++i) {
    a_indices_t(i, 0) = a_lhs_dist(gen);
    a_indices_t(i, 1) = a_rhs_dist(gen);
  }
  Tensor b(DT_FLOAT, adjoint_b ? TensorShape({n, k}) : TensorShape({k, n}));
  b.flat<float>().setRandom();

  SparseTensorDenseMatMulNode(
      g, test::graph::Constant(g, a_indices),
      test::graph::Constant(g, a_values), test::graph::HostConstant(g, a_shape),
      test::graph::Constant(g, b), adjoint_a, adjoint_b);
  return g;
}

// NOLINTBEGIN
#define BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, DEVICE)                  \
  static void                                                                        \
      BM_SparseTensorDenseMatmul##_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE( \
          ::testing::benchmark::State& state) {                                      \
    int64_t items_per_iter = (static_cast<int64_t>(NNZ) * (TB ? K : N));             \
    test::Benchmark(#DEVICE, SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB),          \
                    /*old_benchmark_api*/ false)                                     \
        .Run(state);                                                                 \
    state.SetItemsProcessed(state.iterations() * items_per_iter);                    \
    state.SetBytesProcessed(state.iterations() * items_per_iter *                    \
                            sizeof(float));                                          \
  }                                                                                  \
  BENCHMARK(                                                                         \
      BM_SparseTensorDenseMatmul##_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE);
// NOLINTEND

#define BM_SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB)    \
  BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, cpu); \
  BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, gpu);

BM_SparseTensorDenseMatmul(128, 8, 512, 1, false, false);
BM_SparseTensorDenseMatmul(128, 16, 512, 1, false, false);
BM_SparseTensorDenseMatmul(128, 128, 512, 1, false, false);

BM_SparseTensorDenseMatmul(128, 4096, 4096, 1, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 1, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 1, false, false);

BM_SparseTensorDenseMatmul(128, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(128, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(1024, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(16384, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, false, true);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, true, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, true, true);

// NOLINTBEGIN
#define BM_SparseTensorDenseMatmulDevT(NNZ, M, K, N, TA, TB, DEVICE, T)                \
  static void                                                                           \
      BM_SparseTensorDenseMatmul_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE##_t##T( \
          ::testing::benchmark::State& state) {                                         \
    int64_t items_per_iter = (static_cast<int64_t>(NNZ) * (TB ? K : N));                \
    test::Benchmark(#DEVICE, SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB),             \
                    /*old_benchmark_api*/ false)                                        \
        .Run(state);                                                                    \
    state.SetItemsProcessed(state.iterations() * items_per_iter);                       \
    state.SetBytesProcessed(state.iterations() * items_per_iter *                       \
                            sizeof(float));                                             \
  }                                                                                     \
  BENCHMARK(                                                                            \
      BM_SparseTensorDenseMatmul_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE##_t##T) \
      ->Threads(T);
// NOLINTEND

#define BM_SparseTensorDenseMatmulT(NNZ, M, K, N, TA, TB, T)    \
  BM_SparseTensorDenseMatmulDevT(NNZ, M, K, N, TA, TB, cpu, T);

#define BM_SparseTensorDenseMatmulAllThreads(NNZ, M, K, N, TA, TB) \
  BM_SparseTensorDenseMatmulT(NNZ, M, K, N, TA, TB, 1);           \
  BM_SparseTensorDenseMatmulT(NNZ, M, K, N, TA, TB, 4);           \
  BM_SparseTensorDenseMatmulT(NNZ, M, K, N, TA, TB, 8);           \
  BM_SparseTensorDenseMatmulT(NNZ, M, K, N, TA, TB, 16);

BM_SparseTensorDenseMatmulAllThreads(128,  16,   1024,   1, false, false);
BM_SparseTensorDenseMatmulAllThreads(128,  16,   1024,   8, false, false);
BM_SparseTensorDenseMatmulAllThreads(256,  128,  4096,   1, false, false);
BM_SparseTensorDenseMatmulAllThreads(128,  128,  1024,   8, false, false);
BM_SparseTensorDenseMatmulAllThreads(128,  16,   1024,  64, false, false);
BM_SparseTensorDenseMatmulAllThreads(256,  128,  4096,  16, false, false);
BM_SparseTensorDenseMatmulAllThreads(128,  128,  1024,  64, false, false);
BM_SparseTensorDenseMatmulAllThreads(256,  128,  4096, 128, false, false);
BM_SparseTensorDenseMatmulAllThreads(128,  128,  1024,   1, false, false);
BM_SparseTensorDenseMatmulAllThreads(1024, 256,  4096,   1, false, false);
BM_SparseTensorDenseMatmulAllThreads(128,  4096, 4096, 512, false, false);

}  // end namespace tensorflow
