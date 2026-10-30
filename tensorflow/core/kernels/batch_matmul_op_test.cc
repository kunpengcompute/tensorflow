/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Exercises the BatchMatMulV2 CPU kernels. On KDNN builds
// (--define=enable_kdnn=true, TF_ENABLE_KDNN_OPTS on by default) float inputs
// route through kdnnGemm (batch_size == 1) or kdnnBatchGemm (batch_size > 1)
// inside BaseBatchMatMulOp.
class BatchMatMulOpTest : public OpsTestBase {
 protected:
  void MakeOp(bool adj_x, bool adj_y) {
    TF_ASSERT_OK(NodeDefBuilder("my_op", "BatchMatMulV2")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("adj_x", adj_x)
                     .Attr("adj_y", adj_y)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// 2-D inputs: batch_size == 1, dispatched to kdnnGemm on KDNN builds.
TEST_F(BatchMatMulOpTest, Batch2D) {
  MakeOp(false, false);
  AddInputFromArray<float>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<float>(TensorShape({3, 2}), {7, 8, 9, 10, 11, 12});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected, {58, 64, 139, 154});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// 3-D inputs: batch_size > 1, dispatched to kdnnBatchGemm on KDNN builds.
TEST_F(BatchMatMulOpTest, Batch3D) {
  MakeOp(false, false);
  AddInputFromArray<float>(TensorShape({2, 2, 3}),
                           {1, 2, 3, 4, 5, 6,   // batch 0
                            0, 1, 0, 1, 0, 1});  // batch 1
  AddInputFromArray<float>(TensorShape({2, 3, 2}),
                           {7, 8, 9, 10, 11, 12,  // batch 0
                            1, 2, 3, 4, 5, 6});   // batch 1
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2}));
  test::FillValues<float>(&expected, {58, 64, 139, 154,  // batch 0
                                      3, 4, 6, 8});      // batch 1
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Adjoints are folded into the trans_x / trans_y flags passed to the KDNN
// adapters.
TEST_F(BatchMatMulOpTest, Batch3DAdjX) {
  MakeOp(true, false);
  // Logical a[b] = in0[b]^T = [[1, 2, 3], [4, 5, 6]] per batch.
  AddInputFromArray<float>(TensorShape({2, 3, 2}),
                           {1, 4, 2, 5, 3, 6,   // batch 0
                            1, 4, 2, 5, 3, 6});  // batch 1
  AddInputFromArray<float>(TensorShape({2, 3, 2}),
                           {7, 8, 9, 10, 11, 12,  // batch 0
                            7, 8, 9, 10, 11, 12});  // batch 1
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2}));
  test::FillValues<float>(&expected, {58, 64, 139, 154,  // batch 0
                                      58, 64, 139, 154});  // batch 1
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

}  // namespace
}  // namespace tensorflow
