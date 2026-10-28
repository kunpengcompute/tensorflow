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

// Exercises the Softmax / LogSoftmax CPU kernels. On KDNN builds
// (--define=enable_kdnn=true, TF_ENABLE_KDNN_OPTS on by default) the float
// Softmax case dispatches to kdnnSoftmaxOp inside SoftmaxOp::Compute, while
// LogSoftmax keeps the Eigen path (log_ is true).
class SoftmaxOpTest : public OpsTestBase {
 protected:
  void MakeOp(const char* op_name) {
    TF_EXPECT_OK(NodeDefBuilder("my_op", op_name)
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(SoftmaxOpTest, SoftmaxFloat) {
  MakeOp("Softmax");
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected,
                          {0.09003057f, 0.24472847f, 0.66524096f,  //
                           1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

TEST_F(SoftmaxOpTest, LogSoftmaxFloat) {
  MakeOp("LogSoftmax");
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected,
                          {-2.40760596f, -1.40760596f, -0.40760596f,  //
                           -1.09861229f, -1.09861229f, -1.09861229f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

}  // namespace
}  // namespace tensorflow
