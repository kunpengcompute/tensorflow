/* Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class KPFusedTruncSeqOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_trunc_seq",
                                "KPFusedTruncSeq")
                     .Input(FakeInput(DT_INT32))  // input
                     .Input(FakeInput(DT_INT32))  // min_x
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedTruncSeqOpTest, TestBasic) {  // min_x < dim_1
  MakeOp();

  AddInputFromArray<int32>(TensorShape({2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {3});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {0, 2, 3, 4, 5, 0});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(KPFusedTruncSeqOpTest, TestNoTruncation) {  // min_x >= dim_1
  MakeOp();

  AddInputFromArray<int32>(TensorShape({2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {5});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2, 4}));
  test::FillValues<int32>(&expected, {0, 2, 3, 0, 4, 5, 0, 0});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(KPFusedTruncSeqOpTest, TestInvalidInputDim1D) {
  MakeOp();

  AddInputFromArray<int32>(TensorShape({8}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {5});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.error_message().find("Input data shape must be 2D") !=
              std::string::npos);
}

TEST_F(KPFusedTruncSeqOpTest, TestInvalidInputDim3D) {
  MakeOp();

  AddInputFromArray<int32>(TensorShape({1, 2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {5});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.error_message().find("Input data shape must be 2D") !=
              std::string::npos);
}
TEST_F(KPFusedTruncSeqOpTest, TestNegativeMinX) {
  MakeOp();

  AddInputFromArray<int32>(TensorShape({2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {-5});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.error_message().find("min_x must be positive") !=
              std::string::npos);
}
}  // namespace
}  // namespace tensorflow