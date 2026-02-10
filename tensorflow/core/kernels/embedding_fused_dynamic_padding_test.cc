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

class KPFusedDynamicPaddingOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_dynamic_padding",
                                "KPFusedDynamicPadding")
                     .Input(FakeInput(DT_INT32))  // data
                     .Input(FakeInput(DT_INT32))  // sub_x
                     .Input(FakeInput(DT_INT32))  // less_y
                     .Attr("rank", 0)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedDynamicPaddingOpTest, TestNoPadding) {
  MakeOp();
  // input_data shape : [2, 4] ,dim_last = 4, less_y = 3 -> No padding needed.
  AddInputFromArray<int32>(TensorShape({2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {4});  // sub_x
  AddInputFromArray<int32>(TensorShape({}), {3});  // less_y

  TF_ASSERT_OK(RunOpKernel());

  // output_less: false
  Tensor expected_less(allocator(), DT_BOOL, TensorShape({}));
  test::FillValues<bool>(&expected_less, {false});
  test::ExpectTensorEqual<bool>(expected_less, *GetOutput(0));

  // output: 原样输出
  Tensor expected_out(allocator(), DT_INT32, TensorShape({2, 4}));
  test::FillValues<int32>(&expected_out, {0, 2, 3, 0, 4, 5, 0, 0});
  test::ExpectTensorEqual<int32>(expected_out, *GetOutput(1));
}

TEST_F(KPFusedDynamicPaddingOpTest, TestBasic) {
  MakeOp();
  // input_data shape : [2, 4] ,dim_last = 4, less_y = 6 -> 补两列
  AddInputFromArray<int32>(TensorShape({2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {6});  // sub_x
  AddInputFromArray<int32>(TensorShape({}), {6});  // less_y

  TF_ASSERT_OK(RunOpKernel());

  // output_less: true
  Tensor expected_less(allocator(), DT_BOOL, TensorShape({}));
  test::FillValues<bool>(&expected_less, {true});
  test::ExpectTensorEqual<bool>(expected_less, *GetOutput(0));

  // output: 右侧补零
  Tensor expected_out(allocator(), DT_INT32, TensorShape({2, 6}));
  test::FillValues<int32>(&expected_out, {0, 2, 3, 0, 0, 0, 4, 5, 0, 0, 0, 0});
  test::ExpectTensorEqual<int32>(expected_out, *GetOutput(1));
}

TEST_F(KPFusedDynamicPaddingOpTest, TestNotScalar) {
  MakeOp();
  AddInputFromArray<int32>(TensorShape({2, 4}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 7});  // sub_x
  AddInputFromArray<int32>(TensorShape({}), {6});      // less_y

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.error_message().find("sub_x must be a scalar") !=
              std::string::npos);
}
}  // namespace
}  // namespace tensorflow
