/* Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/

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

class KPFusedDirectHashModOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_direct_hash_mod",
                                "KPFusedDirectHashMod")
                     .Input(FakeInput(DT_INT32))  // data
                     .Input(FakeInput(DT_INT64))  // add_y
                     .Input(FakeInput(DT_INT64))  // mod_y
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedDirectHashModOpTest, TestBasic) {
  MakeOp();

  AddInputFromArray<int32>(TensorShape({1, 8}), {0, 2, 3, 0, 4, 5, 0, 0});
  AddInputFromArray<int64>(TensorShape({}), {4});  // add_y
  AddInputFromArray<int64>(TensorShape({}), {5});  // mod_y

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT64, TensorShape({1, 8}));
  test::FillValues<int64>(&expected, {0, 1, 2, 0, 3, 4, 0, 0});
  test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
}

TEST_F(KPFusedDirectHashModOpTest, TestModBoundary) {
  MakeOp();

  AddInputFromArray<int32>(TensorShape({1, 3}), {1, 2, 3});
  AddInputFromArray<int64>(TensorShape({}), {4});
  AddInputFromArray<int64>(TensorShape({}), {5});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT64, TensorShape({1, 3}));
  test::FillValues<int64>(&expected,
                          {0, 1, 2});  // (1+4)%5=0, (2+4)%5=1, (3+4)%5=2
  test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow