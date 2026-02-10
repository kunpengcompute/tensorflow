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

class KPFusedTopKSegmentMinOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_topk_segment_min",
                                "KPFusedTopKSegmentMin")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedTopKSegmentMinOpTest, TestOrderScore) {
  MakeOp();  // num_partitions = 2

  AddInputFromArray<float>(TensorShape({1,8}), {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
  AddInputFromArray<int32>(TensorShape({}),{8});
  AddInputFromArray<int64>(TensorShape({1, 8}), {1, 2, 3, 4, 5, 6, 7, 8});
  AddInputFromArray<int64>(TensorShape({1, 8}), {11, 12, 13, 14, 15, 16, 17, 18});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_FLOAT, TensorShape({1, 8}));
  test::FillValues<float>(&expected_1,
                          {0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f});
  test::ExpectTensorEqual<float>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_INT64, TensorShape({1, 8}));
  test::FillValues<int64>(&expected_2,
                          {18, 17, 16, 15, 14, 13, 12, 11});
  test::ExpectTensorEqual<int64>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedTopKSegmentMinOpTest, TestUnOrderScore) {
  MakeOp();  // num_partitions = 2

  AddInputFromArray<float>(TensorShape({1,8}), {0.6f, 0.2f, 0.5f, 0.3f, 0.4f, 0.1f, 0.7f, 0.8f});
  AddInputFromArray<int32>(TensorShape({}),{8});
  AddInputFromArray<int64>(TensorShape({1, 8}), {8, 7, 3, 2, 5, 4, 1, 6});
  AddInputFromArray<int64>(TensorShape({1, 8}), {11, 12, 13, 14, 15, 16, 17, 18});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_FLOAT, TensorShape({1, 8}));
  test::FillValues<float>(&expected_1,
                          {0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f});
  test::ExpectTensorEqual<float>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_INT64, TensorShape({1, 8}));
  test::FillValues<int64>(&expected_2,
                          {18, 17, 11, 13, 15, 14, 12, 16});
  test::ExpectTensorEqual<int64>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedTopKSegmentMinOpTest, TestIdenticalScore) {
  MakeOp();  // num_partitions = 2

  AddInputFromArray<float>(TensorShape({1,8}), {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  AddInputFromArray<int32>(TensorShape({}),{8});
  AddInputFromArray<int64>(TensorShape({1, 8}), {8, 7, 3, 2, 5, 4, 1, 6});
  AddInputFromArray<int64>(TensorShape({1, 8}), {18, 12, 16, 14, 15, 13, 17, 11});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_FLOAT, TensorShape({1, 8}));
  test::FillValues<float>(&expected_1,
                          {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test::ExpectTensorEqual<float>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_INT64, TensorShape({1, 8}));
  test::FillValues<int64>(&expected_2,
                          {18, 12, 16, 14, 15, 13, 17, 11});
  test::ExpectTensorEqual<int64>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedTopKSegmentMinOpTest, TestRepeatId) {
  MakeOp();  // num_partitions = 2

  AddInputFromArray<float>(TensorShape({1,8}), {0.6f, 0.2f, 0.5f, 0.3f, 0.4f, 0.1f, 0.7f, 0.8f});
  AddInputFromArray<int32>(TensorShape({}),{8});
  AddInputFromArray<int64>(TensorShape({1, 8}), {8, 7, 3, 2, 3, 5, 1, 6});
  AddInputFromArray<int64>(TensorShape({1, 8}), {18, 12, 16, 14, 15, 13, 17, 11});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({7}));
  test::FillValues<int32>(&expected, {0, 1, 2, 3, 5, 6, 7});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_FLOAT, TensorShape({1, 8}));
  test::FillValues<float>(&expected_1,
                          {0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f});
  test::ExpectTensorEqual<float>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_INT64, TensorShape({1, 8}));
  test::FillValues<int64>(&expected_2,
                          {11, 17, 18, 16, 15, 14, 12, 13});
  test::ExpectTensorEqual<int64>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedTopKSegmentMinOpTest, TestRepeatId2) {
  MakeOp();  // num_partitions = 2

  AddInputFromArray<float>(TensorShape({1,8}), {0.6f, 0.2f, 0.5f, 0.3f, 0.4f, 0.1f, 0.7f, 0.8f});
  AddInputFromArray<int32>(TensorShape({}),{8});
  AddInputFromArray<int64>(TensorShape({1, 8}), {1, 1, 1, 1, 1, 1, 1, 1});
  AddInputFromArray<int64>(TensorShape({1, 8}), {18, 12, 16, 14, 15, 13, 17, 11});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int32>(&expected, {0});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_FLOAT, TensorShape({1, 8}));
  test::FillValues<float>(&expected_1,
                          {0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f});
  test::ExpectTensorEqual<float>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_INT64, TensorShape({1, 8}));
  test::FillValues<int64>(&expected_2,
                          {11, 17, 18, 16, 15, 14, 12, 13});
  test::ExpectTensorEqual<int64>(expected_2, *GetOutput(2));
}

}  // namespace
}  // namespace tensorflow