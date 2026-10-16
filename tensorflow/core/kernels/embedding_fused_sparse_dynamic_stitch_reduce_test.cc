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

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests for the fused SVE reduce kernels:
//   KPFusedSparseDynamicStitch{Sum,Mean,SqrtN,FastSum,FastMean,FastSqrtN}.
// The op gathers rows from N embedding tables according to flattened global
// ids, then sums (or means) rows that share the same segment id.
class KPFusedSparseDynamicStitchReduceOpTest : public OpsTestBase {
 protected:
  void MakeOp(const std::string& name, int N, int64_t emb_tables_len) {
    TF_ASSERT_OK(NodeDefBuilder(name, name)
                     .Input(FakeInput(DT_INT64))     // x
                     .Input(FakeInput(N, DT_FLOAT))  // variables
                     .Input(FakeInput(DT_INT64))     // slice_input
                     .Input(FakeInput(DT_INT32))     // begin
                     .Input(FakeInput(DT_INT32))     // end
                     .Input(FakeInput(DT_INT32))     // strides
                     .Attr("emb_tables_len", emb_tables_len)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  // Standard two-table inputs:
  //   x = {0, 2, 1, 3}; segment_ids (col 0) = {0, 0, 1, 1}.
  // With N == 2 this gathers table0 rows {0, 1} into segment 0 and
  // table1 rows {0, 1} into segment 1.
  void AddStandardInputs() {
    AddInputFromArray<int64>(TensorShape({4}), {0, 2, 1, 3});
    AddInputFromArray<float>(TensorShape({2, 2}),
                             {1.0f, 2.0f, 3.0f, 4.0f});
    AddInputFromArray<float>(TensorShape({2, 2}),
                             {5.0f, 6.0f, 7.0f, 8.0f});
    AddInputFromArray<int64>(TensorShape({4, 1}), {0, 0, 1, 1});
    AddInputFromArray<int32>(TensorShape({2}), {0, 0});
    AddInputFromArray<int32>(TensorShape({2}), {4, 1});
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  }
};

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestSum) {
  MakeOp("KPFusedSparseDynamicStitchSum", 2, 2);
  AddStandardInputs();
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected, {4.0f, 6.0f, 12.0f, 14.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestMean) {
  MakeOp("KPFusedSparseDynamicStitchMean", 2, 2);
  AddStandardInputs();
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected, {2.0f, 3.0f, 6.0f, 7.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestSqrtN) {
  MakeOp("KPFusedSparseDynamicStitchSqrtN", 2, 2);
  AddStandardInputs();
  TF_ASSERT_OK(RunOpKernel());

  // Each segment holds two rows: sum / sqrt(2).
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected,
                          {2.8284271f, 4.2426407f, 8.485281f, 9.899495f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestFastSum) {
  MakeOp("KPFusedSparseDynamicStitchFastSum", 2, 2);
  AddStandardInputs();
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected, {4.0f, 6.0f, 12.0f, 14.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestFastMean) {
  MakeOp("KPFusedSparseDynamicStitchFastMean", 2, 2);
  AddStandardInputs();
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected, {2.0f, 3.0f, 6.0f, 7.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestFastSqrtN) {
  MakeOp("KPFusedSparseDynamicStitchFastSqrtN", 2, 2);
  AddStandardInputs();
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&expected,
                          {2.8284271f, 4.2426407f, 8.485281f, 9.899495f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestFastSumFourTables) {
  // Exercise the fast path with a shift larger than one (N == 4).
  MakeOp("KPFusedSparseDynamicStitchFastSum", 4, 2);

  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 2, 3});
  AddInputFromArray<float>(TensorShape({1, 2}), {1.0f, 2.0f});
  AddInputFromArray<float>(TensorShape({1, 2}), {3.0f, 4.0f});
  AddInputFromArray<float>(TensorShape({1, 2}), {5.0f, 6.0f});
  AddInputFromArray<float>(TensorShape({1, 2}), {7.0f, 8.0f});
  AddInputFromArray<int64>(TensorShape({4, 1}), {0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {4, 1});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2}));
  test::FillValues<float>(&expected, {16.0f, 20.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestFastSumRequiresPowerOfTwo) {
  TF_ASSERT_OK(NodeDefBuilder("kp_fast_sum_bad", "KPFusedSparseDynamicStitchFastSum")
                   .Input(FakeInput(DT_INT64))
                   .Input(FakeInput(3, DT_FLOAT))
                   .Input(FakeInput(DT_INT64))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Attr("emb_tables_len", 2)
                   .Finalize(node_def()));
  Status s = InitOp();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "power of two"));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestUnsortedSegmentIds) {
  MakeOp("KPFusedSparseDynamicStitchSum", 2, 2);

  AddInputFromArray<int64>(TensorShape({4}), {0, 2, 1, 3});
  AddInputFromArray<float>(TensorShape({2, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f});
  AddInputFromArray<float>(TensorShape({2, 2}),
                           {5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int64>(TensorShape({4, 1}), {1, 0, 1, 1});  // unsorted
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {4, 1});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "sorted"));
}

TEST_F(KPFusedSparseDynamicStitchReduceOpTest, TestRowIdOutOfRange) {
  MakeOp("KPFusedSparseDynamicStitchSum", 2, 2);

  AddInputFromArray<int64>(TensorShape({1}), {4});  // table0 row 2 is OOB
  AddInputFromArray<float>(TensorShape({2, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f});
  AddInputFromArray<float>(TensorShape({2, 2}),
                           {5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int64>(TensorShape({1, 1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "row_id out of range"));
}

}  // namespace
}  // namespace tensorflow
