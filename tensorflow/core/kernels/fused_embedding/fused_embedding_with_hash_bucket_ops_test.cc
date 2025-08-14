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

#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class KPLookupEmbeddingByHashOpTest : public OpsTestBase {};

TEST_F(KPLookupEmbeddingByHashOpTest, WithHashBucket) {
  TF_EXPECT_OK(
      NodeDefBuilder("fused_embedding_with_hash_bucket_op", "KPLookupEmbeddingByHash")
          .Input(FakeInput(DT_STRING))
          .Input(FakeInput(DT_FLOAT))
          .Attr("num_buckets", 5)
          .Attr("combiner", 1)
          .Attr("T_weight", DT_FLOAT)
          .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({2, 1}),
                           {"ktfop", "fused_embedding_with_hash_bucket"});
  AddInputFromArray<float>(TensorShape({5, 10}), {3.21, 7.89, 1.45, 9.32, 0.67, 5.43, 2.98, 8.76, 4.12, 6.54,
                                                  0.23, 9.87, 3.56, 7.01, 2.34, 8.09, 5.67, 1.89, 6.78, 4.45,
                                                  0.98, 9.01, 3.45, 7.23, 2.67, 8.34, 5.89, 1.23, 6.45, 4.78,
                                                  0.56, 9.45, 3.78, 7.56, 2.12, 8.67, 5.34, 1.67, 6.23, 4.89,
                                                  0.34, 9.78, 3.12, 7.45, 2.89, 8.23, 5.78, 1.45, 6.67, 4.23});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 10}));
  test::FillValues<float>(
      &expected, {3.21, 7.89, 1.45, 9.32, 0.67, 5.43, 2.98, 8.76, 4.12, 6.54,
                  0.98, 9.01, 3.45, 7.23, 2.67, 8.34, 5.89, 1.23, 6.45, 4.78});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.0);
}
}  // namespace tensorflow
