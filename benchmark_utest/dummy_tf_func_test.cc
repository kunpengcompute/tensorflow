// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "dummy_tf_utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

using tensorflow::Tensor;
using tensorflow::string;

namespace {

TEST(DummyTfFuncTest, ConvertsDynamicDimensionsToOne) {
  tensorflow::TensorShapeProto shape_proto;
  shape_proto.add_dim()->set_size(-1);
  shape_proto.add_dim()->set_size(0);
  shape_proto.add_dim()->set_size(8);

  const tensorflow::TensorShape shape = ConvertToTensorShape(shape_proto);

  ASSERT_EQ(shape.dims(), 3);
  EXPECT_EQ(shape.dim_size(0), 1);
  EXPECT_EQ(shape.dim_size(1), 1);
  EXPECT_EQ(shape.dim_size(2), 8);
}

TEST(DummyTfFuncTest, SplitsLfAndCrLfBatchMessages) {
  const std::vector<string> values =
      SplitBatchModelInput("first\r\n\nsecond\nthird\r\n");

  EXPECT_EQ(values, (std::vector<string>{"first", "second", "third"}));
}

TEST(DummyTfFuncTest, KeepsEmptyMessageAsOneBatchItem) {
  const std::vector<string> values = SplitBatchModelInput("");

  EXPECT_EQ(values, (std::vector<string>{""}));
}

TEST(DummyTfFuncTest, CreatesStringTensorFromBatchValues) {
  const Tensor tensor = CreateStringBatchTensor({"first", "second"});

  ASSERT_EQ(tensor.dims(), 1);
  ASSERT_EQ(tensor.dim_size(0), 2);
  EXPECT_EQ(tensor.dtype(), tensorflow::DT_STRING);
  EXPECT_EQ(tensor.flat<tensorflow::tstring>()(0), "first");
  EXPECT_EQ(tensor.flat<tensorflow::tstring>()(1), "second");
}

TEST(DummyTfFuncTest, CreatesSafeTensorForEmptyBatch) {
  const Tensor tensor = CreateStringBatchTensor({});

  ASSERT_EQ(tensor.dim_size(0), 1);
  EXPECT_TRUE(tensor.flat<tensorflow::tstring>()(0).empty());
}

}  // namespace
