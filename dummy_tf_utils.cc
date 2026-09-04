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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kDefaultBatchSize = 1;

}  // namespace

tensorflow::TensorShape ConvertToTensorShape(
    const tensorflow::TensorShapeProto& shape_proto) {
  std::vector<int64_t> dims;
  for (const auto& dim : shape_proto.dim()) {
    const int64_t size = dim.size() > 0 ? dim.size() : kDefaultBatchSize;
    dims.push_back(size);
  }
  return tensorflow::TensorShape(dims);
}

std::vector<tensorflow::string> SplitBatchModelInput(
    const tensorflow::string& request_message) {
  std::vector<tensorflow::string> batch;
  std::stringstream ss(request_message);
  tensorflow::string line;
  while (std::getline(ss, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      batch.emplace_back(std::move(line));
    }
  }
  if (batch.empty()) {
    batch.emplace_back(request_message);
  }
  return batch;
}

tensorflow::Tensor CreateStringBatchTensor(
    const std::vector<tensorflow::string>& values) {
  const int64_t batch_size = std::max<int64_t>(1, values.size());
  tensorflow::Tensor tensor(tensorflow::DT_STRING,
                            tensorflow::TensorShape({batch_size}));
  auto flat = tensor.flat<tensorflow::tstring>();
  for (int64_t i = 0; i < flat.size(); ++i) {
    flat(i) = values.empty() ? "" : values[static_cast<size_t>(i)];
  }
  return tensor;
}
