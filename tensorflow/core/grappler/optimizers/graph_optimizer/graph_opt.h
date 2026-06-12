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
#ifndef ANNC_TF_GRAPH_OPT_H_
#define ANNC_TF_GRAPH_OPT_H_
#include <type_traits>
#include <unordered_map>

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"

namespace annc {
#define CHECK_NODE_OK(x) \
  if (!(x)) {            \
    return false;        \
  }

static const std::string fusion_appendix = "/kp_fused";

void update_node_indexes(const tensorflow::GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes);

class PatternRewriter {
 public:
  PatternRewriter() {}
  virtual ~PatternRewriter() = default;

  virtual bool match_and_rewrite(
      const tensorflow::NodeDef* node, tensorflow::GraphDef* graph,
      const tensorflow::grappler::GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) = 0;

  virtual std::string name() const { return "PatternRewriter"; };

  const tensorflow::NodeDef* get_node(const std::string& name);
  tensorflow::NodeDef* get_mutable_node(const std::string& name);

  tensorflow::NodeDef* get_operand(const tensorflow::NodeDef* node, std::string op_type);

  const tensorflow::NodeDef* get_user(const tensorflow::NodeDef* node, int index,
                          const std::string& op_type);

  void replace_all_users_with(const tensorflow::NodeDef* old_node, int old_index,
                              const tensorflow::NodeDef* new_node, int new_index,
                              tensorflow::GraphDef* graph);

  bool check_input_dims(const tensorflow::grappler::GraphProperties& graph_properties,
                      const tensorflow::NodeDef* op, int input_index,
                      int expected_dim_size) {
    if (op == nullptr) return false;
    const auto& input_props = graph_properties.GetInputProperties(op->name());
    if (input_index >= static_cast<int>(input_props.size())) {
      return false;
    }
    const tensorflow::TensorShapeProto& shape = input_props[input_index].shape();
    std::string shape_str = "[";
    if (shape.unknown_rank()) {
      shape_str = "[?]";  //  rank
    } else {
      for (int i = 0; i < shape.dim_size(); ++i) {
        if (i > 0) shape_str += ", ";
        // -1 "?"
        auto dim_size = shape.dim(i).size();
        if (dim_size == -1) {
          shape_str += "?";
        } else {
          shape_str += std::to_string(dim_size);
        }
      }
      shape_str += "]";
    }

    LOG(INFO) << "  Full input shape: " << shape_str
              << ", rank: " << (shape.unknown_rank() ? -1 : shape.dim_size())
              << ", expected rank: " << expected_dim_size;

    return shape.dim_size() == expected_dim_size;
  }
  bool check_const_dims(tensorflow::NodeDef* op, int dim_size) {
    if (!((tensorflow::grappler::IsConstant(*op) || tensorflow::grappler::IsHostConstant(*op)) &&
          HasNodeAttr(*op, "value")))
      return false;

    tensorflow::TensorProto* tensor = (*op->mutable_attr())["value"].mutable_tensor();
    if (tensor == nullptr) return false;
    const auto& shape = tensor->tensor_shape();
    if (shape.dim_size() != static_cast<int>(dim_size)) return false;
    return true;
  }

  bool check_const_shape(tensorflow::NodeDef* op, std::vector<int> dims) {
    if (!((tensorflow::grappler::IsConstant(*op) || tensorflow::grappler::IsHostConstant(*op)) &&
          HasNodeAttr(*op, "value")))
      return false;

    tensorflow::TensorProto* tensor = (*op->mutable_attr())["value"].mutable_tensor();
    if (tensor == nullptr) return false;
    const auto& shape = tensor->tensor_shape();
    if (shape.dim_size() != static_cast<int>(dims.size())) return false;
    for (int i = 0; i < shape.dim_size(); ++i) {
      if (shape.dim(i).size() != dims[i]) return false;
    }
    return true;
  }

  template <typename T>
  bool check_const_value(tensorflow::NodeDef* op, std::vector<T> cmp) {
    if (!((tensorflow::grappler::IsConstant(*op) || tensorflow::grappler::IsHostConstant(*op)) &&
          HasNodeAttr(*op, "value")))
      return false;

    tensorflow::TensorProto* tensor = (*op->mutable_attr())["value"].mutable_tensor();
    if (tensor == nullptr) return false;
    const auto& shape = tensor->tensor_shape();
    int64_t dim_size = 1;
    for (int i = 0; i < shape.dim_size(); ++i) {
      dim_size *= shape.dim(i).size();
    }
    if (dim_size < static_cast<int64_t>(cmp.size())) return false;

    if (std::is_same<T, float>::value) {
      const float* data = tensor->mutable_float_val()->data();
      if (data == nullptr)
        data = reinterpret_cast<const float*>(tensor->tensor_content().data());
      if (data == nullptr) return false;
      for (int i = 0; i < static_cast<int>(cmp.size()); ++i) {
        if (std::fabs(data[i] - cmp[i]) >= 1e-5f) return false;
      }
    } else if (std::is_same<T, int>::value) {
      const int* data = tensor->mutable_int_val()->data();
      if (data == nullptr)
        data = reinterpret_cast<const int*>(tensor->tensor_content().data());
      if (data == nullptr) return false;
      for (int i = 0; i < static_cast<int>(cmp.size()); ++i) {
        if (data[i] != cmp[i]) return false;
      }
    } else if (std::is_same<T, int64_t>::value) {
      const int64_t* data = tensor->mutable_int64_val()->data();
      if (data == nullptr)
        data =
            reinterpret_cast<const int64_t*>(tensor->tensor_content().data());
      if (data == nullptr) return false;
      for (int i = 0; i < static_cast<int>(cmp.size()); ++i) {
        if (data[i] != cmp[i]) return false;
      }
    } else {
      // data type do not support
      return false;
    }
    return true;
  }

  bool check_int_attr(const tensorflow::NodeDef* op, std::string name, int value) {
    if (HasNodeAttr(*op, name)) {
      tensorflow::AttrValue attr = op->attr().at(name);
      if (attr.value_case() == tensorflow::AttrValue::kI && attr.i() == value) return true;
    }
    return false;
  }

  tensorflow::GraphDef* graph_;
  std::unordered_map<std::string, int>* indexes_;
};

class GraphOptimizer {
 public:
  GraphOptimizer(tensorflow::GraphDef* graph, tensorflow::grappler::GraphProperties graph_properties) : graph_(graph), props_(graph_properties) {}
  virtual ~GraphOptimizer() = default;

  void register_rewriter(std::unique_ptr<PatternRewriter> rewriter);

  void optimize();

 private:
  tensorflow::GraphDef* graph_;
  tensorflow::grappler::GraphProperties props_;
  std::unordered_map<std::string, int> node_indexes_;
  std::vector<std::unique_ptr<PatternRewriter>> rewriters_;
};

void run_graph_optimization(tensorflow::GraphDef* graph, tensorflow::grappler::GraphProperties props);
}  // namespace annc
#endif  // ANNC_TF_GRAPH_OPT_H_
