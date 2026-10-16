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
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#endif

#include "graph_opt.h"

using namespace tensorflow;
using namespace tensorflow::grappler;

namespace annc {
void update_node_indexes(const GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes) {
  for (int i = 0; i < graph->node_size(); ++i) {
    node_indexes[graph->node(i).name()] = i;
  }
}

void GraphOptimizer::register_rewriter(
    std::unique_ptr<PatternRewriter> rewriter) {
  rewriters_.push_back(std::move(rewriter));
}

void GraphOptimizer::optimize() {
  update_node_indexes(graph_, node_indexes_);
  int node_index = 0;
  const int node_size = graph_->node_size();
  while (node_index < node_size) {
    const NodeDef& node = graph_->node(node_index);
    const std::string& node_name = node.name();
    for (auto& rewriter : rewriters_) {
      if (rewriter->match_and_rewrite(&node, graph_, props_, node_indexes_)) {
        update_node_indexes(graph_, node_indexes_);
        const std::string new_node_name = node_name + fusion_appendix;
        auto it = node_indexes_.find(new_node_name);
        if (it != node_indexes_.end()) {
          node_index = it->second;
        }
        break;
      }
    }
    node_index++;
  }
}

std::string get_node_name(const std::string& name) {
  size_t colon_pos = name.find_last_of(':');
  std::string node_name = name;
  if (colon_pos != std::string::npos) {
    node_name = name.substr(0, colon_pos);
  }
  return node_name;
}

void set_fusedop_attributes(NodeDef* fused,
                            const absl::Span<const absl::string_view> fused_ops,
                            int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);  // required only for BatchNorm
}

const NodeDef* PatternRewriter::get_node(const std::string& name) {
  const std::string node_name = get_node_name(name);
  if (indexes_->find(node_name) == indexes_->end()) return nullptr;
  const int node_index = indexes_->at(node_name);
  return &graph_->node(node_index);
}

NodeDef* PatternRewriter::get_mutable_node(const std::string& name) {
  const std::string node_name = get_node_name(name);
  if (indexes_->find(node_name) == indexes_->end()) return nullptr;
  const int node_index = indexes_->at(node_name);
  return graph_->mutable_node(node_index);
}

NodeDef* PatternRewriter::get_operand(const NodeDef* node,
                                      std::string op_type) {
  for (int i = 0; i < node->input_size(); ++i) {
    NodeDef* operand = get_mutable_node(node->input(i));
    if (operand != nullptr && operand->op() == op_type) return operand;
  }
  return nullptr;
}

const NodeDef* PatternRewriter::get_user(const NodeDef* node, int index,
                                         const std::string& op_type) {
  std::string node_name = node->name();
  if (index) node_name = node_name + ":" + std::to_string(index);
  for (int i = 0; i < graph_->node_size(); ++i) {
    const NodeDef* node = &graph_->node(i);
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == node_name && node->op() == op_type) {
        return node;
      }
    }
  }
  return nullptr;
}

void PatternRewriter::replace_all_users_with(const NodeDef* old_node,
                                             int old_index,
                                             const NodeDef* new_node,
                                             int new_index, GraphDef* graph) {
  std::string old_name = old_node->name();
  if (old_index) old_name = old_name + ":" + std::to_string(old_index);
  std::string new_name = new_node->name();
  if (new_index) new_name = new_name + ":" + std::to_string(new_index);
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == old_name) {
        node->set_input(j, new_name);
      }
    }
  }
}

class KPFusedSparseDynamicStitchRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseDynamicStitch"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "ParallelDynamicStitch" &&
                  node->input_size() % 2 == 0)
    int num_inputs = node->input_size();
    int num_partitions = num_inputs / 2;
    // left branch
    const NodeDef* partition = get_node(node->input(0));
    CHECK_NODE_OK(partition != nullptr &&
                  partition->op() == "DynamicPartition" &&
                  partition->input_size() == 2)
    const NodeDef* range = get_node(partition->input(0));
    CHECK_NODE_OK(range != nullptr && range->op() == "Range" &&
                  range->input_size() == 3)
    // Range start=0, delta=1
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(range->input(0)), {0}))
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(range->input(2)), {1}))
    const NodeDef* size = get_node(range->input(1));
    CHECK_NODE_OK(size != nullptr && IsSize(*size) &&
                  size->input_size() == 1)
    const NodeDef* cast = get_node(partition->input(1));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast) &&
                  cast->input_size() == 1)
    const NodeDef* floor_mod = get_node(cast->input(0));
    CHECK_NODE_OK(floor_mod != nullptr &&
                  floor_mod->op() == "FloorMod" &&
                  floor_mod->input_size() == 2)
    CHECK_NODE_OK(check_const_value<int64_t>(
        get_mutable_node(floor_mod->input(1)), {num_partitions}))

    CHECK_NODE_OK(check_int_attr(node, "N", {num_partitions}))
    CHECK_NODE_OK(check_int_attr(partition, "num_partitions", {num_partitions}))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(size->input(0));
    // right branch
    for (int i = num_partitions; i < num_inputs; ++i) {
      const NodeDef* gather = get_node(node->input(i));
      CHECK_NODE_OK(gather != nullptr &&
                    gather->op() == "GatherV2" &&
                    gather->input_size() == 3)
      // Gather axis=0
      CHECK_NODE_OK(
          check_const_value<int>(get_mutable_node(gather->input(2)), {0}))
      const NodeDef* partition_1 = get_node(gather->input(1));
      CHECK_NODE_OK(partition_1 != nullptr &&
                    partition_1->op() == "DynamicPartition" &&
                    partition_1->input_size() == 2)
      CHECK_NODE_OK(
          check_int_attr(partition_1, "num_partitions", {num_partitions}))
      const NodeDef* floor_div = get_node(partition_1->input(0));
      CHECK_NODE_OK(floor_div != nullptr &&
                    floor_div->op() == "FloorDiv" &&
                    floor_div->input_size() == 2)
      CHECK_NODE_OK(check_const_value<int64_t>(
          get_mutable_node(floor_div->input(1)), {num_partitions}))
      CHECK_NODE_OK(check_input_dims(props, gather, 0, 2));
      fused_node.add_input(gather->input(0));
    }
    (*fused_node.mutable_attr())["N"].set_i(num_partitions);
    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(node, 0, added, 0, graph);
    return true;
  }
};

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

class KPFusedSparseDynamicStitchReduceRewriter : public PatternRewriter {
 public:
  KPFusedSparseDynamicStitchReduceRewriter() = default;

  std::string name() const override { return "KPFusedSparseDynamicStitch"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    VLOG(1) << "KPFusedSparseDynamicStitch check op: candidate; name="
            << node->name() << "; op=" << node->op()
            << "; inputs=" << node->input_size();
    if (!((node->op() == "SparseSegmentSum" ||
           node->op() == "SparseSegmentMean" ||
           node->op() == "SparseSegmentSqrtN") &&
          node->input_size() == 3)) {
      return false;
    }
    VLOG(1) << "KPFusedSparseDynamicStitch match candidate: " << node->name()
            << "; op=" << node->op() << "; inputs=" << node->input_size();
    // The Reshape ("gathered") node between SparseSegment* and
    // ParallelDynamicStitch may be optimized away by earlier passes. Support
    // both layouts: with Reshape we peel it off first, otherwise node->input(0)
    // is the ParallelDynamicStitch directly.
    const NodeDef* gathered = get_node(node->input(0));
    CHECK_NODE_OK(gathered != nullptr)
    const bool has_reshape =
        gathered->op() == "Reshape" && gathered->input_size() == 2;
    VLOG(1) << "KPFusedSparseDynamicStitch check value: has_reshape="
            << has_reshape << "; gathered_op=" << gathered->op();
    const NodeDef* stitch =
        has_reshape ? get_node(gathered->input(0)) : gathered;
    CHECK_NODE_OK(stitch != nullptr && stitch->op() == "ParallelDynamicStitch" &&
                  stitch->input_size() % 2 == 0)
    const int stitch_inputs = stitch->input_size();
    const int partitions = stitch_inputs / 2;
    VLOG(1) << "KPFusedSparseDynamicStitch check value: partitions="
            << partitions;
    const bool use_fast = IsPowerOfTwo(partitions);
    // Fast 与非 Fast 共用融合图，仅按表数量选择解码实现。
    // left branch
    const NodeDef* partition = get_node(stitch->input(0));
    CHECK_NODE_OK(partition != nullptr &&
                  partition->op() == "DynamicPartition" &&
                  partition->input_size() == 2)
    const NodeDef* range = get_node(partition->input(0));
    CHECK_NODE_OK(range != nullptr && range->op() == "Range" &&
                  range->input_size() == 3)
    // Range start=0, delta=1
    NodeDef* range_start = get_mutable_node(range->input(0));
    CHECK_NODE_OK(range_start != nullptr &&
                  check_const_value<int>(range_start, {0}))
    NodeDef* range_delta = get_mutable_node(range->input(2));
    CHECK_NODE_OK(range_delta != nullptr &&
                  check_const_value<int>(range_delta, {1}))
    const NodeDef* size = get_node(range->input(1));
    CHECK_NODE_OK(size != nullptr && IsSize(*size) && size->input_size() == 1)
    const NodeDef* cast = get_node(partition->input(1));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast) && cast->input_size() == 1)
    const NodeDef* floor_mod = get_node(cast->input(0));
    CHECK_NODE_OK(floor_mod != nullptr && floor_mod->op() == "FloorMod" &&
                  floor_mod->input_size() == 2)
    NodeDef* floor_mod_divisor = get_mutable_node(floor_mod->input(1));
    CHECK_NODE_OK(floor_mod_divisor != nullptr &&
                  check_const_value<int64_t>(floor_mod_divisor, {partitions}))

    CHECK_NODE_OK(check_int_attr(stitch, "N", partitions))
    CHECK_NODE_OK(check_int_attr(partition, "num_partitions", partitions))

    // The Reshape ("unique_input") between the size branch and Unique may be
    // optimized away. Support both layouts: with Reshape we peel it off,
    // otherwise size->input(0) is the Unique node directly.
    const NodeDef* unique_input = get_node(size->input(0));
    CHECK_NODE_OK(unique_input != nullptr)
    const bool unique_has_reshape =
        unique_input->op() == "Reshape" && unique_input->input_size() == 2;
    VLOG(1) << "KPFusedSparseDynamicStitch check value: unique_has_reshape="
            << unique_has_reshape << "; unique_input_op=" << unique_input->op();
    const NodeDef* unique =
        unique_has_reshape ? get_node(unique_input->input(0)) : unique_input;
    CHECK_NODE_OK(unique != nullptr && unique->op() == "Unique" &&
                  unique->input_size() == 1)
    // The fused op re-derives the pooling order from the Unique node instead
    // of taking the SparseSegment* indices input, so fusion is only valid
    // when that input is exactly the idx output (port 1) of this Unique.
    CHECK_NODE_OK(node->input(1) == unique->name() + ":1")

    // The Cast ("segment_cast") between SparseSegment* input(2) and
    // StridedSlice may be optimized away. Support both layouts: with Cast we
    // peel it off, otherwise node->input(2) is the StridedSlice directly.
    const NodeDef* segment_cast = get_node(node->input(2));
    CHECK_NODE_OK(segment_cast != nullptr)
    const bool segment_has_cast =
        segment_cast->op() == "Cast" && segment_cast->input_size() == 1;
    VLOG(1) << "KPFusedSparseDynamicStitch check value: segment_has_cast="
            << segment_has_cast << "; segment_cast_op=" << segment_cast->op();
    const NodeDef* strided_slice =
        segment_has_cast ? get_node(segment_cast->input(0)) : segment_cast;
    CHECK_NODE_OK(strided_slice != nullptr &&
                  strided_slice->op() == "StridedSlice" &&
                  strided_slice->input_size() == 4)
    VLOG(1) << "KPFusedSparseDynamicStitch check value: strided_slice required "
               "attrs; begin_mask="
            << HasNodeAttr(*strided_slice, "begin_mask")
            << "; end_mask=" << HasNodeAttr(*strided_slice, "end_mask")
            << "; shrink_axis_mask="
            << HasNodeAttr(*strided_slice, "shrink_axis_mask");
    CHECK_NODE_OK(HasNodeAttr(*strided_slice, "begin_mask") &&
                  HasNodeAttr(*strided_slice, "end_mask") &&
                  HasNodeAttr(*strided_slice, "shrink_axis_mask"))
    CHECK_NODE_OK(check_int_attr(strided_slice, "begin_mask", 1))
    CHECK_NODE_OK(check_int_attr(strided_slice, "end_mask", 1))
    CHECK_NODE_OK(check_int_attr(strided_slice, "shrink_axis_mask", 2))
    CHECK_NODE_OK(check_int_attr(strided_slice, "ellipsis_mask", 0))
    CHECK_NODE_OK(check_int_attr(strided_slice, "new_axis_mask", 0))

    const NodeDef* first_gather = get_node(stitch->input(partitions));
    CHECK_NODE_OK(first_gather != nullptr && first_gather->input_size() == 3)
    const auto& table_props = props.GetInputProperties(first_gather->name());
    VLOG(1) << "KPFusedSparseDynamicStitch check value: table_props size="
            << table_props.size()
            << (table_props.empty()
                    ? ""
                    : absl::StrCat("; shape=",
                                   table_props[0].shape().ShortDebugString()));
    CHECK_NODE_OK(!table_props.empty() &&
                  table_props[0].shape().dim_size() == 2 &&
                  table_props[0].shape().dim_size() > 1)
    const int64_t embedding_width = table_props[0].shape().dim(1).size();
    VLOG(1) << "KPFusedSparseDynamicStitch check value: embedding_width="
            << embedding_width;
    CHECK_NODE_OK(embedding_width > 0)

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    const std::string reduce_suffix =
        node->op() == "SparseSegmentMean"    ? "Mean"
        : node->op() == "SparseSegmentSqrtN" ? "SqrtN"
                                             : "Sum";
    fused_node.set_op(
        absl::StrCat("KPFusedSparseDynamicStitch", use_fast ? "Fast" : "",
                     reduce_suffix));
    fused_node.set_device(node->device());
    fused_node.add_input(unique->input(0));
    // right branch
    for (int i = partitions; i < stitch_inputs; ++i) {
      const NodeDef* gather = get_node(stitch->input(i));
      CHECK_NODE_OK(gather != nullptr && gather->op() == "GatherV2" &&
                    gather->input_size() == 3)
      // Gather axis=0
      NodeDef* gather_axis = get_mutable_node(gather->input(2));
      bool gather_axis_value_found = false;
      int64_t gather_axis_value = 0;
      const char* gather_axis_dtype_name = "unsupported";
      if (gather_axis != nullptr &&
          (IsConstant(*gather_axis) || IsHostConstant(*gather_axis)) &&
          HasNodeAttr(*gather_axis, "value")) {
        const TensorProto& axis_tensor_proto =
            gather_axis->attr().at("value").tensor();
        Tensor axis_tensor;
        if (axis_tensor.FromProto(axis_tensor_proto) &&
            axis_tensor.NumElements() > 0) {
          if (axis_tensor.dtype() == DT_INT32) {
            gather_axis_dtype_name = "DT_INT32";
            gather_axis_value = axis_tensor.flat<int32>()(0);
            gather_axis_value_found = true;
          } else if (axis_tensor.dtype() == DT_INT64) {
            gather_axis_dtype_name = "DT_INT64";
            gather_axis_value = axis_tensor.flat<int64_t>()(0);
            gather_axis_value_found = true;
          }
        }
      }
      VLOG(1) << "KPFusedSparseDynamicStitch check value: gather axis"
              << "; dtype=" << gather_axis_dtype_name
              << "; actual="
              << (gather_axis_value_found
                      ? absl::StrCat(gather_axis_value)
                      : "<unavailable>")
              << "; expected=0";
      CHECK_NODE_OK(gather_axis != nullptr && gather_axis_value_found &&
                        gather_axis_value == 0)
      const NodeDef* partition_1 = get_node(gather->input(1));
      CHECK_NODE_OK(partition_1 != nullptr &&
                    partition_1->op() == "DynamicPartition" &&
                    partition_1->input_size() == 2)
      CHECK_NODE_OK(
          check_int_attr(partition_1, "num_partitions", partitions))
      const NodeDef* floor_div = get_node(partition_1->input(0));
      CHECK_NODE_OK(floor_div != nullptr && floor_div->op() == "FloorDiv" &&
                    floor_div->input_size() == 2)
      NodeDef* floor_div_divisor = get_mutable_node(floor_div->input(1));
      CHECK_NODE_OK(floor_div_divisor != nullptr &&
                    check_const_value<int64_t>(floor_div_divisor,
                                               {partitions}))
      fused_node.add_input(gather->input(0));
    }
    fused_node.add_input(strided_slice->input(0));
    fused_node.add_input(strided_slice->input(1));
    fused_node.add_input(strided_slice->input(2));
    fused_node.add_input(strided_slice->input(3));
    (*fused_node.mutable_attr())["N"].set_i(partitions);
    (*fused_node.mutable_attr())["emb_tables_len"].set_i(embedding_width);
    (*fused_node.mutable_attr())["begin_mask"].set_i(
        strided_slice->attr().at("begin_mask").i());
    (*fused_node.mutable_attr())["end_mask"].set_i(
        strided_slice->attr().at("end_mask").i());
    (*fused_node.mutable_attr())["shrink_axis_mask"].set_i(
        strided_slice->attr().at("shrink_axis_mask").i());

    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(node, 0, added, 0, graph);
    return true;
  }
};

class KPFusedSparseSegmentReduceRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseSegmentReduce"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsStridedSlice(*node) && node->input_size() == 4)
    const NodeDef* shape = get_node(node->input(0));
    CHECK_NODE_OK(shape != nullptr && IsShape(*shape) &&
                  shape->input_size() == 1)
    const NodeDef* ss_reduce = get_node(shape->input(0));
    CHECK_NODE_OK(ss_reduce != nullptr && ss_reduce->input_size() == 3)
    AttrValue combiner;
    if (ss_reduce->op() == "SparseSegmentMean")
      combiner.set_i(1);
    else if (ss_reduce->op() == "SparseSegmentSum")
      combiner.set_i(0);
    else
      return false;
    const NodeDef* cast = get_node(ss_reduce->input(2));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast) &&
                  cast->input_size() == 1)
    const NodeDef* strided_slice = get_node(cast->input(0));
    CHECK_NODE_OK(strided_slice != nullptr &&
                  IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)

    // check fusion conditions
    CHECK_NODE_OK(
        check_const_shape(get_mutable_node(strided_slice->input(1)), {2}))
    CHECK_NODE_OK(check_int_attr(strided_slice, "shrink_axis_mask", 2))
    CHECK_NODE_OK(check_int_attr(strided_slice, "begin_mask", 1))
    CHECK_NODE_OK(check_int_attr(strided_slice, "end_mask", 1))
    CHECK_NODE_OK(check_const_shape(get_mutable_node(node->input(1)), {1}))
    CHECK_NODE_OK(check_int_attr(node, "shrink_axis_mask", 1))

    CHECK_NODE_OK(check_input_dims(props, ss_reduce, 0, 2))
    CHECK_NODE_OK(check_input_dims(props, strided_slice, 0, 2))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(ss_reduce->input(0));      // input_tensor
    fused_node.add_input(ss_reduce->input(1));      // indices
    fused_node.add_input(strided_slice->input(0));  // slice_input
    fused_node.add_input(strided_slice->input(1));  // begin
    fused_node.add_input(node->input(1));           // begin_1
    AddNodeAttr("combiner", combiner, &fused_node);
    AddNodeAttr("Tidx", ss_reduce->attr().at("Tidx"), &fused_node);

    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(ss_reduce, 0, added, 0, graph);
    replace_all_users_with(node, 0, added, 1, graph);
    return true;
  }
};

class KPFusedSparseSegmentReduceNonzeroRewriter : public PatternRewriter {
 public:
  std::string name() const override {
    return "KPFusedSparseSegmentReduceNonzero";
  }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "GatherNd" &&
                  node->input_size() == 2)  // output:2
    const NodeDef* ss_reduce = get_node(node->input(0));
    CHECK_NODE_OK(ss_reduce != nullptr && ss_reduce->input_size() == 3)
    AttrValue combiner;
    if (ss_reduce->op() == "SparseSegmentMean") {
      combiner.set_i(1);
    } else if (ss_reduce->op() == "SparseSegmentSum") {
      combiner.set_i(0);
    } else {
      return false;
    }
    const NodeDef* where = get_node(node->input(1));
    CHECK_NODE_OK(where != nullptr && where->op() == "Where" &&
                  where->input_size() == 1)
    const NodeDef* cast = get_user(where, 0, "Cast");
    CHECK_NODE_OK(cast != nullptr)  // output: 1
    const NodeDef* notequal = get_node(where->input(0));
    CHECK_NODE_OK(notequal != nullptr && IsNotEqual(*notequal) &&
                  notequal->input_size() == 2);
    const NodeDef* zerolike = get_node(notequal->input(1));
    CHECK_NODE_OK(zerolike != nullptr && IsZerosLike(*zerolike) &&
                  zerolike->input_size() == 1)
    const NodeDef* cast_1 = get_node(ss_reduce->input(2));
    CHECK_NODE_OK(cast_1 != nullptr && IsCast(*cast_1) &&
                  cast_1->input_size() == 1)
    const NodeDef* strided_slice = get_node(cast_1->input(0));
    CHECK_NODE_OK(strided_slice != nullptr &&
                  IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* shape = get_user(ss_reduce, 0, "Shape");
    CHECK_NODE_OK(shape != nullptr)
    const NodeDef* cast_2 = get_user(shape, 0, "Cast");  // output: 0
    CHECK_NODE_OK(cast_2 != nullptr)

    CHECK_NODE_OK(
        check_const_shape(get_mutable_node(strided_slice->input(1)), {2}))
    CHECK_NODE_OK(check_int_attr(strided_slice, "shrink_axis_mask", 2))
    CHECK_NODE_OK(check_int_attr(strided_slice, "begin_mask", 1))
    CHECK_NODE_OK(check_int_attr(strided_slice, "end_mask", 1))

    CHECK_NODE_OK(check_input_dims(props, ss_reduce, 0, 1) ||
                  check_input_dims(props, ss_reduce, 0, 2))
    CHECK_NODE_OK(check_input_dims(props, strided_slice, 0, 2))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(ss_reduce->input(0));
    fused_node.add_input(ss_reduce->input(1));
    fused_node.add_input(strided_slice->input(0));
    fused_node.add_input(strided_slice->input(1));
    AddNodeAttr("combiner", combiner, &fused_node);
    AddNodeAttr("Tidx", ss_reduce->attr().at("Tidx"), &fused_node);

    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name()
            << "\n";
    replace_all_users_with(cast_2, 0, added, 0, graph);
    replace_all_users_with(cast, 0, added, 1, graph);
    replace_all_users_with(node, 0, added, 2, graph);
    return true;
  }
};

class KPFusedEmbeddingPaddingRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedEmbeddingPadding"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsReshape(*node))
    const NodeDef* user = get_user(node, 0, "ConcatV2");
    CHECK_NODE_OK(user != nullptr)
    const NodeDef* concat = get_node(node->input(0));
    CHECK_NODE_OK(concat != nullptr && IsConcat(*concat) &&
                  concat->input_size() == 3)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(concat->input(2)), {0}))
    const NodeDef* fill = get_node(concat->input(1));
    CHECK_NODE_OK(fill != nullptr && IsFill(*fill) &&
                  fill->input_size() == 2)
    NodeDef* pack = get_operand(fill, "Pack");
    CHECK_NODE_OK(pack != nullptr && IsPack(*pack) && pack->input_size() == 2)
    NodeDef* fill_const = get_operand(fill, "Const");
    // The zero-padding fill value is float-typed in embedding graphs; also
    // accept int to keep the pattern set 2.15 accepted.
    CHECK_NODE_OK(fill_const != nullptr &&
                  (check_const_value<float>(fill_const, {0}) ||
                   check_const_value<int>(fill_const, {0})))
    NodeDef* sub = get_operand(pack, "Sub");
    CHECK_NODE_OK(sub != nullptr && IsSub(*sub) && sub->input_size() == 2)
    const NodeDef* strided_slice = get_node(sub->input(0));
    CHECK_NODE_OK(strided_slice != nullptr &&
                  IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* cast = get_node(strided_slice->input(0));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast))

    CHECK_NODE_OK(check_input_dims(props, cast, 0, 1))
    CHECK_NODE_OK(check_input_dims(props, concat, 0, 2))
    CHECK_NODE_OK(check_input_dims(props, sub, 1, 0))
    CHECK_NODE_OK(check_input_dims(props, node, 1, 1))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(cast->input(0));    // origin_shape
    fused_node.add_input(concat->input(0));  // input
    fused_node.add_input(sub->input(1));     // input_rows
    fused_node.add_input(node->input(1));    // reshape_sizes
    const NodeDef* pack_left = get_node(pack->input(0));
    const NodeDef* pack_right = get_node(pack->input(1));
    CHECK_NODE_OK(pack_left != nullptr && pack_right != nullptr)
    AddNodeAttr("T", pack->attr().at("T"), &fused_node);
    if (IsConstant(*pack_left) || IsHostConstant(*pack_left)) {
      fused_node.add_input(pack->input(0));
    } else if (IsConstant(*pack_right) || IsHostConstant(*pack_right)) {
      fused_node.add_input(pack->input(1));
    } else {
      return false;
    }

    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(sub, 0, added, 0, graph);
    replace_all_users_with(node, 0, added, 1, graph);
    return true;
  }
};

class KPFusedEmbeddingPaddingFastRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedEmbeddingPaddingFast"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsStridedSlice(*node) && node->input_size() == 4)
    CHECK_NODE_OK(check_const_value<int>(get_mutable_node(node->input(1)), {0}))
    CHECK_NODE_OK(check_const_value<int>(get_mutable_node(node->input(2)), {1}))
    CHECK_NODE_OK(check_const_value<int>(get_mutable_node(node->input(3)), {1}))
    CHECK_NODE_OK(check_int_attr(node, "shrink_axis_mask", 1))
    const NodeDef* shape = get_node(node->input(0));
    CHECK_NODE_OK(shape != nullptr && IsShape(*shape) &&
                  shape->input_size() == 1)
    const NodeDef* reshape = get_node(shape->input(0));
    CHECK_NODE_OK(reshape != nullptr && IsReshape(*reshape) &&
                  reshape->input_size() == 2)
    const NodeDef* concat = get_node(reshape->input(0));
    CHECK_NODE_OK(concat != nullptr && IsConcat(*concat) &&
                  concat->input_size() == 3)
    const NodeDef* fill = get_node(concat->input(1));
    CHECK_NODE_OK(fill != nullptr && IsFill(*fill) &&
                  fill->input_size() == 2)
    const NodeDef* pack = get_node(fill->input(0));
    CHECK_NODE_OK(pack != nullptr && IsPack(*pack) &&
                  pack->input_size() == 2)
    const NodeDef* sub = get_node(pack->input(0));
    CHECK_NODE_OK(sub != nullptr && IsSub(*sub) && sub->input_size() == 2)
    const NodeDef* strided_slice = get_node(sub->input(0));
    CHECK_NODE_OK(strided_slice != nullptr &&
                  IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* cast = get_node(strided_slice->input(0));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast))

    CHECK_NODE_OK(check_input_dims(props, cast, 0, 1))
    CHECK_NODE_OK(check_input_dims(props, concat, 0, 2))
    CHECK_NODE_OK(check_input_dims(props, sub, 1, 0))
    CHECK_NODE_OK(check_input_dims(props, reshape, 1, 1))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(cast->input(0));
    fused_node.add_input(concat->input(0));
    fused_node.add_input(sub->input(1));
    fused_node.add_input(reshape->input(1));
    const NodeDef* pack_left = get_node(pack->input(0));
    const NodeDef* pack_right = get_node(pack->input(1));
    CHECK_NODE_OK(pack_left != nullptr && pack_right != nullptr)
    if (IsConstant(*pack_left) || IsHostConstant(*pack_left)) {
      fused_node.add_input(pack->input(0));
    } else if (IsConstant(*pack_right) || IsHostConstant(*pack_right)) {
      fused_node.add_input(pack->input(1));
    } else {
      return false;
    }
    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(sub, 0, added, 0, graph);
    replace_all_users_with(node, 0, added, 1, graph);
    return true;
  }
};

class KPFusedSparseSelectRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseSelect"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsConcat(*node) && node->input_size() == 3)
    const NodeDef* select_0 = get_node(node->input(0));
    CHECK_NODE_OK(select_0 != nullptr && IsSelect(*select_0) &&
                  select_0->input_size() == 3)
    const NodeDef* select_1 = get_node(select_0->input(2));
    CHECK_NODE_OK(select_1 != nullptr && IsSelect(*select_1) &&
                  select_1->input_size() == 3)
    const NodeDef* fill = get_node(select_1->input(1));
    CHECK_NODE_OK(fill != nullptr && IsFill(*fill) &&
                  fill->input_size() == 2)
    CHECK_NODE_OK(
        check_const_value<float>(get_mutable_node(fill->input(1)), {1.0f}))
    const NodeDef* cast = get_node(select_1->input(2));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast) &&
                  cast->input_size() == 1)
    const NodeDef* equal = get_node(select_1->input(0));
    CHECK_NODE_OK(equal != nullptr && IsEqual(*equal) &&
                  equal->input_size() == 2)
    NodeDef* reshape = get_operand(equal, "Reshape");
    CHECK_NODE_OK(reshape != nullptr && IsReshape(*reshape))
    const NodeDef* greater = get_node(cast->input(0));
    CHECK_NODE_OK(greater != nullptr && IsGreater(*greater) &&
                  greater->input_size() == 2)
    const NodeDef* reshape_4 = get_node(greater->input(0));
    CHECK_NODE_OK(reshape_4 != nullptr && IsReshape(*reshape_4) &&
                  reshape_4->input_size() == 2)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(reshape_4->input(1)), {-1, 1}))
    const NodeDef* equal_1 = get_node(select_0->input(0));
    CHECK_NODE_OK(equal_1 != nullptr && IsEqual(*equal_1) &&
                  equal_1->input_size() == 2)
    NodeDef* reshape_1 = get_operand(equal_1, "Reshape");
    CHECK_NODE_OK(reshape_1 != nullptr && IsReshape(*reshape_1))
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(reshape_1->input(1)), {-1, 1}))

    // right branch
    const NodeDef* select_2 = get_node(node->input(1));
    CHECK_NODE_OK(select_2 != nullptr && IsSelect(*select_2) &&
                  select_2->input_size() == 3)
    const NodeDef* equal_2 = get_node(select_2->input(0));
    CHECK_NODE_OK(equal_2 != nullptr && IsEqual(*equal_2) &&
                  equal_2->input_size() == 2)
    const NodeDef* fill_1 = get_node(select_2->input(2));
    CHECK_NODE_OK(fill_1 != nullptr && IsFill(*fill_1) &&
                  fill_1->input_size() == 2)
    CHECK_NODE_OK(
        check_const_value<float>(get_mutable_node(fill_1->input(1)), {1.0f}))
    const NodeDef* reshape_2 = get_operand(equal_2, "Reshape");
    CHECK_NODE_OK(reshape_2 != nullptr && IsReshape(*reshape_2))
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(reshape_2->input(1)), {-1, 1}))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(reshape_4->input(0));
    fused_node.add_input(reshape_1->input(0));
    fused_node.add_input(reshape_2->input(0));
    std::vector<const NodeDef*> const_inputs = {
        greater,
        equal,
        equal_1,
        equal_2,
    };
    for (const NodeDef* const_node : const_inputs) {
      const NodeDef* left = get_node(const_node->input(0));
      const NodeDef* right = get_node(const_node->input(1));
      if (left != nullptr && (IsConstant(*left) || IsHostConstant(*left))) {
        fused_node.add_input(const_node->input(0));
      } else if (right != nullptr &&
                 (IsConstant(*right) || IsHostConstant(*right))) {
        fused_node.add_input(const_node->input(1));
      } else {
        return false;
      }
    }
    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(reshape_4, 0, added, 0, graph);
    replace_all_users_with(select_0, 0, added, 1, graph);
    replace_all_users_with(node, 0, added, 2, graph);
    return true;
  }
};

class KPFusedGatherRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedGather"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "GatherV2" && node->input_size() == 3)
    CHECK_NODE_OK(check_const_value<int>(get_mutable_node(node->input(2)), {0}))
    const NodeDef* gather = get_node(node->input(0));
    CHECK_NODE_OK(gather != nullptr && gather->op() == "GatherV2" &&
                  gather->input_size() == 3)  // input:0
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(gather->input(2)), {0}))
    const NodeDef* unique = get_node(node->input(1));  // output:1
    CHECK_NODE_OK(unique != nullptr && unique->op() == "Unique" &&
                  unique->input_size() == 1)
    const NodeDef* unique_1 = get_node(unique->input(0));
    CHECK_NODE_OK(unique_1 != nullptr && unique_1->op() == "Unique" &&
                  unique_1->input_size() == 1)
    const NodeDef* strided_slice = get_node(unique_1->input(0));
    CHECK_NODE_OK(strided_slice != nullptr &&
                  IsStridedSlice(*strided_slice))  // input:1 2
    CHECK_NODE_OK(check_int_attr(strided_slice, "shrink_axis_mask", 2))
    CHECK_NODE_OK(check_int_attr(strided_slice, "begin_mask", 1))
    CHECK_NODE_OK(check_int_attr(strided_slice, "end_mask", 1))

    CHECK_NODE_OK(check_input_dims(props, gather, 0, 2))
    CHECK_NODE_OK(check_input_dims(props, strided_slice, 0, 2))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(gather->input(0));
    fused_node.add_input(strided_slice->input(0));
    fused_node.add_input(strided_slice->input(1));
    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(unique_1, 0, added, 0, graph);
    replace_all_users_with(unique_1, 1, added, 1, graph);
    replace_all_users_with(node, 0, added, 2, graph);
    return true;
  }
};

class KPFusedSparseReshapeRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseReshape"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "SparseReshape" && node->input_size() == 3)
    const NodeDef* concat = get_node(node->input(0));
    CHECK_NODE_OK(concat != nullptr && IsConcat(*concat) &&
                  concat->input_size() == 3)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(concat->input(2)), {-1}))
    const NodeDef* reshape = get_node(concat->input(1));
    CHECK_NODE_OK(reshape != nullptr && IsReshape(*reshape) &&
                  reshape->input_size() == 2)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(reshape->input(1)), {-1, 1}))
    const NodeDef* strided_slice = get_node(reshape->input(0));
    CHECK_NODE_OK(strided_slice != nullptr &&
                  IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    CHECK_NODE_OK(check_int_attr(strided_slice, "shrink_axis_mask", 2))
    CHECK_NODE_OK(check_int_attr(strided_slice, "begin_mask", 1))
    CHECK_NODE_OK(check_int_attr(strided_slice, "end_mask", 1))
    const NodeDef* cast_1 = get_node(concat->input(0));
    CHECK_NODE_OK(cast_1 != nullptr && IsCast(*cast_1) &&
                  cast_1->input_size() == 1)
    const NodeDef* reshape_1 = get_node(cast_1->input(0));
    CHECK_NODE_OK(reshape_1 != nullptr && IsReshape(*reshape_1) &&
                  reshape_1->input_size() == 2)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(reshape_1->input(1)), {-1, 1}))
    const NodeDef* range = get_node(reshape_1->input(0));
    CHECK_NODE_OK(range != nullptr && range->op() == "Range" &&
                  range->input_size() == 3)
    // Range start=0, delta=1
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(range->input(0)), {0}))
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(range->input(2)), {1}))
    const NodeDef* cast = get_node(node->input(1));
    CHECK_NODE_OK(cast != nullptr && IsCast(*cast) &&
                  cast->input_size() == 1)
    const NodeDef* pack = get_node(cast->input(0));
    CHECK_NODE_OK(pack != nullptr && IsPack(*pack) &&
                  pack->input_size() == 2)
    const NodeDef* strided_slice_1 = get_node(pack->input(0));
    CHECK_NODE_OK(strided_slice_1 != nullptr &&
                  IsStridedSlice(*strided_slice_1) &&
                  strided_slice_1->input_size() == 4)
    CHECK_NODE_OK(check_const_value<int>(
        get_mutable_node(strided_slice_1->input(1)), {0}))
    CHECK_NODE_OK(check_const_value<int>(
        get_mutable_node(strided_slice_1->input(2)), {1}))
    CHECK_NODE_OK(check_const_value<int>(
        get_mutable_node(strided_slice_1->input(3)), {1}))
    CHECK_NODE_OK(check_int_attr(strided_slice_1, "shrink_axis_mask", 1))
    const NodeDef* shape = get_node(strided_slice_1->input(0));
    CHECK_NODE_OK(shape != nullptr && IsShape(*shape) &&
                  shape->input_size() == 1)

    CHECK_NODE_OK(check_input_dims(props, shape, 0, 2))
    CHECK_NODE_OK(check_input_dims(props, pack, 1, 0))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(shape->input(0));          // slice_input
    fused_node.add_input(strided_slice->input(1));  // begin
    fused_node.add_input(node->input(2));           // new_shape
    fused_node.add_input(pack->input(1));           // pack_const
    AddNodeAttr("T", pack->attr().at("T"), &fused_node);
    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(node, 0, added, 0, graph);
    replace_all_users_with(node, 1, added, 1, graph);
    return true;
  }
};

class KPFusedEmbeddingActionIdGatherRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedEmbeddingActionIdGather"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      const GraphProperties& props,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsConcat(*node) && node->input_size() == 3)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(node->input(2)), {-1}))
    const NodeDef* reshape = get_node(node->input(0));
    CHECK_NODE_OK(reshape != nullptr && IsReshape(*reshape) &&
                  reshape->input_size() == 2)
    const NodeDef* gather = get_node(reshape->input(0));
    const NodeDef* pack_1 = get_node(reshape->input(1));
    CHECK_NODE_OK(gather != nullptr)
    CHECK_NODE_OK(pack_1 != nullptr && IsPack(*pack_1) &&
                  pack_1->input_size() == 2)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(pack_1->input(1)), {-1}))
    CHECK_NODE_OK(gather->op() == "GatherV2" && gather->input_size() == 3)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(gather->input(2)), {0}))
    const NodeDef* gather_1 = get_node(gather->input(0));
    CHECK_NODE_OK(gather_1 != nullptr && gather_1->op() == "GatherV2" &&
                  gather_1->input_size() == 3)
    CHECK_NODE_OK(
        check_const_value<int>(get_mutable_node(gather_1->input(2)), {0}))
    const NodeDef* fill = get_node(node->input(1));
    CHECK_NODE_OK(fill != nullptr && IsFill(*fill) &&
                  fill->input_size() == 2)
    // The zero-padding fill value is float-typed in embedding graphs; also
    // accept int to keep the pattern set 2.15 accepted.
    CHECK_NODE_OK(check_const_value<float>(get_mutable_node(fill->input(1)), {0}) ||
                  check_const_value<int>(get_mutable_node(fill->input(1)), {0}))
    const NodeDef* pack = get_node(fill->input(0));
    CHECK_NODE_OK(pack != nullptr && IsPack(*pack) &&
                  pack->input_size() == 2)

    CHECK_NODE_OK(check_input_dims(props, gather_1, 1, 1) ||
                  check_input_dims(props, gather_1, 1, 2))
    CHECK_NODE_OK(check_input_dims(props, gather, 1, 1 ) ||
                  check_input_dims(props, gather, 1, 2))
    CHECK_NODE_OK(check_input_dims(props, gather_1, 0, 2) ||
                  check_input_dims(props, gather_1, 0, 3))
    CHECK_NODE_OK(check_input_dims(props, pack, 0, 0))
    CHECK_NODE_OK(check_input_dims(props, pack, 1, 0))

    auto nodes = graph->mutable_node();
    NodeDef fused_node;  // built locally; committed only if all checks pass
    fused_node.set_name(node->name() + fusion_appendix);
    fused_node.set_op(name());
    fused_node.set_device(node->device());
    fused_node.add_input(gather_1->input(1));  // indices1
    fused_node.add_input(gather_1->input(0));  // params
    fused_node.add_input(gather->input(1));    // indices2
    fused_node.add_input(pack->input(0));      // pack_dim
    fused_node.add_input(pack->input(1));      // pack
    AddNodeAttr("Tindices1", gather_1->attr().at("Tindices"), &fused_node);
    AddNodeAttr("Tindices2", gather->attr().at("Tindices"), &fused_node);
    // All checks passed: commit the fused node to the graph in one step so a
    // failed match never leaves a partially built node behind.
    NodeDef* added = nodes->Add();
    *added = std::move(fused_node);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << added->op() << "] " << added->name();
    replace_all_users_with(node, 0, added, 0, graph);
    return true;
  }
};

static bool IsEnvEnabled(const char* name) {
  const char* value = getenv(name);
  return value != nullptr && strcmp(value, "1") == 0;
}

bool ANNCAnyFusionEnabled() {
  // Keep this list in sync with the env vars consumed by
  // run_graph_optimization below. Deliberately not cached: callers may
  // change the env between sessions.
  static const char* const kEnvNames[] = {
      "ANNC_FUSED_ALL",
      "ANNC_FUSED_SPS_STITCH",
      "ANNC_FUSED_SPS_STITCH_REDUCE",
      "ANNC_FUSED_SPS_REDUCE",
      "ANNC_FUSED_EMD_PADDING",
      "ANNC_FUSED_EMD_PADDING_FAST",
      "ANNC_FUSED_SPS_SELECT",
      "ANNC_FUSED_GATHER",
      "ANNC_FUSED_SPS_RESHAPE",
      "ANNC_FUSED_EMB_ACTIONID_GATHER",
      "ANNC_FUSED_SPS_REDUCE_NONZERO"};
  for (const char* env_name : kEnvNames) {
    if (IsEnvEnabled(env_name)) return true;
  }
  return false;
}

bool RuntimeSveSupported() {
#if defined(__aarch64__)
  // The KPFusedSparseDynamicStitch*Reduce kernels are compiled with +sve and
  // would SIGILL on a CPU without SVE, so probe the auxiliary vector once
  // per process before letting the rewriter insert them into a graph.
  static const bool supported = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
  return supported;
#else
  return false;
#endif
}

void run_graph_optimization(GraphDef* graph, GraphProperties props) {
  GraphOptimizer optimizer(graph, props);

  const char* annc_fused_all = getenv("ANNC_FUSED_ALL");
  const char* annc_fused_sps_stitch = getenv("ANNC_FUSED_SPS_STITCH");
  const char* annc_fused_sps_stitch_reduce = getenv("ANNC_FUSED_SPS_STITCH_REDUCE");
  const char* annc_fused_sps_reduce = getenv("ANNC_FUSED_SPS_REDUCE");
  const char* annc_fused_emb_padding = getenv("ANNC_FUSED_EMD_PADDING");
  const char* annc_fused_emb_padding_fast =
      getenv("ANNC_FUSED_EMD_PADDING_FAST");
  const char* annc_fused_sps_select = getenv("ANNC_FUSED_SPS_SELECT");
  const char* annc_fused_gather = getenv("ANNC_FUSED_GATHER");
  const char* annc_fused_sps_reshape = getenv("ANNC_FUSED_SPS_RESHAPE");
  const char* annc_fused_emb_actionid_gather =
      getenv("ANNC_FUSED_EMB_ACTIONID_GATHER");
  const char* annc_fused_sps_reduce_nonzero =
      getenv("ANNC_FUSED_SPS_REDUCE_NONZERO");

  bool enable_all =
      (annc_fused_all != nullptr) && strcmp(annc_fused_all, "1") == 0;

  if (RuntimeSveSupported() &&
      (enable_all || (annc_fused_sps_stitch_reduce != nullptr &&
                      strcmp(annc_fused_sps_stitch_reduce, "1") == 0)))
    optimizer.register_rewriter(
        std::make_unique<KPFusedSparseDynamicStitchReduceRewriter>());
  if (enable_all || (annc_fused_sps_stitch != nullptr &&
                     strcmp(annc_fused_sps_stitch, "1") == 0))
    optimizer.register_rewriter(
        std::make_unique<KPFusedSparseDynamicStitchRewriter>());
  if (enable_all || (annc_fused_sps_reduce != nullptr &&
                     strcmp(annc_fused_sps_reduce, "1") == 0))
    optimizer.register_rewriter(
        std::make_unique<KPFusedSparseSegmentReduceRewriter>());
  if (enable_all || (annc_fused_emb_padding_fast != nullptr &&
                     strcmp(annc_fused_emb_padding_fast, "1") == 0))
    optimizer.register_rewriter(
        std::make_unique<KPFusedEmbeddingPaddingFastRewriter>());
  if (enable_all || (annc_fused_emb_padding != nullptr &&
                     strcmp(annc_fused_emb_padding, "1") == 0))
    optimizer.register_rewriter(
        std::make_unique<KPFusedEmbeddingPaddingRewriter>());
  if (enable_all || (annc_fused_sps_select != nullptr &&
                     strcmp(annc_fused_sps_select, "1") == 0))
    optimizer.register_rewriter(
        std::make_unique<KPFusedSparseSelectRewriter>());
  if (enable_all ||
      (annc_fused_gather != nullptr && strcmp(annc_fused_gather, "1") == 0))
    optimizer.register_rewriter(std::make_unique<KPFusedGatherRewriter>());
  if (enable_all || (annc_fused_sps_reshape != nullptr &&
                     strcmp(annc_fused_sps_reshape, "1") == 0))
    optimizer.register_rewriter(
        std::make_unique<KPFusedSparseReshapeRewriter>());
  if (enable_all || annc_fused_emb_actionid_gather != nullptr &&
      strcmp(annc_fused_emb_actionid_gather, "1") == 0)
    optimizer.register_rewriter(
        std::make_unique<KPFusedEmbeddingActionIdGatherRewriter>());
  if (enable_all || annc_fused_sps_reduce_nonzero != nullptr &&
      strcmp(annc_fused_sps_reduce_nonzero, "1") == 0)
    optimizer.register_rewriter(
        std::make_unique<KPFusedSparseSegmentReduceNonzeroRewriter>());

  // Nothing was enabled: skip the node-index rebuild and the rewrite loop.
  if (!optimizer.has_rewriters()) return;
  optimizer.optimize();
}
}  // namespace annc
