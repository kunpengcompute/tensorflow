/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

// Unit tests for the ANNC embedding-fusion graph optimizer (graph_opt.cc).
//
// Each test builds the *unfused* graph that the corresponding PatternRewriter
// is designed to recognize, runs annc::run_graph_optimization with only that
// rewriter enabled, and asserts that the fused node was produced.

#include "tensorflow/core/grappler/optimizers/graph_optimizer/graph_opt.h"

#include "absl/log/internal/vlog_config.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::ops::Cast;
using ::tensorflow::ops::Concat;
using ::tensorflow::ops::Const;
using ::tensorflow::ops::DynamicPartition;
using ::tensorflow::ops::Equal;
using ::tensorflow::ops::Fill;
using ::tensorflow::ops::FloorDiv;
using ::tensorflow::ops::FloorMod;
using ::tensorflow::ops::GatherNd;
using ::tensorflow::ops::GatherV2;
using ::tensorflow::ops::Greater;
using ::tensorflow::ops::NotEqual;
using ::tensorflow::ops::ParallelDynamicStitch;
using ::tensorflow::ops::Placeholder;
using ::tensorflow::ops::Range;
using ::tensorflow::ops::RealDiv;
using ::tensorflow::ops::Reshape;
using ::tensorflow::ops::SelectV2;
using ::tensorflow::ops::Shape;
using ::tensorflow::ops::Size;
using ::tensorflow::ops::SparseSegmentSum;
using ::tensorflow::ops::Stack;
using ::tensorflow::ops::StridedSlice;
using ::tensorflow::ops::Sub;
using ::tensorflow::ops::Unique;
using ::tensorflow::ops::Where;
using ::tensorflow::ops::ZerosLike;

class GraphOptTest : public GrapplerTest {
 protected:
  void SetUp() override {
    // Enable the VLOG(1) diagnostics in graph_opt.cc so the tests also cover
    // those logging statements. VLOG here resolves to the absl implementation
    // (an absl/log header is pulled in after tsl's logging.h and redefines
    // the macro), so the TF_CPP_VMODULE/TF_CPP_MAX_VLOG_LEVEL environment
    // variables used by tsl's logging are ignored. The absl config is updated
    // at runtime instead: PrependVModule registers "graph_opt=1" and refreshes
    // every already-registered VLogSite.
    absl::log_internal::PrependVModule("graph_opt", 1);

    // Disable every rewriter by default; each test opts in to exactly one so
    // that a single graph cannot be rewritten by two different rewriters.
    unsetenv("ANNC_FUSED_ALL");
    unsetenv("ANNC_FUSED_SPS_STITCH");
    unsetenv("ANNC_FUSED_SPS_STITCH_REDUCE");
    unsetenv("ANNC_FUSED_SPS_REDUCE");
    unsetenv("ANNC_FUSED_SPS_REDUCE_NONZERO");
    unsetenv("ANNC_FUSED_EMD_PADDING");
    unsetenv("ANNC_FUSED_EMD_PADDING_FAST");
    unsetenv("ANNC_FUSED_SPS_SELECT");
    unsetenv("ANNC_FUSED_GATHER");
    unsetenv("ANNC_FUSED_SPS_RESHAPE");
    unsetenv("ANNC_FUSED_EMB_ACTIONID_GATHER");
  }

  // Populate static shape information (required by check_input_dims) and run
  // the graph optimizer.
  Status RunGraphOpt(GraphDef* graph) {
    GrapplerItem item;
    item.graph = *graph;
    GraphProperties props(item);
    TF_RETURN_IF_ERROR(props.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/false,
        /*include_input_tensor_values=*/true,
        /*include_output_tensor_values=*/true));
    annc::run_graph_optimization(graph, props);
    return OkStatus();
  }

  int CountOp(const GraphDef& graph, const string& op) {
    return CountOpNodes(graph, op);
  }

  // Pieces of the canonical Stitch+Reduce pattern (see
  // KPFusedSparseDynamicStitchReduce) that the no-fuse tests below vary.
  // gather1_axis selects the axis const of the second table's GatherV2
  // (0 = canonical, non-zero makes the right branch fail to match).
  struct StitchReduceGraph {
    Output stitched;
    Output unique_values;   // Unique:0
    Output unique_indices;  // Unique:1
    Output seg_ids;
  };

  struct StitchReduceMutation {
    bool swap_gather_ports = false;
    bool swap_stitch_ports = false;
    bool split_table_partition = false;
    bool mod_from_x = false;
    bool div_from_x = false;
  };

  StitchReduceGraph BuildStitchReduceGraph(const Scope& s, int gather1_axis) {
    return BuildStitchReduceGraph(s, gather1_axis, StitchReduceMutation());
  }

  StitchReduceGraph BuildStitchReduceGraph(
      const Scope& s, int gather1_axis, const StitchReduceMutation& m) {
    const int64_t kNumTables = 2;  // power of two -> "Fast" fused op
    Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));
    auto unique = Unique(s.WithOpName("unique"), x);
    Output num = Const<int64>(s.WithOpName("num"), kNumTables);
    Output mod = FloorMod(s.WithOpName("mod"), m.mod_from_x ? x : unique.y,
                          num);
    Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
    Output div = FloorDiv(s.WithOpName("div"), m.div_from_x ? x : unique.y,
                          num);
    Output size = Size(s.WithOpName("size"), unique.y);
    Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
    auto stitch_part = DynamicPartition(s.WithOpName("stitch_part"), range,
                                        group_ids, kNumTables);
    auto table_part = DynamicPartition(s.WithOpName("table_part"), div,
                                       group_ids, kNumTables);
    Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                                Placeholder::Shape({10, 4}));
    Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                                Placeholder::Shape({10, 4}));
    Output axis = Const(s.WithOpName("axis"), 0);
    Output axis1 = Const(s.WithOpName("axis1"), gather1_axis);
    Output g0 = GatherV2(s.WithOpName("gather0"), table0,
                         table_part.outputs[m.swap_gather_ports ? 1 : 0], axis);
    Output g1;
    if (m.split_table_partition) {
      auto table_part2 = DynamicPartition(s.WithOpName("table_part2"), div,
                                          group_ids, kNumTables);
      g1 = GatherV2(s.WithOpName("gather1"), table1, table_part2.outputs[1],
                    axis1);
    } else {
      g1 = GatherV2(s.WithOpName("gather1"), table1,
                    table_part.outputs[m.swap_gather_ports ? 0 : 1], axis1);
    }
    Output stitched =
        ParallelDynamicStitch(s.WithOpName("stitch"),
                              m.swap_stitch_ports
                                  ? OutputList{stitch_part.outputs[1],
                                               stitch_part.outputs[0]}
                                  : stitch_part.outputs,
                              {g0, g1})
            .merged;

    Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                     Placeholder::Shape({6, 2}));
    Output begin = Const(s.WithOpName("begin"), {0, 1});
    Output end = Const(s.WithOpName("end"), {0, 2});
    Output strides = Const(s.WithOpName("strides"), {1, 1});
    Output slice = StridedSlice(
        s.WithOpName("slice"), slice_input, begin, end, strides,
        StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
    Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
    return {stitched, unique.y, unique.idx, seg_ids};
  }
};

// GatherV2(GatherV2(data, Unique(Unique(StridedSlice))), ...) -> KPFusedGather
TEST_F(GraphOptTest, KPFusedGather) {
  setenv("ANNC_FUSED_GATHER", "1", 1);

  Scope s = Scope::NewRootScope();
  Output data = Placeholder(s.WithOpName("data"), DT_FLOAT,
                            Placeholder::Shape({50, 12}));
  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({3, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {3, 3});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  auto unique1 = Unique(s.WithOpName("unique1"), slice);
  auto unique2 = Unique(s.WithOpName("unique2"), unique1.y);
  Output axis = Const(s.WithOpName("axis"), 0);
  Output gather1 = GatherV2(s.WithOpName("gather1"), data, unique2.y, axis);
  Output gather2 = GatherV2(s.WithOpName("gather2"), gather1, unique2.idx, axis);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedGather"), 1);
}

// Shape(SparseSegmentSum) + StridedSlice -> KPFusedSparseSegmentReduce
TEST_F(GraphOptTest, KPFusedSparseSegmentReduce) {
  setenv("ANNC_FUSED_SPS_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  Output data = Placeholder(s.WithOpName("data"), DT_FLOAT,
                            Placeholder::Shape({4, 3}));
  Output indices = Placeholder(s.WithOpName("indices"), DT_INT32,
                               Placeholder::Shape({3}));
  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({3, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Placeholder(s.WithOpName("strides"), DT_INT32,
                               Placeholder::Shape({2}));
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
  Output output = SparseSegmentSum(s.WithOpName("ss"), data, indices, seg_ids);
  Output out_shape = Shape(s.WithOpName("out_shape"), output);
  Output begin1 = Const(s.WithOpName("begin1"), {0});
  Output end1 = Const(s.WithOpName("end1"), {1});
  Output strides1 = Const(s.WithOpName("strides1"), {1});
  Output slice1 = StridedSlice(
      s.WithOpName("slice1"), out_shape, begin1, end1, strides1,
      StridedSlice::Attrs().ShrinkAxisMask(1));

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedSparseSegmentReduce"), 1);
}

// ParallelDynamicStitch(DynamicPartition(Range(Size)), GatherV2 branches)
// -> KPFusedSparseDynamicStitch
TEST_F(GraphOptTest, KPFusedSparseDynamicStitch) {
  setenv("ANNC_FUSED_SPS_STITCH", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({4}));
  Output x1 = Reshape(s.WithOpName("x1"), x, {-1});
  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), x1, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output size = Size(s.WithOpName("size"), x1);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto orig_part = DynamicPartition(s.WithOpName("orig_part"), range, group_ids,
                                    kNumTables);

  Output div = FloorDiv(s.WithOpName("div"), x1, num);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);

  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitch = ParallelDynamicStitch(s.WithOpName("stitch"),
                                        orig_part.outputs, {g0, g1})
                      .merged;

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitch"), 1);
}

// SparseSegmentSum(ParallelDynamicStitch(...)) -> KPFusedSparseDynamicStitchSum
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduce) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);
  const int64_t kNumTables = 2;  // power of two -> "Fast" fused op

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));
  auto unique = Unique(s.WithOpName("unique"), x);
  Output unique_values = unique.y;
  Output unique_indices = unique.idx;

  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), unique_values, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output div = FloorDiv(s.WithOpName("div"), unique_values, num);
  Output size = Size(s.WithOpName("size"), unique_values);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto stitch_part = DynamicPartition(s.WithOpName("stitch_part"), range,
                                      group_ids, kNumTables);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);

  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitched = ParallelDynamicStitch(s.WithOpName("stitch"),
                                          stitch_part.outputs, {g0, g1})
                        .merged;

  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({6, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), stitched,
                                    unique_indices, seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  // The Stitch+Reduce fused kernels are compiled with +sve, so on a CPU
  // without SVE the rewriter must not be registered and nothing may fuse.
  // 2 tables is a power of two, so the "Fast" sum variant is selected.
  if (annc::RuntimeSveSupported()) {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 1);
  } else {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
  }
}

// Right branch of the plain Stitch pattern has a non-zero GatherV2 axis: the
// match fails after the fused-node construction has started, and the graph
// must be left completely untouched (no partially built "/kp_fused" node
// missing its attributes).
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchPartialMatchNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({4}));
  Output x1 = Reshape(s.WithOpName("x1"), x, {-1});
  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), x1, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output size = Size(s.WithOpName("size"), x1);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto orig_part = DynamicPartition(s.WithOpName("orig_part"), range, group_ids,
                                    kNumTables);
  Output div = FloorDiv(s.WithOpName("div"), x1, num);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);
  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output bad_axis = Const(s.WithOpName("bad_axis"), 1);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       bad_axis);
  Output stitch = ParallelDynamicStitch(s.WithOpName("stitch"),
                                        orig_part.outputs, {g0, g1})
                      .merged;

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitch"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchSwappedGatherPortsNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({4}));
  Output x1 = Reshape(s.WithOpName("x1"), x, {-1});
  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), x1, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output size = Size(s.WithOpName("size"), x1);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto orig_part = DynamicPartition(s.WithOpName("orig_part"), range, group_ids,
                                    kNumTables);
  Output div = FloorDiv(s.WithOpName("div"), x1, num);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);
  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[1],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[0],
                       axis);
  Output stitch = ParallelDynamicStitch(s.WithOpName("stitch"),
                                        orig_part.outputs, {g0, g1})
                      .merged;

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitch"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchModSourceNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({4}));
  Output x1 = Reshape(s.WithOpName("x1"), x, {-1});
  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), x, num);  // x, not x1
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output size = Size(s.WithOpName("size"), x1);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto orig_part = DynamicPartition(s.WithOpName("orig_part"), range, group_ids,
                                    kNumTables);
  Output div = FloorDiv(s.WithOpName("div"), x1, num);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);
  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitch = ParallelDynamicStitch(s.WithOpName("stitch"),
                                        orig_part.outputs, {g0, g1})
                      .merged;

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitch"), 0);
}

// Right branch of the Stitch+Reduce pattern has a non-zero GatherV2 axis:
// same requirement — the graph must be left completely untouched.
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReducePartialMatchNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceGraph pattern = BuildStitchReduceGraph(s, /*gather1_axis=*/1);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_indices, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

// The fused op re-derives the pooling order from the Unique node, so a
// SparseSegmentSum whose indices do not come from that Unique's idx output
// (here: an independent constant) must not fuse — pooling over a re-ordered
// index tensor would silently change the result.
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceIndependentIndicesNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceGraph pattern = BuildStitchReduceGraph(s, /*gather1_axis=*/0);
  Output indices_const =
      Const<int64>(s.WithOpName("indices_const"), {0, 1, 2, 3, 4, 5});
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    indices_const, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

// Indices wired to the Unique values output (port 0) instead of the idx
// output (port 1): identical shape, different semantics — must not fuse.
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceWrongPortNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceGraph pattern = BuildStitchReduceGraph(s, /*gather1_axis=*/0);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_values, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceSwappedGatherPortsNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceMutation m;
  m.swap_gather_ports = true;
  StitchReduceGraph pattern =
      BuildStitchReduceGraph(s, /*gather1_axis=*/0, m);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_indices, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceSwappedStitchPortsNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceMutation m;
  m.swap_stitch_ports = true;
  StitchReduceGraph pattern =
      BuildStitchReduceGraph(s, /*gather1_axis=*/0, m);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_indices, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceSplitPartitionNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceMutation m;
  m.split_table_partition = true;
  StitchReduceGraph pattern =
      BuildStitchReduceGraph(s, /*gather1_axis=*/0, m);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_indices, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceModSourceNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceMutation m;
  m.mod_from_x = true;
  StitchReduceGraph pattern =
      BuildStitchReduceGraph(s, /*gather1_axis=*/0, m);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_indices, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceDivSourceNoFuse) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);

  Scope s = Scope::NewRootScope();
  StitchReduceMutation m;
  m.div_from_x = true;
  StitchReduceGraph pattern =
      BuildStitchReduceGraph(s, /*gather1_axis=*/0, m);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), pattern.stitched,
                                    pattern.unique_indices, pattern.seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
}

// Cast(StridedSlice) -> SparseSegment* -> Where(NotEqual(ZerosLike)) ->
// KPFusedSparseSegmentReduceNonzero
TEST_F(GraphOptTest, KPFusedSparseSegmentReduceNonzero) {
  setenv("ANNC_FUSED_SPS_REDUCE_NONZERO", "1", 1);

  Scope s = Scope::NewRootScope();
  Output data = Placeholder(s.WithOpName("data"), DT_FLOAT,
                            Placeholder::Shape({100, 5}));
  Output indices = Placeholder(s.WithOpName("indices"), DT_INT32,
                               Placeholder::Shape({50}));
  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({50, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
  Output ss = SparseSegmentSum(s.WithOpName("ss"), data, indices, seg_ids);
  Output zero = ZerosLike(s.WithOpName("zero"), ss);
  Output notequal = NotEqual(s.WithOpName("notequal"), ss, zero);
  Output where = Where(s.WithOpName("where"), notequal);
  Output cast_where = Cast(s.WithOpName("cast_where"), where, DT_INT32);
  Output gather = GatherNd(s.WithOpName("gather"), ss, where);
  Output ss_shape = Shape(s.WithOpName("ss_shape"), ss);
  Output cast_shape = Cast(s.WithOpName("cast_shape"), ss_shape, DT_INT32);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedSparseSegmentReduceNonzero"), 1);
}

// Cast -> StridedSlice -> Sub -> Pack/Fill -> Concat -> Reshape ->
// KPFusedEmbeddingPadding
TEST_F(GraphOptTest, KPFusedEmbeddingPadding) {
  setenv("ANNC_FUSED_EMD_PADDING", "1", 1);

  Scope s = Scope::NewRootScope();
  Output shape_input = Placeholder(s.WithOpName("shape_input"), DT_INT64,
                                   Placeholder::Shape({2}));
  Output data = Placeholder(s.WithOpName("data"), DT_FLOAT,
                            Placeholder::Shape({150, 10}));
  Output input_rows = Placeholder(s.WithOpName("input_rows"), DT_INT32);
  Output reshape_sizes = Placeholder(s.WithOpName("reshape_sizes"), DT_INT32,
                                     Placeholder::Shape({2}));

  Output cast = Cast(s.WithOpName("cast"), shape_input, DT_INT32);
  Output begin = Const(s.WithOpName("begin"), {0});
  Output end = Const(s.WithOpName("end"), {1});
  Output strides = Const(s.WithOpName("strides"), {1});
  Output slice = StridedSlice(s.WithOpName("slice"), cast, begin, end, strides,
                              StridedSlice::Attrs().ShrinkAxisMask(1));
  Output sub = Sub(s.WithOpName("sub"), slice, input_rows);
  Output pack_right = Const(s.WithOpName("pack_right"), 10);
  Output pack = Stack(s.WithOpName("pack"), {sub, pack_right});
  Output fill_value = Const(s.WithOpName("fill_value"), 0.0f);
  Output fill = Fill(s.WithOpName("fill"), pack, fill_value);
  Output axis0 = Const(s.WithOpName("axis0"), 0);
  Output concat = Concat(s.WithOpName("concat"), {data, fill}, axis0);
  Output reshape = Reshape(s.WithOpName("reshape"), concat, reshape_sizes);
  Output axis1 = Const(s.WithOpName("axis1"), 1);
  Output concat2 = Concat(s.WithOpName("concat2"), {reshape, reshape}, axis1);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedEmbeddingPadding"), 1);
}

// KPFusedEmbeddingPadding + extra StridedSlice -> KPFusedEmbeddingPaddingFast
TEST_F(GraphOptTest, KPFusedEmbeddingPaddingFast) {
  setenv("ANNC_FUSED_EMD_PADDING_FAST", "1", 1);

  Scope s = Scope::NewRootScope();
  Output shape_input = Placeholder(s.WithOpName("shape_input"), DT_INT64,
                                   Placeholder::Shape({2}));
  Output data = Placeholder(s.WithOpName("data"), DT_FLOAT,
                            Placeholder::Shape({150, 10}));
  Output input_rows = Placeholder(s.WithOpName("input_rows"), DT_INT32);
  Output reshape_sizes = Placeholder(s.WithOpName("reshape_sizes"), DT_INT32,
                                     Placeholder::Shape({2}));

  Output cast = Cast(s.WithOpName("cast"), shape_input, DT_INT32);
  Output begin = Const(s.WithOpName("begin"), {0});
  Output end = Const(s.WithOpName("end"), {1});
  Output strides = Const(s.WithOpName("strides"), {1});
  Output slice = StridedSlice(s.WithOpName("slice"), cast, begin, end, strides,
                              StridedSlice::Attrs().ShrinkAxisMask(1));
  Output sub = Sub(s.WithOpName("sub"), slice, input_rows);
  Output pack_right = Const(s.WithOpName("pack_right"), 10);
  Output pack = Stack(s.WithOpName("pack"), {sub, pack_right});
  Output fill_value = Const(s.WithOpName("fill_value"), 0.0f);
  Output fill = Fill(s.WithOpName("fill"), pack, fill_value);
  Output axis0 = Const(s.WithOpName("axis0"), 0);
  Output concat = Concat(s.WithOpName("concat"), {data, fill}, axis0);
  Output reshape = Reshape(s.WithOpName("reshape"), concat, reshape_sizes);
  Output reshape_shape = Shape(s.WithOpName("reshape_shape"), reshape);
  Output slice_begin = Const(s.WithOpName("slice_begin"), {0});
  Output slice_end = Const(s.WithOpName("slice_end"), {1});
  Output slice_strides = Const(s.WithOpName("slice_strides"), {1});
  Output slice2 = StridedSlice(s.WithOpName("slice2"), reshape_shape,
                               slice_begin, slice_end, slice_strides,
                               StridedSlice::Attrs().ShrinkAxisMask(1));

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedEmbeddingPaddingFast"), 1);
}

// Reshape + Greater/Equal + Select pattern -> KPFusedSparseSelect
TEST_F(GraphOptTest, KPFusedSparseSelect) {
  setenv("ANNC_FUSED_SPS_SELECT", "1", 1);

  Scope s = Scope::NewRootScope();
  Output input_a = Placeholder(s.WithOpName("input_a"), DT_INT32,
                               Placeholder::Shape({10, 10}));
  Output input_b = Placeholder(s.WithOpName("input_b"), DT_INT32,
                               Placeholder::Shape({10, 10}));
  Output input_c = Placeholder(s.WithOpName("input_c"), DT_INT32,
                               Placeholder::Shape({10, 10}));

  // Reshape inputs
  Output reshape_a = Reshape(s.WithOpName("reshape_a"), input_a, {-1, 1});
  Output reshape_b = Reshape(s.WithOpName("reshape_b"), input_b, {-1, 1});
  Output reshape_c = Reshape(s.WithOpName("reshape_c"), input_c, {-1, 1});

  // First branch: greater -> cast -> fill -> select chain
  Output greater_value = Const(s.WithOpName("greater_value"), 0);
  Output greater_a = Greater(s.WithOpName("greater_a"), reshape_a, greater_value);
  Output cast_a = Cast(s.WithOpName("cast_a"), greater_a, DT_FLOAT);

  Output shape_a = Shape(s.WithOpName("shape_a"), cast_a);
  Output fill_value = Const(s.WithOpName("fill_value"), 1.0f);
  Output fill_a = Fill(s.WithOpName("fill_a"), shape_a, fill_value);

  // Equal checks for b
  Output equal1_value = Const(s.WithOpName("equal1_value"), 4563);
  Output equal2_value = Const(s.WithOpName("equal2_value"), 10831);
  Output equal_1 = Equal(s.WithOpName("equal_1"), reshape_b, equal1_value);
  Output equal_2 = Equal(s.WithOpName("equal_2"), reshape_b, equal2_value);

  // Select cascade for first output
  Output select_1 = SelectV2(s.WithOpName("select_1"), equal_2, fill_a, cast_a);
  Output select_0 = SelectV2(s.WithOpName("select_0"), equal_1, fill_a, select_1);

  // Second branch: realdiv for c
  Output shape_reshape_a1 = Shape(s.WithOpName("shape_reshape_a1"), reshape_a);
  Output fill_div = Fill(s.WithOpName("fill_div"), shape_reshape_a1, fill_value);
  Output div_const = Const(s.WithOpName("div_const"), 1.0f);
  Output realdiv = RealDiv(s.WithOpName("realdiv"), fill_div, div_const);

  Output equal3_value = Const(s.WithOpName("equal3_value"), 3);
  Output equal_3 = Equal(s.WithOpName("equal_3"), reshape_c, equal3_value);
  Output select_2 = SelectV2(s.WithOpName("select_2"), equal_3, realdiv, fill_div);

  // Final concat
  Output axis_neg1 = Const(s.WithOpName("axis_neg1"), -1);
  Output concat = Concat(s.WithOpName("concat"), {select_0, select_2}, axis_neg1);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedSparseSelect"), 1);
}

// StridedSlice + Reshape + Shape + StridedSlice + Stack + Cast + Range +
// Reshape + Cast + Concat + SparseReshape -> KPFusedSparseReshape
TEST_F(GraphOptTest, KPFusedSparseReshape) {
  setenv("ANNC_FUSED_SPS_RESHAPE", "1", 1);

  Scope s = Scope::NewRootScope();
  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({4, 2}));
  Output newshape = Placeholder(s.WithOpName("newshape"), DT_INT64,
                                Placeholder::Shape({2}));
  Output pack_const = Const(s.WithOpName("pack_const"), 2);

  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output reshape = Reshape(s.WithOpName("reshape"), slice, {-1, 1});
  Output shape = Shape(s.WithOpName("shape"), slice_input);
  Output shape_begin = Const(s.WithOpName("shape_begin"), {0});
  Output shape_end = Const(s.WithOpName("shape_end"), {1});
  Output shape_strides = Const(s.WithOpName("shape_strides"), {1});
  Output shape_slice = StridedSlice(
      s.WithOpName("shape_slice"), shape, shape_begin, shape_end,
      shape_strides, StridedSlice::Attrs().ShrinkAxisMask(1));
  Output stack = Stack(s.WithOpName("stack"), {shape_slice, pack_const});
  Output input_shape = Cast(s.WithOpName("input_shape"), stack, DT_INT64);
  Output range_start = Const(s.WithOpName("range_start"), 0);
  Output range_delta = Const(s.WithOpName("range_delta"), 1);
  Output range = Range(s.WithOpName("range"), range_start, shape_slice,
                       range_delta);
  Output range_reshape = Reshape(s.WithOpName("range_reshape"), range, {-1, 1});
  Output range_cast = Cast(s.WithOpName("range_cast"), range_reshape, DT_INT64);
  Output axis_neg1_2 = Const(s.WithOpName("axis_neg1_2"), -1);
  Output concat = Concat(s.WithOpName("concat"), {range_cast, reshape}, axis_neg1_2);
  // SparseReshape expects 3 inputs: indices, input_shape, new_shape
  // We're building a minimal pattern that just uses concat as indices
  // For a real sparse tensor we'd need values too, but the matcher only
  // checks structure
  auto sparse_out = ops::SparseReshape(s.WithOpName("sparse_reshape"), concat,
                                       input_shape, newshape);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedSparseReshape"), 1);
}

// GatherV2(GatherV2) + Pack + Reshape + Fill + Concat ->
// KPFusedEmbeddingActionIdGather
TEST_F(GraphOptTest, KPFusedEmbeddingActionIdGather) {
  setenv("ANNC_FUSED_EMB_ACTIONID_GATHER", "1", 1);

  Scope s = Scope::NewRootScope();
  Output params = Placeholder(s.WithOpName("params"), DT_FLOAT,
                              Placeholder::Shape({80, 300}));
  Output indices1 = Placeholder(s.WithOpName("indices1"), DT_INT32,
                                Placeholder::Shape({8, 10}));
  Output indices2 = Placeholder(s.WithOpName("indices2"), DT_INT32,
                                Placeholder::Shape({5, 6}));
  Output rows = Const(s.WithOpName("rows"), 80);
  Output pack_value = Const(s.WithOpName("pack_value"), 1680);

  Output axis = Const(s.WithOpName("axis"), 0);
  Output gather1 = GatherV2(s.WithOpName("gather1"), params, indices1, axis);
  Output gather2 = GatherV2(s.WithOpName("gather2"), gather1, indices2, axis);
  Output neg_one = Const(s.WithOpName("neg_one"), -1);
  Output pack1 = Stack(s.WithOpName("pack1"), {rows, neg_one});
  Output reshape = Reshape(s.WithOpName("reshape"), gather2, pack1);
  Output pack2 = Stack(s.WithOpName("pack2"), {rows, pack_value});
  Output fill_value = Const(s.WithOpName("fill_value"), 0.0f);
  Output fill = Fill(s.WithOpName("fill"), pack2, fill_value);
  Output axis_neg1_3 = Const(s.WithOpName("axis_neg1_3"), -1);
  Output concat = Concat(s.WithOpName("concat"), {reshape, fill}, axis_neg1_3);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  EXPECT_EQ(CountOp(graph, "KPFusedEmbeddingActionIdGather"), 1);
}

// A dangling input reference: get_node() cannot resolve it, and the rewriter
// must bail out instead of dereferencing the null result. Static shape
// inference is skipped on purpose: the graph is intentionally malformed, and
// the structural checks under test run before any shape lookup.
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchDanglingInputNoCrash) {
  setenv("ANNC_FUSED_SPS_STITCH", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({4}));
  Output x1 = Reshape(s.WithOpName("x1"), x, {-1});
  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), x1, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output size = Size(s.WithOpName("size"), x1);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto orig_part = DynamicPartition(s.WithOpName("orig_part"), range, group_ids,
                                    kNumTables);
  Output div = FloorDiv(s.WithOpName("div"), x1, num);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);
  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 2}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitch = ParallelDynamicStitch(s.WithOpName("stitch"),
                                        orig_part.outputs, {g0, g1})
                      .merged;

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  const int node_count_before = graph.node_size();
  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == "stitch") {
      graph.mutable_node(i)->set_input(0, "does_not_exist:0");
    }
  }

  GrapplerItem item;
  item.graph = graph;
  GraphProperties props(item);
  annc::run_graph_optimization(&graph, props);  // must not crash

  EXPECT_EQ(graph.node_size(), node_count_before);
  EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitch"), 0);
}

// ANNCAnyFusionEnabled must reflect the ANNC_FUSED_* environment: false when
// every variable is unset or holds something other than "1", true as soon as
// one variable is exactly "1" (covers IsEnvEnabled and the env-name loop).
TEST_F(GraphOptTest, ANNCAnyFusionEnabled) {
  // SetUp() cleared all ANNC_FUSED_* variables.
  EXPECT_FALSE(annc::ANNCAnyFusionEnabled());

  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);
  EXPECT_TRUE(annc::ANNCAnyFusionEnabled());

  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "0", 1);
  EXPECT_FALSE(annc::ANNCAnyFusionEnabled());

  unsetenv("ANNC_FUSED_SPS_STITCH_REDUCE");
}

// Test KPFusedSparseDynamicStitchReduce with int64 gather axis (covers line
// 399-401)
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceInt64Axis) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));
  auto unique = Unique(s.WithOpName("unique"), x);
  Output unique_values = unique.y;
  Output unique_indices = unique.idx;

  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), unique_values, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output div = FloorDiv(s.WithOpName("div"), unique_values, num);
  Output size = Size(s.WithOpName("size"), unique_values);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto stitch_part = DynamicPartition(s.WithOpName("stitch_part"), range,
                                      group_ids, kNumTables);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);

  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  // Use int64 for axis instead of int32 to trigger line 399-401
  // But need to wrap it in a Const<int64_t> tensor
  Tensor axis_tensor(DT_INT64, TensorShape({}));
  axis_tensor.scalar<int64_t>()() = 0;
  Output axis = Const(s.WithOpName("axis"), axis_tensor);

  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitched = ParallelDynamicStitch(s.WithOpName("stitch"),
                                          stitch_part.outputs, {g0, g1})
                        .merged;

  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({6, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), stitched,
                                    unique_indices, seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  if (annc::RuntimeSveSupported()) {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 1);
  } else {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
  }
}

// Test with Reshape between Stitch and SparseSegment (covers has_reshape branch)
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceWithReshape) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));
  auto unique = Unique(s.WithOpName("unique"), x);
  Output unique_values = unique.y;
  Output unique_indices = unique.idx;

  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), unique_values, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output div = FloorDiv(s.WithOpName("div"), unique_values, num);
  Output size = Size(s.WithOpName("size"), unique_values);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto stitch_part = DynamicPartition(s.WithOpName("stitch_part"), range,
                                      group_ids, kNumTables);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);

  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output axis = Const(s.WithOpName("axis"), 0);

  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);

  Output stitched = ParallelDynamicStitch(s.WithOpName("stitch"),
                                          stitch_part.outputs, {g0, g1})
                        .merged;

  // Add Reshape between Stitch and SparseSegment (this tests the has_reshape branch)
  Output reshape_shape = Const(s.WithOpName("reshape_shape"), {-1, 4});
  Output reshaped_stitched = Reshape(s.WithOpName("reshape_stitch"), stitched, reshape_shape);

  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({6, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), reshaped_stitched,
                                    unique_indices, seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  if (annc::RuntimeSveSupported()) {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 1);
    // Reshape between Stitch and SparseSegment is skipped during pattern match
    // but remains in the graph (its input redirected to the fused op)
    EXPECT_EQ(CountOp(graph, "Reshape"), 1);
  } else {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
  }
}

// Test with Reshape on Unique input (covers line 333-337)
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceWithUniqueReshape) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));

  // Add Reshape before Unique
  Output reshape_shape = Const(s.WithOpName("reshape_shape"), {-1});
  Output reshaped_x = Reshape(s.WithOpName("reshape_x"), x, reshape_shape);

  auto unique = Unique(s.WithOpName("unique"), reshaped_x);
  Output unique_values = unique.y;
  Output unique_indices = unique.idx;

  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), unique_values, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output div = FloorDiv(s.WithOpName("div"), unique_values, num);

  // Add Reshape between Unique output and Size to test unique_has_reshape
  Output unique_reshape_shape = Const(s.WithOpName("unique_reshape_shape"), {-1});
  Output reshaped_unique = Reshape(s.WithOpName("reshape_unique"), unique_values, unique_reshape_shape);
  Output size = Size(s.WithOpName("size"), reshaped_unique);  // Size on reshaped Unique output

  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto stitch_part = DynamicPartition(s.WithOpName("stitch_part"), range,
                                      group_ids, kNumTables);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);

  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitched = ParallelDynamicStitch(s.WithOpName("stitch"),
                                          stitch_part.outputs, {g0, g1})
                        .merged;

  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT64,
                                   Placeholder::Shape({6, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output slice = StridedSlice(
      s.WithOpName("slice"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  Output seg_ids = Cast(s.WithOpName("seg_ids"), slice, DT_INT32);
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), stitched,
                                    unique_indices, seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  if (annc::RuntimeSveSupported()) {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 1);
  } else {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
  }
}

// Test without Cast on segment_ids (StridedSlice directly, no Cast)
TEST_F(GraphOptTest, KPFusedSparseDynamicStitchReduceNoCast) {
  setenv("ANNC_FUSED_SPS_STITCH_REDUCE", "1", 1);
  const int64_t kNumTables = 2;

  Scope s = Scope::NewRootScope();
  Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));
  auto unique = Unique(s.WithOpName("unique"), x);
  Output unique_values = unique.y;
  Output unique_indices = unique.idx;

  Output num = Const<int64>(s.WithOpName("num"), kNumTables);
  Output mod = FloorMod(s.WithOpName("mod"), unique_values, num);
  Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
  Output div = FloorDiv(s.WithOpName("div"), unique_values, num);
  Output size = Size(s.WithOpName("size"), unique_values);
  Output range = Range(s.WithOpName("range"), Const(s, 0), size, Const(s, 1));
  auto stitch_part = DynamicPartition(s.WithOpName("stitch_part"), range,
                                      group_ids, kNumTables);
  auto table_part = DynamicPartition(s.WithOpName("table_part"), div, group_ids,
                                     kNumTables);

  Output table0 = Placeholder(s.WithOpName("table0"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output table1 = Placeholder(s.WithOpName("table1"), DT_FLOAT,
                              Placeholder::Shape({10, 4}));
  Output axis = Const(s.WithOpName("axis"), 0);
  Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                       axis);
  Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                       axis);
  Output stitched = ParallelDynamicStitch(s.WithOpName("stitch"),
                                          stitch_part.outputs, {g0, g1})
                        .merged;

  // Use int32 slice_input to produce int32 segment_ids directly, no Cast
  Output slice_input = Placeholder(s.WithOpName("slice_input"), DT_INT32,
                                   Placeholder::Shape({6, 2}));
  Output begin = Const(s.WithOpName("begin"), {0, 1});
  Output end = Const(s.WithOpName("end"), {0, 2});
  Output strides = Const(s.WithOpName("strides"), {1, 1});
  Output seg_ids = StridedSlice(
      s.WithOpName("seg_ids"), slice_input, begin, end, strides,
      StridedSlice::Attrs().BeginMask(1).EndMask(1).ShrinkAxisMask(2));
  // No Cast here - seg_ids is directly int32
  Output reduced = SparseSegmentSum(s.WithOpName("reduced"), stitched,
                                    unique_indices, seg_ids);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));
  TF_ASSERT_OK(RunGraphOpt(&graph));

  if (annc::RuntimeSveSupported()) {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 1);
  } else {
    EXPECT_EQ(CountOp(graph, "KPFusedSparseDynamicStitchFastSum"), 0);
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
