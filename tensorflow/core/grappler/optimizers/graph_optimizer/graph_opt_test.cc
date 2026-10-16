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

// Unit tests for the ANNC embedding-fusion graph optimizer (graph_opt.cc).
//
// Each test builds the *unfused* graph that the corresponding PatternRewriter
// is designed to recognize, runs annc::run_graph_optimization with only that
// rewriter enabled, and asserts that the fused node was produced.

#include "tensorflow/core/grappler/optimizers/graph_optimizer/graph_opt.h"

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
using ::tensorflow::ops::Const;
using ::tensorflow::ops::DynamicPartition;
using ::tensorflow::ops::FloorDiv;
using ::tensorflow::ops::FloorMod;
using ::tensorflow::ops::GatherV2;
using ::tensorflow::ops::ParallelDynamicStitch;
using ::tensorflow::ops::Placeholder;
using ::tensorflow::ops::Range;
using ::tensorflow::ops::Reshape;
using ::tensorflow::ops::Shape;
using ::tensorflow::ops::Size;
using ::tensorflow::ops::SparseSegmentSum;
using ::tensorflow::ops::StridedSlice;
using ::tensorflow::ops::Unique;

class GraphOptTest : public GrapplerTest {
 protected:
  void SetUp() override {
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

  StitchReduceGraph BuildStitchReduceGraph(const Scope& s, int gather1_axis) {
    const int64_t kNumTables = 2;  // power of two -> "Fast" fused op
    Output x = Placeholder(s.WithOpName("x"), DT_INT64, Placeholder::Shape({6}));
    auto unique = Unique(s.WithOpName("unique"), x);
    Output num = Const<int64>(s.WithOpName("num"), kNumTables);
    Output mod = FloorMod(s.WithOpName("mod"), unique.y, num);
    Output group_ids = Cast(s.WithOpName("group_ids"), mod, DT_INT32);
    Output div = FloorDiv(s.WithOpName("div"), unique.y, num);
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
    Output g0 = GatherV2(s.WithOpName("gather0"), table0, table_part.outputs[0],
                         axis);
    Output g1 = GatherV2(s.WithOpName("gather1"), table1, table_part.outputs[1],
                         axis1);
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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
