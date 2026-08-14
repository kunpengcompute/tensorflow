/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// FlashAttentionForward
//
//   Inputs:
//     q     : [num_heads, seq_len, d_k]
//     k     : [num_heads, seq_len, d_k]
//     v     : [num_heads, seq_len, d_k]
//     mask  : [seq_len, seq_len]  (additive, broadcast across heads;
//                                  0 = attend, large negative finite value
//                                  e.g. -1e9 = block)
//
//   Outputs:
//     output    : [num_heads, seq_len, d_k]
//     logsumexp : [num_heads, seq_len]   (consumed by backward)
//
//   Constraints:
//     d_k <= 512 (forward auto-tiered: <=96 / <=128 / <=512)
//     seq_len > 0  (no 32-alignment requirement; tail handled in kernel)
// =============================================================================
REGISTER_OP("FlashAttentionForward")
    .Input("q: T")
    .Input("k: T")
    .Input("v: T")
    .Input("mask: T")
    .Output("output: T")
    .Output("logsumexp: T")
    .Attr("T: {float, half, bfloat16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // Q, K, V: rank-3, identical shapes
      ShapeHandle q_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &q_shape));
      ShapeHandle k_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &k_shape));
      ShapeHandle v_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &v_shape));

      ShapeHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(q_shape, k_shape, &merged));
      TF_RETURN_IF_ERROR(c->Merge(merged, v_shape, &merged));

      // mask: rank-2 [seq_len, seq_len]
      ShapeHandle mask_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &mask_shape));

      DimensionHandle num_heads = c->Dim(merged, 0);
      DimensionHandle seq_len   = c->Dim(merged, 1);

      // Cross-check mask dims against q seq_len when both are known.
      DimensionHandle mask_d0 = c->Dim(mask_shape, 0);
      DimensionHandle mask_d1 = c->Dim(mask_shape, 1);
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(mask_d0, seq_len, &unused));
      TF_RETURN_IF_ERROR(c->Merge(mask_d1, seq_len, &unused));

      // output shape == q shape
      c->set_output(0, merged);
      // logsumexp shape: [num_heads, seq_len]
      c->set_output(1, c->MakeShape({num_heads, seq_len}));
      return OkStatus();
    })
    .Doc(R"doc(
Flash Attention forward pass with logsumexp output for backward.

Computes exact scaled dot-product attention using tiled online softmax
(Algorithm 1 from the Flash Attention paper), avoiding materializing the
full N*N attention matrix in HBM.

q: Query tensor of shape [num_heads, seq_len, d_k].
k: Key tensor of shape [num_heads, seq_len, d_k].
v: Value tensor of shape [num_heads, seq_len, d_k].
mask: Additive attention mask of shape [seq_len, seq_len], broadcast across
      heads. Use 0 for "attend" and a large negative FINITE value <= -1e7
      (e.g. -1e9) for "block". Do NOT use -inf: fully blocked rows would
      degenerate to NaN (same as the native softmax path). Tiles whose mask
      elements are all <= -1e7 are skipped entirely (block-sparse fast path).
output: Attention output of shape [num_heads, seq_len, d_k].
logsumexp: Per-row log-sum-exp of shape [num_heads, seq_len], consumed by
           the backward op.

The forward kernel auto-dispatches across three tiers based on d_k:
  d_k <= 96   -> BLOCK 64x64, 4 warps
  d_k <= 128  -> BLOCK 32x32, 2 warps
  d_k <= 512  -> BLOCK 16x16, 1 warp

Forward also applies Split-KV when the GPU is under-occupied: the KV loop is
split into `num_splits` chunks (heuristic target >= 6 full waves, S <= 8),
each producing an unnormalized partial sum, which a merge kernel reduces in
fp32 (mathematically equivalent to the unsplit online softmax; S == 1 is
byte-identical to the unsplit path).
)doc");

// =============================================================================
// FlashAttentionBackward
//
//   Inputs:
//     q, k, v   : [num_heads, seq_len, d_k]   (forward inputs, replayed)
//     mask      : [seq_len, seq_len]          (same tensor as forward)
//     o         : [num_heads, seq_len, d_k]   (forward output)
//     grad_o    : [num_heads, seq_len, d_k]   (upstream gradient)
//     logsumexp : [num_heads, seq_len]        (forward output 1)
//
//   Outputs:
//     grad_q, grad_k, grad_v : same shape as q, k, v
//
//   Constraints:
//     d_k <= 128 (backward kernel uses fixed-size accumulator fragments)
// =============================================================================
REGISTER_OP("FlashAttentionBackward")
    .Input("q: T")
    .Input("k: T")
    .Input("v: T")
    .Input("mask: T")
    .Input("o: T")
    .Input("grad_o: T")
    .Input("logsumexp: T")
    .Output("grad_q: T")
    .Output("grad_k: T")
    .Output("grad_v: T")
    .Attr("T: {float, half, bfloat16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // grad_q/k/v share shapes with q/k/v respectively.
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return OkStatus();
    })
    .Doc(R"doc(
Flash Attention backward pass (Algorithm 2 from the Flash Attention paper).

Takes Q, K, V, mask, O (forward output), grad_O, and logsumexp (from forward),
produces grad_Q, grad_K, grad_V.

Mask must be the same tensor passed to the forward op; the backward
recomputes P = exp((S + mask) - L) and so must see identical additive bias
to match L exactly. Mask semantics follow the forward op: "block" must be a
large negative FINITE value <= -1e7 (e.g. -1e9); fully blocked tiles are
skipped (block-sparse).

q, k, v: Forward inputs, shape [num_heads, seq_len, d_k].
mask: Additive attention mask, shape [seq_len, seq_len].
o: Forward output, shape [num_heads, seq_len, d_k].
grad_o: Upstream gradient, shape [num_heads, seq_len, d_k].
logsumexp: Forward log-sum-exp output, shape [num_heads, seq_len].
grad_q, grad_k, grad_v: Gradients of the loss w.r.t. q, k, v.
)doc");

}  // namespace tensorflow
