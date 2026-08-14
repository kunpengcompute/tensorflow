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

// =============================================================================
// flash_attn_op.cc
//
// OpKernel implementations for FlashAttentionForward / FlashAttentionBackward.
//
// The actual CUDA kernels live in flash_attn_op_gpu.cu.cc and are exposed
// here through functor::LaunchFlashAttentionForward / LaunchFlashAttentionBackward.
//
// Behaviour summary:
//   * Forward consumes Q, K, V (each [num_heads, seq_len, d_k]) and an additive
//     mask of shape [seq_len, seq_len], produces O (same shape as Q) and a
//     per-row logsumexp L of shape [num_heads, seq_len] used by the backward.
//   * Backward consumes Q, K, V, mask, O, dO, L and produces dQ, dK, dV.
//     A per-row scratch buffer D = sum_x(dO * O), shape [num_heads, seq_len],
//     is allocated through ctx->allocate_temp so it lives on the TF GPU
//     allocator (no cudaMalloc/Free on the hot path).
//
// Workspace (v25 tile-map + Split-KV):
//   * Both forward and backward allocate a second ctx->allocate_temp scratch
//     (DT_INT8) sized by FlashAttnFwdWorkspaceBytes / FlashAttnBwdWorkspaceBytes.
//     The launcher builds the mask tile-classification table (TILE_SKIP /
//     FULL / PARTIAL) into it; forward additionally uses it for Split-KV
//     partial sums (O_part / m_part / l_part) when num_splits > 1.
//   * Forward caches the device SM count on first Compute (sm_count_) and
//     feeds it to both the workspace sizing and the launcher, so they agree
//     on num_splits = FaChooseNumSplits(...).
//
// Only float is currently registered. Adding half / bfloat16 would just need
// matching kernel template instantiations in the .cu.cc plus an extra
// REGISTER_KERNEL_BUILDER block here.
// =============================================================================

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/flash_attn_op.h"

namespace tensorflow {

#if GOOGLE_CUDA

// =============================================================================
// FlashAttentionForwardOp
// =============================================================================
template <typename T>
class FlashAttentionForwardOp : public OpKernel {
 public:
  explicit FlashAttentionForwardOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), sm_count_(0) {}

  void Compute(OpKernelContext* ctx) override {
    // ---- Split-KV: query SM count on first Compute (device is now active),
    //      cache for the lifetime of this kernel. Both the workspace sizing
    //      and the launcher consume sm_count_ so they share one
    //      FaChooseNumSplits(...) decision. ----
    if (sm_count_ <= 0) {
      int dev = 0;
      if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
          sm_count_ = prop.multiProcessorCount;
        }
      }
    }

    const Tensor& Q    = ctx->input(0);
    const Tensor& K    = ctx->input(1);
    const Tensor& V    = ctx->input(2);
    const Tensor& mask = ctx->input(3);

    // ---- Shape validation ----
    OP_REQUIRES(ctx, Q.dims() == 3,
                errors::InvalidArgument(
                    "FlashAttentionForward: q must be rank-3 "
                    "[num_heads, seq_len, d_k], got shape ",
                    Q.shape().DebugString()));
    OP_REQUIRES(ctx, K.shape() == Q.shape(),
                errors::InvalidArgument(
                    "FlashAttentionForward: k shape must match q. q=",
                    Q.shape().DebugString(),
                    " k=", K.shape().DebugString()));
    OP_REQUIRES(ctx, V.shape() == Q.shape(),
                errors::InvalidArgument(
                    "FlashAttentionForward: v shape must match q. q=",
                    Q.shape().DebugString(),
                    " v=", V.shape().DebugString()));

    const int64_t num_heads_64 = Q.dim_size(0);
    const int64_t seq_len_64   = Q.dim_size(1);
    const int64_t d_k_64       = Q.dim_size(2);

    OP_REQUIRES(ctx, num_heads_64 > 0 && seq_len_64 > 0 && d_k_64 > 0,
                errors::InvalidArgument(
                    "FlashAttentionForward: all dims of q must be > 0, got ",
                    Q.shape().DebugString()));
    OP_REQUIRES(
        ctx,
        d_k_64 <= 256,
        errors::InvalidArgument(
            "FlashAttentionForward: the value of d_k must be <= 256, got ",
            d_k_64));
    OP_REQUIRES(
        ctx,
        num_heads_64 <= std::numeric_limits<int>::max() &&
            seq_len_64 <= std::numeric_limits<int>::max() &&
            d_k_64 <= std::numeric_limits<int>::max(),
        errors::InvalidArgument(
            "FlashAttentionForward: dims must fit in int32, got ",
            Q.shape().DebugString()));
    OP_REQUIRES(
        ctx,
        num_heads_64 % 4 == 0,
        errors::InvalidArgument(
            "FlashAttentionForward: num_heads must be a multiple of 4, got ",
            num_heads_64));

    const int num_heads = static_cast<int>(num_heads_64);
    const int seq_len   = static_cast<int>(seq_len_64);
    const int d_k       = static_cast<int>(d_k_64);

    OP_REQUIRES(
        ctx,
        mask.dims() == 2 &&
            mask.dim_size(0) == seq_len_64 &&
            mask.dim_size(1) == seq_len_64,
        errors::InvalidArgument(
            "FlashAttentionForward: mask must be [seq_len, seq_len]=[",
            seq_len, ", ", seq_len, "], got ", mask.shape().DebugString()));

    // ---- Allocate outputs ----
    Tensor* output    = nullptr;
    Tensor* logsumexp = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, Q.shape(), &output));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1, TensorShape({num_heads_64, seq_len_64}),
                            &logsumexp));

    // ---- Allocate tile-map + Split-KV workspace via TF GPU allocator ----
    Tensor ws;
    const size_t ws_bytes =
        functor::FlashAttnFwdWorkspaceBytes(num_heads, seq_len, d_k, sm_count_);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DT_INT8,
                            TensorShape({static_cast<int64_t>(ws_bytes)}),
                            &ws));

    // ---- Launch kernel ----
    auto* stream = ctx->eigen_gpu_device().stream();
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("FlashAttentionForward: null GPU stream"));

    OP_REQUIRES_OK(ctx, functor::LaunchFlashAttentionForward<T>(
                            stream,
                            Q.flat<T>().data(),
                            K.flat<T>().data(),
                            V.flat<T>().data(),
                            mask.flat<T>().data(),
                            output->flat<T>().data(),
                            logsumexp->flat<T>().data(),
                            ws.flat<int8>().data(),
                            num_heads, seq_len, d_k, sm_count_));
  }

 private:
  int sm_count_;  // SM count for the Split-KV heuristic (cached on first
                  // Compute; shared by workspace sizing and the launcher).
};

// =============================================================================
// FlashAttentionBackwardOp
// =============================================================================
template <typename T>
class FlashAttentionBackwardOp : public OpKernel {
 public:
  explicit FlashAttentionBackwardOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& Q    = ctx->input(0);
    const Tensor& K    = ctx->input(1);
    const Tensor& V    = ctx->input(2);
    const Tensor& mask = ctx->input(3);
    const Tensor& O    = ctx->input(4);
    const Tensor& dO   = ctx->input(5);
    const Tensor& L    = ctx->input(6);

    // ---- Shape validation (matches forward) ----
    OP_REQUIRES(ctx, Q.dims() == 3,
                errors::InvalidArgument(
                    "FlashAttentionBackward: q must be rank-3, got ",
                    Q.shape().DebugString()));
    OP_REQUIRES(ctx, K.shape() == Q.shape(),
                errors::InvalidArgument(
                    "FlashAttentionBackward: k shape mismatch. q=",
                    Q.shape().DebugString(),
                    " k=", K.shape().DebugString()));
    OP_REQUIRES(ctx, V.shape() == Q.shape(),
                errors::InvalidArgument(
                    "FlashAttentionBackward: v shape mismatch. q=",
                    Q.shape().DebugString(),
                    " v=", V.shape().DebugString()));
    OP_REQUIRES(ctx, O.shape() == Q.shape(),
                errors::InvalidArgument(
                    "FlashAttentionBackward: o shape must match q. q=",
                    Q.shape().DebugString(),
                    " o=", O.shape().DebugString()));
    OP_REQUIRES(ctx, dO.shape() == Q.shape(),
                errors::InvalidArgument(
                    "FlashAttentionBackward: grad_o shape must match q. q=",
                    Q.shape().DebugString(),
                    " grad_o=", dO.shape().DebugString()));

    const int64_t num_heads_64 = Q.dim_size(0);
    const int64_t seq_len_64   = Q.dim_size(1);
    const int64_t d_k_64       = Q.dim_size(2);

    OP_REQUIRES(
        ctx,
        num_heads_64 > 0 && seq_len_64 > 0 && d_k_64 > 0,
        errors::InvalidArgument(
            "FlashAttentionBackward: all dims of q must be > 0, got ",
            Q.shape().DebugString()));
    OP_REQUIRES(
        ctx,
        d_k_64 <= 256,
        errors::InvalidArgument(
            "FlashAttentionBackward: the value of d_k must be <= 256, got ",
            d_k_64));
    OP_REQUIRES(
        ctx,
        num_heads_64 <= std::numeric_limits<int>::max() &&
            seq_len_64 <= std::numeric_limits<int>::max() &&
            d_k_64 <= std::numeric_limits<int>::max(),
        errors::InvalidArgument(
            "FlashAttentionBackward: dims must fit in int32, got ",
            Q.shape().DebugString()));
    OP_REQUIRES(
        ctx,
        num_heads_64 % 4 == 0,
        errors::InvalidArgument(
            "FlashAttentionBackward: num_heads must be a multiple of 4, got ",
            num_heads_64));

    const int num_heads = static_cast<int>(num_heads_64);
    const int seq_len   = static_cast<int>(seq_len_64);
    const int d_k       = static_cast<int>(d_k_64);

    OP_REQUIRES(
        ctx,
        mask.dims() == 2 &&
            mask.dim_size(0) == seq_len_64 &&
            mask.dim_size(1) == seq_len_64,
        errors::InvalidArgument(
            "FlashAttentionBackward: mask must be [seq_len, seq_len]=[",
            seq_len, ", ", seq_len, "], got ", mask.shape().DebugString()));

    OP_REQUIRES(
        ctx,
        L.dims() == 2 &&
            L.dim_size(0) == num_heads_64 &&
            L.dim_size(1) == seq_len_64,
        errors::InvalidArgument(
            "FlashAttentionBackward: logsumexp must be "
            "[num_heads, seq_len]=[",
            num_heads, ", ", seq_len, "], got ", L.shape().DebugString()));

    // ---- Allocate outputs (dQ, dK, dV) ----
    Tensor* dQ = nullptr;
    Tensor* dK = nullptr;
    Tensor* dV = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, Q.shape(), &dQ));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, K.shape(), &dK));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, V.shape(), &dV));

    // ---- Allocate scratch D_buf [num_heads, seq_len] via TF GPU allocator ----
    Tensor d_buf;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::value,
                            TensorShape({num_heads_64, seq_len_64}),
                            &d_buf));

    // ---- Allocate tile-map workspace via TF GPU allocator ----
    Tensor ws;
    const size_t ws_bytes = functor::FlashAttnBwdWorkspaceBytes(seq_len);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DT_INT8,
                            TensorShape({static_cast<int64_t>(ws_bytes)}),
                            &ws));

    // ---- Launch kernel ----
    auto* stream = ctx->eigen_gpu_device().stream();
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("FlashAttentionBackward: null GPU stream"));

    OP_REQUIRES_OK(ctx, functor::LaunchFlashAttentionBackward<T>(
                            stream,
                            Q.flat<T>().data(),
                            K.flat<T>().data(),
                            V.flat<T>().data(),
                            mask.flat<T>().data(),
                            O.flat<T>().data(),
                            dO.flat<T>().data(),
                            L.flat<T>().data(),
                            dQ->flat<T>().data(),
                            dK->flat<T>().data(),
                            dV->flat<T>().data(),
                            d_buf.flat<T>().data(),
                            ws.flat<int8>().data(),
                            num_heads, seq_len, d_k));
  }
};

// =============================================================================
// Kernel registration (GPU, float only - matches the instantiated template
// specializations in flash_attn_op_gpu.cu.cc).
// =============================================================================
#define REGISTER_FLASH_ATTN_GPU(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("FlashAttentionForward")             \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T"),              \
                          FlashAttentionForwardOp<T>);              \
  REGISTER_KERNEL_BUILDER(Name("FlashAttentionBackward")            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T"),              \
                          FlashAttentionBackwardOp<T>);

REGISTER_FLASH_ATTN_GPU(float);

#undef REGISTER_FLASH_ATTN_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
