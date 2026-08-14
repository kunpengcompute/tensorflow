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

#ifndef TENSORFLOW_CORE_KERNELS_FLASH_ATTN_OP_H_
#define TENSORFLOW_CORE_KERNELS_FLASH_ATTN_OP_H_

#include <cstddef>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// gpuStream_t / GpuLaunchKernel / GPU_DYNAMIC_CHECK etc. live here.
// (This is the include that fixes the historical
//  "'gpuStream_t' has not been declared" build error.)
// include "tensorflow/core/util/gpu_kernel_helper.h"
struct CUstream_st;
typedef CUstream_st* gpuStream_t;  
#endif

namespace tensorflow {

// =============================================================================
// Compile-time limits for the Flash Attention kernels.
//
// Forward auto-dispatches across three WMMA tiers, the largest of which
// supports d_k up to kFlashAttnFwdMaxD. Backward currently uses a single
// configuration with a fixed-size accumulator and is capped at
// kFlashAttnBwdMaxD.
// =============================================================================
constexpr int kFlashAttnFwdMaxD = 512;
constexpr int kFlashAttnBwdMaxD = 128;

namespace functor {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// -----------------------------------------------------------------------------
// FlashAttentionForward GPU launch.
//
//   q, k, v        : [num_heads, seq_len, d_k]
//   mask           : [seq_len, seq_len]   (additive, broadcast across heads)
//   output         : [num_heads, seq_len, d_k]   (out)
//   logsumexp      : [num_heads, seq_len]        (out, consumed by backward)
//   workspace      : caller-allocated scratch via ctx->allocate_temp; sized by
//                    FlashAttnFwdWorkspaceBytes(...). Layout (when num_splits
//                    > 1): [tile_map(256B aligned) | O_part | m_part | l_part];
//                    when num_splits == 1 only the tile_map prefix is used.
//   sm_count       : device multiProcessorCount, used by the Split-KV heuristic
//                    (must match the value passed to FlashAttnFwdWorkspaceBytes
//                    so the workspace and the launcher agree on num_splits).
//
// Returns errors::Internal on launch / sync failure.
// -----------------------------------------------------------------------------
template <typename T>
Status LaunchFlashAttentionForward(gpuStream_t stream,
                                   const T* q,
                                   const T* k,
                                   const T* v,
                                   const T* mask,
                                   T* output,
                                   T* logsumexp,
                                   void* workspace,
                                   int num_heads,
                                   int seq_len,
                                   int d_k,
                                   int sm_count);

// -----------------------------------------------------------------------------
// FlashAttentionBackward GPU launch.
//
//   q, k, v        : [num_heads, seq_len, d_k]    (forward inputs)
//   mask           : [seq_len, seq_len]
//   o              : [num_heads, seq_len, d_k]    (forward output)
//   grad_o         : [num_heads, seq_len, d_k]    (upstream gradient)
//   logsumexp      : [num_heads, seq_len]         (from forward)
//   grad_q, grad_k,
//   grad_v         : [num_heads, seq_len, d_k]    (out)
//   d_buf          : [num_heads, seq_len]   (caller-allocated scratch via
//                    ctx->allocate_temp; reused for the Algorithm-2 row-sum
//                    buffer D_i = sum_x(dO * O))
//   workspace      : caller-allocated scratch via ctx->allocate_temp; sized by
//                    FlashAttnBwdWorkspaceBytes(seq_len). Holds the mask
//                    tile-classification table (built once per backward).
//
// Returns errors::Internal on launch / sync failure.
// -----------------------------------------------------------------------------
template <typename T>
Status LaunchFlashAttentionBackward(gpuStream_t stream,
                                    const T* q,
                                    const T* k,
                                    const T* v,
                                    const T* mask,
                                    const T* o,
                                    const T* grad_o,
                                    const T* logsumexp,
                                    T* grad_q,
                                    T* grad_k,
                                    T* grad_v,
                                    T* d_buf,
                                    void* workspace,
                                    int num_heads,
                                    int seq_len,
                                    int d_k);

// -----------------------------------------------------------------------------
// Workspace sizing (host-side, pure functions). The OpKernel calls these to
// size the ctx->allocate_temp scratch; the launchers consume the same byte
// count internally. Both must use the identical Split-KV / tile-map layout.
//
//   FlashAttnFwdWorkspaceBytes : depends on (num_heads, seq_len, d_k, sm_count)
//                                because the Split-KV partial sums are sized
//                                by num_splits = FaChooseNumSplits(...).
//   FlashAttnBwdWorkspaceBytes : depends only on seq_len (a single tile_map of
//                                num_q_tiles * num_kv_tiles bytes).
// -----------------------------------------------------------------------------
std::size_t FlashAttnFwdWorkspaceBytes(int num_heads, int seq_len, int d_k,
                                       int sm_count);
std::size_t FlashAttnBwdWorkspaceBytes(int seq_len);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTN_OP_H_
