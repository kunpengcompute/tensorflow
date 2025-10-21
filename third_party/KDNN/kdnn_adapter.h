/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_KDNN_ADAPTER_H_
#define TENSORFLOW_CORE_UTIL_KDNN_ADAPTER_H_
#include "kdnn.hpp"
#include "tensorflow/core/util/matmul_bcast.h"
#include "kdnn_threadpool.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"

namespace tensorflow {


inline void kdnnParallelGemm(const OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                     const MatMulBCast& bcast, int start, int end, bool trans_x, bool trans_y) {
  const bool should_bcast = bcast.IsBroadcastingRequired();
  const auto& x_batch_indices = bcast.x_batch_indices();
  const auto& y_batch_indices = bcast.y_batch_indices();
  int m = a.dim_size(1);
  int n = b.dim_size(trans_y ? 1 : 2);
  int k = b.dim_size(trans_y ? 2 : 1);
  int stride_a = m * k;
  int stride_b = k * n;
  int stride_c = m * n;
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  // intra_op thread_pool
  thread::ThreadPool* thread_pool = 
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  KDNN::Threading::ActivateThreadpool(&kdnn_tp);
  const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, trans_y ? KDNN::Layout::BA : KDNN::Layout::AB};
  const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo);
  for (int64_t i = start; i < end; ++i) {
    const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
    const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
    gemm.Run(A + x_batch_index * stride_a, B + y_batch_index * stride_b, C + i * stride_c); 
  }
  KDNN::Threading::DeactivateThreadpool();
}

inline void kdnnFusedGemm(OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                    FusedComputationType fusion, const FusedComputationArgs& fusion_args, bool trans_x, bool trans_y) {
  int m = a.dim_size(0);
  int n = b.dim_size(trans_y ? 0 : 1);
  int k = b.dim_size(trans_y ? 1 : 0);
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  const Tensor& bias = ctx->input(2);
  const float *Bias = bias.flat<float>().data();
  if (bias.dims() != 1 || bias.dim_size(0) != n) {
    OP_REQUIRES_OK(ctx, errors::InvalidArgument("bias must be 1-dimensional and match n",
                            bias.shape().DebugString()));
  }
  KDNN::PostOpsDataPtrs po_ptrs;
  KDNN::PostOps post_ops;
  if (fusion == FusedComputationType::kBiasAddWithRelu) {
    post_ops.AppendEltwise(KDNN::ActivationFunction::RELU);
    po_ptrs.push_back(&post_ops);
  }
  // intra_op thread_pool
  thread::ThreadPool* thread_pool = 
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  KDNN::Threading::ActivateThreadpool(&kdnn_tp);
  const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, trans_y ? KDNN::Layout::BA : KDNN::Layout::AB};
  const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo biasInfo = {{1, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  KDNN::Attributes attr;
  attr.SetPostOps(post_ops);
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo, biasInfo, attr);
  gemm.Run(A, B, C, Bias, po_ptrs); 
  KDNN::Threading::DeactivateThreadpool();
}

}// namespace tensorflow
#endif