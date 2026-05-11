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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Fused double gather:
//   Step 1: temp[i, j, :] = params[indices1[i, j], :]  -> shape [I10, I11, P1]
//   Step 2: output[i, j, k, :] = temp[indices2[i, j], k, :]  -> shape [I20, I21, I11, P1]
// Fused: output[i, j, k, :] = params[indices1[indices2[i, j], k], :]
template <typename Tindices1, typename Tindices2>
static void FusedDoubleGatherImpl(OpKernelContext* context,
    const float* params_data, const TensorShape& params_shape,
    const Tindices1* indices1_data, const TensorShape& indices1_shape,
    const Tindices2* indices2_data, const TensorShape& indices2_shape,
    Tensor* output) {
  // params shape: [P0, P1] (2D) or [P0, P1, P2] (3D)
  // indices1 shape: [I10, I11] (2D) or [I10] (1D), values in [0, P0)
  // indices2 shape: [I20, I21] (2D) or [I21] (1D), values in [0, I10) for 2D, [0, P0) for 1D
  // output shape:
  //   2D+2D: [I20, I21, I11, P_row]
  //   1D+2D: [I20, I21, P_row]
  //   2D+1D: [I21, I11, P_row]
  //   1D+1D: [I21, P_row]
  // where P_row = P1 (2D params) or P1*P2 (3D params)
  OP_REQUIRES(context, params_shape.dims() >= 2 && params_shape.dims() <= 3,
      errors::InvalidArgument("params must be 2D or 3D matrix"));
  OP_REQUIRES(context, indices1_shape.dims() >= 1 && indices1_shape.dims() <= 2,
      errors::InvalidArgument("indices1 must be 1D or 2D matrix"));
  OP_REQUIRES(context, indices2_shape.dims() >= 1 && indices2_shape.dims() <= 2,
      errors::InvalidArgument("indices2 must be 1D or 2D matrix"));

  const int P0 = params_shape.dim_size(0);
  // P_row: number of floats per params row (P1 for 2D, P1*P2 for 3D)
  int64_t P_row = params_shape.dim_size(1);
  if (params_shape.dims() == 3) {
    P_row *= params_shape.dim_size(2);
  }
  
  // Handle indices1 dimensions: 1D [I10] or 2D [I10, I11]
  const int indices1_dims = indices1_shape.dims();
  const int I10 = indices1_shape.dim_size(0);
  const int I11 = (indices1_dims == 2) ? indices1_shape.dim_size(1) : 1;
  
  // Handle indices2 dimensions: 1D [I21] or 2D [I20, I21]
  const int indices2_dims = indices2_shape.dims();
  const int I20 = (indices2_dims == 2) ? indices2_shape.dim_size(0) : 1;
  const int I21 = (indices2_dims == 2) ? indices2_shape.dim_size(1) : indices2_shape.dim_size(0);

  // Build output shape based on indices dimensions
  TensorShape output_shape;
  if (indices2_dims == 2) {
    output_shape.AddDim(I20);
  }
  output_shape.AddDim(I21);
  if (indices1_dims == 2) {
    output_shape.AddDim(I11);
  }
  output_shape.AddDim(P_row);

  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape, output));
  VLOG(1) << "fused gather output shape: " << output->shape().DebugString();

  float* output_data = output->flat<float>().data();

  // Fused double gather (parallelized over the outer I20 * I21 work units):
  // 2D+2D: output[i, j, k, :] = params[indices1[indices2[i, j], k], :]
  // 1D+2D: output[i, j, :] = params[indices1[indices2[i, j]], :]
  // 2D+1D: output[j, k, :] = params[indices1[indices2[j], k], :]
  // 1D+1D: output[j, :] = params[indices1[indices2[j]], :]
  //
  // Each work unit (i, j) writes to a disjoint region of output, so no
  // synchronization is needed between parallel tasks.
  // cost_per_unit: 2D indices1 -> I11 * P_row floats copied; 1D -> P_row floats.
  const int64_t total_units = static_cast<int64_t>(I20) * I21;
  const int64_t cost_per_unit =
      (indices1_dims == 2) ? static_cast<int64_t>(I11) * P_row
                           : P_row;

  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  worker_threads->workers->ParallelFor(
      total_units, cost_per_unit,
      [&](int64_t begin, int64_t end) {
        for (int64_t flat = begin; flat < end; ++flat) {
          const int i = static_cast<int>(flat / I21);
          const int j = static_cast<int>(flat % I21);

          Tindices2 idx2 = indices2_data[(indices2_dims == 2) ? (i * I21 + j) : j];

          if (indices1_dims == 2) {
            // 2D indices1: idx2 indexes into first dimension (I10)
            if (TF_PREDICT_FALSE(idx2 < 0 || idx2 >= I10)) {
              context->CtxFailure(errors::InvalidArgument(
                  "FusedGather: indices2[",
                  (indices2_dims == 2) ? i : 0, ",", j, "]=", idx2,
                  " out of range [0, ", I10, ")"));
              return;
            }
            for (int k = 0; k < I11; ++k) {
              Tindices1 idx1 = indices1_data[idx2 * I11 + k];
              if (TF_PREDICT_FALSE(idx1 < 0 || idx1 >= P0)) {
                context->CtxFailure(errors::InvalidArgument(
                    "GatherV2 axis=0: index out of range"));
                return;
              }
              int64_t output_offset;
              if (indices2_dims == 2) {
                // 2D+2D: [i, j, k, :]
                output_offset = ((i * I21 + j) * I11 + k) * P_row;
              } else {
                // 2D+1D: [j, k, :]
                output_offset = (j * I11 + k) * P_row;
              }
              std::memcpy(
                  output_data + output_offset,
                  params_data + idx1 * P_row,
                  sizeof(float) * P_row
              );
            }
          } else {
            // 1D indices1: idx2 indexes into indices1, then idx1 indexes into params
            if (TF_PREDICT_FALSE(idx2 < 0 || idx2 >= I10)) {
              context->CtxFailure(errors::InvalidArgument(
                  "FusedGather: indices2[",
                  (indices2_dims == 2) ? i : 0, ",", j, "]=", idx2,
                  " out of range [0, ", I10, ")"));
              return;
            }
            Tindices1 idx1 = indices1_data[idx2];
            if (TF_PREDICT_FALSE(idx1 < 0 || idx1 >= P0)) {
              context->CtxFailure(errors::InvalidArgument(
                  "GatherV2 axis=0: index out of range"));
              return;
            }
            int64_t output_offset;
            if (indices2_dims == 2) {
              // 1D+2D: [i, j, :]
              output_offset = (i * I21 + j) * P_row;
            } else {
              // 1D+1D: [j, :]
              output_offset = j * P_row;
            }
            std::memcpy(
                output_data + output_offset,
                params_data + idx1 * P_row,
                sizeof(float) * P_row
            );
          }
        }
      });
}


template <typename Tindices1, typename Tindices2>
class KPFusedEmbeddingActionIdGatherOp : public OpKernel {
public:
  explicit KPFusedEmbeddingActionIdGatherOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& indices1 = context->input(0);
    const Tensor& params = context->input(1);
    const Tensor& indices2 = context->input(2);
    const Tensor& pack_dim = context->input(3);

    const Tensor& pack = context->input(4);

    VLOG(1) << "indices1 shape: " << indices1.shape().DebugString();
    VLOG(1) << "params shape: " << params.shape().DebugString();
    VLOG(1) << "indices2 shape: " << indices2.shape().DebugString();
    OP_REQUIRES(
      context,
      indices1.dims() >= 1 && indices1.dims() <= 2,
      errors::InvalidArgument("indices1 dims must be 1 or 2")
    );
    OP_REQUIRES(
      context,
      indices2.dims() >= 1 && indices2.dims() <= 2,
      errors::InvalidArgument("indices2 dims must be 1 or 2")
    );
    OP_REQUIRES(
      context,
      params.dims() >= 2 && params.dims() <= 3,
      errors::InvalidArgument("params dims must = 2 or 3")
    );
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsScalar(pack_dim.shape()),
      errors::InvalidArgument("pack_dim is scalar")
    );
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsScalar(pack.shape()),
      errors::InvalidArgument("pack const is scalar")
    );

    // Fused double gather: directly compute params[indices1[indices2[i]]]
    Tensor gathered;
    FusedDoubleGatherImpl<Tindices1, Tindices2>(
        context,
        params.flat<float>().data(), params.shape(),
        indices1.flat<Tindices1>().data(), indices1.shape(),
        indices2.flat<Tindices2>().data(), indices2.shape(),
        &gathered);
    int pack_size = pack_dim.scalar<int32>()();
    int pack_const = pack.scalar<int32>()();
    OP_REQUIRES(context, pack_size > 0, errors::InvalidArgument("pack_size must > 0"));
    int a_reshaped_cols = gathered.NumElements() / pack_size;
    auto a_reshaped = gathered.shaped<float, 2>({pack_size, a_reshaped_cols});
    Tensor* output;
    int output_cols = a_reshaped_cols + pack_const;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({pack_size, output_cols}), &output));
    auto a_reshaped_data = a_reshaped.data();
    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64_t pack_cost_per_row =
        static_cast<int64_t>(a_reshaped_cols) + pack_const;
    worker_threads->workers->ParallelFor(
        pack_size, pack_cost_per_row,
        [&](int64_t start_row, int64_t end_row) {
          float* base = output->matrix<float>().data();
          for (int64_t row = start_row; row < end_row; ++row) {
            float* dst_row = base + row * (a_reshaped_cols + pack_const);
            std::memcpy(
                dst_row, a_reshaped_data + row * a_reshaped_cols,
                sizeof(float) * a_reshaped_cols
            );
            std::memset(
                dst_row + a_reshaped_cols, 0, sizeof(float) * pack_const
            );
          }
        });
  }
};

#define REGISTER_CPU_KERNEL(Tindices1, Tindices2)                                \
  REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingActionIdGather") \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<Tindices1>("Tindices1") \
                              .TypeConstraint<Tindices2>("Tindices2"), \
                          KPFusedEmbeddingActionIdGatherOp<Tindices1, Tindices2>);

REGISTER_CPU_KERNEL(int64, int32)
REGISTER_CPU_KERNEL(int32, int32)
REGISTER_CPU_KERNEL(int64, int64)
REGISTER_CPU_KERNEL(int32, int64)

}