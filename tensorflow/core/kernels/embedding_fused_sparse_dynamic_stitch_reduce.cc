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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if !defined(__aarch64__)
#error "KPFusedSparseDynamicStitch requires an AArch64 CPU with SVE"
#endif

// This fused kernel is deployed only on SVE-capable AArch64 systems. Keep the
// target attribute on the helpers because TensorFlow's global target is generic
// AArch64 rather than +sve.
#include <arm_sve.h>
#define KPFUSED_HAS_SVE 1
#define KPFUSED_SVE_TARGET __attribute__((target("+sve")))

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

KPFUSED_SVE_TARGET inline void ScaleEmbedding(float* output,
                                               int64_t embedding_width,
                                               float scale) {
  int64_t col = 0;
  const svfloat32_t scale_value = svdup_n_f32(scale);
  while (col < embedding_width) {
    const svbool_t predicate = svwhilelt_b32(col, embedding_width);
    const svfloat32_t output_value = svld1_f32(predicate, output + col);
    svst1_f32(predicate, output + col,
              svmul_f32_x(predicate, output_value, scale_value));
    col += svcntw();
  }
}

// Reduction semantics: plain sum, mean (1/n), or sqrt_n (1/sqrt(n)) segment
// normalization.  Sum performs no final scaling; mean and sqrt_n track the
// group size and normalize once per segment.
enum class ReduceMode { kSum, kMean, kSqrtN };

template <ReduceMode Mode, bool UseFast>
class KPFusedSparseDynamicStitchReduceOp : public OpKernel {
 public:
  explicit KPFusedSparseDynamicStitchReduceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_tables_attr_));
    OP_REQUIRES_OK(
        context, context->GetAttr("emb_tables_len", &emb_tables_len_));
    OP_REQUIRES(context, num_tables_attr_ > 0,
                errors::InvalidArgument("N must be positive"));
    OP_REQUIRES(context, emb_tables_len_ > 0,
                errors::InvalidArgument("emb_tables_len must be positive"));
    if constexpr (UseFast) {
      OP_REQUIRES(context,
                  (num_tables_attr_ & (num_tables_attr_ - 1)) == 0,
                  errors::InvalidArgument(
                      "fast fused reducer requires N to be a power of two"));
      table_mask_ = num_tables_attr_ - 1;
      while ((int64_t{1} << table_shift_) < num_tables_attr_) {
        ++table_shift_;
      }
    }
  }

  KPFUSED_SVE_TARGET void Compute(OpKernelContext* context) override {
    // 参数获取及校验
    const Tensor& x = context->input(0);
    const auto x_flat = x.flat<int64_t>();
    const int num_tables = num_tables_attr_;
    const int slice_input_index = num_tables + 1;
    const int begin_index = num_tables + 2;
    const int end_index = num_tables + 3;
    const int strides_index = num_tables + 4;
    OP_REQUIRES(context, context->num_inputs() == num_tables + 5,
                errors::InvalidArgument("invalid fused op input count"));

    std::vector<const float*> variables(num_tables);
    std::vector<int64_t> variable_rows(num_tables);
    int64_t output_stride = 0;
    for (int i = 0; i < num_tables; ++i) {
      const Tensor& table = context->input(i + 1);
      OP_REQUIRES(context, table.dims() == 2,
                  errors::InvalidArgument("embedding table must be rank 2"));
      OP_REQUIRES(context, table.dim_size(1) == emb_tables_len_,
                  errors::InvalidArgument(
                      "emb_tables_len does not match embedding table width"));
      if (i == 0) output_stride = table.dim_size(1);
      OP_REQUIRES(context, table.dim_size(1) == output_stride,
                  errors::InvalidArgument("embedding table widths differ"));
      variables[i] = table.flat<float>().data();
      variable_rows[i] = table.dim_size(0);
    }

    const Tensor& slice_input = context->input(slice_input_index);
    const Tensor& begin = context->input(begin_index);
    const Tensor& end = context->input(end_index);
    const Tensor& strides = context->input(strides_index);
    OP_REQUIRES(context, slice_input.dims() == 2,
                errors::InvalidArgument("slice_input must be rank 2"));
    OP_REQUIRES(context, begin.NumElements() == 2 && end.NumElements() == 2 &&
                             strides.NumElements() == 2,
                errors::InvalidArgument(
                    "begin/end/strides must have length 2"));
    OP_REQUIRES(context, strides.flat<int32_t>()(0) == 1,
                errors::InvalidArgument("slice row stride must be one"));
    OP_REQUIRES(context, x_flat.size() == slice_input.dim_size(0),
                errors::InvalidArgument(
                    "flattened x size must equal slice_input rows"));
    const int64_t column = begin.flat<int32_t>()(1);
    OP_REQUIRES(context, column >= 0 && column < slice_input.dim_size(1),
                errors::InvalidArgument("slice column is out of range"));

    const auto segment_ids = slice_input.matrix<int64_t>();
    const int64_t input_size = x_flat.size();
    int64_t previous_segment = -1;
    if (input_size > 0) {
      previous_segment = segment_ids(input_size - 1, column);
      OP_REQUIRES(context, previous_segment >= 0,
                  errors::InvalidArgument("segment_ids must be non-negative"));
      OP_REQUIRES(context,
                  previous_segment <= std::numeric_limits<int32_t>::max(),
                  errors::InvalidArgument("segment id is too large"));
    }
    const int64_t batch_size = previous_segment + 1;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, output_stride}),
                                &output_tensor));
    float* output = output_tensor->flat<float>().data();
    if (batch_size > 0) {
      std::fill(output, output + batch_size * output_stride, 0.0f);
    }

    int64_t current_segment = -1;
    int64_t group_size = 0;
    float* group_result = nullptr;
    for (int64_t index = 0; index < input_size; ++index) {
      const int64_t segment_id = segment_ids(index, column);
      OP_REQUIRES(context, segment_id >= current_segment,
                  errors::InvalidArgument(
                      "segment_ids must be sorted for fused reduction"));
      if (segment_id != current_segment) {
        if constexpr (Mode != ReduceMode::kSum) {
          if (group_size > 0) {
            ScaleEmbedding(group_result, output_stride,
                           ReductionScale(group_size));
          }
          group_size = 0;
        }
        current_segment = segment_id;
        group_result = output + segment_id * output_stride;
      }

      const int64_t global_id = x_flat(index);
      OP_REQUIRES(context, global_id >= 0,
                  errors::InvalidArgument("global_id must be non-negative"));
      int64_t table_id;
      int64_t row_id;
      if constexpr (UseFast) {
        const uint64_t unsigned_id = static_cast<uint64_t>(global_id);
        table_id = unsigned_id & table_mask_;
        row_id = unsigned_id >> table_shift_;
      } else {
        table_id = global_id % num_tables;
        row_id = global_id / num_tables;
      }
      OP_REQUIRES(context, row_id < variable_rows[table_id],
                  errors::InvalidArgument("row_id out of range"));
      const float* table_row =
          variables[table_id] + row_id * output_stride;
      for (int64_t j = 0; j < output_stride; ++j) {
        group_result[j] += table_row[j];
      }
      if constexpr (Mode != ReduceMode::kSum) {
        ++group_size;
      }
    }
    if constexpr (Mode != ReduceMode::kSum) {
      if (group_size > 0) {
        ScaleEmbedding(group_result, output_stride,
                       ReductionScale(group_size));
      }
    }
  }

 private:
  // Normalization factor for the current reduction mode: 1/n for mean and
  // 1/sqrt(n) for sqrt_n.  Callers guard this with Mode != kSum, so the kSum
  // instantiation never references it.
  static float ReductionScale(int64_t group_size) {
    if constexpr (Mode == ReduceMode::kMean) {
      return 1.0f / static_cast<float>(group_size);
    }
    return 1.0f / std::sqrt(static_cast<float>(group_size));
  }

  int num_tables_attr_ = 0;
  int emb_tables_len_ = 0;
  int table_shift_ = 0;
  int64_t table_mask_ = 0;
};

REGISTER_KERNEL_BUILDER(
    Name("KPFusedSparseDynamicStitchSum").Device(DEVICE_CPU),
    KPFusedSparseDynamicStitchReduceOp<ReduceMode::kSum, false>);
REGISTER_KERNEL_BUILDER(
    Name("KPFusedSparseDynamicStitchMean").Device(DEVICE_CPU),
    KPFusedSparseDynamicStitchReduceOp<ReduceMode::kMean, false>);
REGISTER_KERNEL_BUILDER(
    Name("KPFusedSparseDynamicStitchSqrtN").Device(DEVICE_CPU),
    KPFusedSparseDynamicStitchReduceOp<ReduceMode::kSqrtN, false>);
REGISTER_KERNEL_BUILDER(
    Name("KPFusedSparseDynamicStitchFastSum").Device(DEVICE_CPU),
    KPFusedSparseDynamicStitchReduceOp<ReduceMode::kSum, true>);
REGISTER_KERNEL_BUILDER(
    Name("KPFusedSparseDynamicStitchFastMean").Device(DEVICE_CPU),
    KPFusedSparseDynamicStitchReduceOp<ReduceMode::kMean, true>);
REGISTER_KERNEL_BUILDER(
    Name("KPFusedSparseDynamicStitchFastSqrtN").Device(DEVICE_CPU),
    KPFusedSparseDynamicStitchReduceOp<ReduceMode::kSqrtN, true>);

}  // namespace
}  // namespace tensorflow