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

#include <arm_neon.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

template <typename T>
class KPFusedTruncSeq : public OpKernel {
 public:
  explicit KPFusedTruncSeq(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_1 = context->input(1);

    OP_REQUIRES(context, input_tensor.dims() == 2,
                errors::InvalidArgument("Input data shape must be 2D"));
    int dim_0 = input_tensor.dim_size(0);
    int dim_1 = input_tensor.dim_size(1);
    auto input_vec = input_tensor.flat<T>().data();
    int min_x = input_1.scalar<int32>()();
    OP_REQUIRES(context, min_x > 0,
                errors::InvalidArgument("min_x must be positive"));
    int64 num_elems = std::min(dim_1, min_x);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({dim_0, num_elems}), &output));
    auto output_data = output->flat<T>().data();

    for (int i = 0; i < dim_0; ++i) {
      const T* src_row = input_vec + i * dim_1;
      T* dst_row = output_data + i * num_elems;
      memcpy(dst_row, src_row, num_elems * sizeof(T));
    }
  }
};

#define REGISTER_KERNEL(type)                                               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("KPFusedTruncSeq").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      KPFusedTruncSeq<type>)

REGISTER_KERNEL(int64);
REGISTER_KERNEL(int32);
#undef REGISTER_KERNEL