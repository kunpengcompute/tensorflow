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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

template <typename T>
class KPFusedDynamicPadding : public OpKernel {
 public:
  explicit KPFusedDynamicPadding(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_1 = context->input(1);
    const Tensor& input_2 = context->input(2);

    OP_REQUIRES(context, input_1.NumElements() == 1,
                errors::InvalidArgument("sub_x must be a scalar"));
    OP_REQUIRES(context, input_2.NumElements() == 1,
                errors::InvalidArgument("less_y must be a scalar"));
    int sub_x = input_1.scalar<int32>()();
    int less_y = input_2.scalar<int32>()();
    int rank = input_tensor.dims();
    int dim_last = input_tensor.dim_size(rank - 1);
    Tensor* output_less = nullptr;
    Tensor* output = nullptr;
    if (dim_last >= less_y) {
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({}), &output_less));
      output_less->scalar<bool>()() = false;

      OP_REQUIRES_OK(
          context, context->allocate_output(1, input_tensor.shape(), &output));
      *output = input_tensor;
      return;
    }

    OP_REQUIRES(context, less_y > 0 && sub_x == less_y,
                errors::InvalidArgument("sub_x must be equal to less_y"));
    int outer_size = 1;
    TensorShape output_shape;
    for (int i = 0; i < rank - 1; ++i) {
      int size = input_tensor.dim_size(i);
      output_shape.AddDim(size);
      outer_size *= size;
    }
    output_shape.AddDim(less_y);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output_less));
    output_less->scalar<bool>()() = true;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &output));

    // Reshape to 2D views for efficient padding operation
    auto input_2d = input_tensor.template shaped<T, 2>({outer_size, dim_last});
    auto output_2d = output->template shaped<T, 2>({outer_size, less_y});
    output_2d.setZero();

    // zero-padding on the right side of the last dimension
    Eigen::array<Eigen::Index, 2> offset{0, 0};
    Eigen::array<Eigen::Index, 2> extent{outer_size, dim_last};

    output_2d.slice(offset, extent) = input_2d;
  }
};

#define REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("KPFusedDynamicPadding")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          KPFusedDynamicPadding<type>)

REGISTER_KERNEL(int64);
REGISTER_KERNEL(int32);
#undef REGISTER_KERNEL
