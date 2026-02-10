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
class KPFusedDirectHashMod : public OpKernel {
 public:
  explicit KPFusedDirectHashMod(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_1 = context->input(1);
    const Tensor& input_2 = context->input(2);

    int64 add_y = input_1.scalar<int64>()();
    int64 mod_y = input_2.scalar<int64>()();

    int64 num_elems = input_tensor.NumElements();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->flat<int64>();
    auto input_flat = input_tensor.flat<T>();

    auto input_eigen =
        Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>>(
            input_flat.data(), num_elems);

    auto output_eigen =
        Eigen::TensorMap<Eigen::Tensor<int64, 1, Eigen::RowMajor>>(
            output_flat.data(), num_elems);

    auto non_zero_mask =
        (input_eigen != static_cast<T>(0)).template cast<bool>();
    auto values = (input_eigen.template cast<int64>() + add_y) % mod_y;

    Eigen::Tensor<int64, 1, Eigen::RowMajor> zeros(num_elems);
    zeros.setZero();

    output_eigen = non_zero_mask.select(values, zeros);
  }
};

#define REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("KPFusedDirectHashMod")    \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          KPFusedDirectHashMod<type>)

REGISTER_KERNEL(int64);
REGISTER_KERNEL(int32);
#undef REGISTER_KERNEL