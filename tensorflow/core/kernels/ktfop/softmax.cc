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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "ktfop.h"

namespace tensorflow {
class KPSoftmaxOp : public OpKernel {
 public:
  explicit KPSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in.shape().DebugString()));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in.shape(), &softmax_out));
    if (logits_in.NumElements() > 0) {
      typename TTypes<float>::ConstMatrix input = logits_in.flat_inner_dims<float>();
      float* input_data = (float *)logits_in.data();
      float* output_data = (float *)softmax_out->data();
      int result = ktfop::Softmax(input_data, output_data, input.dimension(0), input.dimension(1));
      OP_REQUIRES(context, (result == 0),
                errors::InvalidArgument("Invalid argument, error code: ", result));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("KPSoftmax").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    KPSoftmaxOp);

}  // namespace tensorflow