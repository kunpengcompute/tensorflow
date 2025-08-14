/*
 * @Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * @Description:
 * @Version: 1.0
 * @Date: 2025-07-30 17:43:04
 * @LastEditTime: 2025-07-30 17:43:04
 */

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

#include "lookup_embedding_by_hash.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

struct LookupEmbeddingByHashFunctor {
  int operator()(uintptr_t *lookup_embedding, size_t *lookup_length, int32_t batch_size, float *embedding_table, 
                 int64_t embedding_size, int32_t embedding_dims, float *output) {
    return KPLookupEmbeddingByHashImpl::Compute(lookup_embedding, lookup_length, batch_size, embedding_table, 
                                                embedding_size, embedding_dims, output);
  }
};

class KPLookupEmbeddingByHashOp : public OpKernel {
    public:
      explicit KPLookupEmbeddingByHashOp(OpKernelConstruction* context)
               : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
        node_name = context->def().name();
      }

      void Compute(OpKernelContext* context) override {
        float *weight;
        const Tensor& input_tensor = context->input(0);
        const Tensor* weight_tensor = &context->input(1);
        
        if (weight_tensor->dtype() == DT_RESOURCE) {
          Var* variable;
          OP_REQUIRES_OK(context,
                        LookupResource(context, HandleFromInput(context, 1), 
                                        &variable));
          core::ScopedUnref s(variable);
          weight_tensor = variable->tensor();
          OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                      errors::InvalidArgument("Expect float weight in ",
                                              node_name));
        }
        
        auto input = input_tensor.flat<tstring>();
        weight = (float *)weight_tensor->tensor_data().data();
        int64_t batch = input_tensor.dim_size(0);
        int64_t embedding_dims = weight_tensor->dim_size(1);
        uintptr_t cstr_addresses[batch];
        size_t cstr_length[batch];
        for (int i = 0; i < batch; ++i) {
          cstr_addresses[i] = reinterpret_cast<uintptr_t>(input(i).c_str());
          cstr_length[i] = input(i).length();
        }
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, 
                       context->allocate_output(
                        0, TensorShape({batch, embedding_dims}), 
                        &output_tensor));
        float *output = (float *)output_tensor->tensor_data().data();
        LookupEmbeddingByHashFunctor lookfunctor;
        int result = lookfunctor(cstr_addresses, cstr_length, batch,
                                 weight, num_buckets_, embedding_dims, output);
        OP_REQUIRES(context, (result == 0),
                errors::InvalidArgument("Invalid argument, error code: ", result));
      }

    private:
        int64_t num_buckets_;
        std::string node_name;
};
REGISTER_KERNEL_BUILDER(Name("KPLookupEmbeddingByHash").Device(DEVICE_CPU), KPLookupEmbeddingByHashOp);
}  // namespace tensorflow
