#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/kernels/emhash8.hpp"

using namespace tensorflow;

class KPFusedTopKSegmentMin : public OpKernel {
 public:
  explicit KPFusedTopKSegmentMin(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    VLOG(2) << "Executing KPFusedGather operator";
    const Tensor& scores_tensor  = context->input(0);
    const Tensor& k_tensor  = context->input(1);
    const Tensor& ids_tensor = context->input(2);
    const Tensor& attrs_tensor = context->input(3);

    OP_REQUIRES(context, scores_tensor.dims() == 2, errors::Internal("scores_tensor dims must == 2"));
    OP_REQUIRES(context, k_tensor.dims() == 0, errors::Internal("k_tensor dims must == 0"));
    OP_REQUIRES(context, ids_tensor.dims() == 2, errors::Internal("ids_tensor dims must == 2"));
    OP_REQUIRES(context, attrs_tensor.dims() == 2, errors::Internal("attrs_tensor dims must == 2"));

    OP_REQUIRES(context, scores_tensor.dim_size(0) == 1, errors::Internal("scores_tensor batch_size must == 1"));
    OP_REQUIRES(context, ids_tensor.dim_size(0) == 1, errors::Internal("ids_tensor batch_size must == 1"));
    OP_REQUIRES(context, attrs_tensor.dim_size(0) == 1, errors::Internal("attrs_tensor batch_size must == 1"));
    const int64 num_elements = scores_tensor.dim_size(1);
    OP_REQUIRES(context, ids_tensor.dim_size(1) >= num_elements, 
                    errors::InvalidArgument("ids must more than or equal to num_elements"));
    OP_REQUIRES(context, attrs_tensor.dim_size(1) >= num_elements, 
                    errors::InvalidArgument("attrs must more than or equal to num_elements"));

    int32 k = k_tensor.scalar<int32>()();

    OP_REQUIRES(context, k <= num_elements, 
                    errors::InvalidArgument("k must less than or equal to num_elements"));

    auto scores = scores_tensor.matrix<float>();
    auto ids = ids_tensor.matrix<int64>();
    auto attrs = attrs_tensor.matrix<int64>();

    std::vector<int> idx(num_elements);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (scores(0, a) != scores(0, b)) return scores(0, a) > scores(0, b);
        return a < b;
    });

    Tensor* topk_scores_tensor = nullptr;
    Tensor* topk_attrs_tensor = nullptr;
    
    OP_REQUIRES_OK(context, context->allocate_output(
            1, TensorShape({1, k}), &topk_scores_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(
            2, TensorShape({1, k}), &topk_attrs_tensor));

    auto topk_scores = topk_scores_tensor->matrix<float>();
    auto topk_attrs = topk_attrs_tensor->matrix<int64>();

    emhash8::HashMap<int64_t, int32_t> id_to_min_rank;
    std::vector<int32> min_ranks;
    
    id_to_min_rank.reserve(k * 2);
    
    min_ranks.reserve(k);
    
    for (int32 i = 0; i < k; ++i) {
        int32 idx_i = idx[i];
        int64 id = ids(0, idx_i);
        topk_scores(0, i) = scores(0, idx_i);
        topk_attrs(0, i) = attrs(0, idx_i);
        auto it = id_to_min_rank.insert(std::make_pair(id,i));
        if (it.second) {
            min_ranks.push_back(i);
        }
    }

    Tensor* min_ranks_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({static_cast<int64>(min_ranks.size())}), &min_ranks_tensor));
    
    auto min_ranks_output = min_ranks_tensor->vec<int32>();
    for (size_t i = 0; i < min_ranks.size(); ++i) {
        min_ranks_output(i) = min_ranks[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedTopKSegmentMin").Device(DEVICE_CPU),
                        KPFusedTopKSegmentMin);