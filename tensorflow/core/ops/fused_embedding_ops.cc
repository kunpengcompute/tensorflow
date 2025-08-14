/*
 * @Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * @Description:
 * @Version: 1.0
 * @Date: 2025-07-30 17:33:04
 * @LastEditTime: 2025-07-30 17:33:04
 */

#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

REGISTER_OP("KPLookupEmbeddingByHash")
    .Input("lookup: string")
    .Input("weights: T_weight")
    .Attr("num_buckets: int >= 1")
    .Attr("combiner: int")
    .Attr("T_weight: {resource, float}")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle temp;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 1, &temp));
      DimensionHandle emb_size_dim = ctx->UnknownDim();
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_size_dim});
      ctx->set_output(0, output_shape);

      return OkStatus();
    });

}  // namespace tensorflow