#ifndef FAGO_TF_FLASH_ATTN_GRAPH_OPT_H_
#define FAGO_TF_FLASH_ATTN_GRAPH_OPT_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace fago{

// =============================================================================
// Perform Flash Attention forward fusion on GraphDef.
//
// Forward:
//   BatchMatMulV2(Q, K^T) ──┐
//                           ├─ AddV2 ─ Softmax ─ BatchMatMulV2(P, V) ── out
//                   mask ───┘
//   -> FlashAttentionForward(q, k, v, mask)
//      output :0 = attention_out (replace bmm_out)
//             :1 = logsumexp     (used for backword)
// =============================================================================
void run_flash_attn_optimization(tensorflow::GraphDef* graph);

}  // namespace FlashAttnGraphOpt

#endif  // FAGO_TF_FLASH_ATTN_GRAPH_OPT_H_
