/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifdef KDNN_SWITCH_GFLAGS

#include "gflags/gflags.h"

// ANNC graph optimizations（annc 总开关对应 ANNC_FUSED_ALL，其余对应
// ANNC_FUSED_* 环境变量）。
DEFINE_bool(annc, false, "Enable all ANNC fused graph optimizations.");
DEFINE_bool(annc_fused_sps_stitch, false,
            "Enable fused sparse dynamic stitch.");
DEFINE_bool(annc_fused_sps_stitch_reduce, false,
            "Enable fused sparse dynamic stitch reduce.");
DEFINE_bool(annc_fused_sps_reduce, false,
            "Enable fused sparse segment reduce.");
DEFINE_bool(annc_fused_emb_padding, false, "Enable fused embedding padding.");
DEFINE_bool(annc_fused_emb_padding_fast, false,
            "Enable fused embedding padding (fast path).");
DEFINE_bool(annc_fused_sps_select, false, "Enable fused sparse select.");
DEFINE_bool(annc_fused_gather, false, "Enable fused gather.");
DEFINE_bool(annc_fused_sps_reshape, false, "Enable fused sparse reshape.");
DEFINE_bool(annc_fused_emb_actionid_gather, false,
            "Enable fused embedding action_id gather.");
DEFINE_bool(annc_fused_sps_reduce_nonzero, false,
            "Enable fused sparse segment reduce nonzero.");
DEFINE_bool(annc_fused_flashattn_fwd, false,
            "Enable fused flash attention forward.");

// KDNN custom operations（enable_kdnn 对应 TF_ENABLE_KDNN_OPTS 环境变量）。
DEFINE_bool(enable_kdnn, true, "Enable KDNN custom operations.");

#endif  // KDNN_SWITCH_GFLAGS
