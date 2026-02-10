#include "gflags/gflags.h"
// KDNN
DEFINE_bool(enable_kdnn, true, "Enable KDNN Operation");
DEFINE_int64(kdnn_num_threads, -1, "Set KDNN num threads, default same as intra threadpool");

// Switch Combine Optimization
DEFINE_bool(enable_switch_opt, false, "Enable switch chains combine optimization");

// ANNC Graph Optimizations
DEFINE_bool(annc, false, "Enable ANNC optimizations.");
DEFINE_bool(annc_fusion, false, "Enable graph fusion.");
DEFINE_bool(annc_fused_dyn_stitch, false, "Enable fused dynamic stitch.");
DEFINE_bool(annc_fused_seg_reduce, false,
            "Enable fused sparse segment mean/sum.");
DEFINE_bool(annc_fused_emd_padding, false, "Enable fused embedding padding.");
DEFINE_bool(annc_fused_emd_padding_fast, false,
            "Enable fused embedding padding.");
DEFINE_bool(annc_fused_sparse_select, false,
            "Enable fused fast embedding padding.");
DEFINE_bool(annc_fused_gather, false, "Enable fused gather.");
DEFINE_bool(annc_fused_sparse_reshape, false, "Enable fused sparse reshape.");
DEFINE_bool(annc_fused_emd_actionid_gather, false,
            "Enable fused embedding action_id gather.");
DEFINE_bool(annc_fused_seg_reduce_nozero, false,
            "Enable fused sparse segment reduce nozero.");
DEFINE_bool(annc_fused_matmul, false, "Enable fused matmul.");
DEFINE_bool(annc_fused_direct_hash_mod, false, "Enable fused direct hash mod.");
DEFINE_bool(annc_fused_dynamic_padding, false, "Enable fused dynamic padding.");
DEFINE_bool(annc_fused_trunc_seq, false, "Enable fused trunc seq.");
DEFINE_bool(annc_fused_topk_segment_min, false, "Enable fused topk segmentmin.");

// ANNC Constant Folding
DEFINE_int32(annc_cf_matmul_batchnorm, 0, "Enable BatchNorm folding.");
DEFINE_bool(annc_cf_relu, false, "Enable Relu folding.");
DEFINE_string(annc_cf_dump, "", "Enable dump constant folding result.");
DEFINE_bool(annc_cf_dump_text, false, "Enable dump constant folding text result.");