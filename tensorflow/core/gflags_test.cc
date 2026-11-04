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

#include "tensorflow/core/platform/test.h"

#ifdef KDNN_SWITCH_GFLAGS
#include <string>

#include "gflags/gflags.h"

DECLARE_bool(annc);
DECLARE_bool(annc_fused_sps_stitch);
DECLARE_bool(annc_fused_sps_stitch_reduce);
DECLARE_bool(annc_fused_sps_reduce);
DECLARE_bool(annc_fused_emb_padding);
DECLARE_bool(annc_fused_emb_padding_fast);
DECLARE_bool(annc_fused_sps_select);
DECLARE_bool(annc_fused_gather);
DECLARE_bool(annc_fused_sps_reshape);
DECLARE_bool(annc_fused_emb_actionid_gather);
DECLARE_bool(annc_fused_sps_reduce_nonzero);
DECLARE_bool(annc_fused_flashattn_fwd);
DECLARE_bool(enable_kdnn);
#endif  // KDNN_SWITCH_GFLAGS

namespace tensorflow {
namespace {

#ifdef KDNN_SWITCH_GFLAGS

TEST(KdnnGflagsSwitchTest, Defaults) {
  EXPECT_FALSE(FLAGS_annc);
  EXPECT_FALSE(FLAGS_annc_fused_sps_stitch);
  EXPECT_FALSE(FLAGS_annc_fused_sps_stitch_reduce);
  EXPECT_FALSE(FLAGS_annc_fused_sps_reduce);
  EXPECT_FALSE(FLAGS_annc_fused_emb_padding);
  EXPECT_FALSE(FLAGS_annc_fused_emb_padding_fast);
  EXPECT_FALSE(FLAGS_annc_fused_sps_select);
  EXPECT_FALSE(FLAGS_annc_fused_gather);
  EXPECT_FALSE(FLAGS_annc_fused_sps_reshape);
  EXPECT_FALSE(FLAGS_annc_fused_emb_actionid_gather);
  EXPECT_FALSE(FLAGS_annc_fused_sps_reduce_nonzero);
  EXPECT_FALSE(FLAGS_annc_fused_flashattn_fwd);
  EXPECT_TRUE(FLAGS_enable_kdnn);
}

TEST(KdnnGflagsSwitchTest, CommandLineOption) {
  gflags::FlagSaver flag_saver;

  gflags::SetCommandLineOption("annc_fused_gather", "true");
  EXPECT_TRUE(FLAGS_annc_fused_gather);

  gflags::SetCommandLineOption("enable_kdnn", "false");
  EXPECT_FALSE(FLAGS_enable_kdnn);

  EXPECT_FALSE(FLAGS_annc);
}

#else

TEST(KdnnGflagsSwitchTest, EnvBranchSkipped) {
  GTEST_SKIP() << "KDNN_SWITCH_GFLAGS off: env-var branch, no gflags";
}

#endif  // KDNN_SWITCH_GFLAGS

}  // namespace
}  // namespace tensorflow
