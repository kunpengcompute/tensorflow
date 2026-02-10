/* Copyright 2025 Huawei. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SWITCH_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SWITCH_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "absl/status/status.h"

namespace tensorflow {
namespace grappler {

// Optimize TensorFlow subgraphs that shorten chain of switches
class SwitchOptimizer : public GraphOptimizer {
 public:
  SwitchOptimizer() {}
  explicit SwitchOptimizer(RewriterConfig::Toggle opt_level) {}

  ~SwitchOptimizer() override {}

  string name() const override { return "switch_optimizer"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SWITCH_OPTIMIZER_H_
