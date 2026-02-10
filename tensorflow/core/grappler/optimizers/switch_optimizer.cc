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

#include "tensorflow/core/grappler/optimizers/switch_optimizer.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/lib/core/errors.h"
#include "absl/status/status.h"

namespace tensorflow {
namespace grappler {
const string thisPrefix = "swo_";

struct SwitchChain {
    // The producer, usually a data node.
    NodeDef *producer;

    // Ths consumer, usually a compute node guarded by switch node[s]
    NodeDef *consumer;

    // The chain of switches, ordered in reverse
    std::vector <std::pair<NodeDef*, bool>> switches;

    // The chain of predicate, only need their names here
    std::vector <std::pair<string, bool>> predicates;

    // The built string used to match for the repeated chain of predicates
    string match_str;
};

using SwitchChains = std::vector<SwitchChain>;
using Group = std::vector<NodeDef*>;

// SwitchOptimizer reduces a chain of switches.
// For a given chain of switches
// tf.add(tf.raw_ops.Switch(tf.raw_ops.Switch(tf.raw_ops.Switch(data, pred0)[1], pred1)[0], pred2))[1], x)
// Replace it with
// all_preds = tf.logical_and(pred0, tf.logical_and(tf.logical_not(pred1), pred2))
// tf.add(tf.raw_ops.Switch(data, all_preds)[1],x)
// By doing so, number of Switches can be reduced, at the cost of computing more logical_and/not node.
// However, in the real-world models, the same chain of logical_and/not can be reused in many places. Say there
// are other places like this:
// tf.multiply(tf.raw_ops.Switch(tf.raw_ops.Switch(tf.raw_ops.Switch(data2, pred0)[1], pred1)[0], pred2))[1], y)
// to be replaced with
// tf.multiply(data2, tf.raw_ops.Switch(all_preds, all_preds)[1], y)
// Thus this replacement should actually benefit.
absl::Status SwitchOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
  VLOG(0) << "In SwitchOptimizer::Optimize\n";
  *optimized_graph = item.graph;
  NodeMap node_map(optimized_graph);
  Status status;
  utils::MutableGraphView graph_view(optimized_graph, &status);
  TF_RETURN_IF_ERROR(graph_view.SortTopologically(false, {}));
  std::vector<Group> groupList={};
  for (auto& nv: graph_view.GetNodes()) {
    NodeDef* np = nv.node();
    NodeDef& node = *np;
    bool found = false;
    auto all_inputs = node.input();
    for (auto in : all_inputs) {
      string input_name = NodeName(in);
      // check if this switch's input data (which should be another switch) is in the list
      for (Group& nodes: groupList) {
        if (std::any_of(nodes.begin(), nodes.end(), [input_name](NodeDef* n) {
            return IsSwitch(*n) && n->name() == input_name; }))
        {
          // Input is a switch node, and the switch node is in this group
          VLOG(2) << "Pusing node to existing group: " << node.name()<< "\n";
          nodes.push_back(&node);
          found = true;
          break;
        }
      }
      // If found, don't break here as a node may have more than one inputs that are switch
    }
    if (!found) {
      if (IsSwitch(node)) {
        // This is a switch node, and none of its input is in a existing group.
        // Adding this switch to a new group
        Group newGroup;
        VLOG(2) << "Pusing node to new group: " << node.name()<< "\n";
        newGroup.push_back(&node);
        groupList.push_back(newGroup);
      }
      else {
        VLOG(2) << "Not adding into any group: " << node.name()<< "\n";
      }
    }
  }
  for (Group nodes: groupList) {
    VLOG(2) << "Group start: " << nodes.size() << "\n";
    for (auto n: nodes) {
      VLOG(2) << n->name() << "\n";
    }
    VLOG(2) << "Group end\n";
  }
  // Now build switch chains from groups
  SwitchChains chains={};
  for (Group nodes: groupList) {
    std::vector<NodeDef*> switchList = {};
    std::vector<NodeDef*> consumerList = {};
    for (NodeDef * np : nodes) {
      if (IsSwitch(*np))
        switchList.push_back(np);
      else
        consumerList.push_back(np);
    }
    // If a switch has more than two users forming a fork, reject the group
    if (switchList.empty())
      continue;
    bool invalide = false;
    auto last_it = std::prev(switchList.end());
    for (auto it = switchList.begin(); it != last_it; it++) {
      if (node_map.GetOutputs(NodeName((*it)->name())).size() != 1)
        invalide = true;
    }
    if (invalide)
      continue;
    for (NodeDef * consumer: consumerList) {
      SwitchChain chain={};
      chain.consumer = consumer;
      NodeDef * tailSwitch;
      bool found = false;
      for (auto in: consumer->input()) {
        int pos;
        auto it = std::find_if(switchList.begin(), switchList.end(), [in,&pos](NodeDef *n) {
          return n->name() == ParseNodeName(in, &pos);});
        if (it != switchList.end()) {
          tailSwitch = *it;
          // Insert this switch in the front of the list
          chain.switches.insert(chain.switches.begin(), std::make_pair(tailSwitch, bool(pos)));
          found = true;
          break;
        }
      }
      if (found) {
        for (NodeDef * theSwitch = tailSwitch;;) {
          auto switch_data = theSwitch->input(0);
          int data_pos;
          string switch_data_name = ParseNodeName(switch_data, &data_pos);
          auto it = std::find_if(switchList.begin(), switchList.end(), [switch_data_name](NodeDef *n) {
            return n->name() == switch_data_name;});
          if (it != switchList.end()) {
            // Insert this switch in the front of the list
            chain.switches.insert(chain.switches.begin(), std::make_pair(*it, bool(data_pos)));
            theSwitch = *it;
          }
          else {
            // Cannot find further switch in the producer chain, Log the most recent producer and stop
            chain.producer = node_map.GetNode(NodeName(theSwitch->input(0)));
            break;
          }
        }
      }
      else {
        return Status(absl::StatusCode::kNotFound, "Not switch in a chain");
      }
      chains.push_back(std::move(chain));
    }
  }
  VLOG(2) << "Chains:\n";
  for (auto chain: chains) {
    VLOG(2) << "Producer: " << chain.producer->name() << "\n";
    //for (auto s: chain.switches) {
    for (auto it = chain.switches.begin(); it != chain.switches.end(); ++it) {
      VLOG(2) << it->first->name() <<":"<<it->second<< " - ";
    }
    VLOG(2) << "\nConsumer: " << chain.consumer->name() << "\n";
  }

  // Now build predicate chains from the switch chains and match_str
  //using predicateChain = std::pair<std::vector<string>, string>;
  for (auto& sc : chains) {
    if (sc.switches.size() <= 1) continue;
    for (auto it = sc.switches.begin(); it != sc.switches.end(); ++it) {
      auto thisSwitch = *it;
      // Always take the last predicate as true in matching
      auto thisPred = thisSwitch.first->input(1);
      auto thisPredValue = thisSwitch.second;
      //auto thisPredValue = ((std::next(it) == sc.switches.end()) ? true : thisSwitch.second);
      sc.predicates.push_back(std::make_pair(thisPred, thisPredValue));
      sc.match_str += "@";
      sc.match_str += thisPred;
      sc.match_str += "^";
      sc.match_str += std::to_string(thisPredValue);
    }
  }

  VLOG(2) << "Predicate chain strings:\n";
  for (auto p: chains) {
    VLOG(2) << p.match_str <<"\n";
  }

  // Now group the same predicate chains
  std::unordered_map<string, std::vector<SwitchChain>> groupByPredicates;
  for (auto p: chains) {
    auto key = p.match_str;
    groupByPredicates[key].push_back(p);
  }

  VLOG(2) << "Grouped predicate chains:\n";
  for (auto g: groupByPredicates) {
    VLOG(2) << (g.first.empty() ? "<empty>" : g.first) << ": ";
    for (SwitchChain p: g.second) {
      for (auto pred: p.predicates) {
        VLOG(2) << pred.first << ":" << pred.second << "-";
      }
      VLOG(2) << " & ";
    }
    VLOG(2) << "\n";
  }

  for (auto g: groupByPredicates) {
    // Pick up the first chain as a representative for rest of them
    SwitchChain firstChain = *g.second.begin();
    if (firstChain.predicates.size() <= 1) continue;
    auto thisDevice = firstChain.switches[0].first->device();
    bool first_pred = true;
    string prevPredicate = {};
    bool prevPredValue = true;
    for (auto pred: firstChain.predicates) {
      auto thisPredicate = pred.first;
      auto thisPredValue = pred.second;
      if (!thisPredValue) {
        string node_name = thisPrefix + "not_" + thisPredicate;
        if (node_map.GetNode(node_name) != nullptr) {
          thisPredicate = node_name;
        }
        else {
          // Not found in in exist nodes, create a logical not node
          NodeDef* not_op = optimized_graph->add_node();
          not_op->set_op("LogicalNot");
          not_op->add_input(thisPredicate);
          not_op->set_name(node_name);
          not_op->set_device(thisDevice);
          // Not sure if need to set _dtype, as the default LogicalNot does not
          //(*not_op->mutable_attr())["_dtype"].set_type(DT_BOOL);
          tensorflow::TensorShapeProto* shape =
          (*not_op->mutable_attr())["_output_shapes"]
              .mutable_list()
              ->add_shape();
          shape->set_unknown_rank(true);
          node_map.AddNode(node_name, not_op);
          node_map.AddOutput(thisPredicate, node_name);
          thisPredicate = node_name;
        }
      }
      if (!first_pred) {
        string node_name = thisPrefix + prevPredicate + "_and_" + thisPredicate;
        if (node_map.GetNode(node_name) != nullptr) {
          thisPredicate = node_name;
        }
        else {
          // Not found in in exist nodes, create a logical and node
          NodeDef* and_op = optimized_graph->add_node();
          and_op->set_op("LogicalAnd");
          and_op->add_input(prevPredicate);
          and_op->add_input(thisPredicate);
          and_op->set_name(node_name);
          and_op->set_device(thisDevice);
          // Not sure if need to set _dtype, as the default LogicalAnd does not
          // (*and_op->mutable_attr())["_dtype"].set_type(DT_BOOL);
          tensorflow::TensorShapeProto* shape =
          (*and_op->mutable_attr())["_output_shapes"]
              .mutable_list()
              ->add_shape();
          shape->set_unknown_rank(true);
          node_map.AddNode(node_name, and_op);
          node_map.AddOutput(thisPredicate, node_name);
          thisPredicate = node_name;
        }
      }
      first_pred = false;
      prevPredicate = thisPredicate;
      prevPredValue = thisPredValue;
    }

    if (prevPredicate.empty()) continue;
    for (SwitchChain p: g.second) {
      NodeDef * lastSwitch = (*(p.switches.rbegin())).first;
      VLOG(0) << "Last switch before change: \n";
      VLOG(0) << lastSwitch->DebugString();
      auto old_input0 = lastSwitch->input(0);
      auto old_input1 = lastSwitch->input(1);
      lastSwitch->set_input(0, p.producer->name());
      lastSwitch->set_input(1, prevPredicate);
      node_map.UpdateInput(lastSwitch->name(), old_input0, p.producer->name());
      node_map.UpdateInput(lastSwitch->name(), old_input1, prevPredicate);
      if (prevPredValue == false) {
        int consumer_input_size = p.consumer->input().size();
        int pos;
        string lastSwitchName = ParseNodeName(lastSwitch->name(), &pos);
        for (int i = 0; i< consumer_input_size; i++) {
          if (lastSwitchName == p.consumer->input(i)) {
            if (pos != 0) {
              return Status(absl::StatusCode::kNotFound, "consumer of the last switch is not using the false branch");
            }
            p.consumer->set_input(i, lastSwitchName + ":1");
            node_map.UpdateInput(lastSwitch->name(), lastSwitchName, lastSwitchName + ":1");
          }
        }
      }
      VLOG(0) << "Last switch after change: \n";
      VLOG(0) << lastSwitch->DebugString();
    }
  }
  // VLOG(3) << "Optimized graph =\n" << optimized_graph->DebugString();
  return absl::OkStatus();
}

}  // end namespace grappler
}  // namespace tensorflow
