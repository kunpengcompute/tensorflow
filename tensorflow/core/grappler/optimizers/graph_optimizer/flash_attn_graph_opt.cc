#include "flash_attn_graph_opt.h"

using namespace tensorflow;
using namespace tensorflow::grappler;

namespace annc {

constexpr const char* kFwdSuffix = "_flash_attn_fwd";

bool IsControlInput(const std::string& in) {
  return !in.empty() && in[0] == '^';
}

// Remove the control‑dependency prefix ^ and port number :N
std::string GetPureNodeName(const std::string& in) {
  std::string name = in;
  if (!name.empty() && name[0] == '^') name = name.substr(1);
  auto pos = name.find(':');
  if (pos != std::string::npos) name = name.substr(0, pos);
  return name;
}

NodeDef* FindNodeByName(GraphDef* graph, const std::string& name) {
  for (NodeDef& n : *graph->mutable_node()) {
    if (n.name() == name) return &n;
  }
  return nullptr;
}

// Count the number of data consumers of the node
int CountDataConsumers(const GraphDef& graph,
                       const std::string& node_name) {
  int count = 0;
  for (const NodeDef& n : graph.node()) {
    for (const std::string& in : n.input()) {
      if (IsControlInput(in)) continue;
      if (GetPureNodeName(in) == node_name) {
        ++count;
        break;
      }
    }
  }
  return count;
}

// Get pointers to all data consumer nodes of the node
std::vector<NodeDef*> GetDataFanout(GraphDef* graph,
                                    const std::string& node_name) {
  std::vector<NodeDef*> fanouts;
  for (NodeDef& n : *graph->mutable_node()) {
    for (const std::string& in : n.input()) {
      if (IsControlInput(in)) continue;
      if (GetPureNodeName(in) == node_name) {
        fanouts.push_back(&n);
        break;
      }
    }
  }
  return fanouts;
}

std::string OutputStr(const std::string& name, int port) {
  return absl::StrCat(name, ":", port);
}

void ReplaceAllNodeReferences(GraphDef* graph,
                              const std::string& old_node_name,
                              const std::string& new_output_str) {
  for (NodeDef& n : *graph->mutable_node()) {
    for (int i = 0; i < n.input_size(); ++i) {
      const std::string& in = n.input(i);
      if (IsControlInput(in)) continue;
      if (GetPureNodeName(in) == old_node_name) {
        n.set_input(i, new_output_str);
      }
    }
  }
}

bool IsDtypeAllowed(DataType dtype) {
  return dtype == DT_FLOAT || dtype == DT_HALF || dtype == DT_BFLOAT16;
}


struct FwdCandidate {
  NodeDef* bmm_qk = nullptr;
  NodeDef* scale_mul = nullptr;   // Mul(BatchMatMulV2, scale_const)
  NodeDef* add_mask = nullptr;
  NodeDef* softmax = nullptr;
  NodeDef* bmm_out = nullptr;
  std::string mask_input;
};

bool MatchFwd(GraphDef* graph, NodeDef* softmax_node, FwdCandidate* c) {
  c->softmax = softmax_node;

  // 1. The data input of Softmax must be AddV2
  NodeDef* add_mask = nullptr;
  for (const std::string& in : softmax_node->input()) {
    if (IsControlInput(in)) continue;
    NodeDef* n = FindNodeByName(graph, GetPureNodeName(in));
    if (n != nullptr && n->op() == "AddV2") {
      add_mask = n;
      break;
    }
  }
  if (add_mask == nullptr || add_mask->input_size() != 2) return false;
  c->add_mask = add_mask;

  // 2. AddV2 inputs: Mul(scale_mul) and mask
  // (5-op graph: BMM -> Mul(scale) -> AddV2 -> Softmax -> BMM)
  NodeDef* scale_mul = nullptr;
  std::string mask_input;
  for (const std::string& in : add_mask->input()) {
    if (IsControlInput(in)) continue;
    NodeDef* n = FindNodeByName(graph, GetPureNodeName(in));
    if (n != nullptr && n->op() == "Mul") {
      scale_mul = n;
    } else {
      mask_input = in;
    }
  }
  if (scale_mul == nullptr || mask_input.empty()) return false;
  if (scale_mul->input_size() != 2) return false;
  c->scale_mul = scale_mul;
  c->mask_input = mask_input;
 
  // 3. Mul inputs: BatchMatMulV2(bmm_qk) and Const(scale)
  NodeDef* bmm_qk = nullptr;
  for (const std::string& in : scale_mul->input()) {
    if (IsControlInput(in)) continue;
    NodeDef* n = FindNodeByName(graph, GetPureNodeName(in));
    if (n != nullptr && n->op() == "BatchMatMulV2") {
      bmm_qk = n;
      break;
    }
  }
  if (bmm_qk == nullptr) return false;
  if (bmm_qk->input_size() < 2) return false;
  c->bmm_qk = bmm_qk;

  // 4. Softmax has only one data consumer: BatchMatMulV2(bmm_out)
  if (CountDataConsumers(*graph, softmax_node->name()) != 1) return false;
  auto sm_fanouts = GetDataFanout(graph, softmax_node->name());
  if (sm_fanouts.size() != 1 || sm_fanouts[0]->op() != "BatchMatMulV2")
    return false;
  c->bmm_out = sm_fanouts[0];

  // 5. intermediate node has only one data consumer
  if (CountDataConsumers(*graph, bmm_qk->name()) != 1) return false;
  if (CountDataConsumers(*graph, scale_mul->name()) != 1) return false;
  if (CountDataConsumers(*graph, add_mask->name()) != 1) return false;

  // 6. device consistency
  const std::string& device = bmm_qk->device();
  if (scale_mul->device() != device ||
      add_mask->device() != device ||
      softmax_node->device() != device ||
      c->bmm_out->device() != device) {
    return false;
  }

  // 7. dtype is consistent and within the whitelist
  if (!HasNodeAttr(*bmm_qk, "T") || !HasNodeAttr(*scale_mul, "T") ||
      !HasNodeAttr(*add_mask, "T") ||
      !HasNodeAttr(*softmax_node, "T") || !HasNodeAttr(*c->bmm_out, "T"))
    return false;
  DataType dtype = bmm_qk->attr().at("T").type();
  if (scale_mul->attr().at("T").type() != dtype ||
      add_mask->attr().at("T").type() != dtype ||
      softmax_node->attr().at("T").type() != dtype ||
      c->bmm_out->attr().at("T").type() != dtype) {
    return false;
  }
  if (!IsDtypeAllowed(dtype)) return false;

  // 8. bmm_qk = Q·K^T (adj_y==true), bmm_out = P·V (adj_y==false)
  if (!HasNodeAttr(*bmm_qk, "adj_y") || !bmm_qk->attr().at("adj_y").b())
    return false;
  if (!HasNodeAttr(*c->bmm_out, "adj_y") ||
      c->bmm_out->attr().at("adj_y").b()) {
    return false;
  }

  return true;
}

void ApplyFwd(GraphDef* graph, const FwdCandidate& c) {
  std::string v_input;
  for (const std::string& in : c.bmm_out->input()) {
    if (IsControlInput(in)) continue;
    if (GetPureNodeName(in) != c.softmax->name()) {
      v_input = in;
      break;
    }
  }

  NodeDef* fused = graph->add_node();
  fused->set_name(absl::StrCat(c.softmax->name(), kFwdSuffix));
  fused->set_op("FlashAttentionForward");
  fused->set_device(c.bmm_qk->device());
  fused->add_input(c.bmm_qk->input(0));  // q
  fused->add_input(c.bmm_qk->input(1));  // k
  fused->add_input(v_input);             // v
  fused->add_input(c.mask_input);        // mask
  (*fused->mutable_attr())["T"].set_type(c.bmm_qk->attr().at("T").type());

  absl::flat_hash_set<std::string> ctrl;
  auto collect = [&](const NodeDef* n) {
    for (const std::string& in : n->input()) {
      if (IsControlInput(in)) ctrl.insert(in);
    }
  };
  collect(c.bmm_qk);
  collect(c.scale_mul);
  collect(c.add_mask);
  collect(c.softmax);
  collect(c.bmm_out);
  for (const std::string& d : ctrl) fused->add_input(d);

  // bmm_out:0 -> fused:0
  ReplaceAllNodeReferences(graph, c.bmm_out->name(),
                           OutputStr(fused->name(), 0));
  VLOG(0) << "-- Add node: [" << fused->op() << "] " << fused->name();
}

void run_flash_attn_optimization(GraphDef* graph) {

  const char* annc_fused_all = getenv("ANNC_FUSED_ALL");
  const char* annc_fused_flashattn_fwd = getenv("ANNC_FUSED_FLASHATTN_FWD");

  bool enable_all = (annc_fused_all != nullptr) && strcmp(annc_fused_all, "1") == 0;
  bool enable_flashattn_fwd = (annc_fused_flashattn_fwd != nullptr) && strcmp(annc_fused_flashattn_fwd, "1") == 0;

  if (graph == nullptr) return;

  if(enable_all || enable_flashattn_fwd) {
    std::vector<FwdCandidate> candidates;
    absl::flat_hash_set<std::string> processed;
    for (NodeDef& node : *graph->mutable_node()) {
      if (node.op() != "Softmax") continue;
      if (processed.contains(node.name())) continue;
      FwdCandidate c;
      if (MatchFwd(graph, &node, &c)) {
        candidates.push_back(c);
        processed.insert(c.bmm_qk->name());
        processed.insert(c.add_mask->name());
        processed.insert(c.softmax->name());
        processed.insert(c.bmm_out->name());
      }
    }
    for (const auto& c : candidates) ApplyFwd(graph, c);
    VLOG(1) << "FlashAttention forward fusion fused " << candidates.size()
            << " attention subgraphs.";
  }
}

}  // namespace annc
