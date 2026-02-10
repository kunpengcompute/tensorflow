#include <iostream>
#include <sstream>
#include <unordered_set>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/loader_util.h"

#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::Flag;
using tensorflow::string;
using tensorflow::Tensor;

static const size_t kLoadSessionThreadPoolIndex = 0;
static const size_t kRunSessionThreadPoolIndex = 1;

class Tf_Model {
public:
  Tf_Model(const string& model_path) {
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;  // Using 0 index thread pool to load model (contain one thread)
    run_options.set_inter_op_thread_pool(kLoadSessionThreadPoolIndex);
    if (!LoadFromSavedModelBundle(model_path, run_options, session_options, &bundle_)) {
      LOG(ERROR) << "load from SavedModel fail: " << master_model_path;
    }
  }
  ~Tf_Model() {};

  bool Infer() {
    tensorflow::Session* session_ptr = bundle_.GetSession();
    tensorflow::MetaGraphDef& meta_graph = bundle_.meta_graph_def;
    const google::protobuf::Map<string, tensorflow::SignatureDef>& 
        signature_def = bundle_.GetSignatures();
    
    // Session Run Master model
    if (!MockDataSessionRun(signature_def, session_ptr)) {
      LOG(ERROR) << "mock data for session run fail.";
      return false;
    }
    return true;
  }

private:
  bool LoadFromSavedModelBundle(const string& file_path,
                                const tensorflow::RunOptions& run_options,
                                const tensorflow::SessionOptions& session_options,
                                tensorflow::SavedModelBundle* bundle) {
    tensorflow::Status status = tensorflow::LoadSavedModel(session_options, 
                                                           run_options, 
                                                           file_path, 
                                                           {tensorflow::kSavedModelTagServe}, 
                                                           bundle);
    if(!status.ok()) {
      LOG(ERROR) << "Load Session Bundle fail: " << status.ToString();
      return false;
    }
    return true;
  }

  bool MockDataSessionRun(const google::protobuf::Map<string, 
                          tensorflow::SignatureDef>& signature_def,
                          tensorflow::Session* session) {
    for (const auto& io_iter : signature_def) {
      const string& io_key = io_iter.first;
      const tensorflow::SignatureDef& io_def = io_iter.second;
    
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<string> output_node_names;
      std::vector<Tensor> output_tensors;
    
//      // Mock input data
//      for (const auto& input_iter : io_def.inputs()) {
//        std::string input_node_names = input_iter.second.name();
//        Tensor input_tensor(DT_STRING, TensorShape({}));  // Mock input tensor data
//        inputs[input_node_names] = input_tensor;
//      }
      // Get output node name
      for (const auto& output_iter : io_def.outputs()) {
        output_node_names.push_back(output_iter.second.name());
      }
  
      tensorflow::Status run_status =
          session->Run(inputs, output_node_names, {}, &output_tensors);
      if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status.ToString()
                   << ", SignatureDef Key: " << io_key;
        return false;
      }
  
      std::stringstream ss;
      ss << "\nSession Run SignatureDef Key: " << io_key << "\n";
      // Print output tensor using TensorProto
      for (int i = 0; i < output_tensors.size(); ++i) {
        tensorflow::TensorProto output;
        output_tensors[i].AsProtoField(&output);
        ss << "######## " << output_node_names[i] << " ########\n"
           << output.ShortDebugString() << "\n";
      }
      LOG(INFO) << ss.str();
    }
    return true;
  }

private:
  tensorflow::SavedModelBundle bundle_;
};
