#include <iostream>
#include <gflags/gflags.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>
#include <random>
#include <unistd.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::Flag;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::DataType;
using tensorflow::TensorShape;

DECLARE_bool(debug);

static const size_t kLoadSessionThreadPoolIndex = 0;
static const size_t kRunSessionThreadPoolIndex = 1;

// Default batch size for dynamic dimensions (marked as -1 in TensorShape)
static const int64_t kDefaultBatchSize = 1;

// convert TensorShapeProto to TensorShape
TensorShape ConvertToTensorShape(const tensorflow::TensorShapeProto& shape_proto) {
  std::vector<int64_t> dims;
  for (const auto& dim : shape_proto.dim()) {
    int64_t size = dim.size();
    // Replace dynamic dimension (-1) with default batch size
    if (size <= 0) {
      size = kDefaultBatchSize;
    }
    dims.push_back(size);
  }
  return TensorShape(dims);
}

// create a mock tensor with the given dtype and shape
Tensor CreateMockTensor(DataType dtype, const TensorShape& shape) {
  Tensor tensor(dtype, shape);
  
  // Initialize random generator
  static std::random_device rd;
  static std::mt19937 gen(rd());
  
  switch (dtype) {
    case tensorflow::DT_FLOAT: {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      auto flat = tensor.flat<float>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_DOUBLE: {
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      auto flat = tensor.flat<double>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_INT32: {
      std::uniform_int_distribution<int32_t> dist(0, 100);
      auto flat = tensor.flat<int32_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_INT64: {
      std::uniform_int_distribution<int64_t> dist(0, 100);
      auto flat = tensor.flat<int64_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_STRING: {
      auto flat = tensor.flat<tensorflow::tstring>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = "mock_string";
      }
      break;
    }
    case tensorflow::DT_BOOL: {
      std::uniform_int_distribution<int> dist(0, 1);
      auto flat = tensor.flat<bool>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen) == 1;
      }
      break;
    }
    case tensorflow::DT_INT8: {
      std::uniform_int_distribution<int> dist(-128, 127);
      auto flat = tensor.flat<int8_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = static_cast<int8_t>(dist(gen));
      }
      break;
    }
    case tensorflow::DT_UINT8: {
      std::uniform_int_distribution<int> dist(0, 255);
      auto flat = tensor.flat<uint8_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = static_cast<uint8_t>(dist(gen));
      }
      break;
    }
    case tensorflow::DT_INT16: {
      std::uniform_int_distribution<int16_t> dist(-1000, 1000);
      auto flat = tensor.flat<int16_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_UINT16: {
      std::uniform_int_distribution<uint16_t> dist(0, 1000);
      auto flat = tensor.flat<uint16_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_UINT32: {
      std::uniform_int_distribution<uint32_t> dist(0, 100);
      auto flat = tensor.flat<uint32_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    case tensorflow::DT_UINT64: {
      std::uniform_int_distribution<uint64_t> dist(0, 100);
      auto flat = tensor.flat<uint64_t>();
      for (int64_t i = 0; i < flat.size(); ++i) {
        flat(i) = dist(gen);
      }
      break;
    }
    default: {
      LOG(WARNING) << "Unsupported data type for mock tensor: " 
                   << tensorflow::DataTypeString(dtype) 
                   << ", using zero-initialized tensor";
      // Tensor is already zero-initialized by default
      break;
    }
  }
  
  return tensor;
}

class Tf_Model {
public:
  struct TimelineConfig {
    bool enable = false;
    string dir = "/tmp/tf_timeline";
    uint64_t max_dumps = 1;
    bool dump_warmup = false;
    uint64_t every_n = 1;
    bool include_runmeta_pb = false;
    bool log_each_dump = true;
  };

  Tf_Model(const string& model_path, int64_t batch_size = 1)
      : Tf_Model(model_path, batch_size, TimelineConfig{}) {}

  Tf_Model(const string& model_path, int64_t batch_size,
           const TimelineConfig& timeline_config)
      : batch_size_(batch_size),
        timeline_config_(timeline_config),
        request_counter_(0),
        dump_counter_(0) {
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;  // Using 0 index thread pool to load model (contain one thread)
    run_options.set_inter_op_thread_pool(kLoadSessionThreadPoolIndex);
    if (!LoadFromSavedModelBundle(model_path, run_options, session_options, &bundle_)) {
      LOG(ERROR) << "load from SavedModel fail: " << model_path;
    }
    LOG(INFO) << "Model loaded with batch_size=" << batch_size_;
  }
  ~Tf_Model() {};

  bool Infer() {
    const uint64_t request_id = request_counter_.fetch_add(1, std::memory_order_relaxed);
    tensorflow::Session* session_ptr = bundle_.GetSession();
    tensorflow::MetaGraphDef& meta_graph = bundle_.meta_graph_def;
    const google::protobuf::Map<string, tensorflow::SignatureDef>& 
        signature_def = bundle_.GetSignatures();
    
    // Session Run Master model
    if (!MockDataSessionRun(signature_def, session_ptr, request_id)) {
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
                          tensorflow::Session* session,
                          uint64_t request_id) {
    for (const auto& io_iter : signature_def) {
      const string& io_key = io_iter.first;
      const tensorflow::SignatureDef& io_def = io_iter.second;
    
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<string> output_node_names;
      std::vector<Tensor> output_tensors;

      // Mock input data - create tensors based on SignatureDef input specifications
      if (FLAGS_debug) {
        LOG(INFO) << "Preparing inputs for SignatureDef: " << io_key;
      }
      for (const auto& input_iter : io_def.inputs()) {
        const string& input_key = input_iter.first;
        const tensorflow::TensorInfo& tensor_info = input_iter.second;

        // Get tensor name
        string input_tensor_name = tensor_info.name();
        // Get data type
        DataType dtype = tensor_info.dtype();
        // Get tensor shape and convert to TensorShape
        const tensorflow::TensorShapeProto& shape_proto = tensor_info.tensor_shape();
        TensorShape shape = ConvertToTensorShape(shape_proto);
        // Create mock tensor with appropriate data
        Tensor input_tensor = CreateMockTensor(dtype, shape);

        if (FLAGS_debug) {
          LOG(INFO) << "Input '" << input_key << "': name=" << input_tensor_name 
                    << ", dtype=" << tensorflow::DataTypeString(dtype) 
                    << ", shape=" << shape.DebugString();
        }
        
        inputs.emplace_back(input_tensor_name, std::move(input_tensor));
      }
    
      // Get output node name
      for (const auto& output_iter : io_def.outputs()) {
        output_node_names.push_back(output_iter.second.name());
        if (FLAGS_debug) {
          LOG(INFO) << "Output '" << output_iter.first << "': name=" << output_iter.second.name();
        }
      }
  
      tensorflow::Status run_status;
      tensorflow::RunMetadata run_metadata;
      const bool should_collect_timeline = ShouldCollectTimeline(request_id);
      if (should_collect_timeline) {
        tensorflow::RunOptions run_options;
        run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
        run_status = session->Run(
            run_options, inputs, output_node_names, {}, &output_tensors, &run_metadata);
      } else {
        run_status = session->Run(inputs, output_node_names, {}, &output_tensors);
      }
      if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status.ToString()
                   << ", SignatureDef Key: " << io_key;
        return false;
      }
      if (should_collect_timeline) {
        MaybeDumpTimeline(io_key, request_id, run_metadata);
      }
  
      std::stringstream ss;
      if (FLAGS_debug) {
        ss << "\nSession Run SignatureDef Key: " << io_key << "\n";
      }
      // Print output tensor using TensorProto
      for (size_t i = 0; i < output_tensors.size(); ++i) {
        tensorflow::TensorProto output;
        output_tensors[i].AsProtoField(&output);
        if (FLAGS_debug) {
          ss << "######## " << output_node_names[i] << " ########\n"
             << output.ShortDebugString() << "\n";
        }
      }
      if (FLAGS_debug) {
        LOG(INFO) << ss.str();
      }
    }
    return true;
  }

    bool ShouldCollectTimeline(uint64_t request_id) const {
    if (!timeline_config_.enable) {
      return false;
    }
    if (!timeline_config_.dump_warmup && request_id == 0) {
      return false;
    }
    const uint64_t every_n = std::max<uint64_t>(1, timeline_config_.every_n);
    return (request_id % every_n == 0);
  }

  bool ReserveDumpSlot(uint64_t* dump_id) {
    if (!timeline_config_.enable) {
      return false;
    }
    const uint64_t max_dumps = timeline_config_.max_dumps;
    while (true) {
      uint64_t current = dump_counter_.load(std::memory_order_relaxed);
      if (current >= max_dumps) {
        return false;
      }
      if (dump_counter_.compare_exchange_weak(
              current, current + 1, std::memory_order_relaxed)) {
        *dump_id = current;
        return true;
      }
    }
  }

  string SanitizeForFileName(const string& value) const {
    string result = value;
    for (char& c : result) {
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
        c = '_';
      }
    }
    if (result.empty()) {
      return "unknown";
    }
    return result;
  }

  string BuildTraceEventJson(const tensorflow::RunMetadata& run_metadata,
                             const string& signature_key,
                             uint64_t request_id) const {
    std::ostringstream oss;
    oss << "{\n  \"traceEvents\": [\n";
    bool first = true;
    const auto& dev_stats = run_metadata.step_stats().dev_stats();
    for (const auto& dev_stat : dev_stats) {
      for (const auto& node_stat : dev_stat.node_stats()) {
        if (!first) {
          oss << ",\n";
        }
        first = false;
        const int64_t duration_us = std::max<int64_t>(0, node_stat.all_end_rel_micros());
        oss << "    {\"name\":\"" << node_stat.node_name()
            << "\",\"cat\":\"TensorFlow\",\"ph\":\"X\",\"pid\":" << getpid()
            << ",\"tid\":\"" << dev_stat.device() << "\",\"ts\":"
            << node_stat.all_start_micros() << ",\"dur\":" << duration_us
            << ",\"args\":{\"signature\":\"" << signature_key
            << "\",\"request_id\":" << request_id << "}}";
      }
    }
    oss << "\n  ]\n}\n";
    return oss.str();
  }

  bool EnsureDir(const string& dir) const {
    const tensorflow::Status status = tensorflow::Env::Default()->RecursivelyCreateDir(dir);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to create timeline directory: " << dir
                   << ", err=" << status.ToString();
      return false;
    }
    return true;
  }

  bool WriteStringToFile(const string& path, const string& content) const {
    std::unique_ptr<tensorflow::WritableFile> file;
    tensorflow::Status status = tensorflow::Env::Default()->NewWritableFile(path, &file);
    if (!status.ok()) {
      LOG(WARNING) << "NewWritableFile failed, path=" << path
                   << ", err=" << status.ToString();
      return false;
    }
    status = file->Append(content);
    if (!status.ok()) {
      LOG(WARNING) << "Append failed, path=" << path
                   << ", err=" << status.ToString();
      return false;
    }
    status = file->Close();
    if (!status.ok()) {
      LOG(WARNING) << "Close failed, path=" << path
                   << ", err=" << status.ToString();
      return false;
    }
    return true;
  }

  void MaybeDumpTimeline(const string& signature_key, uint64_t request_id,
                         const tensorflow::RunMetadata& run_metadata) {
    uint64_t dump_id = 0;
    if (!ReserveDumpSlot(&dump_id)) {
      return;
    }
    if (!EnsureDir(timeline_config_.dir)) {
      return;
    }

    const uint64_t now_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    std::ostringstream tid_ss;
    tid_ss << std::this_thread::get_id();
    const string safe_signature = SanitizeForFileName(signature_key);
    const string base_name = tensorflow::strings::StrCat(
        "tf_timeline_req_", request_id,
        "_dump_", dump_id,
        "_sig_", safe_signature,
        "_pid_", getpid(),
        "_tid_", tid_ss.str(),
        "_", now_us);

    const string json_path = tensorflow::io::JoinPath(
        timeline_config_.dir, tensorflow::strings::StrCat(base_name, ".json"));
    const string trace_json = BuildTraceEventJson(run_metadata, signature_key, request_id);
    const bool json_ok = WriteStringToFile(json_path, trace_json);

    string pb_path;
    bool pb_ok = false;
    if (timeline_config_.include_runmeta_pb) {
      pb_path = tensorflow::io::JoinPath(
          timeline_config_.dir, tensorflow::strings::StrCat(base_name, ".runmeta.pb"));
      pb_ok = WriteStringToFile(pb_path, run_metadata.SerializeAsString());
    }

    if (timeline_config_.log_each_dump) {
      LOG(INFO) << "TF timeline dump: request_id=" << request_id
                << ", signature_key=" << signature_key
                << ", json_path=" << json_path
                << ", include_runmeta_pb=" << timeline_config_.include_runmeta_pb
                << ", runmeta_pb_path=" << (pb_path.empty() ? "N/A" : pb_path)
                << ", json_ok=" << json_ok
                << ", runmeta_pb_ok=" << pb_ok;
    }
  }

private:
  tensorflow::SavedModelBundle bundle_;
  int64_t batch_size_;
  TimelineConfig timeline_config_;
  std::atomic<uint64_t> request_counter_;
  std::atomic<uint64_t> dump_counter_;
};
