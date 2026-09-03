// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// A client sending requests to server for performance testing.

#include <gflags/gflags.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <butil/logging.h>
#include <butil/time.h>
#include <brpc/channel.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>
#include "dummy.pb.h"

DEFINE_string(attachment, "", "Carry this along with requests");
DEFINE_string(protocol, "baidu_std", "Protocol type. Defined in src/brpc/options.proto");
DEFINE_string(connection_type, "", "Connection type. Available values: single, pooled, short");
DEFINE_string(server, "0.0.0.0:8000", "IP Address of server");
DEFINE_string(load_balancer, "", "The algorithm for load balancing");
DEFINE_int32(timeout_ms, 1000, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");
DEFINE_bool(enable_checksum, false, "Enable checksum or not");
DEFINE_int32(thread_num, 1, "Load test thread number");
DEFINE_int32(test_duration_s, 5, "Load test duration in seconds");
DEFINE_double(max_qps, 0, "Max target QPS (0 means unlimited)");
DEFINE_int32(warmup_duration_s, 1, "Warmup duration in seconds before actual test");
DEFINE_string(input_data_path, "", "Benchmark TSV path. Expected columns: id<TAB>model_input");
DEFINE_bool(input_has_header, true, "Whether input_data_path has a header line");
DEFINE_int32(request_min_batch_size, 1, "Minimum number of benchmark inputs per RPC request");
DEFINE_int32(request_max_batch_size, 1, "Maximum number of benchmark inputs per RPC request");
DEFINE_int32(max_inflight, 0,
             "Maximum in-flight RPC requests. 0 means derive from max_qps and timeout_ms, "
             "or thread_num when max_qps is unlimited");
DEFINE_bool(log_each_latency, false, "Log every completed RPC latency");

struct ThreadStats {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> success{0};
    std::atomic<uint64_t> failure{0};
    std::atomic<uint64_t> dropped{0};
    std::vector<int64_t> latencies_us;
    mutable std::mutex latencies_mutex;  // 保护 latencies_us 的并发访问

    // 禁用拷贝，允许移动
    ThreadStats() = default;
    ThreadStats(const ThreadStats&) = delete;
    ThreadStats& operator=(const ThreadStats&) = delete;
    ThreadStats(ThreadStats&&) = default;
    ThreadStats& operator=(ThreadStats&&) = default;
};

// Real-time stats for monitoring (updated periodically by workers)
struct RealtimeStats {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> success{0};
    std::atomic<uint64_t> failure{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> latency_sum_us{0};  // For calculating avg latency
};

std::string TrimCarriageReturn(std::string value) {
    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    return value;
}

std::vector<std::string> LoadBenchmarkInputs(const std::string& path,
                                             bool has_header) {
    std::vector<std::string> inputs;
    if (path.empty()) {
        inputs.emplace_back("hello world");
        return inputs;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open input_data_path: " << path;
        return inputs;
    }

    std::string line;
    if (has_header) {
        std::getline(file, line);
    }

    uint64_t line_no = has_header ? 1 : 0;
    while (std::getline(file, line)) {
        ++line_no;
        line = TrimCarriageReturn(line);
        if (line.empty()) {
            continue;
        }

        const size_t tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) {
            LOG(WARNING) << "Skip malformed input line " << line_no
                         << ": missing tab separator";
            continue;
        }

        std::string model_input = line.substr(tab_pos + 1);
        if (model_input.empty()) {
            LOG(WARNING) << "Skip malformed input line " << line_no
                         << ": empty model_input";
            continue;
        }
        inputs.emplace_back(std::move(model_input));
    }

    LOG(INFO) << "Loaded " << inputs.size()
              << " benchmark inputs from " << path;
    return inputs;
}

std::string BuildRequestMessage(const std::vector<std::string>& messages,
                                std::atomic<uint64_t>* next_message_index,
                                int batch_size) {
    std::string request_message;
    for (int i = 0; i < batch_size; ++i) {
        const uint64_t message_index =
            next_message_index->fetch_add(1, std::memory_order_relaxed);
        if (!request_message.empty()) {
            request_message.push_back('\n');
        }
        request_message.append(messages[message_index % messages.size()]);
    }
    return request_message;
}

// Helper to calculate P99 from thread stats
int64_t CalculateP99(const std::vector<ThreadStats>& stats) {
    std::vector<int64_t> all_latencies;
    for (const auto& stat : stats) {
        std::lock_guard<std::mutex> lock(stat.latencies_mutex);
        all_latencies.insert(all_latencies.end(),
                             stat.latencies_us.begin(),
                             stat.latencies_us.end());
    }
    if (all_latencies.empty()) {
        return 0;
    }
    std::sort(all_latencies.begin(), all_latencies.end());
    const size_t p99_index = static_cast<size_t>(
        std::ceil(all_latencies.size() * 0.99)) - 1;
    return all_latencies[std::min(p99_index, all_latencies.size() - 1)];
}

// Benchmark report
struct BenchmarkSummary {
    uint64_t total_requests = 0;
    uint64_t total_success = 0;
    uint64_t total_failure = 0;
    uint64_t total_dropped = 0;
    double avg_latency_us = 0.0;
    int64_t p99_latency_us = 0;
};

BenchmarkSummary BuildBenchmarkSummary(const std::vector<ThreadStats>& stats) {
    BenchmarkSummary summary;
    std::vector<int64_t> all_latencies;

    for (const auto& stat : stats) {
        summary.total_requests += stat.total.load(std::memory_order_relaxed);
        summary.total_success += stat.success.load(std::memory_order_relaxed);
        summary.total_failure += stat.failure.load(std::memory_order_relaxed);
        summary.total_dropped += stat.dropped.load(std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(stat.latencies_mutex);
        all_latencies.insert(all_latencies.end(),
                             stat.latencies_us.begin(),
                             stat.latencies_us.end());
    }

    if (all_latencies.empty()) {
        return summary;
    }

    const int64_t total_latency = std::accumulate(
        all_latencies.begin(), all_latencies.end(), static_cast<int64_t>(0));
    summary.avg_latency_us = static_cast<double>(total_latency) /
                             static_cast<double>(all_latencies.size());

    std::sort(all_latencies.begin(), all_latencies.end());
    const size_t p99_index = static_cast<size_t>(
        std::ceil(all_latencies.size() * 0.99)) - 1;
    summary.p99_latency_us = all_latencies[std::min(p99_index, all_latencies.size() - 1)];
    return summary;
}

void ReportBenchmarkStats(const std::vector<ThreadStats>& stats,
                          std::chrono::steady_clock::time_point start_time,
                          double target_duration_s) {
    const BenchmarkSummary summary = BuildBenchmarkSummary(stats);
    const auto actual_duration_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    const double actual_success_qps = (actual_duration_s > 0)
        ? static_cast<double>(summary.total_success) / actual_duration_s
        : 0.0;
    const double actual_request_qps = (actual_duration_s > 0)
        ? static_cast<double>(summary.total_requests) / actual_duration_s
        : 0.0;
    const double target_window_request_qps = (target_duration_s > 0)
        ? static_cast<double>(summary.total_requests) / target_duration_s
        : 0.0;

    LOG(INFO) << "Load test finished.\n"
              << "----------------------------\n"
              << "duration_s                : " << actual_duration_s << "\n"
              << "target_duration_s         : " << target_duration_s << "\n"
              << "total                     : " << summary.total_requests << "\n"
              << "success                   : " << summary.total_success << "\n"
              << "failure                   : " << summary.total_failure << "\n"
              << "dropped                   : " << summary.total_dropped << "\n"
              << "avg_latency_us            : " << summary.avg_latency_us << "\n"
              << "p99_latency_us            : " << summary.p99_latency_us << "\n"
              << "actual_success_qps        : " << actual_success_qps << "\n"
              << "actual_request_qps        : " << actual_request_qps << "\n"
              << "target_window_request_qps : " << target_window_request_qps << "\n"
              << "qps                       : " << actual_success_qps << "\n"
              << "----------------------------";
}


int ResolveMaxInflight(double max_qps, int thread_num) {
    if (FLAGS_max_inflight > 0) {
        return FLAGS_max_inflight;
    }
    if (max_qps > 0 && FLAGS_timeout_ms > 0) {
        return std::max(1, static_cast<int>(
            std::ceil(max_qps * static_cast<double>(FLAGS_timeout_ms) / 1000.0)));
    }
    return std::max(1, thread_num);
}

bool TryReserveInflight(std::atomic<int>* inflight, int max_inflight) {
    int current = inflight->load(std::memory_order_relaxed);
    while (current < max_inflight) {
        if (inflight->compare_exchange_weak(
                current, current + 1, std::memory_order_acquire,
                std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void RecordDropped(ThreadStats* stat, RealtimeStats* rt_stats) {
    if (rt_stats != nullptr) {
        rt_stats->total.fetch_add(1, std::memory_order_relaxed);
        rt_stats->failure.fetch_add(1, std::memory_order_relaxed);
        rt_stats->dropped.fetch_add(1, std::memory_order_relaxed);
    }
    if (stat != nullptr) {
        stat->total.fetch_add(1, std::memory_order_relaxed);
        stat->failure.fetch_add(1, std::memory_order_relaxed);
        stat->dropped.fetch_add(1, std::memory_order_relaxed);
    }
}

void HandleEchoResponse(brpc::Controller* cntl,
                        example::EchoRequest* request,
                        example::EchoResponse* response,
                        ThreadStats* stat,
                        RealtimeStats* rt_stats,
                        std::atomic<int>* inflight,
                        bool record_stats) {
    std::unique_ptr<brpc::Controller> cntl_guard(cntl);
    std::unique_ptr<example::EchoRequest> request_guard(request);
    std::unique_ptr<example::EchoResponse> response_guard(response);

    if (record_stats) {
        const int64_t latency_us = cntl->latency_us();
        if (FLAGS_log_each_latency) {
            LOG(INFO) << "CLIENT_LATENCY log_id=" << cntl->log_id()
                      << " latency_us=" << latency_us
                      << " failed=" << cntl->Failed()
                      << " error_text=" << cntl->ErrorText();
        }
        rt_stats->total.fetch_add(1, std::memory_order_relaxed);
        rt_stats->latency_sum_us.fetch_add(latency_us, std::memory_order_relaxed);
        stat->total.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(stat->latencies_mutex);
            stat->latencies_us.push_back(latency_us);
        }
        if (!cntl->Failed()) {
            rt_stats->success.fetch_add(1, std::memory_order_relaxed);
            stat->success.fetch_add(1, std::memory_order_relaxed);
        } else {
            rt_stats->failure.fetch_add(1, std::memory_order_relaxed);
            stat->failure.fetch_add(1, std::memory_order_relaxed);
        }
    }
    inflight->fetch_sub(1, std::memory_order_release);
}

class EchoDone : public google::protobuf::Closure {
public:
    EchoDone(brpc::Controller* cntl,
             example::EchoRequest* request,
             example::EchoResponse* response,
             ThreadStats* stat,
             RealtimeStats* rt_stats,
             std::atomic<int>* inflight,
             bool record_stats)
        : cntl_(cntl),
          request_(request),
          response_(response),
          stat_(stat),
          rt_stats_(rt_stats),
          inflight_(inflight),
          record_stats_(record_stats) {}

    void Run() override {
        HandleEchoResponse(cntl_, request_, response_, stat_, rt_stats_,
                           inflight_, record_stats_);
        delete this;
    }

private:
    brpc::Controller* cntl_;
    example::EchoRequest* request_;
    example::EchoResponse* response_;
    ThreadStats* stat_;
    RealtimeStats* rt_stats_;
    std::atomic<int>* inflight_;
    bool record_stats_;
};

void RunEchoPhase(example::EchoService_Stub* stub,
                  int duration_s,
                  int thread_num,
                  const std::vector<std::string>& messages,
                  double max_qps,
                  bool record_stats,
                  std::vector<ThreadStats>* stats) {
    if (record_stats && (stats == nullptr || stats->size() < static_cast<size_t>(thread_num))) {
        LOG(ERROR) << "stats must be initialized when record_stats is enabled";
        return;
    }

    std::atomic<int> log_id{0};
    std::atomic<uint64_t> next_message_index{0};
    std::atomic<uint64_t> next_rpc_ticket{0};
    std::atomic<int> inflight{0};
    RealtimeStats rt_stats;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(thread_num));

    const auto start_time = std::chrono::steady_clock::now();
    const auto end_time = start_time + std::chrono::seconds(duration_s);
    const int max_inflight = ResolveMaxInflight(max_qps, thread_num);
    LOG(INFO) << (record_stats ? "Load test" : "Warmup")
              << " RPC schedule: max_qps=" << max_qps
              << ", max_inflight=" << max_inflight
              << ", thread_num=" << thread_num;

    std::thread monitor_thread;
    if (record_stats) {
        monitor_thread = std::thread([&, start_time, end_time, duration_s, max_inflight]() {
            uint64_t last_total = 0;
            auto last_time = start_time;
            bool first_output = true;
            const int num_lines = 10;

            while (!brpc::IsAskedToQuit() && std::chrono::steady_clock::now() < end_time) {
                std::this_thread::sleep_for(std::chrono::seconds(1));

                const auto now = std::chrono::steady_clock::now();
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                const uint64_t current_total = rt_stats.total.load(std::memory_order_relaxed);
                const uint64_t current_success = rt_stats.success.load(std::memory_order_relaxed);
                const uint64_t current_failure = rt_stats.failure.load(std::memory_order_relaxed);
                const uint64_t current_dropped = rt_stats.dropped.load(std::memory_order_relaxed);
                const uint64_t latency_sum = rt_stats.latency_sum_us.load(std::memory_order_relaxed);
                const int current_inflight = inflight.load(std::memory_order_relaxed);

                const auto time_diff = std::chrono::duration<double>(now - last_time).count();
                const uint64_t req_diff = current_total - last_total;
                const double qps = (time_diff > 0) ? static_cast<double>(req_diff) / time_diff : 0.0;

                const uint64_t completed = current_success + current_failure - current_dropped;
                const double avg_latency = (completed > 0)
                    ? static_cast<double>(latency_sum) / static_cast<double>(completed)
                    : 0.0;

                const int64_t p99_latency = CalculateP99(*stats);
                const int remaining = duration_s - static_cast<int>(elapsed);
                const int display_elapsed = std::min(static_cast<int>(elapsed), duration_s);
                const int display_remaining = std::max(remaining, 0);

                if (!first_output) {
                    std::fprintf(stderr, "\033[%dA\033[0G", num_lines);
                }
                first_output = false;

                std::fprintf(stderr, "========== Progress [%d/%d seconds] ==========\n", display_elapsed, duration_s);
                std::fprintf(stderr, "Requests:     %lu   \n", current_total);
                std::fprintf(stderr, "QPS:          %.1f   \n", qps);
                std::fprintf(stderr, "Inflight:     %d/%d   \n", current_inflight, max_inflight);
                std::fprintf(stderr, "Success:      %lu   \n", current_success);
                std::fprintf(stderr, "Failure:      %lu   \n", current_failure);
                std::fprintf(stderr, "Dropped:      %lu   \n", current_dropped);
                std::fprintf(stderr, "Avg Latency:  %.1f us   \n", avg_latency);
                std::fprintf(stderr, "P99 Latency:  %ld us   \n", p99_latency);
                std::fprintf(stderr, "Remaining:    %d seconds   \n", display_remaining);
                std::fflush(stderr);

                last_total = current_total;
                last_time = now;
            }

            std::fprintf(stderr, "\n");
        });
    }

    for (int i = 0; i < thread_num; ++i) {
        workers.emplace_back([&, i]() {
            std::mt19937 rng(static_cast<uint32_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
                static_cast<uint32_t>(i));
            std::uniform_int_distribution<int> batch_dist(
                FLAGS_request_min_batch_size, FLAGS_request_max_batch_size);
            while (!brpc::IsAskedToQuit() && std::chrono::steady_clock::now() < end_time) {
                if (max_qps > 0) {
                    const uint64_t ticket = next_rpc_ticket.fetch_add(1, std::memory_order_relaxed);
                    const auto scheduled_time = start_time +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(static_cast<double>(ticket) / max_qps));
                    if (scheduled_time >= end_time) {
                        break;
                    }
                    std::this_thread::sleep_until(scheduled_time);
                    if (std::chrono::steady_clock::now() >= end_time) {
                        break;
                    }
                }

                ThreadStats* stat = record_stats ? &(*stats)[i] : nullptr;
                if (!TryReserveInflight(&inflight, max_inflight)) {
                    if (record_stats) {
                        RecordDropped(stat, &rt_stats);
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    continue;
                }

                auto* request = new example::EchoRequest;
                auto* response = new example::EchoResponse;
                auto* cntl = new brpc::Controller;

                const int batch_size = batch_dist(rng);
                request->set_message(BuildRequestMessage(
                    messages, &next_message_index, batch_size));
                cntl->set_log_id(log_id.fetch_add(1, std::memory_order_relaxed));
                if (!FLAGS_attachment.empty()) {
                    cntl->request_attachment().append(FLAGS_attachment);
                }
                if (FLAGS_enable_checksum) {
                    cntl->set_request_checksum_type(brpc::CHECKSUM_TYPE_CRC32C);
                }

                stub->Echo(cntl, request, response,
                           new EchoDone(cntl, request, response, stat,
                                        &rt_stats, &inflight, record_stats));
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    while (inflight.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (record_stats && monitor_thread.joinable()) {
        monitor_thread.join();
    }
}

int main(int argc, char* argv[]) {
    // Parse gflags. We recommend you to use gflags as well.
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_thread_num <= 0 || FLAGS_test_duration_s <= 0) {
        LOG(ERROR) << "thread_num and test_duration_s must be greater than 0";
        return -1;
    }
    if (FLAGS_request_min_batch_size <= 0 ||
        FLAGS_request_max_batch_size < FLAGS_request_min_batch_size) {
        LOG(ERROR) << "request_min_batch_size must be > 0 and "
                   << "request_max_batch_size must be >= request_min_batch_size";
        return -1;
    }
    if (FLAGS_max_qps < 0 || FLAGS_max_inflight < 0) {
        LOG(ERROR) << "max_qps and max_inflight must be >= 0";
        return -1;
    }

    const std::vector<std::string> benchmark_inputs =
        LoadBenchmarkInputs(FLAGS_input_data_path, FLAGS_input_has_header);
    if (benchmark_inputs.empty()) {
        LOG(ERROR) << "No benchmark input loaded";
        return -1;
    }

    // A Channel represents a communication line to a Server. Notice that
    // Channel is thread-safe and can be shared by all threads in your program.
    brpc::Channel channel;

    // Initialize the channel, NULL means using default options.
    brpc::ChannelOptions options;
    options.protocol = FLAGS_protocol;
    options.connection_type = FLAGS_connection_type;
    options.timeout_ms = FLAGS_timeout_ms/*milliseconds*/;
    options.max_retry = FLAGS_max_retry;

    if (channel.Init(FLAGS_server.c_str(), FLAGS_load_balancer.c_str(), &options) != 0) {
        LOG(ERROR) << "Fail to initialize channel";
        return -1;
    }

    // Normally, you should not call a Channel directly, but instead construct
    // a stub Service wrapping it. stub can be shared by all threads as well.
    example::EchoService_Stub stub(&channel);

    if (FLAGS_thread_num <= 0) {
        LOG(ERROR) << "thread_num must be greater than 0";
        return -1;
    }

     if (FLAGS_test_duration_s <= 0) {
        LOG(ERROR) << "test_duration_s must be greater than 0";
        return -1;
    }

    // --- Phase 1: Warmup ---
    if (FLAGS_warmup_duration_s > 0) { // 已重构代码
        LOG(INFO) << "Starting Warmup (" << FLAGS_warmup_duration_s << "s)...";
        RunEchoPhase(&stub,
                     FLAGS_warmup_duration_s,
                     FLAGS_thread_num,
                     benchmark_inputs,
                     FLAGS_max_qps,
                     false,
                     nullptr);

        LOG(INFO) << "Warmup finished. Waiting 1s before main test...";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // --- Phase 2: Main Test ---
    LOG(INFO) << "Starting Load Test (" << FLAGS_test_duration_s << "s)...";
    std::vector<ThreadStats> stats(static_cast<size_t>(FLAGS_thread_num));

    const auto start_time = std::chrono::steady_clock::now();
    RunEchoPhase(&stub,
                 FLAGS_test_duration_s,
                 FLAGS_thread_num,
                 benchmark_inputs,
                 FLAGS_max_qps,
                 true,
                 &stats);

    // --- Statistics Aggregation ---
    ReportBenchmarkStats(stats, start_time, FLAGS_test_duration_s);
    return 0;
}
