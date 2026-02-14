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
#include <numeric>
#include <thread>
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

struct ThreadStats {
    uint64_t total = 0;
    uint64_t success = 0;
    uint64_t failure = 0;
    std::vector<int64_t> latencies_us;
};

void RunEchoPhase(example::EchoService_Stub* stub,
                  int duration_s,
                  int thread_num,
                  const std::string& message,
                  double max_qps,
                  bool record_stats,
                  std::vector<ThreadStats>* stats) {
    if (record_stats && (stats == nullptr || stats->size() < static_cast<size_t>(thread_num))) {
        LOG(ERROR) << "stats must be initialized when record_stats is enabled";
        return;
    }

    std::atomic<int> log_id{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(thread_num));

    const auto end_time = std::chrono::steady_clock::now() +
                          std::chrono::seconds(duration_s);
    const double per_thread_qps = (max_qps > 0)
        ? max_qps / static_cast<double>(thread_num)
        : 0.0;
    const auto interval = (per_thread_qps > 0)
        ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / per_thread_qps))
        : std::chrono::steady_clock::duration::zero();

    for (int i = 0; i < thread_num; ++i) {
        workers.emplace_back([&, i]() {
            auto next_send = std::chrono::steady_clock::now();
            while (!brpc::IsAskedToQuit() && std::chrono::steady_clock::now() < end_time) {
                if (interval != std::chrono::steady_clock::duration::zero()) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now < next_send) {
                        std::this_thread::sleep_until(next_send);
                    }
                    next_send = std::chrono::steady_clock::now() + interval;
                }

                example::EchoRequest request;
                example::EchoResponse response;
                brpc::Controller cntl;

                request.set_message(message);
                cntl.set_log_id(log_id.fetch_add(1, std::memory_order_relaxed));
                if (!FLAGS_attachment.empty()) {
                    cntl.request_attachment().append(FLAGS_attachment);
                }
                if (FLAGS_enable_checksum) {
                    cntl.set_request_checksum_type(brpc::CHECKSUM_TYPE_CRC32C);
                }

                stub->Echo(&cntl, &request, &response, nullptr);
                if (!record_stats) {
                    continue;
                }

                (*stats)[i].total += 1;
                (*stats)[i].latencies_us.push_back(cntl.latency_us());
                if (!cntl.Failed()) {
                    (*stats)[i].success += 1;
                } else {
                    (*stats)[i].failure += 1;
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
}

int main(int argc, char* argv[]) {
    // Parse gflags. We recommend you to use gflags as well.
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_thread_num <= 0 || FLAGS_test_duration_s <= 0) {
        LOG(ERROR) << "thread_num and test_duration_s must be greater than 0";
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
                     "warmup",
                     0.0,
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
                 "hello world",
                 FLAGS_max_qps,
                 true,
                 &stats);
    
    // --- Statistics Aggregation ---
    uint64_t total_requests = 0;
    uint64_t total_success = 0;
    uint64_t total_failure = 0;
    std::vector<int64_t> all_latencies;
    for (const auto& stat : stats) {
        total_requests += stat.total;
        total_success += stat.success;
        total_failure += stat.failure;
        all_latencies.insert(all_latencies.end(),
                             stat.latencies_us.begin(),
                             stat.latencies_us.end());
    }

    double avg_latency = 0.0;
    int64_t p99_latency = 0;
    if (!all_latencies.empty()) {
        const int64_t total_latency = std::accumulate(
            all_latencies.begin(), all_latencies.end(), static_cast<int64_t>(0));
        avg_latency = static_cast<double>(total_latency) /
                      static_cast<double>(all_latencies.size());
        std::sort(all_latencies.begin(), all_latencies.end());
        const size_t p99_index = static_cast<size_t>(
            std::ceil(all_latencies.size() * 0.99)) - 1;
        p99_latency = all_latencies[std::min(p99_index, all_latencies.size() - 1)];
    }

    const auto actual_duration_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    const double actual_qps = (actual_duration_s > 0)
        ? static_cast<double>(total_requests) / actual_duration_s
        : 0.0;

    LOG(INFO) << "Load test finished.\n"
              << "----------------------------\n"
              << "duration_s     : " << actual_duration_s << "\n"
              << "total          : " << total_requests << "\n"
              << "success        : " << total_success << "\n"
              << "failure        : " << total_failure << "\n"
              << "avg_latency_us : " << avg_latency << "\n"
              << "p99_latency_us : " << p99_latency << "\n"
              << "qps            : " << actual_qps << "\n"
              << "----------------------------";

    return 0;
}
