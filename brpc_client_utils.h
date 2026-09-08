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

#ifndef BRPC_CLIENT_UTILS_H_
#define BRPC_CLIENT_UTILS_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace predictor {

struct ThreadStats {
  std::atomic<uint64_t> total{0};
  std::atomic<uint64_t> success{0};
  std::atomic<uint64_t> failure{0};
  std::atomic<uint64_t> dropped{0};
  std::vector<int64_t> latencies_us;
  mutable std::mutex latencies_mutex;

  ThreadStats() = default;
  ThreadStats(const ThreadStats&) = delete;
  ThreadStats& operator=(const ThreadStats&) = delete;
  ThreadStats(ThreadStats&&) = delete;
  ThreadStats& operator=(ThreadStats&&) = delete;
};

struct RealtimeStats {
  std::atomic<uint64_t> total{0};
  std::atomic<uint64_t> success{0};
  std::atomic<uint64_t> failure{0};
  std::atomic<uint64_t> dropped{0};
  std::atomic<uint64_t> latency_sum_us{0};
};

struct BenchmarkSummary {
  uint64_t total_requests = 0;
  uint64_t total_success = 0;
  uint64_t total_failure = 0;
  uint64_t total_dropped = 0;
  double avg_latency_us = 0.0;
  int64_t p99_latency_us = 0;
};

std::string TrimCarriageReturn(std::string value);

std::string BuildRequestMessage(const std::vector<std::string>& messages,
                                std::atomic<uint64_t>* next_message_index,
                                int batch_size);

int64_t CalculateP99(const std::vector<ThreadStats>& stats);

BenchmarkSummary BuildBenchmarkSummary(const std::vector<ThreadStats>& stats);

int ResolveMaxInflight(double max_qps, int thread_num,
                       int configured_max_inflight, int timeout_ms);

bool TryReserveInflight(std::atomic<int>* inflight, int max_inflight);

void RecordDropped(ThreadStats* stat, RealtimeStats* rt_stats);

}  // namespace predictor

#endif  // BRPC_CLIENT_UTILS_H_
