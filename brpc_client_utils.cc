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

#include "brpc_client_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace predictor {

std::string TrimCarriageReturn(std::string value) {
  if (!value.empty() && value.back() == '\r') {
    value.pop_back();
  }
  return value;
}

std::string BuildRequestMessage(const std::vector<std::string>& messages,
                                std::atomic<uint64_t>* next_message_index,
                                int batch_size) {
  if (messages.empty() || next_message_index == nullptr || batch_size <= 0) {
    return "";
  }

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

int64_t CalculateP99(const std::vector<ThreadStats>& stats) {
  std::vector<int64_t> all_latencies;
  for (const auto& stat : stats) {
    std::lock_guard<std::mutex> lock(stat.latencies_mutex);
    all_latencies.insert(all_latencies.end(), stat.latencies_us.begin(),
                         stat.latencies_us.end());
  }
  if (all_latencies.empty()) {
    return 0;
  }

  std::sort(all_latencies.begin(), all_latencies.end());
  const size_t p99_index =
      static_cast<size_t>(std::ceil(all_latencies.size() * 0.99)) - 1;
  return all_latencies[std::min(p99_index, all_latencies.size() - 1)];
}

BenchmarkSummary BuildBenchmarkSummary(
    const std::vector<ThreadStats>& stats) {
  BenchmarkSummary summary;
  std::vector<int64_t> all_latencies;

  for (const auto& stat : stats) {
    summary.total_requests += stat.total.load(std::memory_order_relaxed);
    summary.total_success += stat.success.load(std::memory_order_relaxed);
    summary.total_failure += stat.failure.load(std::memory_order_relaxed);
    summary.total_dropped += stat.dropped.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(stat.latencies_mutex);
    all_latencies.insert(all_latencies.end(), stat.latencies_us.begin(),
                         stat.latencies_us.end());
  }

  if (all_latencies.empty()) {
    return summary;
  }

  const int64_t total_latency =
      std::accumulate(all_latencies.begin(), all_latencies.end(), int64_t{0});
  summary.avg_latency_us = static_cast<double>(total_latency) /
                           static_cast<double>(all_latencies.size());
  std::sort(all_latencies.begin(), all_latencies.end());
  const size_t p99_index =
      static_cast<size_t>(std::ceil(all_latencies.size() * 0.99)) - 1;
  summary.p99_latency_us =
      all_latencies[std::min(p99_index, all_latencies.size() - 1)];
  return summary;
}

int ResolveMaxInflight(double max_qps, int thread_num,
                       int configured_max_inflight, int timeout_ms) {
  if (configured_max_inflight > 0) {
    return configured_max_inflight;
  }
  if (max_qps > 0 && timeout_ms > 0) {
    return std::max(
        1, static_cast<int>(std::ceil(max_qps * timeout_ms / 1000.0)));
  }
  return std::max(1, thread_num);
}

bool TryReserveInflight(std::atomic<int>* inflight, int max_inflight) {
  if (inflight == nullptr || max_inflight <= 0) {
    return false;
  }
  int current = inflight->load(std::memory_order_relaxed);
  while (current < max_inflight) {
    if (inflight->compare_exchange_weak(current, current + 1,
                                        std::memory_order_acquire,
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

}  // namespace predictor
