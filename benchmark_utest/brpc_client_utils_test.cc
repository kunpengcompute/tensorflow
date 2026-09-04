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

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace predictor {
namespace {

TEST(BrpcClientUtilsTest, BuildsRoundRobinBatchMessage) {
  const std::vector<std::string> messages = {"first", "second"};
  std::atomic<uint64_t> next_message_index{1};

  EXPECT_EQ(BuildRequestMessage(messages, &next_message_index, 3),
            "second\nfirst\nsecond");
  EXPECT_EQ(next_message_index.load(), 4);
}

TEST(BrpcClientUtilsTest, RejectsInvalidBatchMessageInput) {
  std::atomic<uint64_t> next_message_index{0};
  EXPECT_TRUE(BuildRequestMessage({}, &next_message_index, 1).empty());
  EXPECT_TRUE(BuildRequestMessage({"value"}, nullptr, 1).empty());
  EXPECT_TRUE(
      BuildRequestMessage({"value"}, &next_message_index, 0).empty());
}

TEST(BrpcClientUtilsTest, CalculatesP99AcrossThreads) {
  std::vector<ThreadStats> stats(2);
  for (int64_t latency = 1; latency <= 100; ++latency) {
    stats[static_cast<size_t>(latency % 2)].latencies_us.push_back(latency);
  }

  EXPECT_EQ(CalculateP99(stats), 99);
}

TEST(BrpcClientUtilsTest, ReturnsZeroP99WithoutSamples) {
  std::vector<ThreadStats> stats(1);
  EXPECT_EQ(CalculateP99(stats), 0);
}

TEST(BrpcClientUtilsTest, AggregatesCountersAndLatenciesOnce) {
  std::vector<ThreadStats> stats(2);
  stats[0].total.store(4);
  stats[0].success.store(3);
  stats[0].failure.store(1);
  stats[0].dropped.store(1);
  stats[0].latencies_us = {10, 20, 30};
  stats[1].total.store(3);
  stats[1].success.store(2);
  stats[1].failure.store(1);
  stats[1].latencies_us = {40, 50};

  const BenchmarkSummary summary = BuildBenchmarkSummary(stats);

  EXPECT_EQ(summary.total_requests, 7);
  EXPECT_EQ(summary.total_success, 5);
  EXPECT_EQ(summary.total_failure, 2);
  EXPECT_EQ(summary.total_dropped, 1);
  EXPECT_DOUBLE_EQ(summary.avg_latency_us, 30.0);
  EXPECT_EQ(summary.p99_latency_us, 50);
}

TEST(BrpcClientUtilsTest, ResolvesMaxInflightByPriority) {
  EXPECT_EQ(ResolveMaxInflight(100.0, 4, 8, 1000), 8);
  EXPECT_EQ(ResolveMaxInflight(100.0, 4, 0, 250), 25);
  EXPECT_EQ(ResolveMaxInflight(0.0, 4, 0, 1000), 4);
  EXPECT_EQ(ResolveMaxInflight(0.1, 0, 0, 1), 1);
}

TEST(BrpcClientUtilsTest, EnforcesInflightLimit) {
  std::atomic<int> inflight{0};
  EXPECT_TRUE(TryReserveInflight(&inflight, 2));
  EXPECT_TRUE(TryReserveInflight(&inflight, 2));
  EXPECT_FALSE(TryReserveInflight(&inflight, 2));
  EXPECT_EQ(inflight.load(), 2);
}

TEST(BrpcClientUtilsTest, RecordsDroppedRequestInBothStats) {
  ThreadStats stat;
  RealtimeStats realtime;

  RecordDropped(&stat, &realtime);

  EXPECT_EQ(stat.total.load(), 1);
  EXPECT_EQ(stat.failure.load(), 1);
  EXPECT_EQ(stat.dropped.load(), 1);
  EXPECT_EQ(realtime.total.load(), 1);
  EXPECT_EQ(realtime.failure.load(), 1);
  EXPECT_EQ(realtime.dropped.load(), 1);
}

}  // namespace
}  // namespace predictor
