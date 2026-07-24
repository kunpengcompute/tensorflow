// Benchmark tests for EmbeddingTableLookup operator.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

#include "include/core/embedding_index_to_value_table.h"
#include "proto/custom/embedding_index_to_value_table.pb.h"

namespace tensorflow {
namespace kembedding {
namespace {

using TableEntry = tensorflow::kembedding::TableEntry;

struct BenchmarkConfig {
  const char* name;
  int num_lookup_keys;
  int num_table_keys;
  int emb_dim;
  int values_per_key;
  int max_parallelism;
  bool include_misses;
};

void AppendRaw(std::string* output, const void* data, size_t size) {
  output->append(static_cast<const char*>(data), size);
}

std::string WriteEmbeddingTableFile(const std::vector<TableEntry>& entries) {
  sparse_embedding_table::TableInfo meta;
  meta.set_total_key_size(entries.size());
  meta.set_endianness(sparse_embedding_table::TableInfo::LITTLE);

  std::string serialized_meta;
  CHECK(meta.SerializeToString(&serialized_meta));

  std::string contents;
  const uint32_t meta_size = serialized_meta.size();
  AppendRaw(&contents, &meta_size, sizeof(meta_size));
  contents.append(serialized_meta);

  for (const auto& entry : entries) {
    AppendRaw(&contents, &entry.key, sizeof(entry.key));
    const uint16_t value_count = entry.pairs.size();
    AppendRaw(&contents, &value_count, sizeof(value_count));
    for (const auto& pair : entry.pairs) {
      AppendRaw(&contents, &pair.index, sizeof(pair.index));
    }
    for (const auto& pair : entry.pairs) {
      AppendRaw(&contents, &pair.value, sizeof(pair.value));
    }
  }

  std::string filename =
      io::JoinPath(testing::TmpDir(), "embedding_table_lookup_benchmark.bin");
  CHECK(Env::Default()->CreateUniqueFileName(&filename, ".bin"));
  TF_CHECK_OK(WriteStringToFile(Env::Default(), filename, contents));
  return filename;
}

Tensor CreateResourceHandleTensor(EmbeddingIndexToValueTable* table,
                                  const std::string& device_name) {
  Tensor handle(DT_RESOURCE, TensorShape({}));
  handle.scalar<ResourceHandle>()() =
      ResourceHandle::MakeRefCountingHandle(table, device_name, {}, {});
  return handle;
}

std::vector<TableEntry> BuildTableEntries(int num_table_keys,
                                          int emb_dim,
                                          int values_per_key) {
  std::vector<TableEntry> entries;
  entries.reserve(num_table_keys);
  for (int i = 0; i < num_table_keys; ++i) {
    TableEntry entry;
    entry.key = 100000 + static_cast<uint64_t>(i);
    entry.pairs.reserve(values_per_key);
    for (int j = 0; j < values_per_key; ++j) {
      const int idx =
          std::min((j * emb_dim) / values_per_key, emb_dim - 1);
      entry.pairs.push_back(
          {static_cast<uint16_t>(idx),
           static_cast<float>((i % 97) + j + 1) / 10.0f});
    }
    entries.push_back(std::move(entry));
  }
  return entries;
}

Tensor BuildKeysTensor(int num_lookup_keys,
                       int num_table_keys,
                       bool include_misses) {
  Tensor keys(DT_INT64, TensorShape({num_lookup_keys}));
  auto flat = keys.vec<int64_t>();
  for (int i = 0; i < num_lookup_keys; ++i) {
    if (include_misses && (i % 4 == 0)) {
      flat(i) = 900000 + i;
      continue;
    }
    flat(i) = 100000 + (i % num_table_keys);
  }
  return keys;
}

class EmbeddingTableLookupBenchmark : public OpsTestBase {
 public:
  void TestBody() override {}

  Status InitOpWithConfig(int emb_dim,
                          int max_parallelism,
                          bool force_ref_impl) {
    TF_RETURN_IF_ERROR(NodeDefBuilder("embedding_lookup_benchmark",
                                      "EmbeddingTableLookup")
                           .Input(FakeInput(DT_RESOURCE))
                           .Input(FakeInput(DT_INT64))
                           .Attr("emb_dim", emb_dim)
                           .Attr("max_parallelism", max_parallelism)
                           .Attr("force_ref_impl", force_ref_impl)
                           .Finalize(node_def()));
    return InitOp();
  }

  void AddInputTensor(const Tensor& tensor) {
    Tensor* input = AddInput(tensor.dtype(), tensor.shape());
    *input = tensor;
  }

  const std::string& DeviceName() const { return device_->name(); }
};

void RunBenchmarkCase(benchmark::State& state,
                      const BenchmarkConfig& config,
                      bool force_ref_impl) {
  ASSERT_GT(config.num_lookup_keys, 0);
  ASSERT_GT(config.num_table_keys, 0);
  ASSERT_GT(config.emb_dim, 0);
  ASSERT_GT(config.values_per_key, 0);
  ASSERT_LE(config.values_per_key, config.emb_dim);

  EmbeddingTableLookupBenchmark benchmark;
  TF_ASSERT_OK(benchmark.InitOpWithConfig(
      config.emb_dim, config.max_parallelism, force_ref_impl));

  const std::string filename = WriteEmbeddingTableFile(
      BuildTableEntries(config.num_table_keys, config.emb_dim,
                        config.values_per_key));
  auto* table = new EmbeddingIndexToValueTable();
  EmbeddingTableIterator iter;
  TF_ASSERT_OK(iter.Init(filename, Env::Default()));
  TF_ASSERT_OK(table->Initialize(iter));

  benchmark.AddInputTensor(
      CreateResourceHandleTensor(table, benchmark.DeviceName()));
  benchmark.AddInputTensor(BuildKeysTensor(
      config.num_lookup_keys, config.num_table_keys, config.include_misses));

  TF_ASSERT_OK(benchmark.RunOpKernel());

  for (auto _ : state) {
    TF_ASSERT_OK(benchmark.RunOpKernel());
  }

  state.SetLabel(config.name);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(config.num_lookup_keys));
}

constexpr BenchmarkConfig kSmallSingleThread = {
    "4KKeys_Dim16_4Values_1Thread_Hit100", 4096, 4096, 16, 4, 1, false};
constexpr BenchmarkConfig kMediumParallel = {
    "16KKeys_Dim32_8Values_8Threads_Hit100", 16384, 16384, 32, 8, 8, false};
constexpr BenchmarkConfig kMediumParallelMiss = {
    "16KKeys_Dim32_8Values_8Threads_Hit75", 16384, 16384, 32, 8, 8, true};
constexpr BenchmarkConfig kLargeParallel = {
    "64KKeys_Dim64_16Values_16Threads_Hit100", 65536, 65536, 64, 16, 16,
    false};

#define DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(config_name, force_ref_impl)   \
  void BM_EmbeddingTableLookup_##config_name##_##force_ref_impl(               \
      benchmark::State& state) {                                               \
    RunBenchmarkCase(state, k##config_name, force_ref_impl);                   \
  }                                                                            \
  BENCHMARK(BM_EmbeddingTableLookup_##config_name##_##force_ref_impl)

DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(SmallSingleThread, false);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(SmallSingleThread, true);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(MediumParallel, false);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(MediumParallel, true);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(MediumParallelMiss, false);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(MediumParallelMiss, true);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(LargeParallel, false);
DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK(LargeParallel, true);

#undef DEFINE_EMBEDDING_TABLE_LOOKUP_BENCHMARK

}  // namespace
}  // namespace kembedding
}  // namespace tensorflow
