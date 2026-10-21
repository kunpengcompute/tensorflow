#ifndef KEMBEDDING_CORE_EMBEDDING_INDEX_TO_VALUE_TABLE_H_
#define KEMBEDDING_CORE_EMBEDDING_INDEX_TO_VALUE_TABLE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"

#include "proto/custom/embedding_index_to_value_table.pb.h"

namespace tensorflow {
namespace kembedding {

// A single (index, value) pair stored in the embedding table.
struct ValidIndexValuePair {
  uint16_t index;
  float value;
} __attribute__((packed));

// A lookup result entry: key and its associated index/value pairs.
struct TableEntry {
  uint64_t key = 0;
  std::vector<ValidIndexValuePair> pairs;
};

using EmbeddingRow = std::vector<ValidIndexValuePair>;

// Reads a serialized embedding table from a binary file.
//
// File format:
//   [4 bytes: header_size] [header_size bytes: TableInfo protobuf]
//   For each entry:
//     [8 bytes: key] [2 bytes: pair_count]
//     [pair_count * 2 bytes: indices] [pair_count * 4 bytes: values]
class EmbeddingTableReader {
 public:
  EmbeddingTableReader(const std::string& file_path, Env* env);
  ~EmbeddingTableReader() = default;

  Status status() const { return status_; }

  const sparse_embedding_table::TableInfo& metadata() const {
    return metadata_;
  }

  Status ReadNext(TableEntry* entry);

 private:
  Status ReadHeader(sparse_embedding_table::TableInfo* info);

  std::string file_path_;
  Status status_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::InputBuffer> input_buffer_;
  sparse_embedding_table::TableInfo metadata_;
};

// Iterator over a serialized embedding table file.
class EmbeddingTableIterator {
 public:
  EmbeddingTableIterator()
      : valid_(false),
        rows_read_(0),
        total_rows_(0),
        status_(errors::FailedPrecondition("Iterator not initialized")) {}

  Status Init(const std::string& filename, Env* env) {
    filename_ = filename;
    reader_ = std::make_unique<EmbeddingTableReader>(filename, env);
    status_ = reader_->status();
    if (!status_.ok()) return status_;

    total_rows_ = reader_->metadata().total_key_size();
    valid_ = true;
    rows_read_ = 0;
    Advance();
    return status_;
  }

  void Advance();

  bool Valid() const { return valid_; }

  const TableEntry& entry() const { return entry_; }
  TableEntry& entry() { return entry_; }

  Status status() const { return status_; }
  uint64_t total_rows() const { return total_rows_; }

 private:
  std::unique_ptr<EmbeddingTableReader> reader_;
  TableEntry entry_;
  bool valid_;
  uint64_t rows_read_;
  uint64_t total_rows_;
  std::string filename_;
  Status status_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingTableIterator);
};

// A sharded hash table mapping uint64 keys to sparse embedding rows.
//
// Thread-safe: reads (Find) may happen concurrently with each other.
// Initialization and insertion must be serialized.
class EmbeddingIndexToValueTable : public ResourceBase {
 public:
  EmbeddingIndexToValueTable() = default;
  ~EmbeddingIndexToValueTable() override = default;

  // ResourceBase implementation.
  std::string DebugString() const override {
    return strings::StrCat("EmbeddingIndexToValueTable{size=", num_entries_, "}");
  }

  int64_t MemoryUsed() const override {
    if (!is_initialized()) return 0;
    mutex_lock l(mu_);
    int64_t total = 0;
    for (const auto& shard : shards_) {
      total += shard.size() * (sizeof(uint64_t) + sizeof(EmbeddingRow));
      for (const auto& [key, row] : shard) {
        total += row.size() * sizeof(ValidIndexValuePair);
      }
    }
    return total;
  }

  bool is_initialized() const {
    return is_initialized_.load(std::memory_order_acquire);
  }

  size_t num_entries() const { return num_entries_; }

  // Populates the table from an iterator over a serialized file.
  Status Initialize(EmbeddingTableIterator& iter);

  // Lookup keys and return (row_index, value_index) pairs and values.
  //
  // Returns:
  //   indices: [N, 2] int64 tensor of (key_index, value_index) pairs
  //   values:  [N] float tensor of embedding values
  //   dense_shape: [2] int64 tensor of (num_keys, embedding_dim)
  Status Find(OpKernelContext* ctx, const Tensor& keys,
              int embedding_dim, int max_parallelism,
              Tensor** indices, Tensor** values, Tensor** dense_shape);

  // Optimized variant of Find that pre-allocates output tensors.
  Status FindOpt(OpKernelContext* ctx, const Tensor& keys,
                 int embedding_dim, int max_parallelism,
                 Tensor** indices, Tensor** values, Tensor** dense_shape);

 private:
  Status Insert(uint64_t key, TableEntry& entry);
  void InitializeBuckets(size_t expected_rows);

  static constexpr int kDefaultBucketCount = 128;

  using Bucket = absl::flat_hash_map<uint64_t, EmbeddingRow>;

  mutable mutex mu_;
  std::atomic<bool> is_initialized_{false};
  std::vector<Bucket> shards_;
  size_t bucket_count_ = kDefaultBucketCount;
  size_t num_entries_ = 0;
};

}  // namespace kembedding
}  // namespace tensorflow

#endif  // KEMBEDDING_CORE_EMBEDDING_INDEX_TO_VALUE_TABLE_H_
