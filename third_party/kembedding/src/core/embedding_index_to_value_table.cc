#include "include/core/embedding_index_to_value_table.h"

#include <algorithm>

#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace kembedding {

// =============================================================================
// EmbeddingTableReader
// =============================================================================

EmbeddingTableReader::EmbeddingTableReader(const std::string& file_path,
                                           Env* env)
    : file_path_(file_path) {
  FileStatistics stat;
  status_ = env->Stat(file_path, &stat);
  if (!status_.ok()) return;

  status_ = env->NewRandomAccessFile(file_path, &file_);
  if (!status_.ok()) return;

  input_buffer_ = std::make_unique<io::InputBuffer>(file_.get(), 1024 * 1024);
  status_ = ReadHeader(&metadata_);
}

Status EmbeddingTableReader::ReadHeader(
    sparse_embedding_table::TableInfo* info) {
  uint32_t header_size = 0;
  size_t bytes_read = 0;

  Status s = input_buffer_->ReadNBytes(
      sizeof(header_size), reinterpret_cast<char*>(&header_size), &bytes_read);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to read header size from " << file_path_
               << ": " << s.message();
    return s;
  }

  std::string header_bytes;
  s = input_buffer_->ReadNBytes(header_size, &header_bytes);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to read header (" << header_size << " bytes) from "
               << file_path_ << ": " << s.message();
    return s;
  }

  if (!info->ParseFromString(header_bytes)) {
    LOG(ERROR) << "Failed to parse TableInfo protobuf from " << file_path_;
    return errors::DataLoss("Corrupted embedding table header in ", file_path_);
  }

  return absl::OkStatus();
}

Status EmbeddingTableReader::ReadNext(TableEntry* entry) {
  size_t bytes_read = 0;

  TF_RETURN_IF_ERROR(input_buffer_->ReadNBytes(
      sizeof(uint64_t), reinterpret_cast<char*>(&entry->key), &bytes_read));

  uint16_t pair_count = 0;
  TF_RETURN_IF_ERROR(input_buffer_->ReadNBytes(
      sizeof(uint16_t), reinterpret_cast<char*>(&pair_count), &bytes_read));

  entry->pairs.resize(pair_count);

  for (int i = 0; i < pair_count; ++i) {
    TF_RETURN_IF_ERROR(input_buffer_->ReadNBytes(
        sizeof(uint16_t),
        reinterpret_cast<char*>(&entry->pairs[i].index), &bytes_read));
  }

  for (int i = 0; i < pair_count; ++i) {
    TF_RETURN_IF_ERROR(input_buffer_->ReadNBytes(
        sizeof(float),
        reinterpret_cast<char*>(&entry->pairs[i].value), &bytes_read));
  }

  return absl::OkStatus();
}

// =============================================================================
// EmbeddingTableIterator
// =============================================================================

void EmbeddingTableIterator::Advance() {
  if (!valid_) return;

  status_ = reader_->ReadNext(&entry_);

  if (!status_.ok()) {
    if (errors::IsOutOfRange(status_) && rows_read_ != total_rows_) {
      LOG(WARNING) << filename_ << ": expected " << total_rows_
                   << " entries but only read " << rows_read_;
    }
    valid_ = false;
    return;
  }

  ++rows_read_;
}

// =============================================================================
// EmbeddingIndexToValueTable
// =============================================================================

void EmbeddingIndexToValueTable::InitializeBuckets(size_t expected_rows) {
  shards_.resize(bucket_count_);
  if (expected_rows > 0) {
    size_t per_shard = (expected_rows - 1) / bucket_count_ + 1;
    for (auto& shard : shards_) {
      shard.reserve(per_shard);
    }
  }
}

Status EmbeddingIndexToValueTable::Insert(uint64_t key, TableEntry& entry) {
  size_t bucket = key & (bucket_count_ - 1);
  shards_[bucket].emplace(key, std::move(entry.pairs));
  ++num_entries_;
  return absl::OkStatus();
}

Status EmbeddingIndexToValueTable::Initialize(
    EmbeddingTableIterator& iter) {
  if (!iter.Valid()) {
    return iter.status();
  }

  mutex_lock l(mu_);
  if (is_initialized()) {
    return errors::FailedPrecondition(
        "EmbeddingIndexToValueTable is already initialized.");
  }

  InitializeBuckets(static_cast<size_t>(iter.total_rows()));

  while (iter.Valid()) {
    TF_RETURN_IF_ERROR(Insert(iter.entry().key, iter.entry()));
    iter.Advance();
  }

  if (!errors::IsOutOfRange(iter.status())) {
    return iter.status();
  }

  is_initialized_.store(true, std::memory_order_release);
  return absl::OkStatus();
}

Status EmbeddingIndexToValueTable::Find(
    OpKernelContext* ctx, const Tensor& keys,
    int embedding_dim, int max_parallelism,
    Tensor** output_indices, Tensor** output_values,
    Tensor** output_dense_shape) {
  if (!is_initialized()) {
    return errors::FailedPrecondition(
        "EmbeddingIndexToValueTable has not been initialized.");
  }

  const auto key_values = keys.tensor<int64, 1>();
  const int64 num_keys = keys.NumElements();

  // Collect results from each shard in parallel.
  std::vector<const EmbeddingRow*> shard_results(num_keys, nullptr);

  const DeviceBase::CpuWorkerThreads& worker_threads =
      *ctx->device()->tensorflow_cpu_worker_threads();

  Shard(max_parallelism, worker_threads.workers, num_keys, 10000,
        [&](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
            size_t bucket = key_values(i) & (bucket_count_ - 1);
            auto it = shards_[bucket].find(key_values(i));
            if (it != shards_[bucket].end()) {
              shard_results[i] = &it->second;
            }
          }
        });

  // Gather results into intermediate buffers, then copy to output tensors.
  std::vector<int64_t> index_buf;
  std::vector<float> value_buf;
  index_buf.reserve(num_keys * embedding_dim * 2);
  value_buf.reserve(num_keys * embedding_dim);

  for (int64 i = 0; i < num_keys; ++i) {
    const EmbeddingRow* row = shard_results[i];
    if (row == nullptr) continue;

    for (const auto& pair : *row) {
      index_buf.push_back(i);
      index_buf.push_back(pair.index);
      value_buf.push_back(pair.value);
    }
  }

  const int64 num_hits = value_buf.size();

  TF_RETURN_IF_ERROR(
      ctx->allocate_output(0, {num_hits, 2}, output_indices));
  TF_RETURN_IF_ERROR(
      ctx->allocate_output(1, {num_hits}, output_values));

  std::copy(value_buf.begin(), value_buf.end(),
            (*output_values)->vec<float>().data());
  std::copy(index_buf.begin(), index_buf.end(),
            (*output_indices)->flat<int64>().data());

  TF_RETURN_IF_ERROR(ctx->allocate_output(2, {2}, output_dense_shape));
  (*output_dense_shape)->vec<int64>()(0) = num_keys;
  (*output_dense_shape)->vec<int64>()(1) = embedding_dim;

  return absl::OkStatus();
}

Status EmbeddingIndexToValueTable::FindOpt(
    OpKernelContext* ctx, const Tensor& keys,
    int embedding_dim, int max_parallelism,
    Tensor** output_indices, Tensor** output_values,
    Tensor** output_dense_shape) {
  if (!is_initialized()) {
    return errors::FailedPrecondition(
        "EmbeddingIndexToValueTable has not been initialized.");
  }

  const auto key_values = keys.tensor<int64, 1>();
  const int64 num_keys = keys.NumElements();

  std::vector<const EmbeddingRow*> shard_results(num_keys, nullptr);

  const DeviceBase::CpuWorkerThreads& worker_threads =
      *ctx->device()->tensorflow_cpu_worker_threads();

  Shard(max_parallelism, worker_threads.workers, num_keys, 10000,
        [&](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
            size_t bucket = key_values(i) & (bucket_count_ - 1);
            auto it = shards_[bucket].find(key_values(i));
            if (it != shards_[bucket].end()) {
              shard_results[i] = &it->second;
            }
          }
        });

  int64 num_hits = 0;
  for (int64 i = 0; i < num_keys; ++i) {
    if (shard_results[i] != nullptr) {
      num_hits += shard_results[i]->size();
    }
  }

  TF_RETURN_IF_ERROR(
      ctx->allocate_output(0, {num_hits, 2}, output_indices));
  TF_RETURN_IF_ERROR(
      ctx->allocate_output(1, {num_hits}, output_values));

  int64* indices_ptr = (*output_indices)->flat<int64>().data();
  float* values_ptr = (*output_values)->vec<float>().data();

  int64 pos = 0;
  for (int64 i = 0; i < num_keys; ++i) {
    const EmbeddingRow* row = shard_results[i];
    if (row == nullptr) continue;

    for (const auto& pair : *row) {
      indices_ptr[pos * 2] = i;
      indices_ptr[pos * 2 + 1] = pair.index;
      values_ptr[pos] = pair.value;
      ++pos;
    }
  }

  TF_RETURN_IF_ERROR(ctx->allocate_output(2, {2}, output_dense_shape));
  (*output_dense_shape)->vec<int64>()(0) = num_keys;
  (*output_dense_shape)->vec<int64>()(1) = embedding_dim;

  return absl::OkStatus();
}

}  // namespace kembedding
}  // namespace tensorflow
