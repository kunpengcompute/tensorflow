// Unit tests for EmbeddingTableLookup related ops.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

#include "include/core/embedding_index_to_value_table.h"
#include "proto/custom/embedding_index_to_value_table.pb.h"

namespace tensorflow {
namespace kembedding {
namespace {

	using TableEntry = tensorflow::kembedding::TableEntry;

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
      io::JoinPath(testing::TmpDir(), "embedding_table_lookup_test.bin");
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

class InitializeEmbeddingIndexToValueTableFromTextFileOpTest
    : public OpsTestBase {
 protected:
  void AddInputTensor(const Tensor& tensor) {
    Tensor* input = AddInput(tensor.dtype(), tensor.shape());
    *input = tensor;
  }

  void MakeOp() {
    TF_ASSERT_OK(
        NodeDefBuilder("init_embedding_table",
                       "InitializeEmbeddingIndexToValueTableFromTextFile")
            .Input(FakeInput(DT_RESOURCE))
            .Input(FakeInput(DT_STRING))
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(InitializeEmbeddingIndexToValueTableFromTextFileOpTest,
       InitializesTableFromBinaryFile) {
  MakeOp();

  const std::string filename = WriteEmbeddingTableFile({
      {101, {{0, 1.0f}, {3, 3.5f}}},
      {202, {{1, 2.5f}}},
  });
  Tensor handle = CreateResourceHandleTensor(
      new EmbeddingIndexToValueTable(), device_->name());
  Tensor filename_tensor(DT_STRING, TensorShape({}));
  filename_tensor.scalar<tstring>()() = filename;

  AddInputTensor(handle);
  AddInputTensor(filename_tensor);
  TF_ASSERT_OK(RunOpKernel());

  auto resource_or = handle.scalar<ResourceHandle>()()
                         .GetResource<EmbeddingIndexToValueTable>();
  TF_ASSERT_OK(resource_or.status());
  ASSERT_NE(resource_or.value(), nullptr);
  EXPECT_TRUE(resource_or.value()->is_initialized());
  EXPECT_EQ(resource_or.value()->num_entries(), 2);
}

class EmbeddingTableLookupOpTest : public OpsTestBase {
 protected:
  void AddInputTensor(const Tensor& tensor) {
    Tensor* input = AddInput(tensor.dtype(), tensor.shape());
    *input = tensor;
  }

  void MakeOp(bool force_ref_impl = false) {
    TF_ASSERT_OK(NodeDefBuilder("embedding_lookup", "EmbeddingTableLookup")
                     .Input(FakeInput(DT_RESOURCE))
                     .Input(FakeInput(DT_INT64))
                     .Attr("emb_dim", 4)
                     .Attr("max_parallelism", 1)
                     .Attr("force_ref_impl", force_ref_impl)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(EmbeddingTableLookupOpTest, ProducesExpectedSparseTensorOutputs) {
  MakeOp();

  const std::string filename = WriteEmbeddingTableFile({
      {101, {{0, 1.0f}, {2, 3.0f}}},
      {202, {{1, 2.5f}}},
  });
  auto* table = new EmbeddingIndexToValueTable();
  EmbeddingTableIterator iter;
  TF_ASSERT_OK(iter.Init(filename, Env::Default()));
  TF_ASSERT_OK(table->Initialize(iter));

  AddInputTensor(CreateResourceHandleTensor(table, device_->name()));
  AddInputFromArray<int64_t>(TensorShape({3}), {101, 202, 999});

  TF_ASSERT_OK(RunOpKernel());

  ASSERT_NE(GetOutput(0), nullptr);
  ASSERT_NE(GetOutput(1), nullptr);
  ASSERT_NE(GetOutput(2), nullptr);

  EXPECT_EQ(GetOutput(0)->shape(), TensorShape({3, 2}));
  EXPECT_EQ(GetOutput(1)->shape(), TensorShape({3}));
  EXPECT_EQ(GetOutput(2)->shape(), TensorShape({2}));

  auto indices = GetOutput(0)->matrix<int64_t>();
  EXPECT_EQ(indices(0, 0), 0);
  EXPECT_EQ(indices(0, 1), 0);
  EXPECT_EQ(indices(1, 0), 0);
  EXPECT_EQ(indices(1, 1), 2);
  EXPECT_EQ(indices(2, 0), 1);
  EXPECT_EQ(indices(2, 1), 1);

  auto values = GetOutput(1)->vec<float>();
  EXPECT_FLOAT_EQ(values(0), 1.0f);
  EXPECT_FLOAT_EQ(values(1), 3.0f);
  EXPECT_FLOAT_EQ(values(2), 2.5f);

  auto dense_shape = GetOutput(2)->vec<int64_t>();
  EXPECT_EQ(dense_shape(0), 3);
  EXPECT_EQ(dense_shape(1), 4);
}

TEST_F(EmbeddingTableLookupOpTest,
       ProducesExpectedSparseTensorOutputsWithForceRefImpl) {
  MakeOp(/*force_ref_impl=*/true);

  const std::string filename = WriteEmbeddingTableFile({
      {101, {{0, 1.0f}, {2, 3.0f}}},
      {202, {{1, 2.5f}}},
  });
  auto* table = new EmbeddingIndexToValueTable();
  EmbeddingTableIterator iter;
  TF_ASSERT_OK(iter.Init(filename, Env::Default()));
  TF_ASSERT_OK(table->Initialize(iter));

  AddInputTensor(CreateResourceHandleTensor(table, device_->name()));
  AddInputFromArray<int64_t>(TensorShape({3}), {101, 202, 999});

  TF_ASSERT_OK(RunOpKernel());

  auto indices = GetOutput(0)->matrix<int64_t>();
  EXPECT_EQ(indices(0, 0), 0);
  EXPECT_EQ(indices(0, 1), 0);
  EXPECT_EQ(indices(1, 0), 0);
  EXPECT_EQ(indices(1, 1), 2);
  EXPECT_EQ(indices(2, 0), 1);
  EXPECT_EQ(indices(2, 1), 1);

  auto values = GetOutput(1)->vec<float>();
  EXPECT_FLOAT_EQ(values(0), 1.0f);
  EXPECT_FLOAT_EQ(values(1), 3.0f);
  EXPECT_FLOAT_EQ(values(2), 2.5f);

  auto dense_shape = GetOutput(2)->vec<int64_t>();
  EXPECT_EQ(dense_shape(0), 3);
  EXPECT_EQ(dense_shape(1), 4);
}

}  // namespace
}  // namespace kembedding
}  // namespace tensorflow
