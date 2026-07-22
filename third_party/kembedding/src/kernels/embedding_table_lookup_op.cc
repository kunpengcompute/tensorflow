#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"

#include "include/core/embedding_index_to_value_table.h"
#include "include/core/resource_hash_map.h"

namespace tensorflow {
namespace kembedding {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// InitializeEmbeddingIndexToValueTableFromTextFile
// =============================================================================

REGISTER_OP("InitializeEmbeddingIndexToValueTableFromTextFile")
    .Input("table_handle: resource")
    .Input("filename: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return absl::OkStatus();
    });

class InitializeEmbeddingIndexToValueTableFromTextFileOp : public OpKernel {
 public:
  explicit InitializeEmbeddingIndexToValueTableFromTextFileOp(
      OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    EmbeddingIndexToValueTable* table = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource<EmbeddingIndexToValueTable>(
                 "table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& filename_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(filename_tensor.shape()),
        errors::InvalidArgument(
            "filename must be a scalar string, got shape ",
            filename_tensor.shape().DebugString()));

    const std::string filename = filename_tensor.scalar<tstring>()();
    OP_REQUIRES(
        ctx, !filename.empty(),
        errors::InvalidArgument("filename must not be empty."));

    int64_t memory_before = 0;
    if (ctx->track_allocations()) {
      memory_before = table->MemoryUsed();
    }

    OP_REQUIRES_OK(ctx, LoadTableFromFile(filename, ctx->env(), table));

    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(
          table->MemoryUsed() - memory_before);
    }
  }

 private:
  Status LoadTableFromFile(const std::string& filename, Env* env,
                           EmbeddingIndexToValueTable* table) {
    EmbeddingTableIterator iter;
    TF_RETURN_IF_ERROR(iter.Init(filename, env));

    Status s = table->Initialize(iter);
    if (errors::IsFailedPrecondition(s) && table->is_initialized()) {
      LOG(INFO) << "Embedding table is already initialized from file: "
                << filename;
      return absl::OkStatus();
    }
    return s;
  }

  mutex mu_;
  TF_DISALLOW_COPY_AND_ASSIGN(
      InitializeEmbeddingIndexToValueTableFromTextFileOp);
};

REGISTER_KERNEL_BUILDER(
    Name("InitializeEmbeddingIndexToValueTableFromTextFile")
        .Device(DEVICE_CPU),
    InitializeEmbeddingIndexToValueTableFromTextFileOp);

// =============================================================================
// EmbeddingIndexToValueTable (resource handle op)
// =============================================================================

REGISTER_OP("EmbeddingIndexToValueTable")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

class EmbeddingIndexToValueTableOp : public OpKernel {
 public:
  explicit EmbeddingIndexToValueTableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_created_(false), use_node_name_sharing_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle_));
  }

  ~EmbeddingIndexToValueTableOp() override {
    if (table_created_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<EmbeddingIndexToValueTable>(cinfo_.container(),
                                                    cinfo_.name())
               .ok()) {
        LOG(WARNING) << "Failed to clean up EmbeddingIndexToValueTable in "
                     << cinfo_.container() << "/" << cinfo_.name();
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_created_) {
      OP_REQUIRES_OK(
          ctx, cinfo_.Init(ctx->resource_manager(), def(),
                           use_node_name_sharing_));
    }

    auto creator = [ctx, this](EmbeddingIndexToValueTable** ret)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> Status {
      auto* table = new EmbeddingIndexToValueTable();
      if (!ctx->status().ok()) {
        table->Unref();
        return ctx->status();
      }

      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(
            table->MemoryUsed() + handle_.AllocatedBytes());
      }

      *ret = table;
      return absl::OkStatus();
    };

    EmbeddingIndexToValueTable* table = nullptr;
    OP_REQUIRES_OK(
        ctx, cinfo_.resource_manager()
                 ->LookupOrCreate<EmbeddingIndexToValueTable>(
                     cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    auto h = handle_.scalar<ResourceHandle>();
    h() = MakeResourceHandle<EmbeddingIndexToValueTable>(
        ctx, cinfo_.container(), cinfo_.name());
    ctx->set_output(0, handle_);

    table_created_ = true;
  }

 private:
  mutex mu_;
  Tensor handle_ TF_GUARDED_BY(mu_);
  bool table_created_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingIndexToValueTableOp);
};

REGISTER_KERNEL_BUILDER(
    Name("EmbeddingIndexToValueTable").Device(DEVICE_CPU),
    EmbeddingIndexToValueTableOp);

// =============================================================================
// EmbeddingTableLookup
// =============================================================================

REGISTER_OP("EmbeddingTableLookup")
    .Attr("emb_dim: int = 4")
    .Attr("max_parallelism: int = 8")
    .Attr("force_ref_impl: bool = false")
    .Input("table_handle: resource")
    .Input("keys: int64")
    .Output("indices: int64")
    .Output("values: float")
    .Output("dense_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->UnknownShape());
      c->set_output(2, c->Vector(2));
      return absl::OkStatus();
    });

class EmbeddingTableLookupOp : public OpKernel {
 public:
  explicit EmbeddingTableLookupOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        embedding_dim_(4),
        max_parallelism_(8),
        force_ref_impl_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("emb_dim", &embedding_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_parallelism", &max_parallelism_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("force_ref_impl", &force_ref_impl_));

    OP_REQUIRES(ctx, embedding_dim_ > 0,
                errors::InvalidArgument("emb_dim must be > 0, got ",
                                        embedding_dim_));
    OP_REQUIRES(ctx, max_parallelism_ > 0,
                errors::InvalidArgument("max_parallelism must be > 0, got ",
                                        max_parallelism_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingIndexToValueTable* table = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource<EmbeddingIndexToValueTable>(
                 "table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& keys = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(keys.shape()),
        errors::InvalidArgument("keys must be a 1-D tensor, got shape ",
                                keys.shape().DebugString()));

    Tensor* indices = nullptr;
    Tensor* values = nullptr;
    Tensor* dense_shape = nullptr;

    if (force_ref_impl_) {
      OP_REQUIRES_OK(
          ctx, table->Find(ctx, keys, embedding_dim_, max_parallelism_,
                           &indices, &values, &dense_shape));
    } else {
      OP_REQUIRES_OK(
          ctx, table->FindOpt(ctx, keys, embedding_dim_, max_parallelism_,
                              &indices, &values, &dense_shape));
    }
  }

 private:
  int embedding_dim_;
  int max_parallelism_;
  bool force_ref_impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingTableLookupOp);
};

REGISTER_KERNEL_BUILDER(
    Name("EmbeddingTableLookup").Device(DEVICE_CPU),
    EmbeddingTableLookupOp);

}  // namespace kembedding
}  // namespace tensorflow
