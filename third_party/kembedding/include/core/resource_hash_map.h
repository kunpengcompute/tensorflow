#ifndef KEMBEDDING_CORE_RESOURCE_HASH_MAP_H_
#define KEMBEDDING_CORE_RESOURCE_HASH_MAP_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace kembedding {

template <typename ResourceType>
Status LookupResource(absl::string_view input_name,
                      OpKernelContext* ctx,
                      ResourceType** resource) {
  *resource = nullptr;

  ResourceHandle handle;
  TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));

  ResourceType* raw = nullptr;
  TF_RETURN_IF_ERROR(LookupResource(ctx, handle, &raw));
  if (raw == nullptr) {
    return errors::NotFound(
        "Resource '", handle.name(), "' in container '",
        handle.container(), "' has not been initialized.");
  }

  *resource = raw;
  return absl::OkStatus();
}

}  // namespace kembedding
}  // namespace tensorflow

#endif  // KEMBEDDING_CORE_RESOURCE_HASH_MAP_H_
