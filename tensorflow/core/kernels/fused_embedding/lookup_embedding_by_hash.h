/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <farmhash.h>
#include <cstring>

namespace tensorflow {

enum ReturnCode {
    OK = 0,
    NULL_POINTER = 1,
    INVALID_PARAMETER = 2
};

static inline int Lookup1D(uintptr_t *lookup_embedding, size_t *lookup_length, int32_t batch_size, float *embedding_table, 
                           int64_t embedding_size, int32_t embedding_dims, float *output)
{
    uint64_t embedding_size_u64 = static_cast<uint64_t>(embedding_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        if (lookup_length[i] != 0) {
            uint64_t hash_value = ::util::Fingerprint64((char *)(lookup_embedding[i]), lookup_length[i]);
            uint64_t x = hash_value % embedding_size_u64;
            output[i] = embedding_table[x];
        } else {
            output[i] = 0;
        }
    }
    return OK;
}

static inline int RegularLookup(uintptr_t *lookup_embedding, size_t *lookup_length, int32_t batch_size, float *embedding_table, 
                                int64_t embedding_size, int32_t embedding_dims, float *output)
{
    uint64_t embedding_dims_u64 = static_cast<uint64_t>(embedding_dims);
    uint64_t embedding_size_u64 = static_cast<uint64_t>(embedding_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        if (lookup_length[i] != 0) {
            uint64_t hash_value = ::util::Fingerprint64((char *)(lookup_embedding[i]), lookup_length[i]);

            uint64_t x = hash_value % embedding_size_u64;
            for (uint64_t j = 0; j < embedding_dims; ++j) {
                output[j] = embedding_table[x * embedding_dims + j];
            }
            output += embedding_dims_u64;
        }
    }
    return OK;
}

struct KPLookupEmbeddingByHashImpl {
    static int Compute(uintptr_t *lookup_embedding, size_t *lookup_length, int32_t batch_size, float *embedding_table, 
                int64_t embedding_size, int32_t embedding_dims, float *output) {
    if (output == nullptr || lookup_embedding == nullptr || embedding_table == nullptr || lookup_length == nullptr) {
        return NULL_POINTER;
    }
    if (batch_size < 0 || embedding_size <= 0 || embedding_dims <= 0) {
        return INVALID_PARAMETER;
    }

    if (embedding_dims == 1) {
        return Lookup1D(lookup_embedding, lookup_length, batch_size, embedding_table, 
                        embedding_size, embedding_dims, output);
    } else {
        return RegularLookup(lookup_embedding, lookup_length, batch_size, embedding_table, 
                             embedding_size, embedding_dims, output);
    }
    }
};

} // namespace tensorflow