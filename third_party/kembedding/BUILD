# Kembedding: Embedding Table Lookup Custom TensorFlow Operator
# Minimal build targets for embedding table lookup only.

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object", "tf_cc_test")

# Proto: embedding table serialization format
proto_library(
    name = "kembedding_embedding_index_to_value_table_proto",
    srcs = ["proto/custom/embedding_index_to_value_table.proto"],
)

cc_proto_library(
    name = "kembedding_embedding_index_to_value_table_cc_proto",
    deps = [":kembedding_embedding_index_to_value_table_proto"],
)

# Core: EmbeddingIndexToValueTable — sharded hash table for sparse embedding lookup
cc_library(
    name = "embedding_index_to_value_table",
    srcs = ["src/core/embedding_index_to_value_table.cc"],
    hdrs = ["include/core/embedding_index_to_value_table.h"],
    includes = ["."],
    deps = [
        ":kembedding_embedding_index_to_value_table_cc_proto",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

# Op kernel: EmbeddingTableLookup + InitializeEmbeddingIndexToValueTableFromTextFile
cc_library(
    name = "embedding_table_lookup_op",
    srcs = ["src/kernels/embedding_table_lookup_op.cc"],
    hdrs = ["include/core/resource_hash_map.h"],
    includes = ["."],
    deps = [
        ":embedding_index_to_value_table",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# Shared object for dynamic loading
tf_cc_shared_object(
    name = "kembedding_embedding_table_lookup.so",
    deps = [
        ":embedding_table_lookup_op",
    ],
)

# Unit tests
tf_cc_test(
    name = "embedding_table_lookup_op_test",
    srcs = ["tests/embedding_table_lookup/embedding_table_lookup_op_test.cc"],
    deps = [
        ":embedding_index_to_value_table",
        ":embedding_table_lookup_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:test_main",
        "//tensorflow/core:test",
    ],
)

# Benchmark
tf_cc_test(
    name = "embedding_table_lookup_benchmark",
    srcs = ["tests/embedding_table_lookup/embedding_table_lookup_benchmark.cc"],
    deps = [
        ":embedding_index_to_value_table",
        ":embedding_table_lookup_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:test_main",
        "//tensorflow/core:test",
    ],
)
