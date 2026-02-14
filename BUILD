# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

exports_files(glob(["requirements*"]) + [
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "AUTHORS",
    "LICENSE",
])

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
package(default_visibility = ["//visibility:public"],)

cc_library(
  name = 'tensorflow_lib',
  deps = [
      "//tensorflow/core:tensorflow",
      "//tensorflow/core:framework",
      "//tensorflow/core:protos_all_cc",
      "//tensorflow/core:lib",
      "//tensorflow/core:core_cpu",
      "//tensorflow/core:ops",
      "//tensorflow/core:all_kernels",
      "//tensorflow/core:tensorflow_opensource",
      "//tensorflow/cc/saved_model:reader",
      "//tensorflow/cc/saved_model:loader_lite",
      "//tensorflow/cc/saved_model:tag_constants",
      #'//tensorflow/contrib/session_bundle:session_bundle',
      "//tensorflow/core/profiler:profiler",
  ],
  alwayslink = True,
)

cc_library(
  name = 'dummy_tf_func',
  srcs = [
    'dummy_tf_func.h',
  ],
  deps = [
    ':tensorflow_lib',
  ],
  alwayslink = True,
)

COPTS = [
    "-D__STDC_FORMAT_MACROS",
    "-DBTHREAD_USE_FAST_PTHREAD_MUTEX",
    "-D__const__=__unused__",
    "-D_GNU_SOURCE",
    "-DUSE_SYMBOLIZE",
    "-DNO_TCMALLOC",
    "-D__STDC_LIMIT_MACROS",
    "-D__STDC_CONSTANT_MACROS",
    "-fPIC",
    "-Wno-unused-parameter",
    "-fno-omit-frame-pointer",
] + select({
    "@brpc//bazel/config:brpc_with_glog": ["-DBRPC_WITH_GLOG=1"],
    "//conditions:default": ["-DBRPC_WITH_GLOG=0"],
}) + select({
    "@brpc//bazel/config:brpc_with_rdma": ["-DBRPC_WITH_RDMA=1"],
    "//conditions:default": [""],
})

proto_library(
    name = "dummy_proto",
    srcs = [
        "dummy.proto",
    ],
)

cc_proto_library(
    name = "cc_dummy_proto",
    deps = [
        ":dummy_proto",
    ],
)

cc_binary(
    name = "dummy_server",
    srcs = [
        "dummy_server.cc",
    ],
    copts = COPTS,
    deps = [
        ":cc_dummy_proto",
        "@brpc//:brpc",
        "//:dummy_tf_func",
        "@com_github_google_gperftools//:tcmalloc"
    ],
)

cc_binary(
    name = "dummy_client",
    srcs = [
        "dummy_client.cc",
    ],
    copts = COPTS,
    deps = [
        ":cc_dummy_proto",
        "@brpc//:brpc",
    ],
)
