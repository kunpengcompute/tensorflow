# Quick Start

<!-- md-trans-meta sourceCommit=5c9162943c20095e6b484e3b477f280464376ee5 translatedAt=2026-08-04T03:18:27.505Z pushedAt=2026-08-05T01:07:46.202Z -->

## Installing the Kunpeng TensorFlow

1. Obtain the Kunpeng TensorFlow patch repository.

   ```bash
   git clone -b v1.2.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   ```

2. Obtain the official TensorFlow 2.15.0 baseline.

   ```bash
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```  

3. Create the complete TensorFlow source code with default features.

   ```bash
   python3 patches/V2.15.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   cd ../tensorflow-full-default
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```  

   For other profiles and the standalone Legacy patch, see the
   [Patch Release](../../../patches/V2.15.0/README_en.md) document.
   The `output/` directory preserves the reusable Bazel build cache, manually
   downloaded build dependencies go in `distdir/`, and pip packages are written
   to `output-release/`.

   For guidance on other available combinations and standalone Legacy patches, see [Patch Release Notes](../../../patches/V2.15.0/README.md).

4. Select the required build target based on your actual use scenario. You do not need to build all targets at the same time. The following provides reference commands for two common build targets.

   - TensorFlow pip package:

     ```bash
     bazel --output_base="$PWD/output" build \
       --distdir="$PWD/distdir" \
       -c opt \
       //tensorflow/tools/pip_package:build_pip_package
     ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
     ```

   - TensorFlow C++ dynamic library:

     ```bash
     bazel --output_base="$PWD/output" build \
       --distdir="$PWD/distdir" \
       -c opt \
       //tensorflow:tensorflow_cc
     ```

If you encounter any problem during the compilation, see [TensorFlow Porting Guide](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0017.html).

## Running the Inference Service

Kunpeng TensorFlow can be integrated into an inference service as a TensorFlow dependency. The current version supports building an online inference service based on the open-source TensorFlow Serving.

### TensorFlow Serving

Before use, you need to compile TensorFlow Serving based on the complete TensorFlow source code generated from this repository. For the complete process, see [Installation Guide](./installation_guide.md#building-tensorflow-serving).

#### Using KDNN Operator Optimization

Kunpeng Deep Neural Network Library (KDNN) is a high-performance AI operator library optimized for the Kunpeng platform. Operators such as MatMul, FusedMatMul, and SparseMatMul have been adapted to TensorFlow. Integrating KDNN can reduce the latency of NN operators and enhance model inference performance.

1. Start the server.

    ```bash
    numactl -N 0 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >`numactl -N 0` binds the program's memory allocation to NUMA node 0.

2. Start the performance test on the client.

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >`--cpuset-cpus`: limits the container's processes to execute on the specified CPU cores.
    >`--cpuset-mems`: specifies the memory node bound to the container.

    After the stress test starts, the server displays "KDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable \`TF\_ENABLE\_KDNN\_OPTS=0\`." In this case, the function is enabled successfully.

    KDNN is enabled by default. You can set the environment variable `TF_ENABLE_KDNN_OPTS` to `0` to disable KDNN.

    ![1_zh-cn_image_0000002504453619](figures/1_zh-cn_image_0000002504453619.png)

#### Using ANNC Static Graph Fusion

The Kunpeng TensorFlow ANNC static graph fusion feature provides environment variables to enable operator fusion either individually or globally. This section provides usage examples only. For details, see [API Reference](./api_reference.md).

1. Start the server.

    ```bash
    numactl -N 0  ANNC_FUSED_ALL=1 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >`numactl -N 0` binds the program's memory allocation to NUMA node 0.

2. Start the performance test on the client.

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >`--cpuset-cpus`: limits the container's processes to execute on the specified CPU cores.
    >`--cpuset-mems`: specifies the memory node bound to the container.

This section provides usage examples only. The ANNC static graph fusion feature requires specific subgraphs in the model. If the model does not contain such subgraphs, the ANNC static graph fusion feature will not take effect. You can run a benchmark test with the sample to verify whether the ANNC static graph fusion feature is successfully integrated.

#### ANNC Offline Graph Optimization (Legacy)

>![icon note](public_sys-resources/icon-note.gif) **NOTE:**
>This feature is provided only by an independently frozen Legacy patch and is not part of the currently maintained default feature set.

The Kunpeng TensorFlow ANNC graph compilation optimization feature provides TensorFlow graph fusion, XLA graph fusion, operator optimization, and constant folding optimization. This section provides usage examples only. For details, see [API Reference](./api_reference.md).

1. Perform offline model graph optimization.

    ```bash
    annc-opt -I /base_model/deepfm/1/ -O /optimized_model/deepfm/1/ lookup_embedding_hash
    cp -r /base_model/deepfm/1/variables /optimized_model/deepfm/1/
    ```

2. Set the environment variables.

    ```bash
    export ENABLE_BISHENG_GRAPH_OPT=""
    export OMP_NUM_THREADS=1
    export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=16"
    export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"
    export ANNC_FLAGS="--gemm-opt --graph-opt"
    ```

3. Start the TF Serving service.

    ```bash
    /path/to/tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/optimized_model/deepfm --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1 --xla_cpu_compilation_enabled=true
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >The model specified by `--model_base_path` is not subject to this restriction. You can download and use other models.

4. Start the stress test on the client.

    ```bash
    docker run -it --rm --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8561 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random 
    ```

#### TensorFlow Serving Thread Scheduling Optimization (Legacy)

The Kunpeng TensorFlow Serving thread scheduling optimization feature provides two configuration options: batch operator scheduling and thread affinity isolation. For details, see [API Reference](./api_reference.md).

>![icon note](public_sys-resources/icon-note.gif) **NOTE:**
>This feature has been moved to an independently frozen Legacy patch. It is not part of the currently maintained default profile, and compatibility with KDNN, ANNC, or KEmbedding is not guaranteed.

## Change History

| Release Date | Change History |
| ---- | ---- |
| 2026-09-30 | Third official release.<ul><li>Updated the instructions for creating complete TensorFlow source code by feature combination and selecting build targets as needed.</li><li>Restructured the TensorFlow Serving inference service usage instructions, and clarified the scope of currently maintained features and Legacy functions.</li></ul> |
| 2026-06-30 | Second official release.<ul><li>Added constant folding optimization to the TensorFlow ANNC graph compilation optimization feature.</li><li>Added the TensorFlow ANNC static graph fusion feature with corresponding usage examples.</li></ul> |
| 2026-03-30 | First official release. <ul><li>Added usage examples for TensorFlow–KDNN integration.</li><li>Added the TensorFlow KDNN thread passthrough feature with corresponding usage examples.</li></ul> |
