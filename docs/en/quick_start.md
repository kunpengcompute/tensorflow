# Quick Start

## Installing the Kunpeng TensorFlow

1. Obtain the Kunpeng TensorFlow patch repository.

   ```bash
   git clone -b v1.2.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   ```

2. Fetch the official TensorFlow 2.15.0 baseline.

   ```bash
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

3. Create a TensorFlow source tree with the complete default profile.

   ```bash
   python3 patches/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   cd ../tensorflow-full-default
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   For other profiles and the standalone Legacy patch, see the
   [Patch Release](../../patches/README_en.md) document.
   The `output/` directory preserves the reusable Bazel build cache, manually
   downloaded build dependencies go in `distdir/`, and pip packages are written
   to `output-release/`.

4. Select the build target required for your use case. You do not need to build all targets. The following commands show two common examples.

   - TensorFlow pip package:

     ```bash
     bazel --output_base="$PWD/output" build \
       --distdir="$PWD/distdir" \
       -c opt \
       //tensorflow/tools/pip_package:build_pip_package
     ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
     ```

   - TensorFlow C++ shared library:

     ```bash
     bazel --output_base="$PWD/output" build \
       --distdir="$PWD/distdir" \
       -c opt \
       //tensorflow:tensorflow_cc
     ```

If you encounter any problem during the compilation, see the [TensorFlow Porting Guide](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0001.html) and [TensorFlow Installation Guide](https://tensorflow.google.cn/install/source?hl=en-us).

## Running an Inference Service

Kunpeng TensorFlow can be integrated into an inference service as its TensorFlow dependency. The current release supports online inference based on open-source TensorFlow Serving.

### TensorFlow Serving

Before running the service, build TensorFlow Serving against the complete TensorFlow source tree generated from this repository. For the complete procedure, see [Installation Guide](./installation_guide.md#building-tensorflow-serving).

#### Using KDNN Kernel Optimizations

Kunpeng Deep Neural Network Library (KDNN) is a high-performance AI operator library optimized for the Kunpeng platform. These optimizations are delivered by integrating operators such as MatMul, FusedMatMul, and SparseMatmul into TensorFlow. Integrating KDNN can reduce the latency of Neural Network (NN) operators and greatly improve the model inference performance.

1. Start the server.

    ```bash
    numactl -N 0 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >`numactl -N 0` binds the program's memory allocation to NUMA node 0.

2. Start the performance test on the client.

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >--`--cpuset-cpus`: limits the container's processes to execute on the specified CPU cores.
    >--`--cpuset-mems`: specifies the memory node bound to the container.

    After the stress test starts, the server prints "KDNN custom operations are on.You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable \`TF_ENABLE_KDNN_OPTS=0\`." In this case, the function is enabled successfully.

    KDNN is enabled by default. You can set the environment variable `TF_ENABLE_KDNN_OPTS` to `0` to disable KDNN.

    ![1 zh cn image 0000002504453619](figures/1_zh-cn_image_0000002504453619.png)

#### Using ANNC Static Graph Fusion

The Kunpeng TensorFlow ANNC static graph fusion feature provides environment variables to enable operator fusion either individually or globally. This section shows usage examples only. For details, see [API Reference](./api_reference.md).

1. Start the server.

    ```bash
    numactl -N 0  ANNC_FUSED_ALL=1 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >`numactl -N 0` binds the program's memory allocation to NUMA node 0.

2. Start the performance test on the client.

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    >--`--cpuset-cpus`: limits the container's processes to execute on the specified CPU cores.
    >--`--cpuset-mems`: specifies the memory node bound to the container.

This section is only an example. Because the ANNC static graph fusion feature depends on specific subgraphs in the model, it will not take effect if the model does not contain such subgraphs. You can run the benchmark test cases to verify whether the ANNC static graph fusion feature has been successfully integrated. For details, see "Verification After Adaptation" in the section "TensorFlow ANNC Static Graph Fusion" of the [Installation Guide](./installation_guide.md).

#### Legacy: ANNC Graph Compilation Optimization

>![icon note](public_sys-resources/icon-note.gif) **NOTE:**
>This feature is provided only by the standalone frozen Legacy patch and is not part of the maintained default feature sets.

Kunpeng's TensorFlow ANNC for graph compilation provides TensorFlow graph fusion, XLA graph fusion, operator optimization, and constant folding optimization. This section provides simple examples. For details, see [API Reference](./api_reference.md).

1. Perform TensorFlow graph fusion.

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

#### Legacy: TensorFlow Serving Thread Scheduling Optimization

Kunpeng's TensorFlow Serving thread scheduling feature provides two configuration options: batch operator scheduling and thread affinity isolation. For details, see [API Reference](./api_reference.md).

>![icon note](public_sys-resources/icon-note.gif) **NOTE:**
>This feature is provided only by the standalone frozen Legacy patch. It is not
>part of the maintained default profiles and is not guaranteed to work with
>KDNN, ANNC, or KEmbedding.

## Description

| Release Date | Change History |
| ---- | ---- |
| 2026-09-30 | This is the third official release. <ul><li>Updated the instructions for creating a complete TensorFlow source tree from a feature set and selecting build targets as needed.</li><li>Restructured the TensorFlow Serving inference service guidance and clarified the scope of maintained and Legacy features.</li></ul> |
| 2026-06-30 | This is the second official release. <ul><li>Added the description for constant folding optimization to the TensorFlow ANNC for graph compilation documentation. </li><li>Added the TensorFlow ANNC static graph fusion feature, including the corresponding examples.</li></ul> |
| 2026-03-30 | This is the first official release. <ul><li>Added the example of integrating TensorFlow with KDNN. </li><li>Added the TensorFlow KDNN thread passthrough feature, including the corresponding examples.</li></ul> |
