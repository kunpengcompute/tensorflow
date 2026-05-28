# Quick Start

## Installing Kunpeng TensorFlow

1. Obtain the TensorFlow open-source code.

   ```bash
   git clone -b v2.15.0 <https://github.com/tensorflow/tensorflow.git> open-tensorflow
   ```

2. Obtain the Kunpeng TensorFlow optimization patch.

   ```bash
   git clone -b master <https://gitcode.com/BoostKit/tensorflow.git> sra-tensorflow
   ```  

3. Enable the optimization patch.

   ```bash
   cp sra-tensorflow/0001-boostsra-tensorflow.patch open-tensorflow
   cd open-tensorflow
   patch -p1 < 0001-boostsra-tensorflow.patch
   ```  

4. Compile the pip package.

   ```bash
   bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
   ```  

5. Compile `libtensorflow_cc.so`.

   ```bash
   bazel build --config=opt //tensorflow/libtensorflow_cc.so
   ```  

If you encounter any problem during the compilation, follow the instructions in the documents: [TensorFlow Porting Guide](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0001.html); [TensorFlow Install](https://tensorflow.google.cn/install/source)

## Sample Running

To use TensorFlow Serving to start an inference stress test, see section [Starting the Service and Performing a Pressure Test](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0012.html) in the _TensorFlow Serving Porting Guide_.

### TensorFlow ANNC for Graph Compilation Optimization

Kunpeng's TensorFlow ANNC feature provides three optimization functions: TensorFlow graph fusion, XLA graph fusion, and operator optimization. For details, see [API Reference](./api_reference.md).

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

    >![note](public_sys-resources/icon-note.gif) **Note:**
    >The model specified by `--model_base_path` is not subject to this restriction. You can download and use other models.

4. Start the stress test on the client.

    ```bash
    docker run -it --rm --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8561 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random 
    ```

### TensorFlow Serving Thread Scheduling

Kunpeng's TensorFlow Serving thread scheduling feature provides two configuration options: batch operator scheduling and thread affinity isolation. For details, see [API Reference](./api_reference.md).

### TensorFlow–KDNN Integration

Kunpeng Deep Neural Network Library (KDNN) is a high-performance AI operator library optimized for the Kunpeng platform. These optimizations are delivered by integrating operators such as MatMul, FusedMatMul, and SparseMatmul into TensorFlow. Integrating KDNN can reduce the latency of Neural Network (NN) operators and greatly improve the model inference performance.

1. Start the server.

    ```bash
    numactl -N 0 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1 --xla_cpu_compilation_enabled=true
    ```

    >![note](public_sys-resources/icon-note.gif) **Note:**
    >`numactl -N 0` binds the program's memory allocation to NUMA node 0.

2. Start the performance test on the client.

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![note](public_sys-resources/icon-note.gif) **Note:**
    >--`--cpuset-cpus`: limits the container's processes to execute on the specified CPU cores.
    >--`--cpuset-mems`: specifies the memory node bound to the container.

    After the stress test starts, the server prints "KDNN custom operations are on.You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable \`TF_ENABLE_KDNN_OPTS=0\`." In this case, the function is enabled successfully.

    KDNN is enabled by default. You can set the environment variable `TF_ENABLE_KDNN_OPTS` to `0` to disable KDNN.

    ![image_0000002504453619](figures/1_zh-cn_image_0000002504453619.png)
