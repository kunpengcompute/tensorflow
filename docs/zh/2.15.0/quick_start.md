# 快速入门

## 安装鲲鹏TensorFlow

1. 获取鲲鹏TensorFlow补丁仓。

   ```bash
   git clone -b v1.2.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   ```

2. 获取官方TensorFlow 2.15.0基线。

   ```bash
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

3. 创建完整默认特性的TensorFlow源码。

   ```bash
   python3 patches/2.15.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   cd ../tensorflow-full-default
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   `output/`用于复用Bazel构建缓存，手动下载的构建依赖统一放入`distdir/`，pip包输出到`output-release/`。

   其他可用组合及独立Legacy补丁的使用指导，请参见《[补丁发布说明](../../../patches/2.15.0/README.md)》。

4. 根据实际使用场景选择需要的构建目标，无需同时构建全部目标。以下提供两种常用构建目标的参考命令。

   - TensorFlow pip包：

     ```bash
     bazel --output_base="$PWD/output" build \
       --distdir="$PWD/distdir" \
       -c opt \
       //tensorflow/tools/pip_package:build_pip_package
     ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
     ```

   - TensorFlow C++动态库：
  
     ```bash
     bazel --output_base="$PWD/output" build \
       --distdir="$PWD/distdir" \
       -c opt \
       //tensorflow:tensorflow_cc
     ```

如果在编译过程中遇到任何问题，请参见《[TensorFlow 移植指南](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0001.html)》。

## 运行推理服务

鲲鹏TensorFlow可以作为TensorFlow依赖，集成到推理服务中。当前版本支持基于开源TensorFlow Serving，构建在线推理服务。

### TensorFlow Serving

使用前需基于本仓库生成的完整TensorFlow源码编译TensorFlow Serving。完整流程请参见《[安装指南](./installation_guide.md#构建tensorflow-serving)》。

#### 使用KDNN算子优化

KDNN（Kunpeng Deep Neural Network Library，鲲鹏DNN库）是华为提供的基于鲲鹏平台进行优化的高性能AI算子库。其中MatMul、FusedMatMul、SparseMatMul算子已经适配TensorFlow。集成KDNN可以降低NN类算子的时延，增强模型推理性能。

1. 启动服务端。

    ```bash
    numactl -N 0 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm
    ```

    >![icon note](public_sys-resources/icon-note.gif) **说明：**
    >**numactl -N 0**表示将程序绑定到第0个NUMA节点上运行。

2. 启动客户端性能测试。

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![icon note](public_sys-resources/icon-note.gif) **说明：**
    >--cpuset-cpus：设置容器绑定的CPU核编号。
    >--cpuset-mems：设置容器绑定的NUMA内存节点。

    性能测试启动后，服务端显示“KDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable \`TF\_ENABLE\_KDNN\_OPTS=0\`”字样，即表示使能成功。

    KDNN默认使能，可以通过设置环境变量TF\_ENABLE\_KDNN\_OPTS=0关闭KDNN。

    ![1 zh cn image 0000002504453619](figures/1_zh-cn_image_0000002504453619.png)

#### 使用ANNC静态图融合

鲲鹏TensorFlow ANNC静态图融合特性提供了环境变量开关，即算子可单独开启融合以及算子可全部开启融合。本章节仅提供使用样例，具体使用说明参见《[API参考](./api_reference.md)》。

1. 启动服务端。

    ```bash
    numactl -N 0  ANNC_FUSED_ALL=1 /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/path/to/model_zoo/models/deepfm
    ```

    >![icon note](public_sys-resources/icon-note.gif) **说明：**
    >**numactl -N 0**表示将程序绑定到第0个NUMA节点上运行。

2. 启动客户端性能测试。

    ```bash
    docker run -it --rm --cpuset-cpus="$(cat /sys/devices/system/node/node0/cpulist)" --cpuset-mems="0" --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8000 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

    >![icon note](public_sys-resources/icon-note.gif) **说明：**
    >--cpuset-cpus：设置容器绑定的CPU核编号。
    >--cpuset-mems：设置容器绑定的NUMA内存节点。

本章节仅为使用样例，由于ANNC静态图融合功能需要模型中包含特定子图，若模型中不包含特定子图，则ANNC静态图融合功能不会生效。可以通过运行benchmark测试，用样例验证是否成功集成了ANNC静态图融合特性。

#### ANNC离线图优化（Legacy）

>![icon note](public_sys-resources/icon-note.gif) **说明：**
>该功能仅由独立冻结的Legacy补丁提供，不属于当前维护的默认特性组合。

鲲鹏TensorFlow ANNC图编译优化特性提供了TensorFlow图融合、XLA图融合、算子优化和常量折叠优化特性。本章节仅提供使用样例，具体使用说明请参见《[API参考](./api_reference.md)》。

1. 模型离线图优化。

    ```bash
    annc-opt -I /base_model/deepfm/1/ -O /optimized_model/deepfm/1/ lookup_embedding_hash
    cp -r /base_model/deepfm/1/variables /optimized_model/deepfm/1/
    ```

2. 设置环境变量。

    ```bash
    export ENABLE_BISHENG_GRAPH_OPT=""
    export OMP_NUM_THREADS=1
    export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=16"
    export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"
    export ANNC_FLAGS="--gemm-opt --graph-opt"
    ```

3. 启动TF Serving服务。

    ```bash
    /path/to/tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8889 --model_name=deepfm --model_base_path=/optimized_model/deepfm --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1 --xla_cpu_compilation_enabled=true
    ```

    >![icon note](public_sys-resources/icon-note.gif) **说明：**
    >“--model\_base\_path”所指定的模型不在此限制，用户可自行下载或使用其他模型。

4. 启动客户端压测。

    ```bash
    docker run -it --rm --net host  nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 28:28:1 -p 8561 -f perf.csv -m deepfm --service-kind tfserving -i grpc --request-distribution poisson -b 128  -u localhost:8889 --percentile 99 --input-data=random
    ```

#### TensorFlow Serving线程调度优化（Legacy）

鲲鹏TensorFlow Serving线程调度优化特性提供了算子批量调度和线程亲和性隔离两个特性，具体使用说明请参见《[API参考](./api_reference.md)》。

>![icon note](public_sys-resources/icon-note.gif) **说明：**
>该特性已归入独立冻结的Legacy补丁，不属于当前维护的默认Profile，也不保证与KDNN、ANNC或KEmbedding组合兼容。

## 修订记录

| 发布日期 | 修订记录 |
| ---- | ---- |
| 2026-09-30 | 第三次正式发布。<ul><li>更新按特性组合创建完整TensorFlow源码及按需选择构建目标的操作指导。</li><li>重构TensorFlow Serving推理服务使用说明，并明确当前维护特性与Legacy功能的使用范围。</li></ul> |
| 2026-06-30 | 第二次正式发布。<ul><li>TensorFlow ANNC图编译优化特性增加常量折叠优化特性内容。</li><li>新增TensorFlow ANNC静态图融合特性，增加对应使用样例。</li></ul> |
| 2026-03-30 | 第一次正式发布。 <ul><li>新增TensorFlow集成KDNN使用示例。</li><li>新增TensorFlow KDNN线程直通特性，增加对应使用示例。</li></ul> |
