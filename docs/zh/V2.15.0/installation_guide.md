# 安装指南

本文按照“准备TensorFlow源码、构建推理服务、验证特性”的顺序介绍安装流程。

## 准备构建环境

当前维护版本的构建环境如下所示。

| 项目 | 版本或要求 |
| --- | --- |
| CPU | <ul><li>鲲鹏920 7282C处理器</li><li>鲲鹏950 7592C处理器</li></ul> |
| OS | <ul><li>openEuler 22.03 LTS SP3</li><li>openEuler 24.03 LTS SP3</li></ul> |
| GCC/G++ | 12.3.1 |
| Bazel | 6.5.0 |
| Python | 3.11.x |

安装基础依赖并启用GCC 12.3.1。

```bash
yum install -y gcc-toolset-12-gcc* git patch patchelf perl \
  python3 python3-devel tar unzip wget zip
export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
```

TensorFlow和TensorFlow Serving均使用Bazel 6.5.0构建。Bazel安装方法请参见《TensorFlow 移植指南》的“[安装Bazel](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html)”章节。

## 准备TensorFlow源码

### 选择特性组合

| 特性组合 | 包含内容 | 对应补丁（按应用顺序） | 适用场景 |
| --- | --- | --- | --- |
| `common-only` | 公共构建集成（common）与兼容性改动 | `patches/V2.15.0/feature/0001-tensorflow_2.15.0-common.patch` | 检查公共改动，不启用加速特性。 |
| `kdnn-core` | 公共构建集成（common）、KDNN算子优化 | `patches/V2.15.0/feature/0001-tensorflow_2.15.0-common.patch`<br>`patches/V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch` | 使用KDNN算子优化。 |
| `kdnn-annc` | 公共构建集成（common）、KDNN算子优化、ANNC静态图融合 | `patches/V2.15.0/feature/0001-tensorflow_2.15.0-common.patch`<br>`patches/V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch`<br>`patches/V2.15.0/feature/0003-tensorflow_2.15.0-annc.patch` | 使用KDNN和ANNC静态图融合。 |
| `full-default` | 公共构建集成（common）、KDNN算子优化、ANNC静态图融合、KEmbedding自定义算子 | `patches/V2.15.0/feature/0001-tensorflow_2.15.0-common.patch`<br>`patches/V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch`<br>`patches/V2.15.0/feature/0003-tensorflow_2.15.0-annc.patch`<br>`patches/V2.15.0/feature/0004-tensorflow_2.15.0-kembedding.patch` | 使用当前维护的全部特性。 |

### 生成完整源码

1. 下载补丁仓并获取官方TensorFlow基线。

   ```bash
   git clone -b v1.2.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

2. 根据需要创建TensorFlow完整源码。以下以完整默认特性为例。

   ```bash
   python3 patches/V2.15.0/prepare_source.py \
     --feature-set full-default \
     --output-dir /path/to/tensorflow
   ```

   如需其他组合，只需修改`--feature-set`。输出目录包含官方TensorFlow `v2.15.0`完整源码、所选补丁以及自动生成的`tensorflow/feature_copts.bzl`。

3. 在TensorFlow源码根目录创建统一的构建目录。

   ```bash
   cd /path/to/tensorflow
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   `output/`用于复用Bazel构建缓存，手动下载的构建依赖统一放入`distdir/`，pip包输出到`output-release/`。

## 准备特性依赖

### KDNN

`kdnn-core`、`kdnn-annc`和`full-default`均需要KDNN头文件和静态库。

1. 获取并安装[KDNN软件包](https://gitcode.com/boostkit/boostsra/releases/download/v1.1.0/BoostKit-boostcore-kdnn_3.0.0.zip)。

   ```bash
   rpm -ivh boostcore-kdnn-3.0.0-1.aarch64.rpm
   ```

   安装后，头文件位于`/usr/local/kdnn/include`，线程池和OpenMP库分别位于
   `/usr/local/kdnn/lib/threadpool`和`/usr/local/kdnn/lib/omp`。
   TensorFlow集成使用线程池版本。

2. 将KDNN头文件和线程池静态库放入生成的TensorFlow源码。

   ```bash
   export TF_PATH=/path/to/tensorflow
   mkdir -p $TF_PATH/third_party/KDNN/src
   cp -r /usr/local/kdnn/include $TF_PATH/third_party/KDNN/
   cp /usr/local/kdnn/lib/threadpool/libkdnn.a \
     $TF_PATH/third_party/KDNN/src/
   ```

3. 应用KDNN头文件适配补丁。

   ```bash
   cd $TF_PATH/third_party/KDNN
   patch -p0 < tensorflow_kdnn_include_adapter.patch
   ```

### ANNC静态图融合

ANNC静态图融合代码已经包含在`kdnn-annc`和`full-default`特性组合中，不需要重复下载或应用TensorFlow补丁。构建前仅需完成KDNN依赖准备。

### KEmbedding

KEmbedding代码已经包含在`full-default`特性组合中，无需额外下载源码。若只需要KEmbedding动态库，可在生成的TensorFlow源码中单独构建：

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  //third_party/kembedding:kembedding_embedding_table_lookup.so
```

构建产物为`bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so`。

## 构建推理服务

当前版本支持基于开源TensorFlow Serving构建推理服务。

### 构建TensorFlow Serving

1. 按照《搜推排序模型推理Benchmark》的“[编译TensorFlow Serving](https://www.hikunpeng.com/document/detail/zh/SRA/perfEval/benchmarksra/kunpengmodelzoo_06_0011.html)”章节准备TensorFlow Serving源码、Bazel和编译依赖。

2. 编译时将`--tensorflow_dir`指向本指南生成的TensorFlow完整源码。

   ```bash
   cd /path/to/serving
   sh compile_serving.sh \
     --tensorflow_dir /path/to/tensorflow \
     --features gcc12
   ```

3. 检查构建产物。

   ```text
   /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
   ```

TensorFlow Serving通过本地TensorFlow源码完成集成。更换特性组合时，无需重新整理Serving源码，只需重新生成对应TensorFlow源码、准备其依赖并重新构建Serving。

## 构建TensorFlow产物

不使用推理服务时，也可以直接在生成的源码中按需构建TensorFlow目标。

### TensorFlow pip包

构建TensorFlow pip包。

```bash
cd /path/to/tensorflow
export TF_PYTHON_VERSION=3.11
./configure
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  -c opt \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

### TensorFlow C++动态库

构建TensorFlow C++动态库。

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  -c opt \
  //tensorflow:tensorflow_cc
```

## 验证特性

### KDNN

1. 进入测试目录并查询支持的模块。

   ```bash
   cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark
   python main.py --list
   ```

2. 从查询结果中选择算子并运行测试。

   ```bash
   python main.py --op {op_name} --performance_test False
   ```

   执行通过即表示相关算子集成成功。

### ANNC静态图融合

进入测试目录后，执行测试命令。

```bash
cd /path/to/tensorflow/tensorflow/python/grappler/embedding_fused_test
python main.py --list
python main.py --op {op_name} --performance_test False
```

ANNC静态图融合只对满足特定结构和输入约束的子图生效。

### KEmbedding

进入测试目录后，执行测试命令。

```bash
cd /path/to/tensorflow
bazel test //third_party/kembedding:embedding_table_lookup_op_test \
  --test_output=errors
bazel run //third_party/kembedding:embedding_table_lookup_benchmark
```

## Legacy功能

Legacy补丁包含历史Runtime调度、旧融合Embedding、ANNC图编译和旧XLA执行等功能，只允许独立应用到官方TensorFlow `v2.15.0`基线。Legacy补丁不依赖common，不属于当前维护的特性组合，也不保证与KDNN、ANNC静态图融合或KEmbedding补丁兼容。

```bash
git clone -b v2.15.0 https://github.com/tensorflow/tensorflow.git tensorflow-legacy
cd tensorflow-legacy
git apply --check \
  /path/to/sra-tensorflow/patches/V2.15.0/frozen_feature/tensorflow_2.15.0-legacy.patch
git apply \
  /path/to/sra-tensorflow/patches/V2.15.0/frozen_feature/tensorflow_2.15.0-legacy.patch

mkdir -p output distdir output-release
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  --config=fused_embedding \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

需要构建TensorFlow Serving时，仍使用“[构建TensorFlow Serving](#构建tensorflow-serving)”中的流程，将`--tensorflow_dir`改为Legacy源码目录。

## 常见问题

编译TensorFlow和TensorFlow Serving时，可参考以下故障处理文档：

- [TensorFlow源码编译证书校验失败](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0012.html)
- [TensorFlow Serving依赖下载失败](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0014.html)
- [获取org_boost依赖失败](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0015.html)
- [Golang网站证书不可用](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0016.html)
- [upb.c编译语法报错](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0017.html)

## 修订记录

| 发布日期 | 修订记录 |
| ---- | ---- |
| 2026-09-30 | 第三次正式发布。重构安装流程，合并各特性重复的环境与编译步骤。 |
| 2026-06-30 | 第二次正式发布。<ul><li>TensorFlow ANNC图编译优化特性增加常量折叠优化特性内容。</li><li>新增TensorFlow ANNC静态图融合特性适配环境和安装指导内容。</li></ul> |
| 2026-03-30 | 第一次正式发布。 <ul><li>新增TensorFlow集成KDNN的安装步骤内容。</li><li>新增TensorFlow KDNN线程直通特性适配环境和安装指导内容。</li></ul> |
