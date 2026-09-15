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

TensorFlow和TensorFlow Serving均使用Bazel 6.5.0构建。Bazel安装方法请参见《TensorFlow 移植指南》的“[安装Bazel](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0009.html)”章节。

## 准备TensorFlow源码

### 选择特性组合

| 特性组合 | 包含内容 | 对应补丁（按应用顺序） | 适用场景 |
| --- | --- | --- | --- |
| common-only | 公共构建集成（common）与兼容性改动 | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch | 检查公共改动，不启用加速特性。 |
| kdnn-core | 公共构建集成（common）、KDNN算子优化 | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch<br>patches/Tensorflow_V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch | 使用KDNN算子优化。 |
| kdnn-annc | 公共构建集成（common）、KDNN算子优化、ANNC静态图融合 | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch<br>patches/Tensorflow_V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch<br>patches/Tensorflow_V2.15.0/feature/0003-tensorflow_2.15.0-annc.patch | 使用KDNN和ANNC静态图融合。 |
| full-default | 公共构建集成（common）、KDNN算子优化、ANNC静态图融合、KEmbedding自定义算子 | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch<br>patches/Tensorflow_V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch<br>patches/Tensorflow_V2.15.0/feature/0003-tensorflow_2.15.0-annc.patch<br>patches/Tensorflow_V2.15.0/feature/0004-tensorflow_2.15.0-kembedding.patch | 使用当前维护的全部特性。 |

### 生成完整源码

下载补丁仓并获取官方TensorFlow基线。

   ```bash
   git clone https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

根据需要创建TensorFlow完整源码。以下以完整默认特性为例。

   ```bash
   python3 patches/Tensorflow_V2.15.0/prepare_source.py \
     --feature-set full-default \
     --output-dir /path/to/tensorflow
   ```

   如需其他组合，只需修改`--feature-set`。输出目录包含官方TensorFlow `v2.15.0`完整源码、所选补丁以及自动生成的`tensorflow/feature_copts.bzl`。

在TensorFlow源码根目录创建统一的构建目录。

   ```bash
   cd /path/to/tensorflow
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   `output/`用于复用Bazel构建缓存，手动下载的构建依赖统一放入`distdir/`，pip包输出到`output-release/`。

## 准备特性依赖

### KDNN

`kdnn-core`、`kdnn-annc`和`full-default`均需要KDNN头文件和静态库。

#### 下载KDNN软件包及校验文件

访问[KDNN 发布网页](https://gitcode.com/boostkit/boostsra/releases)，找到套件发行版v1.4.0，下载下面两个文件:

- KDNN 软件包: `BoostKit-boostcore-kdnn_3.2.0.zip`
- SHA256 校验文件: `BoostKit-boostcore-kdnn_3.2.0.zip.sha256`

#### 软件包完整性校验

##### 简介

为了检查软件包在传输或存储过程中是否因网络或设备问题而不完整，在获取到软件包后，需要对软件包的完整性进行校验，通过了校验的软件包才能部署。<br>
这里通过对比校验文件中记录的校验值和手动方式计算的软件包校验值，判断软件包是否完整。若两个值相同，说明文件完整，否则，文件完整性被破坏，请重新获取软件包。

##### 前提条件

在校验软件包完整性之前，需要准备如下文件：

- 软件包：BoostKit-boostcore-kdnn_xxx.zip。
- 校验文件：同名的.sha256文件。

##### 操作指导

文件完整性校验操作步骤如下：

1. 计算软件包的sha256校验值。linux执行命令如下：

   ```bash
   sha256sum BoostKit-boostcore-kdnn_xxx.zip
   ```

   windows执行命令如下：

   ```bash
   certutil -hashfile BoostKit-boostcore-kdnn_xxx.zip SHA256
   ```

   命令执行完成后，输出校验值。
2. 对比步骤 1 计算的校验值与校验文件中的 SHA256 值是否一致<br>
   如果校验值一致说明文件完整，如果校验值不一致则可以确认文件完整性已被破坏，需要重新获取。

#### 安装KDNN软件包

   ```bash
   unzip BoostKit-boostcore-kdnn_3.2.0.zip
   rpm -ivh boostcore-kdnn-3.2.0-1.aarch64.rpm
   ```

   安装后，头文件位于`/usr/local/kdnn/include`，线程池和OpenMP库分别位于
   `/usr/local/kdnn/lib/threadpool`和`/usr/local/kdnn/lib/omp`。
   TensorFlow集成使用线程池版本。

将KDNN头文件和线程池静态库放入生成的TensorFlow源码。

   ```bash
   export TF_PATH=/path/to/tensorflow
   mkdir -p $TF_PATH/third_party/KDNN/src
   cp -r /usr/local/kdnn/include $TF_PATH/third_party/KDNN/
   cp /usr/local/kdnn/lib/threadpool/libkdnn.a \
     $TF_PATH/third_party/KDNN/src/
   ```

应用KDNN头文件适配补丁。

   ```bash
   cd $TF_PATH/third_party/KDNN
   patch -p0 < tensorflow_kdnn_include_adapter.patch
   ```

### ANNC静态图融合

ANNC静态图融合代码已经包含在`kdnn-annc`和`full-default`特性组合中，不需要重复下载或应用TensorFlow补丁。构建前仅需完成KDNN依赖准备。

### KEmbedding

KEmbedding代码已经包含在`full-default`特性组合中，无需额外下载源码。

>![](public_sys-resources/icon-note.gif) **说明：**
>
>- 若只需要KEmbedding动态库，可在生成的TensorFlow源码中单独构建，若需构建完整Tensorflow产物则跳过此步骤。

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  //third_party/kembedding:kembedding_embedding_table_lookup.so
```

构建产物为`bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so`。

## 构建TensorFlow产物

### TensorFlow pip包

构建TensorFlow pip包。

```bash
cd /path/to/tensorflow
export TF_PYTHON_VERSION=3.11
./configure
```

依次按照如下选项配置，未显示N或n的选项使用回车`[Enter]`确认。

```bash
You have bazel 6.5.0- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python3.11]:


Found possible Python library paths:
  /usr/lib/python3.11/site-packages
  /usr/lib64/python3.11/site-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3.11/site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: N
No CUDA support will be enabled for TensorFlow.

Do you want to use Clang to build TensorFlow? [Y/n]: n
GCC will be used to compile TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).
        --config=monolithic     # Config for mostly static monolithic build.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v1             # Build with TensorFlow 1 API instead of TF 2 API.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=nogcp          # Disable GCP support.
        --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```

开始执行编译。

```bash
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  -c opt \
  --define=enable_kdnn=True \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

>![](public_sys-resources/icon-note.gif) **说明：**
>
>- `--define=enable_kdnn=True`用于开启KDNN算子优化，选择`kdnn-core`、`kdnn-annc`或`full-default`特性组合时添加该选项，`common-only`组合不包含KDNN代码，无需添加。

编译完成后，检查产物目录。

```bash
ls ./output-release
```

回显类似如下信息时，表示pip包构建成功。

```text
tensorflow-2.15.0-cp311-cp311-linux_aarch64.whl
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

>![](public_sys-resources/icon-note.gif) **说明：**
>
>- 需完成构建TensorFlow pip包并安装后，才能验证特性。

安装命令。

```bash
pip install /path/to/tensorflow/output-release/tensorflow-2.15.0-cp311-cp311-linux_aarch64.whl
```

安装完成后，切换到TensorFlow源码目录以外的任意目录（如用户主目录）执行以下命令验证。

```bash
cd ~
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

>![](public_sys-resources/icon-note.gif) **说明：**
>
>- 请勿在TensorFlow源码根目录下执行验证命令，否则Python会优先加载源码目录下的`tensorflow/`包而非已安装的pip包，导致导入报错。

回显类似如下信息时，表示TensorFlow安装成功。

```text
2.15.0
```

### KDNN

进入测试目录并查询支持的模块。

   ```bash
   cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark
   python main.py --list
   ```

从查询结果中选择算子并运行测试。

   ```bash
   numactl -C 0-15 python main.py --op {op_name} --performance_test True
   ```

   执行通过即表示相关算子集成成功。

### ANNC静态图融合

进入测试目录后，执行测试命令。

```bash
cd /path/to/tensorflow/tensorflow/python/grappler/embedding_fused_test
python main.py --list
python main.py --op {op_name} --performance_test True
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

## 构建推理服务（可选）

当前版本支持基于开源TensorFlow Serving构建推理服务。

### 构建TensorFlow Serving

按照《搜推排序模型推理Benchmark》的“[编译TensorFlow Serving](https://www.hikunpeng.com/document/detail/zh/SRA/perfEval/benchmarksra/kunpengmodelzoo_06_0011.html)”章节准备TensorFlow Serving源码、Bazel和编译依赖。

编译时将`--tensorflow_dir`指向本指南生成的TensorFlow完整源码。

   ```bash
   cd /path/to/serving
   sh compile_serving.sh \
     --tensorflow_dir /path/to/tensorflow \
     --features gcc12
   ```

检查构建产物。

   ```text
   /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
   ```

TensorFlow Serving通过本地TensorFlow源码完成集成。更换特性组合时，无需重新整理Serving源码，只需重新生成对应TensorFlow源码、准备其依赖并重新构建Serving。

## Legacy功能

Legacy补丁包含历史Runtime调度、旧融合Embedding、ANNC图编译和旧XLA执行等功能，只允许独立应用到官方TensorFlow `v2.15.0`基线。

>![](public_sys-resources/icon-note.gif) **说明：**
>
>- Legacy补丁不依赖common，不属于当前维护的特性组合，也不保证与KDNN、ANNC静态图融合或KEmbedding补丁兼容。

```bash
git clone -b v2.15.0 https://github.com/tensorflow/tensorflow.git tensorflow-legacy
cd tensorflow-legacy
git apply --check \
  /path/to/sra-tensorflow/patches/Tensorflow_V2.15.0/frozen_feature/tensorflow_2.15.0-legacy.patch
git apply \
  /path/to/sra-tensorflow/patches/Tensorflow_V2.15.0/frozen_feature/tensorflow_2.15.0-legacy.patch

mkdir -p output distdir output-release
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  --config=fused_embedding \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

需要构建TensorFlow Serving时，仍使用“[构建TensorFlow Serving](#构建tensorflow-serving)”中的流程，将`--tensorflow_dir`改为Legacy源码目录。

## 常见问题

编译TensorFlow和TensorFlow Serving时，可参考以下故障处理文档。

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
