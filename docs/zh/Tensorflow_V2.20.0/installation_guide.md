# 安装指南

本文按照“准备TensorFlow源码、构建推理服务、验证特性”的顺序介绍安装流程。

## 准备构建环境

当前维护版本的构建环境如下所示。

| 项目 | 版本或要求 |
| --- | --- |
| CPU | 鲲鹏950 7592C处理器 |
| OS | openEuler 24.03 LTS SP3 |
| GCC/G++ | 12.3.1 |
| Bazel | 7.4.1 |
| Python | 3.11.x |
| CUDA  | 12.8.0 |
| CuDNN | 9.5.0 |

安装基础依赖。**当前维护特性中，仅 ANNC 特性包含 GPU 算子（FlashAttentionForward），其他特性（KDNN、KEmbedding）均为纯 CPU 优化**。如需使用 ANNC 的 GPU 算子，需安装 CUDA/CuDNN（CUDA 安装路径建议为 /usr/local/cuda-12.8）；纯 CPU 构建（仅使用 ANNC 的 KPFused CPU 算子）无需安装 CUDA。

TensorFlow V2.20.0使用官方指定的Bazel 7.4.1构建。Bazel安装方法请参见《TensorFlow 移植指南》的“[安装Bazel](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html)”章节。

>![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
>
>CUDA/CuDNN 仅 ANNC 特性的 GPU 算子（FlashAttentionForward）需要。KDNN、KEmbedding 特性均为纯 CPU 优化，不涉及 GPU。纯 CPU 构建可跳过”CUDA 安装指导”章节，直接进入”准备 TensorFlow 源码”。

### CUDA安装指导

#### 检查硬件与系统

  1. 检查显卡是否存在。

     ```bash
     lspci | grep -i nvidia
     ```

  2. 安装依赖。

     ```bash
     yum install -y gcc make dkms
     yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
     ```

  3. 确认GPU支持CUDA：访问“[NVIDIA CUDA-GPU列表](https://developer.nvidia.com/cuda-gpus)”确认你的GPU支持CUDA。

#### 安装CUDA

  1. 禁用nouveau驱动。
  
     检查系统是否加载nouveau驱动，若有输出则代表启用了nouveau驱动，需要禁用nouveau驱动。

     ```bash
     smod | grep nouveau
     ```
  
  2. 下载CUDA工具包。

     CUDA 12.8官网下载链接“[CUDA Toolkit 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=RHEL&target_version=8)”。

     下载对应版本的CUDA工具包，且仅需下载CUDA工具包，无需单独下载GPU驱动。CUDA工具包`cuda_12.8.0_***_linux_sbsa.run`内含GPU驱动。

  3. 安装CUDA工具包。

     安装工具包，赋予可执行权限并运行.run文件，根据提示完成安装。

     ```bash
     ./cuda_12.8.0_***_linux_sbsa.run
     ```

     添加环境变量。安装完成后，需要将CUDA路径添加到系统环境变量（建议采用默认安装路径`/usr/local/cuda-12.8`）。

  4. 验证安装。
  
     检查CUDA版本，若输出CUDA 12.8.0相关信息，则安装成功。

     ```bash
     nvcc --version
     ```

## 准备TensorFlow源码

### 选择特性组合

| 特性组合 | 包含内容 | 对应补丁（按应用顺序） | 适用场景 |
| --- | --- | --- | --- |
| common-only | 公共构建集成（common）与gtest覆盖率测试框架 | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch | 检查公共改动，不启用加速特性。 |
| kdnn-core | 公共构建集成（common）、KDNN算子优化 | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-kdnn.patch | 使用KDNN算子优化。 |
| kdnn-annc | 公共构建集成（common）、KDNN算子优化、ANNC静态图融合 | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-kdnn.patch<br>patches/Tensorflow_V2.20.0/feature/0003-tensorflow_2.20.0-annc.patch | 使用KDNN和ANNC静态图融合。 |
| full-default | 公共构建集成（common）、KDNN算子优化、ANNC静态图融合、KEmbedding自定义算子 | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-kdnn.patch<br>patches/Tensorflow_V2.20.0/feature/0003-tensorflow_2.20.0-annc.patch<br>patches/Tensorflow_V2.20.0/feature/0004-tensorflow_2.20.0-kembedding.patch | 使用当前维护的全部特性。 |

### 生成完整源码

1. 下载补丁仓并获取官方TensorFlow基线。

   ```bash
   git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
   cd kunpeng-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0
   ```

2. 根据需要创建TensorFlow完整源码。纯CPU构建与GPU构建均使用`full-default`组合，构建时通过是否启用CUDA区分。

   ```bash
   # 创建包含全部特性的TensorFlow完整源码（纯CPU与GPU构建通用）
   python3 patches/Tensorflow_V2.20.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   ```

   如需其他组合，只需修改`--feature-set`。输出目录包含官方TensorFlow `v2.20.0`完整源码、所选补丁。

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

1. 将KDNN软件包解压缩，并使用rpm -ivh命令安装。

   ```bash
   unzip BoostKit-boostcore-kdnn_3.2.0.zip
   rpm -ivh boostcore-kdnn-3.2.0-1.aarch64.rpm
   ```

   安装后，头文件位于`/usr/local/kdnn/include`，线程池和OpenMP库分别位于`/usr/local/kdnn/lib/threadpool`和`/usr/local/kdnn/lib/omp`。
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

> ![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
> 
> ANNC 特性包含 GPU 算子，。

构建前需完成 KDNN 依赖准备。

### KEmbedding

1. KEmbedding代码已经包含在`full-default`特性组合中，无需额外下载源码。

   ```bash
   cd /path/to/tensorflow
   bazel build //third_party/kembedding:kembedding_embedding_table_lookup.so
   ```

   >![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
   >
   >- 若只需要KEmbedding动态库，可在生成的TensorFlow源码中单独构建，若需构建完整Tensorflow产物则跳过此步骤。

2. 构建产物为`bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so`。

## 构建TensorFlow产物

不使用推理服务时，也可以直接在生成的源码中按需构建TensorFlow目标。

### 构建TensorFlow pip包

1. 配置bazel。

   ```bash
   cd /path/to/tensorflow
   ./configure
   You have bazel 7.4.1 installed.
   Please specify the location of python. [Default is /usr/bin/python3]: 


   Found possible Python library paths:
   /usr/lib/python3.11/site-packages
   /usr/lib64/python3.11/site-packages
   /usr/local/lib/python3.11/site-packages
   /usr/local/lib64/python3.11/site-packages
   Please input the desired Python library path to use.  Default is [/usr/lib/python3.11/site-packages]

   Do you wish to build TensorFlow with ROCm support? [y/N]: n
   No ROCm support will be enabled for TensorFlow.

   Do you wish to build TensorFlow with CUDA support? [y/N]: n
   No CUDA support will be enabled for TensorFlow.

   Do you want to use Clang to build TensorFlow? [Y/n]: n
   GCC will be used to compile TensorFlow.

   Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]: 


   Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
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

2. 构建TensorFlow pip包。

   ```bash
   cd /path/to/tensorflow
   bazel build \
   --experimental_repo_remote_exec \
   --cxxopt=-std=c++17 \
   --host_cxxopt=-std=c++17 \
   --copt=-march=armv8.5-a \
   --host_copt=-march=armv8.5-a \
   --check_direct_dependencies=off \
   -c opt \
   //tensorflow/tools/pip_package:wheel
   ```

   构建`eigen-core`和`full-default`特性组合时，追加`--copt=-DEIGEN_NEON_USE_KGEMM=1`以开启Eigen KGemm矩阵乘加速。

   ```bash
   bazel build \
   --experimental_repo_remote_exec \
   --cxxopt=-std=c++17 \
   --host_cxxopt=-std=c++17 \
   --copt=-march=armv8.5-a \
   --host_copt=-march=armv8.5-a \
   --copt=-DEIGEN_NEON_USE_KGEMM=1 \
   --check_direct_dependencies=off \
   -c opt \
   //tensorflow/tools/pip_package:wheel
   ```

3. 编译完成后，检查产物目录。

   ```bash
   ls bazel-bin/tensorflow/tools/pip_package/wheel_house/
   ```

   回显类似如下信息时，表示pip包构建成功。

   ```output
   tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
   ```

### 安装tensorflow pip包

```bash
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
```

### 构建GPU版本

**若要构建支持FAGO特性的TensorFlow，请编译包含GPU支持的TensorFlow。**

`fago-core`和`full-default`特性组合包含FAGO静态图融合算子，其算子实现依赖CUDA，必须构建GPU版本。`common-only`和`eigen-core`组合不包含GPU算子，使用上述CPU流程构建即可。

#### 环境要求

除“[准备构建环境](#准备构建环境)”的基础依赖外，GPU版本还需要以下环境。

| 项目 | 版本或要求 |
| --- | --- |
| CUDA | 12.8.0 |
| CuDNN | 9.5.0 |

CUDA安装路径建议为`/usr/local/cuda-12.8`。

1. 安装CUDA。

   1. 检查硬件与系统：检查显卡是否存在，并安装依赖。

      ```bash
      lspci | grep -i nvidia
      yum install -y gcc make dkms
      yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
      ```

      确认GPU支持CUDA：访问“[NVIDIA CUDA-GPU列表](https://developer.nvidia.com/cuda-gpus)”确认你的GPU支持CUDA。

   2. 禁用nouveau驱动。

      检查系统是否加载nouveau驱动，若有输出则代表启用了nouveau驱动，需要禁用nouveau驱动。

      ```bash
      lsmod | grep nouveau
      ```

   3. 下载CUDA工具包。

      CUDA 12.8官网下载链接“[CUDA Toolkit 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=RHEL&target_version=8)”。

      下载对应版本的CUDA工具包，且仅需下载CUDA工具包，无需单独下载GPU驱动。CUDA工具包`cuda_12.8.0_***_linux_sbsa.run`内含GPU驱动。

   4. 安装CUDA工具包。

      安装工具包，赋予可执行权限并运行.run文件，根据提示完成安装。

      ```bash
      ./cuda_12.8.0_***_linux_sbsa.run
      ```

      添加环境变量。安装完成后，需要将CUDA路径添加到系统环境变量（建议采用默认安装路径`/usr/local/cuda-12.8`）。

   5. 验证安装。

      检查CUDA版本，若输出CUDA 12.8.0相关信息，则安装成功。

      ```bash
      nvcc --version
      ```

#### 配置bazel

1. 开启CUDA支持。

   ```bash
   cd /path/to/tensorflow
   ./configure
   ```

   在交互式配置中，`Do you wish to build TensorFlow with CUDA support? [y/N]`选择`y`，并按提示完成其后各项CUDA参数配置，未显示N或n的选项使用回车`[Enter]`确认。关键配置项如下。

   ```bash
   Do you wish to build TensorFlow with ROCm support? [y/N]: n
   No ROCm support will be enabled for TensorFlow.

   Do you wish to build TensorFlow with CUDA support? [y/N]: y
   CUDA support will be enabled for TensorFlow.

   Please specify the hermetic CUDA version you want to use or leave empty to use the default version. 12.8.0


   Please specify the hermetic cuDNN version you want to use or leave empty to use the default version. 9.5.0


   Please specify a list of comma-separated CUDA compute capabilities you want to build with.
   You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
   Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 8.9 


   Please specify the local CUDA path you want to use or leave empty to use the default version. 


   Please specify the local CUDNN path you want to use or leave empty to use the default version. 


   Please specify the local NCCL path you want to use or leave empty to use the default version. 


   Do you want to use clang as CUDA compiler? [Y/n]: n
   nvcc will be used as CUDA compiler.

   Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 
   ```

2. 打开`.tf_configure.bazelrc`，在`build --action_env LD_LIBRARY_PATH=...`行前添加以下build配置。

   ```bash
   build:cuda --repo_env TF_NEED_CUDA=1
   build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
   build:cuda --@local_config_cuda//:enable_cuda
   build:cuda --config=cuda_version
   ```

#### 构建GPU版本pip包

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  --experimental_repo_remote_exec \
  --verbose_failures \
  --config=cuda --config=cuda_nvcc --config=cuda_wheel \
  --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 \
  --copt=-march=armv8.5-a --linkopt=-fuse-ld=gold \
  --config=opt \
  --action_env=CLANG_CUDA=0 \
  --repo_env=CLANG_CUDA=0 \
  --repo_env=PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  --repo_env=PIP_TRUSTED_HOST=mirrors.aliyun.com \
  --@local_config_cuda//cuda:override_include_cuda_libs=true \
  --repo_env=CC=/usr/bin/gcc \
  --repo_env=CXX=/usr/bin/g++ \
  --repo_env=CUDA_PATH=/usr/local/cuda-12.8 \
  //tensorflow/tools/pip_package:wheel
```

安装方式同“[安装tensorflow pip包](#安装tensorflow-pip包)”。

## 验证特性

>![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
>
>需完成构建TensorFlow pip包并安装后，才能验证特性。请勿在TensorFlow源码根目录下执行验证命令，否则Python会优先加载源码目录下的`tensorflow/`包而非已安装的pip包，导致导入报错。

### ANNC静态图融合

#### CPU 算子验证

进入测试目录后，执行测试命令。

```bash
cd tensorflow/python/grappler/embedding_fused_test
python main.py --list
python main.py --op {op_name}
```

ANNC CPU 算子静态图融合只对满足特定结构和输入约束的子图生效。

#### GPU 算子验证

进入测试目录后，执行测试命令。

```bash
cd tensorflow/python/grappler/flash_attn_opt_test
python main.py --list
python main.py --op {op_name}
```

ANNC GPU 算子（FlashAttentionForward）静态图融合只对满足特定结构和输入约束的子图生效，需 CUDA 环境支持。

### KDNN

1. 进入测试目录并查询支持的模块。

   ```bash
   cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark
   python main.py --list
   ```

2. 从查询结果中选择算子并运行测试。

   ```bash
   numactl -C 0-15 python main.py --op {op_name} --performance_test True
   ```

   执行通过即表示相关算子集成成功。

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

1. 编译时将`--tensorflow_dir`指向本指南生成的TensorFlow完整源码。

   ```bash
   cd /path/to/serving
   sh compile_serving.sh \
     --tensorflow_dir /path/to/tensorflow \
     --features gcc12
   ```

2. 检查构建产物。

   ```output
   /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
   ```

TensorFlow Serving通过本地TensorFlow源码完成集成。更换特性组合时，无需重新整理Serving源码，只需重新生成对应TensorFlow源码、准备其依赖并重新构建Serving。

## 故障案例

编译TensorFlow时，可参考以下故障处理指导。

### 提示找不到libnvrtc-builtins.so.12.5文件报错

#### 问题现象描述

Tensorflow编译安装后，python import tensorflow报错，提示“libnvrtc-builtins.so.12.5: cannot open shared object file: No such file or directory”。详细信息如下所示。

```bash
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "/usr/local/lib64/python3.11/site-packages/tensorflow/__init__.py", line 40, in <module>
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow # pylint: disable=unused-import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib64/python3.11/site-packages/tensorflow/python/pywrap_tensorflow.py", line 37, in <module>
self_check.preload_check()
File "/usr/local/lib64/python3.11/site-packages/tensorflow/python/platform/self_check.py", line 63, in preload_check
from tensorflow.python.platform import _pywrap_cpu_feature_guard
ImportError: libnvrtc-builtins.so.12.5: cannot open shared object file: No such file or directory
```

#### 根本原因分析

TensorFlow构建脚本中硬编码了libnvrtc-builtins.so.12.5作为依赖版本，但实际安装的是CUDA 12.8，对应库文件为libnvrtc-builtins.so.12.8，运行时动态链接器找不到对应版本的库，触发导入失败。

#### 解决方案及效果

NVIDIA保证CUDA同大版本（12.x）内的nvrtc-builtins库二进制向下兼容，直接创建版本软链接即可正常运行，不影响功能与性能。

1. 执行以下命令进行修复。

    ```bash
    cd /usr/local/cuda-12.8/lib64
    ln -sf libnvrtc-builtins.so.12.8 libnvrtc-builtins.so.12.5
    ldconfig
    ```

2. 验证修复。

    ```bash
    python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
    ```

执行上述命令后，不会再报错，可以正常输出tensorflow系统路径。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 02 | 2026-12-30 | 第二次正式发布。<ul><li>新增TensorFlow KDNN算子优化特性的适配环境和安装指导内容。</li><li>新增TensorFlow ANNC静态图融合特性的适配环境和安装指导内容。</li><li>新增KEmbedding自定义算子库的适配环境和安装指导内容。</li></ul> |
| 01 | 2026-09-30 | 第一次正式发布。<ul><li>新增TensorFlow 2.20.0的安装步骤内容。</li><li>新增TensorFlow FAGO静态图融合特性的适配环境和安装指导内容。</li></ul> |
