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

安装基础依赖并启用CUDA，CUDA安装路径建议为/usr/local/cuda-12.8。

TensorFlow V2.20.0使用官方指定的Bazel 7.4.1构建。Bazel安装方法请参见《TensorFlow 移植指南》的“[安装Bazel](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html)”章节。

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
| common-fago | 公共构建集成（common）、FAGO静态图融合 | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-fago.patch | 使用FAGO静态图融合。 |

### 生成完整源码

1. 下载补丁仓并获取官方TensorFlow基线。

   ```bash
   git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
   cd kunpeng-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0
   ```

2. 根据需要创建TensorFlow完整源码。以下以完整默认特性为例。

   ```bash
   python3 patches/Tensorflow_V2.20.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   ```

   如需其他组合，只需修改`--feature-set`。输出目录包含官方TensorFlow `v2.20.0`完整源码、所选补丁。

## 构建TensorFlow产物

不使用推理服务时，也可以直接在生成的源码中按需构建TensorFlow目标。

### 构建TensorFlow pip包

1. 配置bazel。

   ```bash
   cd ../tensorflow-full-default
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

   Do you wish to build TensorFlow with CUDA support? [y/N]: y
   CUDA support will be enabled for TensorFlow.

   Please specify the hermetic CUDA version you want to use or leave empty to use the default version. 12.8.0


   Please specify the hermetic cuDNN version you want to use or leave empty to use the default version. 9.5.0


   Please specify a list of comma-separated CUDA compute capabilities you want to build with.
   You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
   Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 


   Please specify the local CUDA path you want to use or leave empty to use the default version. 


   Please specify the local CUDNN path you want to use or leave empty to use the default version. 


   Please specify the local NCCL path you want to use or leave empty to use the default version. 


   Do you want to use clang as CUDA compiler? [Y/n]: n
   nvcc will be used as CUDA compiler.

   Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


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
  
2. 打开`.tf_configure.bazelrc`，在`build --action_env LD_LIBRARY_PATH=...`行前添加以下build 配置。

   ```bash
   build:cuda --repo_env TF_NEED_CUDA=1
   build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
   build:cuda --@local_config_cuda//:enable_cuda
   build:cuda --config=cuda_version
   ```

3. 构建TensorFlow pip包。

   ```bash
   bazel build \
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

### 安装tensorflow pip包

```bash
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
```

## 验证特性

### FAGO静态图融合

进入测试目录后，执行测试命令。

```bash
cd tensorflow/python/grappler/flash_attn_opt_test
python main.py --list
python main.py --op {op_name} --performance_test False > result.txt
```

FAGO静态图融合只对满足特定结构和输入约束的子图生效。

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

TensorFlow 构建脚本中硬编码了 libnvrtc-builtins.so.12.5 作为依赖版本，但实际安装的是 CUDA 12.8，对应库文件为 libnvrtc-builtins.so.12.8，运行时动态链接器找不到对应版本的库，触发导入失败。

#### 解决方案及效果

NVIDIA 保证 CUDA 同大版本（12.x）内的 nvrtc-builtins 库二进制向下兼容，直接创建版本软链接即可正常运行，不影响功能与性能。

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

| 文档版本 | 发布日期 | 修订记录 |
| ---- | ---- | ---- |
| 01 | 2026-09-30 | 第一次正式发布。|
