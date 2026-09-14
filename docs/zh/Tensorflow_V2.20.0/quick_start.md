# 快速入门

## 安装鲲鹏TensorFlow

1. 获取鲲鹏TensorFlow补丁仓。

   ```bash
   git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
   cd kunpeng-tensorflow
   ```

2. 获取官方TensorFlow v2.20.0基线。

   ```bash
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0
   ```

3. 创建完整默认特性的TensorFlow源码。

   ```bash
   python3 patches/Tensorflow_V2.20.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   ```

   其他可用组合的使用指导，请参见《[补丁发布说明](../../../patches/Tensorflow_V2.20.0/README.md)》。

4. 根据实际使用场景选择需要的构建目标，无需同时构建全部目标。以下提供pip构建目标的参考命令。

   4.1. 配置bazel。

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

   4.2. 打开`.tf_configure.bazelrc`，在`build --action_env LD_LIBRARY_PATH=...`行前添加以下build 配置。

      ```bash
      build:cuda --repo_env TF_NEED_CUDA=1
      build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
      build:cuda --@local_config_cuda//:enable_cuda
      build:cuda --config=cuda_version
      ```

   4.3. 构建TensorFlow pip包。

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

   4.4. 安装TensorFlow pip包。
  
     ```bash
     pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
     ```

5. 调用Tensorflow接口。

   5.1. C++接口调用，预期可以输出Tensorflow C++共享库依赖的系统动态库。

     ```bash
     ldd /usr/local/lib64/python3.11/site-packages/tensorflow/libtensorflow_cc.so.2
     ```

   5.2. Python接口调用，预期可以输出tensorflow lib路径。

     ```bash
     cd ..
     python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
     ```

如果在编译过程中遇到任何问题，请参见《[TensorFlow 移植指南](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0001.html)》。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 01 | 2026-09-30 | 第一次正式发布。 |
