# Quick Start

<!-- md-trans-meta sourceCommit=7cc97f4b89994805c365839c45d47415fd06ffdb translatedAt=2026-10-12T06:48:30.720Z pushedAt=2026-10-13T02:43:28.000Z -->

## Installing the Kunpeng TensorFlow

1. Obtain the Kunpeng TensorFlow patch repository.

   ```bash
   git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
   cd kunpeng-tensorflow
   ```

2. Obtain the official TensorFlow v2.20.0 baseline.

   ```bash
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0
   ```

3. Create the complete TensorFlow source code with default features.

   ```bash
   python3 patches/Tensorflow_V2.20.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   ```

   For usage instructions of other available combinations, see [Patch Release](../../../patches/Tensorflow_V2.20.0/README_en.md).

4. Select the required build target based on your actual use scenario. You do not need to build all targets at the same time. The following provides the reference commands for the pip build target.

   4.1 Configure Bazel.

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
      Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 8.9


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

   4.2 Open `.tf_configure.bazelrc` and add the following build configuration before the line `build --action_env LD_LIBRARY_PATH=...`.

      ```bash
      build:cuda --repo_env TF_NEED_CUDA=1
      build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
      build:cuda --@local_config_cuda//:enable_cuda
      build:cuda --config=cuda_version
      ```

   4.3 Build the TensorFlow pip package.

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

   4.4 Install the TensorFlow pip package.

     ```bash
     pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
     ```

5. Call TensorFlow interfaces.

   5.1 C++ interface call. The expected output is the system dynamic libraries that the TensorFlow C++ shared library depends on.

     ```bash
     ldd /usr/local/lib64/python3.11/site-packages/tensorflow/libtensorflow_cc.so.2
     ```

   5.2 Python interface call. The expected output is the TensorFlow lib path.

     ```bash
     cd ..
     python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
     ```

If you encounter any problem during the compilation, see [TensorFlow Porting Guide](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0001.html).

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-09-30 | This is the first official release. |
