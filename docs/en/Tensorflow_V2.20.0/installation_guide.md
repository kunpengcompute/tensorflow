# Installation Guide

<!-- md-trans-meta sourceCommit=7cc97f4b89994805c365839c45d47415fd06ffdb translatedAt=2026-10-12T06:48:38.195Z pushedAt=2026-10-13T03:37:46.244Z -->

This document describes the installation process in the following order: preparing the TensorFlow source code, building the inference service, and verifying the features.

## Preparing the Build Environment

The build environment for the currently maintained version is as follows.

| Item | Version or Requirement |
| --- | --- |
| CPU | Kunpeng 950 processor |
| OS | openEuler 24.03 LTS SP3 |
| GCC/G++ | 12.3.1 |
| Bazel | 7.4.1 |
| Python | 3.11.x |
| CUDA  | 12.8.0 |
| cuDNN | 9.5.0 |

Install the basic dependencies and enable CUDA. The recommended CUDA installation path is `/usr/local/cuda-12.8`.

TensorFlow V2.20.0 is built using the officially specified Bazel 7.4.1. For the Bazel installation method, see the "[Installing Bazel](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html)" section in the *TensorFlow Porting Guide*.

### CUDA Installation Guide

#### Checking Hardware and System

  1. Check whether any GPU exists.

     ```bash
     lspci | grep -i nvidia
     ```

  2. Install dependencies.

     ```bash
     yum install -y gcc make dkms
     yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
     ```

  3. Visit the [NVIDIA CUDA GPU Compute Capability List](https://developer.nvidia.com/cuda-gpus) to confirm that your GPU supports CUDA.

#### Installing CUDA

  1. Disable the nouveau driver.
  
     Check whether the system has loaded the nouveau driver. If there is any output, the nouveau driver is enabled and must be disabled.

     ```bash
     smod | grep nouveau
     ```
  
  2. Download the CUDA toolkit.

     Official CUDA 12.8 download link: [CUDA Toolkit 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=RHEL&target_version=8)

     Download the CUDA toolkit for the corresponding version. Only the CUDA toolkit needs to be downloaded; there is no need to download the GPU driver separately. The CUDA toolkit `cuda_12.8.0_***_linux_sbsa.run` includes the GPU driver.

  3. Install the CUDA toolkit.

     Install the toolkit, grant executable permissions and run the .run file, then follow the prompts to complete the installation.

     ```bash
     ./cuda_12.8.0_***_linux_sbsa.run
     ```

     Add environment variables. After installation is complete, add the CUDA path to the system environment variables (the default installation path `/usr/local/cuda-12.8` is recommended).

  4. Verify the installation.
  
     Check the CUDA version. If CUDA 12.8.0-related information is output, the installation is successful.

     ```bash
     nvcc --version
     ```

## Preparing TensorFlow Source Code

### Selecting a Feature Combination

| Feature Combination | Contents | Corresponding Patches (in Application Order) | Applicable Scenario |
| --- | --- | --- | --- |
| common-fago | Common build integration (common), FAGO static graph fusion | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-fago.patch | Use FAGO static graph fusion |

### Generating Complete Source Code

1. Download the patch repository and obtain the official TensorFlow baseline.

   ```bash
   git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
   cd kunpeng-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0
   ```

2. Create the complete TensorFlow source code as needed. The following example uses the full default feature set.

   ```bash
   python3 patches/Tensorflow_V2.20.0/prepare_source.py \
     --feature-set full-default \
     --output-dir ../tensorflow-full-default
   ```

   For other combinations, simply modify `--feature-set`. The output directory contains the complete source code of the official TensorFlow `v2.20.0` and the selected patches.

## Building TensorFlow Artifacts

When the inference service is not used, TensorFlow targets can also be built on demand directly from the generated source code.

### Building the TensorFlow pip Package

1. Configure Bazel.

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
  
2. Open `.tf_configure.bazelrc` and add the following build configuration before the line `build --action_env LD_LIBRARY_PATH=...`.

   ```bash
   build:cuda --repo_env TF_NEED_CUDA=1
   build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
   build:cuda --@local_config_cuda//:enable_cuda
   build:cuda --config=cuda_version
   ```

3. Build the TensorFlow pip package.

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

### Installing the TensorFlow pip Package

```bash
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
```

## Feature Verification

### FAGO Static Graph Fusion

Navigate to the test directory and execute the test command.

```bash
cd tensorflow/python/grappler/flash_attn_opt_test
python main.py --list
python main.py --op {op_name} --performance_test False > result.txt
```

FAGO static graph fusion takes effect only on subgraphs that meet specific structural and input constraints.

## Troubleshooting Cases

When compiling TensorFlow, refer to the following troubleshooting case.

### Missing libnvrtc-builtins.so.12.5

#### Symptom

After TensorFlow is compiled and installed, running `python import tensorflow` reports an error, prompting "libnvrtc-builtins.so.12.5: cannot open shared object file: No such file or directory". The detailed information is as follows.

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

#### Root Cause Analysis

The TensorFlow build script hardcodes `libnvrtc-builtins.so.12.5` as the dependency version, but the actually installed version is CUDA 12.8, whose corresponding library file is `libnvrtc-builtins.so.12.8`. At runtime, the dynamic linker cannot find the library of the expected version, causing the import to fail.

#### Solution and Effect

NVIDIA guarantees binary backward compatibility of the nvrtc-builtins library within the same major CUDA version (12.x). Creating a version symlink directly allows normal operation without affecting functionality or performance.

1. Run the following command to fix the issue.

    ```bash
    cd /usr/local/cuda-12.8/lib64
    ln -sf libnvrtc-builtins.so.12.8 libnvrtc-builtins.so.12.5
    ldconfig
    ```

2. Verify the fix.

    ```bash
    python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
    ```

After running the above commands, no error is reported, and the TensorFlow system path can be output normally.

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-09-30 | This is the first official release. |
