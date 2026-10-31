# Installation Guide

This document describes the installation process in the following order: preparing the TensorFlow source code, building TensorFlow artifacts, and verifying the features.

## Preparing the Build Environment

The build environment for the currently maintained version is as follows.

| Item | Version or Requirement |
| --- | --- |
| CPU | Kunpeng 950 7592C processor |
| OS | openEuler 24.03 LTS SP3 |
| GCC/G++ | 12.3.1 |
| Bazel | 7.4.1 |
| Python | 3.11.x |

Install the basic dependencies.

   ```bash
   yum install -y gcc gcc-c++ git patch patchelf perl \
   python3 python3-devel tar unzip wget zip
   ```

TensorFlow is built using Bazel 7.4.1. For the Bazel installation method, see the "[Installing Bazel](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html)" section in the *TensorFlow Porting Guide*.

>![Note icon](public_sys-resources/icon-note.gif) **Note:**
>
>The default installation procedure builds the CPU version. To build TensorFlow with FAGO static graph fusion support, see the "[Building the GPU Version](#building-the-gpu-version)" section, which additionally requires CUDA and cuDNN.

## Preparing TensorFlow Source Code

### Selecting a Feature Combination

| Category | Feature Combination | Contents | Corresponding Patches (in Application Order) | Applicable Scenario |
| --- | --- | --- | --- | --- |
| CPU | common-only | Common build integration (common) | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch | Validate common changes without acceleration features. |
| CPU | eigen-core | Common build integration (common), Eigen KGemm MatMul acceleration | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0005-tensorflow_2.20.0-eigen-kgemm.patch | Use Eigen KGemm MatMul acceleration. |
| GPU | fago-core | Common build integration (common), FAGO static graph fusion | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-fago.patch | Use FAGO static graph fusion. |
| GPU | full-default | Common build integration (common), FAGO static graph fusion, Eigen KGemm MatMul acceleration | patches/Tensorflow_V2.20.0/feature/0001-tensorflow_2.20.0-common.patch<br>patches/Tensorflow_V2.20.0/feature/0002-tensorflow_2.20.0-fago.patch<br>patches/Tensorflow_V2.20.0/feature/0005-tensorflow_2.20.0-eigen-kgemm.patch | Use all currently maintained features. |

> ![Note icon](public_sys-resources/icon-note.gif) **Note:**
>
> CPU and GPU features are independent, and only the CPU features are integrated by default, so they can be built with the CPU build flow. The GPU features include FAGO static graph fusion, which requires GPU capability; complete the configuration in "[Building the GPU Version](#building-the-gpu-version)" before building them. Compiling `fago-core` or `full-default` with the CPU build flow fails. To use only the CPU acceleration, select `common-only` or `eigen-core`.

### Generating Complete Source Code

1. Download the patch repository and obtain the official TensorFlow baseline.

   ```bash
   git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
   cd kunpeng-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0
   ```

2. Create the complete TensorFlow source code as needed. The following example uses the CPU feature combination `eigen-core`.

   ```bash
   python3 patches/Tensorflow_V2.20.0/prepare_source.py \
     --feature-set eigen-core \
     --output-dir /path/to/tensorflow
   ```

   For other combinations, simply modify `--feature-set`, which accepts the feature combinations listed in the table above. Before using a GPU feature combination, complete the GPU environment configuration described in "[Building the GPU Version](#building-the-gpu-version)". `/path/to/tensorflow` indicates the TensorFlow source root directory. The output directory contains the complete source code of the official TensorFlow `v2.20.0` and the selected patches.

3. Create the build directories in the TensorFlow source root directory.

   ```bash
   cd /path/to/tensorflow
   mkdir -p output ../repo_cache ../download
   ```

   `output/` is used to reuse the Bazel build cache, `../repo_cache` caches external dependency repositories, and `../download` stores manually downloaded build dependency packages.

## Building TensorFlow Artifacts

When the inference service is not used, TensorFlow targets can also be built on demand directly from the generated source code. This section describes the CPU build flow and applies to the `common-only` and `eigen-core` feature combinations. To use `fago-core` or `full-default`, see "[Building the GPU Version](#building-the-gpu-version)".

### Building the TensorFlow pip Package

1. Configure the build.

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

2. Build the TensorFlow pip package.

   ```bash
   cd /path/to/tensorflow
   bazel --output_user_root="$PWD/output" build \
   --jobs=64 \
   --experimental_repo_remote_exec \
   --repository_cache="$PWD/../repo_cache" \
   --distdir="$PWD/../download" \
   --cxxopt=-std=c++17 \
   --host_cxxopt=-std=c++17 \
   --copt=-march=armv8.5-a \
   --host_copt=-march=armv8.5-a \
   --linkopt=-Wl,--stub-group-size=0x2000000 \
   --check_direct_dependencies=off \
   -c opt \
   //tensorflow/tools/pip_package:wheel
   ```

   For the `eigen-core` feature combination, append `--copt=-DEIGEN_NEON_USE_KGEMM=1` to enable Eigen KGemm MatMul acceleration. If the GPU feature combination `full-default` also requires this acceleration, append the same option to the build command in "[Building the GPU Version](#building-the-gpu-version)".

   ```bash
   bazel --output_user_root="$PWD/output" build \
   --jobs=64 \
   --experimental_repo_remote_exec \
   --repository_cache="$PWD/../repo_cache" \
   --distdir="$PWD/../download" \
   --cxxopt=-std=c++17 \
   --host_cxxopt=-std=c++17 \
   --copt=-march=armv8.5-a \
   --host_copt=-march=armv8.5-a \
   --copt=-DEIGEN_NEON_USE_KGEMM=1 \
   --linkopt=-Wl,--stub-group-size=0x2000000 \
   --check_direct_dependencies=off \
   -c opt \
   //tensorflow/tools/pip_package:wheel
   ```

3. After the build is complete, check the artifact directory.

   ```bash
   ls bazel-bin/tensorflow/tools/pip_package/wheel_house/
   ```

   When the output is similar to the following, the pip package is built successfully.

   ```output
   tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
   ```

### Installing the TensorFlow pip Package

```bash
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.20.0.dev0+selfbuilt-cp311-cp311-linux_aarch64.whl
```

### Building the GPU Version

**To build TensorFlow with FAGO support, build TensorFlow with GPU support enabled.**

The `fago-core` and `full-default` feature combinations contain FAGO static graph fusion operators, whose implementation depends on CUDA, so a GPU-enabled build is mandatory. The `common-only` and `eigen-core` combinations do not contain GPU operators and can be built with the CPU procedure above.

#### Environment Requirements

In addition to the basic dependencies described in "[Preparing the Build Environment](#preparing-the-build-environment)", the GPU version requires the following environment.

| Item | Version or Requirement |
| --- | --- |
| CUDA | 12.8.0 |
| cuDNN | 9.5.0 |

The recommended CUDA installation path is `/usr/local/cuda-12.8`.

1. Install CUDA.

   1. Check the hardware and system: check whether a GPU is present and install the dependencies.

      ```bash
      lspci | grep -i nvidia
      yum install -y gcc make dkms
      yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
      ```

      Confirm that the GPU supports CUDA: visit the "[NVIDIA CUDA-GPU list](https://developer.nvidia.com/cuda-gpus)" to confirm that your GPU supports CUDA.

   2. Disable the nouveau driver.

      Check whether the nouveau driver is loaded. If there is output, the nouveau driver is enabled and must be disabled.

      ```bash
      lsmod | grep nouveau
      ```

   3. Download the CUDA toolkit.

      CUDA 12.8 download link: "[CUDA Toolkit 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=RHEL&target_version=8)".

      Download the CUDA toolkit of the corresponding version. Only the CUDA toolkit is required; the GPU driver does not need to be downloaded separately. The CUDA toolkit `cuda_12.8.0_***_linux_sbsa.run` contains the GPU driver.

   4. Install the CUDA toolkit.

      Grant the executable permission to the toolkit, run the .run file, and complete the installation as prompted.

      ```bash
      ./cuda_12.8.0_***_linux_sbsa.run
      ```

      Add the environment variables. After the installation is complete, add the CUDA path to the system environment variables (the default installation path `/usr/local/cuda-12.8` is recommended).

   5. Verify the installation.

      Check the CUDA version. If information about CUDA 12.8.0 is displayed, the installation is successful.

      ```bash
      nvcc --version
      ```

#### Configuring the Build

1. Enable CUDA support.

   ```bash
   cd /path/to/tensorflow
   ./configure
   ```

   In the interactive configuration, answer `y` to `Do you wish to build TensorFlow with CUDA support? [y/N]` and complete the subsequent CUDA parameter configuration as prompted. Press `[Enter]` to confirm options that do not show N or n. The key configuration items are as follows.

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

2. Open `.tf_configure.bazelrc` and add the following build configuration before the `build --action_env LD_LIBRARY_PATH=...` line.

   ```bash
   build:cuda --repo_env TF_NEED_CUDA=1
   build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
   build:cuda --@local_config_cuda//:enable_cuda
   build:cuda --config=cuda_version
   ```

#### Building the GPU Version pip Package

```bash
cd /path/to/tensorflow
bazel --output_user_root="$PWD/output" build \
  --repository_cache="$PWD/../repo_cache" \
  --distdir="$PWD/../download" \
  --experimental_repo_remote_exec \
  --verbose_failures \
  --config=cuda --config=cuda_nvcc --config=cuda_wheel \
  --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 \
  --copt=-march=armv8.5-a \
  --host_copt=-march=armv8.5-a \
  --linkopt=-fuse-ld=gold --linkopt=-Wl,--stub-group-size=0x2000000 \
  --config=opt \
  --action_env=CLANG_CUDA=0 \
  --repo_env=CLANG_CUDA=0 \
  --repo_env=PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  --repo_env=PIP_TRUSTED_HOST=mirrors.aliyun.com \
  --@local_config_cuda//cuda:override_include_cuda_libs=true \
  --repo_env=CC=/usr/bin/gcc \
  --repo_env=CXX=/usr/bin/g++ \
  --repo_env=CUDA_PATH=/usr/local/cuda-12.8 \
  --check_direct_dependencies=off \
  //tensorflow/tools/pip_package:wheel
```

The installation method is the same as that described in "[Installing the TensorFlow pip Package](#installing-the-tensorflow-pip-package)".

## Feature Verification

>![Note icon](public_sys-resources/icon-note.gif) **Note:**
>
>The features can be verified only after the TensorFlow pip package is built and installed. Do not run the verification commands in the TensorFlow source root directory, otherwise Python loads the `tensorflow/` package in the source directory instead of the installed pip package, causing an import error.

### FAGO Static Graph Fusion

The GPU version of TensorFlow must be installed. Navigate to the test directory and execute the test command.

```bash
cd tensorflow/python/grappler/flash_attn_opt_test
python main.py --list
python main.py --op {op_name} --performance_test False > result.txt
```

FAGO static graph fusion takes effect only on subgraphs that meet specific structural and input constraints.

### Eigen KGemm MatMul Acceleration

The CPU version is sufficient for verification. Eigen KGemm acceleration takes effect on 2-D `tf.matmul` operands with contiguous memory and a single contraction dimension. The larger the weight matrix (K), the more significant the acceleration. For a GPU build artifact, the scripts below must explicitly target the CPU; otherwise MatMul runs on the GPU and Eigen KGemm is not exercised.

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> When KDNN operator optimization and Eigen KGemm MatMul acceleration are enabled at the same time (for example, using the `full-default` profile and appending `--copt=-DEIGEN_NEON_USE_KGEMM=1` to the build command), fp32 MatMul is handled by KDNN first and KGemm does not take effect. The two do not conflict, and the computation result is not affected. To accelerate MatMul with KGemm, use the `eigen-core` profile, which does not contain KDNN, or set the `TF_ENABLE_KDNN_OPTS=0` environment variable at runtime to disable KDNN (this also disables other KDNN operator optimizations such as Softmax).

1. Verify the correctness of the MatMul result.

   ```bash
   cd ~
   python3 -c "
   import numpy as np, tensorflow as tf
   rng = np.random.RandomState(0)
   a = rng.normal(0, 0.1, size=(59, 1570)).astype(np.float32)
   b = rng.normal(0, 0.1, size=(1570, 256)).astype(np.float32)
   with tf.device('/CPU:0'):
       got = tf.matmul(tf.constant(a), tf.constant(b))
   print('device =', got.device)
   got = got.numpy()
   exp = np.matmul(a, b)
   print('max_abs_err =', np.max(np.abs(got - exp)))
   print('allclose =', np.allclose(got, exp, rtol=1e-4, atol=1e-5))
   "
   ```

   When the output is similar to the following, the MatMul result is correct.

   ```output
   device = /job:localhost/replica:0/task:0/device:CPU:0
   max_abs_err = 1.132e-06
   allclose = True
   ```

2. Compare the acceleration effect.

   ```bash
   cd ~
   python3 -c "
   import time, numpy as np, tensorflow as tf
   tf.config.threading.set_intra_op_parallelism_threads(16)
   rng = np.random.RandomState(0)
   with tf.device('/CPU:0'):
       a = tf.constant(rng.normal(0, 0.1, size=(59, 1570)).astype(np.float32))
       b = tf.constant(rng.normal(0, 0.1, size=(1570, 256)).astype(np.float32))
       f = tf.function(lambda: tf.matmul(a, b)).get_concrete_function()
   print('device =', f().device)
   for _ in range(500):
       f()
   t = time.perf_counter()
   for _ in range(100):
       f()
   print('avg_us = %.1f' % ((time.perf_counter() - t) / 100 * 1e6))
   "
   ```

   The time consumed by build artifacts with Eigen KGemm acceleration enabled and disabled can be compared. Taking the `59x1570x256` shape as an example, the time consumed with acceleration enabled is about half of that with acceleration disabled.

3. Run the performance test script to compare the two build artifacts.

   The `tensorflow/python/kernel_tests/benchmark` directory of the TensorFlow source code provides the performance test scripts. The script installs the baseline wheel and runs all cases, then installs the candidate wheel and repeats, and finally writes a comparison report. `--warmup` should be 100 or more; otherwise the first samples are inflated.

   If a GPU build artifact is used, mask the GPU so that MatMul runs on the CPU.

   ```bash
   cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark

   CUDA_VISIBLE_DEVICES= python3 compare_wheels.py \
     --python /usr/bin/python3 \
     --baseline-wheel  /path/to/dist/tensorflow-nokgemm-*.whl \
     --baseline-label  nokgemm \
     --candidate-wheel /path/to/dist/tensorflow-kgemm-*.whl \
     --candidate-label kgemm \
     --intra-threads 16 \
     --warmup 500 \
     --iterations 100 \
     --samples 7 \
     --output-dir ./kgemm-vs-nokgemm
   ```

   `comparison.md` lists the time and speedup of every shape, and `summarize_comparison.py` groups the results by model.

   ```bash
   python3 summarize_comparison.py ./kgemm-vs-nokgemm/comparison.md
   ```

   > ![Note icon](public_sys-resources/icon-note.gif) **Note:**
   >
   > Bind the test process to CPUs within a single NUMA node with `taskset`, and keep `--intra-threads` equal to that CPU count to reduce jitter caused by cross-node memory access.

   `tensorflow/python/kernel_tests/math_ops/matmul_model_shapes_test.py` checks the correctness of the same set of shapes and can be run with `bazel test //tensorflow/python/kernel_tests/math_ops:matmul_model_shapes_test_cpu`.

## FAQs

When compiling TensorFlow, refer to the following troubleshooting case.

### Missing libnvrtc-builtins.so.12.5

>![Note icon](public_sys-resources/icon-note.gif) **Note:**
>
>This issue occurs only when building the GPU version.

**Symptom**

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

**Key Process and Root Cause Analysis**

The TensorFlow build script hardcodes `libnvrtc-builtins.so.12.5` as the dependency version, but the actually installed version is CUDA 12.8, whose corresponding library file is `libnvrtc-builtins.so.12.8`. At runtime, the dynamic linker cannot find the library of the expected version, causing the import to fail.

**Conclusion, Solution, and Effect**

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

| Document Version | Release Date | Description |
| ---- | ---- | ---- |
| 02 | 2026-12-30 | Second official release.<ul><li>The default installation procedure is changed to the CPU version, and GPU-related content is moved to the "Building the GPU Version" section.</li><li>Added the installation instructions and verification steps for the Eigen KGemm MatMul acceleration feature.</li><li>Fixed a typo in the nouveau driver disabling command.</li></ul> |
| 01 | 2026-09-30 | First official release. |
