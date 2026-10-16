# Installation Guide

<!-- md-trans-meta sourceCommit=e98cf0d4e88f1b0406fb0a2fff9214e7c26cad25 translatedAt=2026-10-13T11:08:15.888Z pushedAt=2026-10-13T12:33:28.005Z -->

This document describes the installation process in the following order: preparing the TensorFlow source code, building the inference service, and verifying the features.

## Preparing the Build Environment

The build environment for the currently maintained version is as follows.

| Item | Version or Requirement |
| --- | --- |
| CPU | <ul><li>New Kunpeng 920 processor model</li><li>Kunpeng 950 processor</li></ul> |
| OS | <ul><li>openEuler 22.03 LTS SP3</li><li>openEuler 24.03 LTS SP3</li></ul> |
| GCC/G++ | 12.3.1 |
| Bazel | 6.5.0 |
| Python | 3.11.x |

Install the basic dependencies and enable GCC 12.3.1.

   ```bash
   yum install -y gcc-toolset-12-gcc* git patch patchelf perl \
   python3 python3-devel tar unzip wget zip
   export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
   export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
   ```

Both TensorFlow and TensorFlow Serving are built using Bazel 6.5.0. For the Bazel installation method, see [Installing Bazel](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0009.html).

## Preparing TensorFlow Source Code

### Selecting a Feature Combination

| Feature Combination | Contents | Corresponding Patches (in Application Order) | Applicable Scenario |
| --- | --- | --- | --- |
| common-only | Public build integration (common) and compatibility changes | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch | Inspect shared changes without acceleration features |
| kdnn-core | common + KDNN | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch<br>patches/Tensorflow_V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch | Use KDNN kernel optimizations |
| kdnn-annc | common + KDNN + ANNC | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch<br>patches/Tensorflow_V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch<br>patches/Tensorflow_V2.15.0/feature/0003-tensorflow_2.15.0-annc.patch | Use KDNN and ANNC static graph fusion |
| full-default | common + KDNN + ANNC + KEmbedding | patches/Tensorflow_V2.15.0/feature/0001-tensorflow_2.15.0-common.patch<br>patches/Tensorflow_V2.15.0/feature/0002-tensorflow_2.15.0-kdnn.patch<br>patches/Tensorflow_V2.15.0/feature/0003-tensorflow_2.15.0-annc.patch<br>patches/Tensorflow_V2.15.0/feature/0004-tensorflow_2.15.0-kembedding.patch | Use all currently maintained features |

### Generating Complete Source Code

1. Download the patch repository and obtain the official TensorFlow baseline.

   ```bash
   git clone https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

2. Create the complete TensorFlow source code as needed. The following example uses the full default feature set.

   ```bash
   python3 patches/Tensorflow_V2.15.0/prepare_source.py \
     --feature-set full-default \
     --output-dir /path/to/tensorflow
   ```

   For other combinations, simply modify `--feature-set`. `/path/to/tensorflow` indicates the TensorFlow source root directory, and the output directory contains the complete source code of the official TensorFlow `v2.15.0`, the selected patches, and the automatically generated `tensorflow/feature_copts.bzl`.

3. Create a unified build directory in the TensorFlow source root directory.

   ```bash
   cd /path/to/tensorflow
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   `output/` is used to reuse the Bazel build cache, manually downloaded build dependencies are placed in `distdir/`, and pip packages are output to `output-release/`.

## Preparing Feature Dependencies

### KDNN

`kdnn-core`, `kdnn-annc`, and `full-default` all require KDNN header files and static libraries.

#### Downloading the KDNN Package and Checksum File

Go to the [KDNN releases page](https://gitcode.com/boostkit/boostsra/releases), locate release v1.4.0, and download the following files:

- KDNN package: `BoostKit-boostcore-kdnn_3.2.0.zip`
- SHA256 checksum file: `BoostKit-boostcore-kdnn_3.2.0.zip.sha256`

#### Verifying Package Integrity

##### Overview

To check whether the software package is incomplete due to network or device issues during transmission or storage, verify the integrity of the software package after obtaining it. Only a software package that passes verification can be deployed.<br>
Here, we determine whether the software package is complete by comparing the checksum recorded in the checksum file with the checksum calculated manually. If the two values match, the package is intact. Otherwise, download the package again.

##### Prerequisites

Prepare the following files before verifying package integrity:

- Package: `BoostKit-boostcore-kdnn_xxx.zip`
- Checksum file: a `.sha256` file with the same name as the package

##### Procedure

To verify package integrity, perform the following steps:

1. Calculate the SHA256 checksum of the package. On Linux, run the following command:

   ```bash
   sha256sum BoostKit-boostcore-kdnn_xxx.zip
   ```

   On Windows, run the following command:

   ```bash
   certutil -hashfile BoostKit-boostcore-kdnn_xxx.zip SHA256
   ```

   After the command is executed, the checksum is displayed.
2. Compare the checksum calculated in step 1 with the SHA256 value in the checksum file.<br>
   If they match, the package is intact. Otherwise, download the package again.

#### Installing the KDNN Software Package

1. Decompress the KDNN software package and install it using the `rpm -ivh` command.

   ```bash
   unzip BoostKit-boostcore-kdnn_3.2.0.zip
   rpm -ivh boostcore-kdnn-3.2.0-1.aarch64.rpm
   ```

   After installation, the header files are located at `/usr/local/kdnn/include`, and the thread pool and OpenMP libraries are located at `/usr/local/kdnn/lib/threadpool` and `/usr/local/kdnn/lib/omp`, respectively.
   TensorFlow integration uses the thread pool version.

2. Place the KDNN header files and the thread pool static library into the generated TensorFlow source code.

   ```bash
    export TF_PATH=/path/to/tensorflow
    mkdir -p $TF_PATH/third_party/KDNN/src
    cp -r /usr/local/kdnn/include $TF_PATH/third_party/KDNN/
    cp /usr/local/kdnn/lib/threadpool/libkdnn.a \
     $TF_PATH/third_party/KDNN/src/
   ```

3. Apply the KDNN header file adaptation patch.

   ```bash
   cd $TF_PATH/third_party/KDNN
   patch -p0 < tensorflow_kdnn_include_adapter.patch
   ```

### ANNC Static Graph Fusion

The ANNC static graph fusion code is already included in the `kdnn-annc` and `full-default` feature combinations, so there is no need to download it again or apply the TensorFlow patch. Before building, you only need to complete the KDNN dependency preparation.

### KEmbedding

1. The KEmbedding code is already included in the `full-default` feature combination, and no additional source code download is required.

   ```bash
   cd /path/to/tensorflow
   bazel --output_base="$PWD/output" build \
   --distdir="$PWD/distdir" \
   //third_party/kembedding:kembedding_embedding_table_lookup.so
   ```

   >![note](public_sys-resources/icon-note.gif) **NOTE**
   >
   >- If only the KEmbedding dynamic library is needed, it can be built separately from the generated TensorFlow source code.
   >- Skip this step if the complete TensorFlow artifacts need to be built.

2. The build artifact is `bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so`.

## Building TensorFlow Artifacts

>![note](public_sys-resources/icon-note.gif) **Build option:**
>
>`--define=enable_kdnn=true` enables KDNN kernel optimizations. Add this option when using the `kdnn-core`, `kdnn-annc`, or `full-default` feature combinations. The `common-only` combination does not include KDNN code and does not require this option. The build examples below assume that KDNN is enabled.

### TensorFlow pip Package

1. Configure the TensorFlow build.

   ```bash
   cd /path/to/tensorflow
   export TF_PYTHON_VERSION=3.11
   ./configure
   ```

2. Configure the following options in sequence. For options that do not display `N` or `n`, press `[Enter]` to confirm.

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

3. Build the TensorFlow pip package.

   ```bash
   bazel --output_base="$PWD/output" build \
   --distdir="$PWD/distdir" \
   -c opt \
   --define=enable_kdnn=true \
   //tensorflow/tools/pip_package:build_pip_package
   ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
   ```

4. Check the output directory after the build completes.

   ```bash
   ls ./output-release
   ```

   The pip package is built successfully when output similar to the following is displayed.

   ```output
   tensorflow-2.15.0-cp311-cp311-linux_aarch64.whl
   ```

### TensorFlow C++ Dynamic Library

Build the TensorFlow C++ dynamic library.

   ```bash
   cd /path/to/tensorflow
   bazel --output_base="$PWD/output" build \
   --distdir="$PWD/distdir" \
   -c opt \
   --define=enable_kdnn=true \
   //tensorflow:tensorflow_cc
   ```

## Feature Verification

>![Note](public_sys-resources/icon-note.gif) **NOTE**
>
>You must build and install the TensorFlow pip package before you can verify the features.

1. Run the installation command.

   ```bash
   pip install /path/to/tensorflow/output-release/tensorflow-2.15.0-cp311-cp311-linux_aarch64.whl
   ```

2. After the installation is complete, switch to any directory other than the TensorFlow source directory (for example, the user home directory) and run the following command to verify.

   ```bash
   cd ~
   python3 -c "import tensorflow as tf; print(tf.__version__)"
   ```

   >![Note](public_sys-resources/icon-note.gif) **NOTE**
   >
   >Do not run the verification command in the TensorFlow source root directory. Otherwise, Python preferentially loads the `tensorflow/` package in the source directory instead of the installed pip package, causing an import error.

   If the output is similar to the following, TensorFlow is installed successfully.

   ```output
   2.15.0
   ```

### KDNN

1. Enter the test directory and query the list of supported operators.

   ```bash
   cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark
   python main.py --list
   ```

2. Select an operator from the query results and run the test.

   ```bash
   numactl -C 0-15 python main.py --op {op_name} --performance_test True
   ```

   Taking the batch_matmul_ops operator as an example, when the output is similar to the following, the related operator has been integrated successfully.

   ```output
   Function testing: batch_matmul_2D_fp32_bcFalse_taTrue_tbTrue ...
   Function test passed
   Performance testing: MatMul_2D_79_1570_256_False_False ...
   Testing: MatMul_2D_79_1570_256_False_False ...
   ✅ Profile has been persisted to: bench_logs/plugins/profile/MatMul_2D_79_1570_256_False_False_kdnn_0
   Testing: MatMul_2D_79_1570_256_False_False ...
   ✅ Profile has been persisted to: bench_logs/plugins/profile/MatMul_2D_79_1570_256_False_False_kdnn_1
   Performance test passed, performance improved by 187.43%, elapsed time with KDNN enabled is 150.41us, and 432.33us with it disabled
   ```

### ANNC Static Graph Fusion

1. Enter the test directory and query the list of supported operators.

   ```bash
   cd /path/to/tensorflow/tensorflow/python/grappler/embedding_fused_test
   python main.py --list
   ```

2. Select an operator from the query results and run the test.

   ```bash
   python main.py --op {op_name} --performance_test True
   ```

   Taking the KPFusedEmbeddingActionIdGather_op operator as an example, when the output is similar to the following, the related operator has been integrated successfully.

   ```output
   Function testing: KPFusedEmbeddingActionIdGather_case_1_int32_int32 ...
   Function test passed
   Performance testing: KPFusedEmbeddingActionIdGather_case_1_int32_int32 ...
   Testing: KPFusedEmbeddingActionIdGather_case_1_int32_int32 ...
   Testing: KPFusedEmbeddingActionIdGather_case_1_int32_int32 ...
   Performance test passed, performance improved by 291.88%, elapsed time with embedding operator fusion enabled is 52.41us, and 205.38us with it disabled
   ```

   >![Note](public_sys-resources/icon-note.gif) **NOTE**
   >
   >ANNC static graph fusion takes effect only on subgraphs that meet specific structural and input constraints.

### KEmbedding

1. Enter the test directory.

   ```bash
   cd /path/to/tensorflow
   ```

2. Run the unit tests.

   ```bash
   bazel test //third_party/kembedding:embedding_table_lookup_op_test \
   --test_output=all   --nocache_test_results
   ```

   When the output is similar to the following, the unit tests have passed.

   ```output
   //third_party/kembedding:embedding_table_lookup_op_test                  PASSED in 0.2s
   ```

3. Run the benchmark tests.

   ```bash
   bazel run //third_party/kembedding:embedding_table_lookup_benchmark -- --benchmark_filter=.
   ```

   When the output is similar to the following, the benchmark tests have passed.

   ```output
   Benchmark                                                 Time             CPU   Iterations UserCounters...
   -----------------------------------------------------------------------------------------------------------
   BM_EmbeddingTableLookup_SmallSingleThread_false       44547 ns        44428 ns        15791 items_per_second=92.1944M/s 4KKeys_Dim16_4Values_1Thread_Hit100
   BM_EmbeddingTableLookup_SmallSingleThread_true        98719 ns        98461 ns         7046 items_per_second=41.6004M/s 4KKeys_Dim16_4Values_1Thread_Hit100
   BM_EmbeddingTableLookup_MediumParallel_false         339120 ns       335686 ns         2164 items_per_second=48.8075M/s 16KKeys_Dim32_8Values_8Threads_Hit100
   BM_EmbeddingTableLookup_MediumParallel_true          776258 ns       773776 ns          879 items_per_second=21.1741M/s 16KKeys_Dim32_8Values_8Threads_Hit100
   BM_EmbeddingTableLookup_MediumParallelMiss_false     321051 ns       318884 ns         2226 items_per_second=51.3792M/s 16KKeys_Dim32_8Values_8Threads_Hit75
   BM_EmbeddingTableLookup_MediumParallelMiss_true      794667 ns       793124 ns          882 items_per_second=20.6576M/s 16KKeys_Dim32_8Values_8Threads_Hit75
   BM_EmbeddingTableLookup_LargeParallel_false         1873011 ns      1865223 ns          340 items_per_second=35.1357M/s 64KKeys_Dim64_16Values_16Threads_Hit100
   BM_EmbeddingTableLookup_LargeParallel_true          6129549 ns      6100689 ns           84 items_per_second=10.7424M/s 64KKeys_Dim64_16Values_16Threads_Hit100
   ```

## (Optional) Building the Inference Service

The current release supports building an inference service based on the open-source TensorFlow Serving.

### Building TensorFlow Serving

Prepare the TensorFlow Serving source code, Bazel, and build dependencies by following the "[Compiling TensorFlow Serving](https://www.hikunpeng.com/document/detail/en/SRA/perfEval/benchmarksra/kunpengmodelzoo_06_0011.html)" section in the *Inference Performance Benchmark Testing for Search and Recommendation Ranking Models*.

1. During compilation, point `--tensorflow_dir` to the complete TensorFlow source code generated by this guide.

   ```bash
   cd /path/to/serving
   sh compile_serving.sh \
     --tensorflow_dir /path/to/tensorflow \
     --features gcc12
   ```

2. Check the build artifacts.

   ```output
   /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
   ```

TensorFlow Serving is integrated through the local TensorFlow source code. When switching feature combinations, there is no need to reorganize the Serving source code; simply regenerate the corresponding TensorFlow source code, prepare its dependencies, and rebuild Serving.

## Legacy Features

The Legacy patch contains features such as historical Runtime scheduling, legacy fused Embedding, ANNC graph compilation, and legacy XLA execution. It is only allowed to be independently applied to the official TensorFlow `v2.15.0` baseline.

>![note](public_sys-resources/icon-note.gif) **NOTE**
>
>The Legacy patch does not depend on `common`, is not part of the currently maintained feature combinations, and is not guaranteed to be compatible with KDNN, ANNC static graph fusion, or KEmbedding patches.

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

When TensorFlow Serving needs to be built, still follow the process in [Building TensorFlow Serving](#building-tensorflow-serving) and change `--tensorflow_dir` to the Legacy source code directory.

## FAQ

When compiling TensorFlow and TensorFlow Serving, refer to the following troubleshooting cases:

- [Failed to Verify the Certificate When Compiling TensorFlow Source Code](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0012.html)
- [Failed to Download the TF-Serving Source Code Dependency](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0014.html)
- [Failed to Obtain the Dependency of org_boost](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0015.html)
- [No Golang Website Certificate](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0016.html)
- [Syntax Error Reported During upb.c Compilation](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0017.html)

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 03 | 2026-09-30 | This is the third official release. Restructured the installation process by consolidating duplicate environment and compilation steps across features. |
| 02 | 2026-06-30 | This is the second official release.<ul><li>Added constant folding optimization to the TensorFlow ANNC graph compilation optimization feature.</li><li>Added adaptation environment and installation guide for the TensorFlow ANNC static graph fusion feature.</li></ul> |
| 01 | 2026-03-30 | This is the first official release. <ul><li>Added installation steps for integrating TensorFlow with KDNN.</li><li>Added adaptation environment and installation guide for the TensorFlow KDNN thread passthrough feature.</li></ul> |
