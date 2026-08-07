# Installation Guide

<!-- md-trans-meta sourceCommit=348350b006f776e1fe13fb0fb53fe94efa080a45 translatedAt=2026-08-05T03:52:05.826Z pushedAt=2026-08-05T07:56:06.344Z -->

This document describes the installation process in the following order: preparing the TensorFlow source code, building the reasoning service, and verifying the features.

## Preparing the Build Environment

The build environment for the currently maintained version is as follows.

| Item | Version or Requirement |
| --- | --- |
| CPU | <ul><li>Kunpeng 920 processor</li><li>Kunpeng 950 processor</li></ul> |
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

Both TensorFlow and TensorFlow Serving are built using Bazel 6.5.0. For the Bazel installation method, see the "[Installing Bazel](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html)" section in *TensorFlow Porting Guide*.

## Preparing TensorFlow Source Code

### Selecting a Feature Combination

| Feature Combination | Contents | Corresponding Patches (in App Order) | Applicable Scenario |
| --- | --- | --- | --- |
| `common-only` | Public build integration (common) and compatibility changes | `patches/feature/0001-tensorflow_2.15.0-common.patch` | Check public changes without enabling acceleration features. |
| `kdnn-core` | Public build integration (common), KDNN operator optimization | `patches/feature/0001-tensorflow_2.15.0-common.patch`<br>`patches/feature/0002-tensorflow_2.15.0-kdnn.patch` | Use KDNN operator optimization. |
| `kdnn-annc` | Public build integration (common), KDNN operator optimization, ANNC static graph fusion | `patches/feature/0001-tensorflow_2.15.0-common.patch`<br>`patches/feature/0002-tensorflow_2.15.0-kdnn.patch`<br>`patches/feature/0003-tensorflow_2.15.0-annc.patch` | Use KDNN and ANNC static graph fusion. |
| `full-default` | Public build integration (common), KDNN operator optimization, ANNC static graph fusion, KEmbedding custom operator | `patches/feature/0001-tensorflow_2.15.0-common.patch`<br>`patches/feature/0002-tensorflow_2.15.0-kdnn.patch`<br>`patches/feature/0003-tensorflow_2.15.0-annc.patch`<br>`patches/feature/0004-tensorflow_2.15.0-kembedding.patch` | Use all currently maintained features. |

### Generating Complete Source Code

1. Download the patch repository and obtain the official TensorFlow baseline.

   ```bash
   git clone -b v1.2.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

2. Create the complete TensorFlow source code as needed. The following example uses the full default feature set.

   ```bash
   python3 patches/prepare_source.py \
     --feature-set full-default \
     --output-dir /path/to/tensorflow
   ```

   For other combinations, simply modify `--feature-set`. The output directory contains the complete source code of the official TensorFlow `v2.15.0`, the selected patches, and the automatically generated `tensorflow/feature_copts.bzl`.

3. Create a unified build directory at the root of the TensorFlow source tree.

   ```bash
   cd /path/to/tensorflow
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   `output/` is used to reuse the Bazel build cache. Manually downloaded build dependencies are placed in `distdir/`, and the pip package is output to `output-release/`.

## Preparing Feature Dependencies

### KDNN

`kdnn-core`, `kdnn-annc`, and `full-default` all require KDNN header files and static libraries.

1. Obtain and install the [KDNN software package](https://gitcode.com/boostkit/boostsra/releases/download/v1.1.0/BoostKit-boostcore-kdnn_3.0.0.zip).

   ```bash
   rpm -ivh boostcore-kdnn-3.0.0-1.aarch64.rpm
   ```

   After installation, the header files are located at `/usr/local/kdnn/include`, and the thread pool and OpenMP libraries are located at
   `/usr/local/kdnn/lib/threadpool` and `/usr/local/kdnn/lib/omp`, respectively.
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

The ANNC static graph fusion code is already included in the `kdnn-annc` and `full-default` feature combinations, so there is no need to download or apply TensorFlow patches again. Only the KDNN dependency preparation needs to be completed before building.

### KEmbedding

The KEmbedding code is already included in the `full-default` feature combination, and no additional source code download is required. If only the KEmbedding dynamic library is needed, it can be built separately from the generated TensorFlow source code:

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  //third_party/kembedding:kembedding_embedding_table_lookup.so
```

The build artifact is `bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so`.

## Building the Inference Service

The current release supports building an inference service based on the open-source TensorFlow Serving.

### Building TensorFlow Serving

1. Prepare the TensorFlow Serving source code, Bazel, and build dependencies by following the "[Compiling TensorFlow Serving](https://www.hikunpeng.com/document/detail/en/SRA/perfEval/benchmarksra/kunpengmodelzoo_06_0011.html)" section in *Search and Recommendation Ranking Model Inference Benchmark*.

2. During compilation, point `--tensorflow_dir` to the complete TensorFlow source code generated by this guide.

   ```bash
   cd /path/to/serving
   sh compile_serving.sh \
     --tensorflow_dir /path/to/tensorflow \
     --features gcc12
   ```

3. Check the build artifacts.

   ```text
   /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
   ```

TensorFlow Serving is integrated through the local TensorFlow source code. When switching feature combinations, there is no need to reorganize the Serving source code; simply regenerate the corresponding TensorFlow source code, prepare its dependencies, and rebuild Serving.

## Building TensorFlow Artifacts

When the reasoning service is not used, TensorFlow targets can also be built on demand directly from the generated source code.

### TensorFlow pip package

Build the TensorFlow pip package.

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

### TensorFlow C++ Dynamic Library

Build the TensorFlow C++ dynamic library.

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  -c opt \
  //tensorflow:tensorflow_cc
```

## Feature Verification

### KDNN

1. Enter the test directory and query the supported modules.

   ```bash
   cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark
   python main.py --list
   ```

2. Select an operator from the query results and run the test.

   ```bash
   python main.py --op {op_name} --performance_test False
   ```

   A successful execution indicates that the relevant operator has been integrated successfully.

### ANNC Static Graph Fusion

After entering the test directory, execute the test command.

```bash
cd /path/to/tensorflow/tensorflow/python/grappler/embedding_fused_test
python main.py --list
python main.py --op {op_name} --performance_test False
```

ANNC static graph fusion takes effect only on subgraphs that meet specific structural and input constraints.

### KEmbedding

Navigate to the test directory and execute the test command.

```bash
cd /path/to/tensorflow
bazel test //third_party/kembedding:embedding_table_lookup_op_test \
  --test_output=errors
bazel run //third_party/kembedding:embedding_table_lookup_benchmark
```

## Legacy Features

The Legacy patch contains features such as historical Runtime scheduling, legacy fused Embedding, ANNC graph compilation, and legacy XLA execution. It is only allowed to be independently applied to the official TensorFlow `v2.15.0` baseline. The Legacy patch does not depend on common, is not part of the currently maintained feature combination, and is not guaranteed to be compatible with KDNN, ANNC static graph fusion, or KEmbedding patches.

```bash
git clone -b v2.15.0 https://github.com/tensorflow/tensorflow.git tensorflow-legacy
cd tensorflow-legacy
git apply --check \
  /path/to/sra-tensorflow/patches/frozen_feature/tensorflow_2.15.0-legacy.patch
git apply \
  /path/to/sra-tensorflow/patches/frozen_feature/tensorflow_2.15.0-legacy.patch

mkdir -p output distdir output-release
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  --config=fused_embedding \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

When TensorFlow Serving needs to be built, still follow the process in "[Building TensorFlow Serving](#building-tensorflow-serving)" and change `--tensorflow_dir` to the Legacy source code directory.

## FAQs

When compiling TensorFlow and TensorFlow Serving, refer to the following troubleshooting documents:

- [Failed to Verify the Certificate When Compiling TensorFlow Source Code](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0012.html)

- [Failed to Download the TF-Serving Source Code Dependency](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0014.html)

- [Failed to Obtain the Dependency of org_boost](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0015.html)

- [No Golang Website Certificate](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0016.html)

- [Syntax Error Reported During upb.c Compilation](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0017.html)

## Change History

| Release Date | Description |
| ---- | ---- |
| 2026-09-30 | This is the third official release. Restructured the installation process by consolidating duplicate environment and compilation steps across features. |
| 2026-06-30 | This is the second official release.<ul><li>Added constant folding optimization to the TensorFlow ANNC graph compilation optimization feature.</li><li>Added adaptation environment and installation guide for the TensorFlow ANNC static graph fusion feature.</li></ul> |
| 2026-03-30 | This is the first official release. <ul><li>Added installation steps for integrating TensorFlow with KDNN.</li><li>Added adaptation environment and installation guide for the TensorFlow KDNN thread passthrough feature.</li></ul> |
