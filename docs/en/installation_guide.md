# Installation Guide

This guide follows a single workflow: prepare the TensorFlow source, build an inference service, and verify the selected features. Shared environment and source preparation steps are performed once; feature sections describe only additional dependencies and validation.

## Preparing the Build Environment

The reference environment for the maintained release is as follows.

| Item | Version or Requirement |
| --- | --- |
| CPU | Kunpeng 920 or Kunpeng 950 processor |
| OS | openEuler 24.03 LTS SP3 |
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

TensorFlow and TensorFlow Serving use Bazel 6.5.0. For installation instructions, see [Installing Bazel](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_06_0008.html) in the _TensorFlow Porting Guide_.

## Creating the TensorFlow Source

### Selecting a Feature Set

| Feature Set | Contents | Patches (Application Order) | When to Use |
| --- | --- | --- | --- |
| `common-only` | Shared build and compatibility changes | `patches/dist/0001-tensorflow_2.15.0-common.patch` | Inspect shared changes without acceleration features |
| `kdnn-core` | common + KDNN | `patches/dist/0001-tensorflow_2.15.0-common.patch`<br>`patches/dist/0002-tensorflow_2.15.0-kdnn.patch` | Use KDNN kernel optimizations |
| `kdnn-annc` | common + KDNN + ANNC | `patches/dist/0001-tensorflow_2.15.0-common.patch`<br>`patches/dist/0002-tensorflow_2.15.0-kdnn.patch`<br>`patches/dist/0003-tensorflow_2.15.0-annc.patch` | Use KDNN and ANNC static graph fusion |
| `full-default` | common + KDNN + ANNC + KEmbedding | `patches/dist/0001-tensorflow_2.15.0-common.patch`<br>`patches/dist/0002-tensorflow_2.15.0-kdnn.patch`<br>`patches/dist/0003-tensorflow_2.15.0-annc.patch`<br>`patches/dist/0004-tensorflow_2.15.0-kembedding.patch` | Use all currently maintained features |

### Generating the Complete Source

1. Clone the patch repository and fetch the official TensorFlow baseline.

   ```bash
   git clone -b v1.2.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow
   cd sra-tensorflow
   git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
   git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0
   ```

2. Create the complete TensorFlow source for the required feature set. The following example uses all maintained features.

   ```bash
   python3 patches/prepare_source.py \
     --feature-set full-default \
     --output-dir /path/to/tensorflow
   ```

   To use another set, change only `--feature-set`. The output contains the complete official TensorFlow `v2.15.0` source, the selected patches, and the generated `tensorflow/feature_copts.bzl`.

3. Create the shared build directories in the TensorFlow source root.

   ```bash
   cd /path/to/tensorflow
   mkdir -p output distdir output-release
   export TF_PYTHON_VERSION=3.11
   ```

   The `output/` directory preserves the reusable Bazel build cache, manually
   downloaded build dependencies go in `distdir/`, and pip packages are written
   to `output-release/`.

## Preparing Feature Dependencies

### KDNN

The `kdnn-core`, `kdnn-annc`, and `full-default` sets require the KDNN headers and static library.

1. Obtain and install the [KDNN package](https://gitcode.com/boostkit/boostsra/releases/download/v1.1.0/BoostKit-boostcore-kdnn_3.0.0.zip).

   ```bash
   rpm -ivh boostcore-kdnn-3.0.0-1.aarch64.rpm
   ```

   After installation, the headers are under `/usr/local/kdnn/include`. The
   thread-pool and OpenMP libraries are under `/usr/local/kdnn/lib/threadpool`
   and `/usr/local/kdnn/lib/omp`, respectively. TensorFlow integration uses
   the thread-pool variant.

2. Copy the KDNN headers and thread-pool library into the generated TensorFlow source.

   ```bash
   export TF_PATH=/path/to/tensorflow
   mkdir -p $TF_PATH/third_party/KDNN/src
   cp -r /usr/local/kdnn/include $TF_PATH/third_party/KDNN/
   cp /usr/local/kdnn/lib/threadpool/libkdnn.a \
     $TF_PATH/third_party/KDNN/src/
   ```

3. Apply the KDNN header adapter patch.

   ```bash
   cd $TF_PATH/third_party/KDNN
   patch -p0 < tensorflow_kdnn_include_adapter.patch
   ```

### ANNC Static Graph Fusion

The `kdnn-annc` and `full-default` feature sets already contain the ANNC static graph fusion source. No additional TensorFlow patch is required. Prepare only the KDNN dependency before building.

### KEmbedding

The `full-default` feature set already contains the KEmbedding source. To build only its shared library:

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  //third_party/kembedding:kembedding_embedding_table_lookup.so
```

The output is `bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so`.

## Building an Inference Service

The current release supports an inference service based on open-source TensorFlow Serving.

### Building TensorFlow Serving

1. Follow [Compiling TensorFlow Serving](https://www.hikunpeng.com/document/detail/en/SRA/perfEval/benchmarksra/kunpengmodelzoo_06_0011.html) in _Inference Performance Benchmark Testing for Search and Recommendation Ranking Models_ to prepare the TensorFlow Serving source, Bazel, and build dependencies.

2. Point `--tensorflow_dir` to the complete TensorFlow source generated by this guide.

   ```bash
   cd /path/to/serving
   sh compile_serving.sh \
     --tensorflow_dir /path/to/tensorflow \
     --features gcc12
   ```

3. Check the generated server binary:

   ```text
   /path/to/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
   ```

TensorFlow Serving integrates the selected features through the local TensorFlow source. To change the feature set, regenerate TensorFlow, prepare its dependencies, and rebuild Serving. The Serving source layout does not need to be changed.

## Building TensorFlow Artifacts

When an inference service is not required, build TensorFlow targets directly from the generated source as needed.

### TensorFlow pip Package

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

### TensorFlow C++ Shared Library

```bash
cd /path/to/tensorflow
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  -c opt \
  //tensorflow:tensorflow_cc
```

## Verifying Features

### KDNN

```bash
cd /path/to/tensorflow/tensorflow/python/kernel_tests/benchmark
python main.py --list
python main.py --op {op_name} --performance_test False
```

Select an `op_name` from the listed modules. A successful run verifies the
corresponding operator integration.

### ANNC Static Graph Fusion

```bash
cd /path/to/tensorflow/tensorflow/python/grappler/embedding_fused_test
python main.py --list
python main.py --op {op_name} --performance_test False
```

Static graph fusion takes effect only when a graph contains a supported subgraph with valid input constraints.

### KEmbedding

```bash
cd /path/to/tensorflow
bazel test //third_party/kembedding:embedding_table_lookup_op_test \
  --test_output=errors
bazel run //third_party/kembedding:embedding_table_lookup_benchmark
```

## Legacy Features

The Legacy patch contains historical runtime scheduling, the old fused embedding implementation, ANNC graph compilation, and old XLA execution features. It applies independently and only to the official TensorFlow `v2.15.0` baseline.

```bash
git clone -b v2.15.0 https://github.com/tensorflow/tensorflow.git tensorflow-legacy
cd tensorflow-legacy
git apply --check \
  /path/to/sra-tensorflow/patches/frozen/0005-tensorflow_2.15.0-legacy.patch
git apply \
  /path/to/sra-tensorflow/patches/frozen/0005-tensorflow_2.15.0-legacy.patch

mkdir -p output distdir output-release
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  --config=fused_embedding \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

To build TensorFlow Serving, follow [Building TensorFlow Serving](#building-tensorflow-serving) and point `--tensorflow_dir` to the Legacy source.

>![icon note](public_sys-resources/icon-note.gif) **NOTE:**
>The Legacy patch does not depend on `common`, is not part of the maintained feature sets, and is not guaranteed to work with KDNN, ANNC static graph fusion, or KEmbedding.

## FAQs

For TensorFlow and TensorFlow Serving build failures, see:

- [TensorFlow source certificate verification failure](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_06_0012.html)
- [TensorFlow Serving dependency download failure](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0014.html)
- [Failed to obtain the org_boost dependency](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0015.html)
- [Missing Golang website certificate](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0016.html)
- [upb.c compilation syntax error](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0017.html)

## Description

| Release Date | Change History |
| --- | --- |
| 2026-09-30 | This is the third official release. Restructured installation by TensorFlow source, feature dependencies, and inference service build layers, consolidating duplicated environment and compilation steps. |
| 2026-06-30 | This is the second official release. <ul><li>Added the description for constant folding optimization to the TensorFlow ANNC for graph compilation documentation. </li><li>Added the environment support and installation guide for TensorFlow ANNC static graph fusion. </li></ul> |
| 2026-03-30 | This is the first official release. <ul><li>Added installation steps for TensorFlow with KDNN integration. </li><li>Added the environment support and installation guide for KDNN thread passthrough.</li></ul> |
