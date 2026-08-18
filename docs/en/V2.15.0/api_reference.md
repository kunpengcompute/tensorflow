# API Reference

<!-- md-trans-meta sourceCommit=43da085d0d418cfad09a9ba22bf97c508f3b8e0f translatedAt=2026-08-04T03:19:09.402Z pushedAt=2026-08-05T07:33:43.161Z -->

## Using the TensorFlow KDNN Thread Passthrough Feature

The TensorFlow KDNN thread passthrough feature is controlled by a process environment variable, as described below.

**Variable Type**

Process environment variable

**Variable Name**

`TF_ENABLE_KDNN_OPTS`

**Variable Function**

Controls the enablement of KDNN optimization features.

**Variable Value**

* 0: KDNN is disabled.

* 1: KDNN is enabled.

**Sample**

Set the process environment variable `TF_ENABLE_KDNN_OPTS` before the first invocation of a KDNN operator. For example, in Python, the process environment variable can be set using the command `os.environ['TF_ENABLE_KDNN_OPTS'] = str(1)`.

## SparseMatmul Multi-threading Optimization

The SparseMatmul operator is a KDNN operator for calculating the product of a sparse matrix and a dense matrix. It supports single-precision FP32 inputs. This operator is the core component of the Neural Network (NN) layers in recommendation models.

**Interface Description**

The operator is designed based on the compressed sparse row (CSR) storage structure. It skips zero blocks during loading and computing to maximize the efficiency of computational and memory bandwidth utilization. The core computing kernel has been optimized for the Kunpeng platform by leveraging SIMD (supporting the NEON instruction set), implementing multi-threading optimization.

**Interface Type**

Internal computing interface.

**Input Parameter**

| Parameter | Type | Description |
| ----------- | ------------- | ---------- |
| tp | KDNN::Threading::ThreadpoolIface * | KDNN thread pool interface, used for multi-threaded parallel execution. |
| alpha | const FLOAT | Scaling factor. |
| mat | const JOIN(spmat_csr_, _t) * | Sparse matrix (CSR format). |
| x | const FLOAT * | Dense matrix. |
| columns | const KDNN_INT | Number of matrix columns. |
| ldx | const KDNN_INT | Stride of matrix `x`. |
| beta | const FLOAT | Accumulation scaling factor. |
| y | FLOAT * | Output matrix. |
| ldy | const KDNN_INT | Stride of matrix `y`. |

**Output Parameter**

None. The result is returned through the `y` parameter.

**Interface Change**

The `ThreadpoolIface *tp` parameter is added to the API function signature to transfer the thread pool instance.

* Before modification

  ```c
  kdnn_sparse_status_t kdnn_sparse_scsrmm(
      const kdnn_sparse_operation_t opt, ...);
  ```

* After modification

  ```c
  kdnn_sparse_status_t kdnn_sparse_scsrmm(
      KDNN::Threading::ThreadpoolIface *tp,
      const kdnn_sparse_operation_t opt, ...);
  ```

**Interface Source File**

The interface source code files are `third_party/kdnn/kdnn_adapter.h` and `tensorflow/core/kernels/sparse_tensor_dense_matmul_op.cc`.

## EmbeddingTableLookup

The EmbeddingTableLookup operator, a custom operator in the KEmbedding operator library, is used to efficiently perform sparse embedding lookup.

**Interface Description**

This operator retrieves sparse embeddings from the resource table `EmbeddingIndexToValueTable` based on keys and outputs a standard `SparseTensor` triplet.

* `indices`: contains the coordinates of the hit non-zero elements.

* `values`: contains the corresponding embedding values.

* `dense_shape`: contains the shape of the complete embedding matrix.

**Interface Type**

TensorFlow OpKernel class.

**Input Parameter**

| Parameter | Type | Description |
| ---------------- | ------------- | ------ |
| keys | int64 tensor | List of embedding keys to look up, whose shape is `[key_cnt]`. |
| table_handle | resource | Resource table handle, created through `EmbeddingIndexToValueTable` and loaded with embedding data. |

**Output Parameter**

| Parameter | Type | Description |
| --------------- | ------------- | ------------- |
| indices | int64 tensor | Indices of the SparseTensor, whose shape is `[N, 2]`, where N is the total number of non-zero elements hit, and the second dimension is `[row, col]`. |
| values | float tensor | Values of the SparseTensor, whose shape is `[N]`, corresponding one-to-one with indices. |
| dense_shape | int64 tensor | Dense shape, whose shape is `[2]`, with the value `[key_cnt, emb_dim]`. |

**Core Attribute**

| Attribute | Description |
| ----------- | --------- |
| emb_dim | Embedding dimension, used to construct the output `dense_shape`. |

**Interface Source File**

`third_party/kembedding/src/kernels/embedding_table_lookup_op.cc`

**Other Related Operators**

* <code>EmbeddingIndexToValueTable</code>: creates a resource table handle.

* <code>InitializeEmbeddingIndexToValueTableFromTextFile</code>: initializes the resource table from a binary file.

The retention of `TextFile` in this interface name is due to historical naming. The actual input file read is not a text file, but a kembedding binary table file described below.

**Sample**

```python
import tensorflow as tf

# Load the kembedding custom operator dynamic library.
kembedding_module = tf.load_op_library(
    'path/to/bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so'
)

# Create and initialize the resource table.
with tf.compat.v1.Session() as sess:
    # Step 1: Create the resource table handle.
    table_handle = kembedding_module.embedding_index_to_value_table()

    # Step 2: Load embedding table data from the kembedding binary table file.
    # Note: The interface name contains text_file due to historical naming; filename should be the path to the binary table file.
    sess.run(
        kembedding_module.initialize_embedding_index_to_value_table_from_text_file(
            table_handle=table_handle,
            filename='path/to/embedding_table.bin'
        )
    )

    # Step 3: Perform batch lookup.
    keys = tf.constant([101, 202, 999], dtype=tf.int64)
    indices, values, dense_shape = kembedding_module.embedding_table_lookup(
        table_handle=table_handle,
        keys=keys,
        emb_dim=4
    )

    # Obtain the result.
    result_indices, result_values, result_shape = sess.run(
        [indices, values, dense_shape]
    )

    # Result example:
    # indices = [[0, 0], [0, 2], [1, 1]]
    # values = [1.0, 3.0, 2.5]
    # dense_shape = [3, 4]
```

## Using the TensorFlow ANNC Static Graph Fusion Feature

The TensorFlow ANNC static graph fusion feature is enabled or disabled by environment variables. For details, see [Table 1 Environment variables for enabling or disabling ANNC static graph fusion](#table473618378218)

The default value of each feature switch is `0`, indicating that the ANNC static graph fusion function is disabled. To use this function, you need to manually set the environment variable before graph optimization. For example, you can set the environment variable in Python as follows:

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
```

**Table 1** Environment variables for enabling or disabling ANNC static graph fusion<a id="table473618378218"></a>

| Environment Variable | Type | Value | Function |
| -------- | ------------ | --------------- | ------------------ |
| ANNC_FUSED_EMB_ACTIONID_GATHER | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedEmbeddingActionIdGather operator. |
| ANNC_FUSED_GATHER | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedGather operator. |
| ANNC_FUSED_EMD_PADDING | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedEmbeddingPadding operator. |
| ANNC_FUSED_EMD_PADDING_FAST | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedEmbeddingPaddingFast operator. |
| ANNC_FUSED_SPS_STITCH | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedSparseDynamicStitch operator. |
| ANNC_FUSED_SPS_RESHAPE | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedSparseReshape operator. |
| ANNC_FUSED_SPS_REDUCE | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedSparseSegmentReduce operator. |
| ANNC_FUSED_SPS_REDUCE_NONZERO | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedSparseSegmentReduceNonzero operator. |
| ANNC_FUSED_SPS_SELECT | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for the KPFusedSparseSelect operator. |
| ANNC_FUSED_ALL | Process environment variable | `1`: Enable `0`: Disable | Enables ANNC static graph fusion for all ANNC fusion operators. |

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> Operator fusion will not be performed for any of the above operators if and only if `ANNC_FUSED_ALL` is set to `0` and the environment variable corresponding to the specific operator is also set to `0`.

## Legacy: Using the TensorFlow ANNC Graph Compilation Optimization Feature

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> The following interfaces are provided only by an independently frozen legacy patch and are not part of the currently maintained default profile.

TensorFlow ANNC for graph compilation optimization provides the following features: TensorFlow graph fusion, XLA graph fusion, operator optimization, and constant folding optimization. This document describes APIs of each feature and the steps required to enable them.

### TensorFlow Graph Fusion

The TensorFlow graph fusion interface commands and usage examples are as follows.

**Terminal Command Line Interface**

`annc-opt`

**Interface Function**

Graph fusion startup command.

**Parameters**

* `-I /path/to/save_model.pb`: Model pending graph fusion

* `-O /path/to/new_save_model.pb`: Model after graph fusion

* `pass`: Graph fusion strategy (currently supports `lookup_embedding_hash`).

**Sample**

```bash
annc-opt -I /base_model/wide_and_deep/1/ -O /optimized_model/wide_and_deep/1/ lookup_embedding_hash
cp -r /base_model/wide_and_deep/1/variables /optimized_model/wide_and_deep/1/
```

### XLA Graph Fusion

The XLA graph fusion interface commands and usage examples are as follows.

**Interface Function**

Compiles ANNC and enables XLA graph fusion optimization.

**Environment Variable**

`ANNC_FLAGS`

**Value Range**

The feature is enabled when the environment variable value is `"--graph-opt"`.

Sample

```bash
export ANNC_FLAGS="--graph-opt"
```

### Operator Optimization

The optimized operator interfaces include redundant operator, matrix operator, and Softmax operator interfaces. Their usage is shown in [**Table 1** Operator optimization interface](#operator-optimization-interface).

**Table 1** Operator optimization interface<a id="operator-optimization-interface"></a>

| Interface | Interface | Environment Variable | Value Range | Usage Example |
| ---- | ---- | ---- | ---- | ---- |
| Redundant operator optimization interface | Enables redundant operator optimization. | ENABLE_BISHENG_GRAPH_OPT | The feature is enabled when the environment variable is non-empty. | `export ENABLE_BISHENG_GRAPH_OPT=""` |
| Matrix operator optimization interface | Enables matrix operator optimization. | ANNC_FLAGS | The feature is enabled when the environment variable value is `"--gemm-opt"`. | `export ANNC_FLAGS="--gemm-opt"` |
| Softmax operator optimization interface | Enables Softmax operator optimization. | XLA_FLAGS | The feature is enabled when the environment variable value is `"--xla_cpu_enable_xnnpack=true"`. | `export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"` |

### Constant Folding Optimization

When both constant folding and graph optimization are enabled, constant folding must be enabled first.

The constant folding model conversion interface is used as follows.

**Terminal Command Line Interface**

`annc-opt`

**Interface Function**

Constant folding startup command.

**Parameters**

* `-I /path/to/save_model.pb`: Model pending constant folding

* `-O /path/to/new_save_model.pb`: Model after constant folding

* `pass`: `layout_matmul`

Sample

```bash
annc-opt -I /base_model/wide_and_deep/1/ -O /folding/wide_and_deep/1/ layout_matmul
```

The constant folding optimization interface is used as follows.

**Environment Variable**

`ANNC_FLAGS`

**Interface Function**

Enables constant folding optimization.

**Value Range**

The feature is enabled when the environment variable value is `"--layout-matmul"`.

**Sample**

```bash
export ANNC_FLAGS="--layout-matmul"
```

## Legacy: Using the TensorFlow Serving Thread Scheduling Feature

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> The following interfaces are provided only by an independently frozen legacy patch and are not part of the currently maintained default profile.

Kunpeng's TensorFlow Serving thread scheduling feature provides two configuration options: batch operator scheduling and thread affinity isolation. You can configure the options based on your specific requirements.

To use TensorFlow Serving to start an inference stress test, see section [Starting the Service and Performing a Pressure Test](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0012.html) in the *TensorFlow Serving Porting Guide*.

### Batch Operator Scheduling

The batch operator scheduling interface is used as follows.

**TF Serving Command Line Interface**

`--batch_op_scheduling`

**Interface Function**

Enables operator scheduling optimization and XLA thread pool management optimization features.

**Parameter Type**

bool

**Value Range**

* true: The feature is enabled.

* false: The feature is disabled. The default value is false.

**Recommended Scenario**

When the single-core inference latency meets service requirements, this option can be configured to improve inference concurrency and throughput.

**Recommended Configuration**

* `--tensorflow_intra_op_parallelism=1`: Sets the intra-op parallelism to 1.

* `--tensorflow_inter_op_parallelism=80`: Sets the inter-op parallelism to the number of CPU cores.

* `--batch_op_scheduling=true`: Enables the batch operator scheduling feature.

**Sample**

```bash
/path/to/tensorflow_model_server  --port=8850 \
  --rest_api_port=8851 \
  --model_base_path=/path/to/saved_model/ \
  --model_name=model \
  --tensorflow_intra_op_parallelism=1 \
  --tensorflow_inter_op_parallelism=80 \
  --batch_op_scheduling=true
```

### Thread Affinity Isolation

The thread affinity isolation interface is used as follows.

**TF Serving Command Line Interface**

`--task_affinity_isolation`

**Interface Function**

Enables the thread affinity isolation feature, which supports two isolation modes:

* Sequential core binding allocates TensorFlow computing threads to the first K cores and TF Serving communication threads to remaining cores.

* Interleaved core binding (applicable when hyper-threading is enabled) assigns TensorFlow threads to physical cores and TF Serving communication threads to virtual cores.

**Parameter Type**

std::string

**Parameter Format**

`mode;m-n;k`, with the default value `0`.

**Value Range**

**Table 1** Thread affinity isolation parameter values<a id="thread-affinity-isolation-parameter-values"></a>

<a name="table12688064377"></a>

<table><thead align="left"><tr id="row468814643718"><th class="cellrowborder" valign="top" width="7.43925607439256%" id="mcps1.2.5.1.1"><p id="p176881266377"><a name="p176881266377"></a><a name="p176881266377"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="15.448455154484552%" id="mcps1.2.5.1.2"><p id="p14656122373711"><a name="p14656122373711"></a><a name="p14656122373711"></a>Value Range</p>
</th>
<th class="cellrowborder" valign="top" width="27.90720927907209%" id="mcps1.2.5.1.3"><p id="p1768920617371"><a name="p1768920617371"></a><a name="p1768920617371"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="49.2050794920508%" id="mcps1.2.5.1.4"><p id="p3689265373"><a name="p3689265373"></a><a name="p3689265373"></a>Constraint</p>
</th>
</tr>
</thead>
<tbody><tr id="row168915612378"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p196891768371"><a name="p196891768371"></a><a name="p196891768371"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p265616232373"><a name="p265616232373"></a><a name="p265616232373"></a><code>0</code>, <code>1</code>, or <code>2</code></p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><a name="ul1088015211937"></a><a name="ul1088015211937"></a><ul id="ul1088015211937"><li><code>0</code>: (OFF) Thread affinity is disabled. </li><li><code>1</code>: (ORDER) Cores are bound in sequence. </li><li><code>2</code>: (INTERVAL) Cores are bound in an interleaved manner.</li></ul>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p46891769376"><a name="p46891769376"></a><a name="p46891769376"></a>When <code>mode</code> is set to <code>0</code>, <code>m-n</code> and <code>k</code> are invalid and can be omitted.</p>
</td>
</tr>
<tr id="row8689116183715"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p176894612375"><a name="p176894612375"></a><a name="p176894612375"></a>m-n</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p26571123153717"><a name="p26571123153717"></a><a name="p26571123153717"></a>Available CPU cores</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p768915663712"><a name="p768915663712"></a><a name="p768915663712"></a><span>The core binding range is [m, n]</span>.</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p7689660371"><a name="p7689660371"></a><a name="p7689660371"></a>m ≤ n</p>
</td>
</tr>
<tr id="row3273102020388"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p1427332019385"><a name="p1427332019385"></a><a name="p1427332019385"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p7273122013811"><a name="p7273122013811"></a><a name="p7273122013811"></a>Available CPU cores</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p727352063819"><a name="p727352063819"></a><a name="p727352063819"></a>Number of cores allocated to the TensorFlow thread.</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p6273142017388"><a name="p6273142017388"></a><a name="p6273142017388"></a>k ≤ n - m + 1 (the total number of bound cores). When <code>mode</code> is set to <code>2</code>, <code>k</code> is invalid and can be omitted.</p>
</td>
</tr>
</tbody>
</table>

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> `numactl` is a tool used to control and manage the NUMA architecture on Linux. It can be installed using Yum.
>
> ```bash
> yum install -y numactl numactl-devel
> ```
>
> For example, `numactl -C 0-79 -m 0` indicates that the TF Serving service runs on the cores of NUMA node 0, so that CPU resources can be fully utilized. `-C` and `-m` specify cores and memory of NUMA node 0, respectively.

**Recommended Scenario**

* When TensorFlow scheduling is used, sequential core binding is recommended.

* When the thread affinity isolation feature is enabled together with the `--batch_op_scheduling` option and hyper-threading is turned on, interleaved core binding is recommended.

**Sample**

Consider a server with 160 physical cores, 320 cores in total with hyper-threading enabled, 4 NUMA nodes, and 80 cores per NUMA node.

* If running with the TensorFlow scheduling mode, the following running parameters can be used as a reference:

   ```bash
   numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=75 --tensorflow_inter_op_parallelism=75 --task_affinity_isolation="1;0-79;75"
   ```

* If the `--batch_op_scheduling` option is enabled, it is recommended to set the --tensorflow_inter_op_parallelism parameter to the number of physical cores. Other running parameters can be referenced as follows:

   ```bash
   numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=40 --batch_op_scheduling=true --task_affinity_isolation="2;0-79"
   ```

## Change History

| Release Date | Description |
| ---- | ---- |
| 2026-09-30 | This is the third official release.<ul><li>Added the KEmbedding custom operator library, providing the EmbeddingTableLookup operator description.</li><li>Added the KDNN SparseMatmul multi-threading optimization interface description.</li><li>Optimized document structure.</li></ul> |
| 2026-06-30 | This is the second official release. <ul><li>Added the description for constant folding optimization to the TensorFlow ANNC for graph compilation documentation. </li><li>Added the description for the TensorFlow ANNC static graph fusion feature. </li></ul> |
| 2026-03-30 | This is the first official release. Added the TensorFlow KDNN thread passthrough feature usage instructions. |
