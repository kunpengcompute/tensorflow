# API Reference

## Feature Description of TensorFlow ANNC for Graph Compilation Optimization

TensorFlow ANNC for graph compilation optimization provides the following features: TensorFlow graph fusion, XLA graph fusion, operator optimization, and constant folding optimization. This document describes APIs of each feature and the steps required to enable them.

### TensorFlow Graph Fusion

The TensorFlow graph fusion interface commands and usage examples are shown below.

**Command Line Interface**

<code>annc-opt</code>

**Function**

Triggers the graph fusion feature.

**Parameter Description**

* <code>-I /path/to/save_model.pb</code>: model before graph fusion
* <code>-O /path/to/new_save_model.pb</code>: model after graph fusion
* <code>pass</code>: graph fusion strategy (Currently, <code>lookup_embedding_hash</code> is supported.)

**Example**

```bash
annc-opt -I /base_model/wide_and_deep/1/ -O /optimized_model/wide_and_deep/1/ lookup_embedding_hash
cp -r /base_model/wide_and_deep/1/variables /optimized_model/wide_and_deep/1/
```

### XLA Graph Fusion

The XLA graph fusion interface commands and usage examples are shown below.

**Function**

Compiles ANNC and enables XLA graph fusion optimization.

**Environment Variable**

<code>ANNC_FLAGS</code>

**Value**

Enables the feature when the environment variable is <code>--graph-opt</code>.

**Example**

```bash
export ANNC_FLAGS="--graph-opt"
```

### Operator Optimization

The optimized operator interfaces include redundant operators, matrix operators, and Softmax operators, and their usage is shown in [**Table 1** Operator Optimization Interfaces](#Operator Optimization Interfaces).

**Table 1** Operator Optimization Interfaces<a id="Operator Optimization Interfaces"></a>

| Name | Function | Environment Variable | Value | Example |
| ---- | ---- | ---- | ---- | ---- |
| redundant operator optimization | Enables redundant operator optimization. | ENABLE_BISHENG_GRAPH_OPT | Enables the feature when the environment variable is not null. | `export ENABLE_BISHENG_GRAPH_OPT=""` |
| matrix operator optimization | Enables matrix operator optimization. | ANNC_FLAGS | Enables the feature when the environment variable is <code>--gemm-opt</code>.| `export ANNC_FLAGS="--gemm-opt"` |
| Softmax operator optimization | Enables Softmax operator optimization. | XLA_FLAGS | Enables the feature when the environment variable is <code>--xla_cpu_enable_xnnpack=true</code>. | `export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"` |

### Constant Folding Optimization

When both constant folding and graph optimization are enabled, constant folding must be performed first.

The constant folding model conversion interface is used as shown below.

**Command Line Interface**

`annc-opt`

**Function**

Triggers the constant folding feature.

**Parameter Description**

* <code>-I /path/to/save_model.pb</code>: model before constant folding
* <code>-O /path/to/new_save_model.pb</code>: model after constant folding
* <code>pass</code>: layout_matmul

**Example**

```bash
annc-opt -I /base_model/wide_and_deep/1/ -O /folding/wide_and_deep/1/ layout_matmul
```

The Constant folding interfaces are shown below.

**Environment Variable**

`ANNC_FLAGS`

**Function**

Enables constant folding optimization.

**Value**

Enables the feature when the environment variable is <code>--layout-matmul</code>.

**Example**

```bash
export ANNC_FLAGS="--layout-matmul"
```

## Feature Description of TensorFlow Serving Thread Scheduling

Kunpeng's TensorFlow Serving thread scheduling feature provides two configuration options: batch operator scheduling and thread affinity isolation. You can configure the options based on your specific requirements.

To use TensorFlow Serving to start an inference stress test, see section [Starting the Service and Performing a Pressure Test](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0012.html) in the _TensorFlow Serving Porting Guide_.

### Batch Operator Scheduling

The Batch Operator Scheduling interface is used as shown below.

**TF Serving Command Line Interface**

`--batch_op_scheduling`

**Function**

Enables the operator scheduling optimization and XLA thread pool management optimization features.

**Parameter Type**

bool

**Value Range**

<code>true</code> or <code>false</code>. Set it to <code>true</code> to enable the feature or <code>false</code> to disable the feature.

**Recommended Scenario**

Recommended when single-core inference latency meets requirements. This option enhances concurrent processing capability and overall throughput.

**Recommended Configuration**

* <code>--tensorflow_intra_op_parallelism=1</code>: Sets the intra-operator parallelism degree to 1.
* <code>--tensorflow_inter_op_parallelism=80</code>: Sets the inter-operator parallelism degree to the number of CPU cores.
* <code>--batch_op_scheduling=true</code>: Enables the batch operator scheduling feature.

**Example**

```bash
/path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=80 --batch_op_scheduling=true
```

### Thread Affinity Isolation

The Thread Affinity Isolation interface is used as shown below.

**TF Serving Command Line Interface**

`--task_affinity_isolation`

**Function**

Enables the thread affinity isolation feature, which offers two isolation methods:

* Sequential core binding allocates TensorFlow computing threads to the first K cores and TF Serving communication threads to remaining cores.
* Interleaved core binding (applicable when hyper-threading is enabled) assigns TensorFlow threads to physical cores and TF Serving communication threads to virtual cores.

**Parameter Type**

std::string

**Parameter Format**

mode;m-n;k. The default value is <code>0</code>.

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

>![](public_sys-resources/icon-note.gif) **NOTE:**
>`numactl` is a tool used to control and manage the NUMA architecture on Linux. It can be installed using Yum.
>
>```bash
>yum install -y numactl numactl-devel
>```
>
>For example, `numactl -C 0-79 -m 0` indicates that the TF Serving service runs on the cores of NUMA node 0, so that CPU resources can be fully utilized. `-C` and `-m` specify cores and memory of NUMA node 0, respectively.

**Recommended Scenario**

* When TensorFlow scheduling is used, sequential core binding is recommended.
* When both batch operator scheduling and thread affinity isolation are used, and hyper-threading is enabled, interleaved core binding is recommended.

**Example**

A server has four Non-Uniform Memory Access (NUMA) nodes, each containing 40 physical cores (160 in total) or 80 logical cores (320 in total) with hyper-threading enabled.

* For TensorFlow scheduling mode, use these reference parameters:

  ```bash
  numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=75 --tensorflow_inter_op_parallelism=75 --task_affinity_isolation="1;0-79;75"
  ```

* With <code>--batch_op_scheduling</code> enabled, set <code>--tensorflow_inter_op_parallelism</code> to match the physical core count, use these reference parameters:

  ```bash
  numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=40 --batch_op_scheduling=true --task_affinity_isolation="2;0-79"
  ```

## Using the TensorFlow KDNN Thread Passthrough Feature

The TensorFlow KDNN thread passthrough feature is controlled by a Process environment variable. as detailed below.

**Type**

Process environment variable

**Environment Variable**

`TF_ENABLE_KDNN_OPTS`

**Function**

Enables or disables KDNN.

**Value Range**

* 0: KDNN disabled
* 1: KDNN enabled

**Example**

Before the first KDNN operator call, configure <code>TF_ENABLE_KDNN_OPTS</code>, for example, <code>os.environ['TF_ENABLE_KDNN_OPTS'] = str(1)</code> in the python environment.

## Usage of the TensorFlow ANNC Static Graph Fusion Feature

The TensorFlow ANNC static graph fusion feature is enabled or disabled by environment variables. For details, see [Table 1 Environment variables for enabling or disabling ANNC static graph fusion](#table47361837821)

The default value of each environment variable is `0`, indicating that the ANNC static graph fusion function is disabled. To use this function, you need to manually set the environment variable before the graph compilation. For example, you can set the environment variable in Python as follows:

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
```

**Table 1** Environment variables for enabling or disabling ANNC static graph fusion

<a name="table47361837821"></a>

| Environment Variable | Type | Value | Function |
| ---- | -------- | ---- | ---- |
| ANNC_FUSED_EMB_ACTIONID_GATHER | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedEmbeddingActionIdGather operator. |
| ANNC_FUSED_GATHER | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedGather operator. |
| ANNC_FUSED_EMD_PADDING | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedEmbeddingPadding operator. |
| ANNC_FUSED_EMD_PADDING_FAST | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedEmbeddingPaddingFast operator. |
| ANNC_FUSED_SPS_STITCH | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseDynamicStitch operator. |
| ANNC_FUSED_SPS_RESHAPE | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseReshape operator. |
| ANNC_FUSED_SPS_REDUCE | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseSegmentReduce operator. |
| ANNC_FUSED_SPS_REDUCE_NONZERO | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseSegmentReduceNonzero operator. |
| ANNC_FUSED_SPS_SELECT | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseSelect operator. |
| ANNC_FUSED_ALL | Process environment variable | <code>1</code>: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for all ANNC fusion operators. |

>![](public_sys-resources/icon-note.gif) **NOTE:**
>Operator fusion will not be performed for any of the above operators if and only if `ANNC_FUSED_ALL` is set to `0` and the environment variable corresponding to the specific operator is also set to `0`.

## Description

| Release Date | Change History |
| ---- | ---- |
| 2026-06-30 | This is the second official release. <ul><li>Added the description for constant folding optimization to the TensorFlow ANNC for graph compilation documentation. </li><li>Added the description for the TensorFlow ANNC static graph fusion feature. </li></ul> |
| 2026-03-30 | This is the first official release. <ul><li>Added the description for the TensorFlow KDNN thread passthrough feature.</li></ul> |
