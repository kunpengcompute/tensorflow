# API参考

## TensorFlow ANNC图编译优化特性使用说明

TensorFlow ANNC图编译优化特性提供了TensorFlow图融合、XLA图融合、算子优化、常量折叠优化特性，本章节提供各特性接口与接口使能步骤。

### TensorFlow图融合

TensorFlow图融合接口命令与使用示例如下所示。

**终端命令行接口**

`annc-opt`

**接口功能**

图融合启动命令。

**参数说明**

* -I /path/to/save_model.pb：待图融合的模型
* -O /path/to/new_save_model.pb：图融合之后的模型
* pass：图融合策略（当前支持lookup_embedding_hash）

**使用示例**

```bash
annc-opt -I /base_model/wide_and_deep/1/ -O /optimized_model/wide_and_deep/1/ lookup_embedding_hash
cp -r /base_model/wide_and_deep/1/variables /optimized_model/wide_and_deep/1/
```

### XLA图融合

XLA图融合接口命令与使用示例如下所示。

**接口功能**

编译ANNC，使能XLA图融合优化。

**环境变量**

`ANNC_FLAGS`

**取值范围**

环境变量为“--graph-opt”开启特性。

**使用示例**

```bash
export ANNC_FLAGS="--graph-opt"
```

### 算子优化

已优化的算子接口有冗余算子、矩阵算子和Softmax算子，使用方式如[**表 1** 算子优化接口](#算子优化接口)所示。

**表 1** 算子优化接口<a id="算子优化接口"></a>

| 接口名称 | 接口功能 | 环境变量 | 取值范围 | 使用示例 |
| ---- | ---- | ---- | ---- | ---- |
| 冗余算子优化接口 | 使能冗余算子优化。 | ENABLE_BISHENG_GRAPH_OPT | 环境变量非空时开启特性。| `export ENABLE_BISHENG_GRAPH_OPT=""` |
| 矩阵算子优化接口 | 使能矩阵算子优化。 | ANNC_FLAGS | 环境变量值为“--gemm-opt”开启特性。| `export ANNC_FLAGS="--gemm-opt"` |
| Softmax算子优化接口 | 使能Softmax算子优化。 | XLA_FLAGS | 环境变量值为“--xla_cpu_enable_xnnpack=true”开启特性。 | `export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"` |

### 常量折叠优化

常量折叠和图优化同时开启时，需先开启常量折叠优化。

常量折叠模型转换接口使用如下所示。

**终端命令行接口**

`annc-opt`

**接口功能**

常量折叠启动命令。

**参数说明**

* -I /path/to/save_model.pb：待常量折叠的模型
* -O /path/to/new_save_model.pb：常量折叠之后的模型
* pass：layout_matmul

**使用示例**

```bash
annc-opt -I /base_model/wide_and_deep/1/ -O /folding/wide_and_deep/1/ layout_matmul
```

常量折叠优化接口使用如下所示。

**环境变量**

`ANNC_FLAGS`

**接口功能**

使能常量折叠优化。

**取值范围**

环境变量值为“--layout-matmul”开启特性。

**使用示例**

```bash
export ANNC_FLAGS="--layout-matmul"
```

## TensorFlow Serving线程调度特性使用说明

鲲鹏TensorFlow Serving线程调度优化通过命令行提供了算子批量调度和线程亲和性隔离两个特性开关，用户可根据实际场景自行配置。

使用TF Serving启动推理压测指导请参见《TensorFlow Serving推理部署框架移植指南》的“[启动服务并压测](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0012.html)”章节。

### 算子批量调度

算子批量调度接口使用如下所示。

**TF Serving命令行接口**

`--batch_op_scheduling`

**接口功能**

使能算子调度优化和XLA线程池管理优化特性。

**参数类型**

bool

**取值范围**

* true为真，表示开启特性。
* false表示关闭特性，默认为false。

**推荐场景**

单核推理时延可满足业务要求，可配置该选项提升推理并发能力和吞吐量。

**推荐配置**

* --tensorflow_intra_op_parallelism=1，算子内并行度设置为1。
* --tensorflow_inter_op_parallelism=80，算子间并行度设置为CPU核数。
* --batch_op_scheduling=true，开启算子批量调度特性。

**使用示例**

```bash
/path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=80 --batch_op_scheduling=true
```

### 线程亲和性隔离

线程亲和性隔离接口使用如下所示。

**TF Serving命令行接口**

`--task_affinity_isolation`

**接口功能**

使能线程亲和性隔离特性，有两种隔离方式：

* 顺序绑核，TensorFlow计算线程绑定到前K个核，TF Serving通信线程绑定到其余核。
* 交叉绑核，适用于开启超线程的场景，将TensorFlow线程绑定到物理核，TF Serving通信线程绑定到虚拟核。

**参数类型**

std::string

**参数格式**

mode;m-n;k，默认0。

**取值范围**

**表 1** 线程亲和性隔离参数格式取值说明<a id="线程亲和性隔离参数格式取值说明"></a>

<a name="table12688064377"></a>
<table><thead align="left"><tr id="row468814643718"><th class="cellrowborder" valign="top" width="7.43925607439256%" id="mcps1.2.5.1.1"><p id="p176881266377"><a name="p176881266377"></a><a name="p176881266377"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.448455154484552%" id="mcps1.2.5.1.2"><p id="p14656122373711"><a name="p14656122373711"></a><a name="p14656122373711"></a>取值范围</p>
</th>
<th class="cellrowborder" valign="top" width="27.90720927907209%" id="mcps1.2.5.1.3"><p id="p1768920617371"><a name="p1768920617371"></a><a name="p1768920617371"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="49.2050794920508%" id="mcps1.2.5.1.4"><p id="p3689265373"><a name="p3689265373"></a><a name="p3689265373"></a>约束</p>
</th>
</tr>
</thead>
<tbody><tr id="row168915612378"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p196891768371"><a name="p196891768371"></a><a name="p196891768371"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p265616232373"><a name="p265616232373"></a><a name="p265616232373"></a>0、1、2</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><a name="ul1088015211937"></a><a name="ul1088015211937"></a><ul id="ul1088015211937"><li>0：OFF，不使能线程亲和。</li><li>1：ORDER，按顺序绑核。</li><li>2：INTERVAL，交叉绑核。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p46891769376"><a name="p46891769376"></a><a name="p46891769376"></a>mode=0时，m-n、k两个参数无效（可不填）。</p>
</td>
</tr>
<tr id="row8689116183715"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p176894612375"><a name="p176894612375"></a><a name="p176894612375"></a>m-n</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p26571123153717"><a name="p26571123153717"></a><a name="p26571123153717"></a>可用的CPU核</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p768915663712"><a name="p768915663712"></a><a name="p768915663712"></a><span>绑核范围[m, n]</span>。</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p7689660371"><a name="p7689660371"></a><a name="p7689660371"></a><span>m </span><span>&lt;</span><span>= n</span>。</p>
</td>
</tr>
<tr id="row3273102020388"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p1427332019385"><a name="p1427332019385"></a><a name="p1427332019385"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p7273122013811"><a name="p7273122013811"></a><a name="p7273122013811"></a>可用的CPU核</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p727352063819"><a name="p727352063819"></a><a name="p727352063819"></a>分配给TensorFlow线程的核数。</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p6273142017388"><a name="p6273142017388"></a><a name="p6273142017388"></a>k &lt;= n - m + 1，即不大于绑核总数；mode=2时，参数k无效（可不填）。</p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **说明：** 
>numactl是一个在Linux系统上用于控制和管理NUMA（非统一内存访问，Non-Uniform Memory Access）架构的工具。可通过yum工具安装：
>
>```bash
>yum install -y numactl numactl-devel
>```
>
>**numactl -C 0-79 -m 0**是限定TF Serving服务运行在NUMA 0对应的核上，以该方式启动可以充分利用CPU资源，-C指定NUMA 0对应的核，-m指的是使用NUMA 0对应的内存。

**推荐场景**

* 使用TensorFlow调度方式运行时，推荐设置为顺序绑核。
* 与--batch_op_scheduling选项同时使能，并开启超线程时，推荐设置为交叉绑核。

**使用示例**

一台160个物理核的服务器，开启超线程共320个核心，4个NUMA，每个NUMA上80个核心。

* 如果使用TensorFlow调度方式运行，运行参数可参考：

  ```bash
  numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=75 --tensorflow_inter_op_parallelism=75 --task_affinity_isolation="1;0-79;75"
  ```

* 如果使能了--batch_op_scheduling选项，--tensorflow_inter_op_parallelism参数推荐设置为物理核数量，其他运行参数可参考：

  ```bash
  numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=40 --batch_op_scheduling=true --task_affinity_isolation="2;0-79"
  ```

## TensorFlow KDNN线程直通特性使用说明

TensorFlow KDNN线程直通特性通过进程环境变量控制，具体说明如下所示。

**变量类型**

进程环境变量

**变量名称**

`TF_ENABLE_KDNN_OPTS`

**变量功能**

控制KDNN优化特性的启用。

**变量取值**

* 0：KDNN特性关闭。

* 1：KDNN特性开启。

**使用示例**

首次调用KDNN算子前设置进程环境变量TF_ENABLE_KDNN_OPTS，例如python内可以通过命令os.environ['TF_ENABLE_KDNN_OPTS'] = str(1)设置进程环境变量。

## TensorFlow ANNC静态图融合特性使用说明

TensorFlow ANNC静态图融合特性开关通过环境变量开关控制，具体说明如[表1 ANNC静态图融合特性开关](#table47361837821)所示。

特性开关默认取值为0，即关闭ANNC静态图融合功能，如需使用需要手动在图编译阶段前设置环境变量开启，以Python语言为例可以通过如下方式设置：

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
```

**表 1**  ANNC静态图融合特性开关

<a name="table47361837821"></a>

| 开关名 | 类型 | 取值 | 功能 |
| ---- | -------- | ---- | ---- |
| ANNC_FUSED_EMB_ACTIONID_GATHER | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedEmbeddingActionIdGather算子开启ANNC静态图融合。 |
| ANNC_FUSED_GATHER | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedGather算子开启ANNC静态图融合。 |
| ANNC_FUSED_EMD_PADDING | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedEmbeddingPadding算子开启ANNC静态图融合。 |
| ANNC_FUSED_EMD_PADDING_FAST | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedEmbeddingPaddingFast算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_STITCH | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedSparseDynamicStitch算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_RESHAPE | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedSparseReshape算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_REDUCE | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedSparseSegmentReduce算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_REDUCE_NONZERO | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedSparseSegmentReduceNonzero算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_SELECT | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于KPFusedSparseSelect算子开启ANNC静态图融合。 |
| ANNC_FUSED_ALL | 进程环境变量 | 1：开启 <br> 0：关闭 | 用于所有ANNC融合算子开启ANNC静态图融合。 |

>![](public_sys-resources/icon-note.gif) **说明：** 
>以上算子当且仅当ANNC_FUSED_ALL=0且算子对应的环境变量为0时，该算子不会进行算子融合。

## 修订记录

| 发布日期 | 修订记录 |
| ---- | ---- |
| 2026-06-30 | 第二次正式发布。<ul><li>TensorFlow ANNC图编译优化特性增加常量折叠优化特性接口说明内容。</li><li>新增TensorFlow ANNC静态图融合特性使用说明内容。</li></ul> |
| 2026-03-30 | 第一次正式发布。 <ul><li>新增TensorFlow KDNN线程直通特性使用说明内容。</li></ul>|
