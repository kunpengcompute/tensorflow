# Feature Description

## TensorFlow ANNC for Graph Compilation Optimization

### Overview

This section describes the basic concepts and implementation principles of the TensorFlow Accelerated Neural Network Compiler (ANNC) for graph compilation optimization.

Kunpeng BoostKit provides this TensorFlow ANNC feature to enhance TensorFlow Serving (TF Serving) inference performance. ANNC is a compiler dedicated to accelerating neural network computing. It focuses on technologies including computational graph optimization, generation and integration of high-performance fused operators, and efficient code generation. These capabilities significantly improve inference performance in recommendation scenarios. ANNC is an extended acceleration suite. It is built on open-source Open Accelerated Linear Algebra (OpenXLA), and hosted in the ANNC repository maintained by the openEuler community. The suite includes optimizations tailored for the Kunpeng platform, such as TensorFlow graph fusion, Accelerated Linear Algebra (XLA) graph fusion, and operator optimization.

The ANNC optimization feature integrates with the TensorFlow inference framework and XLA through compilation options and code patches. The following new features are introduced for TensorFlow Serving/TensorFlow 2.15:

- TensorFlow graph fusion: fusion and rewriting of graphs at the TensorFlow model level.
- XLA graph fusion: XLA graph fusion enhanced by ANNC.
- Operator optimization: ANNC-driven operator optimization.

>![note](public_sys-resources/icon-note.gif) **Note:**
>OpenXLA is an open ecosystem consisting of high-performance, portable, and scalable machine learning infrastructure components.
>XLA is an open-source compiler for machine learning. It optimizes models from the TensorFlow framework, to enable efficient execution across various hardware platforms including GPUs, CPUs, and machine learning accelerators.

### Software Architecture

[**Figure 1** TF Serving software architecture](#tf-serving-software-architecture) shows the TF Serving software architecture. [**Table 1** TF Serving software component functions](#tf-serving-software-component-functions) describes the functions of each component.

**Figure 1** TF Serving software architecture<a name="fig2460131971612"></a><a id="tf-serving-software-architecture"></a>

![tf-serving-software-architecture](figures/tf-serving-software-architecture.png "TF-Serving software architecture")

**Table 1** TF Serving software component functions<a id="tf-serving-software-component-functions"></a>

<a name="table17527817415"></a>
<table><thead align="left"><tr id="row13527611645"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.3.1.1"><p id="p1452751848"><a name="p1452751848"></a><a name="p1452751848"></a>Component</p>
</th>
<th class="cellrowborder" valign="top" width="80%" id="mcps1.2.3.1.2"><p id="p3527312418"><a name="p3527312418"></a><a name="p3527312418"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9527411342"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p652710118414"><a name="p652710118414"></a><a name="p652710118414"></a>TF Serving</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p12527316419"><a name="p12527316419"></a><a name="p12527316419"></a>Dedicated, high-performance inference server optimized for TensorFlow model deployment</p>
</td>
</tr>
<tr id="row890710021716"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p175272111416"><a name="p175272111416"></a><a name="p175272111416"></a>SavedModel</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p185271018417"><a name="p185271018417"></a><a name="p185271018417"></a>TensorFlow's standardized model format enabling seamless model import, inference, and retraining across diverse TensorFlow implementations</p>
</td>
</tr>
<tr id="row552715117416"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p060405316012"><a name="p060405316012"></a><a name="p060405316012"></a>Graph Fusion</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p1860416535018"><a name="p1860416535018"></a><a name="p1860416535018"></a>ANNC graph fusion component</p>
</td>
</tr>
<tr id="row1552751643"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p652710113419"><a name="p652710113419"></a><a name="p652710113419"></a>TensorFlow</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p3527201147"><a name="p3527201147"></a><a name="p3527201147"></a>Open-source machine learning framework specializing in deep learning model training and inference</p>
</td>
</tr>
<tr id="row1145126151312"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p134819201318"><a name="p134819201318"></a><a name="p134819201318"></a>ANNC</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p183481199139"><a name="p183481199139"></a><a name="p183481199139"></a>AI compiler optimized for machine learning models, which can compile models into high-performance executable code</p>
</td>
</tr>
<tr id="row53481395136"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p117197117117"><a name="p117197117117"></a><a name="p117197117117"></a>XLA Extension</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p3719131119116"><a name="p3719131119116"></a><a name="p3719131119116"></a>ANNC XLA extension</p>
</td>
</tr>
<tr id="row512919311905"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p134526191311"><a name="p134526191311"></a><a name="p134526191311"></a>XLA</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p204518611318"><a name="p204518611318"></a><a name="p204518611318"></a><span>Open-source compiler for machine learning</span></p>
</td>
</tr>
<tr id="row116041953806"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p161299311305"><a name="p161299311305"></a><a name="p161299311305"></a>Kernels</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p2129143115012"><a name="p2129143115012"></a><a name="p2129143115012"></a>TensorFlow operator implementation</p>
</td>
</tr>
</tbody>
</table>

### Application Scenarios<a name="ZH-CN_TOPIC_0000002550048255"></a>

The TensorFlow Serving ANNC feature is mainly used in recommendation systems and advertising delivery. It can greatly improve inference performance for coarse-ranking models in high-concurrency scenarios, boosting throughput while significantly reducing latency.

### Principles<a name="ZH-CN_TOPIC_0000002550048263"></a>

This section describes the TensorFlow/XLA optimization features.

**TensorFlow Graph Fusion<a name="section2050553619512"></a>**

Some subgraphs in TensorFlow models contain redundant computations. By identifying specific graph patterns, you can fuse multiple operators in the subgraphs into one fused operator. This avoids extra work, optimizes memory access, and improves model inference performance. For details, see [**Figure 1** TensorFlow graph fusion](#tensorflow-graph-fusion). This function enables graph fusion and rewriting at the TensorFlow model level on the frontend, and supports manual creation of custom fused operators on the backend.

**Figure 1** TensorFlow graph fusion<a name="fig836316691915"></a><a id="tensorflow-graph-fusion"></a>

![tensorflow-graph-fusion](figures/tensorflow-graph-fusion.png "TensorFlow graph fusion diagram")

**XLA Graph Fusion<a name="section1049725715115"></a>**

XLA provides multiple hardware-agnostic graph fusion optimization policies. However, the resulting cluster (including the fused parts) may still contain redundant computations. For example, sub-expressions are repeated or can be merged across different fusion operations. For details, see [**Figure 2** XLA graph fusion](#xla-graph-fusion). This function aims to identify redundant computations after fusion, such as the F1 operations. Redundant computations can be eliminated using pre-fusion policies, such as the fusion of F4, F5, and F6 operations, to further improve the model inference efficiency.

**Figure 2** XLA graph fusion<a name="fig5154159193015"></a><a id="xla-graph-fusion"></a>

![xla-graph-fusion](figures/xla-graph-fusion.png "XLA graph fusion diagram")

**Operator Optimization<a name="section287331525216"></a>**

This feature performs operator optimization across stages, including offloading the Matrix Multiplication (MatMul) operator to XLA, calling the General Matrix Multiplication (GEMM) operation interface provided by Open Basic Linear Algebra Subprograms (OpenBLAS), and replacing the Softmax function with a more efficient implementation. In addition, it identifies specific operation patterns to eliminate redundant computations and further improve the model inference performance. For example, in scenarios where multiple slices are concatenated, redundant slicing operations are removed.

For details about function configuration, see <a href="./quick_start.md">Quick Start</a>.

## TensorFlow Serving Thread Scheduling

### Overview

This section describes the basic concepts and implementation principles of the thread scheduling optimization feature for TensorFlow Serving.

Kunpeng BoostKit developed a thread scheduling optimization solution to enhance TF Serving inference performance. TensorFlow employs inter-operator thread pools to parallelize independent operators, this approach can lead to task contention in high-concurrency scenarios when multiple sessions share the same thread pool, substantially degrading computational efficiency for entire graphs. Kunpeng BoostKit's solution addresses this limitation through refined operator scheduling algorithms and advanced thread management optimizations, delivering significant throughput improvements for concurrent model inference.

Implemented as patches integrated into openEuler's `sra_tensorflow_adapter` repository, these optimizations introduce two new configuration parameters for TF Serving/TensorFlow 2.15:

- Batch operator scheduling (`--batch_op_scheduling`): Enables the operator scheduling optimization and XLA thread pool management optimization features. When single-core inference latency meets requirements, this option can be used to enhance concurrent processing capability and overall throughput.
- Thread affinity isolation (`--task_affinity_isolation`): Provides the following isolation methods: When the TensorFlow scheduling mode is used, sequential core binding is recommended. When this option is enabled together with the `--batch_op_scheduling` option, and hyper-threading is enabled, interleaved core binding is recommended.
  - Sequential core binding allocates TensorFlow computing threads to the first K cores and TF Serving communication threads to remaining cores.
  - Interleaved core binding (applicable when hyper-threading is enabled) assigns TensorFlow threads to physical cores and TF Serving communication threads to virtual cores.

>![note](public_sys-resources/icon-note.gif) **Note:**
>XLA serves as TensorFlow's optimizing compiler, specifically designed to enhance the execution speed of linear algebra operations. By transforming TensorFlow computational graphs into highly efficient, hardware-specific instructions, XLA delivers significant performance improvements.

### Software Architecture

[**Figure 1** TF Serving software architecture](#tf-serving-software-architecture-1) shows the TF Serving software architecture. [**Table 1** TF Serving component functions](#tf-serving-component-functions) describes the functions of each module.

**Figure 1** TF Serving software architecture<a name="fig2460131971612"></a><a id="tf-serving-software-architecture-1"></a>

![tf-serving-software-architecture-1](figures/tf-serving-software-architecture-1.png "TF-Serving software architecture")

**Table 1** TF Serving component functions<a id="tf-serving-component-functions"></a>

<a name="table17527817415"></a>
<table><thead align="left"><tr id="row13527611645"><th class="cellrowborder" valign="top" width="23.02%" id="mcps1.2.3.1.1"><p id="p1452751848"><a name="p1452751848"></a><a name="p1452751848"></a>Component</p>
</th>
<th class="cellrowborder" valign="top" width="76.98%" id="mcps1.2.3.1.2"><p id="p3527312418"><a name="p3527312418"></a><a name="p3527312418"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9527411342"><td class="cellrowborder" valign="top" width="23.02%" headers="mcps1.2.3.1.1 "><p id="p652710118414"><a name="p652710118414"></a><a name="p652710118414"></a>TF Serving</p>
</td>
<td class="cellrowborder" valign="top" width="76.98%" headers="mcps1.2.3.1.2 "><p id="p12527316419"><a name="p12527316419"></a><a name="p12527316419"></a>Dedicated, high-performance inference server optimized for TensorFlow model deployment</p>
</td>
</tr>
<tr id="row552715117416"><td class="cellrowborder" valign="top" width="23.02%" headers="mcps1.2.3.1.1 "><p id="p652710113419"><a name="p652710113419"></a><a name="p652710113419"></a>TensorFlow</p>
</td>
<td class="cellrowborder" valign="top" width="76.98%" headers="mcps1.2.3.1.2 "><p id="p3527201147"><a name="p3527201147"></a><a name="p3527201147"></a>Open-source machine learning framework specializing in deep learning model training and inference</p>
</td>
</tr>
<tr id="row1552751643"><td class="cellrowborder" valign="top" width="23.02%" headers="mcps1.2.3.1.1 "><p id="p175272111416"><a name="p175272111416"></a><a name="p175272111416"></a>SavedModel</p>
</td>
<td class="cellrowborder" valign="top" width="76.98%" headers="mcps1.2.3.1.2 "><p id="p185271018417"><a name="p185271018417"></a><a name="p185271018417"></a>TensorFlow's standardized model format enabling seamless model import, inference, and retraining across diverse TensorFlow implementations</p>
</td>
</tr>
</tbody>
</table>

### Application scenarios

The TF Serving thread scheduling optimization feature delivers adaptable solutions for diverse inference workloads:

- It can greatly improve inference performance for coarse-ranking models in high-concurrency scenarios, boosting throughput while significantly reducing latency.
- Effectively optimizes latency-sensitive, low-concurrency scenarios through proper thread management parameter configuration.

### Principles<a name="ZH-CN_TOPIC_0000002518448414"></a>

This section details TF Serving's thread pool architecture for inference, clarifying the principles of the feature to guide optimal configuration decisions.

**Figure 1** TF Serving thread pool overview<a name="fig158087456394"></a><a id="tf-serving-thread-pool-overview"></a>

![tf-serving-thread-pool-overview](figures/tf-serving-thread-pool-overview.png "TF Serving thread pool overview")

The inference threads in TF Serving fall into two functional categories: communication threads and computing threads.

Communication threads:

- `grpcpp_sync_ser` threads manage client inference requests (including parsing, inference triggering, and response delivery).

Computing threads:

- `tf_Compute` threads coordinate parallel tasks across operators.
- `tf_numa_-1_Eige` threads execute intra-operator parallel tasks.

XLA-enabled deployments create threads for XLA computation.

- `host_executor` threads coordinate parallel tasks across XLA operators.
- `tf_XLAEigen` threads execute intra-XLA-operator parallel tasks.

[**Figure 2** Inference request handling process](#inference-request-handling-process) shows the overall inference request handling process.

**Figure 2** Inference request handling process<a name="fig1746025495015"></a><a id="inference-request-handling-process"></a>

![inference-request-handling-process](figures/inference-request-handling-process.png "inference-request-handling-process")

Client inference requests are parsed by `grpcpp_sync_ser` threads before triggering session-based inference execution. Parallel operator processing occurs through `tf_Compute or host_executor` threads, with `tf_numa_-1_Eige` or `tf_XLAEigen` threads handling intra-operator parallel computing.

Kunpeng BoostKit improves the operator scheduling algorithm and uses batch operator scheduling. [**Figure 3** Inference process after optimization](#inference-process-after-optimization) shows the overall inference process.

**Figure 3** Inference process after optimization<a name="fig1321324116542"></a><a id="inference-process-after-optimization"></a>

![inference-process-after-optimization](figures/inference-process-after-optimization.png "inference-process-after-optimization")

Client inference requests are parsed by `grpcpp_sync_ser` threads before triggering session-based inference, with operators running sequentially in `tf_Compute` threads (disabling intra-operator parallelism).

This optimization reduces cross-session interference, enabling lower per-session inference latency, improved TF Serving concurrency, and additional gains from thread affinity isolation between communication and computing threads.

The thread scheduling feature enables:

- Batch operator scheduling (via `--batch_op_scheduling`) for enhanced throughput in high-concurrency scenarios
- Optimized XLA thread pool management, enabled alongside batch operator scheduling, to schedule XLA operators onto the current thread, thereby reducing context switching overhead.
- Configurable thread affinity isolation (via `--task_affinity_isolation`) for binding communication and computing threads to different CPU cores

For details about function configuration, see <a href="./quick_start.md">Quick Start</a>.
