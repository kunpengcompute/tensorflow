# Feature Introduction

<!-- md-trans-meta sourceCommit=c700db08dbad7073836f1f07e4120868e3d93a59 translatedAt=2026-10-12T06:49:27.355Z pushedAt=2026-10-13T02:13:34.227Z -->

## TensorFlow FAGO Static Graph Fusion

### Overview

This section introduces the basic concepts and implementation principles of the TensorFlow FAGO static graph fusion optimization feature.

To improve TensorFlow inference performance, Kunpeng BoostKit proposes a TensorFlow FAGO static graph fusion optimization solution. Kunpeng BoostKit offers the FlashAttentionForward custom operator, and during the graph optimization phase, the remapper mechanism is used to replace computational subgraphs that match specific characteristics with the custom operator. Static graph fusion achieves end-to-end performance improvement by eliminating intermediate memory overhead and optimizing memory access logic.

The FAGO static graph fusion feature switch is integrated into TensorFlow via a code patch, added on top of TensorFlow 2.20.
When the FAGO static graph fusion feature is enabled, if a subgraph of the computational graph conforms to a specific structure and its inputs and outputs satisfy the constraints, the eligible subgraph will be replaced with the corresponding custom operator during the graph optimization phase.

### Software Architecture

The software architecture of FAGO static graph fusion is shown in the following figure.

**Figure 1** Software architecture of FAGO static graph fusion

![FAGO-static-graph-fusion-architecture](./figures/FAGO-static-graph-fusion-architecture.png)

### Supported Custom Operator Specifications and Usage Constraints

This section describes the currently supported custom operator and its usage constraints.

#### FlashAttentionForward

**Open-source Subgraph Structure**

![FlashAttentionForward](figures/FlashAttentionForward.png "FlashAttentionForward open-source subgraph structure")

**Input/Output Constraints**

| Input Name | Data Type | Shape |
| -------- | ----------- | ------- |
| Input 1 | float | 3D tensor |
| Input 2 | float | 3D tensor |
| Input 3 | float | 3D tensor |
| Input 4 | float | 2D tensor |

| Output Name | Data Type | Shape |
| -------- | -------- | ------- |
| Output 1 | float | 3D tensor |
| Output 2 | float | 2D tensor |

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> When FlashAttentionForward operator fusion is enabled, if a subgraph meeting the requirements exists, it will be replaced with the corresponding custom operator during the graph optimization phase; otherwise, the open-source TensorFlow interface is used.

### Application Scenarios

TensorFlow FAGO static graph fusion is mainly used in high-concurrency inference scenarios, where it significantly reduces inference latency.

### Principles

This section describes the FAGO static graph fusion optimization feature to help users better utilize it.

After FAGO static graph fusion is enabled, a subgraph meeting the requirements is replaced with the corresponding custom operator during the graph optimization phase, thereby reducing intermediate memory overhead, optimizing memory access logic, and achieving end-to-end performance improvement.

**Figure 2** Operator fusion principle <a name="fig2"></a>

![FAGO static graph fusion principle](figures/FAGO-static-graph-fusion-principle.png "/FAGO static graph fusion principle")

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-09-30 | This is the first official release. |
