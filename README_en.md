# Introduction to Kunpeng TensorFlow

## Latest Updates

- [2025.09.30]: Added the TensorFlow ANNC for graph compilation optimization feature, providing optimizations including computational graph optimization, and generation and integration of high-performance fused operators.
- [2025.06.30]: Released the TensorFlow Serving thread scheduling optimization feature for the first time.

## Project Introduction

Kunpeng TensorFlow is a high-performance inference acceleration extension based on open-source TensorFlow. It focuses on efficient execution in search, recommendation, and advertising inference scenarios. It significantly improves throughput and cuts latency for model inference through in-depth enhancements in graph optimization, operators, and runtime, providing top performance for AI applications based on Kunpeng CPUs.

Pixie provides the following features:

- ANNC for graph compilation optimization: It leverages Kunpeng hardware affinity operators and TensorFlow graph fusion technologies to accelerate model inference.
- Thread scheduling optimization: It uses Kunpeng affinity operator scheduling and thread pool management technologies to optimize TensorFlow operator scheduling in high-concurrency scenarios.

**Figure 1** Project architecture<a name="fig1326111445508"></a>
![project-architecture](./docs/en/figures/project-architecture.png "project-architecture")

- Executor layer: runtime optimization
- Kernel layer: custom operators, which provide high-performance DNN operators based on KDNN.
- XLA layer: provides the Kunpeng graph compiler based on ANNC.

## Feature Description

<table>
<thead align="left">
<tr id="row1291816372202">
<th class="cellrowborder" valign="top" width="9.780978097809781%" id="mcps1.1.4.1.1"><p id="p291823714205">Feature</p></th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019">Description</p></th>
</tr>
</thead>
<tbody>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1918123710208">Thread scheduling optimization</p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p491893752010">Refines operator scheduling algorithms and provides other thread management optimizations, delivering throughput improvements for concurrent model inference.</p></td>
</tr>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1918123710208">ANNC for graph compilation optimization</p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p491893752010">ANNC is a compiler dedicated to accelerating neural network computing. It focuses on technologies including computational graph optimization, generation and integration of high-performance fused operators, and efficient code generation and optimization. These capabilities significantly improve inference performance in recommendation scenarios.</p></td>
</tr>
</tbody>
</table>

For details about the features of Kunpeng TensorFlow, see [Feature Introduction](./docs/en/feature_introduction.md).

## Release Notes

For details about the version updates of Kunpeng TensorFlow, see [Release Notes](./docs/en/release_notes.md).

## Directory Structure

 ```bash
tensorflow
├── 0001-tensorflow_2.15.0-optimize.patch // TensorFlow patch file
├── LICENSE                                   // License file
├── README.md                                 // Open-source repository introduction
└── docs                                      // Documentation
 ```

## Documents

<table>
<thead align="left">
<tr id="row1291816372202">
<th class="cellrowborder" valign="top" width="9.780978097809781%" id="mcps1.1.4.1.1"><p id="p291823714205">Resource Type</p></th>
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016">Resource Name</p></th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019">Resource Description</p></th>
</tr>
</thead>
<tbody>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1918123710208">Document</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/en/release_notes.md">Release Notes</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p491893752010">Provides basic information and feature updates of each Kunpeng TensorFlow release.</p></td>
</tr>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1918123710208">Document</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/en/feature_introduction.md">Feature Introduction</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p491893752010">Describes the Kunpeng TensorFlow features.</p></td>
</tr>
<tr id="row939116371143">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1039163711413">Document</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p03913372046"><a href="./docs/en/quick_start.md">Quick Start</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p1139217371746">Provides guidance for getting started with Kunpeng TensorFlow.</p></td>
</tr>
<tr id="row2918153732017">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p598512211214">Document</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p17918337172020"><a href="./docs/en/installation_guide.md">Installation Guide</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p15918183742018">Describes how to compile and install Kunpeng TensorFlow.</p></td>
</tr>
<tr id="row2918153732017">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p598512211214">Document</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p17918337172020"><a href="./docs/en/api_reference.md">API Reference</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p15918183742018">Describes how to use Kunpeng TensorFlow APIs.</p></td>
</tr>
</tbody>
</table>

## Disclaimer

This code repository contributes to the TensorFlow community. It strictly adheres to the coding style and methods, as well as security design of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

This project is licensed under Apache License 2.0. For details, see the <a href="./docs/LICENSE">LICENSE</a> file.

This project document is licensed under CC-BY 4.0. For details, see the <a href="./docs/LICENSE">LICENSE</a> file.

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit [issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md). For details, see [Contribution Guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in the [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Kunpeng TensorFlow is jointly developed by the following Huawei department:

- Kunpeng Computing BoostKit Development Dept

Thank you for every PR from the community. We welcome your contributions to Kunpeng TensorFlow!
