# Introduction to Kunpeng TensorFlow

## Latest Updates

- [2026-09-30]: Added the KEmbedding custom operator library with EmbeddingTableLookup and added KDNN SparseMatmul multithreading optimization. Restructured patch releases into maintained common, KDNN, ANNC static graph fusion, and KEmbedding groups; and froze runtime, the old fused embedding implementation, and old XLA execution features in a standalone Legacy patch.
- [2026-06-30]: Added the TensorFlow ANNC static graph fusion feature, adapted to the Kunpeng 950 7592C processor and supporting operators such as KPFusedGather and KPFusedSparseReshape. Added the constant folding optimization to TensorFlow ANNC for graph compilation, adapted to the Kunpeng 950 7592C processor.
- [2026-03-30]: Added the TensorFlow KDNN thread passthrough feature, supporting operators such as batchmatmul, concat, and softmax.
- [2025-09-30]: Added the TensorFlow ANNC for graph compilation optimization feature, providing optimizations including computational graph optimization, and generation and integration of high-performance fused operators.
- [2025-06-30]: Released the TensorFlow Serving thread scheduling optimization feature for the first time.

## Project Introduction

Kunpeng TensorFlow is a high-performance inference acceleration extension based
on open-source TensorFlow, focusing on efficient execution for search,
recommendation, and advertising inference workloads. The maintained patch
series covers shared build integration, KDNN kernel optimizations, ANNC static
graph fusion, and the KEmbedding custom operator.

Historical runtime scheduling, the old fused embedding implementation, and old XLA execution changes are
frozen in a standalone Legacy patch. They are not part of the maintained
default profiles and are not guaranteed to work with the maintained patches.

## Feature Description

| Feature | Status | Description |
| --- | --- | --- |
| KDNN thread passthrough | Maintained | Passes the TensorFlow thread pool to KDNN to reduce scheduling overhead. |
| KDNN SparseMatmul multithreading optimization | Maintained | Uses data parallelism, lock-free execution, and load balancing for sparse matrix multiplication. |
| ANNC static graph fusion | Maintained | Replaces matching embedding subgraphs with fused operators through the remapper. |
| KEmbedding custom operator | Maintained | Provides EmbeddingTableLookup for recommendation inference workloads. |
| Runtime, old fused embedding, and old XLA execution | Legacy | Published as a standalone frozen patch outside the default profiles. |

For details about Kunpeng TensorFlow features, see [Feature Introduction](./docs/en/feature_introduction.md).

## Patch Release

All patches use the pinned official TensorFlow `v2.15.0` commit
`6887368d6d46223f460358323c4b76d61d1558a8`.

| Profile | Included Patches | When to Use |
| --- | --- | --- |
| `common-only` | common | Build and compatibility changes only, with no acceleration features |
| `kdnn-core` | common + kdnn | KDNN kernel optimizations; recommended as the basic configuration |
| `kdnn-annc` | common + kdnn + annc | KDNN plus ANNC static graph fusion |
| `full-default` | common + kdnn + annc + kembedding | All currently maintained features, including KEmbedding |

The maintained series generates `tensorflow/feature_copts.bzl` for each
feature set. Use [`patches/prepare_source.py`](./patches/README_en.md) to create
a buildable source tree.

The Legacy patch applies independently to the official baseline. It does not
depend on `common` and is not guaranteed to work with other patches. See the
[Patch Release](./patches/README_en.md) document for details.

## Directory Structure

```bash
tensorflow
├── patches
│   ├── dist                                            # common, KDNN, ANNC, and KEmbedding patches
│   ├── frozen                                          # Standalone Legacy patch
│   ├── manifest.json                                   # Groups and profiles
│   ├── prepare_source.py                               # Complete source creation tool
│   ├── patch_manager.py                                # Patch maintenance and verification tool
│   └── SHA256SUMS                                      # Artifact checksums
├── LICENSE                                             # License file
├── README.md                                           # Chinese project introduction
├── README_en.md                                        # English project introduction
└── docs                                                # Chinese and English documentation
```

## Version Description

For details about the updates of the Kunpeng TensorFlow version, see [Release Notes](./docs/en/release_notes.md).

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

This project is licensed under Apache License 2.0. For details, see the [LICENSE](./LICENSE).

The documentation of this project is released under the CC-BY 4.0 license. For details, see the [LICENSE](./docs/LICENSE).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit [issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md). For details, see [Contribution Guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Kunpeng TensorFlow is jointly developed by the following Huawei department:

- Kunpeng Computing BoostKit Development Dept

Thank you for every PR from the community. We welcome your contributions to Kunpeng TensorFlow!
