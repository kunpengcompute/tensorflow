# Introduction to Kunpeng TensorFlow

<!-- md-trans-meta sourceCommit=ce9dacf78e8286760abd9d08a5f4573f1749dced translatedAt=2026-08-31T03:36:39.719Z pushedAt=2026-09-11T10:56:31.003Z -->

English|[简体中文](./README.md)

## Latest Updates

- \[2026-09-30\]: Introduced the FlashAttentionGraphOptimization (FAGO) static graph fusion feature in TensorFlow V2.20.0, supporting the FlashAttentionForward operator. Added the KEmbedding custom operator library with EmbeddingTableLookup and added KDNN SparseMatmul multithreading optimization in TensorFlow V2.15.0; restructured patch releases into maintained common, KDNN, ANNC static graph fusion, and KEmbedding groups; and froze runtime, the old fused embedding implementation, and old XLA execution features into a standalone Legacy patch.

- \[2026-06-30\]: Added the TensorFlow ANNC static graph fusion feature in TensorFlow V2.15.0, adapted to the Kunpeng 950 processor and supporting operators such as KPFusedGather and KPFusedSparseReshape. Added constant folding optimization to TensorFlow ANNC for graph compilation, adapted to the Kunpeng 950 processor.

- \[2026-03-30\]: Added the TensorFlow KDNN thread passthrough feature in TensorFlow V2.15.0, supporting operators such as batchmatmul, concat, and softmax.

- \[2025-09-30\]: Added the TensorFlow ANNC for graph compilation optimization feature in TensorFlow V2.15.0, providing optimizations including computational graph optimization, and generation and integration of high-performance fused operators.

- \[2025-06-30\]: Released the TensorFlow Serving thread scheduling optimization feature for the first time.

## Project Introduction

Kunpeng TensorFlow is a high-performance inference acceleration extension based on open-source TensorFlow, focusing on efficient execution for search, recommendation, and advertising inference workloads.

The maintained patch series covers shared build integration, KDNN kernel optimizations, ANNC static graph fusion, and the KEmbedding custom operator.

Historical runtime scheduling, the old fused embedding implementation, and old XLA execution changes are frozen into a standalone Legacy patch. It is not part of the maintained default profiles and is not guaranteed to work with the maintained patches.

## Directory Structure

```bash
tensorflow
└── patches
    └── Tensorflow_V2.15.0                              # TensorFlow 2.15.0 patch directory
        ├── feature                                     # Common build integration (common), KDNN kernel optimizations, ANNC static graph fusion, and KEmbedding custom operator
        ├── frozen_feature                              # Standalone Legacy patch
        ├── manifest.json                               # Patch grouping and profile definition
        ├── prepare_source.py                           # Complete source code creation tool
        ├── patch_manager.py                            # Patch maintenance and verification tool
        └── SHA256SUMS                                  # Release file checksum
    └── Tensorflow_V2.20.0                              # TensorFlow 2.20.0 patch directory
        ├── feature                                     # Common build integration (common), FAGO static graph fusion
        ├── manifest.json                               # Patch grouping and profile definition
        ├── prepare_source.py                           # Complete source code creation tool
        ├── patch_manager.py                            # Patch maintenance and verification tool
        └── SHA256SUMS                                  # Release file checksum
├── LICENSE                                             # License file
├── README.md                                           # Project introduction
├── README_en.md                                        # Project introduction in English
└── docs                                                # Documentation
    └── zh                                              # Chinese documentation directory
        └── Tensorflow_V2.15.0                          # TensorFlow 2.15.0 directory
            ├── figures                                 # Image resource directory
            ├── api_reference.md                        # API Reference
            ├── quick_start.md                          # Quick Start
            ├── release_notes.md                        # Release Notes
            ├── installation_guide.md                   # Installation Guide
            └── feature_introduction.md                 # Feature Introduction
        └── Tensorflow_V2.20.0                          # TensorFlow 2.20.0 directory
            ├── figures                                 # Image resource directory
            ├── api_reference.md                        # API Reference
            ├── quick_start.md                          # Quick Start
            ├── release_notes.md                        # Release Notes
            ├── installation_guide.md                   # Installation Guide
            └── feature_introduction.md                 # Feature Introduction
    └── en                                              # English documentation directory
        └── Tensorflow_V2.15.0                          # Tensorflow 2.15.0 directory
            ├── figures                                 # Image resource directory
            ├── api_reference.md                        # API Reference
            ├── quick_start.md                          # Quick Start
            ├── release_notes.md                        # Release Notes
            ├── installation_guide.md                   # Installation Guide
            └── feature_introduction.md                 # Feature Introduction
        └── Tensorflow_V2.20.0                          # TensorFlow 2.20.0 directory
            ├── figures                                 # Image resource directory
            ├── api_reference.md                        # API Reference
            ├── quick_start.md                          # Quick Start
            ├── release_notes.md                        # Release Notes
            ├── installation_guide.md                   # Installation Guide
            └── feature_introduction.md                 # Feature Introduction
```

## Tensorflow V2.15.0

### Tensorflow V2.15.0 Feature Description

| Feature | Status | Description |
| --- | --- | --- |
| KDNN thread passthrough | Maintained | Passes the TensorFlow thread pool to KDNN to reduce scheduling overhead. |
| KDNN SparseMatmul multithreading optimization | Maintained | Uses data parallelism, lock-free execution, and load balancing for sparse matrix multiplication. |
| ANNC static graph fusion | Maintained | Replaces matching embedding subgraphs with fused operators through the remapper. |
| KEmbedding custom operator | Maintained | Provides EmbeddingTableLookup for recommendation inference workloads. |
| Runtime, old fused embedding, and old XLA execution | Legacy | Published as a standalone frozen patch outside the default profiles. |

For detailed information about the features of Kunpeng TensorFlow V2.15.0, see [Feature Introduction](./docs/en/Tensorflow_V2.15.0/feature_introduction.md).

### TensorFlow V2.15.0 Patch Release

All patches use the pinned official TensorFlow `v2.15.0` commit `6887368d6d46223f460358323c4b76d61d1558a8`.

| Profile | Included Patches | When to Use |
| --- | --- | --- |
| `common-only` | common | Build and compatibility changes only, with no acceleration features. |
| `kdnn-core` | common + KDNN | KDNN kernel optimizations; recommended as the basic configuration. |
| `kdnn-annc` | common + KDNN + ANNC | KDNN plus ANNC static graph fusion. |
| `full-default` | common + KDNN + ANNC + KEmbedding | All currently maintained features, including KEmbedding. |

The maintained series requires generating `tensorflow/feature_copts.bzl` for each feature set. See [README_en](./patches/Tensorflow_V2.15.0/README_en.md) in the `patches` directory to build the source.

The Legacy patch can only be applied independently to the official baseline. It does not depend on the common build integration (common) and is not guaranteed to work with other patches. See [README_en](./patches/Tensorflow_V2.15.0/README_en.md) for details.

## Tensorflow V2.20.0

### Tensorflow V2.20.0 Feature Description

| Feature | Status | Description |
| --- | --- | --- |
| FlashAttentionGraphOptimization (FAGO) static graph fusion | Maintained | Provides the FlashAttentionForward operator for recommendation inference scenarios. |

For detailed information about the features of Kunpeng TensorFlow V2.20.0, see [Feature Introduction](./docs/en/Tensorflow_V2.20.0/feature_introduction.md).

### TensorFlow V2.20.0 Patch Release

All patches use the pinned official TensorFlow `v2.20.0` commit `bf5899deaf70fa45173c5c7b8dc9ace8824dc980`.

| Profile | Included Patches | When to Use |
| --- | --- | --- |
| `common-only` | common | Build and compatibility changes only, with no acceleration features. |
| `fago` | common+FAGO | FlashAttention fused operator optimization. |

The maintained series requires generating `tensorflow/feature_copts.bzl` for each feature set. See [README_en](./patches/Tensorflow_V2.20.0/README_en.md) in the `patches` directory to build the source.

## Version Description

For the version updates of Kunpeng TensorFlow V2.15.0, see [Release Notes](./docs/en/Tensorflow_V2.15.0/release_notes.md).

For the version updates of Kunpeng TensorFlow V2.20.0, see [Release Notes](./docs/en/Tensorflow_V2.20.0/release_notes.md).

## Documents

<table>
<thead align="left">
<tr id="row1291816372202">
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016">TensorFlow V2.15.0 Learning Resource</p></th>
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.3"><p id="p13918183762016">TensorFlow V2.20.0 Learning Resource</p></th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.4"><p id="p89181437152019">Description</p></th>
</tr>
</thead>
<tbody>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/en/Tensorflow_V2.15.0/release_notes.md">Release Notes</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p2091893722011"><a href="./docs/en/Tensorflow_V2.20.0/release_notes.md">Release Notes</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p491893752010">Provides basic information and feature updates of each Kunpeng TensorFlow release.</p></td>
</tr>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/en/Tensorflow_V2.15.0/feature_introduction.md">Feature Introduction</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p2091893722011"><a href="./docs/en/Tensorflow_V2.20.0/feature_introduction.md">Feature Introduction</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p491893752010">Describes the Kunpeng TensorFlow features.</p></td>
</tr>
<tr id="row939116371143">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p03913372046"><a href="./docs/en/Tensorflow_V2.15.0/quick_start.md">Quick Start</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p03913372046"><a href="./docs/en/Tensorflow_V2.20.0/quick_start.md">Quick Start</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p1139217371746">Provides guidance for getting started with Kunpeng TensorFlow.</p></td>
</tr>
<tr id="row2918153732017">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p17918337172020"><a href="./docs/en/Tensorflow_V2.15.0/installation_guide.md">Installation Guide</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p17918337172020"><a href="./docs/en/Tensorflow_V2.20.0/installation_guide.md">Installation Guide</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p15918183742018">Describes how to compile and install Kunpeng TensorFlow.</p></td>
</tr>
<tr id="row2918153732017">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p17918337172020"><a href="./docs/en/Tensorflow_V2.15.0/api_reference.md">API Reference</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p17918337172020"><a href="./docs/en/Tensorflow_V2.20.0/api_reference.md">API Reference</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p15918183742018">Describes how to use Kunpeng TensorFlow APIs.</p></td>
</tr>
</tbody>
</table>

## Disclaimer

This code repository contributes to the TensorFlow community. It strictly adheres to the coding style and methods, as well as security design of the open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

This project is licensed under Apache License 2.0. For details, see the [LICENSE](./LICENSE).

The documentation of this project is released under the CC-BY 4.0 license. For details, see the [LICENSE](./docs/LICENSE).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit [issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md). For details, see [Contribution Guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Thank you for every PR from the community. We welcome your contributions to Kunpeng TensorFlow!
