# 鲲鹏TensorFlow介绍

简体中文|[English](./README_en.md)

## 最新消息

- \[2026.09.30\]：Tensorflow V2.20.0新增FlashAttentionGraphOptimization(FAGO)静态图融合特性，支持FlashAttentionForward算子。Tensorflow V2.15.0新增KEmbedding自定义算子库，提供EmbeddingTableLookup算子；新增KDNN SparseMatmul多线程优化；重构补丁发布方式，按公共构建集成（common）、KDNN算子优化、ANNC静态图融合和KEmbedding自定义算子四个维度进行拆分和补丁维护，将Runtime、旧融合Embedding及旧XLA执行能力冻结为独立Legacy补丁。
- \[2026.06.30\]：Tensorflow V2.15.0新增TensorFlow ANNC静态图融合特性, 适配鲲鹏950 7592C处理器，支持KPFusedGather、KPFusedSparseReshape等算子；TensorFlow ANNC图编译优化特性新增常量折叠优化，适配鲲鹏950 7592C处理器。
- \[2026.03.30\]：Tensorflow V2.15.0新增TensorFlow KDNN线程直通特性, 支持batchmatmul、concat、softmax等算子对接KDNN。
- \[2025.09.30\]：Tensorflow V2.15.0新增TensorFlow ANNC图编译优化特性，提供计算图优化，高性能融合算子生成与对接等优化技术。
- \[2025.06.30\]：TensorFlow Serving线程调度优化特性首次发布。

## 项目介绍

鲲鹏TensorFlow是基于开源TensorFlow的高性能推理加速扩展，聚焦于搜推广推理场景下的高效执行。

当前维护的补丁系列覆盖公共构建集成（common）、KDNN算子优化、ANNC静态图融合，以及KEmbedding自定义算子。

历史Runtime调度、旧融合Embedding和旧XLA执行等功能已冻结为独立Legacy补丁。Legacy补丁不属于当前维护的默认特性组合，也不保证与其他特性补丁兼容。

## 目录结构

```bash
tensorflow
└── patches
    └── Tensorflow_V2.15.0                              # Tensorflow 2.15.0补丁目录
        ├── feature                                     # 公共构建集成（common）、KDNN算子优化、ANNC静态图融合和KEmbedding自定义算子
        ├── frozen_feature                              # 独立Legacy补丁
        ├── manifest.json                               # 补丁分组和Profile定义
        ├── prepare_source.py                           # 完整源码创建工具
        ├── patch_manager.py                            # 补丁维护与验证工具
        └── SHA256SUMS                                  # 发布文件校验值
    └── Tensorflow_V2.20.0                              # Tensorflow 2.20.0补丁目录
        ├── feature                                     # 公共构建集成（common）、FAGO静态图融合
        ├── manifest.json                               # 补丁分组和Profile定义
        ├── prepare_source.py                           # 完整源码创建工具
        ├── patch_manager.py                            # 补丁维护与验证工具
        └── SHA256SUMS                                  # 发布文件校验值
├── LICENSE                                             # License文件
├── README.md                                           # 项目介绍
├── README_en.md                                        # 英文项目介绍
└── docs                                                # 文档
    └── zh                                              # 中文文档目录
        └── Tensorflow_V2.15.0                          # Tensorflow 2.15.0目录
            ├── figures                                 # 图片资源目录
            ├── api_reference.md                        # API参考
            ├── quick_start.md                          # 快速入门
            ├── release_notes.md                        # 版本说明书
            ├── installation_guide.md                   # 安装指南
            └── feature_introduction.md                 # 特性介绍
        └── Tensorflow_V2.20.0                          # Tensorflow 2.20.0目录
            ├── figures                                 # 图片资源目录
            ├── api_reference.md                        # API参考
            ├── quick_start.md                          # 快速入门
            ├── release_notes.md                        # 版本说明书
            ├── installation_guide.md                   # 安装指南
            └── feature_introduction.md                 # 特性介绍
    └── en                                              # 英文文档目录
        └── Tensorflow_V2.15.0                          # Tensorflow 2.15.0目录
            ├── figures                                 # 图片资源目录
            ├── api_reference.md                        # API参考
            ├── quick_start.md                          # 快速入门
            ├── release_notes.md                        # 版本说明书
            ├── installation_guide.md                   # 安装指导
            └── feature_introduction.md                 # 特性介绍
        └── Tensorflow_V2.20.0                          # Tensorflow 2.20.0目录
            ├── figures                                 # 图片资源目录
            ├── api_reference.md                        # API参考
            ├── quick_start.md                          # 快速入门
            ├── release_notes.md                        # 版本说明书
            ├── installation_guide.md                   # 安装指导
            └── feature_introduction.md                 # 特性介绍    
```

## Tensorflow V2.15.0

### Tensorflow V2.15.0特性介绍

| 特性 | 状态 | 简介 |
| --- | --- | --- |
| KDNN线程直通 | 维护中 | 将TensorFlow线程池透传到KDNN算子库，减少线程调度开销。 |
| KDNN SparseMatmul多线程优化 | 维护中 | 通过数据并行、无锁设计和负载均衡降低稀疏矩阵乘法耗时。 |
| ANNC静态图融合 | 维护中 | 通过remapper机制将符合固定结构的Embedding子图替换为融合算子。 |
| KEmbedding自定义算子 | 维护中 | 提供面向推荐推理场景的EmbeddingTableLookup算子。 |
| Runtime、旧融合Embedding及旧XLA执行优化 | Legacy | 作为独立冻结补丁提供，不进入当前默认组合。 |

关于鲲鹏TensorFlow V2.15.0的特性详细介绍请参见《[特性介绍](./docs/zh/Tensorflow_V2.15.0/feature_introduction.md)》。

### Tensorflow V2.15.0补丁发布说明

所有补丁均基于官方TensorFlow `v2.15.0`固定提交`6887368d6d46223f460358323c4b76d61d1558a8`。

| 配置 | 包含的补丁 | 适用场景 |
| --- | --- | --- |
| `common-only` | common | 只包含构建和兼容性改动，不启用加速特性。 |
| `kdnn-core` | common + KDNN | 启用KDNN算子优化，推荐作为基础版本。 |
| `kdnn-annc` | common + KDNN + ANNC | 在KDNN基础上增加ANNC静态图融合。 |
| `full-default` | common + KDNN + ANNC + KEmbedding | 启用当前维护的全部特性，包括KEmbedding自定义算子。 |

当前维护的补丁系列需要按特性组合生成 `tensorflow/feature_copts.bzl`，请参见`patches`目录下的《[README](./patches/Tensorflow_V2.15.0/README.md)》构建源码。

Legacy补丁只允许独立应用到官方基线，不依赖公共构建集成（common），也不保证与其他补丁兼容。详细说明请参见《[README](./patches/Tensorflow_V2.15.0/README.md)》。

## Tensorflow V2.20.0

### Tensorflow V2.20.0特性介绍

| 特性 | 状态 | 简介 |
| --- | --- | --- |
| FlashAttentionGraphOptimization(FAGO)静态图融合 | 维护中 | 提供面向推荐推理场景的FlashAttentionForward算子。 |

关于鲲鹏TensorFlow V2.20.0的特性详细介绍请参见《[特性介绍](./docs/zh/Tensorflow_V2.20.0/feature_introduction.md)》。

### Tensorflow V2.20.0.补丁发布说明

所有补丁均基于官方TensorFlow `v2.20.0`固定提交`bf5899deaf70fa45173c5c7b8dc9ace8824dc980`。

| 配置 | 包含的补丁 | 适用场景 |
| --- | --- | --- |
| `common-only` | common | 只包含构建和兼容性改动，不启用加速特性。|
| `fago` | common+FAGO | 启用FlashAttention融合算子优化。 |

当前维护的补丁系列需要按特性组合生成 `tensorflow/feature_copts.bzl`，请参见`patches`目录下的《[README](./patches/Tensorflow_V2.20.0/README.md)》构建源码。

## 版本说明

关于鲲鹏TensorFlow V2.15.0的版本更新情况请参见《[版本说明书](./docs/zh/Tensorflow_V2.15.0/release_notes.md)》。

关于鲲鹏TensorFlow V2.20.0的版本更新情况请参见《[版本说明书](./docs/zh/Tensorflow_V2.20.0/release_notes.md)》。

## 学习文档

<table>
<thead align="left">
<tr id="row1291816372202">
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016">Tensorflow V2.15.0 学习资源名称</p></th>
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.3"><p id="p13918183762016">Tensorflow V2.20.0 学习资源名称</p></th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.4"><p id="p89181437152019">学习资源简介</p></th>
</tr>
</thead>
<tbody>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/zh/Tensorflow_V2.15.0/release_notes.md">版本说明书</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p2091893722011"><a href="./docs/zh/Tensorflow_V2.20.0/release_notes.md">版本说明书</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p491893752010">提供鲲鹏TensorFlow每个发布版本的基础信息和特性更新信息。</p></td>
</tr>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/zh/Tensorflow_V2.15.0/feature_introduction.md">特性介绍</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p2091893722011"><a href="./docs/zh/Tensorflow_V2.20.0/feature_introduction.md">特性介绍</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p491893752010">提供鲲鹏TensorFlow特性介绍。</p></td>
</tr>
<tr id="row939116371143">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p03913372046"><a href="./docs/zh/Tensorflow_V2.15.0/quick_start.md">快速入门</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p03913372046"><a href="./docs/zh/Tensorflow_V2.20.0/quick_start.md">快速入门</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p1139217371746">提供鲲鹏TensorFlow快速入门指导。</p></td>
</tr>
<tr id="row2918153732017">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p17918337172020"><a href="./docs/zh/Tensorflow_V2.15.0/installation_guide.md">安装指南</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p17918337172020"><a href="./docs/zh/Tensorflow_V2.20.0/installation_guide.md">安装指南</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p15918183742018">提供鲲鹏TensorFlow编译安装方法指导。</p></td>
</tr>
<tr id="row2918153732017">
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p17918337172020"><a href="./docs/zh/Tensorflow_V2.15.0/api_reference.md">API参考</a></p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.3"><p id="p17918337172020"><a href="./docs/zh/Tensorflow_V2.20.0/api_reference.md">API参考</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.4"><p id="p15918183742018">提供鲲鹏TensorFlow API使用参考。</p></td>
</tr>
</tbody>
</table>

## 免责声明

此代码仓计划参与TensorFlow社区开源，编码风格遵照开源软件，继承开源软件安全设计，不破坏开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

本项目采用Apache License 2.0许可证。详见<a href="./LICENSE">LICENSE</a>文件

本项目文档适用CC-BY 4.0许可证，具体请参见<a href="./docs/LICENSE">LICENSE</a>文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[贡献指南](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

感谢来自社区的每一个PR，欢迎贡献鲲鹏TensorFlow！
