# 项目介绍<a name="ZH-CN_TOPIC_0000002442690778"></a>

鲲鹏Tensorflow是基于开源Tensorflow的高性能推理加速扩展，聚焦于搜推广推理场景下的高效执行。通过在图优化、算子、Runtime等方面进行了深度的性能增强，显著提升了模型推理的吞吐量和时延表现，为AI应用提供基于鲲鹏CPU极致性能。

主要特性包括：

-   图优化：识别Embedding常见子图进行融合，减少图执行开销；
-   算子优化：针对核心算子提供鲲鹏亲和实现，结合鲲鹏硬件底层指令集提升计算效率；
-   运行时优化：引入轻量化调度器，提升并发执行效率。

**图 1**  项目架构<a name="fig47161150181513"></a>  
![](./docs/guide/images/architecture.png "项目架构")

-   Executor层：运行时优化。
-   Kernel层：自定义算子，基于KDNN提供鲲鹏高性能DNN算子。
-   XLA层：基于ANNC提供鲲鹏图编译器。

# 版本说明<a name="ZH-CN_TOPIC_0000002442853110"></a>

<a name="table2197946132516"></a>
<table><thead align="left"><tr id="row42091246172511"><th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.1"><p id="p12095461252"><a name="p12095461252"></a><a name="p12095461252"></a>Kunpeng Tensorflow</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.2"><p id="p14209646192510"><a name="p14209646192510"></a><a name="p14209646192510"></a>Stock TensorFlow</p>
</th>
</tr>
</thead>
<tbody><tr id="row62095469253"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p162092046182518"><a name="p162092046182518"></a><a name="p162092046182518"></a>v2.15.0</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p17209114618255"><a name="p17209114618255"></a><a name="p17209114618255"></a>2.15</p>
</td>
</tr>
</tbody>
</table>

# 环境部署<a name="ZH-CN_TOPIC_0000002476133021"></a>

**硬件要求<a name="section19905171013614"></a>**

鲲鹏TensorFlow扩展提供对鲲鹏920系列处理器的支持。

**软件要求<a name="section2088201912369"></a>**

操作系统：openEuler 22.03 LTS SP3、Kernel=5.10.0

软件：

-   GCC/G++：10.3.1
-   Bazel：6.5.0
-   Python 3.9
-   鲲鹏AI编译器：ANNC

**安装方式<a name="section49733392819"></a>**

1.  拉取代码

    ```
    git clone -b {tensorflow_tag} https://gitcode.com/boostkit/tensorflow.git 
    ```

    其中，tensorflow\_tag需要替换为release版本的tag点。

1.  编译

    -   编译pip包

    ```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 
    ```

    -   编译libtensorflow\_cc.so

    ```
    bazel build --config=opt //tensorflow/libtensorflow_cc.so 
    ```

    如果在编译过程中遇到任何问题，可参考以下文档：

    [TensorFlow 移植指南](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0001.html)

    [TensorFlow 安装](https://tensorflow.google.cn/install/source?hl=zh-cn)

# 快速上手<a name="ZH-CN_TOPIC_0000002442693246"></a>

**线程调度优化特性<a name="section2696329173210"></a>**

为提升TensorFlow Serving（以下简称TF Serving）推理性能，鲲鹏BoostKit提出了TensorFlow Serving线程调度优化方案。传统TensorFlow使用算子间的线程池并行计算不同的算子，虽可实现没有数据依赖的算子的并发执行，但在高并发场景下，多Session共享算子间线程池会导致任务抢占，严重降低整图计算效率。针对这一痛点，鲲鹏BoostKit改进了算子调度算法，并加入了其他线程管理优化，有效提升了高并发场景下的模型推理吞吐量。

快速上手：[TensorFlow Serving线程调度优化](https://www.hikunpeng.com/document/detail/zh/SRA/accelFeatures/TFTSO/kunpengsra_tfserving_20_0002.html)

**ANNC特性<a name="section147391344338"></a>**

鲲鹏BoostKit提出了Tensorflow Serving ANNC优化方案。ANNC是专注于加速神经网络计算的编译器，聚焦于通过计算图优化，高性能融合算子生成和对接技术，高效代码生成和优化能力，加速推荐的推理性能。ANNC作为基于开源OpenXLA的扩展加速套件，发布在openEuler组织的ANNC开源仓，具有鲲鹏亲和的优化特性，包括Tensorflow图融合、XLA图融合、算子优化能力。

快速上手：[Tensorflow Serving ANNC](https://gitcode.com/boostkit/tensorflow/tree/master/%E5%BE%85%E5%8F%91%E5%B8%83)

# 贡献指南<a name="ZH-CN_TOPIC_0000002476053213"></a>

如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交isssues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。

# 免责声明<a name="ZH-CN_TOPIC_0000002442853118"></a>

此代码仓计划参与Tensorflow社区开源，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

# 许可证书<a name="ZH-CN_TOPIC_0000002476133025"></a>

[Apache License 2.0](https://gitcode.com/boostkit/tensorflow/blob/master/LICENSE)  通过下载并使用此源码及其附带的软件，您即同意遵守软件许可协议中的条款和条件。

