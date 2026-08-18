# 版本说明书

## 版本配套说明

### 产品版本信息

<a name="table62675726"></a>

<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p1597721693713"><a name="p1597721693713"></a><a name="p1597721693713"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p11923034"><a name="p11923034"></a><a name="p11923034"></a><span id="text14311218114"><a name="text14311218114"></a><a name="text14311218114"></a>26.2.RC1</span></p>
</td>
</tr>
<tr id="row1930811171892"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>鲲鹏TensorFlow推理优化</span></p>
</td>
</tr>
<tr id="row19308111718"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>v1.2.0</span></p>
</td>
</tr>
</tbody>
</table>

### 操作系统、编译器和CPU配套说明

<a name="table59918346913"></a>

<table><thead align="left"><tr id="row134231235497"><th class="cellrowborder" valign="top" width="31.009999999999998%" id="mcps1.1.4.1.1"><p id="p1542383511912"><a name="p1542383511912"></a><a name="p1542383511912"></a>操作系统</p>
</th>
<th class="cellrowborder" valign="top" width="34.72%" id="mcps1.1.4.1.2"><p id="p8423193516910"><a name="p8423193516910"></a><a name="p8423193516910"></a>CPU类型</p>
</th>
<th class="cellrowborder" valign="top" width="34.27%" id="mcps1.1.4.1.3"><p id="p94235351693"><a name="p94235351693"></a><a name="p94235351693"></a>编译器</p>
</th>
</tr>
</thead>
<tbody>
<tr id="row207032381509"><td class="cellrowborder" valign="top" width="31.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p177041382017"><a name="p177041382017"></a><a name="p177041382017"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="34.72%" headers="mcps1.1.4.1.2 "><p id="p173654366528"><a name="p173654366528"></a><a name="p173654366528"></a>鲲鹏920 7282C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="34.27%" headers="mcps1.1.4.1.3 "><p id="p10125515012"><a name="p10125515012"></a><a name="p10125515012"></a>GCC 12.3.1</p>
</td>
</tr>
<tr id="row207032381509"><td class="cellrowborder" valign="top" width="31.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p177041382017"><a name="p177041382017"></a><a name="p177041382017"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="34.72%" headers="mcps1.1.4.1.2 "><p id="p173654366528"><a name="p173654366528"></a><a name="p173654366528"></a>鲲鹏950 7592C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="34.27%" headers="mcps1.1.4.1.3 "><p id="p10125515012"><a name="p10125515012"></a><a name="p10125515012"></a>GCC 12.3.1</p>
</td>
</tr>
</tbody>
</table>

### 病毒扫描结果

不涉及软件包发布，不涉及病毒扫描。

## 版本使用注意事项

### 使用注意事项

请参见《[安装指南](./installation_guide.md)》。

## v1.2.0

### 更新说明

- 新增KEmbedding算子库中的EmbeddingTableLookup算子，进一步丰富推理模型的特征处理能力。
- 新增SparseMatmul多线程优化，该优化按输出矩阵的列块进行划分，每个线程处理独立的列块。通过无锁设计、负载均衡和内存优化等技术，提升了稀疏矩阵乘法在多核CPU场景下的计算性能。

**修改特性<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

无

**删除特性<a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## v1.1.0

### 更新说明

- 新增TensorFlow ANNC静态图融合特性，适配鲲鹏950 7592C处理器，支持KPFusedEmbeddingActionIdGather等算子静态图融合。
- 新增TensorFlow ANNC常量折叠优化，适配鲲鹏950 7592C处理器。

**修改特性<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

无

**删除特性<a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## v1.0.0

### 更新说明

新增TensorFlow KDNN线程直通特性，支持batchmatmul、concat和softmax等算子对接KDNN。

**修改特性<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

无

**删除特性<a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## 25.2.RC1

### 更新说明

**新增特性<a name="zh-cn_topic_0000002518399190_section11862975"></a>**

<a name="zh-cn_topic_0000002518399190_table41916133"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000002518399190_row28804032"><th class="cellrowborder" valign="top" width="23.419999999999998%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002518399190_p4697041"><a name="zh-cn_topic_0000002518399190_p4697041"></a><a name="zh-cn_topic_0000002518399190_p4697041"></a>算法名称</p>
</th>
<th class="cellrowborder" valign="top" width="76.58%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002518399190_p44916036"><a name="zh-cn_topic_0000002518399190_p44916036"></a><a name="zh-cn_topic_0000002518399190_p44916036"></a>更新说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002518399190_row15189941135213"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002518399190_p11189174116528"><a name="zh-cn_topic_0000002518399190_p11189174116528"></a><a name="zh-cn_topic_0000002518399190_p11189174116528"></a>TensorFlow图融合</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002518399190_p1218964125212"><a name="zh-cn_topic_0000002518399190_p1218964125212"></a><a name="zh-cn_topic_0000002518399190_p1218964125212"></a>新增TensorFlow模型层面的图融合与图重写功能。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002518399190_row84721825541"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002518399190_p14945115594019"><a name="zh-cn_topic_0000002518399190_p14945115594019"></a><a name="zh-cn_topic_0000002518399190_p14945115594019"></a>XLA算子融合</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002518399190_p515310015517"><a name="zh-cn_topic_0000002518399190_p515310015517"></a><a name="zh-cn_topic_0000002518399190_p515310015517"></a>新增ANNC XLA图融合特性。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002518399190_row01931052122211"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002518399190_p103321455202715"><a name="zh-cn_topic_0000002518399190_p103321455202715"></a><a name="zh-cn_topic_0000002518399190_p103321455202715"></a>算子优化</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002518399190_p1519435212217"><a name="zh-cn_topic_0000002518399190_p1519435212217"></a><a name="zh-cn_topic_0000002518399190_p1519435212217"></a>新增ANNC算子优化特性。</p>
</td>
</tr>
</tbody>
</table>

**修改特性<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

无

**删除特性<a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## 25.1.RC1

### 更新说明

**新增特性<a name="section11862975"></a>**

<a name="table41916133"></a>

<table><thead align="left"><tr id="row28804032"><th class="cellrowborder" valign="top" width="23.419999999999998%" id="mcps1.1.3.1.1"><p id="p4697041"><a name="p4697041"></a><a name="p4697041"></a>算法名称</p>
</th>
<th class="cellrowborder" valign="top" width="76.58%" id="mcps1.1.3.1.2"><p id="p44916036"><a name="p44916036"></a><a name="p44916036"></a>更新说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row84721825541"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="p1682142025412"><a name="p1682142025412"></a><a name="p1682142025412"></a>算子批量调度</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="p515310015517"><a name="p515310015517"></a><a name="p515310015517"></a>新增算子批量调度和XLA线程池管理优化特性。</p>
</td>
</tr>
<tr id="row01931052122211"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="p14194952182212"><a name="p14194952182212"></a><a name="p14194952182212"></a>线程亲和性隔离</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="p1519435212217"><a name="p1519435212217"></a><a name="p1519435212217"></a>新增线程亲和性隔离特性。</p>
</td>
</tr>
</tbody>
</table>

**修改特性<a name="section16450949161512"></a>**

无

**删除特性<a name="section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## 版本配套文档

### v1.2.0版本配套文档

<a name="table41916133"></a>

<table><thead align="left"><tr id="row28804032"><th class="cellrowborder" valign="top" width="45.019999999999996%" id="mcps1.1.4.1.1"><p id="p4697041"><a name="p4697041"></a><a name="p4697041"></a>文档名称</p>
</th>
<th class="cellrowborder" valign="top" width="38.019999999999996%" id="mcps1.1.4.1.2"><p id="p44916036"><a name="p44916036"></a><a name="p44916036"></a>内容简介</p>
</th>
<th class="cellrowborder" valign="top" width="16.96%" id="mcps1.1.4.1.3"><p id="p14320308"><a name="p14320308"></a><a name="p14320308"></a>交付形式</p>
</th>
</tr>
</thead>
<tbody><tr id="row19094280">
<td class="cellrowborder" valign="top" width="40.510000000000005%" headers="mcps1.1.4.1.1"><p id="p1341193722116">《版本说明书》</p></td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p2042183752117"><a name="p2042183752117"></a>本文档提供<span id="text133997593113"><a name="text133997593113"></a><a name="text133997593113"></a>鲲鹏TensorFlow</span>的版本发布信息。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p12419194564814"><a name="p12419194564814"></a><a name="p12419194564814"></a>开源仓</p>
</td>
</tr>
<tr id="row1941037152117">
<td class="cellrowborder" valign="top" width="40.510000000000005%" headers="mcps1.1.4.1.1">《特性介绍》</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1914345202019"><a name="p1914345202019"></a><a name="p1914345202019"></a>本文档提供<span id="text1176394151219"><a name="text1176394151219"></a><a name="text1176394151219"></a>鲲鹏TensorFlow优化</span>特性使用指导。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p164410548486"><a name="p164410548486"></a><a name="p164410548486"></a>开源仓</p>
</td>
</tr>
<tr id="row17314122818119"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p9534164291118"><a name="p9534164291118"></a><a name="p9534164291118"></a>《安装指南》</p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1353414214111"><a name="p1353414214111"></a><a name="p1353414214111"></a>本文档提供鲲鹏TensorFlow的安装使用指导。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p1164511541488"><a name="p1164511541488"></a><a name="p1164511541488"></a>开源仓</p>
</td>
</tr>
<tr id="row1021013311110"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p18535154291116"><a name="p18535154291116"></a><a name="p18535154291116"></a>《快速入门》</p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1353554215117"><a name="p1353554215117"></a><a name="p1353554215117"></a>本文档提供鲲鹏TensorFlow推理优化特性快速入门指导。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p2646175434816"><a name="p2646175434816"></a><a name="p2646175434816"></a>开源仓</p>
</td>
</tr>

<tr id="row1021013311110"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p18535154291116"><a name="p18535154291116"></a><a name="p18535154291116"></a>《API参考》</p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1353554215117"><a name="p1353554215117"></a><a name="p1353554215117"></a>本文档提供鲲鹏TensorFlow推理优化特性API使用说明。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p2646175434816"><a name="p2646175434816"></a><a name="p2646175434816"></a>开源仓</p>
</td>
</tr>
</tbody>
</table>

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/tensorflow)浏览和获取相关文档。
