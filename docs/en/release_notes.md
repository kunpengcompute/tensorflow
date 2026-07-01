# Release Notes

## Version Mapping

### Product Information

<a name="table62675726"></a>

<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p1597721693713"><a name="p1597721693713"></a><a name="p1597721693713"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p11923034"><a name="p11923034"></a><a name="p11923034"></a><span id="text14311218114"><a name="text14311218114"></a><a name="text14311218114"></a>26.1.RC1</span></p>
</td>
</tr>
<tr id="row1930811171892"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>Software Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>Kunpeng TensorFlow Inference Optimization</span></p>
</td>
</tr>
<tr id="row19308111718"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>Software Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>v1.1.0</span></p>
</td>
</tr>
</tbody>
</table>

### OS, Compiler, and CPU

<a name="table59918346913"></a>

<table><thead align="left"><tr id="row134231235497"><th class="cellrowborder" valign="top" width="31.009999999999998%" id="mcps1.1.4.1.1"><p id="p1542383511912"><a name="p1542383511912"></a><a name="p1542383511912"></a>OS</p>
</th>
<th class="cellrowborder" valign="top" width="34.72%" id="mcps1.1.4.1.2"><p id="p8423193516910"><a name="p8423193516910"></a><a name="p8423193516910"></a>CPU</p>
</th>
<th class="cellrowborder" valign="top" width="34.27%" id="mcps1.1.4.1.3"><p id="p94235351693"><a name="p94235351693"></a><a name="p94235351693"></a>Compiler</p>
</th>
</tr>
</thead>
<tbody>
<tr id="row207032381509"><td class="cellrowborder" valign="top" width="31.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p177041382017"><a name="p177041382017"></a><a name="p177041382017"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="34.72%" headers="mcps1.1.4.1.2 "><p id="p173654366528"><a name="p173654366528"></a><a name="p173654366528"></a>New Kunpeng 920 processor model</p>
</td>
<td class="cellrowborder" valign="top" width="34.27%" headers="mcps1.1.4.1.3 "><p id="p10125515012"><a name="p10125515012"></a><a name="p10125515012"></a>GCC 12.3.1</p>
</td>
</tr>
<tr id="row207032381509"><td class="cellrowborder" valign="top" width="31.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p177041382017"><a name="p177041382017"></a><a name="p177041382017"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="34.72%" headers="mcps1.1.4.1.2 "><p id="p173654366528"><a name="p173654366528"></a><a name="p173654366528"></a>Kunpeng 950 processor</p>
</td>
<td class="cellrowborder" valign="top" width="34.27%" headers="mcps1.1.4.1.3 "><p id="p10125515012"><a name="p10125515012"></a><a name="p10125515012"></a>GCC 12.3.1</p>
</td>
</tr>
</tbody>
</table>

### Virus Scan Results

Software package release and the following virus scanning are not involved.

## Important Notes

### Precautions

For details, see [Installation Guide](./installation_guide.md).

## v1.1.0

### Change Description

- Added the TensorFlow ANNC static graph fusion feature, adapted to Kunpeng 950 processor and supporting static graph fusion of operators such as KPFusedEmbeddingActionIdGather.
- Added the TensorFlow ANNC constant folding optimization, adapted to Kunpeng 950 processor.

**Modified Features<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

None

**Removed Features <a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

None

### Resolved Issues

None

### Known Issues

None

## v1.0.0

### Change Description

Added the TensorFlow KDNN thread passthrough feature, supporting operators such as batchmatmul, concat, and softmax.

**Modified Features<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

None

**Removed Features <a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

None

### Resolved Issues

None

### Known Issues

None

## 25.2.RC1

### Change Description

**New Features <a name="zh-cn_topic_0000002518399190_section11862975"></a>**

<a name="zh-cn_topic_0000002518399190_table41916133"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000002518399190_row28804032"><th class="cellrowborder" valign="top" width="23.419999999999998%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002518399190_p4697041"><a name="zh-cn_topic_0000002518399190_p4697041"></a><a name="zh-cn_topic_0000002518399190_p4697041"></a>Algorithm</p>
</th>
<th class="cellrowborder" valign="top" width="76.58%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002518399190_p44916036"><a name="zh-cn_topic_0000002518399190_p44916036"></a><a name="zh-cn_topic_0000002518399190_p44916036"></a>Update Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002518399190_row15189941135213"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002518399190_p11189174116528"><a name="zh-cn_topic_0000002518399190_p11189174116528"></a><a name="zh-cn_topic_0000002518399190_p11189174116528"></a>TensorFlow graph fusion</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002518399190_p1218964125212"><a name="zh-cn_topic_0000002518399190_p1218964125212"></a><a name="zh-cn_topic_0000002518399190_p1218964125212"></a>Added the graph fusion and graph rewriting functions at the TensorFlow model level.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002518399190_row84721825541"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002518399190_p14945115594019"><a name="zh-cn_topic_0000002518399190_p14945115594019"></a><a name="zh-cn_topic_0000002518399190_p14945115594019"></a>XLA operator fusion</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002518399190_p515310015517"><a name="zh-cn_topic_0000002518399190_p515310015517"></a><a name="zh-cn_topic_0000002518399190_p515310015517"></a>Added the ANNC XLA graph fusion feature.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002518399190_row01931052122211"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002518399190_p103321455202715"><a name="zh-cn_topic_0000002518399190_p103321455202715"></a><a name="zh-cn_topic_0000002518399190_p103321455202715"></a>Operator optimization</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002518399190_p1519435212217"><a name="zh-cn_topic_0000002518399190_p1519435212217"></a><a name="zh-cn_topic_0000002518399190_p1519435212217"></a>Added the ANNC operator optimization feature.</p>
</td>
</tr>
</tbody>
</table>

**Modified Features<a name="zh-cn_topic_0000002518399190_section16450949161512"></a>**

None

**Removed Features <a name="zh-cn_topic_0000002518399190_section9218125814159"></a>**

None

### Resolved Issues

None

### Known Issues

None

## 25.1.RC1

### Change Description

**New Features<a name="section11862975"></a>**

<a name="table41916133"></a>

<table><thead align="left"><tr id="row28804032"><th class="cellrowborder" valign="top" width="23.419999999999998%" id="mcps1.1.3.1.1"><p id="p4697041"><a name="p4697041"></a><a name="p4697041"></a>Algorithm Name</p>
</th>
<th class="cellrowborder" valign="top" width="76.58%" id="mcps1.1.3.1.2"><p id="p44916036"><a name="p44916036"></a><a name="p44916036"></a>Update Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row84721825541"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="p1682142025412"><a name="p1682142025412"></a><a name="p1682142025412"></a>Batch operator scheduling</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="p515310015517"><a name="p515310015517"></a><a name="p515310015517"></a>Added the operator scheduling optimization and XLA thread pool management optimization features.</p>
</td>
</tr>
<tr id="row01931052122211"><td class="cellrowborder" valign="top" width="23.419999999999998%" headers="mcps1.1.3.1.1 "><p id="p14194952182212"><a name="p14194952182212"></a><a name="p14194952182212"></a>Thread affinity isolation</p>
</td>
<td class="cellrowborder" valign="top" width="76.58%" headers="mcps1.1.3.1.2 "><p id="p1519435212217"><a name="p1519435212217"></a><a name="p1519435212217"></a>Added the thread affinity isolation feature.</p>
</td>
</tr>
</tbody>
</table>

**Modified Features<a name="section16450949161512"></a>**

None

**Removed Features<a name="section9218125814159"></a>**

None

### Resolved Issues

None

### Known Issues

None

## Related Documentation

### v1.1.0 Related Documentation

<a name="table41916133"></a>

<table><thead align="left"><tr id="row28804032"><th class="cellrowborder" valign="top" width="45.019999999999996%" id="mcps1.1.4.1.1"><p id="p4697041"><a name="p4697041"></a><a name="p4697041"></a>Document Name</p>
</th>
<th class="cellrowborder" valign="top" width="38.019999999999996%" id="mcps1.1.4.1.2"><p id="p44916036"><a name="p44916036"></a><a name="p44916036"></a>Overview</p>
</th>
<th class="cellrowborder" valign="top" width="16.96%" id="mcps1.1.4.1.3"><p id="p14320308"><a name="p14320308"></a><a name="p14320308"></a>Delivery Method</p>
</th>
</tr>
</thead>
<tbody><tr id="row19094280">
<td class="cellrowborder" valign="top" width="40.510000000000005%" headers="mcps1.1.4.1.1"><p id="p1341193722116">Release Notes</p></td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p2042183752117"><a name="p2042183752117"></a>Provides the version release information about <span id="text133997593113"><a name="text133997593113"></a><a name="text133997593113"></a> Kunpeng TensorFlow</span>.</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p12419194564814"><a name="p12419194564814"></a><a name="p12419194564814"></a>Open-source repository</p>
</td>
</tr>
<tr id="row1941037152117">
<td class="cellrowborder" valign="top" width="40.510000000000005%" headers="mcps1.1.4.1.1">Feature Introduction</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1914345202019"><a name="p1914345202019"></a><a name="p1914345202019"></a>Describes how to use Kunpeng TensorFlow optimization features.</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p164410548486"><a name="p164410548486"></a><a name="p164410548486"></a>Open-source repository</p>
</td>
</tr>
<tr id="row17314122818119"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p9534164291118"><a name="p9534164291118"></a><a name="p9534164291118"></a>Installation Guide</p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1353414214111"><a name="p1353414214111"></a><a name="p1353414214111"></a>Describes how to install and use Kunpeng TensorFlow.</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p1164511541488"><a name="p1164511541488"></a><a name="p1164511541488"></a>Open-source repository</p>
</td>
</tr>
<tr id="row1021013311110"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p18535154291116"><a name="p18535154291116"></a><a name="p18535154291116"></a>Quick Start</p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1353554215117"><a name="p1353554215117"></a><a name="p1353554215117"></a>Provides a quick start guide to the Kunpeng TensorFlow inference optimization features.</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p2646175434816"><a name="p2646175434816"></a><a name="p2646175434816"></a>Open-source repository</p>
</td>
</tr>

<tr id="row1021013311110"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p18535154291116"><a name="p18535154291116"></a><a name="p18535154291116"></a>API Reference</p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1353554215117"><a name="p1353554215117"></a><a name="p1353554215117"></a>Describes how to use the APIs of the Kunpeng TensorFlow inference optimization features.</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p2646175434816"><a name="p2646175434816"></a><a name="p2646175434816"></a>Open-source repository</p>
</td>
</tr>
</tbody>
</table>

### Obtaining Documentation

Visit the [open-source repository](https://gitcode.com/boostkit/tensorflow) to view or download related documents.
