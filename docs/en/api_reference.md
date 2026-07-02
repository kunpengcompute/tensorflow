# API Reference

## Feature Description of TensorFlow ANNC for Graph Compilation Optimization

TensorFlow ANNC for graph compilation optimization provides the following features: TensorFlow graph fusion, XLA graph fusion, operator optimization, and constant folding optimization. This document describes APIs of each feature and the steps required to enable them.

### TensorFlow Graph Fusion

[**Table 1** TensorFlow graph fusion interface](#tensorflow-graph-fusion-interface) shows how to use the TensorFlow graph fusion interface.

**Table 1** TensorFlow graph fusion interface<a id="tensorflow-graph-fusion-interface"></a>

<a name="table76971117203416"></a>
<table><tbody><tr id="row169713174343"><th class="firstcol" valign="top" width="29.73%" id="mcps1.2.3.1.1"><p id="p15790132610344"><a name="p15790132610344"></a><a name="p15790132610344"></a>Command Line Interface</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.1.1 "><p id="p1579018267345"><a name="p1579018267345"></a><a name="p1579018267345"></a>annc-opt</p>
</td>
</tr>
<tr id="row1669771715348"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.2.1"><p id="p1579042663418"><a name="p1579042663418"></a><a name="p1579042663418"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.2.1 "><p id="p1379052618344"><a name="p1379052618344"></a><a name="p1379052618344"></a>Triggers the graph fusion feature.</p>
</td>
</tr>
<tr id="row8697217173418"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.3.1"><p id="p1779017261344"><a name="p1779017261344"></a><a name="p1779017261344"></a>Parameter Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.3.1 "><a name="ul187905267346"></a><a name="ul187905267346"></a><ul id="ul187905267346"><li><code>-I /path/to/save_model.pb</code>: model before graph fusion</li><li><code>-O /path/to/new_save_model.pb</code>: model after graph fusion</li><li><code>pass</code>: graph fusion strategy (Currently, <code>lookup_embedding_hash</code> is supported.)</li></ul>
</td>
</tr>
<tr id="row63001135194219"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.4.1"><p id="p123001235104219"><a name="p123001235104219"></a><a name="p123001235104219"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.4.1 "><a name="screen12790152618347"></a><a name="screen12790152618347"></a><pre class="screen" codetype="Linux" id="screen12790152618347">annc-opt -I /base_model/wide_and_deep/1/ -O /optimized_model/wide_and_deep/1/ lookup_embedding_hash
cp -r /base_model/wide_and_deep/1/variables /optimized_model/wide_and_deep/1/</pre>
</td>
</tr>
</tbody>
</table>

### XLA Graph Fusion

[**Table 2** XLA graph fusion interface](#xla-graph-fusion-interface) describes the XLA graph fusion interface.

**Table 2** XLA graph fusion interface<a id="xla-graph-fusion-interface"></a>

<a name="table8136171014352"></a>
<table><tbody><tr id="row19137101013358"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p288371293515"><a name="p288371293515"></a><a name="p288371293515"></a>Environment Variable</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p188831212153511"><a name="p188831212153511"></a><a name="p188831212153511"></a>ANNC_FLAGS</p>
</td>
</tr>
<tr id="row213711083519"><th class="firstcol" valign="top" width="20%" id="mcps1.2.3.2.1"><p id="p16883312203510"><a name="p16883312203510"></a><a name="p16883312203510"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.2.1 "><p id="p28831312113515"><a name="p28831312113515"></a><a name="p28831312113515"></a>Compiles ANNC and enables XLA graph fusion optimization.</p>
</td>
</tr>
<tr id="row121375104358"><th class="firstcol" valign="top" width="20%" id="mcps1.2.3.3.1"><p id="p688451210355"><a name="p688451210355"></a><a name="p688451210355"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.3.1 "><a name="screen1488413125356"></a><a name="screen1488413125356"></a><pre class="screen" codetype="Linux" id="screen1488413125356">export ANNC_FLAGS="--graph-opt"</pre>
</td>
</tr>
<tr id="row31101755184211"><th class="firstcol" valign="top" width="20%" id="mcps1.2.3.4.1"><p id="p161113554427"><a name="p161113554427"></a><a name="p161113554427"></a>Value</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.4.1 "><p id="p1711155513421"><a name="p1711155513421"></a><a name="p1711155513421"></a>Enables the feature when the environment variable is <code>--graph-opt</code>.</p>
</td>
</tr>
</tbody>
</table>

### Operator Optimization

The operator optimization interfaces are described as in [**Table 3** Interface for redundant operator optimization](#interface-for-redundant-operator-optimization), [**Table 4** Interface for matrix operator optimization](#interface-for-matrix-operator-optimization), and [**Table 5** Interface for Softmax operator optimization](#interface-for-softmax-operator-optimization).

**Table 3** Interface for redundant operator optimization<a id="interface-for-redundant-operator-optimization"></a>

<a name="table7827156103620"></a>
<table><tbody><tr id="row0828268367"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p036710983611"><a name="p036710983611"></a><a name="p036710983611"></a>Environment Variable</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p336713923617"><a name="p336713923617"></a><a name="p336713923617"></a>ENABLE_BISHENG_GRAPH_OPT</p>
</td>
</tr>
<tr id="row982812643613"><th class="firstcol" valign="top" width="19.950000000000003%" id="mcps1.2.3.2.1"><p id="p436719943616"><a name="p436719943616"></a><a name="p436719943616"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="80.05%" headers="mcps1.2.3.2.1 "><p id="p9367189103617"><a name="p9367189103617"></a><a name="p9367189103617"></a>Enables redundant operator optimization.</p>
</td>
</tr>
<tr id="row882813618368"><th class="firstcol" valign="top" width="19.950000000000003%" id="mcps1.2.3.3.1"><p id="p736799173611"><a name="p736799173611"></a><a name="p736799173611"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="80.05%" headers="mcps1.2.3.3.1 "><a name="screen133671699365"></a><a name="screen133671699365"></a><pre class="screen" codetype="Linux" id="screen133671699365">export ENABLE_BISHENG_GRAPH_OPT=""</pre>
</td>
</tr>
<tr id="row21211838413"><th class="firstcol" valign="top" width="19.950000000000003%" id="mcps1.2.3.4.1"><p id="p18121931415"><a name="p18121931415"></a><a name="p18121931415"></a>Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.05%" headers="mcps1.2.3.4.1 "><p id="p1112110374117"><a name="p1112110374117"></a><a name="p1112110374117"></a>Enables the feature when the environment variable is not null.</p>
</td>
</tr>
</tbody>
</table>

**Table 4** Interface for matrix operator optimization<a id="interface-for-matrix-operator-optimization"></a>

<a name="table33723106451"></a>
<table><tbody><tr id="row10372161084512"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p952011910407"><a name="p952011910407"></a><a name="p952011910407"></a>Environment Variable</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p1852091911401"><a name="p1852091911401"></a><a name="p1852091911401"></a>ANNC_FLAGS</p>
</td>
</tr>
<tr id="row8372111014518"><th class="firstcol" valign="top" width="19.650000000000002%" id="mcps1.2.3.2.1"><p id="p152016194407"><a name="p152016194407"></a><a name="p152016194407"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="80.35%" headers="mcps1.2.3.2.1 "><p id="p17520151914012"><a name="p17520151914012"></a><a name="p17520151914012"></a>Enables matrix operator optimization.</p>
</td>
</tr>
<tr id="row12372161044513"><th class="firstcol" valign="top" width="19.650000000000002%" id="mcps1.2.3.3.1"><p id="p1520181944016"><a name="p1520181944016"></a><a name="p1520181944016"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="80.35%" headers="mcps1.2.3.3.1 "><a name="screen052011199406"></a><a name="screen052011199406"></a><pre class="screen" codetype="Linux" id="screen052011199406">export ANNC_FLAGS="--gemm-opt"</pre>
</td>
</tr>
<tr id="row173721910154515"><th class="firstcol" valign="top" width="19.650000000000002%" id="mcps1.2.3.4.1"><p id="p2154140184320"><a name="p2154140184320"></a><a name="p2154140184320"></a>Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.35%" headers="mcps1.2.3.4.1 "><p id="p6549165120430"><a name="p6549165120430"></a><a name="p6549165120430"></a>Enables the feature when the environment variable is <code>--gemm-opt</code>.</p>
</td>
</tr>
</tbody>
</table>

**Table 5** Interface for Softmax operator optimization<a id="interface-for-softmax-operator-optimization"></a>

<a name="table1173712184620"></a>
<table><tbody><tr id="row77372244611"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p103831220124013"><a name="p103831220124013"></a><a name="p103831220124013"></a>Environment Variable</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p9383820184016"><a name="p9383820184016"></a><a name="p9383820184016"></a>XLA_FLAGS</p>
</td>
</tr>
<tr id="row2073713210467"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.2.1"><p id="p133831020144016"><a name="p133831020144016"></a><a name="p133831020144016"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.2.1 "><p id="p43831320174010"><a name="p43831320174010"></a><a name="p43831320174010"></a>Enables Softmax operator optimization.</p>
</td>
</tr>
<tr id="row07373294619"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.3.1"><p id="p2038317205401"><a name="p2038317205401"></a><a name="p2038317205401"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.3.1 "><a name="screen13383182017408"></a><a name="screen13383182017408"></a><pre class="screen" codetype="Linux" id="screen13383182017408">export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"</pre>
</td>
</tr>
<tr id="row873762194620"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.4.1"><p id="p114856413449"><a name="p114856413449"></a><a name="p114856413449"></a>Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.4.1 "><p id="p748524184411"><a name="p748524184411"></a><a name="p748524184411"></a>Enables the feature when the environment variable is <code>--xla_cpu_enable_xnnpack=true</code>.</p>
</td>
</tr>
</tbody>
</table>

### Constant Folding Optimization

When both constant folding and graph optimization are enabled, constant folding must be performed first.

The Constant folding interfaces are shown in [**Table 6** Interface for model conversion using constant folding](#interface-for-model-conversion-using-constant-folding) and [**Table 7** Constant folding interface](#constant-folding-interface).

**Table 6** Interface for model conversion using constant folding<a id="interface-for-model-conversion-using-constant-folding"></a>

<a name="table769711172034"></a>
<table><tbody><tr id="row169713174343"><th class="firstcol" valign="top" width="29.73%" id="mcps1.2.3.1.1"><p id="p15790132610344"><a name="p15790132610344"></a><a name="p15790132610344"></a>Command Line Interface</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.1.1 "><p id="p1579018267345"><a name="p1579018267345"></a><a name="p1579018267345"></a>annc-opt</p>
</td>
</tr>
<tr id="row1669771715348"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.2.1"><p id="p1579042663418"><a name="p1579042663418"></a><a name="p1579042663418"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.2.1 "><p id="p1379052618344"><a name="p1379052618344"></a><a name="p1379052618344"></a>Triggers the constant folding feature.</p>
</td>
</tr>
<tr id="row8697217173418"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.3.1"><p id="p1779017261344"><a name="p1779017261344"></a><a name="p1779017261344"></a>Parameter Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.3.1 "><a name="ul187905267346"></a><a name="ul187905267346"></a><ul id="ul187905267346"><li><code>-I /path/to/save_model.pb</code>: model before constant folding</li><li><code>-O /path/to/new_save_model.pb</code>: model after constant folding</li><li><code>pass</code>: layout_matmul</li></ul>
</td>
</tr>
<tr id="row63001135194219"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.4.1"><p id="p123001235104219"><a name="p123001235104219"></a><a name="p123001235104219"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.4.1 "><a name="screen12790152618347"></a><a name="screen12790152618347"></a><pre class="screen" codetype="Linux" id="screen12790152618347">annc-opt -I /base_model/wide_and_deep/1/ -O /folding/wide_and_deep/1/ layout_matmul
</pre>
</td>
</tr>
</tbody>
</table>

**Table 7** Constant folding interface<a id="constant-folding-interface"></a>

<a name="table11737121846"></a>
<table><tbody><tr id="row77372244611"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p103831220124013"><a name="p103831220124013"></a><a name="p103831220124013"></a>Environment Variable</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p9383820184016"><a name="p9383820184016"></a><a name="p9383820184016"></a>ANNC_FLAGS</p>
</td>
</tr>
<tr id="row2073713210467"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.2.1"><p id="p133831020144016"><a name="p133831020144016"></a><a name="p133831020144016"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.2.1 "><p id="p43831320174010"><a name="p43831320174010"></a><a name="p43831320174010"></a>Enables constant folding optimization.</p>
</td>
</tr>
<tr id="row07373294619"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.3.1"><p id="p2038317205401"><a name="p2038317205401"></a><a name="p2038317205401"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.3.1 "><a name="screen13383182017408"></a><a name="screen13383182017408"></a><pre class="screen" codetype="Linux" id="screen13383182017408">export ANNC_FLAGS="--layout-matmul"</pre>
</td>
</tr>
<tr id="row873762194620"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.4.1"><p id="p114856413449"><a name="p114856413449"></a><a name="p114856413449"></a>Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.4.1 "><p id="p748524184411"><a name="p748524184411"></a><a name="p748524184411"></a>Enables the feature when the environment variable is <code>--layout-matmul</code>.</p>
</td>
</tr>
</tbody>
</table>

## Feature Description of TensorFlow Serving Thread Scheduling

### Batch Operator Scheduling

Kunpeng's TensorFlow Serving thread scheduling feature provides two configuration options: batch operator scheduling and thread affinity isolation. You can configure the options based on your specific requirements.

To use TensorFlow Serving to start an inference stress test, see section [Starting the Service and Performing a Pressure Test](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0012.html) in the _TensorFlow Serving Porting Guide_.

<a name="table4841125121312"></a>
<table><tbody><tr id="row3842145111316"><th class="firstcol" valign="top" width="26.89%" id="mcps1.1.3.1.1"><p id="p14842175119139"><a name="p14842175119139"></a><a name="p14842175119139"></a>TF Serving Command Line Interface</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.1.1 "><p id="p11842451111312"><a name="p11842451111312"></a><a name="p11842451111312"></a>--batch_op_scheduling</p>
</td>
</tr>
<tr id="row178421151111312"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.2.1"><p id="p10908168201417"><a name="p10908168201417"></a><a name="p10908168201417"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.2.1 "><p id="p2842125114131"><a name="p2842125114131"></a><a name="p2842125114131"></a>Enables the operator scheduling optimization and XLA thread pool management optimization features.</p>
</td>
</tr>
<tr id="row1484211519134"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.3.1"><p id="p18842451191311"><a name="p18842451191311"></a><a name="p18842451191311"></a>Parameter Type</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.3.1 "><p id="p12842145119131"><a name="p12842145119131"></a><a name="p12842145119131"></a>bool</p>
</td>
</tr>
<tr id="row1842205131315"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.4.1"><p id="p1984220514137"><a name="p1984220514137"></a><a name="p1984220514137"></a>Value Range</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.4.1 "><p id="p5842205111310"><a name="p5842205111310"></a><a name="p5842205111310"></a><code>true</code> or <code>false</code>. Set it to <code>true</code> to enable the feature or <code>false</code> to disable the feature.</p>
</td>
</tr>
<tr id="row5842205181319"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.5.1"><p id="p08426514136"><a name="p08426514136"></a><a name="p08426514136"></a>Recommended Scenario</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.5.1 "><p id="p1184275114135"><a name="p1184275114135"></a><a name="p1184275114135"></a>Recommended when single-core inference latency meets requirements. This option enhances concurrent processing capability and overall throughput.</p>
</td>
</tr>
<tr id="row1484295116136"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.6.1"><p id="p8842125115139"><a name="p8842125115139"></a><a name="p8842125115139"></a>Recommended Configuration</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.6.1 "><a name="ul545260191512"></a><a name="ul545260191512"></a><ul id="ul545260191512"><li><code>--tensorflow_intra_op_parallelism=1</code>: Sets the intra-operator parallelism degree to 1.</li><li><code>--tensorflow_inter_op_parallelism=80</code>: Sets the inter-operator parallelism degree to the number of CPU cores.</li><li><code>--batch_op_scheduling=true</code>: Enables the batch operator scheduling feature.</li></ul>
</td>
</tr>
<tr id="row128426514133"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.7.1"><p id="p88425512131"><a name="p88425512131"></a><a name="p88425512131"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.7.1 "><a name="screen130812517152"></a><a name="screen130812517152"></a><pre class="screen" codetype="Linux" id="screen130812517152">/path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=80 --batch_op_scheduling=true</pre>
</td>
</tr>
</tbody>
</table>

### Thread Affinity Isolation

<a name="table103198278154"></a>
<table><tbody><tr id="row231920279158"><th class="firstcol" valign="top" width="26.89%" id="mcps1.1.3.1.1"><p id="p631912272155"><a name="p631912272155"></a><a name="p631912272155"></a>TF Serving Command Line Interface</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.1.1 "><p id="p14319182719156"><a name="p14319182719156"></a><a name="p14319182719156"></a>--task_affinity_isolation</p>
</td>
</tr>
<tr id="row12319182741517"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.2.1"><p id="p331992771516"><a name="p331992771516"></a><a name="p331992771516"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.2.1 "><p id="p65711337151515"><a name="p65711337151515"></a><a name="p65711337151515"></a>Enables the thread affinity isolation feature, which offers two isolation methods:</p>
<a name="ul5571143761519"></a><a name="ul5571143761519"></a><ul id="ul5571143761519"><li>Sequential core binding allocates TensorFlow computing threads to the first K cores and TF Serving communication threads to remaining cores. </li><li>Interleaved core binding (applicable when hyper-threading is enabled) assigns TensorFlow threads to physical cores and TF Serving communication threads to virtual cores.</li></ul>
</td>
</tr>
<tr id="row1319162721514"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.3.1"><p id="p631952721519"><a name="p631952721519"></a><a name="p631952721519"></a>Parameter Type</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.3.1 "><p id="p12319927121515"><a name="p12319927121515"></a><a name="p12319927121515"></a>std::string</p>
</td>
</tr>
<tr id="row5938164721511"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.4.1"><p id="p189382479158"><a name="p189382479158"></a><a name="p189382479158"></a>Parameter Format</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.4.1 "><p id="p4938164716155"><a name="p4938164716155"></a><a name="p4938164716155"></a>mode;m-n;k. The default value is <code>0</code>.</p>
</td>
</tr>
<tr id="row1831912274154"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.5.1"><p id="p4319127111519"><a name="p4319127111519"></a><a name="p4319127111519"></a>Value Range</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.5.1 "><p id="p1231902713153"><a name="p1231902713153"></a><a name="p1231902713153"></a>For details, see <a href="#thread-affinity-isolation-parameter-values">Thread affinity isolation parameter values.</a></p>
</td>
</tr>
<tr id="row7319727101519"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.6.1"><p id="p15319127171510"><a name="p15319127171510"></a><a name="p15319127171510"></a>Recommended Scenario</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.6.1 "><a name="ul685628191620"></a><a name="ul685628191620"></a><ul id="ul685628191620"><li>When TensorFlow scheduling is used, sequential core binding is recommended.</li><li>When both batch operator scheduling and thread affinity isolation are used, and hyper-threading is enabled, interleaved core binding is recommended.</li></ul>
</td>
</tr>
<tr id="row1232011272159"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.7.1"><p id="p1432022761515"><a name="p1432022761515"></a><a name="p1432022761515"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.7.1 "><p id="p1152862014166"><a name="p1152862014166"></a><a name="p1152862014166"></a>A server has four Non-Uniform Memory Access (NUMA) nodes, each containing 40 physical cores (160 in total) or 80 logical cores (320 in total) with hyper-threading enabled.</p>
<a name="ul652816204167"></a><a name="ul652816204167"></a><ul id="ul652816204167"><li>For TensorFlow scheduling mode, use these reference parameters: <a name="screen1352813201166"></a><a name="screen1352813201166"></a><pre class="screen" codetype="Linux" id="screen1352813201166">numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=75 --tensorflow_inter_op_parallelism=75 --task_affinity_isolation="1;0-79;75"</pre>
</li></ul>
<a name="ul14528172051619"></a><a name="ul14528172051619"></a><ul id="ul14528172051619"><li>With <code>--batch_op_scheduling</code> enabled, set <code>--tensorflow_inter_op_parallelism</code> to match the physical core count, use these reference parameters: <a name="screen4528020161615"></a><a name="screen4528020161615"></a><pre class="screen" codetype="Linux" id="screen4528020161615">numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=40 --batch_op_scheduling=true --task_affinity_isolation="2;0-79"</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

**Table 1** Thread affinity isolation parameter values<a id="thread-affinity-isolation-parameter-values"></a>

<a name="table12688064377"></a>
<table><thead align="left"><tr id="row468814643718"><th class="cellrowborder" valign="top" width="7.43925607439256%" id="mcps1.2.5.1.1"><p id="p176881266377"><a name="p176881266377"></a><a name="p176881266377"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="15.448455154484552%" id="mcps1.2.5.1.2"><p id="p14656122373711"><a name="p14656122373711"></a><a name="p14656122373711"></a>Value Range</p>
</th>
<th class="cellrowborder" valign="top" width="27.90720927907209%" id="mcps1.2.5.1.3"><p id="p1768920617371"><a name="p1768920617371"></a><a name="p1768920617371"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="49.2050794920508%" id="mcps1.2.5.1.4"><p id="p3689265373"><a name="p3689265373"></a><a name="p3689265373"></a>Constraint</p>
</th>
</tr>
</thead>
<tbody><tr id="row168915612378"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p196891768371"><a name="p196891768371"></a><a name="p196891768371"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p265616232373"><a name="p265616232373"></a><a name="p265616232373"></a><code>0</code>, <code>1</code>, or <code>2</code></p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><a name="ul1088015211937"></a><a name="ul1088015211937"></a><ul id="ul1088015211937"><li><code>0</code>: (OFF) Thread affinity is disabled. </li><li><code>1</code>: (ORDER) Cores are bound in sequence. </li><li><code>2</code>: (INTERVAL) Cores are bound in an interleaved manner.</li></ul>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p46891769376"><a name="p46891769376"></a><a name="p46891769376"></a>When <code>mode</code> is set to <code>0</code>, <code>m-n</code> and <code>k</code> are invalid and can be omitted.</p>
</td>
</tr>
<tr id="row8689116183715"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p176894612375"><a name="p176894612375"></a><a name="p176894612375"></a>m-n</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p26571123153717"><a name="p26571123153717"></a><a name="p26571123153717"></a>Available CPU cores</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p768915663712"><a name="p768915663712"></a><a name="p768915663712"></a><span>The core binding range is [m, n]</span>.</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p7689660371"><a name="p7689660371"></a><a name="p7689660371"></a>m ≤ n</p>
</td>
</tr>
<tr id="row3273102020388"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p1427332019385"><a name="p1427332019385"></a><a name="p1427332019385"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p7273122013811"><a name="p7273122013811"></a><a name="p7273122013811"></a>Available CPU cores</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p727352063819"><a name="p727352063819"></a><a name="p727352063819"></a>Number of cores allocated to the TensorFlow thread.</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p6273142017388"><a name="p6273142017388"></a><a name="p6273142017388"></a>k ≤ n - m + 1 (the total number of bound cores). When <code>mode</code> is set to <code>2</code>, <code>k</code> is invalid and can be omitted.</p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **NOTE:**
>`numactl` is a tool used to control and manage the NUMA architecture on Linux. It can be installed using Yum.
>
>```bash
>yum install -y numactl numactl-devel
>```
>
>For example, `numactl -C 0-79 -m 0` indicates that the TF Serving service runs on the cores of NUMA node 0, so that CPU resources can be fully utilized. `-C` and `-m` specify cores and memory of NUMA node 0, respectively.

## Using the TensorFlow KDNN Thread Passthrough Feature

The TensorFlow KDNN thread passthrough feature is controlled by a KDNN environment variable shown in [Table 1 KDNN environment variable](#table473618378218)

**Table 1** KDNN environment variable

<a name="table473618378218"></a>
<table><tbody><tr id="row273612371926"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.1.1"><p id="p37361437223"><a name="p37361437223"></a><a name="p37361437223"></a>KDNN Environment Variable</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.1.1 "><p id="p147351257418"><a name="p147351257418"></a><a name="p147351257418"></a>TF_ENABLE_KDNN_OPTS</p>
</td>
</tr>
<tr id="row2736637329"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.2.1"><p id="p17544180133510"><a name="p17544180133510"></a><a name="p17544180133510"></a>Type</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.2.1 "><p id="p135441705354"><a name="p135441705354"></a><a name="p135441705354"></a>Process environment variable</p>
</td>
</tr>
<tr id="row1373713374215"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.3.1"><p id="p173719376217"><a name="p173719376217"></a><a name="p173719376217"></a>Function</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.3.1 "><p id="p15737237026"><a name="p15737237026"></a><a name="p15737237026"></a>Enables or disables KDNN.</p>
</td>
</tr>
<tr id="row17371371026"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.4.1"><p id="p187371371426"><a name="p187371371426"></a><a name="p187371371426"></a>Value Range</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.4.1 "><p id="p139187124325"><a name="p139187124325"></a><a name="p139187124325"></a><code>0</code>: KDNN disabled</p>
<p id="p18324912320"><a name="p18324912320"></a><a name="p18324912320"></a><code>1</code>: KDNN enabled</p>
</td>
</tr>
<tr id="row17372371123"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.5.1"><p id="p14737103710220"><a name="p14737103710220"></a><a name="p14737103710220"></a>Example</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.5.1 "><p id="p1926119477390"><a name="p1926119477390"></a><a name="p1926119477390"></a>Before the first KDNN operator call, configure <code>TF_ENABLE_KDNN_OPTS</code>, for example, <strong id="b166491125328"><a name="b166491125328"></a><a name="b166491125328"></a><code>os.environ['TF_ENABLE_KDNN_OPTS'] = str(1)</code></strong> in the python environment.</p>
</td>
</tr>
</tbody>
</table>

## Usage of the TensorFlow ANNC Static Graph Fusion Feature

The TensorFlow ANNC static graph fusion feature is enabled or disabled by environment variables. For details, see [Table 1 Environment variables for enabling or disabling ANNC static graph fusion](#table473618378218)

The default value of each environment variable is `0`, indicating that the ANNC static graph fusion function is disabled. To use this function, you need to manually set the environment variable before the graph compilation. For example, you can set the environment variable in Python as follows:

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
```

**Table 1** Environment variables for enabling or disabling ANNC static graph fusion

| Environment Variable | Type | Value | Function |
| ---- | -------- | ---- | ---- |
| ANNC_FUSED_EMB_ACTIONID_GATHER | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedEmbeddingActionIdGather operator. |
| ANNC_FUSED_GATHER | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedGather operator. |
| ANNC_FUSED_EMD_PADDING | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedEmbeddingPadding operator. |
| ANNC_FUSED_EMD_PADDING_FAST | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedEmbeddingPaddingFast operator. |
| ANNC_FUSED_SPS_STITCH | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseDynamicStitch operator. |
| ANNC_FUSED_SPS_RESHAPE | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseReshape operator. |
| ANNC_FUSED_SPS_REDUCE | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseSegmentReduce operator. |
| ANNC_FUSED_SPS_REDUCE_NONZERO | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseSegmentReduceNonzero operator. |
| ANNC_FUSED_SPS_SELECT | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for the KPFusedSparseSelect operator. |
| ANNC_FUSED_ALL | Process environment variable | **1**: enabled<br> <code>0</code>: disabled | Enables ANNC static graph fusion for all ANNC fusion operators. |

>![](public_sys-resources/icon-note.gif) **NOTE:**
>Operator fusion will not be performed for any of the above operators if and only if `ANNC_FUSED_ALL` is set to `0` and the environment variable corresponding to the specific operator is also set to `0`.

## Description

| Release Date | Change History |
| ---- | ---- |
| 2026-06-30 | This is the second official release. <ul><li>Added the description for constant folding optimization to the TensorFlow ANNC for graph compilation documentation. </li><li>Added the description for the TensorFlow ANNC static graph fusion feature. </li></ul> |
| 2026-03-30 | This is the first official release. <ul><li>Added the description for the TensorFlow KDNN thread passthrough feature.</li></ul> |
