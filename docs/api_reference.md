# API参考

## TensorFlow ANNC图编译优化特性使用说明

TensorFlow ANNC图编译优化特性提供了TensorFlow图融合、XLA图融合、算子优化三种优化特性，本章节提供各特性的使能步骤。

**TensorFlow图融合<a name="section788964518814"></a>**

TensorFlow图融合接口使用如[**表 1** TensorFlow图融合接口](#TensorFlow图融合接口)所示。

**表 1** TensorFlow图融合接口<a id="TensorFlow图融合接口"></a>

<a name="table76971117203416"></a>
<table><tbody><tr id="row169713174343"><th class="firstcol" valign="top" width="29.73%" id="mcps1.2.3.1.1"><p id="p15790132610344"><a name="p15790132610344"></a><a name="p15790132610344"></a>终端命令行接口</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.1.1 "><p id="p1579018267345"><a name="p1579018267345"></a><a name="p1579018267345"></a>annc-opt</p>
</td>
</tr>
<tr id="row1669771715348"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.2.1"><p id="p1579042663418"><a name="p1579042663418"></a><a name="p1579042663418"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.2.1 "><p id="p1379052618344"><a name="p1379052618344"></a><a name="p1379052618344"></a>图融合启动命令。</p>
</td>
</tr>
<tr id="row8697217173418"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.3.1"><p id="p1779017261344"><a name="p1779017261344"></a><a name="p1779017261344"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.3.1 "><a name="ul187905267346"></a><a name="ul187905267346"></a><ul id="ul187905267346"><li>-I /path/to/save_model.pb：待图融合的模型</li><li>-O /path/to/new_save_model.pb：图融合之后的模型</li><li>pass：图融合策略（当前支持lookup_embedding_hash）</li></ul>
</td>
</tr>
<tr id="row63001135194219"><th class="firstcol" valign="top" width="19.73%" id="mcps1.2.3.4.1"><p id="p123001235104219"><a name="p123001235104219"></a><a name="p123001235104219"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.2.3.4.1 "><a name="screen12790152618347"></a><a name="screen12790152618347"></a><pre class="screen" codetype="Linux" id="screen12790152618347">annc-opt -I /base_model/wide_and_deep/1/ -O /optimized_model/wide_and_deep/1/ lookup_embedding_hash
cp -r /base_model/wide_and_deep/1/variables /optimized_model/wide_and_deep/1/</pre>
</td>
</tr>
</tbody>
</table>

**XLA图融合<a name="section14991842122818"></a>**

XLA图融合接口如[**表 2** XLA图融合接口](#XLA图融合接口)所示。

**表 2** XLA图融合接口<a id="XLA图融合接口"></a>

<a name="table8136171014352"></a>
<table><tbody><tr id="row19137101013358"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p288371293515"><a name="p288371293515"></a><a name="p288371293515"></a>环境变量</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p188831212153511"><a name="p188831212153511"></a><a name="p188831212153511"></a>ANNC_FLAGS</p>
</td>
</tr>
<tr id="row213711083519"><th class="firstcol" valign="top" width="20%" id="mcps1.2.3.2.1"><p id="p16883312203510"><a name="p16883312203510"></a><a name="p16883312203510"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.2.1 "><p id="p28831312113515"><a name="p28831312113515"></a><a name="p28831312113515"></a>编译ANNC，使能XLA图融合优化。</p>
</td>
</tr>
<tr id="row121375104358"><th class="firstcol" valign="top" width="20%" id="mcps1.2.3.3.1"><p id="p688451210355"><a name="p688451210355"></a><a name="p688451210355"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.3.1 "><a name="screen1488413125356"></a><a name="screen1488413125356"></a><pre class="screen" codetype="Linux" id="screen1488413125356">export ANNC_FLAGS="--graph-opt"</pre>
</td>
</tr>
<tr id="row31101755184211"><th class="firstcol" valign="top" width="20%" id="mcps1.2.3.4.1"><p id="p161113554427"><a name="p161113554427"></a><a name="p161113554427"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.4.1 "><p id="p1711155513421"><a name="p1711155513421"></a><a name="p1711155513421"></a>环境变量为<span class="parmvalue" id="parmvalue871883818189"><a name="parmvalue871883818189"></a><a name="parmvalue871883818189"></a>“--graph-opt”</span>开启特性。</p>
</td>
</tr>
</tbody>
</table>

**算子优化<a name="section56815743116"></a>**

算子优化接口如[**表 3** 冗余算子优化接口](#冗余算子优化接口)，[**表 4** 矩阵算子优化接口](#矩阵算子优化接口)，[**表 5** Softmax算子优化接口](#Softmax算子优化接口)所示。

**表 3** 冗余算子优化接口<a id="冗余算子优化接口"></a>

<a name="table7827156103620"></a>
<table><tbody><tr id="row0828268367"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p036710983611"><a name="p036710983611"></a><a name="p036710983611"></a>环境变量</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p336713923617"><a name="p336713923617"></a><a name="p336713923617"></a>ENABLE_BISHENG_GRAPH_OPT</p>
</td>
</tr>
<tr id="row982812643613"><th class="firstcol" valign="top" width="19.950000000000003%" id="mcps1.2.3.2.1"><p id="p436719943616"><a name="p436719943616"></a><a name="p436719943616"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="80.05%" headers="mcps1.2.3.2.1 "><p id="p9367189103617"><a name="p9367189103617"></a><a name="p9367189103617"></a>使能冗余算子优化。</p>
</td>
</tr>
<tr id="row882813618368"><th class="firstcol" valign="top" width="19.950000000000003%" id="mcps1.2.3.3.1"><p id="p736799173611"><a name="p736799173611"></a><a name="p736799173611"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="80.05%" headers="mcps1.2.3.3.1 "><a name="screen133671699365"></a><a name="screen133671699365"></a><pre class="screen" codetype="Linux" id="screen133671699365">export ENABLE_BISHENG_GRAPH_OPT=""</pre>
</td>
</tr>
<tr id="row21211838413"><th class="firstcol" valign="top" width="19.950000000000003%" id="mcps1.2.3.4.1"><p id="p18121931415"><a name="p18121931415"></a><a name="p18121931415"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="80.05%" headers="mcps1.2.3.4.1 "><p id="p1112110374117"><a name="p1112110374117"></a><a name="p1112110374117"></a>环境变量非空时开启特性。</p>
</td>
</tr>
</tbody>
</table>

**表 4** 矩阵算子优化接口<a id="矩阵算子优化接口"></a>

<a name="table33723106451"></a>
<table><tbody><tr id="row10372161084512"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p952011910407"><a name="p952011910407"></a><a name="p952011910407"></a>环境变量</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p1852091911401"><a name="p1852091911401"></a><a name="p1852091911401"></a>ANNC_FLAGS</p>
</td>
</tr>
<tr id="row8372111014518"><th class="firstcol" valign="top" width="19.650000000000002%" id="mcps1.2.3.2.1"><p id="p152016194407"><a name="p152016194407"></a><a name="p152016194407"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="80.35%" headers="mcps1.2.3.2.1 "><p id="p17520151914012"><a name="p17520151914012"></a><a name="p17520151914012"></a>使能矩阵算子优化。</p>
</td>
</tr>
<tr id="row12372161044513"><th class="firstcol" valign="top" width="19.650000000000002%" id="mcps1.2.3.3.1"><p id="p1520181944016"><a name="p1520181944016"></a><a name="p1520181944016"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="80.35%" headers="mcps1.2.3.3.1 "><a name="screen052011199406"></a><a name="screen052011199406"></a><pre class="screen" codetype="Linux" id="screen052011199406">export ANNC_FLAGS="--gemm-opt"</pre>
</td>
</tr>
<tr id="row173721910154515"><th class="firstcol" valign="top" width="19.650000000000002%" id="mcps1.2.3.4.1"><p id="p2154140184320"><a name="p2154140184320"></a><a name="p2154140184320"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="80.35%" headers="mcps1.2.3.4.1 "><p id="p6549165120430"><a name="p6549165120430"></a><a name="p6549165120430"></a>环境变量为<span class="parmvalue" id="parmvalue54155320196"><a name="parmvalue54155320196"></a><a name="parmvalue54155320196"></a>“--gemm-opt”</span>开启特性。</p>
</td>
</tr>
</tbody>
</table>

**表 5** Softmax算子优化接口<a id="Softmax算子优化接口"></a>

<a name="table1173712184620"></a>
<table><tbody><tr id="row77372244611"><th class="firstcol" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p103831220124013"><a name="p103831220124013"></a><a name="p103831220124013"></a>环境变量</p>
</th>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.1 "><p id="p9383820184016"><a name="p9383820184016"></a><a name="p9383820184016"></a>XLA_FLAGS</p>
</td>
</tr>
<tr id="row2073713210467"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.2.1"><p id="p133831020144016"><a name="p133831020144016"></a><a name="p133831020144016"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.2.1 "><p id="p43831320174010"><a name="p43831320174010"></a><a name="p43831320174010"></a>使能Softmax算子优化。</p>
</td>
</tr>
<tr id="row07373294619"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.3.1"><p id="p2038317205401"><a name="p2038317205401"></a><a name="p2038317205401"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.3.1 "><a name="screen13383182017408"></a><a name="screen13383182017408"></a><pre class="screen" codetype="Linux" id="screen13383182017408">export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"</pre>
</td>
</tr>
<tr id="row873762194620"><th class="firstcol" valign="top" width="20.06%" id="mcps1.2.3.4.1"><p id="p114856413449"><a name="p114856413449"></a><a name="p114856413449"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="79.94%" headers="mcps1.2.3.4.1 "><p id="p748524184411"><a name="p748524184411"></a><a name="p748524184411"></a>环境变量为<span class="parmvalue" id="parmvalue147231513181917"><a name="parmvalue147231513181917"></a><a name="parmvalue147231513181917"></a>“--xla_cpu_enable_xnnpack=true”</span>开启特性。</p>
</td>
</tr>
</tbody>
</table>

## TensorFlow Serving线程调度特性使用说明

**算子批量调度<a name="section788964518814"></a>**
鲲鹏TensorFlow Serving线程调度优化通过命令行提供了算子批量调度和线程亲和性隔离两个特性开关，用户可根据实际场景自行配置。

使用TF Serving启动推理压测指导请参见《TensorFlow Serving推理部署框架 移植指南》的“[启动服务并压测](https://www.hikunpeng.com/document/detail/zh/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_02_0012.html)”章节。

<a name="table4841125121312"></a>
<table><tbody><tr id="row3842145111316"><th class="firstcol" valign="top" width="26.89%" id="mcps1.1.3.1.1"><p id="p14842175119139"><a name="p14842175119139"></a><a name="p14842175119139"></a>TF Serving命令行接口</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.1.1 "><p id="p11842451111312"><a name="p11842451111312"></a><a name="p11842451111312"></a>--batch_op_scheduling</p>
</td>
</tr>
<tr id="row178421151111312"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.2.1"><p id="p10908168201417"><a name="p10908168201417"></a><a name="p10908168201417"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.2.1 "><p id="p2842125114131"><a name="p2842125114131"></a><a name="p2842125114131"></a>使能算子调度优化和XLA线程池管理优化特性。</p>
</td>
</tr>
<tr id="row1484211519134"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.3.1"><p id="p18842451191311"><a name="p18842451191311"></a><a name="p18842451191311"></a>参数类型</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.3.1 "><p id="p12842145119131"><a name="p12842145119131"></a><a name="p12842145119131"></a>bool</p>
</td>
</tr>
<tr id="row1842205131315"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.4.1"><p id="p1984220514137"><a name="p1984220514137"></a><a name="p1984220514137"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.4.1 "><p id="p5842205111310"><a name="p5842205111310"></a><a name="p5842205111310"></a>true/false。true为真，表示开启特性，默认为false。</p>
</td>
</tr>
<tr id="row5842205181319"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.5.1"><p id="p08426514136"><a name="p08426514136"></a><a name="p08426514136"></a>推荐场景</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.5.1 "><p id="p1184275114135"><a name="p1184275114135"></a><a name="p1184275114135"></a>单核推理时延可满足业务要求，可配置该选项提升推理并发能力和吞吐量。</p>
</td>
</tr>
<tr id="row1484295116136"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.6.1"><p id="p8842125115139"><a name="p8842125115139"></a><a name="p8842125115139"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.6.1 "><a name="ul545260191512"></a><a name="ul545260191512"></a><ul id="ul545260191512"><li>--tensorflow_intra_op_parallelism=1，算子内并行度设置为1；</li><li>--tensorflow_inter_op_parallelism=80，算子间并行度设置为CPU核数；</li><li>--batch_op_scheduling=true，开启算子批量调度特性。</li></ul>
</td>
</tr>
<tr id="row128426514133"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.7.1"><p id="p88425512131"><a name="p88425512131"></a><a name="p88425512131"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.7.1 "><a name="screen130812517152"></a><a name="screen130812517152"></a><pre class="screen" codetype="Linux" id="screen130812517152">/path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=80 --batch_op_scheduling=true</pre>
</td>
</tr>
</tbody>
</table>

**线程亲和性隔离<a name="section106451202096"></a>**

<a name="table103198278154"></a>
<table><tbody><tr id="row231920279158"><th class="firstcol" valign="top" width="26.89%" id="mcps1.1.3.1.1"><p id="p631912272155"><a name="p631912272155"></a><a name="p631912272155"></a>TF Serving命令行接口</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.1.1 "><p id="p14319182719156"><a name="p14319182719156"></a><a name="p14319182719156"></a>--task_affinity_isolation</p>
</td>
</tr>
<tr id="row12319182741517"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.2.1"><p id="p331992771516"><a name="p331992771516"></a><a name="p331992771516"></a>接口功能</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.2.1 "><p id="p65711337151515"><a name="p65711337151515"></a><a name="p65711337151515"></a>使能线程亲和性隔离特性，有两种隔离方式：</p>
<a name="ul5571143761519"></a><a name="ul5571143761519"></a><ul id="ul5571143761519"><li>顺序绑核，TensorFlow计算线程绑定到前K个核，TF Serving通信线程绑定到其余核。</li><li>交叉绑核，适用于开启超线程的场景，将TensorFlow线程绑定到物理核，TF Serving通信线程绑定到虚拟核。</li></ul>
</td>
</tr>
<tr id="row1319162721514"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.3.1"><p id="p631952721519"><a name="p631952721519"></a><a name="p631952721519"></a>参数类型</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.3.1 "><p id="p12319927121515"><a name="p12319927121515"></a><a name="p12319927121515"></a>std::string。</p>
</td>
</tr>
<tr id="row5938164721511"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.4.1"><p id="p189382479158"><a name="p189382479158"></a><a name="p189382479158"></a>参数格式</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.4.1 "><p id="p4938164716155"><a name="p4938164716155"></a><a name="p4938164716155"></a>mode;m-n;k，默认0。</p>
</td>
</tr>
<tr id="row1831912274154"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.5.1"><p id="p4319127111519"><a name="p4319127111519"></a><a name="p4319127111519"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.5.1 "><p id="p1231902713153"><a name="p1231902713153"></a><a name="p1231902713153"></a>请参见<a href="#线程亲和性隔离参数格式取值说明">线程亲和性隔离参数格式取值说明</a>。</p>
</td>
</tr>
<tr id="row7319727101519"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.6.1"><p id="p15319127171510"><a name="p15319127171510"></a><a name="p15319127171510"></a>推荐场景</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.6.1 "><a name="ul685628191620"></a><a name="ul685628191620"></a><ul id="ul685628191620"><li>使用TensorFlow调度方式运行时，推荐设置为顺序绑核；</li><li>与--batch_op_scheduling选项同时使能，并开启超线程时，推荐设置为交叉绑核。</li></ul>
</td>
</tr>
<tr id="row1232011272159"><th class="firstcol" valign="top" width="16.89%" id="mcps1.1.3.7.1"><p id="p1432022761515"><a name="p1432022761515"></a><a name="p1432022761515"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="83.11%" headers="mcps1.1.3.7.1 "><p id="p1152862014166"><a name="p1152862014166"></a><a name="p1152862014166"></a>一台160个物理核的服务器，开启超线程共320个核心，4个NUMA，每个NUMA上80个核心。</p>
<a name="ul652816204167"></a><a name="ul652816204167"></a><ul id="ul652816204167"><li>如果使用TensorFlow调度方式运行，运行参数可参考：<a name="screen1352813201166"></a><a name="screen1352813201166"></a><pre class="screen" codetype="Linux" id="screen1352813201166">numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=75 --tensorflow_inter_op_parallelism=75 --task_affinity_isolation="1;0-79;75"</pre>
</li></ul>
<a name="ul14528172051619"></a><a name="ul14528172051619"></a><ul id="ul14528172051619"><li>如果使能了--batch_op_scheduling选项，--tensorflow_inter_op_parallelism参数推荐设置为物理核数量，其他运行参数可参考：<a name="screen4528020161615"></a><a name="screen4528020161615"></a><pre class="screen" codetype="Linux" id="screen4528020161615">numactl -C 0-79 -m 0 /path/to/tensorflow_model_server  --port=8850 --rest_api_port=8851 --model_base_path=/path/to/saved_model/ --model_name=model --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=40 --batch_op_scheduling=true --task_affinity_isolation="2;0-79"</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

**表 1** 线程亲和性隔离参数格式取值说明<a id="线程亲和性隔离参数格式取值说明"></a>

<a name="table12688064377"></a>
<table><thead align="left"><tr id="row468814643718"><th class="cellrowborder" valign="top" width="7.43925607439256%" id="mcps1.2.5.1.1"><p id="p176881266377"><a name="p176881266377"></a><a name="p176881266377"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.448455154484552%" id="mcps1.2.5.1.2"><p id="p14656122373711"><a name="p14656122373711"></a><a name="p14656122373711"></a>取值范围</p>
</th>
<th class="cellrowborder" valign="top" width="27.90720927907209%" id="mcps1.2.5.1.3"><p id="p1768920617371"><a name="p1768920617371"></a><a name="p1768920617371"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="49.2050794920508%" id="mcps1.2.5.1.4"><p id="p3689265373"><a name="p3689265373"></a><a name="p3689265373"></a>约束</p>
</th>
</tr>
</thead>
<tbody><tr id="row168915612378"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p196891768371"><a name="p196891768371"></a><a name="p196891768371"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p265616232373"><a name="p265616232373"></a><a name="p265616232373"></a>0、1、2</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><a name="ul1088015211937"></a><a name="ul1088015211937"></a><ul id="ul1088015211937"><li>0：OFF，不使能线程亲和。</li><li>1：ORDER，按顺序绑核。</li><li>2：INTERVAL，交叉绑核。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p46891769376"><a name="p46891769376"></a><a name="p46891769376"></a>mode=0时，m-n、k两个参数无效（可不填）。</p>
</td>
</tr>
<tr id="row8689116183715"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p176894612375"><a name="p176894612375"></a><a name="p176894612375"></a>m-n</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p26571123153717"><a name="p26571123153717"></a><a name="p26571123153717"></a>可用的CPU核</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p768915663712"><a name="p768915663712"></a><a name="p768915663712"></a><span>绑核范围[m, n]</span>。</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p7689660371"><a name="p7689660371"></a><a name="p7689660371"></a><span>m </span><span>&lt;</span><span>= n</span>。</p>
</td>
</tr>
<tr id="row3273102020388"><td class="cellrowborder" valign="top" width="7.43925607439256%" headers="mcps1.2.5.1.1 "><p id="p1427332019385"><a name="p1427332019385"></a><a name="p1427332019385"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="15.448455154484552%" headers="mcps1.2.5.1.2 "><p id="p7273122013811"><a name="p7273122013811"></a><a name="p7273122013811"></a>可用的CPU核</p>
</td>
<td class="cellrowborder" valign="top" width="27.90720927907209%" headers="mcps1.2.5.1.3 "><p id="p727352063819"><a name="p727352063819"></a><a name="p727352063819"></a>分配给TensorFlow线程的核数。</p>
</td>
<td class="cellrowborder" valign="top" width="49.2050794920508%" headers="mcps1.2.5.1.4 "><p id="p6273142017388"><a name="p6273142017388"></a><a name="p6273142017388"></a>k &lt;= n - m + 1，即不大于绑核总数；mode=2时，参数k无效（可不填）。</p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **说明：** 
>numactl是一个在Linux系统上用于控制和管理NUMA（非统一内存访问，Non-Uniform Memory Access）架构的工具。可通过yum工具安装：
>
>```bash
>yum install -y numactl numactl-devel
>```
>
>**numactl -C 0-79 -m 0**是限定TF Serving服务运行在NUMA 0对应的核上，以该方式启动可以充分利用CPU资源，-C指定NUMA 0对应的核，-m指的是使用NUMA 0对应的内存。


## TensorFlow KDNN线程直通特性使用说明


**TensorFlow KDNN线程直通<a name="section1912343520617"></a>**

TensorFlow KDNN线程直通特性开关通过KDNN特性开关控制，具体说明如[表1 KDNN特性开关](#table473618378218)所示

**表 1**  KDNN特性开关

<a name="table473618378218"></a>
<table><tbody><tr id="row273612371926"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.1.1"><p id="p37361437223"><a name="p37361437223"></a><a name="p37361437223"></a>KDNN特性开关</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.1.1 "><p id="p147351257418"><a name="p147351257418"></a><a name="p147351257418"></a>TF_ENABLE_KDNN_OPTS</p>
</td>
</tr>
<tr id="row2736637329"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.2.1"><p id="p17544180133510"><a name="p17544180133510"></a><a name="p17544180133510"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.2.1 "><p id="p135441705354"><a name="p135441705354"></a><a name="p135441705354"></a>进程环境变量</p>
</td>
</tr>
<tr id="row1373713374215"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.3.1"><p id="p173719376217"><a name="p173719376217"></a><a name="p173719376217"></a>功能</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.3.1 "><p id="p15737237026"><a name="p15737237026"></a><a name="p15737237026"></a>控制开启KDNN优化特性</p>
</td>
</tr>
<tr id="row17371371026"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.4.1"><p id="p187371371426"><a name="p187371371426"></a><a name="p187371371426"></a>变量取值</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.4.1 "><p id="p139187124325"><a name="p139187124325"></a><a name="p139187124325"></a>0：KDNN关闭</p>
<p id="p18324912320"><a name="p18324912320"></a><a name="p18324912320"></a>1：KDNN开启</p>
</td>
</tr>
<tr id="row17372371123"><th class="firstcol" valign="top" width="17.11%" id="mcps1.2.3.5.1"><p id="p14737103710220"><a name="p14737103710220"></a><a name="p14737103710220"></a>使用示例</p>
</th>
<td class="cellrowborder" valign="top" width="82.89%" headers="mcps1.2.3.5.1 "><p id="p1926119477390"><a name="p1926119477390"></a><a name="p1926119477390"></a>首次调用KDNN算子前设置进程环境变量TF_ENABLE_KDNN_OPTS，例如python内可以通过命令<strong id="b166491125328"><a name="b166491125328"></a><a name="b166491125328"></a>os.environ['TF_ENABLE_KDNN_OPTS'] = str(1)</strong>设置进程环境变量。</p>
</td>
</tr>
</tbody>
</table>