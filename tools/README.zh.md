# TensorFlow Benchmark 工具

## Serving Benchmark Sweep

在 `benchmark_infer_workspace` 容器中使用 `run_serving_benchmark.sh` 运行标准 baseline-vs-DNN 推理压测。默认参数与 `wd_dcn` benchmark 约定一致：batch 范围 `1-100`，目标 QPS 为 `100 350 600 850`，warmup `5s`，测试时长 `30s`，默认使用 `/usr/local/bin` 下的 server/client 二进制，并从 `/proc/<predictor_server_pid>/stat` 采集服务端 CPU 使用率。

示例：

```bash
cd /workspace/tensorflow

MODEL_NAME=din_mmoe \
PORT=8892 \
./tools/run_serving_benchmark.sh
```

默认 `RUN_MODES="baseline dnn"` 会运行：

```text
baseline: TF_ENABLE_ONEDNN_OPTS=0, --enable_kdnn=false
dnn:      TF_ENABLE_ONEDNN_OPTS=1, --enable_kdnn=true
```

可用 `RUN_BASELINE=false`、`RUN_DNN=false`，或 `RUN_MODES="baseline"` /
`RUN_MODES="dnn"` 限制压测模式。当前 `predictor_server` 的 gflag 仍名为
`--enable_kdnn`；脚本模式、metadata、CSV 和 Markdown 输出统一使用
`dnn`。

只有在有意验证本地构建产物时，才覆盖 `BIN_SERVER` / `BIN_CLIENT`。

常用覆盖参数：

```bash
MODEL_NAME=din_mmoe \
MODEL=/workspace/din_mmoe/1 \
DATA=/workspace/benchmark_dataset/din_mmoe.tsv \
OUT_DIR=/workspace/tmp/din_mmoe_benchmark_manual \
QPS_LIST="100 350 600 850" \
REQUEST_MIN_BATCH_SIZE=1 \
REQUEST_MAX_BATCH_SIZE=100 \
./tools/run_serving_benchmark.sh
```

脚本会在 `OUT_DIR` 中写入原始 client/server 日志、逐秒 CPU 采样、`metadata.txt`、`results.csv`，以及可以直接放进 README 的 Markdown 表格 `results.md`。

## 采集 Serving Timeline

在宿主机上使用 `collect_serving_timeline.py`，通过当前推理容器采集一次 `predictor_server` TensorFlow timeline。该脚本会启动 `predictor_server`，运行 `brpc_client`，等待生成 `*.runmeta.pb`，再用 `runmetadata_to_timeline.py` 转换为 Chrome trace JSON，最后停止它自己启动的 server。

采集 `din_mmoe`、`batch_size=50` 的示例：

```bash
cd /home/c00913906/tensorflow

python3 tools/collect_serving_timeline.py \
  --model-name din_mmoe \
  --batch-size 50 \
  --port 8891
```

默认路径约定：

```text
container: benchmark_infer_workspace
model: /workspace/<model-name>/1
input data: /workspace/benchmark_dataset/<model-name>.tsv
tensorflow dir: /workspace/tensorflow
timeline root: /workspace/timeline
```

`--port` 只是首选起始端口。如果该端口被占用，脚本会自动选择下一个空闲端口。

如果模型或数据集不在默认位置，可以显式传入路径：

```bash
python3 tools/collect_serving_timeline.py \
  --model-name wd_dcn \
  --model-path /workspace/sra_benchmark/modelzoo/wd_dcn/result/version_06151425/saved_model/1/ \
  --input-data-path /workspace/benchmark_dataset/wd_dcn.tsv \
  --warmup-duration-s 5 \
  --batch-size 10 \
  --run-name wd_dcn_batch_10
```
