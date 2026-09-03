# TensorFlow Benchmark Tools

## Serving Benchmark Sweep

Use `run_serving_benchmark.sh` inside `benchmark_infer_workspace` to run the
standard baseline-vs-DNN serving sweep. Defaults match the `wd_dcn` benchmark
contract: batch range `1-100`, target QPS `100 350 600 850`, warmup `5s`, test
duration `30s`, `/usr/local/bin` server/client binaries, and CPU sampling from
`/proc/<predictor_server_pid>/stat`.

```bash
cd /workspace/tensorflow

MODEL_NAME=din_mmoe \
PORT=8892 \
./tools/run_serving_benchmark.sh
```

The default `RUN_MODES="baseline dnn"` runs:

```text
baseline: TF_ENABLE_ONEDNN_OPTS=0, --enable_kdnn=false
dnn:      TF_ENABLE_ONEDNN_OPTS=1, --enable_kdnn=true
```

Use `RUN_BASELINE=false`, `RUN_DNN=false`, or `RUN_MODES="baseline"` /
`RUN_MODES="dnn"` to limit the sweep. The `predictor_server` gflag is still
named `--enable_kdnn`; script modes, metadata, CSV, and Markdown output use
`dnn`.

Override `BIN_SERVER` / `BIN_CLIENT` only when intentionally testing locally
built binaries instead of the image-provided `/usr/local/bin` versions.

Useful overrides:

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

The script writes raw client/server logs, per-second CPU samples,
`metadata.txt`, `results.csv`, and a README-ready Markdown table in
`results.md`.

## Collect Serving Timeline

Use `collect_serving_timeline.py` from the host to collect one
`predictor_server` TensorFlow timeline through the active inference container.
The helper starts `predictor_server`, runs `brpc_client`, waits for
`*.runmeta.pb`, converts it to Chrome trace JSON with
`runmetadata_to_timeline.py`, and stops the server it started.

Example for `din_mmoe` with 50 samples grouped into each client RPC:

```bash
cd /home/c00913906/tensorflow

./tools/collect_serving_timeline.py \
  --model-name din_mmoe \
  --batch-size 50 \
  --port 8891
```

`--batch-size` only sets the client-side `--request_min_batch_size` and
`--request_max_batch_size`. The helper does not set a server-side batch
parameter.

Defaults assume:

```text
container: benchmark_infer_workspace
model: /workspace/<model-name>/1
input data: /workspace/benchmark_dataset/<model-name>.tsv
tensorflow dir: /workspace/tensorflow
timeline root: /workspace/timeline
```

The requested port is only the preferred starting point. If it is busy, the
helper chooses the next free port.

For models or datasets stored elsewhere, pass explicit paths:

```bash
./tools/collect_serving_timeline.py \
  --model-name wd_dcn \
  --model-path /workspace/wd_dcn/1 \
  --input-data-path /workspace/benchmark_dataset/wd_dcn.tsv \
  --batch-size 10 \
  --run-name wd_dcn_batch_10
```
