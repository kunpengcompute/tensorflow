# Dummy Server 构建指导

本文档指导用户基于 Dockerfile 构建镜像，启动容器并编译 `predictor_server`。

## 环境要求

| 项目     | 要求                      |
| ------ | ----------------------- |
| 架构     | aarch64 (ARM64)         |
| OS     | openEuler 24.03 LTS SP3 |
| Docker | Docker Engine / CLI 25.x |
| Compose | Docker Compose v2，可通过 `docker compose` 调用 |
| 磁盘     | 至少 30GB 可用空间            |

### 安装 Moby / Docker Compose

openEuler 24.03 LTS SP3 仓库中的 `docker-engine` 版本较旧，不能识别
Docker CLI plugins，因此无法使用 `docker compose` 子命令。推荐安装
`moby-engine` 和 `moby-client`：

```bash
sudo dnf install -y moby-engine moby-client
sudo systemctl enable --now docker
```

如果系统中已安装旧版 `docker-engine`，可切换到 Moby 包：

```bash
sudo dnf swap -y docker-engine moby-engine
sudo dnf install -y moby-client
sudo systemctl restart docker
```

`moby-engine` 提供 Docker daemon，`moby-client` 提供 `/usr/bin/docker`
客户端。安装后验证：

```bash
docker --version
docker info
```

安装 Docker Compose v2 插件（aarch64）：

```bash
sudo mkdir -p /usr/local/lib/docker/cli-plugins
curl -fL https://github.com/docker/compose/releases/download/v5.1.4/docker-compose-linux-aarch64 \
  -o /tmp/docker-compose
sudo install -m 0755 /tmp/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
```

如果服务器访问 GitHub 需要代理，可在 `curl` 中增加代理参数，例如：

```bash
curl -fL --proxy http://127.0.0.1:11082 \
  https://github.com/docker/compose/releases/download/v5.1.4/docker-compose-linux-aarch64 \
  -o /tmp/docker-compose
```

验证 Compose：

```bash
docker compose version
```

预期输出类似：

```text
Docker Compose version v5.1.4
```

## 1. 准备离线依赖包

部分依赖因网络原因需提前下载，放置到 `/home/workspase/download/` 目录：

```bash
mkdir -p /home/workspase/download
```

所需文件：

| 文件                                                                     | 来源                               | 下载地址                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cpython-3.11.10+20241016-aarch64-unknown-linux-gnu-install_only.tar.gz | indygreg/python-build-standalone | [https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-aarch64-unknown-linux-gnu-install_only.tar.gz](https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-aarch64-unknown-linux-gnu-install_only.tar.gz) |
| bazel-skylib-1.8.1.tar.gz                                              | bazelbuild/bazel-skylib          | [https://github.com/bazelbuild/bazel-skylib/releases/download/1.8.1/bazel-skylib-1.8.1.tar.gz](https://github.com/bazelbuild/bazel-skylib/releases/download/1.8.1/bazel-skylib-1.8.1.tar.gz)                                                                                                                   |
| rules_cc-0.0.17.tar.gz                                                 | bazelbuild/rules_cc              | [https://github.com/bazelbuild/rules_cc/releases/download/0.0.17/rules_cc-0.0.17.tar.gz](https://github.com/bazelbuild/rules_cc/releases/download/0.0.17/rules_cc-0.0.17.tar.gz)                                                                                                                               |

> **注意**：以上为 GitHub 直链，如服务器无法访问 GitHub，可在可联网的机器上下载后通过 scp 传输到服务器。

在可联网的机器上下载后，通过 scp 传输到服务器：

```bash
scp cpython-3.11.10+20241016-aarch64-unknown-linux-gnu-install_only.tar.gz \
    bazel-skylib-1.8.1.tar.gz \
    rules_cc-0.0.17.tar.gz \
    <user>@<server_ip>:/home/workspase/download/
```

确认文件完整性：

```bash
ls -lh /home/workspase/download/
```

预期输出：

```
bazel-skylib-1.8.1.tar.gz
cpython-3.11.10+20241016-aarch64-unknown-linux-gnu-install_only.tar.gz
rules_cc-0.0.17.tar.gz
```

## 2. 构建 Docker 镜像

在项目根目录（包含 Dockerfile 的目录）执行：

```bash
cd /home/c00913906/tensorflow
docker build --network=host \
  -t benchmark-infer:v1.0.0 .
```

构建阶段使用 `--network=host`，便于镜像内访问宿主机上的代理服务。
Dockerfile 默认代理地址为 `http://127.0.0.1:11082`，使用前请改成
当前机器实际配置的代理地址；也可以通过 build arg 覆盖：

```bash
docker build --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:11082 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:11082 \
  -t benchmark-infer:v1.0.0 .
```

镜像包含以下组件：

| 组件        | 版本                      |
| --------- | ----------------------- |
| OS        | openEuler 24.03 LTS SP3 |
| GCC / G++ | 14.3.1 (gcc-toolset-14) |
| JDK       | OpenJDK 21              |
| Bazel     | 7.4.1                   |

构建完成后验证：

```bash
docker run --rm benchmark-infer:v1.0.0 bash -c "gcc --version | head -1 && bazel --version"
```

预期输出：

```
gcc (GCC) 14.3.1 20250523 (openEuler 14.3.1-14.oe2403sp3)
bazel 7.4.1
```

## 3. 启动容器

推荐使用 `docker-compose.infer.yml` 快速拉起常驻推理容器。默认配置为：

- `privileged: true`
- `network_mode: host`
- CPU 绑定默认 `0-95`
- CFS period 固定 `100000`，默认规格 `CPU_CORES=16`，启动脚本自动换算为 `cpu_quota=1600000`
- 默认目录映射 `${HOST_WORKSPACE:-/home/c00913906}:/workspace`

默认启动：

```bash
cd /home/c00913906/tensorflow
./compose-infer.sh up -d
```

如需调整 CPU 绑定、CFS 规格或宿主机映射目录，复制 `.env.tmp` 为 `.env` 后修改：

```bash
cd /home/c00913906/tensorflow
cp .env.tmp .env
vi .env
./compose-infer.sh up -d
```

`.env` 中只需要配置规格，例如 `CPU_CORES=32` 会自动换算为
`cpu_quota=3200000`；`cpu_period` 固定为 `100000`。宿主机映射目录通过
`HOST_WORKSPACE` 配置，容器内路径固定为 `/workspace`。

`docker-compose.infer.yml` 要求 Docker Compose v2 通过 `docker compose`
调用，当前只绑定 CPU 核心，不绑定 NUMA 内存节点。

等价 `docker run` 示例：

```bash
docker run -d \
    --name brpc_server \
    --privileged \
    --cpuset-cpus=0-95 \
    --cpu-period=100000 \
    --cpu-quota=1600000 \
    --network=host \
    -v /home/c00913906:/workspace \
    benchmark-infer:v1.0.0 \
    tail -f /dev/null
```

参数说明：

| 参数                         | 说明                                    |
| -------------------------- | ------------------------------------- |
| `--privileged`             | 特权启动，便于使用性能调优和系统观测能力                 |
| `--cpuset-cpus=0-95`       | 绑定 CPU 核心 0-95，请根据实际核心范围调整            |
| `--cpu-period=100000`      | CFS 调度周期，默认 100ms                       |
| `--cpu-quota=1600000`      | CFS 配额，默认 16C；如需 32C 可设为 `3200000`      |
| `--network=host`           | 与宿主机共享网络栈，容器直接使用宿主机网络                 |
| `-v /home/c00913906:/workspace` | 将宿主机工作目录映射到容器 `/workspace`        |
| `tail -f /dev/null`        | 保持容器前台运行，避免退出                         |

## 4. 编译 predictor_server 和 brpc_client

进入容器：

```bash
docker exec -it brpc_server bash
```

在容器内执行编译：

```bash
cd /data/tensorflow

bazel --output_user_root=./output build --noenable_bzlmod --experimental_repo_remote_exec --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --copt=-O3 --host_copt=-O3 --copt=-march=armv8.5-a --host_copt=-I/usr/local/include --linkopt=-L/usr/local/lib64 --host_linkopt=-L/usr/local/lib64 --linkopt=-Wl,-rpath,/usr/local/lib64 --host_linkopt=-Wl,-rpath,/usr/local/lib64 --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} --action_env=TF_SYSTEM_LIBS="boringssl,snappy" --distdir=/data/download --check_direct_dependencies=off //:predictor_server

bazel --output_user_root=./output build --noenable_bzlmod --experimental_repo_remote_exec --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --copt=-O3 --host_copt=-O3 --copt=-march=armv8.5-a --host_copt=-I/usr/local/include --linkopt=-L/usr/local/lib64 --host_linkopt=-L/usr/local/lib64 --linkopt=-Wl,-rpath,/usr/local/lib64 --host_linkopt=-Wl,-rpath,/usr/local/lib64 --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} --action_env=TF_SYSTEM_LIBS="boringssl,snappy" --distdir=/data/download --check_direct_dependencies=off //:brpc_client
```

编译参数说明：

| 参数                                 | 说明                                                             |
| ---------------------------------- | -------------------------------------------------------------- |
| `--output_user_root=./output`      | Bazel 输出根目录设为项目下的 `output/`                                    |
| `--noenable_bzlmod`                | 使用与 `dev_2.20.0` 基线一致的 WORKSPACE 依赖管理                       |
| `--cxxopt=-std=c++17`              | C++编译标准为 C++17                                                 |
| `--host_cxxopt=-std=c++17`         | 主机工具编译标准为 C++17                                                |
| `--copt=-O3`                       | 目标 C/C++ 代码启用 `-O3` 优化                                        |
| `--host_copt=-O3`                  | 主机构建工具启用 `-O3` 优化                                             |
| `--copt=-march=armv8.5-a`          | 针对 ARMv8.5-A 架构优化                                              |
| `--action_env=TF_SYSTEM_LIBS`      | 传递系统库环境变量                                                      |
| `--define=use_system_libs=openssl` | 使用系统 OpenSSL                                                   |
| `--distdir=/data/download`         | 优先从本地 `/data/download` 查找依赖包（对应宿主机 `/home/workspase/download`） |
| `--check_direct_dependencies=off`  | 关闭直接依赖检查，避免版本冲突告警                                              |

编译产物路径：

```
bazel-bin/predictor_server
bazel-bin/brpc_client
```

镜像内置的标准压测二进制路径：

```
/usr/local/bin/predictor_server
/usr/local/bin/brpc_client
```

启动 server。baseline 模式需要显式关闭 oneDNN，并关闭 DNN 优化开关：

```bash
TF_NUM_INTEROP_THREADS=16 \
TF_NUM_INTRAOP_THREADS=16 \
TF_ENABLE_ONEDNN_OPTS=0 \
numactl -C 0-15 /usr/local/bin/predictor_server \
  --enable_kdnn=false \
  --model_path=/data/sra_benchmark/modelzoo/wd_dcn/result/version_b_graphopt_sparse0065_emb12x/saved_model/1 \
  --thread_num=16
```

DNN 模式则显式开启 oneDNN，并开启 DNN 优化开关：

```bash
TF_NUM_INTEROP_THREADS=16 \
TF_NUM_INTRAOP_THREADS=16 \
TF_ENABLE_ONEDNN_OPTS=1 \
numactl -C 0-15 /usr/local/bin/predictor_server \
  --enable_kdnn=true \
  --model_path=/data/sra_benchmark/modelzoo/wd_dcn/result/version_b_graphopt_sparse0065_emb12x/saved_model/1 \
  --thread_num=16
```

当前 `predictor_server` 的 DNN 优化 gflag 仍名为 `--enable_kdnn`。
标准压测推荐使用 `tools/run_serving_benchmark.sh`，脚本会为
`baseline` 设置 `TF_ENABLE_ONEDNN_OPTS=0 --enable_kdnn=false`，为 `dnn`
设置 `TF_ENABLE_ONEDNN_OPTS=1 --enable_kdnn=true`，并默认使用镜像内置的
`/usr/local/bin/predictor_server` 和 `/usr/local/bin/brpc_client`。

启动 client：

```bash
/usr/local/bin/brpc_client --server=127.0.0.1:8000 --input_data_path=/data/sra_benchmark/modelzoo/wd_dcn/result/version_b_infer.tsv --thread_num=16   --test_duration_s=10 --warmup_duration_s=5 \
  --max_qps=1000 \
  --max_inflight=100 \
  --request_min_batch_size=1 \
  --request_max_batch_size=100
```

`brpc_client` 的 `--max_qps` 按全局 RPC QPS 调度，不再按线程均分；`--max_inflight` 限制最大 in-flight RPC 数，超过时请求记为 dropped/failure，不发送到 server。`--request_min_batch_size` 和 `--request_max_batch_size` 只控制每个 RPC 内包装多少条样本。

## 5. 常用运维命令

```bash
# 查看容器状态
docker ps -f name=brpc_server

# 进入容器
docker exec -it brpc_server bash

# 停止容器
docker stop brpc_server

# 启动已停止的容器
docker start brpc_server

# 删除容器
docker rm -f brpc_server

# 清理 Bazel 编译缓存（在容器内）
bazel clean --expunge
```
