# TensorFlow 2.15.0 补丁发布说明

本目录发布鲲鹏 TensorFlow 基于官方 TensorFlow `v2.15.0` 的补丁。固定基线为：

```text
6887368d6d46223f460358323c4b76d61d1558a8
```

发布来源记录在 `SOURCE_COMMIT`，补丁校验值记录在 `SHA256SUMS`。

## 正常补丁系列

| 顺序 | 分组 | 说明 |
| --- | --- | --- |
| 1 | `common` | 公共 Bazel、仓库集成和通用编译修复 |
| 2 | `kdnn` | KDNN 对接、KDNN 算子和 SparseMatmul 优化 |
| 3 | `annc` | ANNC 静态图融合算子及测试 |
| 4 | `kembedding` | KEmbedding 自定义算子、测试和性能脚本 |

支持以下构建组合：

| Profile | 分组 | 用途 |
| --- | --- | --- |
| `common-only` | common | 检查公共改动 |
| `kdnn-core` | common + kdnn | 最小推荐产品组合 |
| `kdnn-annc` | common + kdnn + annc | KDNN 与 ANNC |
| `full-default` | common + kdnn + annc + kembedding | 完整默认组合 |

`tensorflow/feature_copts.bzl` 会根据特性组合动态生成。请使用
`prepare_source.py` 创建可构建源码，不要仅按顺序手工应用补丁。

```bash
git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
cd kunpeng-tensorflow
git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
git fetch tensorflow-upstream refs/tags/v2.15.0:refs/tags/v2.15.0

python3 patches/prepare_source.py \
  --feature-set kdnn-core \
  --output-dir ../tensorflow-kdnn-core
```

切换 `--feature-set` 即可创建其他组合。KDNN profile 还需要按照 KDNN 安装文档提供对应的头文件和库。

## Legacy 补丁

`frozen/0005-tensorflow_2.15.0-legacy.patch` 是独立、冻结的历史功能集合，包含
旧融合 Embedding、批量调度、线程亲和性和 XLA 执行改动。

Legacy 补丁：

- 只允许直接应用到官方 TensorFlow `v2.15.0` 基线。
- 不依赖 `common`，也不属于上面的正常补丁顺序。
- 不保证与 KDNN、ANNC 或 KEmbedding 补丁组合兼容。
- 后续不再合入功能，也不维护内部拆分逻辑。

```bash
git clone -b v2.15.0 https://github.com/tensorflow/tensorflow.git tensorflow-legacy
cd tensorflow-legacy
git apply --check ../kunpeng-tensorflow/patches/frozen/0005-tensorflow_2.15.0-legacy.patch
git apply ../kunpeng-tensorflow/patches/frozen/0005-tensorflow_2.15.0-legacy.patch

mkdir -p output distdir output-release
bazel --output_base="$PWD/output" build \
  --distdir="$PWD/distdir" \
  --config=fused_embedding \
  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output-release
```

## 完整性检查

```bash
cd patches
sha256sum -c SHA256SUMS
```

`prepare_source.py` 用于创建完整源码，`manifest.json` 和 `patch_manager.py`
用于维护和验证补丁。`dist/` 下的文件是生成物，不应手工修改；`frozen/`
下的 Legacy 补丁也不参与重新生成。
