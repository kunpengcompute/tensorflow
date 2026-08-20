# TensorFlow 2.20.0 补丁发布说明

本文档介绍鲲鹏 TensorFlow 基于官方 TensorFlow `v2.20.0` 的补丁。固定基线为。

```text
bf5899deaf70fa45173c5c7b8dc9ace8824dc980
```

发布来源记录在 `SOURCE_COMMIT`，补丁校验值记录在 `SHA256SUMS`。

## 正常补丁系列

| 顺序 | 分组 | 说明 |
| --- | --- | --- |
| 1 | `common` | 公共 Bazel、仓库集成和通用编译修复 |
| 2 | `fago` | FAGO 静态图融合算子及测试 |

支持以下构建组合：

| Profile | 分组 | 用途 |
| --- | --- | --- |
| `common-only` | common | 检查公共改动 |
| `common-fago` | common + annc | FAGO 静态图融合特性 |

`tensorflow/feature_copts.bzl` 会根据特性组合动态生成。请使用
`prepare_source.py` 创建可构建源码，不要仅按顺序手工应用补丁。

```bash
git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
cd kunpeng-tensorflow
git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0

python3 patches/Tensorflow_V2.20.0/prepare_source.py \
  --feature-set fago-core \
  --output-dir ../tensorflow-fago-core
```

切换 `--feature-set` 即可创建其他组合。

## 完整性检查

执行以下命令检查补丁完整性。

```bash
cd patches/Tensorflow_V2.20.0
sha256sum -c SHA256SUMS
```

`prepare_source.py` 用于创建完整源码，`manifest.json` 
用于维护补丁。`feature/` 下的文件是生成物，不应手工修改。
