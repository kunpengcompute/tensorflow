# TensorFlow v2.20.0补丁发布说明

本文档介绍鲲鹏TensorFlow基于官方TensorFlow `v2.20.0`的补丁。固定基线为以下内容。

```text
bf5899deaf70fa45173c5c7b8dc9ace8824dc980
```

发布来源记录在`SOURCE_COMMIT`，补丁校验值记录在`SHA256SUMS`。

## 正常补丁系列

| 顺序 | 分组 | 说明 |
| --- | --- | --- |
| 1 | common | 公共Bazel仓库集成、通用编译修复及gtest覆盖率测试框架 |
| 2 | kdnn | KDNN加速库集成及稀疏矩阵乘快路径 |
| 3 | annc | ANNC静态图融合算子及图优化 |
| 4 | kembedding | KEmbedding Embedding表查找算子及基准测试 |

支持以下构建组合。

| Profile | 分组 | 用途 |
| --- | --- | --- |
| common-only | common | 检查公共改动 |
| kdnn-core | common+kdnn | KDNN加速特性（CPU） |
| kdnn-annc | common+kdnn+annc | KDNN加速及ANNC静态图融合 |
| full-default | common+kdnn+annc+kembedding | 全部特性 |

`tensorflow/feature_copts.bzl` 会根据特性组合动态生成。请使用`prepare_source.py`创建可构建源码，不要仅按顺序手工应用补丁。

```bash
git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
cd kunpeng-tensorflow
git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0

python3 patches/Tensorflow_V2.20.0/prepare_source.py \
  --feature-set full-default \
  --output-dir ../tensorflow-full-default
```

切换`--feature-set`即可创建其他组合。

## 完整性检查

执行以下命令检查补丁完整性。

```bash
cd patches/Tensorflow_V2.20.0
sha256sum -c SHA256SUMS
```

`prepare_source.py`用于创建完整源码，`manifest.json`用于维护补丁。`feature/`下的文件是生成物，不应手工修改。
