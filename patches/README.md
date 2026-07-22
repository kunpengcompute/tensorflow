# TensorFlow补丁开发与发布说明

本项目在开发阶段维护一份完整的TensorFlow源码，再按特性拆分为多个发布补丁，
不为每个特性单独维护分支。

所有补丁都基于官方TensorFlow `v2.15.0`的固定提交：

```text
6887368d6d46223f460358323c4b76d61d1558a8
```

如果本地仓库没有该提交，请先从
`https://github.com/tensorflow/tensorflow.git`获取`v2.15.0`标签。

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `manifest.json` | 定义补丁分组、文件归属和可用Profile |
| `patch_manager.py` | 审计文件归属、生成补丁并验证应用结果 |
| `prepare_source.py` | 根据Profile创建一份可直接构建的完整源码 |
| `feature/` | 自动生成的维护中特性补丁 |
| `frozen_feature/` | 独立保存、不再拆分维护的Legacy补丁 |
| `publish_to_master.py` | 将补丁和发布资料同步到`master` worktree |
| `release/` | 发布到`master`时使用的说明文档模板 |

## 补丁组成

维护中的补丁按以下顺序应用：

| 顺序 | 分组 | 内容 |
| --- | --- | --- |
| 1 | `common` | 公共Bazel配置、仓库集成和通用编译修复 |
| 2 | `kdnn` | KDNN算子优化和SparseMatmul |
| 3 | `annc` | ANNC静态图融合算子及测试 |
| 4 | `kembedding` | KEmbedding算子、依赖库、测试和性能脚本 |

目前提供四种Profile：

| Profile | 补丁组合 | 用途 |
| --- | --- | --- |
| `common-only` | common | 单独检查公共改动 |
| `kdnn-core` | common + kdnn | 最小支持的产品组合 |
| `kdnn-annc` | common + kdnn + annc | 使用KDNN和ANNC静态图融合 |
| `full-default` | common + kdnn + annc + kembedding | 使用全部维护中特性 |

## 补丁如何拆分

`manifest.json`为每个源码路径指定唯一的补丁分组。`patch_manager.py`会完成三项
工作：

1. `audit`：检查每个改动文件是否有且只有一个归属分组。
2. `generate`：根据归属结果生成`feature/`下的补丁。
3. `verify`：从官方基线依次应用各Profile，并检查`full-default`能否完整还原
   开发源码。

`tensorflow/tensorflow.bzl`属于`common`，只提供稳定的公共编译入口。可选特性
应把自己的编译选项放在`tensorflow/patch_features/`下，并在`manifest.json`
的`feature_copts`中登记。工具会按Profile生成
`tensorflow/feature_copts.bzl`，避免多个特性直接修改同一个公共文件。

特性代码继续使用TensorFlow原有的平台选择逻辑。例如，ANNC在AArch64平台
默认参与构建，不需要额外传入`--config=annc`。

## 开发与验证

修改尚未提交时，使用`WORKTREE`直接验证当前工作区：

```bash
python3 patches/patch_manager.py --source WORKTREE audit
python3 patches/patch_manager.py --source WORKTREE generate
python3 patches/patch_manager.py --source WORKTREE verify
```

`WORKTREE`会读取已跟踪修改以及有明确归属的未跟踪文件，不需要执行
`git add`，也不会修改真实Git索引。

修改提交后，可以改用`HEAD`复核提交内容：

```bash
python3 patches/patch_manager.py --source HEAD audit
python3 patches/patch_manager.py --source HEAD generate
python3 patches/patch_manager.py --source HEAD verify
```

`feature/`中的补丁由工具自动生成，不要手工修改。

## 创建完整源码

不要只按顺序手工应用补丁来准备编译目录。使用`prepare_source.py`选择Profile，
工具会检出官方基线、应用对应补丁，并生成`tensorflow/feature_copts.bzl`等
必要文件。

例如，创建`kdnn-core`源码：

```bash
python3 patches/prepare_source.py \
  --feature-set kdnn-core \
  --output-dir /path/to/tensorflow-kdnn-core
```

手工应用单个补丁适合检查diff范围，不适合作为完整源码的标准准备方式。

## 修改补丁归属

相对官方基线发生变化的每个路径只能属于一个分组。开发新特性时应遵循以下
规则：

- 特性独有文件由对应特性分组维护。
- 多个特性共用的修改应归入`common`。
- 如果特性必须接入公共文件，先将特性实现拆到独立文件，再通过稳定入口调用。
- 不要在`manifest.json`中添加相互重叠的路径规则。
- 提交前确保`git diff --check`和四个Profile验证全部通过。

## Legacy补丁

历史Runtime调度、TensorFlow框架内融合Embedding、线程亲和性和旧XLA执行改动
保存在：

```text
patches/frozen_feature/tensorflow_2.15.0-legacy.patch
```

Legacy补丁具有以下约束：

- 直接基于官方TensorFlow `v2.15.0`基线生成。
- 不属于`manifest.json`管理的正常补丁系列。
- 不依赖`common`，也不保证与KDNN、ANNC或KEmbedding组合兼容。
- 后续不再合入新功能，也不由`patch_manager.py`重新生成。

## 发布到master

特性开发保留在`patch-refactor-2.15`分支。发布时使用独立的`master` worktree，
避免在开发分支中混入发布资料修改：

```bash
git worktree add /home/c00913906/tensorflow_master_release master

python3 patches/publish_to_master.py \
  --source HEAD \
  --master-worktree /home/c00913906/tensorflow_master_release
```

发布脚本会：

1. 审计`manifest.json`并重新生成四个维护中补丁。
2. 验证全部Profile。
3. 将维护中补丁发布到`patches/feature/`。
4. 将Legacy补丁发布到`patches/frozen_feature/`。
5. 刷新README、工具脚本、`SOURCE_COMMIT`和`SHA256SUMS`。
6. 删除根目录下废弃的旧补丁文件。

脚本只修改目标worktree，不会自动提交。完成检查后，在`master` worktree中
单独提交发布内容。
