# TensorFlow 2.15.0 补丁发布验证报告模板

> **请勿直接修改本模板。**
>
> 开发者执行验证前，应复制本文件并创建一份真实验证报告，例如
> `VALIDATION_REPORT_YYYYMMDD.md`。验证环境、命令、结果、日志和问题均应记录在
> 真实报告中，模板只用于维护统一的验证范围和报告结构。

## 使用方式

本模板可以直接作为Agent输入，用于指导Agent自动完成补丁验证。向Agent提供本模板时，还应提供待验证仓库、发布目录和编译容器。Agent必须遵循以下要求：

1. 先复制本模板创建真实验证报告，不得将结果写入模板。
2. 从 `patches/SOURCE_COMMIT`、`patches/manifest.json`、Git工作树和容器配置中读取实际信息，不得沿用示例值。
3. 按“发布文件完整性、补丁应用、编译验证、单元测试”的顺序执行。
4. 每完成一项立即更新真实报告，记录状态、耗时、产物或日志路径。
5. 验证失败时保留现场，在“问题记录”中说明现象和处理结果，不得将失败项标记为通过。
6. 只有全部必测项完成后才能填写验证结论。无法执行的项目必须标记为“阻塞”或“不适用”，并说明原因。

状态统一使用：`待执行`、`执行中`、`通过`、`失败`、`阻塞`、`不适用`。

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 验证日期 | `<YYYY-MM-DD>` |
| 官方基线 | `<从manifest.json读取>` |
| 发布记录提交 | `<从SOURCE_COMMIT读取>` |
| 本轮验证源码树 | `<Git提交或工作树tree ID，并注明工作树是否包含未提交修改>` |
| 发布分支 | `<分支名称>` |
| 编译镜像 | `<本地自定义镜像名称，并明确说明非TensorFlow官方发布镜像>` |
| 编译容器 | `<容器名称>` |
| CPU范围 | `<cpuset>` |
| 容器权限 | `<是否启用privileged>` |
| 源码挂载路径 | `<容器内路径>` |
| Bazel缓存 | `<output_base路径>` |
| 离线依赖目录 | `<distdir路径>` |

## 验证范围

本次验证覆盖发布文件完整性、补丁应用、各维护中Profile的编译与测试，以及独立Legacy补丁。维护中Profile按照 `patches/manifest.json` 定义执行；Legacy补丁只应用到官方基线，不与其他补丁组合。

## 验证结果

### 发布文件完整性

| 检查项 | 状态 | 结果或日志 |
| --- | --- | --- |
| `SHA256SUMS`校验 | 待执行 |  |
| `manifest.json`格式校验 | 待执行 |  |
| 补丁归属审计 | 待执行 |  |

### 补丁应用

| Profile | 补丁组合 | 状态 | 验证目录 | 结果或日志 |
| --- | --- | --- | --- | --- |
| `common-only` | common | 待执行 |  |  |
| `kdnn-core` | common + kdnn | 待执行 |  |  |
| `kdnn-annc` | common + kdnn + annc | 待执行 |  |  |
| `full-default` | common + kdnn + annc + kembedding | 待执行 |  |  |
| `legacy` | legacy | 待执行 |  |  |

### 编译验证

| Profile | 构建目标 | 状态 | 耗时 | 构建产物或日志 |
| --- | --- | --- | --- | --- |
| `common-only` | TensorFlow pip包 | 待执行 |  |  |
| `kdnn-core` | TensorFlow pip包 | 待执行 |  |  |
| `kdnn-annc` | TensorFlow pip包 | 待执行 |  |  |
| `full-default` | TensorFlow pip包 | 待执行 |  |  |
| `legacy` | TensorFlow pip包 | 待执行 |  |  |

### 单元测试

| Profile | 测试目标 | 状态 | 耗时 | 结果或日志 |
| --- | --- | --- | --- | --- |
| `kdnn-core` | KDNN相关单元测试 | 待执行 |  |  |
| `kdnn-annc` | ANNC相关单元测试 | 待执行 |  |  |
| `full-default` | KEmbedding相关单元测试 | 待执行 |  |  |

## 问题记录

| 编号 | Profile | 问题描述 | 处理结果 | 状态 |
| --- | --- | --- | --- | --- |

## 验证结论

待全部验证项完成后填写。
