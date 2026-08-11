## TensorFlow FAGO静态图融合特性使用说明

FlashAttentionGraphOptimization(FAGO)静态图融合特性开关通过环境变量开关控制，具体说明如[表1 FAGO静态图融合特性开关](#table473618378218)所示。

特性开关默认取值为0，即关闭FAGO静态图融合功能，如需使用需要手动在图优化前设置环境变量开启，以Python语言为例可以通过如下方式设置：

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
os.environ['ANNC_FUSED_FLASHATTN_FWD'] = '1'
```

**表 1**  FAGO静态图融合特性开关<a id="table473618378218"></a>

| 开关名   | 类型  | 取值  | 功能   |
| -------- | ------------ | --------------- | ------------------ |
| ANNC_FUSED_ALL | 进程环境变量 | 1：开启 0：关闭 | 用于开启FAGO所有融合算子静态图融合。    |
| ANNC_FUSED_FLASHATTN_FWD | 进程环境变量 | 1：开启 0：关闭 | 用于开启FlashAttentionForword算子静态图融合。    |

> ![icon note](public_sys-resources/icon-note.gif) **说明：**
> 以上算子当且仅当ANNC_FUSED_ALL=0且算子对应的环境变量为0时，该算子不会进行算子融合。

## 修订记录

| 发布日期   | 修订记录         |
| ---------- | ---------------- |
| 2026-09-30 | 第一次正式发布。新增TensorFlow FAGO静态图融合特性使用说明内容。 |
