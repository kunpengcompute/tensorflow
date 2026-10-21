# API参考

## TensorFlow ANNC静态图融合特性使用说明

ANNC 静态图融合包含 CPU 算子和 GPU 算子。**GPU 算子需要 CUDA 环境支持，CPU 算子无此要求**。

TensorFlow ANNC静态图融合特性开关通过环境变量开关控制。所有算子共用总开关`ANNC_FUSED_ALL`，总开关与各算子开关是"或"的关系：两者任一开启（取值为1）即启用对应算子的融合，仅当两者均未开启时该算子才不进行融合。特性开关默认取值为0，即关闭ANNC静态图融合功能，如需使用需要手动在图优化前设置环境变量开启，以Python语言为例可以通过如下方式开启全部融合算子（含CPU和GPU算子）。

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
```

也可以只开启指定算子，例如仅开启GPU算子FlashAttentionForward的融合。

```python
import os
os.environ['ANNC_FUSED_FLASHATTN_FWD'] = '1'
```

**表 1**  ANNC 静态图融合算子开关<a id="table473618378219"></a>

| 开关名 | 类型 | 取值 | 算子类型 | 功能 |
| -------- | ------------ | --------------- | -------- | ------------------ |
| ANNC_FUSED_EMB_ACTIONID_GATHER | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedEmbeddingActionIdGather算子开启ANNC静态图融合。 |
| ANNC_FUSED_GATHER | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedGather算子开启ANNC静态图融合。 |
| ANNC_FUSED_EMD_PADDING | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedEmbeddingPadding算子开启ANNC静态图融合。 |
| ANNC_FUSED_EMD_PADDING_FAST | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedEmbeddingPaddingFast算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_STITCH | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedSparseDynamicStitch算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_STITCH_REDUCE | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedSparseDynamicStitchKPFusedSparseDynamicStitch(Fast)\[Sum\|Mean\|SqrtN\]算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_RESHAPE | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedSparseReshape算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_REDUCE | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedSparseSegmentReduce算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_REDUCE_NONZERO | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedSparseSegmentReduceNonzero算子开启ANNC静态图融合。 |
| ANNC_FUSED_SPS_SELECT | 进程环境变量 | 1：开启 0：关闭 | CPU | 用于KPFusedSparseSelect算子开启ANNC静态图融合。 |
| ANNC_FUSED_FLASHATTN_FWD | 进程环境变量 | 1：开启 0：关闭 | GPU | 用于FlashAttentionForward算子开启ANNC静态图融合，需CUDA环境支持。 |
| ANNC_FUSED_ALL | 进程环境变量 | 1：开启 0：关闭 | CPU/GPU共用 | 总开关，用于开启所有ANNC融合算子（包括CPU和GPU算子）。 |

> ![icon note](public_sys-resources/icon-note.gif) **说明：**
> 各算子当且仅当`ANNC_FUSED_ALL`与该算子对应的环境变量均未开启（取值不为1）时，该算子不会进行算子融合，两者任一开启即启用融合。GPU算子还需运行环境具备CUDA支持。

## TensorFlow KDNN算子优化特性使用说明

TensorFlow KDNN算子优化特性通过进程环境变量控制。

- KDNN特性**默认关闭**，需手动设置环境变量开启。
- 稀疏矩阵乘稀疏转稠密开关是KDNN特性内的子开关，仅在KDNN特性开启时生效，此特性**默认开启**，即在满足稀疏度条件时自动转稠密计算。

**使用示例**

首次调用KDNN算子前设置进程环境变量，以Python语言为例：

```python
import os
# 开启KDNN特性
os.environ['TF_ENABLE_KDNN_OPTS'] = '1'
# 可选：关闭稀疏矩阵乘稀疏转稠密（默认为开启）
os.environ['TF_ENABLE_KDNN_SPARSE_MATMUL_TO_DENSE'] = '0'
```

**表 2**  KDNN算子优化特性开关<a id="table473618378220"></a>

| 开关名 | 类型 | 取值 | 默认状态 | 功能 |
| -------- | ------------ | --------------- | ------ | ------------------ |
| TF_ENABLE_KDNN_OPTS | 进程环境变量 | 0：关闭<br>1：开启 | 关闭 | KDNN算子优化特性总开关，控制KDNN优化特性的启用。 |
| TF_ENABLE_KDNN_SPARSE_MATMUL_TO_DENSE | 进程环境变量 | 0：关闭<br>1：开启<br>其他：按默认值1处理 | 开启 | 控制SparseMatmul算子的转稠密计算路径开启。 |

> ![icon note](public_sys-resources/icon-note.gif) **说明：**
> 稀疏矩阵乘稀疏转稠密开关是KDNN特性的子开关，仅在`TF_ENABLE_KDNN_OPTS`取值为1时生效。

## KEmbedding算子库EmbeddingTableLookup算子说明

EmbeddingTableLookup算子是KEmbedding算子库中的一个自定义算子，用于高效执行稀疏Embedding查找操作。

**接口描述**

从资源表`EmbeddingIndexToValueTable`中按key查找稀疏Embedding，并输出标准SparseTensor三元组。

* `indices`：包含命中非零元素的坐标。
* `values`：包含对应的embedding值。
* `dense_shape`：包含完整embedding矩阵的形状。

**接口类型**

TensorFlow OpKernel类。

**输入参数**

| 参数名称 | 类型 | 描述 |
| ---------------- | ------------- | ------ |
| keys | int64张量 | 需要查找的Embedding key列表，形状为`[key_cnt]`。 |
| table_handle | resource | 资源表句柄，通过`EmbeddingIndexToValueTable`创建并已加载Embedding数据。 |

**输出参数**

| 参数名称 | 类型 | 描述 |
| --------------- | ------------- | ------------- |
| indices | int64张量 | SparseTensor的索引，形状为`[N, 2]`，N为命中非零元素总数，第二维为`[row, col]`。 |
| values | float张量 | SparseTensor的值，形状为`[N]`，与indices一一对应。 |
| dense_shape | int64张量 | 稠密形状，形状为`[2]`，值为`[key_cnt, emb_dim]`。 |

**关键属性**

| 属性名称 | 描述 |
| ----------- | --------- |
| emb_dim | Embedding维度，用于构造输出的dense_shape。 |

**接口源码文件**

`third_party/kembedding/src/kernels/embedding_table_lookup_op.cc`

**其他相关算子**

* `EmbeddingIndexToValueTable`：创建资源表句柄。
* `InitializeEmbeddingIndexToValueTableFromTextFile`：从二进制文件初始化资源表。
  
  该接口名称中保留`TextFile`是历史命名，实际读取的输入文件不是文本文件，而是下文所述的 kembedding二进制表文件。

**使用示例**

```python
import tensorflow as tf

# 加载 kembedding 自定义算子动态库
kembedding_module = tf.load_op_library(
    'path/to/bazel-bin/third_party/kembedding/kembedding_embedding_table_lookup.so'
)

# 创建资源表并初始化
with tf.compat.v1.Session() as sess:
    # 步骤1：创建资源表句柄
    table_handle = kembedding_module.embedding_index_to_value_table()

    # 步骤2：从 kembedding 二进制表文件加载 embedding 表数据。
    # 注意：接口名包含 text_file 是历史命名，filename 应传入二进制表文件路径。
    sess.run(
        kembedding_module.initialize_embedding_index_to_value_table_from_text_file(
            table_handle=table_handle,
            filename='path/to/embedding_table.bin'
        )
    )

    # 步骤3：执行批量查找
    keys = tf.constant([101, 202, 999], dtype=tf.int64)
    indices, values, dense_shape = kembedding_module.embedding_table_lookup(
        table_handle=table_handle,
        keys=keys,
        emb_dim=4
    )

    # 获取结果
    result_indices, result_values, result_shape = sess.run(
        [indices, values, dense_shape]
    )

    # 结果示例：
    # indices = [[0, 0], [0, 2], [1, 1]]
    # values = [1.0, 3.0, 2.5]
    # dense_shape = [3, 4]
```

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 02 | 2026-12-30 | 第二次正式发布。<ul><li>新增TensorFlow KDNN算子优化特性使用说明内容。</li><li>新增KEmbedding自定义算子库，提供EmbeddingTableLookup算子说明内容。</li><li>新增TensorFlow ANNC静态图融合特性使用说明内容。</li></ul> |
| 01 | 2026-09-30 | 第一次正式发布。 <ul><li>新增TensorFlow FAGO静态图融合特性使用说明内容。</li></ul> |
