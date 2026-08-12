import tensorflow as tf
import numpy as np
from framework.runner import TestCase


# -----------------------------------------------------------------------------
# 核心函数: 原始 5 算子 Attention 子图 (含 1/√d_k scale, 与 FlashAttention kernel 对齐)
#   BatchMatMulV2(Q, K, adj_y=True)  ->  Q·Kᵀ       : [h, s, s]
#   Mul(·, 1/√d_k)                    ->  (Q·Kᵀ)/√d_k : [h, s, s]
#   AddV2(·, mask)                    ->  S+mask       : [h, s, s]
#   Softmax(·)                        ->  P            : [h, s, s]
#   BatchMatMulV2(P, V, adj_y=False) ->  P·V          : [h, s, d_k]
#
# 融合 pass 会把该子图替换为:
#   FlashAttentionForward(q, k, v, mask) -> output, logsumexp
# (kernel 内部自带 1/√d_k scale, 故融合节点无需单独传 scale)
#
# 注意: 必须用 tf.compat.v1.raw_ops (而非 tf.raw_ops). 在 graph mode 下 tf.raw_ops
# 会把算子包进 PartitionedCall 子图, 导致 grappler 无法匹配 default graph 顶层的节点,
# is_fused_op_exist 和 extract_op_total_time 也无法按节点名找到算子.
# tf.compat.v1.raw_ops 直接在 default graph 顶层建图, 节点名可见可控.
# -----------------------------------------------------------------------------
@tf.function
def AttentionForwardRef(q, k, v, mask, scale):
    # Q·Kᵀ : adj_y=True 表示对 K 转置最后一维
    s = tf.compat.v1.raw_ops.BatchMatMulV2(x=q, y=k, adj_y=True)   # [h, s, s]
    s = tf.compat.v1.raw_ops.Mul(x=s, y=scale)                     # 1/√d_k (与 kernel 对齐)
    s = tf.compat.v1.raw_ops.AddV2(x=s, y=mask)                    # + additive mask
    p = tf.compat.v1.raw_ops.Softmax(logits=s)                     # [h, s, s]
    out = tf.compat.v1.raw_ops.BatchMatMulV2(x=p, y=v, adj_y=False)  # [h, s, d_k]
    return out


def FlashAttn_graph(input, meta):
    """根据核心函数入参名创建 placeholder, 返回结果与 feed_dict."""
    # 性能测试的 parse_performance_time 不调用 tf.reset_default_graph(),
    # 功能测试的 execute_variant 会先 reset 再调用本函数.
    # 在此统一 reset, 保证每次建图节点名始终为 PartitionedCall/... (无 _1 后缀),
    # 使 start_op_name / end_op_name 稳定可匹配.
    # tf.reset_default_graph()

    num_heads = meta["num_heads"]
    seq_len = meta["seq_len"]
    d_k = meta["d_k"]
    dtype = meta.get("dtype", np.float32)
    tf_dtype = tf.as_dtype(dtype)

    tf.compat.v1.reset_default_graph()

    # q/k/v: [num_heads, seq_len, d_k]
    q = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="q")
    k = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="k")
    v = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="v")
    # mask: [seq_len, seq_len], additive, 广播到 num_heads
    mask = tf.compat.v1.placeholder(tf_dtype, shape=[seq_len, seq_len], name="mask")
     # scale = 1/√d_k, 用 Const 节点 (pass 可识别 Const 标量, 且融合后 kernel
    # 内部已自带 1/√d_k scale, 该 Const 仅用于让 5-op 子图与 kernel 语义对齐).
    scale_val = np.array(1.0 / np.sqrt(d_k), dtype=dtype)
    scale = tf.constant(scale_val, dtype=tf_dtype, name="scale")

    feed_dict = {
        q: input["q"],
        k: input["k"],
        v: input["v"],
        mask: input["mask"],
    }

    result = AttentionForwardRef(q, k, v, mask, scale)
    return result, feed_dict

# =============================================================================
# 负向测试: 构造故意违反融合匹配条件的图, 验证融合 pass 正确拒绝
# =============================================================================
# --- 负向1: 缺少 Mul(scale), 直接用 AddV2(BMM, mask), 违反 MatchFwd 第2步 ---
# AddV2 的两个输入中找不到 Mul → 匹配失败
@tf.function
def _AttentionFwdNoScale(q, k, v, mask, scale):
    s = tf.compat.v1.raw_ops.BatchMatMulV2(x=q, y=k, adj_y=True)
    # 跳过 Mul(scale), 直接 AddV2
    s = tf.compat.v1.raw_ops.AddV2(x=s, y=mask)
    p = tf.compat.v1.raw_ops.Softmax(logits=s)
    out = tf.compat.v1.raw_ops.BatchMatMulV2(x=p, y=v, adj_y=False)
    return out
 
 
def FlashAttn_no_scale_graph(input, meta):
    num_heads, seq_len, d_k = meta["num_heads"], meta["seq_len"], meta["d_k"]
    dtype = meta.get("dtype", np.float32)
    tf_dtype = tf.as_dtype(dtype)
    tf.compat.v1.reset_default_graph()
    q = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="q")
    k = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="k")
    v = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="v")
    mask = tf.compat.v1.placeholder(tf_dtype, shape=[seq_len, seq_len], name="mask")
    scale = tf.constant(np.array(1.0 / np.sqrt(d_k), dtype=dtype), dtype=tf_dtype, name="scale")
    feed_dict = {q: input["q"], k: input["k"], v: input["v"], mask: input["mask"]}
    result = _AttentionFwdNoScale(q, k, v, mask, scale)
    return result, feed_dict
 
 
# --- 负向2: 缺少 AddV2(mask), 直接用 Softmax(Mul), 违反 MatchFwd 第1步 ---
# Softmax 的输入不是 AddV2 → 匹配失败
@tf.function
def _AttentionFwdNoMask(q, k, v, mask, scale):
    s = tf.compat.v1.raw_ops.BatchMatMulV2(x=q, y=k, adj_y=True)
    s = tf.compat.v1.raw_ops.Mul(x=s, y=scale)
    # 跳过 AddV2(mask), 直接 Softmax
    p = tf.compat.v1.raw_ops.Softmax(logits=s)
    out = tf.compat.v1.raw_ops.BatchMatMulV2(x=p, y=v, adj_y=False)
    return out
 
 
def FlashAttn_no_mask_graph(input, meta):
    num_heads, seq_len, d_k = meta["num_heads"], meta["seq_len"], meta["d_k"]
    dtype = meta.get("dtype", np.float32)
    tf_dtype = tf.as_dtype(dtype)
    tf.compat.v1.reset_default_graph()
    q = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="q")
    k = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="k")
    v = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="v")
    mask = tf.compat.v1.placeholder(tf_dtype, shape=[seq_len, seq_len], name="mask")
    scale = tf.constant(np.array(1.0 / np.sqrt(d_k), dtype=dtype), dtype=tf_dtype, name="scale")
    feed_dict = {q: input["q"], k: input["k"], v: input["v"], mask: input["mask"]}
    result = _AttentionFwdNoMask(q, k, v, mask, scale)
    return result, feed_dict

# --- 负向3: 缺少Softmax(Mul), 违反 MatchFwd 第1步 ---
# BatchMatMul 的输入不是 Softmax → 匹配失败
@tf.function
def _AttentionFwdNoSoftMax(q, k, v, mask, scale):
    s = tf.compat.v1.raw_ops.BatchMatMulV2(x=q, y=k, adj_y=True)   # [h, s, s]
    s = tf.compat.v1.raw_ops.Mul(x=s, y=scale)                     # 1/√d_k (与 kernel 对齐)
    s = tf.compat.v1.raw_ops.AddV2(x=s, y=mask)                    # + additive mask
    # 跳过SoftMax
    out = tf.compat.v1.raw_ops.BatchMatMulV2(x=s, y=v, adj_y=False)  # [h, s, d_k]
    return out
 
 
def FlashAttn_no_softmax_graph(input, meta):
    num_heads, seq_len, d_k = meta["num_heads"], meta["seq_len"], meta["d_k"]
    dtype = meta.get("dtype", np.float32)
    tf_dtype = tf.as_dtype(dtype)
    tf.compat.v1.reset_default_graph()
    q = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="q")
    k = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="k")
    v = tf.compat.v1.placeholder(tf_dtype, shape=[num_heads, seq_len, d_k], name="v")
    mask = tf.compat.v1.placeholder(tf_dtype, shape=[seq_len, seq_len], name="mask")
    scale = tf.constant(np.array(1.0 / np.sqrt(d_k), dtype=dtype), dtype=tf_dtype, name="scale")
    feed_dict = {q: input["q"], k: input["k"], v: input["v"], mask: input["mask"]}
    result = _AttentionFwdNoSoftMax(q, k, v, mask, scale)
    return result, feed_dict
# -----------------------------------------------------------------------------
# 输入构造
# -----------------------------------------------------------------------------
def _xavier_randn(shape, dtype, rng):
    """Xavier 初始化的正态分布, 对齐 C++ MakeRandom 的幅度."""
    fan_in = shape[0]
    fan_out = int(np.prod(shape)) // fan_in if fan_in > 0 else 1
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    arr = (rng.randn(*shape) * stddev).astype(np.float32)
    return arr.astype(dtype)

def _build_mask(seq_len, use_mask, dtype):
    """构造 additive mask (向量化实现, 适配大 seq_len=4096).
    use_mask=False: 全 0 (不屏蔽任何位置)
    use_mask=True : 因果 mask, 上三角为 -1e9 (大负有限值, 符合算子 doc 约束)
    """
    if not use_mask:
        return np.zeros([seq_len, seq_len], dtype=dtype)
    # 向量化因果 mask: 上三角(k=1)填 -1e9
    mask = np.triu(np.full([seq_len, seq_len], -1e9, dtype=np.float32), k=1)
    return mask.astype(dtype)


def build_input(meta):
    """通用输入构造函数, 由 meta 控制 d_k / dtype / mask."""
    dtype = meta.get("dtype", np.float32)
    use_mask = meta.get("use_mask", True)
    h, s, d = meta["num_heads"], meta["seq_len"], meta["d_k"]
    seed = meta["d_k"] * 1000 + meta["num_heads"]
    rng = np.random.RandomState(seed)
    input = {
        "q": _xavier_randn([h, s, d], dtype, rng),
        "k": _xavier_randn([h, s, d], dtype, rng),
        "v": _xavier_randn([h, s, d], dtype, rng),
        "mask": _build_mask(s, use_mask=use_mask, dtype=dtype),
    }
    return input

# -----------------------------------------------------------------------------
# 数值校验: FlashAttention kernel 内部使用 WMMA fp16-fragment + fp32
# accumulator 做矩阵乘, 参考子图用纯 fp32 BatchMatMulV2. 两者存在 ~1e-3
# 量级的固有精度差 (fp32→fp16 转换引入), 非逻辑错误.
# 框架默认容差 rtol=1e-3, atol=1e-4 对纯 fp32 计算合理, 但对 WMMA fp16
# compute 过严, 故此处提供适配容差.
# -----------------------------------------------------------------------------
# def check_fn_flash_attn(A, B, meta):
#     if len(A) != len(B):
#         return False
#     for i in range(len(A)):
#         np.testing.assert_allclose(A[i], B[i], rtol=1e-4, atol=1e-5)
#     return True

# -----------------------------------------------------------------------------
# TestCase 列表
#
# 测试矩阵: d_k ∈ {4, 8, 16, 32, 64}, seq_len=4096, 均属前向 tier 1 (d_k<=96)
# 所有 d_k 均跑 fp32 + 融合; 选 d_k=64 额外跑 fp16 与不融合对照.
#
# fused_op_name    : 融合后算子名 "FlashAttentionForward"
# start_op_name    : 融合前子图起始节点 (BatchMatMulV2: Q·Kᵀ)
# end_op_name      : 融合前子图终止节点 (BatchMatMulV2: P·V)
# is_fused=True    : 启用 flash_attn 融合 pass
# is_fused=False   : 不融合, 作为对照基准
# -----------------------------------------------------------------------------
def get_test_cases():
    SEQ_LEN = 4096
    NUM_HEADS = 8
    DK_LIST = [4, 8, 16, 32, 64]

    cases = []

    # ---- 负向功能测试: 故意违反融合条件, 预期融合不通过，仅进行功能测试 ----
    # 负向1: 缺少 Mul(scale) → MatchFwd 第2步
    cases.append(TestCase(
        name="FlashAttn_fwd_dk64_fail_no_scale",
        op_fn=FlashAttn_no_scale_graph,
        input_fn=build_input,
        fused_op_name="FlashAttentionForward",
        start_op_name="",
        end_op_name="",
        is_fused=False,  #该case预期融合为FALSE
        num_iters=0,
        optimize_percent=0,
        meta={"num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "d_k": 64,
              "dtype": np.float32, "use_mask": True},
    ))
 
    # 负向2: 缺少 AddV2(mask) → MatchFwd 第1步
    cases.append(TestCase(
        name="FlashAttn_fwd_dk64_fail_no_mask",
        op_fn=FlashAttn_no_mask_graph,
        input_fn=build_input,
        fused_op_name="FlashAttentionForward",
        start_op_name="",
        end_op_name="",
        is_fused=False,   #该case预期融合为FALSE
        num_iters=0,
        optimize_percent=0,
        meta={"num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "d_k": 64,
              "dtype": np.float32, "use_mask": True},
    ))

    # 负向3: 缺少 SoftMax
    cases.append(TestCase(
        name="FlashAttn_fwd_dk64_fail_no_softmax",
        op_fn=FlashAttn_no_softmax_graph,
        input_fn=build_input,
        fused_op_name="FlashAttentionForward",
        start_op_name="",
        end_op_name="",
        is_fused=False,   #该case预期融合为FALSE
        num_iters=0,
        optimize_percent=0,
        meta={"num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "d_k": 64,
                "dtype": np.float32, "use_mask": True},
    ))

    # 正向功能测试：完整图
    cases.append(TestCase(
        name=f"FlashAttn_fwd_dk64_s{SEQ_LEN}_causal_fused",
        op_fn=FlashAttn_graph,
        input_fn=build_input,
        fused_op_name="FlashAttentionForward",
        start_op_name="PartitionedCall/BatchMatMulV2",
        end_op_name="PartitionedCall/BatchMatMulV2_1",
        is_fused=True,
        num_iters=0,
        optimize_percent=0,
        meta={"num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "d_k": 64,
            "dtype": np.float32, "use_mask": True},
    ))
    # ---- 性能测试：5 个 d_k 的 fp32 + 因果 mask + 融合 ----
    # 注: kernel 当前仅注册了 DT_FLOAT (见 flash_attn_op.cc:329 REGISTER_FLASH_ATTN_GPU(float)).
    # half/bfloat16 需在 .cu.cc 补 template 实例化并在 .cc 补 REGISTER 后才能启用.
    #
    # 性能对比:
    for d_k in DK_LIST:
        cases.append(TestCase(
            name=f"FlashAttn_fwd_dk{d_k}_s{SEQ_LEN}_causal_fused",
            op_fn=FlashAttn_graph,
            input_fn=build_input,
            fused_op_name="FlashAttentionForward",
            start_op_name="PartitionedCall/BatchMatMulV2",
            end_op_name="PartitionedCall/BatchMatMulV2_1",
            is_fused=True,
            num_iters=100,
            optimize_percent=50,
            meta={"num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "d_k": d_k,
                "dtype": np.float32, "use_mask": True},
        ))
        
    return cases
