/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// =============================================================================
// flash_attn_op_gpu.cu.cc
//
// Flash Attention forward + backward CUDA kernels (WMMA TF16-style fp16
// fragments + fp32 accumulators), packaged as TF 2.20 internal kernel
// launchers returning absl::Status.
//
// Forward auto-dispatches across three tiers based on d_k:
//   d_k in [48, 96]                 -> BLOCK 64x64, 4 warps, MIN_BLOCKS=1
//   d_k in [24, 48], SM_count >= 80 -> BLOCK 64x64, 4 warps, MIN_BLOCKS=1
//   d_k <= 128                      -> BLOCK 32x32, 2 warps, MIN_BLOCKS=2
//   d_k <= kFlashAttnFwdMaxD (512)  -> BLOCK 16x16, 1 warp,  MIN_BLOCKS=4
//
// Forward additionally applies Split-KV when the GPU is under-occupied
// (heuristic target >= 6 full waves, S <= 8). num_splits == 1 is byte-
// identical to the unsplit path; num_splits > 1 writes unnormalized partial
// sums (O_part / m_part / l_part) reduced by flash_attn_fwd_merge in fp32.
//
// Backward uses a single configuration (BLOCK 32x32, 2 warps) and is capped
// at kFlashAttnBwdMaxD (128). Both forward and backward build a mask
// tile-classification table (TILE_SKIP / FULL / PARTIAL) once per launch and
// skip fully-blocked tiles (block-sparse fast path).
//
// The device kernel bodies are byte-identical to the standalone external
// implementation that has already been validated against a numpy reference;
// only the host-side launchers were rewritten to use GpuLaunchKernel and
// return absl::Status with errors::Internal / InvalidArgument.
// =============================================================================

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/flash_attn_op.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <mma.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

using namespace nvcuda;  // brings in wmma::

// =============================================================================
// Mask tile classification (forward + backward shared).
//
// Built once per launch by fa_build_tile_map; tiles whose mask elements are
// all <= FLASH_ATTN_MASK_SKIP_THRESH are skipped entirely (exp(...) == 0).
// =============================================================================
enum : uint8_t { TILE_SKIP = 0, TILE_FULL = 1, TILE_PARTIAL = 2 };

constexpr float FLASH_ATTN_MASK_SKIP_THRESH = -1e7f;

// =============================================================================
// Forward kernel constants & SMEM size helper
// =============================================================================
namespace fa_fwd {

constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 16;
constexpr int WARP_SIZE = 32;

constexpr int SKEW_HALF = 8;                      /* +16B */
constexpr int SKEW_FP32 = 4;                      /* +16B */

/* SMEM 切分 (含 skew):
 *   sQ      [BLOCK_M, ld_h]  fp16    ld_h = d_k_padded + SKEW_HALF
 *   sK      [BLOCK_N, ld_h]  fp16
 *   sV      [BLOCK_N, ld_h]  fp16
 *   sS/sP   [BLOCK_M, ld_s]  fp32    ld_s = BLOCK_N + SKEW_FP32
 *   sO      [BLOCK_M, ld_o]  fp32    ld_o = d_k_padded + SKEW_FP32
 */
template <int BLOCK_M, int BLOCK_N>
__host__ __device__ inline int smem_bytes_for(int d_k_padded) {
    const int ld_h = d_k_padded + SKEW_HALF;
    const int ld_o = d_k_padded + SKEW_FP32;
    const int ld_s = BLOCK_N + SKEW_FP32;
    int sQ_b   = BLOCK_M * ld_h * (int)sizeof(__half);
    int sK_b   = BLOCK_N * ld_h * (int)sizeof(__half);
    int sV_b   = BLOCK_N * ld_h * (int)sizeof(__half);
    int sPS_b  = BLOCK_M * ld_s * (int)sizeof(float);
    int sO_b   = BLOCK_M * ld_o * (int)sizeof(float);
    int sMisc  = 3 * BLOCK_M * (int)sizeof(float);
    return sQ_b + sK_b + sV_b + sPS_b + sO_b + sMisc;
}

}  // namespace fa_fwd


static __device__ __forceinline__ void fa_fwd_load_tile_fp16(
    const float* __restrict__ src,
    __half*      __restrict__ dst,
    const int row_start, const int block_rows,
    const int seq_len, const int d_k, const int d_k_padded,
    const int ld_h, const int head_offset,
    const int tid, const int num_threads)
{
    if ((d_k & 1) == 0) {
        const int half_cols = d_k_padded / 2;
        for (int idx = tid; idx < block_rows * half_cols; idx += num_threads) {
            int r  = idx / half_cols;
            int c  = (idx % half_cols) * 2;
            int gr = row_start + r;
            float2 v = make_float2(0.f, 0.f);
            if (gr < seq_len && c + 1 < d_k)
                v = *reinterpret_cast<const float2*>(
                        &src[head_offset + gr * d_k + c]);
            else if (gr < seq_len && c < d_k)
                v.x = src[head_offset + gr * d_k + c];
            *reinterpret_cast<__half2*>(&dst[r * ld_h + c]) =
                __floats2half2_rn(v.x, v.y);
        }
    } else {
        for (int idx = tid; idx < block_rows * d_k_padded; idx += num_threads) {
            int r  = idx / d_k_padded;
            int c  = idx % d_k_padded;
            int gr = row_start + r;
            float v = 0.f;
            if (gr < seq_len && c < d_k)
                v = src[head_offset + gr * d_k + c];
            dst[r * ld_h + c] = __float2half(v);
        }
    }
}

static __device__ __forceinline__ void fa_fwd_load_tile_pair_fp16(
    const float* __restrict__ srcA,
    const float* __restrict__ srcB,
    __half*      __restrict__ dstA,
    __half*      __restrict__ dstB,
    const int row_start, const int block_rows,
    const int seq_len, const int d_k, const int d_k_padded,
    const int ld_h, const int head_offset,
    const int tid, const int num_threads)
{
    if ((d_k & 1) == 0) {
        const int half_cols = d_k_padded / 2;
        for (int idx = tid; idx < block_rows * half_cols; idx += num_threads) {
            int r  = idx / half_cols;
            int c  = (idx % half_cols) * 2;
            int gr = row_start + r;
            float2 va = make_float2(0.f, 0.f);
            float2 vb = make_float2(0.f, 0.f);
            if (gr < seq_len && c + 1 < d_k) {
                va = *reinterpret_cast<const float2*>(
                         &srcA[head_offset + gr * d_k + c]);
                vb = *reinterpret_cast<const float2*>(
                         &srcB[head_offset + gr * d_k + c]);
            } else if (gr < seq_len && c < d_k) {
                va.x = srcA[head_offset + gr * d_k + c];
                vb.x = srcB[head_offset + gr * d_k + c];
            }
            *reinterpret_cast<__half2*>(&dstA[r * ld_h + c]) =
                __floats2half2_rn(va.x, va.y);
            *reinterpret_cast<__half2*>(&dstB[r * ld_h + c]) =
                __floats2half2_rn(vb.x, vb.y);
        }
    } else {
        for (int idx = tid; idx < block_rows * d_k_padded; idx += num_threads) {
            int r  = idx / d_k_padded;
            int c  = idx % d_k_padded;
            int gr = row_start + r;
            float va = 0.f, vb = 0.f;
            if (gr < seq_len && c < d_k) {
                va = srcA[head_offset + gr * d_k + c];
                vb = srcB[head_offset + gr * d_k + c];
            }
            dstA[r * ld_h + c] = __float2half(va);
            dstB[r * ld_h + c] = __float2half(vb);
        }
    }
}

/* ========================================================================
 * mask tile 分类
 *
 * grid = (q_tiles, kv_tiles), 每 block 归约一个 tile 的 min/max:
 *   所有元素 <= FLASH_ATTN_MASK_SKIP_THRESH -> TILE_SKIP
 *   所有元素 == 0.0f                        -> TILE_FULL
 *   其它                                    -> TILE_PARTIAL
 * ======================================================================== */
__global__ void fa_build_tile_map(
    const float* __restrict__ mask,
    const int seq_len,
    const int tile_m,
    const int tile_n,
    const int num_kv_tiles,
    uint8_t* __restrict__ tile_map)
{
    const int q_tile  = blockIdx.x;
    const int kv_tile = blockIdx.y;
    const int tid     = threadIdx.x;

    const int row_start = q_tile  * tile_m;
    const int col_start = kv_tile * tile_n;

    /* 逐 tile 归约 min/max (越界元素不参与判定) */
    float t_min =  INFINITY;
    float t_max = -INFINITY;
    const int tile_elems = tile_m * tile_n;
    for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
        int r = row_start + idx / tile_n;
        int c = col_start + idx % tile_n;      /* 连续 c -> 合并读 */
        if (r < seq_len && c < seq_len) {
            float v = mask[r * seq_len + c];
            t_min = fminf(t_min, v);
            t_max = fmaxf(t_max, v);
        }
    }

    /* warp 内归约 */
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        t_min = fminf(t_min, __shfl_down_sync(0xffffffff, t_min, off));
        t_max = fmaxf(t_max, __shfl_down_sync(0xffffffff, t_max, off));
    }

    /* warp 间归约 (blockDim.x = 128 -> 4 warps) */
    __shared__ float s_min[4];
    __shared__ float s_max[4];
    const int warp_id = tid / 32;
    if (tid % 32 == 0) {
        s_min[warp_id] = t_min;
        s_max[warp_id] = t_max;
    }
    __syncthreads();

    if (tid == 0) {
        float blk_min = s_min[0];
        float blk_max = s_max[0];
        for (int w = 1; w < (int)blockDim.x / 32; w++) {
            blk_min = fminf(blk_min, s_min[w]);
            blk_max = fmaxf(blk_max, s_max[w]);
        }
        uint8_t tt;
        if (blk_max <= FLASH_ATTN_MASK_SKIP_THRESH)
            tt = TILE_SKIP;                 /* exp 后必为 0, 整 tile 不算 */
        else if (blk_min == 0.f && blk_max == 0.f)
            tt = TILE_FULL;                 /* 无需做 mask 加法, 也不读 mask */
        else
            tt = TILE_PARTIAL;              /* 需要逐元素叠加 mask */
        tile_map[q_tile * num_kv_tiles + kv_tile] = tt;
    }
}

// -----------------------------------------------------------------------------
// Host launcher for fa_build_tile_map (shared by forward + backward).
// -----------------------------------------------------------------------------
static Status LaunchFaBuildTileMap(gpuStream_t stream,
                                   const float* mask, int seq_len,
                                   int tile_m, int tile_n,
                                   uint8_t* tile_map) {
    const int num_q_tiles  = (seq_len + tile_m - 1) / tile_m;
    const int num_kv_tiles = (seq_len + tile_n - 1) / tile_n;
    dim3 g(num_q_tiles, num_kv_tiles);
    dim3 b(128);
    return GpuLaunchKernel(fa_build_tile_map, g, b, /*smem=*/0, stream,
                           mask, seq_len, tile_m, tile_n, num_kv_tiles, tile_map);
}

/* ========================================================================
 * tier 选择
 * ======================================================================== */
static inline void fa_fwd_tier_block(int d_k, int* bm, int* bn)
{
    if (d_k >= 48 && d_k <= 96) {
        /* Tier 0: 推荐系统 d_k∈{48,64,96}
         * d_k<48 时 BLOCK 64x64 的 SMEM 占用影响 Tensor Core 利用率,
         * 下沉到 Tier 1 */
        *bm = 64; *bn = 64;
    } else if (d_k <= 128) {
        /* Tier 1: d_k∈(0,48)∪(96,128] - 保证小 d_k 的 occupancy, 适配 Qwen3 dense */
        *bm = 32; *bn = 32;
    } else {
        /* Tier 2: d_k <= 512, 适配 Qwen3.5/3.6 Gated Attention */
        *bm = 16; *bn = 16;
    }
}

/* ========================================================================
 * Split-KV 启发式 (改动点5 落地)
 *   目标: ≥6 个满 wave, S 上限 8. S==1 时与原前向逐字节一致 (4050 零变化).
 * ======================================================================== */
static inline int fa_fwd_approx_blocks_per_sm(int bm) {
    /* 各 tier 在 smem 限制下近似驻留 block/SM (Tier1=5 由 ncu 实测) */
    if (bm == 64) return 2;       /* Tier 0 */
    if (bm == 32) return 5;       /* Tier 1 */
    return 8;                     /* Tier 2 */
}

static int FaChooseNumSplits(int d_k, int seq_len, int num_heads, int sm_count) {
    if (sm_count <= 0) return 1;
    int bm = 0, bn = 0;
    fa_fwd_tier_block(d_k, &bm, &bn);
    const int num_q_blocks = (seq_len + bm - 1) / bm;
    const int occ = fa_fwd_approx_blocks_per_sm(bm);
    const int64_t blocks_per_wave = (int64_t)sm_count * occ;
    int S = 1;
    while (S < 8 && (int64_t)num_q_blocks * num_heads * S < 6 * blocks_per_wave)
        S *= 2;
    return S;
}

/* ========================================================================
 * Split-KV merge kernel: 对每个 (head, row) 归约 S 个 split 的部分和
 *   m = max_s m_s,  l = Σ l_s·e^{m_s−m},  O = Σ sO_s·e^{m_s−m} / l,  L = m+log l
 *   全 fp32, 与完整 online softmax 数学等价 (不引入额外精度损失)
 * ======================================================================== */
__global__ void flash_attn_fwd_merge(
    const float* __restrict__ O_part,   /* [H, S, N, d_k] */
    const float* __restrict__ m_part,   /* [H, S, N] */
    const float* __restrict__ l_part,   /* [H, S, N] */
    float* __restrict__ O,              /* [H, N, d_k] */
    float* __restrict__ L,              /* [H, N] */
    int num_splits, int num_heads, int seq_len, int d_k)
{
    const int head = blockIdx.x;
    const int row  = blockIdx.y;
    const int tid  = threadIdx.x;
    if (row >= seq_len) return;
    (void)num_heads;  /* 仅作 grid.x 范围, 内部不直接用 */

    const size_t out_row   = (size_t)head * seq_len + row;     /* 输出 O/L 行偏移 [H*N] */
    const size_t head_step = (size_t)num_splits * seq_len;     /* partials 的 [H] 步长 = S*N */
    const size_t row_base  = (size_t)head * head_step + row;   /* head*S*N + row */

    /* 1. m = max over splits (仅 l>0 的 split 参与) */
    float m = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
        const size_t idx = row_base + (size_t)s * seq_len;     /* head*S*N + s*N + row */
        float ls = l_part[idx];
        if (ls > 0.f) m = fmaxf(m, m_part[idx]);
    }

    /* 2. 权重 w[s] = e^{m_s−m} (l<=0 的 split w=0, 不参与); l = Σ l_s·w[s] */
    float w[8];
    float l = 0.f;
    for (int s = 0; s < num_splits; s++) {
        const size_t idx = row_base + (size_t)s * seq_len;
        float ls = l_part[idx];
        float ws = (ls > 0.f) ? __expf(m_part[idx] - m) : 0.f;
        w[s] = ws;
        l += ls * ws;
    }

    /* 3. O[col] = Σ_s O_part[head][s][row][col] · w[s] / l */
    for (int c = tid; c < d_k; c += blockDim.x) {
        float o = 0.f;
        for (int s = 0; s < num_splits; s++) {
            if (w[s] != 0.f)
                o += O_part[(row_base + (size_t)s * seq_len) * d_k + c] * w[s];
        }
        O[out_row * d_k + c] = (l > 0.f) ? (o / l) : 0.f;
    }

    /* 4. L = m + log(l) */
    if (tid == 0)
        L[out_row] = (l > 0.f) ? (m + logf(l)) : -INFINITY;
}

size_t FlashAttnFwdWorkspaceBytes(int num_heads, int seq_len, int d_k, int sm_count)
{
    if (seq_len <= 0 || d_k <= 0) return 0;
    int bm = 0, bn = 0;
    fa_fwd_tier_block(d_k, &bm, &bn);
    const size_t num_q_tiles  = (size_t)((seq_len + bm - 1) / bm);
    const size_t num_kv_tiles = (size_t)((seq_len + bn - 1) / bn);
    const size_t tile_map_bytes = num_q_tiles * num_kv_tiles * sizeof(uint8_t);
    const size_t aligned_tm = (tile_map_bytes + 255) & ~size_t(255);  /* 256B 对齐 */
    const int S = FaChooseNumSplits(d_k, seq_len, num_heads, sm_count);
    if (S <= 1) return aligned_tm;
    const size_t o_part = (size_t)S * num_heads * seq_len * d_k * sizeof(float);
    const size_t m_part = (size_t)S * num_heads * seq_len * sizeof(float);
    const size_t l_part = (size_t)S * num_heads * seq_len * sizeof(float);
    return aligned_tm + o_part + m_part + l_part;
}

/* ========================================================================
 * 前向 kernel
 * ======================================================================== */
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
flash_attn_fwd_wmma_tmpl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    const uint8_t* __restrict__ tile_map,
    const int   seq_len,
    const int   d_k,
    const int   d_k_padded,
    const int   num_kd_tiles,
    const float scale,
    float* __restrict__ O,
    float* __restrict__ L,
    const int   num_splits,
    float* __restrict__ O_part,    /* [num_splits, num_heads, seq_len, d_k] (num_splits>1) */
    float* __restrict__ m_part,    /* [num_splits, num_heads, seq_len] */
    float* __restrict__ l_part)    /* [num_splits, num_heads, seq_len] */
{
    using namespace fa_fwd;

    static_assert(NUM_WARPS == BLOCK_M / WMMA_M, "warps must own 16 rows each");
    static_assert(BLOCK_N % WMMA_N == 0, "BLOCK_N must be mult of WMMA_N");
    static_assert(BLOCK_N % 2 == 0,      "BLOCK_N must be even");

    constexpr int NUM_THREADS    = NUM_WARPS * WARP_SIZE;
    constexpr int N_FRAGS_S      = BLOCK_N / WMMA_N;
    constexpr int N_KV_FRAGS     = BLOCK_N / WMMA_K;
    constexpr int HALF_BN        = BLOCK_N / 2;
    constexpr int ROWS_PER_WARP  = BLOCK_M / NUM_WARPS;   /* == 16 */

    constexpr int LD_S = BLOCK_N + SKEW_FP32;   /* sS 行距 (fp32) */
    constexpr int LD_P = BLOCK_N + SKEW_HALF;   /* sP 行距 (fp16, 复用 sS 内存) */
    const int ld_h = d_k_padded + SKEW_HALF;    /* sQ/sK/sV 行距 (fp16) */
    const int ld_o = d_k_padded + SKEW_FP32;    /* sO 行距 (fp32) */

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int q_block_idx = blockIdx.x;
    const int head        = blockIdx.y;

    const int head_offset = head * seq_len * d_k;
    const int q_row_start = q_block_idx * BLOCK_M;

    /* ---- SMEM 切分 ---- */
    extern __shared__ __align__(16) uint8_t fa_smem_raw[];
    uint8_t* p = fa_smem_raw;

    __half* sQ = reinterpret_cast<__half*>(p);
    p += BLOCK_M * ld_h * sizeof(__half);
    __half* sK = reinterpret_cast<__half*>(p);
    p += BLOCK_N * ld_h * sizeof(__half);
    __half* sV = reinterpret_cast<__half*>(p);
    p += BLOCK_N * ld_h * sizeof(__half);

    float*  sS = reinterpret_cast<float*>(p);
    __half* sP = reinterpret_cast<__half*>(p);   /* alias (行距 LD_P) */
    p += BLOCK_M * LD_S * sizeof(float);

    float* sO = reinterpret_cast<float*>(p);
    p += BLOCK_M * ld_o * sizeof(float);

    float* sm      = reinterpret_cast<float*>(p); p += BLOCK_M * sizeof(float);
    float* sl      = reinterpret_cast<float*>(p); p += BLOCK_M * sizeof(float);
    float* s_alpha = reinterpret_cast<float*>(p); /* 结束 */

    /* ---- Init: sO=0, sm=-INF, sl=0 ---- */
    for (int i = tid; i < BLOCK_M * ld_o; i += NUM_THREADS)
        sO[i] = 0.f;
    if (tid < BLOCK_M) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.f;
    }

    /* ---- Load Q tile fp32 -> fp16, 向量化 ---- */
    fa_fwd_load_tile_fp16(Q, sQ, q_row_start, BLOCK_M,
                          seq_len, d_k, d_k_padded, ld_h, head_offset,
                          tid, NUM_THREADS);
    __syncthreads();

    const int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;

    /* Split-KV: 本 block 只遍历 [kv_tile_begin, kv_tile_end). num_splits==1 时全覆盖 */
    const int split_idx     = blockIdx.z;
    const int kv_per_split  = (num_kv_tiles + num_splits - 1) / num_splits;
    const int kv_tile_begin = split_idx * kv_per_split;
    int       kv_tile_end   = kv_tile_begin + kv_per_split;
    if (kv_tile_end > num_kv_tiles) kv_tile_end = num_kv_tiles;

    const uint8_t* tmap = tile_map + q_block_idx * num_kv_tiles;

    for (int j = kv_tile_begin; j < kv_tile_end; j++) {
        const uint8_t tt = tmap[j];
        if (tt == TILE_SKIP) continue;   /* K/V 加载、S、softmax、PV 跳过 */

        const int kv_row_start = j * BLOCK_N;

        /* ---- Load K, V tile 向量化---- */
        fa_fwd_load_tile_pair_fp16(K, V, sK, sV, kv_row_start, BLOCK_N,
                                   seq_len, d_k, d_k_padded, ld_h, head_offset,
                                   tid, NUM_THREADS);
        __syncthreads();

        /* ================================================================
         * Step 1: S = Q @ K^T
         * ================================================================ */
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            s_frag[N_FRAGS_S];
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++)
            wmma::fill_fragment(s_frag[n], 0.f);

        for (int kd = 0; kd < num_kd_tiles; kd++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> q_frag;
            wmma::load_matrix_sync(q_frag,
                sQ + warp_id * WMMA_M * ld_h + kd * WMMA_K,
                ld_h);

            #pragma unroll
            for (int n = 0; n < N_FRAGS_S; n++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag,
                    sK + n * WMMA_N * ld_h + kd * WMMA_K,
                    ld_h);
                wmma::mma_sync(s_frag[n], q_frag, k_frag, s_frag[n]);
            }
        }

        /* scale + store 到 sS */
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++) {
            #pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; i++)
                s_frag[n].x[i] *= scale;
            wmma::store_matrix_sync(
                sS + warp_id * WMMA_M * LD_S + n * WMMA_N,
                s_frag[n], LD_S, wmma::mem_row_major);
        }
        __syncthreads();

        if (tt == TILE_PARTIAL) {
            for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += NUM_THREADS) {
                int r  = idx / BLOCK_N;
                int c  = idx % BLOCK_N;
                int gr = q_row_start  + r;
                int gc = kv_row_start + c;
                if (gr < seq_len && gc < seq_len)
                    sS[r * LD_S + c] += mask[gr * seq_len + gc]; /* 连续 gc -> 全合并 */
            }
            __syncthreads();
        }

        /* ================================================================
         * Step 2: Online softmax + β-fold into sP
         *   (mask 已折进 sS, 仅保留边界 -INF 判断)
         * ================================================================ */
        const int row       = tid / 2;
        const int col_group = tid % 2;
        const int col_start = col_group * HALF_BN;
        const int q_row = q_row_start + row;

        float my_vals[HALF_BN];
        float my_max = -INFINITY;

        #pragma unroll
        for (int c = 0; c < HALF_BN; c++) {
            int kv_col = kv_row_start + col_start + c;
            float s_val;
            if (kv_col < seq_len && q_row < seq_len) {
                s_val = sS[row * LD_S + col_start + c];
            } else {
                s_val = -INFINITY;
            }
            my_vals[c] = s_val;
            my_max = fmaxf(my_max, s_val);
        }
        float partner_max = __shfl_xor_sync(0xffffffff, my_max, 1);
        float row_max = fmaxf(my_max, partner_max);

        /* 计算 p_tilde = exp(s - row_max) */
        float my_sum = 0.f;
        #pragma unroll
        for (int c = 0; c < HALF_BN; c++) {
            my_vals[c] = __expf(my_vals[c] - row_max);
            my_sum   += my_vals[c];
        }
        float partner_sum = __shfl_xor_sync(0xffffffff, my_sum, 1);
        float row_sum = my_sum + partner_sum;

        /* 更新 m, l */
        float old_m = sm[row];
        float old_l = sl[row];
        float new_m = fmaxf(old_m, row_max);
        float alpha = __expf(old_m  - new_m);
        float beta  = __expf(row_max - new_m);
        float new_l = alpha * old_l + beta * row_sum;

        if (col_group == 0) {
            sm[row]      = new_m;
            sl[row]      = new_l;
            s_alpha[row] = alpha;
        }
        __syncthreads();

        /* 写 β·P 到 sP */
        #pragma unroll
        for (int c = 0; c < HALF_BN; c++) {
            sP[row * LD_P + col_start + c] = __float2half(beta * my_vals[c]);
        }

        /* ================================================================
         * Step 3: Pre-scale sO by α  (避免运行时除法)
         * ================================================================ */
        #pragma unroll
        for (int rr = 0; rr < ROWS_PER_WARP; rr++) {
            int r = warp_id * ROWS_PER_WARP + rr;
            float alpha_r = s_alpha[r];
            for (int c = lane_id; c < d_k_padded; c += WARP_SIZE)
                sO[r * ld_o + c] *= alpha_r;
        }
        __syncthreads();

        /* ================================================================
         * Step 4: O += β·P @ V
         * ================================================================ */
        for (int dt = 0; dt < num_kd_tiles; dt++) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
                o_frag;
            wmma::load_matrix_sync(o_frag,
                sO + warp_id * WMMA_M * ld_o + dt * WMMA_N,
                ld_o, wmma::mem_row_major);

            #pragma unroll
            for (int k = 0; k < N_KV_FRAGS; k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag,
                    sP + warp_id * WMMA_M * LD_P + k * WMMA_K,
                    LD_P);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag,
                    sV + k * WMMA_K * ld_h + dt * WMMA_N,
                    ld_h);

                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            wmma::store_matrix_sync(
                sO + warp_id * WMMA_M * ld_o + dt * WMMA_N,
                o_frag, ld_o, wmma::mem_row_major);
        }
        __syncthreads();
    }

    if (num_splits <= 1) {
        /* ---- 写回 O = sO / sl, L = m + log(sl)  (与原实现逐字节一致) ---- */
        for (int idx = tid; idx < BLOCK_M * d_k; idx += NUM_THREADS) {
            int r  = idx / d_k;
            int c  = idx % d_k;
            int gr = q_row_start + r;
            if (gr < seq_len) {
                float val = sO[r * ld_o + c] / sl[r];
                O[head_offset + gr * d_k + c] = val;
            }
        }
        if (tid < BLOCK_M) {
            int gr = q_row_start + tid;
            if (gr < seq_len)
                L[head * seq_len + gr] = sm[tid] + logf(sl[tid]);
        }
    } else {
        /* ---- Split-KV: 写未归一化部分和, 由 merge kernel 归一化 ----
         *   O_part[S][H][N][d_k] = sO (未除 sl), m_part/l_part = sm/sl */
        const size_t part_base = ((size_t)head * num_splits + split_idx) * seq_len;
        for (int idx = tid; idx < BLOCK_M * d_k; idx += NUM_THREADS) {
            int r  = idx / d_k;
            int c  = idx % d_k;
            int gr = q_row_start + r;
            if (gr < seq_len)
                O_part[(part_base + gr) * d_k + c] = sO[r * ld_o + c];
        }
        if (tid < BLOCK_M) {
            int gr = q_row_start + tid;
            if (gr < seq_len) {
                m_part[part_base + gr] = sm[tid];
                l_part[part_base + gr] = sl[tid];
            }
        }
    }
}

// =============================================================================
// Forward tier launcher helper
// =============================================================================
template <int BM, int BN, int NW, int MB>
static Status FwdLaunchTier(gpuStream_t stream,
                            const float* Q, const float* K, const float* V,
                            const float* mask, const uint8_t* tile_map,
                            float* O, float* L,
                            int num_heads, int seq_len,
                            int d_k, int d_k_padded,
                            int num_kd_tiles, float scale,
                            int num_splits,
                            float* O_part, float* m_part, float* l_part) {
  auto kernel = flash_attn_fwd_wmma_tmpl<BM, BN, NW, MB>;
  int smem = fa_fwd::smem_bytes_for<BM, BN>(d_k_padded);

  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) {
      return errors::Internal(
          "FlashAttn:fwd cudaFuncSetAttribute failed: ",
          cudaGetErrorString(e),
          " (BM=", BM, " BN=", BN, " d_k=", d_k, " smem=", smem, ")");
    }
  }

  const int num_q_blocks = (seq_len + BM - 1) / BM;
  dim3 grid(num_q_blocks, num_heads, num_splits);   // split in z (Split-KV)
  dim3 block(NW * 32);

  return GpuLaunchKernel(kernel, grid, block, smem, stream,
                         Q, K, V, mask, tile_map, seq_len, d_k, d_k_padded,
                         num_kd_tiles, scale, O, L,
                         num_splits, O_part, m_part, l_part);
}

// =============================================================================
// Public launcher: forward (float specialization)
// =============================================================================
template <>
Status LaunchFlashAttentionForward<float>(gpuStream_t stream,
                                          const float* q,
                                          const float* k,
                                          const float* v,
                                          const float* mask,
                                          float* output,
                                          float* logsumexp,
                                          void* workspace,
                                          int num_heads,
                                          int seq_len,
                                          int d_k,
                                          int sm_count) {
  using namespace fa_fwd;

  if (num_heads <= 0 || seq_len <= 0 || d_k <= 0) {
    return errors::InvalidArgument(
        "FlashAttn:fwd invalid args: num_heads=", num_heads,
        " seq_len=", seq_len, " d_k=", d_k);
  }
  if (workspace == nullptr) {
    return errors::InvalidArgument(
        "FlashAttn:fwd workspace is null (need FlashAttnFwdWorkspaceBytes(...) "
        "bytes on device)");
  }

  const int d_k_padded   = ((d_k + WMMA_K - 1) / WMMA_K) * WMMA_K;
  const int num_kd_tiles = d_k_padded / WMMA_K;

  if (d_k_padded > kFlashAttnFwdMaxD) {
    return errors::InvalidArgument(
        "FlashAttn:fwd unsupported d_k=", d_k,
        " (padded=", d_k_padded, " > kFlashAttnFwdMaxD=", kFlashAttnFwdMaxD,
        "). Raise kFlashAttnFwdMaxD in flash_attn_op.h to support larger d_k.");
  }

  int bm = 0, bn = 0;
  fa_fwd_tier_block(d_k, &bm, &bn);

  // Build the mask tile-classification table at the selected tier's tile size.
  uint8_t* tile_map = static_cast<uint8_t*>(workspace);
  TF_RETURN_IF_ERROR(
      LaunchFaBuildTileMap(stream, mask, seq_len, bm, bn, tile_map));

  // Split-KV: choose num_splits by sm_count; workspace layout
  // [tile_map | O_part | m_part | l_part] (256B-aligned tile_map prefix).
  const int num_splits =
      FaChooseNumSplits(d_k, seq_len, num_heads, sm_count);
  float* O_part = nullptr;
  float* m_part = nullptr;
  float* l_part = nullptr;
  if (num_splits > 1) {
    const size_t num_q_tiles  = (size_t)((seq_len + bm - 1) / bm);
    const size_t num_kv_tiles = (size_t)((seq_len + bn - 1) / bn);
    size_t tile_map_bytes = num_q_tiles * num_kv_tiles * sizeof(uint8_t);
    size_t aligned_tm = (tile_map_bytes + 255) & ~size_t(255);
    O_part = reinterpret_cast<float*>(tile_map + aligned_tm);
    m_part = O_part + (size_t)num_splits * num_heads * seq_len * d_k;
    l_part = m_part + (size_t)num_splits * num_heads * seq_len;
  }

  const float scale = 1.f / sqrtf(static_cast<float>(d_k));

  // Three-tier dispatch.
  if (bm == 64) {
    // Tier 0
    TF_RETURN_IF_ERROR(FwdLaunchTier<64, 64, 4, 1>(
        stream, q, k, v, mask, tile_map, output, logsumexp,
        num_heads, seq_len, d_k, d_k_padded, num_kd_tiles, scale,
        num_splits, O_part, m_part, l_part));
  } else if (bm == 32) {
    // Tier 1
    TF_RETURN_IF_ERROR(FwdLaunchTier<32, 32, 2, 2>(
        stream, q, k, v, mask, tile_map, output, logsumexp,
        num_heads, seq_len, d_k, d_k_padded, num_kd_tiles, scale,
        num_splits, O_part, m_part, l_part));
  } else {
    // Tier 2
    TF_RETURN_IF_ERROR(FwdLaunchTier<16, 16, 1, 4>(
        stream, q, k, v, mask, tile_map, output, logsumexp,
        num_heads, seq_len, d_k, d_k_padded, num_kd_tiles, scale,
        num_splits, O_part, m_part, l_part));
  }

  // Split-KV merge: normalize partial sums into final O / L.
  if (num_splits > 1) {
    dim3 mgrid(num_heads, seq_len);
    dim3 mblock(32);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        flash_attn_fwd_merge, mgrid, mblock, /*smem=*/0, stream,
        O_part, m_part, l_part, output, logsumexp,
        num_splits, num_heads, seq_len, d_k));
  }

  return absl::OkStatus();
}

// =============================================================================
// Backward kernel constants
// =============================================================================
namespace fa_bwd {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;

constexpr int WARP_SIZE   = 32;
constexpr int NUM_WARPS   = BLOCK_M / WMMA_M;        // 2
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;   // 64

constexpr int N_FRAGS_S  = BLOCK_N / WMMA_N;         // 2
constexpr int N_Q_FRAGS  = BLOCK_M / WMMA_K;         //2  (dV/dK 内层 K 循环)
constexpr int N_KV_FRAGS = BLOCK_N / WMMA_K;         // 2  (dQ 内层 K 循环)

/* dQ/dK/dV 的大小上限 */
constexpr int MAX_KD_TILES = kFlashAttnBwdMaxD / WMMA_K;

constexpr int SKEW_HALF = 8;                         // +16B
constexpr int SKEW_FP32 = 4;                         // +16B

constexpr int LD_S     = BLOCK_N + SKEW_FP32;        // 36: sS/sdP 行距 (fp32)
constexpr int LD_P     = BLOCK_N + SKEW_HALF;        // 40: sP/sdS 行距 (fp16, 复用 sS 内存)
constexpr int LD_STAGE = WMMA_N + SKEW_FP32;         // 20: 写回 staging 行距 (原 16, 消冲突)

/*  BLOCK_M*LD_P*2B (2560B) <= BLOCK_M*LD_S*4B (4608B) */
static_assert(BLOCK_M * LD_P * (int)sizeof(__half)
           <= BLOCK_M * LD_S * (int)sizeof(float),
              "sP/sdS alias must fit inside sS region");
/* NUM_WARPS*WMMA_M*LD_STAGE (640) <= BLOCK_M*LD_S (1152) */
static_assert(NUM_WARPS * WMMA_M * LD_STAGE <= BLOCK_M * LD_S,
              "write-back staging must fit inside sdP region");

} // namespace fa_bwd


static __device__ __forceinline__ void fa_bwd_load_tile_pair_fp16(
    const float* __restrict__ srcA,
    const float* __restrict__ srcB,
    __half*      __restrict__ dstA,
    __half*      __restrict__ dstB,
    const int row_start, const int block_rows,
    const int seq_len, const int d_k, const int d_k_padded,
    const int ld_h, const int head_offset,
    const int tid, const int num_threads)
{
    if ((d_k & 1) == 0) {
        const int half_cols = d_k_padded / 2;
        for (int idx = tid; idx < block_rows * half_cols; idx += num_threads) {
            int r  = idx / half_cols;
            int c  = (idx % half_cols) * 2;
            int gr = row_start + r;
            float2 va = make_float2(0.f, 0.f);
            float2 vb = make_float2(0.f, 0.f);
            if (gr < seq_len && c + 1 < d_k) {
                va = *reinterpret_cast<const float2*>(
                         &srcA[head_offset + gr * d_k + c]);
                vb = *reinterpret_cast<const float2*>(
                         &srcB[head_offset + gr * d_k + c]);
            } else if (gr < seq_len && c < d_k) {
                va.x = srcA[head_offset + gr * d_k + c];
                vb.x = srcB[head_offset + gr * d_k + c];
            }
            *reinterpret_cast<__half2*>(&dstA[r * ld_h + c]) =
                __floats2half2_rn(va.x, va.y);
            *reinterpret_cast<__half2*>(&dstB[r * ld_h + c]) =
                __floats2half2_rn(vb.x, vb.y);
        }
    } else {
        for (int idx = tid; idx < block_rows * d_k_padded; idx += num_threads) {
            int r  = idx / d_k_padded;
            int c  = idx % d_k_padded;
            int gr = row_start + r;
            float va = 0.f, vb = 0.f;
            if (gr < seq_len && c < d_k) {
                va = srcA[head_offset + gr * d_k + c];
                vb = srcB[head_offset + gr * d_k + c];
            }
            dstA[r * ld_h + c] = __float2half(va);
            dstB[r * ld_h + c] = __float2half(vb);
        }
    }
}

/* ========================================================================
 * Kernel 0: 基本 O(N·d) 操作, 暂时不用上 Tensor Core
 * ======================================================================== */
__global__ void flash_attn_precompute_D_wmma(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    const int N,
    const int d_k,
    float* __restrict__ D_buf)
{
    const int tid  = threadIdx.x;
    const int bq   = blockIdx.x;
    const int head = blockIdx.y;
    const int row  = bq * blockDim.x + tid;

    if (row >= N) return;

    const int offset = head * N * d_k + row * d_k;
    float d = 0.f;
    for (int x = 0; x < d_k; x++)
        d += dO[offset + x] * O[offset + x];
    D_buf[head * N + row] = d;
}

/* ========================================================================
 * 共享内存切分 (dQ kernel)
 *
 * 内存布局:
 *   sQ   [BLOCK_M, ld_h]  fp16    ld_h = d_k_padded + SKEW_HALF
 *   sdO  [BLOCK_M, ld_h]  fp16
 *   sK   [BLOCK_N, ld_h]  fp16
 *   sV   [BLOCK_N, ld_h]  fp16
 * ======================================================================== */
__host__ __device__ inline int fa_bwd_dq_smem_bytes(int d_k_padded)
{
    using namespace fa_bwd;
    const int ld_h = d_k_padded + SKEW_HALF;
    int s_qd_b   = 2 * BLOCK_M * ld_h * (int)sizeof(__half);   // sQ + sdO
    int s_kv_b   = 2 * BLOCK_N * ld_h * (int)sizeof(__half);   // sK + sV
    int s_spds_b = BLOCK_M * LD_S * (int)sizeof(float);         // sS/sP/sdS
    int s_dp_b   = BLOCK_M * LD_S * (int)sizeof(float);         // sdP
    int s_meta_b = 2 * BLOCK_M * (int)sizeof(float);            // sL + sD
    return s_qd_b + s_kv_b + s_spds_b + s_dp_b + s_meta_b;
}

/* ========================================================================
 * Kernel 1: 一个 block 处理一个 Q tile, 遍历所有 KV tiles，dQ 累加
 * ======================================================================== */
__global__ void __launch_bounds__(fa_bwd::NUM_THREADS, 2)
flash_attn_bwd_dq_wmma(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    const uint8_t* __restrict__ tile_map,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    const int   seq_len,
    const int   d_k,
    const int   d_k_padded,
    const int   num_kd_tiles,
    const float scale,
    float* __restrict__ dQ)
{
    using namespace fa_bwd;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int q_block_idx = blockIdx.x;
    const int head        = blockIdx.y;

    const int head_offset = head * seq_len * d_k;
    const int q_row_start = q_block_idx * BLOCK_M;

    const int ld_h = d_k_padded + SKEW_HALF;

    /* ---- 共享内存切分 ---- */
    extern __shared__ __align__(16) uint8_t fa_bwd_smem_raw[];
    uint8_t* p = fa_bwd_smem_raw;

    __half* sQ = reinterpret_cast<__half*>(p);
    p += BLOCK_M * ld_h * sizeof(__half);

    __half* sdO = reinterpret_cast<__half*>(p);
    p += BLOCK_M * ld_h * sizeof(__half);

    __half* sK = reinterpret_cast<__half*>(p);
    p += BLOCK_N * ld_h * sizeof(__half);

    __half* sV = reinterpret_cast<__half*>(p);
    p += BLOCK_N * ld_h * sizeof(__half);

    /* sS/sP/sdS 三者复用同一段, fp32 行距 LD_S, fp16 行距 LD_P */
    float*  sS  = reinterpret_cast<float*>(p);
    __half* sP  = reinterpret_cast<__half*>(p);   /* alias */
    __half* sdS = reinterpret_cast<__half*>(p);   /* alias */
    p += BLOCK_M * LD_S * sizeof(float);

    float* sdP = reinterpret_cast<float*>(p);
    p += BLOCK_M * LD_S * sizeof(float);

    float* sL = reinterpret_cast<float*>(p); p += BLOCK_M * sizeof(float);
    float* sD = reinterpret_cast<float*>(p); /*p += BLOCK_M * sizeof(float);*/

    /* ---- 加载 Q, dO (向量化) ---- */
    fa_bwd_load_tile_pair_fp16(Q, dO, sQ, sdO, q_row_start, BLOCK_M,
                               seq_len, d_k, d_k_padded, ld_h, head_offset,
                               tid, NUM_THREADS);
    /* 加载 L 和 D (每线程一行) */
    if (tid < BLOCK_M) {
        int gr = q_row_start + tid;
        if (gr < seq_len) {
            sL[tid] = L    [head * seq_len + gr];
            sD[tid] = D_buf[head * seq_len + gr];
        } else {
            sL[tid] = 0.f;
            sD[tid] = 0.f;
        }
    }
    __syncthreads();

    if (num_kd_tiles > MAX_KD_TILES) return;

    /* ---- dQ 累加 ---- */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        dq_acc[MAX_KD_TILES];
    #pragma unroll
    for (int dt = 0; dt < MAX_KD_TILES; dt++)
        wmma::fill_fragment(dq_acc[dt], 0.f);

    /* ---- 主循环: 遍历所有 KV ---- */
    const int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;

    /* 本 Q tile 对应的分类表行 */
    const uint8_t* tmap = tile_map + q_block_idx * num_kv_tiles;

    for (int j = 0; j < num_kv_tiles; j++) {
        const uint8_t tt = tmap[j];
        if (tt == TILE_SKIP) continue;   /* P=0 ⇒ dS=0, 整 tile 对 dQ 无贡献 */

        const int kv_row_start = j * BLOCK_N;

        /* ---- 加载 K, V tile (向量化) ---- */
        fa_bwd_load_tile_pair_fp16(K, V, sK, sV, kv_row_start, BLOCK_N,
                                   seq_len, d_k, d_k_padded, ld_h, head_offset,
                                   tid, NUM_THREADS);
        __syncthreads();

        /* ================================================================
         * Step A: dP = dO @ V^T
         * ================================================================ */
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            dp_frag[N_FRAGS_S];
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++)
            wmma::fill_fragment(dp_frag[n], 0.f);

        for (int kd = 0; kd < num_kd_tiles; kd++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> do_frag;
            wmma::load_matrix_sync(do_frag,
                sdO + warp_id * WMMA_M * ld_h + kd * WMMA_K,
                ld_h);

            #pragma unroll
            for (int n = 0; n < N_FRAGS_S; n++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> v_frag;
                wmma::load_matrix_sync(v_frag,
                    sV + n * WMMA_N * ld_h + kd * WMMA_K,
                    ld_h);
                wmma::mma_sync(dp_frag[n], do_frag, v_frag, dp_frag[n]);
            }
        }
        /* 把 dP 存到 sdP (fp32) */
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++) {
            wmma::store_matrix_sync(
                sdP + warp_id * WMMA_M * LD_S + n * WMMA_N,
                dp_frag[n], LD_S, wmma::mem_row_major);
        }

        /* ================================================================
         * Step B: S = Q @ K^T
         * ================================================================ */
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            s_frag[N_FRAGS_S];
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++)
            wmma::fill_fragment(s_frag[n], 0.f);

        for (int kd = 0; kd < num_kd_tiles; kd++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> q_frag;
            wmma::load_matrix_sync(q_frag,
                sQ + warp_id * WMMA_M * ld_h + kd * WMMA_K,
                ld_h);

            #pragma unroll
            for (int n = 0; n < N_FRAGS_S; n++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag,
                    sK + n * WMMA_N * ld_h + kd * WMMA_K,
                    ld_h);
                wmma::mma_sync(s_frag[n], q_frag, k_frag, s_frag[n]);
            }
        }
        /* scale + store 到 sS */
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++) {
            #pragma unroll
            for (int e = 0; e < s_frag[n].num_elements; e++)
                s_frag[n].x[e] *= scale;
            wmma::store_matrix_sync(
                sS + warp_id * WMMA_M * LD_S + n * WMMA_N,
                s_frag[n], LD_S, wmma::mem_row_major);
        }
        __syncthreads();


        if (tt == TILE_PARTIAL) {
            for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += NUM_THREADS) {
                int r  = idx / BLOCK_N;
                int c  = idx % BLOCK_N;
                int gr = q_row_start  + r;
                int gc = kv_row_start + c;
                if (gr < seq_len && gc < seq_len)
                    sS[r * LD_S + c] += mask[gr * seq_len + gc]; /* 连续 gc -> 全合并 */
            }
            __syncthreads();
        }

        /* ================================================================
         * Step C: P = exp(S - L), dS = P * (dP - D), 写入 sdS (fp16)
         *   (mask 已折进 sS; ★ 改动点7: expf -> __expf, 与前向一致)
         * ================================================================ */
        const int row       = tid / 2;
        const int col_group = tid % 2;
        const int col_start = col_group * (BLOCK_N / 2);
        const int q_row     = q_row_start + row;
        const float my_L = sL[row];
        const float my_D = sD[row];

        float P_local[BLOCK_N / 2];
        #pragma unroll
        for (int c = 0; c < BLOCK_N / 2; c++) {
            int kv_col = kv_row_start + col_start + c;
            if (kv_col < seq_len && q_row < seq_len) {
                P_local[c] = __expf(sS[row * LD_S + col_start + c] - my_L);
            } else {
                P_local[c] = 0.f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < BLOCK_N / 2; c++) {
            float dp = sdP[row * LD_S + col_start + c];
            float ds = P_local[c] * (dp - my_D);
            sdS[row * LD_P + col_start + c] = __float2half(ds);
        }
        __syncthreads();
        /* ================================================================
         * Step D: dQ += dS @ K
         * ================================================================ */
        for (int dt = 0; dt < num_kd_tiles; dt++) {
            #pragma unroll
            for (int k = 0; k < N_KV_FRAGS; k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> ds_frag;
                wmma::load_matrix_sync(ds_frag,
                    sdS + warp_id * WMMA_M * LD_P + k * WMMA_K,
                    LD_P);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> k_frag_b;
                wmma::load_matrix_sync(k_frag_b,
                    sK + k * WMMA_K * ld_h + dt * WMMA_N,
                    ld_h);

                wmma::mma_sync(dq_acc[dt], ds_frag, k_frag_b, dq_acc[dt]);
            }
        }
        __syncthreads();
    }

    /* ---- 写回 dQ (staging 行距 LD_STAGE, 消写回 bank conflict) ---- */
    for (int dt = 0; dt < num_kd_tiles; dt++) {
        #pragma unroll
        for (int e = 0; e < dq_acc[dt].num_elements; e++)
            dq_acc[dt].x[e] *= scale;

        wmma::store_matrix_sync(
            sdP + warp_id * WMMA_M * LD_STAGE,
            dq_acc[dt], LD_STAGE, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int local_idx = e * WARP_SIZE + lane_id;
            int lr = local_idx / WMMA_N;
            int lc = local_idx % WMMA_N;
            int gr = q_row_start + warp_id * WMMA_M + lr;
            int gc = dt * WMMA_N + lc;
            if (gr < seq_len && gc < d_k) {
                float val = sdP[warp_id * WMMA_M * LD_STAGE + lr * LD_STAGE + lc];
                dQ[head_offset + gr * d_k + gc] = val;
            }
        }
        __syncthreads();
    }
}

/* ========================================================================
 * 共享内存切分 (dKV kernel)
 * ======================================================================== */
__host__ __device__ inline int fa_bwd_dkv_smem_bytes(int d_k_padded)
{
    return fa_bwd_dq_smem_bytes(d_k_padded);
}

/* ========================================================================
 * Kernel 2: dK / dV kernel，一个 block 处理一个 KV tile, 遍历所有 Q tile
 * ======================================================================== */
__global__ void __launch_bounds__(fa_bwd::NUM_THREADS, 2)
flash_attn_bwd_dkv_wmma(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    const uint8_t* __restrict__ tile_map,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    const int   seq_len,
    const int   d_k,
    const int   d_k_padded,
    const int   num_kd_tiles,
    const float scale,
    float* __restrict__ dK,
    float* __restrict__ dV)
{
    using namespace fa_bwd;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int kv_block_idx = blockIdx.x;
    const int head         = blockIdx.y;

    const int head_offset   = head * seq_len * d_k;
    const int kv_row_start  = kv_block_idx * BLOCK_N;

    /* fp16 tile 行距 (skew) */
    const int ld_h = d_k_padded + SKEW_HALF;

    /* ---- 共享内存切分 ---- */
    extern __shared__ __align__(16) uint8_t fa_dkv_smem_raw[];
    uint8_t* p = fa_dkv_smem_raw;

    __half* sQ = reinterpret_cast<__half*>(p);
    p += BLOCK_M * ld_h * sizeof(__half);

    __half* sdO = reinterpret_cast<__half*>(p);
    p += BLOCK_M * ld_h * sizeof(__half);

    __half* sK = reinterpret_cast<__half*>(p);
    p += BLOCK_N * ld_h * sizeof(__half);

    __half* sV = reinterpret_cast<__half*>(p);
    p += BLOCK_N * ld_h * sizeof(__half);

    float*  sS  = reinterpret_cast<float*>(p);
    __half* sP  = reinterpret_cast<__half*>(p);   /* alias */
    __half* sdS = reinterpret_cast<__half*>(p);   /* alias */
    p += BLOCK_M * LD_S * sizeof(float);

    float* sdP = reinterpret_cast<float*>(p);
    p += BLOCK_M * LD_S * sizeof(float);

    float* sL = reinterpret_cast<float*>(p);
    p += BLOCK_M * sizeof(float);
    float* sD = reinterpret_cast<float*>(p);

    /* ---- 加载 K, V tile (向量化) ---- */
    fa_bwd_load_tile_pair_fp16(K, V, sK, sV, kv_row_start, BLOCK_N,
                               seq_len, d_k, d_k_padded, ld_h, head_offset,
                               tid, NUM_THREADS);
    __syncthreads();

    /* ---- dK, dV 累加 ---- */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        dk_acc[MAX_KD_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        dv_acc[MAX_KD_TILES];
    #pragma unroll
    for (int dt = 0; dt < MAX_KD_TILES; dt++) {
        wmma::fill_fragment(dk_acc[dt], 0.f);
        wmma::fill_fragment(dv_acc[dt], 0.f);
    }

    /* ---- 主循环: 遍历所有 Q tile ---- */
    const int num_q_tiles  = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;   /* tile map 列数 */

    for (int i = 0; i < num_q_tiles; i++) {
        const uint8_t tt = tile_map[i * num_kv_tiles + kv_block_idx];
        if (tt == TILE_SKIP) continue;   /* P=0 ⇒ 对 dK/dV 无贡献, 整 tile 跳过 */

        const int q_row_start = i * BLOCK_M;

        /* 加载 Q, dO, L, D */
        fa_bwd_load_tile_pair_fp16(Q, dO, sQ, sdO, q_row_start, BLOCK_M,
                                   seq_len, d_k, d_k_padded, ld_h, head_offset,
                                   tid, NUM_THREADS);
        if (tid < BLOCK_M) {
            int gr = q_row_start + tid;
            if (gr < seq_len) {
                sL[tid] = L    [head * seq_len + gr];
                sD[tid] = D_buf[head * seq_len + gr];
            } else {
                sL[tid] = 0.f;
                sD[tid] = 0.f;
            }
        }
        __syncthreads();

        /* ---- Step A: dP = dO @ V^T (WMMA) ---- */
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            dp_frag[N_FRAGS_S];
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++)
            wmma::fill_fragment(dp_frag[n], 0.f);

        for (int kd = 0; kd < num_kd_tiles; kd++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> do_frag;
            wmma::load_matrix_sync(do_frag,
                sdO + warp_id * WMMA_M * ld_h + kd * WMMA_K,
                ld_h);

            #pragma unroll
            for (int n = 0; n < N_FRAGS_S; n++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> v_frag;
                wmma::load_matrix_sync(v_frag,
                    sV + n * WMMA_N * ld_h + kd * WMMA_K,
                    ld_h);
                wmma::mma_sync(dp_frag[n], do_frag, v_frag, dp_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++) {
            wmma::store_matrix_sync(
                sdP + warp_id * WMMA_M * LD_S + n * WMMA_N,
                dp_frag[n], LD_S, wmma::mem_row_major);
        }

        /* ---- Step B: S = Q @ K^T (WMMA) ---- */
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            s_frag[N_FRAGS_S];
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++)
            wmma::fill_fragment(s_frag[n], 0.f);

        for (int kd = 0; kd < num_kd_tiles; kd++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> q_frag;
            wmma::load_matrix_sync(q_frag,
                sQ + warp_id * WMMA_M * ld_h + kd * WMMA_K,
                ld_h);

            #pragma unroll
            for (int n = 0; n < N_FRAGS_S; n++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag,
                    sK + n * WMMA_N * ld_h + kd * WMMA_K,
                    ld_h);
                wmma::mma_sync(s_frag[n], q_frag, k_frag, s_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < N_FRAGS_S; n++) {
            #pragma unroll
            for (int e = 0; e < s_frag[n].num_elements; e++)
                s_frag[n].x[e] *= scale;
            wmma::store_matrix_sync(
                sS + warp_id * WMMA_M * LD_S + n * WMMA_N,
                s_frag[n], LD_S, wmma::mem_row_major);
        }
        __syncthreads();

        /* ================================================================
         * ★ TILE_PARTIAL 时把 mask 以合并方式折进 sS
         * ================================================================ */
        if (tt == TILE_PARTIAL) {
            for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += NUM_THREADS) {
                int r  = idx / BLOCK_N;
                int c  = idx % BLOCK_N;
                int gr = q_row_start  + r;
                int gc = kv_row_start + c;
                if (gr < seq_len && gc < seq_len)
                    sS[r * LD_S + c] += mask[gr * seq_len + gc]; /* 连续 gc -> 全合并 */
            }
            __syncthreads();
        }

        /* ---- Step C: 把 S -> P, 写入 sP (fp16)
         *   (mask 已折进 sS; ★ 改动点7: expf -> __expf) */
        const int row       = tid / 2;
        const int col_group = tid % 2;
        const int col_start = col_group * (BLOCK_N / 2);
        const int q_row     = q_row_start + row;
        const float my_L = sL[row];
        const float my_D = sD[row];

        float P_local[BLOCK_N / 2];
        #pragma unroll
        for (int c = 0; c < BLOCK_N / 2; c++) {
            int kv_col = kv_row_start + col_start + c;
            if (kv_col < seq_len && q_row < seq_len) {
                P_local[c] = __expf(sS[row * LD_S + col_start + c] - my_L);
            } else {
                P_local[c] = 0.f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < BLOCK_N / 2; c++) {
            sP[row * LD_P + col_start + c] = __float2half(P_local[c]);
        }
        __syncthreads();

        /* ================================================================
         * Step D: dV += P^T @ dO  (WMMA)
         * dV 的输出维度: [BLOCK_N, d_k_padded]
         * ================================================================ */
        for (int dt = 0; dt < num_kd_tiles; dt++) {
            #pragma unroll
            for (int k = 0; k < N_Q_FRAGS; k++) {
                /* P^T 的 fragment: A 矩阵, col_major */
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> pt_frag;
                wmma::load_matrix_sync(pt_frag,
                    sP + k * WMMA_K * LD_P + warp_id * WMMA_M,
                    LD_P);

                /* dO: B 矩阵, row_major */
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> do_frag;
                wmma::load_matrix_sync(do_frag,
                    sdO + k * WMMA_K * ld_h + dt * WMMA_N,
                    ld_h);

                wmma::mma_sync(dv_acc[dt], pt_frag, do_frag, dv_acc[dt]);
            }
        }
        /* sP 与 sdS 别名复用同一段 smem: 需等所有 warp 读完 sP 再写 sdS
         * (原实现此处缺 barrier, 属既有竞争, 本次一并修复) */
        __syncthreads();

        /* ================================================================
         * Step E: dS = P * (dP - D), 写入 sdS (fp16)
         * ================================================================ */
        #pragma unroll
        for (int c = 0; c < BLOCK_N / 2; c++) {
            float dp = sdP[row * LD_S + col_start + c];
            float ds = P_local[c] * (dp - my_D);
            sdS[row * LD_P + col_start + c] = __float2half(ds);
        }
        __syncthreads();

        /* ================================================================
         * Step F: dK += dS^T @ Q  (col_major)
         * ================================================================ */
        for (int dt = 0; dt < num_kd_tiles; dt++) {
            #pragma unroll
            for (int k = 0; k < N_Q_FRAGS; k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> dst_frag;
                wmma::load_matrix_sync(dst_frag,
                    sdS + k * WMMA_K * LD_P + warp_id * WMMA_M,
                    LD_P);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> q_frag;
                wmma::load_matrix_sync(q_frag,
                    sQ + k * WMMA_K * ld_h + dt * WMMA_N,
                    ld_h);

                wmma::mma_sync(dk_acc[dt], dst_frag, q_frag, dk_acc[dt]);
            }
        }
        __syncthreads();
    }

    /* ---- 写回 dK 和 dV ---- */
    for (int dt = 0; dt < num_kd_tiles; dt++) {
        /* ---- dK ---- */
        #pragma unroll
        for (int e = 0; e < dk_acc[dt].num_elements; e++)
            dk_acc[dt].x[e] *= scale;

        wmma::store_matrix_sync(
            sdP + warp_id * WMMA_M * LD_STAGE,
            dk_acc[dt], LD_STAGE, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int local_idx = e * WARP_SIZE + lane_id;
            int lr = local_idx / WMMA_N;
            int lc = local_idx % WMMA_N;
            int gr = kv_row_start + warp_id * WMMA_M + lr;
            int gc = dt * WMMA_N + lc;
            if (gr < seq_len && gc < d_k) {
                float val = sdP[warp_id * WMMA_M * LD_STAGE + lr * LD_STAGE + lc];
                dK[head_offset + gr * d_k + gc] = val;
            }
        }
        __syncthreads();

        /* ---- dV ---- */
        wmma::store_matrix_sync(
            sdP + warp_id * WMMA_M * LD_STAGE,
            dv_acc[dt], LD_STAGE, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int local_idx = e * WARP_SIZE + lane_id;
            int lr = local_idx / WMMA_N;
            int lc = local_idx % WMMA_N;
            int gr = kv_row_start + warp_id * WMMA_M + lr;
            int gc = dt * WMMA_N + lc;
            if (gr < seq_len && gc < d_k) {
                float val = sdP[warp_id * WMMA_M * LD_STAGE + lr * LD_STAGE + lc];
                dV[head_offset + gr * d_k + gc] = val;
            }
        }
        __syncthreads();
    }
}

size_t FlashAttnBwdWorkspaceBytes(int seq_len)
{
    using namespace fa_bwd;
    if (seq_len <= 0) return 0;
    const size_t num_q_tiles  = (size_t)((seq_len + BLOCK_M - 1) / BLOCK_M);
    const size_t num_kv_tiles = (size_t)((seq_len + BLOCK_N - 1) / BLOCK_N);
    return num_q_tiles * num_kv_tiles * sizeof(uint8_t);
}

// =============================================================================
// Public launcher: backward (float specialization)
// =============================================================================
template <>
Status LaunchFlashAttentionBackward<float>(gpuStream_t stream,
                                           const float* q,
                                           const float* k,
                                           const float* v,
                                           const float* mask,
                                           const float* o,
                                           const float* grad_o,
                                           const float* logsumexp,
                                           float* grad_q,
                                           float* grad_k,
                                           float* grad_v,
                                           float* d_buf,
                                           void* workspace,
                                           int num_heads,
                                           int seq_len,
                                           int d_k) {
  using namespace fa_bwd;

  if (num_heads <= 0 || seq_len <= 0 || d_k <= 0) {
    return errors::InvalidArgument(
        "FlashAttn:bwd invalid args: num_heads=", num_heads,
        " seq_len=", seq_len, " d_k=", d_k);
  }

  const int d_k_padded   = ((d_k + WMMA_K - 1) / WMMA_K) * WMMA_K;
  const int num_kd_tiles = d_k_padded / WMMA_K;

  if (d_k_padded > kFlashAttnBwdMaxD) {
    return errors::InvalidArgument(
        "FlashAttn:bwd unsupported d_k=", d_k,
        " (padded=", d_k_padded, " > kFlashAttnBwdMaxD=", kFlashAttnBwdMaxD,
        "). Backward kernel uses fixed-size accumulators; raise "
        "kFlashAttnBwdMaxD in flash_attn_op.h to extend.");
  }
  if (workspace == nullptr) {
    return errors::InvalidArgument(
        "FlashAttn:bwd workspace is null (need FlashAttnBwdWorkspaceBytes("
        "seq_len) bytes on device)");
  }

  const float scale       = 1.f / sqrtf(static_cast<float>(d_k));
  const int num_q_blocks  = (seq_len + BLOCK_M - 1) / BLOCK_M;
  const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

  uint8_t* tile_map = static_cast<uint8_t*>(workspace);

  // ---- Step 1: precompute D_i = sum_x dO * O ----
  {
    const int kThreads = 32;
    dim3 grid((seq_len + kThreads - 1) / kThreads, num_heads);
    dim3 block(kThreads);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        flash_attn_precompute_D_wmma, grid, block, /*smem=*/0, stream,
        grad_o, o, seq_len, d_k, d_buf));
  }

  // ---- Step 1.5: build mask tile-classification table (dQ + dKV shared) ----
  TF_RETURN_IF_ERROR(
      LaunchFaBuildTileMap(stream, mask, seq_len, BLOCK_M, BLOCK_N, tile_map));

  // ---- Step 2: dQ kernel ----
  {
    int smem = fa_bwd_dq_smem_bytes(d_k_padded);
    if (smem > 48 * 1024) {
      cudaError_t e = cudaFuncSetAttribute(
          flash_attn_bwd_dq_wmma,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      if (e != cudaSuccess) {
        return errors::Internal(
            "FlashAttn:bwd:dq cudaFuncSetAttribute failed: ",
            cudaGetErrorString(e), " (smem=", smem, ")");
      }
    }
    dim3 grid(num_q_blocks, num_heads);
    dim3 block(NUM_THREADS);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        flash_attn_bwd_dq_wmma, grid, block, smem, stream,
        q, k, v, mask, tile_map, grad_o, logsumexp, d_buf,
        seq_len, d_k, d_k_padded, num_kd_tiles, scale, grad_q));
  }

  // ---- Step 3: dK / dV kernel ----
  {
    int smem = fa_bwd_dkv_smem_bytes(d_k_padded);
    if (smem > 48 * 1024) {
      cudaError_t e = cudaFuncSetAttribute(
          flash_attn_bwd_dkv_wmma,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      if (e != cudaSuccess) {
        return errors::Internal(
            "FlashAttn:bwd:dkv cudaFuncSetAttribute failed: ",
            cudaGetErrorString(e), " (smem=", smem, ")");
      }
    }
    dim3 grid(num_kv_blocks, num_heads);
    dim3 block(NUM_THREADS);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        flash_attn_bwd_dkv_wmma, grid, block, smem, stream,
        q, k, v, mask, tile_map, grad_o, logsumexp, d_buf,
        seq_len, d_k, d_k_padded, num_kd_tiles, scale, grad_k, grad_v));
  }

  return absl::OkStatus();
}

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
