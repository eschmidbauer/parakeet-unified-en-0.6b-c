/*  encoder.c – FastConformer encoder forward pass. */

#include "parakeet.h"
#include "blas.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define PK_ENC_NEON 1
#endif
#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define PK_ENC_AVX2 1
#endif

int pk_verbose = 1;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static float *alloc_f(int n)
{
    return (float *)calloc(n, sizeof(float));
}

/* ── Bump-allocator workspace for conformer blocks ───────────────────────── */

typedef struct {
    float *buf;
    size_t cap;   /* capacity in floats */
} CfmWS;

/* Carve n floats from the workspace. Not freed individually — the caller
 * logically resets (off=0) between sub-functions. */
typedef struct { float *buf; size_t off; } WSA;

static float *ws_get(WSA *a, size_t n)
{
    float *p = a->buf + a->off;
    a->off += n;
    return p;
}

/* Compute the peak workspace (in floats) for a single conformer sub-function.
 * self_attention dominates; feed_forward and conv_module are smaller. */
static size_t conformer_ws_peak(int T)
{
    int D = PK_D_MODEL, H = PK_N_HEADS, Dh = PK_HEAD_DIM, FF = PK_FF_DIM;
    int L = 2 * T - 1;

    /* self_attention peak (all bump-allocated buffers simultaneously): */
    size_t sa = 0;
    sa += (size_t)T * D;           /* ln */
    sa += (size_t)T * 3 * D;      /* QKV (fused) */
    sa += (size_t)H * T * Dh * 2; /* Q_u, Q_v */
    sa += (size_t)L * D;           /* P */
    sa += (size_t)H * T * Dh * 2; /* K_h, V_h */
    sa += (size_t)H * L * Dh;     /* P_h */
    sa += (size_t)H * T * T;      /* ac (also reused as scores) */
    sa += (size_t)H * T * L;      /* bd */
    sa += (size_t)H * T * (2*T);  /* relative_shift padded */
    sa += (size_t)H * T * T;      /* relative_shift result */
    sa += (size_t)H * T * Dh;     /* attn_h */
    sa += (size_t)T * D;           /* attn */
    sa += (size_t)T * D;           /* out */

    /* feed_forward peak: */
    size_t ff = (size_t)T * D + (size_t)T * FF + (size_t)T * D;

    /* conv_module peak: */
    size_t cm = (size_t)T * D      /* ln */
              + (size_t)D * T      /* cf */
              + (size_t)2 * D * T  /* pw1 */
              + (size_t)D * T      /* gated */
              + (size_t)D * T      /* dw */
              + (size_t)D * T;     /* pw2 */

    size_t peak = sa;
    if (ff > peak) peak = ff;
    if (cm > peak) peak = cm;
    return peak;
}

/* ── Pre-encode: convolutional subsampling ────────────────────────────────── */

/*  Input:  mel [128 × T]   (channel-first)
 *  Output: out [1024 × T'] where T' = T / 8 (approximately)
 *
 *  Pipeline (ONNX graph order):
 *   1. Transpose mel [B, 128, T] → [B, T, 128]
 *   2. Unsqueeze → [B, 1, T, 128]  (Conv2D spatial: H=T, W=128)
 *   3. Conv2D [256,1,3,3] stride=2 pad=1  → [256, T/2, 64]    + ReLU
 *   4. DWConv2D [256,1,3,3] stride=2 groups=256
 *      + PWConv2D [256,256,1,1]           → [256, T/4, 32]    + ReLU
 *   5. DWConv2D [256,1,3,3] stride=2 groups=256
 *      + PWConv2D [256,256,1,1]           → [256, T/8, 16]    + ReLU
 *   6. Reshape to [B, T', 256*16] = [B, T', 4096]
 *   7. Linear(4096,1024) → [B, T', 1024]
 */
static int pre_encode(const PreEncodeWeights *w,
                      const float *mel, int T,
                      float **out_ptr)
{
    int H = T;
    int W = PK_N_MELS;
    int C = 1;

    float *mel_t = alloc_f(T * PK_N_MELS);
    for (int t = 0; t < T; t++)
        for (int m = 0; m < PK_N_MELS; m++)
            mel_t[t * PK_N_MELS + m] = mel[m * T + t];

    int H1 = (H + 2 - 3) / 2 + 1;
    int W1 = (W + 2 - 3) / 2 + 1;
    float *s1 = alloc_f(PK_SUB_CHANNELS * H1 * W1);
    pk_conv2d(mel_t, w->conv0_w, w->conv0_b, s1,
              C, H, W, PK_SUB_CHANNELS, 3, 3, 2, 2, 1, 1, 1);
    free(mel_t);
    pk_relu(s1, PK_SUB_CHANNELS * H1 * W1);

    int H2 = (H1 + 2 - 3) / 2 + 1;
    int W2 = (W1 + 2 - 3) / 2 + 1;
    float *s2_dw = alloc_f(PK_SUB_CHANNELS * H2 * W2);
    pk_conv2d(s1, w->conv2_w, w->conv2_b, s2_dw,
              PK_SUB_CHANNELS, H1, W1, PK_SUB_CHANNELS, 3, 3, 2, 2, 1, 1,
              PK_SUB_CHANNELS);
    free(s1);

    float *s2 = alloc_f(PK_SUB_CHANNELS * H2 * W2);
    pk_conv2d(s2_dw, w->conv3_w, w->conv3_b, s2,
              PK_SUB_CHANNELS, H2, W2, PK_SUB_CHANNELS, 1, 1, 1, 1, 0, 0, 1);
    free(s2_dw);
    pk_relu(s2, PK_SUB_CHANNELS * H2 * W2);

    int H3 = (H2 + 2 - 3) / 2 + 1;
    int W3 = (W2 + 2 - 3) / 2 + 1;
    float *s3_dw = alloc_f(PK_SUB_CHANNELS * H3 * W3);
    pk_conv2d(s2, w->conv5_w, w->conv5_b, s3_dw,
              PK_SUB_CHANNELS, H2, W2, PK_SUB_CHANNELS, 3, 3, 2, 2, 1, 1,
              PK_SUB_CHANNELS);
    free(s2);

    float *s3 = alloc_f(PK_SUB_CHANNELS * H3 * W3);
    pk_conv2d(s3_dw, w->conv6_w, w->conv6_b, s3,
              PK_SUB_CHANNELS, H3, W3, PK_SUB_CHANNELS, 1, 1, 1, 1, 0, 0, 1);
    free(s3_dw);
    pk_relu(s3, PK_SUB_CHANNELS * H3 * W3);

    int Tp = H3;
    int feat_dim = PK_SUB_CHANNELS * W3;
    float *flat = alloc_f(Tp * feat_dim);
    for (int c = 0; c < PK_SUB_CHANNELS; c++)
        for (int h = 0; h < H3; h++)
            for (int w3 = 0; w3 < W3; w3++)
                flat[h * feat_dim + c * W3 + w3] = s3[c * H3 * W3 + h * W3 + w3];
    free(s3);

    float *proj = alloc_f(Tp * PK_D_MODEL);
    pk_wmatmul(flat, &w->out_w, proj, Tp, PK_D_MODEL, feat_dim, 1.0f, 0.0f);
    free(flat);
    for (int t = 0; t < Tp; t++)
        pk_bias_add(proj + t * PK_D_MODEL, w->out_b, PK_D_MODEL);

    float *result = alloc_f(PK_D_MODEL * Tp);
    for (int d = 0; d < PK_D_MODEL; d++)
        for (int t = 0; t < Tp; t++)
            result[d * Tp + t] = proj[t * PK_D_MODEL + d];
    free(proj);

    *out_ptr = result;
    return Tp;
}

/* ── Feed-forward module ─────────────────────────────────────────────────── */

static void feed_forward(const float *norm_w, const float *norm_b,
                         const FFWeights *ff,
                         float *x, int T, CfmWS *ws)
{
    int D = PK_D_MODEL;
    int FF = PK_FF_DIM;
    WSA a = {ws->buf, 0};

    float *ln  = ws_get(&a, T * D);
    float *h   = ws_get(&a, T * FF);
    float *out = ws_get(&a, T * D);

    pk_layer_norm_rows(x, norm_w, norm_b, ln, T, D);

    pk_wmatmul(ln, &ff->linear1_w, h, T, FF, D, 1.0f, 0.0f);
    for (int t = 0; t < T; t++)
        pk_bias_add(h + t * FF, ff->linear1_b, FF);
    pk_swish(h, T * FF);

    pk_wmatmul(h, &ff->linear2_w, out, T, D, FF, 1.0f, 0.0f);
    for (int t = 0; t < T; t++)
        pk_bias_add(out + t * D, ff->linear2_b, D);

    {
        int n = T * D, i = 0;
#if PK_ENC_NEON
        float32x4_t half = vdupq_n_f32(0.5f);
        for (; i + 4 <= n; i += 4)
            vst1q_f32(x + i, vfmaq_f32(vld1q_f32(x + i), vld1q_f32(out + i), half));
#elif PK_ENC_AVX2
        __m256 half = _mm256_set1_ps(0.5f);
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(x + i, _mm256_fmadd_ps(
                _mm256_loadu_ps(out + i), half, _mm256_loadu_ps(x + i)));
#endif
        for (; i < n; i++)
            x[i] += 0.5f * out[i];
    }
}

/* ── Relative positional encoding ────────────────────────────────────────── */

static float *slice_pos_encoding(const float *pe_table, int T, int D)
{
    int start = PK_PE_CENTER - T + 1;
    int len = 2 * T - 1;
    float *pe = alloc_f(len * D);
    memcpy(pe, pe_table + start * D, len * D * sizeof(float));
    return pe;
}

static void relative_shift(float *bd, int H, int T, WSA *a)
{
    int L = 2 * T - 1;
    int Lp = 2 * T;

    float *padded = ws_get(a, H * T * Lp);
    for (int h = 0; h < H; h++) {
        for (int r = 0; r < T; r++) {
            padded[h * T * Lp + r * Lp + 0] = 0.0f;
            memcpy(padded + h * T * Lp + r * Lp + 1,
                   bd + h * T * L + r * L,
                   L * sizeof(float));
        }
    }

    float *result = ws_get(a, H * T * T);
    for (int h = 0; h < H; h++) {
        const float *src = padded + h * T * Lp + T;
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                result[h * T * T + i * T + j] = src[i * L + j];
    }

    memcpy(bd, result, H * T * T * sizeof(float));
}

/* ── Batched per-head GEMM ───────────────────────────────────────────────── */

struct batched_head_args {
    int H;
    CBLAS_TRANSPOSE transB;
    int M, N, K;
    const float *A; int strideA, lda;
    const float *B; int strideB, ldb;
    float       *C; int strideC, ldc;
};

static void batched_head_worker(int tid, int nt, void *aux)
{
    struct batched_head_args *a = (struct batched_head_args *)aux;
    int h0 = (a->H * tid) / nt;
    int h1 = (a->H * (tid + 1)) / nt;
    for (int h = h0; h < h1; h++)
        cblas_sgemm_st(CblasRowMajor, CblasNoTrans, a->transB,
                       a->M, a->N, a->K, 1.0f,
                       a->A + (size_t)h * a->strideA, a->lda,
                       a->B + (size_t)h * a->strideB, a->ldb,
                       0.0f, a->C + (size_t)h * a->strideC, a->ldc);
}

static void batched_head_gemm(int H, CBLAS_TRANSPOSE transB,
                              int M, int N, int K,
                              const float *A, int strideA, int lda,
                              const float *B, int strideB, int ldb,
                              float       *C, int strideC, int ldc)
{
    struct batched_head_args args = {
        H, transB, M, N, K,
        A, strideA, lda,
        B, strideB, ldb,
        C, strideC, ldc
    };
    pk_parallel(pk_pool(), batched_head_worker, &args);
}

/* ── Multi-head self-attention with relative positional encoding ─────────── */

static void self_attention(const float *norm_w, const float *norm_b,
                           const MHAWeights *mha,
                           const float *pe_slice,
                           float *x, int T, CfmWS *ws)
{
    int D = PK_D_MODEL;
    int H = PK_N_HEADS;
    int Dh = PK_HEAD_DIM;
    WSA a = {ws->buf, 0};

    float *ln = ws_get(&a, T * D);
    pk_layer_norm_rows(x, norm_w, norm_b, ln, T, D);

    /* ── Fused QKV projection: one [T, D] @ [D, 3D] matmul ── */
    float *QKV = ws_get(&a, T * 3 * D);
    pk_wmatmul(ln, &mha->linear_qkv_w, QKV, T, 3 * D, D, 1.0f, 0.0f);
    for (int t = 0; t < T; t++)
        pk_bias_add(QKV + t * 3 * D, mha->linear_qkv_b, 3 * D);

    /* ── Head gather directly from interleaved QKV [T, 3D] ──
     * QKV row layout: [Q0..Q1023, K0..K1023, V0..V1023] (stride 3D).
     * For head h: Q slice at offset h*Dh, K at D+h*Dh, V at 2D+h*Dh. */
    float *Q_u = ws_get(&a, H * T * Dh);
    float *Q_v = ws_get(&a, H * T * Dh);
    float *K_h = ws_get(&a, H * T * Dh);
    float *V_h = ws_get(&a, H * T * Dh);

    for (int h = 0; h < H; h++) {
        const float *pu = mha->pos_bias_u + h * Dh;
        const float *pv = mha->pos_bias_v + h * Dh;
        for (int t = 0; t < T; t++) {
            const float *row = QKV + t * 3 * D;
            const float *q_src = row + h * Dh;
            const float *k_src = row + D + h * Dh;
            const float *v_src = row + 2 * D + h * Dh;
            float *qu_dst = Q_u + h * T * Dh + t * Dh;
            float *qv_dst = Q_v + h * T * Dh + t * Dh;
            /* Q_u = Q + pos_bias_u, Q_v = Q + pos_bias_v (SIMD, Dh=128) */
#if PK_ENC_NEON
            for (int d = 0; d < Dh; d += 4) {
                float32x4_t q = vld1q_f32(q_src + d);
                vst1q_f32(qu_dst + d, vaddq_f32(q, vld1q_f32(pu + d)));
                vst1q_f32(qv_dst + d, vaddq_f32(q, vld1q_f32(pv + d)));
            }
#elif PK_ENC_AVX2
            for (int d = 0; d < Dh; d += 8) {
                __m256 q = _mm256_loadu_ps(q_src + d);
                _mm256_storeu_ps(qu_dst + d, _mm256_add_ps(q, _mm256_loadu_ps(pu + d)));
                _mm256_storeu_ps(qv_dst + d, _mm256_add_ps(q, _mm256_loadu_ps(pv + d)));
            }
#else
            for (int d = 0; d < Dh; d++) {
                qu_dst[d] = q_src[d] + pu[d];
                qv_dst[d] = q_src[d] + pv[d];
            }
#endif
            memcpy(K_h + h * T * Dh + t * Dh, k_src, Dh * sizeof(float));
            memcpy(V_h + h * T * Dh + t * Dh, v_src, Dh * sizeof(float));
        }
    }

    /* Positional encoding projection */
    float *P = ws_get(&a, (2 * T - 1) * D);
    pk_wmatmul(pe_slice, &mha->linear_pos_w, P, 2 * T - 1, D, D, 1.0f, 0.0f);

    /* P_by_head [H, 2T-1, Dh] */
    int L = 2 * T - 1;
    float *P_h = ws_get(&a, H * L * Dh);
    for (int h = 0; h < H; h++)
        for (int p = 0; p < L; p++)
            memcpy(P_h + h * L * Dh + p * Dh,
                   P + p * D + h * Dh, Dh * sizeof(float));

    /* AC = Q_u @ K^T per head: [H, T, T] */
    float *ac = ws_get(&a, H * T * T);
    batched_head_gemm(H, CblasTrans, T, T, Dh,
                      Q_u, T * Dh, Dh, K_h, T * Dh, Dh, ac, T * T, T);

    /* BD = Q_v @ P^T per head: [H, T, 2T-1] */
    float *bd = ws_get(&a, H * T * L);
    batched_head_gemm(H, CblasTrans, T, L, Dh,
                      Q_v, T * Dh, Dh, P_h, L * Dh, Dh, bd, T * L, L);

    /* Relative shift on bd: [H, T, 2T-1] → [H, T, T] (in-place) */
    relative_shift(bd, H, T, &a);

    /* ── Fused scores combine: ac = (ac + bd) / sqrt(Dh), in-place (SIMD) ── */
    float scale = 1.0f / sqrtf((float)Dh);
    {
        int n = H * T * T, i = 0;
#if PK_ENC_NEON
        float32x4_t vs = vdupq_n_f32(scale);
        for (; i + 4 <= n; i += 4)
            vst1q_f32(ac + i, vmulq_f32(vaddq_f32(vld1q_f32(ac + i),
                                                    vld1q_f32(bd + i)), vs));
#elif PK_ENC_AVX2
        __m256 vs = _mm256_set1_ps(scale);
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(ac + i, _mm256_mul_ps(_mm256_add_ps(
                _mm256_loadu_ps(ac + i), _mm256_loadu_ps(bd + i)), vs));
#endif
        for (; i < n; i++)
            ac[i] = (ac[i] + bd[i]) * scale;
    }

    /* Softmax per [h, t, :] — ac is now the scores. */
    pk_softmax_rows(ac, H * T, T);

    /* attn_out = scores @ V per head: [T, T] @ [T, Dh] = [T, Dh] */
    float *attn_h = ws_get(&a, H * T * Dh);
    batched_head_gemm(H, CblasNoTrans, T, Dh, T,
                      ac, T * T, T, V_h, T * Dh, Dh, attn_h, T * Dh, Dh);

    /* Transpose attn_h [H, T, Dh] → attn [T, D] (concatenate heads) */
    float *attn = ws_get(&a, T * D);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            memcpy(attn + t * D + h * Dh, attn_h + h * T * Dh + t * Dh,
                   Dh * sizeof(float));

    /* Output projection + bias + residual */
    float *out = ws_get(&a, T * D);
    pk_wmatmul(attn, &mha->linear_out_w, out, T, D, D, 1.0f, 0.0f);
    for (int t = 0; t < T; t++)
        pk_bias_add(out + t * D, mha->linear_out_b, D);
    {
        int n = T * D, i = 0;
#if PK_ENC_NEON
        for (; i + 4 <= n; i += 4)
            vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(out + i)));
#elif PK_ENC_AVX2
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(x + i, _mm256_add_ps(
                _mm256_loadu_ps(x + i), _mm256_loadu_ps(out + i)));
#endif
        for (; i < n; i++)
            x[i] += out[i];
    }
}

/* ── Convolution module ──────────────────────────────────────────────────── */

static void conv_module(const float *norm_w, const float *norm_b,
                        const ConvModWeights *cw,
                        float *x, int T, CfmWS *ws)
{
    int D = PK_D_MODEL;
    WSA a = {ws->buf, 0};

    float *ln = ws_get(&a, T * D);
    pk_layer_norm_rows(x, norm_w, norm_b, ln, T, D);

    float *cf = ws_get(&a, D * T);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            cf[d * T + t] = ln[t * D + d];

    float *pw1 = ws_get(&a, 2 * D * T);
    pk_pointwise_conv1d(cf, cw->pw1_w, cw->pw1_b, pw1, D, 2 * D, T);

    float *gated = ws_get(&a, D * T);
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < T; t++) {
            float va = pw1[d * T + t];
            float vb = pw1[(D + d) * T + t];
            float s = 1.0f / (1.0f + expf(-vb));
            gated[d * T + t] = va * s;
        }
    }

    float *dw = ws_get(&a, D * T);
    pk_depthwise_conv1d(gated, cw->dw_w, cw->dw_b, dw, D, T, PK_CONV_KERNEL);

    pk_swish(dw, D * T);

    float *pw2 = ws_get(&a, D * T);
    pk_pointwise_conv1d(dw, cw->pw2_w, cw->pw2_b, pw2, D, D, T);

    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            x[t * D + d] += pw2[d * T + t];
}

/* ── Single conformer block ──────────────────────────────────────────────── */

static void conformer_block(const ConformerBlock *blk,
                            const float *pe_slice,
                            float *x, int T, CfmWS *ws)
{
    feed_forward(blk->norm_ff1_w, blk->norm_ff1_b, &blk->ff1, x, T, ws);
    self_attention(blk->norm_sa_w, blk->norm_sa_b, &blk->mha, pe_slice, x, T, ws);
    conv_module(blk->norm_conv_w, blk->norm_conv_b, &blk->conv, x, T, ws);
    feed_forward(blk->norm_ff2_w, blk->norm_ff2_b, &blk->ff2, x, T, ws);
    pk_layer_norm_rows_inplace(x, blk->norm_out_w, blk->norm_out_b, T, PK_D_MODEL);
}

/* ── Full encoder ────────────────────────────────────────────────────────── */

static void encode_blocks(const PkModel *m, float *x, int T)
{
    float *pe_slice = slice_pos_encoding(m->enc.pe_table, T, PK_D_MODEL);

    CfmWS ws;
    ws.cap = conformer_ws_peak(T);
    ws.buf = (float *)malloc(ws.cap * sizeof(float));

    for (int i = 0; i < PK_N_BLOCKS; i++) {
        if (pk_verbose)
            fprintf(stderr, "  encoder block %d/%d\r", i + 1, PK_N_BLOCKS);
        conformer_block(&m->enc.blocks[i], pe_slice, x, T, &ws);
    }

    free(ws.buf);
    free(pe_slice);
    if (pk_verbose) fprintf(stderr, "\n");
}

static void encode_blocks_chunked(const PkModel *m, const float *x, int T,
                                  float *out)
{
    int C  = PK_CHUNK_SIZE;
    int ov = PK_CHUNK_OVERLAP;
    int n_chunks = (T + C - 1) / C;
    int D = PK_D_MODEL;

    int common_in_len = C + 2 * ov;
    if (common_in_len > T) common_in_len = T;
    float *pe_common = slice_pos_encoding(m->enc.pe_table, common_in_len, D);

    /* Workspace sized for the largest chunk. */
    CfmWS ws;
    ws.cap = conformer_ws_peak(common_in_len);
    ws.buf = (float *)malloc(ws.cap * sizeof(float));

    for (int ci = 0; ci < n_chunks; ci++) {
        int out_start = ci * C;
        int out_end   = out_start + C;
        if (out_end > T) out_end = T;
        int out_len = out_end - out_start;

        int in_start = out_start - ov;
        if (in_start < 0) in_start = 0;
        int in_end = out_end + ov;
        if (in_end > T) in_end = T;
        int in_len = in_end - in_start;

        int left_ov = out_start - in_start;

        if (pk_verbose)
            fprintf(stderr, "  chunk %d/%d: frames [%d, %d) input [%d, %d)\r",
                    ci + 1, n_chunks, out_start, out_end, in_start, in_end);

        float *chunk = alloc_f(in_len * D);
        memcpy(chunk, x + (size_t)in_start * D, (size_t)in_len * D * sizeof(float));

        float *pe_slice;
        int own_pe = 0;
        if (in_len == common_in_len) {
            pe_slice = pe_common;
        } else {
            pe_slice = slice_pos_encoding(m->enc.pe_table, in_len, D);
            own_pe = 1;
        }

        for (int i = 0; i < PK_N_BLOCKS; i++)
            conformer_block(&m->enc.blocks[i], pe_slice, chunk, in_len, &ws);

        if (own_pe) free(pe_slice);

        memcpy(out + (size_t)out_start * D,
               chunk + (size_t)left_ov * D,
               (size_t)out_len * D * sizeof(float));
        free(chunk);
    }

    free(ws.buf);
    free(pe_common);
    if (pk_verbose) fprintf(stderr, "\n");
}

/* Pre-encode mel → [T' × D] time-first scaled frames. */
static int pre_encode_and_scale(const PkModel *m, const float *mel, int n_frames,
                                float **out_x)
{
    float *sub;
    int Tp = pre_encode(&m->enc.pre_encode, mel, n_frames, &sub);

    float scale = sqrtf((float)PK_D_MODEL);
    float *x = alloc_f(Tp * PK_D_MODEL);
    for (int t = 0; t < Tp; t++)
        for (int d = 0; d < PK_D_MODEL; d++)
            x[t * PK_D_MODEL + d] = sub[d * Tp + t] * scale;
    free(sub);

    *out_x = x;
    return Tp;
}

/* ── Public encoder API ──────────────────────────────────────────────────── */

/* Decide whether to chunk based on the PK_CHUNK env var:
 *   unset / "auto" : auto-chunk when Tp > PK_CHUNK_THRESHOLD (default)
 *   "0" / "off"    : force full-sequence (never chunk)
 *   "1" / "on"     : force chunked (regardless of length)
 */
static int should_chunk(int Tp)
{
    const char *env = getenv("PK_CHUNK");
    if (env) {
        if (env[0] == '0' || strcmp(env, "off") == 0) return 0;
        if (env[0] == '1' || strcmp(env, "on")  == 0) return 1;
    }
    return Tp > PK_CHUNK_THRESHOLD;
}

int pk_encoder(const PkModel *m, const float *mel, int n_frames,
               float **out, int *out_len)
{
    float *x;
    int Tp = pre_encode_and_scale(m, mel, n_frames, &x);

    if (should_chunk(Tp)) {
        if (pk_verbose)
            fprintf(stderr, "  chunked: %d frames, chunk=%d, overlap=%d\n",
                    Tp, PK_CHUNK_SIZE, PK_CHUNK_OVERLAP);
        float *encoded = alloc_f(Tp * PK_D_MODEL);
        encode_blocks_chunked(m, x, Tp, encoded);
        free(x);
        *out = encoded;
    } else {
        encode_blocks(m, x, Tp);
        *out = x;
    }

    *out_len = Tp;
    return 0;
}

int pk_encoder_chunked(const PkModel *m, const float *mel, int n_frames,
                       float **out, int *out_len)
{
    float *x;
    int Tp = pre_encode_and_scale(m, mel, n_frames, &x);

    if (pk_verbose)
        fprintf(stderr, "  chunked: %d frames, chunk=%d, overlap=%d\n",
                Tp, PK_CHUNK_SIZE, PK_CHUNK_OVERLAP);
    float *encoded = alloc_f(Tp * PK_D_MODEL);
    encode_blocks_chunked(m, x, Tp, encoded);
    free(x);

    *out = encoded;
    *out_len = Tp;
    return 0;
}
