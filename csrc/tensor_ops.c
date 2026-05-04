/*  tensor_ops.c – Elementwise + reduction kernels for parakeet inference.
 *
 *  Hot ops (LayerNorm, softmax, depthwise conv, Swish) get explicit NEON /
 *  AVX2 inner loops plus a batched "_rows" variant that parallelises across
 *  rows through the thread pool. Scalar fallbacks are always provided.
 *
 *  The SIMD expf helpers use a standard 5-term polynomial approximation
 *  around 0 with a 2^k range-reduction step; accurate to ~1e-6 relative —
 *  well inside float32 precision for softmax and sigmoid.
 */

#include "parakeet.h"
#include "blas.h"
#include "threadpool.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define PK_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define PK_AVX2 1
#endif

/* ── Matrix multiply (row-major) ─────────────────────────────────────────── */

void pk_matmul(const float *A, const float *B, float *C,
               int M, int N, int K, float alpha, float beta)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

void pk_matmul_q(const float *A, const signed char *B, float B_scale,
                 float *C, int M, int N, int K, float alpha, float beta)
{
    cblas_sgemm_qb(M, N, K, alpha, A, K, B, B_scale, N, beta, C, N);
}

void pk_matmul_q4(const float *A, const unsigned char *B, const float *scales,
                  float *C, int M, int N, int K, float alpha, float beta)
{
    cblas_sgemm_q4b(M, N, K, alpha, A, K, B, scales, N / 2, beta, C, N);
}

void pk_wmatmul(const float *A, const Weight *W, float *C,
                int M, int N, int K, float alpha, float beta)
{
    if (W->bits == 4)
        pk_matmul_q4(A, (const unsigned char *)W->data, W->scales,
                     C, M, N, K, alpha, beta);
    else if (W->bits == 8)
        pk_matmul_q(A, (const signed char *)W->data, W->scale,
                    C, M, N, K, alpha, beta);
    else
        pk_matmul(A, (const float *)W->data, C, M, N, K, alpha, beta);
}

/* ── SIMD expf approximation ─────────────────────────────────────────────── */
/*
 * Range-reduction identity: e^x = 2^(x*log2(e)) = 2^k * e^r  where
 * k = round(x*log2(e)), r = x - k*ln(2). r lies in [-ln(2)/2, ln(2)/2] so
 * a 5-term Taylor polynomial evaluates e^r to ~1e-7 absolute error. The
 * 2^k scaling is done by injecting (k+127)<<23 into the exponent field of a
 * float. Input clamped to [-87, 87] to keep 2^k inside float range.
 */

#if PK_NEON
static inline float32x4_t pk_exp_neon(float32x4_t x)
{
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(87.0f));

    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2   = vdupq_n_f32(0.69314718055994529f);

    float32x4_t kf = vrndnq_f32(vmulq_f32(x, log2e));
    int32x4_t   ki = vcvtq_s32_f32(kf);
    float32x4_t r  = vsubq_f32(x, vmulq_f32(kf, ln2));

    /* Horner's method for polynomial e^r ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 */
    float32x4_t y = vdupq_n_f32(1.0f / 120.0f);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f),  y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f),   y, r);
    y = vfmaq_f32(vdupq_n_f32(0.5f),          y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f),          y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f),          y, r);

    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(pow2k));
}
#endif

#if PK_AVX2
static inline __m256 pk_exp_avx2(__m256 x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)),
                      _mm256_set1_ps(87.0f));

    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2   = _mm256_set1_ps(0.69314718055994529f);

    __m256  kf = _mm256_round_ps(_mm256_mul_ps(x, log2e),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i ki = _mm256_cvtps_epi32(kf);
    __m256  r  = _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2));

    __m256 y = _mm256_set1_ps(1.0f / 120.0f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 24.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 6.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));

    __m256i pow2k = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2k));
}
#endif

/* Scalar horizontal-sum helpers. */
#if PK_NEON
static inline float pk_hsum_neon(float32x4_t v) { return vaddvq_f32(v); }
static inline float pk_hmax_neon(float32x4_t v) { return vmaxvq_f32(v); }
#endif
#if PK_AVX2
static inline float pk_hsum_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}
static inline float pk_hmax_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m1);
}
#endif

/* ── Layer normalization ─────────────────────────────────────────────────── */

void pk_layer_norm(const float *x, const float *gamma, const float *beta_,
                   float *out, int len)
{
    /* Pass 1 + 2: mean and variance via SIMD horizontal sums. */
#if PK_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) vsum = vaddq_f32(vsum, vld1q_f32(x + i));
    float sum = pk_hsum_neon(vsum);
    for (; i < len; i++) sum += x[i];
    float mean = sum / (float)len;

    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vvar  = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmean);
        vvar = vfmaq_f32(vvar, d, d);
    }
    float var_sum = pk_hsum_neon(vvar);
    for (; i < len; i++) { float d = x[i] - mean; var_sum += d * d; }
    float var = var_sum / (float)len;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    /* Pass 3: out = (x - mean) * inv_std * gamma + beta */
    float32x4_t vinv = vdupq_n_f32(inv_std);
    i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t xv = vsubq_f32(vld1q_f32(x + i), vmean);
        float32x4_t g  = vld1q_f32(gamma + i);
        float32x4_t b  = vld1q_f32(beta_ + i);
        vst1q_f32(out + i, vfmaq_f32(b, vmulq_f32(xv, vinv), g));
    }
    for (; i < len; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta_[i];

#elif PK_AVX2
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    float sum = pk_hsum_avx2(vsum);
    for (; i < len; i++) sum += x[i];
    float mean = sum / (float)len;

    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar  = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        vvar = _mm256_fmadd_ps(d, d, vvar);
    }
    float var_sum = pk_hsum_avx2(vvar);
    for (; i < len; i++) { float d = x[i] - mean; var_sum += d * d; }
    float var = var_sum / (float)len;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    __m256 vinv = _mm256_set1_ps(inv_std);
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 xv = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        __m256 g  = _mm256_loadu_ps(gamma + i);
        __m256 b  = _mm256_loadu_ps(beta_ + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(_mm256_mul_ps(xv, vinv), g, b));
    }
    for (; i < len; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta_[i];

#else
    double mean_d = 0.0;
    for (int i = 0; i < len; i++) mean_d += x[i];
    float mean = (float)(mean_d / len);

    double var_d = 0.0;
    for (int i = 0; i < len; i++) { double d = x[i] - mean; var_d += d * d; }
    float inv_std = 1.0f / sqrtf((float)(var_d / len) + 1e-5f);

    for (int i = 0; i < len; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta_[i];
#endif
}

/* Row-batched LN – parallel across rows. */
struct ln_args {
    const float *x; const float *gamma; const float *beta_;
    float *out; int n_rows; int D;
};

static void ln_worker(int tid, int nt, void *aux)
{
    struct ln_args *a = (struct ln_args *)aux;
    int first = ((long long)a->n_rows * tid) / nt;
    int last  = ((long long)a->n_rows * (tid + 1)) / nt;
    for (int r = first; r < last; r++)
        pk_layer_norm(a->x + (size_t)r * a->D,
                      a->gamma, a->beta_,
                      a->out + (size_t)r * a->D, a->D);
}

void pk_layer_norm_rows(const float *x, const float *gamma, const float *beta_,
                        float *out, int n_rows, int D)
{
    struct ln_args args = {x, gamma, beta_, out, n_rows, D};
    pk_parallel(pk_pool(), ln_worker, &args);
}

/* Row-batched in-place LN – parallel across rows, no separate output buffer.
 * Each worker uses a D-sized stack buffer to normalize one row at a time. */
struct ln_ip_args {
    float *x; const float *gamma; const float *beta_; int n_rows; int D;
};

static void ln_ip_worker(int tid, int nt, void *aux)
{
    struct ln_ip_args *a = (struct ln_ip_args *)aux;
    int first = ((long long)a->n_rows * tid) / nt;
    int last  = ((long long)a->n_rows * (tid + 1)) / nt;
    float row_buf[PK_D_MODEL];  /* 4 KB on stack — safe for any thread */
    for (int r = first; r < last; r++) {
        float *row = a->x + (size_t)r * a->D;
        pk_layer_norm(row, a->gamma, a->beta_, row_buf, a->D);
        memcpy(row, row_buf, a->D * sizeof(float));
    }
}

void pk_layer_norm_rows_inplace(float *x, const float *gamma, const float *beta_,
                                int n_rows, int D)
{
    struct ln_ip_args args = {x, gamma, beta_, n_rows, D};
    pk_parallel(pk_pool(), ln_ip_worker, &args);
}

/* ── Softmax ─────────────────────────────────────────────────────────────── */

void pk_softmax(float *x, int len)
{
#if PK_NEON
    /* Pass 1: max */
    float32x4_t vmx = vdupq_n_f32(x[0]);
    int i = 0;
    for (; i + 4 <= len; i += 4) vmx = vmaxq_f32(vmx, vld1q_f32(x + i));
    float mx = pk_hmax_neon(vmx);
    for (; i < len; i++) if (x[i] > mx) mx = x[i];

    /* Pass 2: x = exp(x - mx); sum */
    float32x4_t vsum = vdupq_n_f32(0.0f);
    float32x4_t vmx4 = vdupq_n_f32(mx);
    i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t e = pk_exp_neon(vsubq_f32(vld1q_f32(x + i), vmx4));
        vst1q_f32(x + i, e);
        vsum = vaddq_f32(vsum, e);
    }
    float sum = pk_hsum_neon(vsum);
    for (; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }

    /* Pass 3: normalise */
    float32x4_t vinv = vdupq_n_f32(1.0f / sum);
    i = 0;
    for (; i + 4 <= len; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    for (; i < len; i++) x[i] *= (1.0f / sum);

#elif PK_AVX2
    __m256 vmx = _mm256_set1_ps(x[0]);
    int i = 0;
    for (; i + 8 <= len; i += 8) vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(x + i));
    float mx = pk_hmax_avx2(vmx);
    for (; i < len; i++) if (x[i] > mx) mx = x[i];

    __m256 vsum = _mm256_setzero_ps();
    __m256 vmx8 = _mm256_set1_ps(mx);
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 e = pk_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmx8));
        _mm256_storeu_ps(x + i, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum = pk_hsum_avx2(vsum);
    for (; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }

    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 8 <= len; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < len; i++) x[i] *= (1.0f / sum);

#else
    float mx = x[0];
    for (int i = 1; i < len; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < len; i++) x[i] *= inv;
#endif
}

struct softmax_rows_args {
    float *x; int n_rows; int len;
};

static void softmax_rows_worker(int tid, int nt, void *aux)
{
    struct softmax_rows_args *a = (struct softmax_rows_args *)aux;
    int first = ((long long)a->n_rows * tid) / nt;
    int last  = ((long long)a->n_rows * (tid + 1)) / nt;
    for (int r = first; r < last; r++)
        pk_softmax(a->x + (size_t)r * a->len, a->len);
}

void pk_softmax_rows(float *x, int n_rows, int len)
{
    struct softmax_rows_args args = {x, n_rows, len};
    pk_parallel(pk_pool(), softmax_rows_worker, &args);
}

/* ── Swish: x * sigmoid(x) ──────────────────────────────────────────────── */

void pk_swish(float *x, int len)
{
#if PK_NEON
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t e = pk_exp_neon(vnegq_f32(v));
        float32x4_t s = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(x + i, vmulq_f32(v, s));
    }
    for (; i < len; i++) x[i] = x[i] / (1.0f + expf(-x[i]));

#elif PK_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 e = pk_exp_avx2(_mm256_sub_ps(_mm256_setzero_ps(), v));
        __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, s));
    }
    for (; i < len; i++) x[i] = x[i] / (1.0f + expf(-x[i]));

#else
    for (int i = 0; i < len; i++) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        x[i] *= s;
    }
#endif
}

/* ── ReLU ────────────────────────────────────────────────────────────────── */

void pk_relu(float *x, int len)
{
#if PK_NEON
    const float32x4_t z = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) vst1q_f32(x + i, vmaxq_f32(vld1q_f32(x + i), z));
    for (; i < len; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#elif PK_AVX2
    const __m256 z = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) _mm256_storeu_ps(x + i, _mm256_max_ps(_mm256_loadu_ps(x + i), z));
    for (; i < len; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#else
    for (int i = 0; i < len; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#endif
}

/* ── Bias add ────────────────────────────────────────────────────────────── */

void pk_bias_add(float *x, const float *b, int len)
{
#if PK_NEON
    int i = 0;
    for (; i + 4 <= len; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(b + i)));
    for (; i < len; i++) x[i] += b[i];
#elif PK_AVX2
    int i = 0;
    for (; i + 8 <= len; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(b + i)));
    for (; i < len; i++) x[i] += b[i];
#else
    for (int i = 0; i < len; i++) x[i] += b[i];
#endif
}

/* ── Conv2D (supports groups, stride, padding) ───────────────────────────── */

void pk_conv2d(const float *x, const float *w, const float *b,
               float *out, int Cin, int H, int W,
               int Cout, int Kh, int Kw, int sh, int sw, int ph, int pw,
               int groups)
{
    int Hout = (H + 2 * ph - Kh) / sh + 1;
    int Wout = (W + 2 * pw - Kw) / sw + 1;
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < Cout_per_group; oc++) {
            int oc_abs = g * Cout_per_group + oc;
            for (int oh = 0; oh < Hout; oh++) {
                for (int ow = 0; ow < Wout; ow++) {
                    float val = b ? b[oc_abs] : 0.0f;
                    for (int ic = 0; ic < Cin_per_group; ic++) {
                        int ic_abs = g * Cin_per_group + ic;
                        for (int kh = 0; kh < Kh; kh++) {
                            for (int kw = 0; kw < Kw; kw++) {
                                int ih = oh * sh - ph + kh;
                                int iw = ow * sw - pw + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float xv = x[ic_abs * H * W + ih * W + iw];
                                    float wv = w[oc_abs * Cin_per_group * Kh * Kw
                                                 + ic * Kh * Kw + kh * Kw + kw];
                                    val += xv * wv;
                                }
                            }
                        }
                    }
                    out[oc_abs * Hout * Wout + oh * Wout + ow] = val;
                }
            }
        }
    }
}

/* ── 1D depthwise convolution (symmetric padding, parallel over channels) ── */

/* Per-channel kernel: symmetric pad=(K-1)/2. Vectorises along the time axis.
 * Splits into [0, pad), [pad, T-pad) with full K inner loop, [T-pad, T). */
static void depthwise_conv1d_channel(const float *x, const float *w, float bias,
                                     float *out, int T, int K)
{
    int pad = (K - 1) / 2;

    /* Left boundary. */
    for (int t = 0; t < pad && t < T; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) {
            int src_t = t - pad + k;
            if (src_t >= 0 && src_t < T) val += x[src_t] * w[k];
        }
        out[t] = val;
    }

    /* Interior: t in [pad, T - pad). Full K without bounds check. */
    int t_lo = pad;
    int t_hi = T - pad;
    if (t_hi < t_lo) t_hi = t_lo;

#if PK_NEON
    int t = t_lo;
    float32x4_t vbias = vdupq_n_f32(bias);
    for (; t + 4 <= t_hi; t += 4) {
        float32x4_t acc = vbias;
        for (int k = 0; k < K; k++)
            acc = vfmaq_n_f32(acc, vld1q_f32(x + t - pad + k), w[k]);
        vst1q_f32(out + t, acc);
    }
    for (; t < t_hi; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) val += x[t - pad + k] * w[k];
        out[t] = val;
    }
#elif PK_AVX2
    int t = t_lo;
    __m256 vbias = _mm256_set1_ps(bias);
    for (; t + 8 <= t_hi; t += 8) {
        __m256 acc = vbias;
        for (int k = 0; k < K; k++)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + t - pad + k),
                                  _mm256_set1_ps(w[k]), acc);
        _mm256_storeu_ps(out + t, acc);
    }
    for (; t < t_hi; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) val += x[t - pad + k] * w[k];
        out[t] = val;
    }
#else
    for (int t = t_lo; t < t_hi; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) val += x[t - pad + k] * w[k];
        out[t] = val;
    }
#endif

    /* Right boundary. */
    for (int t = t_hi; t < T; t++) {
        if (t < pad) continue;  /* already handled above if T < 2*pad */
        float val = bias;
        for (int k = 0; k < K; k++) {
            int src_t = t - pad + k;
            if (src_t >= 0 && src_t < T) val += x[src_t] * w[k];
        }
        out[t] = val;
    }
}

struct dw_args {
    const float *x; const float *w; const float *b;
    float *out; int C; int T; int K;
};

static void dw_worker(int tid, int nt, void *aux)
{
    struct dw_args *a = (struct dw_args *)aux;
    int first = ((long long)a->C * tid) / nt;
    int last  = ((long long)a->C * (tid + 1)) / nt;
    for (int c = first; c < last; c++) {
        float bias = a->b ? a->b[c] : 0.0f;
        depthwise_conv1d_channel(a->x + (size_t)c * a->T,
                                 a->w + (size_t)c * a->K,
                                 bias,
                                 a->out + (size_t)c * a->T,
                                 a->T, a->K);
    }
}

void pk_depthwise_conv1d(const float *x, const float *w, const float *b,
                         float *out, int C, int T, int K)
{
    struct dw_args args = {x, w, b, out, C, T, K};
    pk_parallel(pk_pool(), dw_worker, &args);
}

/* ── Pointwise conv1d (kernel=1): essentially a matmul ──────────────────── */

void pk_pointwise_conv1d(const float *x, const float *w, const float *b,
                         float *out, int Cin, int Cout, int T)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Cout, T, Cin, 1.0f, w, Cin, x, T, 0.0f, out, T);

    if (b) {
        for (int c = 0; c < Cout; c++)
            for (int t = 0; t < T; t++)
                out[c * T + t] += b[c];
    }
}

/* ── LSTM cell (one time-step) ───────────────────────────────────────────── */

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

void pk_lstm_step(const float *x_t, int input_size,
                  const float *Wi, const float *Wr, const float *B,
                  float *h, float *c, int hidden_size)
{
    int H = hidden_size;
    float *gates = (float *)malloc(4 * H * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                4 * H, input_size, 1.0f, Wi, input_size, x_t, 1,
                0.0f, gates, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                4 * H, H, 1.0f, Wr, H, h, 1,
                1.0f, gates, 1);

    for (int i = 0; i < 4 * H; i++)
        gates[i] += B[i] + B[4 * H + i];

    float *gi = gates;
    float *go = gates + H;
    float *gf = gates + 2 * H;
    float *gc = gates + 3 * H;

    for (int j = 0; j < H; j++) {
        float i_g = sigmoid_f(gi[j]);
        float o_g = sigmoid_f(go[j]);
        float f_g = sigmoid_f(gf[j]);
        float c_g = tanhf(gc[j]);

        c[j] = f_g * c[j] + i_g * c_g;
        h[j] = o_g * tanhf(c[j]);
    }

    free(gates);
}
