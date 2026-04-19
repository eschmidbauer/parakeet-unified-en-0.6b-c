/*  blas.c – Hand-rolled sgemm + sgemv (row-major, fp32).
 *
 *  Register-blocked SIMD kernels:
 *    • ARM NEON     (aarch64, Apple Silicon, Linux on ARM)
 *    • x86_64 AVX2  (Linux, Windows, macOS Intel)
 *    • Scalar       (portable fallback)
 *
 *  Phase-1 optimisations (over the plain 4×16 NEON/AVX2 kernel):
 *    1. SIMD edge kernels for MR ∈ {1, 2, 3} so every row – regardless of
 *       M % 4 – goes through the vectorised path. This matters for Parakeet
 *       specifically because T=39 after subsampling, giving 3 edge rows per
 *       matmul that would otherwise fall to scalar code.
 *    2. Software prefetch (`__builtin_prefetch`) of the next-K row of B in
 *       the inner loop. On L1-bound gemms it covers the load latency.
 *
 *  The main tile stays 4×16 – measurements show this is optimal for the
 *  tall-and-thin matrices typical of this model.
 *
 *  Compile with `-O3 -march=native -ffast-math`.
 */
#include "parakeet.h"
#include "blas.h"
#include "threadpool.h"
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

/* How many K-iterations ahead to prefetch B rows for. */
#define PK_PF_DIST 8

/* Portable prefetch macro. Compilers without support get a no-op. */
#if defined(__GNUC__) || defined(__clang__)
  #define PK_PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#else
  #define PK_PREFETCH(addr) ((void)0)
#endif

/* ── SGEMV ────────────────────────────────────────────────────────────────── */

void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                 int M, int N,
                 float alpha, const float *A, int lda,
                 const float *X, int incX,
                 float beta, float *Y, int incY)
{
    (void)order;

    if (trans == CblasNoTrans) {
        for (int i = 0; i < M; i++) {
            const float *a = A + i * lda;
            float acc = 0.0f;
            if (incX == 1) {
                for (int j = 0; j < N; j++) acc += a[j] * X[j];
            } else {
                for (int j = 0; j < N; j++) acc += a[j] * X[j * incX];
            }
            Y[i * incY] = alpha * acc + beta * Y[i * incY];
        }
    } else {
        if (beta == 0.0f) {
            for (int j = 0; j < N; j++) Y[j * incY] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; j++) Y[j * incY] *= beta;
        }
        for (int i = 0; i < M; i++) {
            float xi = alpha * X[i * incX];
            const float *a = A + i * lda;
            if (incY == 1) {
                for (int j = 0; j < N; j++) Y[j] += a[j] * xi;
            } else {
                for (int j = 0; j < N; j++) Y[j * incY] += a[j] * xi;
            }
        }
    }
}

/* ── SGEMM helpers ────────────────────────────────────────────────────────── */

static void scale_c(float *C, int M, int N, int ldc, float beta)
{
    if (beta == 1.0f) return;
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) memset(C + i * ldc, 0, N * sizeof(float));
        return;
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * ldc + j] *= beta;
}

/* ── NEON kernels: MR×16 NN (MR = 1, 2, 3, 4) ─────────────────────────────── */

#if PK_NEON

/* Generic MR×16 NN kernel – templated on MR as a compile-time constant.
 * With MR as a template parameter we get fully-unrolled row-loops at every
 * call site. GCC/Clang inline the whole thing; the 4×16 version compiles
 * identically to the hand-written one. */
#define NEON_KERNEL_NN(MR_VAL, SUFFIX)                                     \
static inline void kernel_nn_##SUFFIX##x16_neon(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    float32x4_t c[MR_VAL][4];                                              \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        c[i][0] = vld1q_f32(C + i * ldc + 0);                              \
        c[i][1] = vld1q_f32(C + i * ldc + 4);                              \
        c[i][2] = vld1q_f32(C + i * ldc + 8);                              \
        c[i][3] = vld1q_f32(C + i * ldc + 12);                             \
    }                                                                      \
    for (int k = 0; k < K; k++) {                                          \
        if (k + PK_PF_DIST < K)                                            \
            PK_PREFETCH(B + (k + PK_PF_DIST) * ldb);                       \
        float32x4_t b0 = vld1q_f32(B + k * ldb + 0);                       \
        float32x4_t b1 = vld1q_f32(B + k * ldb + 4);                       \
        float32x4_t b2 = vld1q_f32(B + k * ldb + 8);                       \
        float32x4_t b3 = vld1q_f32(B + k * ldb + 12);                      \
        for (int i = 0; i < MR_VAL; i++) {                                 \
            float32x4_t a = vdupq_n_f32(alpha * A[i * lda + k]);           \
            c[i][0] = vfmaq_f32(c[i][0], a, b0);                           \
            c[i][1] = vfmaq_f32(c[i][1], a, b1);                           \
            c[i][2] = vfmaq_f32(c[i][2], a, b2);                           \
            c[i][3] = vfmaq_f32(c[i][3], a, b3);                           \
        }                                                                  \
    }                                                                      \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        vst1q_f32(C + i * ldc + 0,  c[i][0]);                              \
        vst1q_f32(C + i * ldc + 4,  c[i][1]);                              \
        vst1q_f32(C + i * ldc + 8,  c[i][2]);                              \
        vst1q_f32(C + i * ldc + 12, c[i][3]);                              \
    }                                                                      \
}

NEON_KERNEL_NN(1, 1)
NEON_KERNEL_NN(2, 2)
NEON_KERNEL_NN(3, 3)
NEON_KERNEL_NN(4, 4)

/* NEON MR×16 NT kernel (dot-product per output element). */
#define NEON_KERNEL_NT(MR_VAL, SUFFIX)                                     \
static inline void kernel_nt_##SUFFIX##x16_neon(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        const float *a = A + i * lda;                                      \
        float *c = C + i * ldc;                                            \
        for (int j = 0; j < 16; j++) {                                     \
            const float *b = B + j * ldb;                                  \
            float32x4_t acc = vdupq_n_f32(0.0f);                           \
            int k = 0;                                                     \
            for (; k + 4 <= K; k += 4) {                                   \
                float32x4_t av = vld1q_f32(a + k);                         \
                float32x4_t bv = vld1q_f32(b + k);                         \
                acc = vfmaq_f32(acc, av, bv);                              \
            }                                                              \
            float s = vaddvq_f32(acc);                                     \
            for (; k < K; k++) s += a[k] * b[k];                           \
            c[j] += alpha * s;                                             \
        }                                                                  \
    }                                                                      \
}

NEON_KERNEL_NT(1, 1)
NEON_KERNEL_NT(2, 2)
NEON_KERNEL_NT(3, 3)
NEON_KERNEL_NT(4, 4)

#endif /* PK_NEON */

/* ── AVX2 kernels: MR×16 NN (MR = 1, 2, 3, 4) ─────────────────────────────── */

#if PK_AVX2

#define AVX2_KERNEL_NN(MR_VAL, SUFFIX)                                     \
static inline void kernel_nn_##SUFFIX##x16_avx2(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    __m256 c0[MR_VAL], c1[MR_VAL];                                         \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        c0[i] = _mm256_loadu_ps(C + i * ldc + 0);                          \
        c1[i] = _mm256_loadu_ps(C + i * ldc + 8);                          \
    }                                                                      \
    for (int k = 0; k < K; k++) {                                          \
        if (k + PK_PF_DIST < K)                                            \
            PK_PREFETCH(B + (k + PK_PF_DIST) * ldb);                       \
        __m256 b0 = _mm256_loadu_ps(B + k * ldb + 0);                      \
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);                      \
        for (int i = 0; i < MR_VAL; i++) {                                 \
            __m256 a = _mm256_set1_ps(alpha * A[i * lda + k]);             \
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);                         \
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);                         \
        }                                                                  \
    }                                                                      \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        _mm256_storeu_ps(C + i * ldc + 0, c0[i]);                          \
        _mm256_storeu_ps(C + i * ldc + 8, c1[i]);                          \
    }                                                                      \
}

AVX2_KERNEL_NN(1, 1)
AVX2_KERNEL_NN(2, 2)
AVX2_KERNEL_NN(3, 3)
AVX2_KERNEL_NN(4, 4)

#define AVX2_KERNEL_NT(MR_VAL, SUFFIX)                                     \
static inline void kernel_nt_##SUFFIX##x16_avx2(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        const float *a = A + i * lda;                                      \
        float *c = C + i * ldc;                                            \
        for (int j = 0; j < 16; j++) {                                     \
            const float *b = B + j * ldb;                                  \
            __m256 acc = _mm256_setzero_ps();                              \
            int k = 0;                                                     \
            for (; k + 8 <= K; k += 8) {                                   \
                __m256 av = _mm256_loadu_ps(a + k);                        \
                __m256 bv = _mm256_loadu_ps(b + k);                        \
                acc = _mm256_fmadd_ps(av, bv, acc);                        \
            }                                                              \
            __m128 lo = _mm256_castps256_ps128(acc);                       \
            __m128 hi = _mm256_extractf128_ps(acc, 1);                     \
            __m128 s4 = _mm_add_ps(lo, hi);                                \
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));             \
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));         \
            float s = _mm_cvtss_f32(s1);                                   \
            for (; k < K; k++) s += a[k] * b[k];                           \
            c[j] += alpha * s;                                             \
        }                                                                  \
    }                                                                      \
}

AVX2_KERNEL_NT(1, 1)
AVX2_KERNEL_NT(2, 2)
AVX2_KERNEL_NT(3, 3)
AVX2_KERNEL_NT(4, 4)

#endif /* PK_AVX2 */

/* ── Scalar fallback kernels (only compiled when no SIMD is available) ────── */

#if !PK_NEON && !PK_AVX2
static inline void kernel_nn_generic(int MR_, int K, float alpha,
                                     const float *A, int lda,
                                     const float *B, int ldb,
                                     float *C, int ldc)
{
    for (int i = 0; i < MR_; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            float a = alpha * A[i * lda + k];
            const float *b = B + k * ldb;
            for (int j = 0; j < 16; j++) c[j] += a * b[j];
        }
    }
}

static inline void kernel_nt_generic(int MR_, int K, float alpha,
                                     const float *A, int lda,
                                     const float *B, int ldb,
                                     float *C, int ldc)
{
    for (int i = 0; i < MR_; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < 16; j++) {
            const float *b = B + j * ldb;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[k] * b[k];
            c[j] += alpha * acc;
        }
    }
}
#endif

/* ── Kernel dispatch: call the right MR variant based on remaining rows ───── */

static void dispatch_nn(int MR_, int K, float alpha,
                        const float *A, int lda,
                        const float *B, int ldb,
                        float *C, int ldc)
{
#if PK_NEON
    switch (MR_) {
        case 1: kernel_nn_1x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nn_2x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nn_3x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nn_4x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#elif PK_AVX2
    switch (MR_) {
        case 1: kernel_nn_1x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nn_2x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nn_3x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nn_4x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#else
    kernel_nn_generic(MR_, K, alpha, A, lda, B, ldb, C, ldc);
#endif
}

static void dispatch_nt(int MR_, int K, float alpha,
                        const float *A, int lda,
                        const float *B, int ldb,
                        float *C, int ldc)
{
#if PK_NEON
    switch (MR_) {
        case 1: kernel_nt_1x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nt_2x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nt_3x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nt_4x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#elif PK_AVX2
    switch (MR_) {
        case 1: kernel_nt_1x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nt_2x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nt_3x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nt_4x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#else
    kernel_nt_generic(MR_, K, alpha, A, lda, B, ldb, C, ldc);
#endif
}

/* ── Column-edge (N < 16) SIMD kernels ────────────────────────────────────── */

/* NN col-edge: vectorise as much of the N cols as possible with the widest
 * SIMD available (NEON 4-wide, AVX2 8-wide), then handle the scalar tail. */
static void update_edge_nn(int M, int N, int K, float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float *C, int ldc)
{
#if PK_NEON
    int Nv4 = (N / 4) * 4;   /* cols covered by NEON vectors */
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            if (k + PK_PF_DIST < K)
                PK_PREFETCH(B + (k + PK_PF_DIST) * ldb);
            float a_s = alpha * A[i * lda + k];
            float32x4_t a = vdupq_n_f32(a_s);
            const float *b = B + k * ldb;
            int j = 0;
            for (; j + 4 <= Nv4; j += 4) {
                float32x4_t bv = vld1q_f32(b + j);
                float32x4_t cv = vld1q_f32(c + j);
                vst1q_f32(c + j, vfmaq_f32(cv, a, bv));
            }
            for (; j < N; j++) c[j] += a_s * b[j];
        }
    }
#elif PK_AVX2
    int Nv8 = (N / 8) * 8;   /* cols covered by AVX2 vectors (8-wide) */
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            if (k + PK_PF_DIST < K)
                PK_PREFETCH(B + (k + PK_PF_DIST) * ldb);
            float a_s = alpha * A[i * lda + k];
            __m256 a = _mm256_set1_ps(a_s);
            const float *b = B + k * ldb;
            int j = 0;
            for (; j + 8 <= Nv8; j += 8) {
                __m256 bv = _mm256_loadu_ps(b + j);
                __m256 cv = _mm256_loadu_ps(c + j);
                _mm256_storeu_ps(c + j, _mm256_fmadd_ps(a, bv, cv));
            }
            for (; j < N; j++) c[j] += a_s * b[j];
        }
    }
#else
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            float a = alpha * A[i * lda + k];
            const float *b = B + k * ldb;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
    }
#endif
}

/* NT col-edge: each output c[j] is a dot product of two contiguous vectors
 * a[0..K) and b[j][0..K). That's exactly what the main NT kernel does but
 * for fewer cols — vectorise on the K axis. */
static void update_edge_nt(int M, int N, int K, float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float *C, int ldc)
{
#if PK_NEON
    for (int i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            const float *b = B + j * ldb;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                float32x4_t bv = vld1q_f32(b + k);
                acc = vfmaq_f32(acc, av, bv);
            }
            float s = vaddvq_f32(acc);
            for (; k < K; k++) s += a[k] * b[k];
            c[j] += alpha * s;
        }
    }
#elif PK_AVX2
    for (int i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            const float *b = B + j * ldb;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a + k);
                __m256 bv = _mm256_loadu_ps(b + k);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            float s = _mm_cvtss_f32(s1);
            for (; k < K; k++) s += a[k] * b[k];
            c[j] += alpha * s;
        }
    }
#else
    for (int i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            const float *b = B + j * ldb;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[k] * b[k];
            c[j] += alpha * acc;
        }
    }
#endif
}

/* ── SGEMM NN path (packed B, cache-blocked) ──────────────────────────────── */

#define MR 4
#define NR 16

/* Cache block sizes. Sized so B_pack (KC × NC) ≈ 256 KB, fitting L2 on both
 * Apple M-series (≥16 MB L2) and typical Intel/AMD chips (≥256 KB L2).
 * KC divides typical K values (1024, 4096) evenly; NC likewise for 1024/4096. */
#define KC 256
#define NC 256
#define B_PACK_BYTES ((size_t)KC * NC * sizeof(float))

/* Thread-local B_pack buffer, allocated once on first use and freed at
 * thread exit. Avoids aligned_alloc+free inside every sgemm call – which
 * adds up at ~2300 calls per long-audio encoder pass. */
static __thread float *tls_B_pack = NULL;

static float *get_B_pack(void)
{
    if (tls_B_pack == NULL)
        tls_B_pack = (float *)aligned_alloc(64, B_PACK_BYTES);
    return tls_B_pack;
}

/* Pack one [kb × NR] slice of B into a contiguous k-major, NR-interleaved
 * buffer. After packing, pack[k * NR + j] == B[k * ldb + j] so the existing
 * 4×16 micro-kernels consume the data unchanged with ldb_new = NR. */
static inline void pack_B_panel(const float *B, int ldb,
                                float *pack, int kb)
{
    for (int k = 0; k < kb; k++)
        memcpy(pack + k * NR, B + k * ldb, NR * sizeof(float));
}

static void pack_B(const float *B, int ldb, float *pack, int kb, int nb_full)
{
    int panels = nb_full / NR;
    for (int p = 0; p < panels; p++)
        pack_B_panel(B + p * NR, ldb, pack + p * (size_t)kb * NR, kb);
}

/* Pack a [kb × NR] slice of int8 B into fp32 pack, dequantising with `scale`.
 * Used by the quantised-B sgemm path. */
static inline void pack_B_panel_q(const signed char *B, int ldb,
                                  float *pack, int kb, float scale)
{
    for (int k = 0; k < kb; k++) {
        const signed char *src = B + k * ldb;
        float *dst = pack + k * NR;
#if PK_NEON
        int8x16_t i8 = vld1q_s8(src);
        int16x8_t lo = vmovl_s8(vget_low_s8(i8));
        int16x8_t hi = vmovl_s8(vget_high_s8(i8));
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
        float32x4_t vs = vdupq_n_f32(scale);
        vst1q_f32(dst + 0,  vmulq_f32(f0, vs));
        vst1q_f32(dst + 4,  vmulq_f32(f1, vs));
        vst1q_f32(dst + 8,  vmulq_f32(f2, vs));
        vst1q_f32(dst + 12, vmulq_f32(f3, vs));
#elif PK_AVX2
        __m128i i8 = _mm_loadu_si128((const __m128i *)src);
        __m256i i32_lo = _mm256_cvtepi8_epi32(i8);
        __m256i i32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(i8, 8));
        __m256 f_lo = _mm256_cvtepi32_ps(i32_lo);
        __m256 f_hi = _mm256_cvtepi32_ps(i32_hi);
        __m256 vs = _mm256_set1_ps(scale);
        _mm256_storeu_ps(dst + 0, _mm256_mul_ps(f_lo, vs));
        _mm256_storeu_ps(dst + 8, _mm256_mul_ps(f_hi, vs));
#else
        for (int j = 0; j < NR; j++) dst[j] = (float)src[j] * scale;
#endif
    }
}

static void pack_B_q(const signed char *B, int ldb, float *pack,
                     int kb, int nb_full, float scale)
{
    int panels = nb_full / NR;
    for (int p = 0; p < panels; p++)
        pack_B_panel_q(B + p * NR, ldb, pack + p * (size_t)kb * NR, kb, scale);
}

/* Args passed to the parallel sgemm worker. */
struct sgemm_nn_args {
    int M, N, K;
    float alpha;
    const float *A; int lda;
    const float *B; int ldb;
    float *C; int ldc;
};

/* Each thread: process its slice of the M dimension across all (kc, jc)
 * blocks, using its own private B_pack. Work is split on MR-aligned
 * boundaries so no two threads touch the same C row. */
static void sgemm_nn_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_nn_args *a = (struct sgemm_nn_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldb = a->ldb, ldc = a->ldc;
    float alpha = a->alpha;
    const float *A = a->A, *B = a->B;
    float *C = a->C;

    int Nf = (N / NR) * NR;

    /* Partition M into per-thread bands aligned to MR. */
    int tiles = (M + MR - 1) / MR;
    int first_tile = (tiles * tid) / nthreads;
    int last_tile  = (tiles * (tid + 1)) / nthreads;
    int m_start = first_tile * MR;
    int m_end   = last_tile  * MR;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;

    /* Per-thread packing buffer – reused across calls (thread-local). */
    float *B_pack = get_B_pack();
    if (!B_pack) {
        /* Correct-but-slow fallback if the pool alloc failed. */
        int i = m_start;
        while (i < m_end) {
            int mr = m_end - i; if (mr > MR) mr = MR;
            for (int j = 0; j < Nf; j += NR)
                dispatch_nn(mr, K, alpha, A + i * lda, lda,
                            B + j, ldb, C + i * ldc + j, ldc);
            if (Nf < N)
                update_edge_nn(mr, N - Nf, K, alpha, A + i * lda, lda,
                               B + Nf, ldb, C + i * ldc + Nf, ldc);
            i += mr;
        }
        return;
    }

    for (int kc = 0; kc < K; kc += KC) {
        int kb = (kc + KC <= K) ? KC : (K - kc);

        for (int jc = 0; jc < Nf; jc += NC) {
            int nb      = (jc + NC <= Nf) ? NC : (Nf - jc);
            int nb_full = (nb / NR) * NR;

            pack_B(B + kc * ldb + jc, ldb, B_pack, kb, nb_full);

            int i = m_start;
            while (i < m_end) {
                int mr = m_end - i; if (mr > MR) mr = MR;
                int panels = nb_full / NR;
                for (int p = 0; p < panels; p++) {
                    dispatch_nn(mr, kb, alpha,
                                A + i * lda + kc,              lda,
                                B_pack + (size_t)p * kb * NR,  NR,
                                C + i * ldc + jc + p * NR,     ldc);
                }
                i += mr;
            }
        }

        if (Nf < N) {
            int i = m_start;
            while (i < m_end) {
                int mr = m_end - i; if (mr > MR) mr = MR;
                update_edge_nn(mr, N - Nf, kb, alpha,
                               A + i * lda + kc,      lda,
                               B + kc * ldb + Nf,     ldb,
                               C + i * ldc + Nf,      ldc);
                i += mr;
            }
        }
    }
}

static void sgemm_nn(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    struct sgemm_nn_args args = {
        M, N, K, alpha, A, lda, B, ldb, C, ldc
    };
    pk_parallel(pk_pool(), sgemm_nn_worker, &args);
}

/* ── Quantised-B NN path (int8 B, fp32 A/C) ───────────────────────────────── */

struct sgemm_qb_args {
    int M, N, K;
    float alpha;
    const float       *A; int lda;
    const signed char *B; int ldb;
    float B_scale;
    float *C; int ldc;
};

static void sgemm_qb_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_qb_args *a = (struct sgemm_qb_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldb = a->ldb, ldc = a->ldc;
    float alpha = a->alpha;
    const float       *A = a->A;
    const signed char *B = a->B;
    float *C = a->C;
    float bs = a->B_scale;

    int Nf = (N / NR) * NR;

    int tiles = (M + MR - 1) / MR;
    int first_tile = (tiles * tid) / nthreads;
    int last_tile  = (tiles * (tid + 1)) / nthreads;
    int m_start = first_tile * MR;
    int m_end   = last_tile  * MR;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;

    float *B_pack = get_B_pack();
    if (!B_pack) {
        /* Correct-but-slow fallback: dequantise one column at a time. */
        for (int i = m_start; i < m_end; i++) {
            for (int k = 0; k < K; k++) {
                float ax = alpha * A[i * lda + k];
                for (int j = 0; j < N; j++)
                    C[i * ldc + j] += ax * bs * (float)B[k * ldb + j];
            }
        }
        return;
    }

    for (int kc = 0; kc < K; kc += KC) {
        int kb = (kc + KC <= K) ? KC : (K - kc);

        for (int jc = 0; jc < Nf; jc += NC) {
            int nb      = (jc + NC <= Nf) ? NC : (Nf - jc);
            int nb_full = (nb / NR) * NR;

            /* Pack + dequantise B slice into fp32 pack buffer. */
            pack_B_q(B + kc * ldb + jc, ldb, B_pack, kb, nb_full, bs);

            int i = m_start;
            while (i < m_end) {
                int mr = m_end - i; if (mr > MR) mr = MR;
                int panels = nb_full / NR;
                for (int p = 0; p < panels; p++) {
                    dispatch_nn(mr, kb, alpha,
                                A + i * lda + kc,              lda,
                                B_pack + (size_t)p * kb * NR,  NR,
                                C + i * ldc + jc + p * NR,     ldc);
                }
                i += mr;
            }
        }

        /* N-edge: per-element scalar dequant + multiply-accumulate. */
        if (Nf < N) {
            for (int i = m_start; i < m_end; i++) {
                float *c = C + i * ldc + Nf;
                for (int k = 0; k < kb; k++) {
                    float ax = alpha * A[i * lda + (kc + k)];
                    const signed char *b = B + (kc + k) * ldb + Nf;
                    int j = 0;
#if PK_NEON
                    int Nv = ((N - Nf) / 4) * 4;
                    float32x4_t va = vdupq_n_f32(ax * bs);
                    for (; j + 4 <= Nv; j += 4) {
                        int16x4_t i16 = vget_low_s16(
                            vmovl_s8(vld1_s8(b + j)));
                        float32x4_t bv = vcvtq_f32_s32(vmovl_s16(i16));
                        float32x4_t cv = vld1q_f32(c + j);
                        vst1q_f32(c + j, vfmaq_f32(cv, va, bv));
                    }
#endif
                    for (; j < N - Nf; j++) c[j] += ax * bs * (float)b[j];
                }
            }
        }
    }
}

void cblas_sgemm_qb(int M, int N, int K,
                    float alpha, const float *A, int lda,
                    const signed char *B, float B_scale, int ldb,
                    float beta, float *C, int ldc)
{
    scale_c(C, M, N, ldc, beta);
    if (alpha == 0.0f || B_scale == 0.0f) return;

    struct sgemm_qb_args args = {
        M, N, K, alpha, A, lda, B, ldb, B_scale, C, ldc
    };
    pk_parallel(pk_pool(), sgemm_qb_worker, &args);
}

/* ── Quantised-B NN path (int4 B with per-channel scales, fp32 A/C) ──────── */

/* Int4 packing format: B is [K, N] where each row of N values is stored
 * as N/2 bytes (2 signed 4-bit values per byte, low nibble first).
 * Per-channel means one scale per row (scales[K]).
 * ldb = N/2 (bytes per row). */

static inline void pack_B_panel_q4(const unsigned char *B, int ldb,
                                   const float *scales, int row_base,
                                   int col_base, int n_groups,
                                   float *pack, int kb)
{
    /* Unpack 2 int4 values per byte, dequantise with per-group scale.
     * NR = 16: we need 16 fp32 outputs per row-iteration = 8 bytes.
     * scales layout: [K, n_groups], accessed as scales[k * n_groups + g]. */
    int G = PK_INT4_GROUP;
    for (int k = 0; k < kb; k++) {
        const unsigned char *src = B + k * ldb;  /* ldb = N/2 bytes */
        float *dst = pack + k * NR;
        /* Look up the group scale for this (row, column-group) pair. */
        int group_idx = col_base / G;
        float s = scales[(row_base + k) * n_groups + group_idx];
#if PK_NEON
        /* Load 8 bytes = 16 int4 values */
        uint8x8_t raw = vld1_u8(src);
        /* Expand to 16-byte vector for processing */
        int8x16_t raw16 = vreinterpretq_s8_u8(vcombine_u8(raw, raw));
        /* Low nibbles: shift left 4, then arithmetic shift right 4 to sign-extend */
        int8x8_t lo8 = vshr_n_s8(vshl_n_s8(vget_low_s8(raw16), 4), 4);
        /* High nibbles: arithmetic shift right 4 */
        int8x8_t hi8 = vshr_n_s8(vget_low_s8(raw16), 4);
        /* Interleave: lo[0],hi[0],lo[1],hi[1],... = columns 0,1,2,3,... */
        int8x8x2_t zipped = vzip_s8(lo8, hi8);
        int8x16_t i8 = vcombine_s8(zipped.val[0], zipped.val[1]);
        /* Convert to fp32 and scale */
        int16x8_t lo16 = vmovl_s8(vget_low_s8(i8));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(i8));
        float32x4_t vs = vdupq_n_f32(s);
        vst1q_f32(dst + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vs));
        vst1q_f32(dst + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vs));
        vst1q_f32(dst + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vs));
        vst1q_f32(dst + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vs));
#elif PK_AVX2
        /* Load 8 bytes, unpack 16 int4 → 16 int8 → 16 fp32 */
        __m128i raw64 = _mm_loadl_epi64((const __m128i *)src);
        /* Low nibbles: (x << 4) >> 4 to sign-extend */
        __m128i lo_i8 = _mm_srai_epi16(_mm_slli_epi16(raw64, 12), 12);
        /* High nibbles: x >> 4 (arithmetic) */
        __m128i hi_i8 = _mm_srai_epi16(raw64, 4);
        /* Mask to keep only the sign-extended nibble in each byte */
        __m128i mask = _mm_set1_epi16(0x00FF);
        lo_i8 = _mm_and_si128(lo_i8, mask);
        hi_i8 = _mm_and_si128(hi_i8, mask);
        /* Interleave: lo[0],hi[0],lo[1],hi[1]... using unpacklo bytes */
        __m128i interleaved = _mm_or_si128(lo_i8, _mm_slli_epi16(hi_i8, 8));
        /* Sign-extend bytes to int32 and convert to float */
        __m256i i32_lo = _mm256_cvtepi8_epi32(interleaved);
        __m256i i32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(interleaved, 8));
        __m256 vs = _mm256_set1_ps(s);
        _mm256_storeu_ps(dst + 0, _mm256_mul_ps(_mm256_cvtepi32_ps(i32_lo), vs));
        _mm256_storeu_ps(dst + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(i32_hi), vs));
#else
        for (int j = 0; j < NR / 2; j++) {
            int lo = (int)(signed char)((src[j] & 0x0F) << 4) >> 4;
            int hi = (int)(signed char)(src[j] & 0xF0) >> 4;
            dst[2*j]     = lo * s;
            dst[2*j + 1] = hi * s;
        }
#endif
    }
}

static void pack_B_q4(const unsigned char *B, int ldb, const float *scales,
                      int row_base, int col_base, int n_groups,
                      float *pack, int kb, int nb_full)
{
    int panels = nb_full / NR;
    for (int p = 0; p < panels; p++)
        pack_B_panel_q4(B + p * (NR / 2), ldb, scales, row_base,
                        col_base + p * NR, n_groups,
                        pack + p * (size_t)kb * NR, kb);
}

struct sgemm_q4b_args {
    int M, N, K;
    float alpha;
    const float         *A; int lda;
    const unsigned char *B; int ldb;   /* ldb = N/2 (bytes per row) */
    const float         *scales;       /* [K * n_groups] per-group scales */
    int                  n_groups;     /* N / PK_INT4_GROUP */
    float *C; int ldc;
};

static void sgemm_q4b_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_q4b_args *a = (struct sgemm_q4b_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldb = a->ldb, ldc = a->ldc;
    float alpha = a->alpha;
    const float         *A = a->A;
    const unsigned char *B = a->B;
    const float         *scales = a->scales;
    int n_groups = a->n_groups;
    int G = PK_INT4_GROUP;
    float *C = a->C;

    int Nf = (N / NR) * NR;

    int tiles = (M + MR - 1) / MR;
    int first_tile = (tiles * tid) / nthreads;
    int last_tile  = (tiles * (tid + 1)) / nthreads;
    int m_start = first_tile * MR;
    int m_end   = last_tile  * MR;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;

    float *B_pack = get_B_pack();
    if (!B_pack) {
        /* Scalar fallback */
        for (int i = m_start; i < m_end; i++)
            for (int k = 0; k < K; k++) {
                const unsigned char *brow = B + k * ldb;
                for (int j = 0; j < N / 2; j++) {
                    int col = 2 * j;
                    float s = scales[k * n_groups + col / G];
                    int lo = (int)(signed char)((brow[j] & 0x0F) << 4) >> 4;
                    int hi = (int)(signed char)(brow[j] & 0xF0) >> 4;
                    C[i * ldc + col]     += alpha * A[i * lda + k] * (float)lo * s;
                    C[i * ldc + col + 1] += alpha * A[i * lda + k] * (float)hi * s;
                }
            }
        return;
    }

    for (int kc = 0; kc < K; kc += KC) {
        int kb = (kc + KC <= K) ? KC : (K - kc);

        for (int jc = 0; jc < Nf; jc += NC) {
            int nb      = (jc + NC <= Nf) ? NC : (Nf - jc);
            int nb_full = (nb / NR) * NR;

            pack_B_q4(B + kc * ldb + jc / 2, ldb, scales, kc,
                      jc, n_groups, B_pack, kb, nb_full);

            int i = m_start;
            while (i < m_end) {
                int mr = m_end - i; if (mr > MR) mr = MR;
                int panels = nb_full / NR;
                for (int p = 0; p < panels; p++) {
                    dispatch_nn(mr, kb, alpha,
                                A + i * lda + kc,              lda,
                                B_pack + (size_t)p * kb * NR,  NR,
                                C + i * ldc + jc + p * NR,     ldc);
                }
                i += mr;
            }
        }

        /* N-edge: scalar dequant */
        if (Nf < N) {
            for (int i = m_start; i < m_end; i++) {
                float *c = C + i * ldc + Nf;
                for (int k = 0; k < kb; k++) {
                    const unsigned char *brow = B + (kc + k) * ldb + Nf / 2;
                    for (int j = 0; j < (N - Nf) / 2; j++) {
                        int col = Nf + 2 * j;
                        float s = scales[(kc + k) * n_groups + col / G];
                        int lo = (int)(signed char)((brow[j] & 0x0F) << 4) >> 4;
                        int hi = (int)(signed char)(brow[j] & 0xF0) >> 4;
                        c[2*j]     += alpha * A[i * lda + (kc + k)] * (float)lo * s;
                        c[2*j + 1] += alpha * A[i * lda + (kc + k)] * (float)hi * s;
                    }
                }
            }
        }
    }
}

void cblas_sgemm_q4b(int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const unsigned char *B, const float *scales, int ldb,
                     float beta, float *C, int ldc)
{
    scale_c(C, M, N, ldc, beta);
    if (alpha == 0.0f) return;

    int n_groups = N / PK_INT4_GROUP;
    struct sgemm_q4b_args args = {
        M, N, K, alpha, A, lda, B, ldb, scales, n_groups, C, ldc
    };
    pk_parallel(pk_pool(), sgemm_q4b_worker, &args);
}

struct sgemm_nt_args {
    int M, N, K;
    float alpha;
    const float *A; int lda;
    const float *B; int ldb;
    float *C; int ldc;
};

/* NT doesn't need packing (B is already accessed contiguously on the K axis
 * per output element). Just partition M across threads. */
static void sgemm_nt_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_nt_args *a = (struct sgemm_nt_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldb = a->ldb, ldc = a->ldc;
    float alpha = a->alpha;
    const float *A = a->A, *B = a->B;
    float *C = a->C;

    int Nf = (N / NR) * NR;

    int tiles = (M + MR - 1) / MR;
    int first_tile = (tiles * tid) / nthreads;
    int last_tile  = (tiles * (tid + 1)) / nthreads;
    int m_start = first_tile * MR;
    int m_end   = last_tile  * MR;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;

    int i = m_start;
    while (i < m_end) {
        int mr = m_end - i; if (mr > MR) mr = MR;
        for (int j = 0; j < Nf; j += NR) {
            dispatch_nt(mr, K, alpha,
                        A + i * lda,     lda,
                        B + j * ldb,     ldb,
                        C + i * ldc + j, ldc);
        }
        if (Nf < N) {
            update_edge_nt(mr, N - Nf, K, alpha,
                           A + i * lda,       lda,
                           B + Nf * ldb,      ldb,
                           C + i * ldc + Nf,  ldc);
        }
        i += mr;
    }
}

static void sgemm_nt(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    struct sgemm_nt_args args = {
        M, N, K, alpha, A, lda, B, ldb, C, ldc
    };
    pk_parallel(pk_pool(), sgemm_nt_worker, &args);
}

/* ── Rarely used TN / TT paths (triple loop) ──────────────────────────────── */

static void sgemm_tn(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            float a = alpha * A[k * lda + i];
            const float *b = B + k * ldb;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
    }
}

static void sgemm_tt(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[k * lda + i] * B[j * ldb + k];
            C[i * ldc + j] += alpha * acc;
        }
}

/* ── Public SGEMM ─────────────────────────────────────────────────────────── */

void cblas_sgemm(CBLAS_ORDER order,
                 CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                 int M, int N, int K,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb,
                 float beta, float *C, int ldc)
{
    (void)order;
    scale_c(C, M, N, ldc, beta);
    if (alpha == 0.0f) return;

    int tA = (transA == CblasTrans);
    int tB = (transB == CblasTrans);
    if      (!tA && !tB) sgemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if (!tA &&  tB) sgemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if ( tA && !tB) sgemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else                 sgemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

/* Single-threaded sgemm — calls NN/NT workers directly as tid=0/nthreads=1
 * so the optimised SIMD+packing kernels are used without thread dispatch. */
void cblas_sgemm_st(CBLAS_ORDER order,
                    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                    int M, int N, int K,
                    float alpha, const float *A, int lda,
                    const float *B, int ldb,
                    float beta, float *C, int ldc)
{
    (void)order;
    scale_c(C, M, N, ldc, beta);
    if (alpha == 0.0f) return;

    int tA = (transA == CblasTrans);
    int tB = (transB == CblasTrans);
    if (!tA && !tB) {
        struct sgemm_nn_args args = { M, N, K, alpha, A, lda, B, ldb, C, ldc };
        sgemm_nn_worker(0, 1, &args);
    } else if (!tA && tB) {
        struct sgemm_nt_args args = { M, N, K, alpha, A, lda, B, ldb, C, ldc };
        sgemm_nt_worker(0, 1, &args);
    } else if (tA && !tB) {
        sgemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    } else {
        sgemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    }
}
