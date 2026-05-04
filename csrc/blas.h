/*  blas.h – Minimal CBLAS-compatible API (sgemm + sgemv, row-major only).
 *
 *  Hand-rolled replacement for Apple Accelerate / OpenBLAS. Only the routines
 *  and enum values actually used by parakeet.c are provided. Values of the
 *  enum constants match the upstream CBLAS ABI so the call sites are
 *  source-compatible.
 */
#ifndef PK_BLAS_H
#define PK_BLAS_H

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum {
    CblasNoTrans = 111,
    CblasTrans   = 112
} CBLAS_TRANSPOSE;

/* Y = alpha * op(A) * X + beta * Y
 *   op(A) = A         if trans == CblasNoTrans   (A is M×N, Y is M, X is N)
 *   op(A) = A^T       if trans == CblasTrans     (A is M×N, Y is N, X is M)
 *
 * Row-major: A[i, j] = A[i * lda + j].
 * incX, incY are element strides through X and Y respectively.
 */
void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                 int M, int N,
                 float alpha, const float *A, int lda,
                 const float *X, int incX,
                 float beta, float *Y, int incY);

/* C = alpha * op(A) * op(B) + beta * C
 *   op(A) is M×K, op(B) is K×N, C is M×N.
 *   Row-major: A[i,j] = A[i*lda + j] (and likewise B, C).
 */
void cblas_sgemm(CBLAS_ORDER order,
                 CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                 int M, int N, int K,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb,
                 float beta, float *C, int ldc);

/* Single-threaded sgemm — same interface as cblas_sgemm but never dispatches
 * to the thread pool. Use this inside pk_parallel workers to avoid deadlock. */
void cblas_sgemm_st(CBLAS_ORDER order,
                    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                    int M, int N, int K,
                    float alpha, const float *A, int lda,
                    const float *B, int ldb,
                    float beta, float *C, int ldc);

/* Quantised-B sgemm: A is fp32 row-major, B is int8 row-major with a single
 * fp32 scale (symmetric, zero-point = 0). Only NoTrans on A, NoTrans on B is
 * supported. The dequantisation happens inside the pack step so the inner
 * kernel remains fp32. */
void cblas_sgemm_qb(int M, int N, int K,
                    float alpha, const float *A, int lda,
                    const signed char *B, float B_scale, int ldb,
                    float beta, float *C, int ldc);

/* Int4 quantised-B sgemm: B is packed int4 (2 values per byte, low nibble first)
 * with per-channel (per-row) fp32 scales. ldb = N/2 (bytes per row).
 * N must be even. */
void cblas_sgemm_q4b(int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const unsigned char *B, const float *scales, int ldb,
                     float beta, float *C, int ldc);

#endif /* PK_BLAS_H */
