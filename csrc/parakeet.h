/*  parakeet.h – Pure-C inference for parakeet-unified-en-0.6b
 *
 *  Encoder  : FastConformer (24 blocks, d=1024, 8 heads, ff=4096)
 *  Decoder  : 2-layer LSTM (h=640) + Joint network
 *  Decoding : Greedy RNN-T
 */
#ifndef PARAKEET_H
#define PARAKEET_H

#include <stdint.h>
#include <stddef.h>

/* ── Model constants ──────────────────────────────────────────────────────── */

#define PK_D_MODEL      1024
#define PK_N_HEADS      8
#define PK_HEAD_DIM     (PK_D_MODEL / PK_N_HEADS)   /* 128 */
#define PK_FF_DIM       4096
#define PK_N_BLOCKS     24
#define PK_CONV_KERNEL  9

#define PK_N_MELS       128
#define PK_SUB_CHANNELS 256
#define PK_SUB_FACTOR   8     /* total subsampling: 3× stride-2 = 8× */

#define PK_LSTM_HIDDEN  640
#define PK_LSTM_LAYERS  2
#define PK_VOCAB        1025  /* 1024 SentencePiece + 1 blank */
#define PK_BLANK        1024
#define PK_JOINT_DIM    640

#define PK_EMBED_DIM    640

/* Audio */
#define PK_SAMPLE_RATE  16000
#define PK_N_FFT        512
#define PK_WIN_LEN      400
#define PK_HOP_LEN      160

/* ── Weight structures ────────────────────────────────────────────────────── */

#define PK_INT4_GROUP  32    /* columns per int4 quantization group */

/* Weight holder for sgemm weight matrices – fp32, int8, or int4 quant.
 *
 *   bits == 32 : `data` is a const float *      (fp32 path)
 *   bits == 8  : `data` is a const int8_t *     (int8 per-tensor, dequant during pack_B)
 *   bits == 4  : `data` is packed int4 pairs    (int4 per-group, 2 values per byte,
 *                `scales` points to [K * N/GROUP] float scales)
 *
 * Symmetric quantisation (zero_point = 0) is assumed throughout. */
typedef struct {
    const void  *data;
    float        scale;     /* per-tensor scale (int8), unused for fp32/int4 */
    const float *scales;    /* per-channel scale array [K rows], int4 only */
    int          bits;      /* 32, 8, or 4 */
} Weight;

typedef struct {
    /* pre_encode.conv subsampling – kept fp32 even in int8 mode because the
     * conv2d kernel doesn't go through the packed-B sgemm path. */
    const float *conv0_w;   /* [256, 1, 3, 3]   */
    const float *conv0_b;   /* [256]            */
    const float *conv2_w;   /* [256, 1, 3, 3]   */
    const float *conv2_b;
    const float *conv3_w;   /* [256, 256, 1, 1] */
    const float *conv3_b;
    const float *conv5_w;   /* [256, 1, 3, 3]   */
    const float *conv5_b;
    const float *conv6_w;   /* [256, 256, 1, 1] */
    const float *conv6_b;

    /* pre_encode.out linear: 4096 → 1024. This IS an sgemm weight. */
    Weight       out_w;     /* [4096, 1024]     */
    const float *out_b;     /* [1024]           */
} PreEncodeWeights;

typedef struct {
    /* Feed-forward: linear1 [d→ff] + Swish + linear2 [ff→d] */
    Weight       linear1_w; /* [1024, 4096] */
    const float *linear1_b; /* [4096]       */
    Weight       linear2_w; /* [4096, 1024] */
    const float *linear2_b; /* [1024]       */
} FFWeights;

typedef struct {
    Weight       linear_q_w;    /* [1024, 1024] */
    const float *linear_q_b;
    Weight       linear_k_w;
    const float *linear_k_b;
    Weight       linear_v_w;
    const float *linear_v_b;
    Weight       linear_pos_w;
    Weight       linear_out_w;
    const float *linear_out_b;
    const float *pos_bias_u;    /* [8, 128] – fp32 */
    const float *pos_bias_v;    /* [8, 128] – fp32 */

    /* Fused QKV: Wq|Wk|Wv concatenated along columns at load time.
     * Eliminates 2 of 3 pk_parallel dispatches for QKV projection. */
    Weight       linear_qkv_w;  /* [D, 3*D] = [1024, 3072] */
    float       *linear_qkv_b;  /* [3*D] = [3072], owned */
    void        *qkv_w_buf;     /* owned storage for fused weight data */
} MHAWeights;

typedef struct {
    const float *pw1_w;  /* [2048, 1024, 1] -> treat as [2048, 1024] */
    const float *pw1_b;  /* [2048] */
    const float *dw_w;   /* [1024, 1, 9]    -> treat as [1024, 9]   */
    const float *dw_b;   /* [1024] */
    const float *pw2_w;  /* [1024, 1024, 1] -> treat as [1024, 1024] */
    const float *pw2_b;  /* [1024] */
} ConvModWeights;

typedef struct {
    const float *norm_ff1_w, *norm_ff1_b;       /* [1024] each */
    FFWeights ff1;
    const float *norm_sa_w, *norm_sa_b;         /* [1024] each */
    MHAWeights mha;
    const float *norm_conv_w, *norm_conv_b;     /* [1024] each */
    ConvModWeights conv;
    const float *norm_ff2_w, *norm_ff2_b;       /* [1024] each */
    FFWeights ff2;
    const float *norm_out_w, *norm_out_b;       /* [1024] each */
} ConformerBlock;

#define PK_PE_TABLE_SIZE 9999   /* pre-computed positional encoding size */
#define PK_PE_CENTER     4999   /* (size-1)/2 */

/* Chunked encoder inference — splits long sequences into fixed-size chunks
 * to reduce peak memory from O(T²) to O(C²). */
#define PK_CHUNK_SIZE      400  /* encoder frames per chunk (~25s audio)   */
#define PK_CHUNK_OVERLAP   8    /* overlap frames on each side of a chunk  */
#define PK_CHUNK_THRESHOLD 500  /* auto-chunk when T' exceeds this         */

typedef struct {
    PreEncodeWeights pre_encode;
    const float     *pe_table;  /* [9999, 1024] pre-computed positional encoding */
    ConformerBlock   blocks[PK_N_BLOCKS];
} EncoderWeights;

typedef struct {
    const float *embed_w;           /* [1025, 640]     */

    /* ONNX LSTM format: W_i [1, 4*h, input], W_r [1, 4*h, h], B [1, 8*h] */
    const float *lstm0_Wi;          /* [1, 2560, 640]  */
    const float *lstm0_Wr;          /* [1, 2560, 640]  */
    const float *lstm0_B;           /* [1, 5120]       */
    const float *lstm1_Wi;          /* [1, 2560, 640]  */
    const float *lstm1_Wr;          /* [1, 2560, 640]  */
    const float *lstm1_B;           /* [1, 5120]       */

    /* Joint network (always fp32 even in int8 mode; dequantised at
     * extraction for simplicity since they're small). */
    const float *joint_enc_w;       /* [1024, 640]     */
    const float *joint_enc_b;       /* [640]           */
    const float *joint_pred_w;      /* [640, 640]      */
    const float *joint_pred_b;      /* [640]           */
    const float *joint_out_w;       /* [640, 1025]     */
    const float *joint_out_b;       /* [1025]          */
} DecoderWeights;

typedef struct {
    EncoderWeights enc;
    DecoderWeights dec;
} PkModel;

/* ── LSTM state ───────────────────────────────────────────────────────────── */

typedef struct {
    float h[PK_LSTM_LAYERS][PK_LSTM_HIDDEN];
    float c[PK_LSTM_LAYERS][PK_LSTM_HIDDEN];
} LSTMState;

/* ── Verbosity ───────────────────────────────────────────────────────────── */

/* Controls encoder progress output to stderr. Default 1 (on).
 * Set to 0 before calling pk_encoder to suppress per-block/chunk lines. */
extern int pk_verbose;

/* ── API ──────────────────────────────────────────────────────────────────── */

/* Load weights from binary + JSON manifest. Returns 0 on success. */
int  pk_load_weights(PkModel *m, const char *bin_path, const char *json_path);
void pk_free_weights(PkModel *m);

/* Mel spectrogram: audio (float32 mono 16kHz) → mel [n_mels × n_frames].
 * Caller must free *out_mel. Returns n_frames. */
int  pk_mel_spectrogram(const float *audio, int n_samples,
                        float **out_mel, int *out_frames);

/* Encoder: mel [128 × T] → enc_out [T' × 1024] (time-first, row-major).
 * Auto-dispatches to chunked path when T' > PK_CHUNK_THRESHOLD.
 * Caller must free *out. Returns 0 on success. */
int  pk_encoder(const PkModel *m, const float *mel, int n_frames,
                float **out, int *out_len);

/* Force chunked encoder regardless of sequence length. Same API. */
int  pk_encoder_chunked(const PkModel *m, const float *mel, int n_frames,
                        float **out, int *out_len);

/* Decoder step: one prediction network step.
 * token_id → pred_out [640]. Mutates state in-place. */
void pk_decoder_step(const PkModel *m, int token_id,
                     LSTMState *state, float *pred_out);

/* Joint: enc_t [1024] + pred [640] → logits [1025]. */
void pk_joint(const PkModel *m, const float *enc_t, const float *pred,
              float *logits);

/* Greedy RNN-T decode: enc_out [T' × 1024] (time-first) → token IDs.
 * Returns number of tokens. Caller must free *out_tokens. */
int  pk_greedy_decode(const PkModel *m, const float *enc_out, int T,
                      int **out_tokens);

/* ── Tensor ops (tensor_ops.c) ────────────────────────────────────────────── */

/* C = alpha * A @ B + beta * C  (row-major, CBLAS) */
void pk_matmul(const float *A, const float *B, float *C,
               int M, int N, int K, float alpha, float beta);

/* Same but B is int8 with per-tensor scale; dequantisation happens inside
 * the packed sgemm inner kernel. A and C remain fp32. */
void pk_matmul_q(const float *A, const signed char *B, float B_scale, float *C,
                 int M, int N, int K, float alpha, float beta);

/* Dispatch wrapper: picks fp32 or int8 matmul based on W->scale. */
void pk_wmatmul(const float *A, const Weight *W, float *C,
                int M, int N, int K, float alpha, float beta);

void pk_layer_norm(const float *x, const float *gamma, const float *beta_,
                   float *out, int len);

/* Row-batched layer norm: apply pk_layer_norm to each of n_rows rows of
 * length D, in parallel. x and out are [n_rows × D] row-major.
 * gamma and beta_ are shared across rows, shape [D]. */
void pk_layer_norm_rows(const float *x, const float *gamma, const float *beta_,
                        float *out, int n_rows, int D);

/* In-place row-batched layer norm — normalises x in-place using a small
 * stack buffer per thread. Avoids a separate output allocation + memcpy. */
void pk_layer_norm_rows_inplace(float *x, const float *gamma, const float *beta_,
                                int n_rows, int D);

/* x = x * sigmoid(x) */
void pk_swish(float *x, int len);

void pk_relu(float *x, int len);

void pk_softmax(float *x, int len);

/* Row-batched softmax: apply pk_softmax to each of n_rows rows of length len,
 * in parallel. x is [n_rows × len] row-major, softmax applied over the last dim. */
void pk_softmax_rows(float *x, int n_rows, int len);

/* bias add: x[i] += b[i] for i in [0, len) */
void pk_bias_add(float *x, const float *b, int len);

/* 1D depthwise convolution: groups=C, kernel_size=K, stride=1, causal padding
 * x: [C × T], w: [C × K], b: [C], out: [C × T] */
void pk_depthwise_conv1d(const float *x, const float *w, const float *b,
                         float *out, int C, int T, int K);

/* Conv2D: [Cout, Cin, Kh, Kw] stride=(sh, sw) pad=(ph, pw)
 * x: [Cin × H × W], out: [Cout × H' × W'] */
void pk_conv2d(const float *x, const float *w, const float *b,
               float *out, int Cin, int H, int W,
               int Cout, int Kh, int Kw, int sh, int sw, int ph, int pw,
               int groups);

/* 1D pointwise (kernel=1) convolution = matmul: x[T, Cin] @ w[Cin, Cout] + b
 * x: [Cout × T] (channel-first), w: [Cout, Cin, 1] */
void pk_pointwise_conv1d(const float *x, const float *w, const float *b,
                         float *out, int Cin, int Cout, int T);

/* LSTM cell: one time-step. Wi [4h, input], Wr [4h, h], Bi+Br [8h] */
void pk_lstm_step(const float *x_t, int input_size,
                  const float *Wi, const float *Wr, const float *B,
                  float *h, float *c, int hidden_size);

#endif /* PARAKEET_H */
