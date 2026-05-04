/*  decoder.c – LSTM prediction network + Joint network + greedy RNN-T. */

#include "parakeet.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "blas.h"

/* ── Decoder step: embedding → 2-layer LSTM → pred_out[640] ─────────────── */

void pk_decoder_step(const PkModel *m, int token_id,
                     LSTMState *state, float *pred_out)
{
    const DecoderWeights *d = &m->dec;
    int H = PK_LSTM_HIDDEN;

    /* Embedding lookup: token_id → [640] */
    const float *emb = d->embed_w + token_id * PK_EMBED_DIM;

    /* LSTM layer 0: input = emb [640], hidden = 640 */
    /* Wi is stored as [1, 4H, input_size] → skip leading dim of 1 */
    pk_lstm_step(emb, PK_EMBED_DIM,
                 d->lstm0_Wi, d->lstm0_Wr, d->lstm0_B,
                 state->h[0], state->c[0], H);

    /* LSTM layer 1: input = h[0] [640], hidden = 640 */
    pk_lstm_step(state->h[0], H,
                 d->lstm1_Wi, d->lstm1_Wr, d->lstm1_B,
                 state->h[1], state->c[1], H);

    /* Output is h[1] */
    memcpy(pred_out, state->h[1], H * sizeof(float));
}

/* ── Joint network ───────────────────────────────────────────────────────── */

void pk_joint(const PkModel *m, const float *enc_t, const float *pred,
              float *logits)
{
    const DecoderWeights *d = &m->dec;

    /* enc_proj = enc_t @ joint.enc.weight + bias
     * enc_t: [1024], weight: [1024, 640] row-major → y = w^T @ x
     * sgemv Trans: y[640] = w[1024,640]^T @ x[1024] */
    float enc_proj[PK_JOINT_DIM];
    cblas_sgemv(CblasRowMajor, CblasTrans,
                PK_D_MODEL, PK_JOINT_DIM, 1.0f,
                d->joint_enc_w, PK_JOINT_DIM, enc_t, 1,
                0.0f, enc_proj, 1);
    pk_bias_add(enc_proj, d->joint_enc_b, PK_JOINT_DIM);

    /* pred_proj = pred @ joint.pred.weight + bias
     * pred: [640], weight: [640, 640] → y = w^T @ x */
    float pred_proj[PK_JOINT_DIM];
    cblas_sgemv(CblasRowMajor, CblasTrans,
                PK_JOINT_DIM, PK_JOINT_DIM, 1.0f,
                d->joint_pred_w, PK_JOINT_DIM, pred, 1,
                0.0f, pred_proj, 1);
    pk_bias_add(pred_proj, d->joint_pred_b, PK_JOINT_DIM);

    /* Add + ReLU */
    float hidden[PK_JOINT_DIM];
    for (int i = 0; i < PK_JOINT_DIM; i++) {
        hidden[i] = enc_proj[i] + pred_proj[i];
        if (hidden[i] < 0.0f) hidden[i] = 0.0f;
    }

    /* out = hidden @ joint.out.weight + bias
     * hidden: [640], weight: [640, 1025] → y = w^T @ x */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                PK_JOINT_DIM, PK_VOCAB, 1.0f,
                d->joint_out_w, PK_VOCAB, hidden, 1,
                0.0f, logits, 1);
    pk_bias_add(logits, d->joint_out_b, PK_VOCAB);
}

/* ── Greedy RNN-T decoding ───────────────────────────────────────────────── */

int pk_greedy_decode(const PkModel *m, const float *enc_out, int T,
                     int **out_tokens)
{
    /* enc_out: [T × 1024] (time-first, row-major) */
    LSTMState state;
    memset(&state, 0, sizeof(state));

    int max_tokens = T * 10;
    int *tokens = (int *)malloc(max_tokens * sizeof(int));
    int n_tokens = 0;

    float pred_out[PK_LSTM_HIDDEN];
    float logits[PK_VOCAB];

    /* Initial decoder step with blank token */
    int last_token = PK_BLANK;
    pk_decoder_step(m, last_token, &state, pred_out);

    for (int t = 0; t < T; t++) {
        /* Encoder output for time step t: contiguous [1024] row */
        const float *enc_t = enc_out + t * PK_D_MODEL;

        /* Inner loop: emit tokens until blank */
        for (int s = 0; s < 10; s++) {
            pk_joint(m, enc_t, pred_out, logits);

            /* Argmax */
            int best = 0;
            float best_val = logits[0];
            for (int v = 1; v < PK_VOCAB; v++) {
                if (logits[v] > best_val) {
                    best_val = logits[v];
                    best = v;
                }
            }

            if (best == PK_BLANK)
                break;

            if (n_tokens >= max_tokens) {
                max_tokens *= 2;
                tokens = (int *)realloc(tokens, max_tokens * sizeof(int));
            }
            tokens[n_tokens++] = best;
            last_token = best;

            pk_decoder_step(m, last_token, &state, pred_out);
        }
    }

    *out_tokens = tokens;
    return n_tokens;
}
