/*  main.c – Parakeet ASR inference from WAV file(s).
 *
 *  Usage:  parakeet <weights_dir> <audio.wav> [audio2.wav ...]
 *
 *  The weights_dir must contain:
 *    weights.bin, weights.json, tokenizer.model
 *
 *  Multiple WAV files are transcribed concurrently (one thread each),
 *  sharing a single loaded model.
 *
 *  Example:
 *    parakeet c_weights_fp32 speech.wav
 *    parakeet c_weights_int8 a.wav b.wav c.wav
 */

#include "parakeet.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* External: wav.c */
int wav_read(const char *path, float **out_samples, int *out_count, int *out_sr);

/* External: sentencepiece.c */
typedef struct SPModel SPModel;
SPModel *sp_model_load(const char *path);
void     sp_model_free(SPModel *m);
char    *sp_model_decode(const SPModel *m, const int *ids, int n);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Per-request context for threaded transcription ── */

typedef struct {
    int              id;        /* 0-based request index */
    const char      *wav_path;
    const PkModel   *model;    /* shared, read-only */
    const SPModel   *sp;       /* shared, read-only */
    char            *result;   /* output: caller frees */
    float            audio_dur;/* audio duration in seconds */
    double           elapsed;  /* total wall-clock seconds */
    double           t_mel;    /* mel spectrogram time */
    double           t_enc;    /* encoder time */
    double           t_dec;    /* decoder time */
    int              error;    /* 0 = success */
} TranscribeJob;

static void *transcribe_thread(void *arg)
{
    TranscribeJob *job = (TranscribeJob *)arg;
    double t0 = now_sec();

    /* Read audio */
    float *audio;
    int n_samples, sr;
    if (wav_read(job->wav_path, &audio, &n_samples, &sr) != 0) {
        fprintf(stderr, "[%d] failed to read WAV: %s\n", job->id, job->wav_path);
        job->error = 1;
        return NULL;
    }
    job->audio_dur = (float)n_samples / sr;
    fprintf(stderr, "[%d] %s: %d samples, %d Hz, %.2fs\n",
            job->id, job->wav_path, n_samples, sr, job->audio_dur);

    if (sr != PK_SAMPLE_RATE)
        fprintf(stderr, "[%d] WARNING: expected %d Hz, got %d Hz\n",
                job->id, PK_SAMPLE_RATE, sr);

    /* Mel spectrogram */
    double t1 = now_sec();
    float *mel;
    int n_frames;
    pk_mel_spectrogram(audio, n_samples, &mel, &n_frames);
    free(audio);
    job->t_mel = now_sec() - t1;

    /* Encoder */
    t1 = now_sec();
    float *enc_out;
    int enc_len;
    pk_encoder(job->model, mel, n_frames, &enc_out, &enc_len);
    free(mel);
    job->t_enc = now_sec() - t1;

    /* Greedy decode */
    t1 = now_sec();
    int *tokens;
    int n_tokens = pk_greedy_decode(job->model, enc_out, enc_len, &tokens);
    free(enc_out);
    job->t_dec = now_sec() - t1;

    /* Detokenize */
    if (job->sp) {
        job->result = sp_model_decode(job->sp, tokens, n_tokens);
    } else {
        /* Fallback: space-separated token IDs.
         * Max token value is 1024 (4 digits) + space = 5 chars each, +1 NUL. */
        size_t cap = (size_t)n_tokens * 6 + 1;
        job->result = (char *)malloc(cap);
        int off = 0;
        for (int i = 0; i < n_tokens; i++)
            off += snprintf(job->result + off, cap - off, "%d%s",
                            tokens[i], i < n_tokens - 1 ? " " : "");
    }
    free(tokens);

    job->elapsed = now_sec() - t0;
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <weights_dir> <audio.wav> [audio2.wav ...]\n",
                argv[0]);
        return 1;
    }

    /* Build weight paths */
    char bin_buf[1024], json_buf[1024], tok_buf[1024];
    snprintf(bin_buf,  sizeof(bin_buf),  "%s/weights.bin",     argv[1]);
    snprintf(json_buf, sizeof(json_buf), "%s/weights.json",    argv[1]);
    snprintf(tok_buf,  sizeof(tok_buf),  "%s/tokenizer.model", argv[1]);

    /* ── Load model (once) ── */
    fprintf(stderr, "loading model …\n");
    double t0 = now_sec();

    PkModel model;
    memset(&model, 0, sizeof(model));
    if (pk_load_weights(&model, bin_buf, json_buf) != 0) {
        fprintf(stderr, "failed to load weights\n");
        return 1;
    }

    SPModel *sp = sp_model_load(tok_buf);
    if (!sp)
        fprintf(stderr, "  tokenizer not found at %s – will output token IDs\n",
                tok_buf);

    /* Eagerly init the thread pool so the first encoder call doesn't pay for it. */
    pk_pool();

    fprintf(stderr, "  loaded in %.2fs\n", now_sec() - t0);

    /* ── Launch one thread per WAV file ── */
    int n_jobs = argc - 2;
    TranscribeJob *jobs = (TranscribeJob *)calloc(n_jobs, sizeof(TranscribeJob));
    pthread_t     *tids = (pthread_t *)malloc(n_jobs * sizeof(pthread_t));

    /* Suppress per-block encoder progress when running concurrently —
     * multiple threads writing \r lines to stderr produces garbage. */
    if (n_jobs > 1)
        pk_verbose = 0;

    fprintf(stderr, "transcribing %d file%s …\n", n_jobs, n_jobs > 1 ? "s" : "");
    t0 = now_sec();

    for (int i = 0; i < n_jobs; i++) {
        jobs[i].id       = i;
        jobs[i].wav_path = argv[2 + i];
        jobs[i].model    = &model;
        jobs[i].sp       = sp;

        if (n_jobs == 1) {
            /* Single file: run directly, no extra thread. */
            transcribe_thread(&jobs[i]);
        } else {
            if (pthread_create(&tids[i], NULL, transcribe_thread, &jobs[i]) != 0) {
                fprintf(stderr, "[%d] pthread_create failed\n", i);
                jobs[i].error = 1;
            }
        }
    }

    if (n_jobs > 1) {
        for (int i = 0; i < n_jobs; i++)
            if (!jobs[i].error)
                pthread_join(tids[i], NULL);
    }

    double total = now_sec() - t0;

    /* ── Print results in order ── */
    for (int i = 0; i < n_jobs; i++) {
        if (jobs[i].error) continue;
        float rtf = (jobs[i].audio_dur > 0.0f)
                   ? (float)jobs[i].elapsed / jobs[i].audio_dur : 0.0f;
        fprintf(stderr, "[%d] %s  mel=%.2fs  enc=%.2fs  dec=%.2fs  "
                        "total=%.2fs  audio=%.2fs  RTF=%.2f\n",
                i, jobs[i].wav_path,
                jobs[i].t_mel, jobs[i].t_enc, jobs[i].t_dec,
                jobs[i].elapsed, jobs[i].audio_dur, rtf);
        printf("%s%s%s\n",
               n_jobs > 1 ? jobs[i].wav_path : "",
               n_jobs > 1 ? ": " : "",
               jobs[i].result);
        free(jobs[i].result);
    }

    if (n_jobs > 1)
        fprintf(stderr, "total: %.2fs\n", total);

    free(jobs);
    free(tids);
    sp_model_free(sp);
    pk_free_weights(&model);
    return 0;
}
