/*  mel.c – Mel spectrogram computation matching NeMo's preprocessor.
 *
 *  Config (from parakeet-unified-en-0.6b):
 *    sample_rate = 16000, n_fft = 512, win_length = 400, hop_length = 160,
 *    n_mels = 128, power = 2.0, norm = "slaney", mel_scale = "slaney",
 *    center = true, window = hann
 */

#include "parakeet.h"
#include "threadpool.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Hann window ─────────────────────────────────────────────────────────── */

static void hann_window(float *w, int N)
{
    for (int i = 0; i < N; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / N));
}

/* ── Real FFT (radix-2 Cooley-Tukey) ─────────────────────────────────────── */

/* In-place FFT of N complex values (interleaved real/imag). N must be power of 2. */
static void fft_inplace(float *data, int N)
{
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2 * i], ti = data[2 * i + 1];
            data[2 * i] = data[2 * j];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j] = tr;
            data[2 * j + 1] = ti;
        }
    }

    /* Cooley-Tukey butterflies */
    for (int len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * (float)M_PI / len;
        float wr = cosf(angle), wi = sinf(angle);
        for (int i = 0; i < N; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j, v = i + j + len / 2;
                float tr = data[2 * v] * cur_r - data[2 * v + 1] * cur_i;
                float ti = data[2 * v] * cur_i + data[2 * v + 1] * cur_r;
                data[2 * v]     = data[2 * u]     - tr;
                data[2 * v + 1] = data[2 * u + 1] - ti;
                data[2 * u]     += tr;
                data[2 * u + 1] += ti;
                float new_r = cur_r * wr - cur_i * wi;
                float new_i = cur_r * wi + cur_i * wr;
                cur_r = new_r;
                cur_i = new_i;
            }
        }
    }
}

/* ── Mel filterbank (Slaney normalization) ───────────────────────────────── */

static float hz_to_mel(float hz)
{
    /* Slaney mel scale: linear below 1000 Hz, log above */
    if (hz < 1000.0f)
        return hz * 3.0f / 200.0f;
    else
        return 15.0f + 27.0f * logf(hz / 1000.0f) / logf(6.4f);
}

static float mel_to_hz(float mel)
{
    if (mel < 15.0f)
        return mel * 200.0f / 3.0f;
    else
        return 1000.0f * expf((mel - 15.0f) * logf(6.4f) / 27.0f);
}

/* Build mel filterbank: [n_mels × (n_fft/2+1)] */
static float *make_mel_filterbank(int n_mels, int n_fft, int sr)
{
    int n_freqs = n_fft / 2 + 1;
    float *fb = (float *)calloc(n_mels * n_freqs, sizeof(float));

    float mel_lo = hz_to_mel(0.0f);
    float mel_hi = hz_to_mel((float)sr / 2.0f);

    float *mels = (float *)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++)
        mels[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1));

    float *freqs = (float *)malloc(n_freqs * sizeof(float));
    for (int i = 0; i < n_freqs; i++)
        freqs[i] = (float)sr * i / n_fft;

    for (int m = 0; m < n_mels; m++) {
        float lo = mels[m], center = mels[m + 1], hi = mels[m + 2];
        /* Slaney normalization: 2 / (hi - lo) */
        float norm = 2.0f / (hi - lo);
        for (int f = 0; f < n_freqs; f++) {
            float hz = freqs[f];
            float val = 0.0f;
            if (hz >= lo && hz <= center)
                val = (hz - lo) / (center - lo);
            else if (hz > center && hz <= hi)
                val = (hi - hz) / (hi - center);
            fb[m * n_freqs + f] = val * norm;
        }
    }

    free(mels);
    free(freqs);
    return fb;
}

/* ── Parallel mel spectrogram ───────────────────────────────────────────── */

struct mel_stft_args {
    const float *padded;
    int          padded_len;
    const float *win;
    int          win_len;
    const float *fb;
    int          n_fft;
    int          n_freqs;
    int          n_mels;
    int          n_frames;
    int          hop_len;
    float       *mel;
};

static void mel_stft_worker(int tid, int nt, void *aux)
{
    struct mel_stft_args *a = (struct mel_stft_args *)aux;
    int f0 = (a->n_frames * tid) / nt;
    int f1 = (a->n_frames * (tid + 1)) / nt;
    if (f0 >= f1) return;

    /* Per-thread scratch buffers */
    float *fft_buf = (float *)malloc(2 * a->n_fft * sizeof(float));
    float *power   = (float *)malloc(a->n_freqs * sizeof(float));

    for (int f = f0; f < f1; f++) {
        int start = f * a->hop_len;

        memset(fft_buf, 0, 2 * a->n_fft * sizeof(float));
        for (int i = 0; i < a->win_len && (start + i) < a->padded_len; i++)
            fft_buf[2 * i] = a->padded[start + i] * a->win[i];

        fft_inplace(fft_buf, a->n_fft);

        for (int k = 0; k < a->n_freqs; k++) {
            float re = fft_buf[2 * k], im = fft_buf[2 * k + 1];
            power[k] = re * re + im * im;
        }

        for (int m = 0; m < a->n_mels; m++) {
            float val = 0.0f;
            for (int k = 0; k < a->n_freqs; k++)
                val += a->fb[m * a->n_freqs + k] * power[k];
            a->mel[m * a->n_frames + f] = logf(val + 5.96046448e-08f);
        }
    }

    free(power);
    free(fft_buf);
}

/* ── Mel spectrogram ─────────────────────────────────────────────────────── */

int pk_mel_spectrogram(const float *audio, int n_samples,
                       float **out_mel, int *out_frames)
{
    int n_fft    = PK_N_FFT;
    int win_len  = PK_WIN_LEN;
    int hop_len  = PK_HOP_LEN;
    int n_mels   = PK_N_MELS;
    int n_freqs  = n_fft / 2 + 1;

    /* Center padding */
    int pad = n_fft / 2;
    int padded_len = n_samples + 2 * pad;
    float *padded = (float *)calloc(padded_len, sizeof(float));
    memcpy(padded + pad, audio, n_samples * sizeof(float));

    /* Reflect padding */
    for (int i = 0; i < pad; i++) {
        padded[pad - 1 - i] = audio[i + 1 < n_samples ? i + 1 : 0];
        int j = n_samples - 2 - i;
        padded[pad + n_samples + i] = j >= 0 ? audio[j] : audio[0];
    }

    int n_frames = (padded_len - n_fft) / hop_len + 1;

    /* Window */
    float *win = (float *)malloc(win_len * sizeof(float));
    hann_window(win, win_len);

    /* Mel filterbank */
    float *fb = make_mel_filterbank(n_mels, n_fft, PK_SAMPLE_RATE);

    /* Output: [n_mels × n_frames] */
    float *mel = (float *)calloc(n_mels * n_frames, sizeof(float));

    /* STFT + mel filterbank — parallel across frames */
    struct mel_stft_args args = {
        padded, padded_len, win, win_len, fb,
        n_fft, n_freqs, n_mels, n_frames, hop_len, mel
    };
    pk_parallel(pk_pool(), mel_stft_worker, &args);

    free(fb);
    free(win);
    free(padded);

    /* Per-channel normalization: (mel - mean) / (std + 1e-5) */
    for (int m = 0; m < n_mels; m++) {
        float *row = mel + m * n_frames;
        double mean = 0.0;
        for (int f = 0; f < n_frames; f++) mean += row[f];
        mean /= n_frames;

        double var = 0.0;
        for (int f = 0; f < n_frames; f++) {
            double d = row[f] - mean;
            var += d * d;
        }
        var /= n_frames;
        float inv_std = (float)(1.0 / sqrt(var + 1e-5));

        for (int f = 0; f < n_frames; f++)
            row[f] = (row[f] - (float)mean) * inv_std;
    }

    *out_mel = mel;
    *out_frames = n_frames;
    return 0;
}
