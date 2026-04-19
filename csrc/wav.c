/*  wav.c – Minimal WAV file reader (16-bit PCM or 32-bit float, mono). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int wav_read(const char *path, float **out_samples, int *out_count, int *out_sr)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    /* RIFF header */
    char riff[4];
    uint32_t file_size;
    char wave[4];
    fread(riff, 1, 4, f);
    fread(&file_size, 4, 1, f);
    fread(wave, 1, 4, f);

    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "%s: not a WAV file\n", path);
        fclose(f);
        return -1;
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    int found_fmt = 0, found_data = 0;

    while (!found_data) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            uint32_t byte_rate;
            uint16_t block_align;
            fread(&byte_rate, 4, 1, f);
            fread(&block_align, 2, 1, f);
            fread(&bits_per_sample, 2, 1, f);
            /* Skip any extra format bytes */
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
            found_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = 1;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        fprintf(stderr, "%s: missing fmt or data chunk\n", path);
        fclose(f);
        return -1;
    }

    if (num_channels != 1) {
        fprintf(stderr, "%s: expected mono, got %d channels\n", path, num_channels);
        fclose(f);
        return -1;
    }

    int n_samples;
    float *samples;

    if (audio_format == 1 && bits_per_sample == 16) {
        /* PCM 16-bit */
        n_samples = data_size / 2;
        int16_t *raw = (int16_t *)malloc(data_size);
        fread(raw, 1, data_size, f);
        samples = (float *)malloc(n_samples * sizeof(float));
        for (int i = 0; i < n_samples; i++)
            samples[i] = raw[i] / 32768.0f;
        free(raw);
    } else if (audio_format == 3 && bits_per_sample == 32) {
        /* IEEE float 32-bit */
        n_samples = data_size / 4;
        samples = (float *)malloc(data_size);
        fread(samples, 1, data_size, f);
    } else {
        fprintf(stderr, "%s: unsupported format %d/%d-bit\n",
                path, audio_format, bits_per_sample);
        fclose(f);
        return -1;
    }

    fclose(f);

    *out_samples = samples;
    *out_count = n_samples;
    *out_sr = (int)sample_rate;
    return 0;
}
