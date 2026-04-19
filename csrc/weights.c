/*  weights.c – Load weight binary + JSON manifest into PkModel.
 *
 *  Supports two on-disk formats, distinguished by the per-tensor "dtype"
 *  field in the JSON manifest:
 *    - "float32" : raw fp32 contiguous data
 *    - "int8"    : int8 data with a fp32 scalar "scale" (symmetric, zp=0)
 *
 *  The loader populates `Weight` fields (sgemm weight matrices) with either
 *  an fp32 pointer + scale=0, or an int8 pointer + nonzero scale. Plain
 *  `const float *` fields (biases, LayerNorm weights, pos_bias_u/v,
 *  pre_encode convs, conv module weights, LSTM weights, embedding) must be
 *  stored fp32 in the manifest.
 */

#include "parakeet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ── Minimal JSON parser ─────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    long long   offset;
    int         numel;
    int         is_int8;       /* 1 if dtype=="int8"     */
    int         is_int4;       /* 1 if dtype=="int4"     */
    float       scale;         /* valid when is_int8==1  */
    long long   scales_offset; /* byte offset of per-channel scales (int4) */
    int         n_scales;      /* number of per-channel scales (int4, = K) */
} WEntry;

static int skip_ws(const char *s, int i)
{
    while (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t') i++;
    return i;
}

static int parse_string(const char *s, int i, const char **start, int *len)
{
    i++;
    *start = s + i;
    int j = i;
    while (s[j] != '"') j++;
    *len = j - i;
    return j + 1;
}

static int parse_ll(const char *s, int i, long long *val)
{
    *val = 0;
    int neg = 0;
    if (s[i] == '-') { neg = 1; i++; }
    while (s[i] >= '0' && s[i] <= '9') {
        *val = *val * 10 + (s[i] - '0');
        i++;
    }
    if (neg) *val = -*val;
    return i;
}

/* Parse a JSON number (float or int) into a double. Returns new index. */
static int parse_number(const char *s, int i, double *val)
{
    *val = 0.0;
    int neg = 0;
    if (s[i] == '-') { neg = 1; i++; }
    double x = 0.0;
    while (s[i] >= '0' && s[i] <= '9') {
        x = x * 10 + (s[i] - '0');
        i++;
    }
    if (s[i] == '.') {
        i++;
        double frac = 0.0, div = 1.0;
        while (s[i] >= '0' && s[i] <= '9') {
            frac = frac * 10 + (s[i] - '0');
            div *= 10.0;
            i++;
        }
        x += frac / div;
    }
    if (s[i] == 'e' || s[i] == 'E') {
        i++;
        int eneg = 0, eval = 0;
        if (s[i] == '-') { eneg = 1; i++; }
        else if (s[i] == '+') { i++; }
        while (s[i] >= '0' && s[i] <= '9') { eval = eval * 10 + (s[i] - '0'); i++; }
        double mul = 1.0;
        for (int k = 0; k < eval; k++) mul *= 10.0;
        x = eneg ? x / mul : x * mul;
    }
    *val = neg ? -x : x;
    return i;
}

static WEntry *parse_manifest(const char *json, int *count)
{
    int cap = 1024, n = 0;
    WEntry *entries = (WEntry *)malloc(cap * sizeof(WEntry));

    int i = 0;
    i = skip_ws(json, i);
    if (json[i] != '{') return NULL;
    i++;

    while (1) {
        i = skip_ws(json, i);
        if (json[i] == '}') break;
        if (json[i] == ',') i++;
        i = skip_ws(json, i);

        const char *name_start;
        int name_len;
        i = parse_string(json, i, &name_start, &name_len);

        char *name = (char *)malloc(name_len + 1);
        memcpy(name, name_start, name_len);
        name[name_len] = '\0';

        i = skip_ws(json, i);
        i++;            /* : */
        i = skip_ws(json, i);

        if (json[i] != '{') { free(name); continue; }
        i++;

        long long offset = 0, numel = 0;
        int is_int8 = 0, is_int4 = 0;
        float scale = 0.0f;
        long long scales_offset = 0;
        int n_scales = 0;

        while (json[i] != '}') {
            i = skip_ws(json, i);
            if (json[i] == '}') break;
            if (json[i] == ',') { i++; continue; }

            const char *key_start;
            int key_len;
            i = parse_string(json, i, &key_start, &key_len);
            i = skip_ws(json, i);
            i++;                  /* : */
            i = skip_ws(json, i);

            if (key_len == 6 && strncmp(key_start, "offset", 6) == 0) {
                i = parse_ll(json, i, &offset);
            } else if (key_len == 5 && strncmp(key_start, "numel", 5) == 0) {
                i = parse_ll(json, i, &numel);
            } else if (key_len == 5 && strncmp(key_start, "dtype", 5) == 0) {
                /* string: "int8", "int4", or "float32" */
                if (json[i] == '"') {
                    const char *vs; int vl;
                    i = parse_string(json, i, &vs, &vl);
                    if (vl == 4 && strncmp(vs, "int8", 4) == 0) is_int8 = 1;
                    else if (vl == 4 && strncmp(vs, "int4", 4) == 0) is_int4 = 1;
                } else {
                    while (json[i] != ',' && json[i] != '}') i++;
                }
            } else if (key_len == 13 && strncmp(key_start, "scales_offset", 13) == 0) {
                i = parse_ll(json, i, &scales_offset);
            } else if (key_len == 8 && strncmp(key_start, "n_scales", 8) == 0) {
                long long ns; i = parse_ll(json, i, &ns); n_scales = (int)ns;
            } else if (key_len == 5 && strncmp(key_start, "scale", 5) == 0) {
                double v; i = parse_number(json, i, &v); scale = (float)v;
            } else {
                /* Skip unknown value (string / array / number). */
                if (json[i] == '"') {
                    const char *d; int dl;
                    i = parse_string(json, i, &d, &dl);
                } else if (json[i] == '[') {
                    int depth = 1; i++;
                    while (depth > 0) {
                        if (json[i] == '[') depth++;
                        else if (json[i] == ']') depth--;
                        i++;
                    }
                } else {
                    while (json[i] != ',' && json[i] != '}') i++;
                }
            }
        }
        i++;

        if (n >= cap) {
            cap *= 2;
            entries = (WEntry *)realloc(entries, cap * sizeof(WEntry));
        }
        entries[n].name          = name;
        entries[n].offset        = offset;
        entries[n].numel         = (int)numel;
        entries[n].is_int8       = is_int8;
        entries[n].is_int4       = is_int4;
        entries[n].scale         = scale;
        entries[n].scales_offset = scales_offset;
        entries[n].n_scales      = n_scales;
        n++;
    }

    *count = n;
    return entries;
}

/* ── Weight loading ──────────────────────────────────────────────────────── */

static void  *g_mmap_ptr = NULL;
static size_t g_mmap_len = 0;

/* Find entry by name – returns NULL if not found. */
static const WEntry *find_entry(const WEntry *entries, int n, const char *name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(entries[i].name, name) == 0) return &entries[i];
    return NULL;
}

/* Lookup fp32 pointer. Warns if the entry is int8 (caller should be using
 * lookup_weight for those). */
static const float *lookup_fp32(const WEntry *entries, int n,
                                const void *base, const char *name)
{
    const WEntry *e = find_entry(entries, n, name);
    if (!e) {
        fprintf(stderr, "WARNING: fp32 weight '%s' not found\n", name);
        return NULL;
    }
    if (e->is_int8) {
        fprintf(stderr, "ERROR: '%s' is int8 but fp32 expected\n", name);
        return NULL;
    }
    return (const float *)((const char *)base + e->offset);
}

/* Populate a Weight struct – works for fp32, int8, and int4 entries. */
static void lookup_weight(const WEntry *entries, int n, const void *base,
                          const char *name, Weight *out)
{
    const WEntry *e = find_entry(entries, n, name);
    if (!e) {
        fprintf(stderr, "WARNING: weight '%s' not found\n", name);
        out->data   = NULL;
        out->scale  = 0.0f;
        out->scales = NULL;
        out->bits   = 32;
        return;
    }
    out->data = (const void *)((const char *)base + e->offset);
    if (e->is_int4) {
        out->scale  = 0.0f;
        out->scales = (const float *)((const char *)base + e->scales_offset);
        out->bits   = 4;
    } else if (e->is_int8) {
        out->scale  = e->scale;
        out->scales = NULL;
        out->bits   = 8;
    } else {
        out->scale  = 0.0f;
        out->scales = NULL;
        out->bits   = 32;
    }
}

int pk_load_weights(PkModel *m, const char *bin_path, const char *json_path)
{
    /* Read JSON manifest */
    FILE *jf = fopen(json_path, "r");
    if (!jf) { perror(json_path); return -1; }
    fseek(jf, 0, SEEK_END);
    long jlen = ftell(jf);
    fseek(jf, 0, SEEK_SET);
    char *json = (char *)malloc(jlen + 1);
    fread(json, 1, jlen, jf);
    json[jlen] = '\0';
    fclose(jf);

    int n_entries;
    WEntry *entries = parse_manifest(json, &n_entries);
    free(json);
    if (!entries) { fprintf(stderr, "failed to parse manifest\n"); return -1; }
    int n_int8 = 0, n_int4 = 0;
    for (int i = 0; i < n_entries; i++) {
        if (entries[i].is_int8) n_int8++;
        if (entries[i].is_int4) n_int4++;
    }
    int n_fp32 = n_entries - n_int8 - n_int4;
    fprintf(stderr, "manifest: %d tensors (%d int4, %d int8, %d fp32)\n",
            n_entries, n_int4, n_int8, n_fp32);

    /* mmap binary weights */
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) { perror(bin_path); return -1; }
    struct stat st;
    fstat(fd, &st);
    g_mmap_len = st.st_size;
    g_mmap_ptr = mmap(NULL, g_mmap_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (g_mmap_ptr == MAP_FAILED) { perror("mmap"); return -1; }

    const void *base = g_mmap_ptr;

#define F(field, name) m->field = lookup_fp32(entries, n_entries, base, name)
#define W(field, name) lookup_weight(entries, n_entries, base, name, &m->field)

    /* Pre-encode (conv weights stay fp32 even in int8 manifest). */
    F(enc.pre_encode.conv0_w, "pre_encode.conv.0.weight");
    F(enc.pre_encode.conv0_b, "pre_encode.conv.0.bias");
    F(enc.pre_encode.conv2_w, "pre_encode.conv.2.weight");
    F(enc.pre_encode.conv2_b, "pre_encode.conv.2.bias");
    F(enc.pre_encode.conv3_w, "pre_encode.conv.3.weight");
    F(enc.pre_encode.conv3_b, "pre_encode.conv.3.bias");
    F(enc.pre_encode.conv5_w, "pre_encode.conv.5.weight");
    F(enc.pre_encode.conv5_b, "pre_encode.conv.5.bias");
    F(enc.pre_encode.conv6_w, "pre_encode.conv.6.weight");
    F(enc.pre_encode.conv6_b, "pre_encode.conv.6.bias");
    W(enc.pre_encode.out_w,   "pre_encode.out.weight");
    F(enc.pre_encode.out_b,   "pre_encode.out.bias");

    F(enc.pe_table,           "pos_encoding.pe_table");

    /* Conformer blocks */
    for (int i = 0; i < PK_N_BLOCKS; i++) {
        ConformerBlock *b = &m->enc.blocks[i];
        char buf[256];

#define BF(field, fmt, ...) do { \
    snprintf(buf, sizeof(buf), fmt, __VA_ARGS__); \
    b->field = lookup_fp32(entries, n_entries, base, buf); \
} while(0)
#define BW(field, fmt, ...) do { \
    snprintf(buf, sizeof(buf), fmt, __VA_ARGS__); \
    lookup_weight(entries, n_entries, base, buf, &b->field); \
} while(0)

        BF(norm_ff1_w, "layers.%d.norm_feed_forward1.weight", i);
        BF(norm_ff1_b, "layers.%d.norm_feed_forward1.bias", i);
        BW(ff1.linear1_w, "layers.%d.feed_forward1.linear1.weight", i);
        BF(ff1.linear1_b, "layers.%d.feed_forward1.linear1.bias", i);
        BW(ff1.linear2_w, "layers.%d.feed_forward1.linear2.weight", i);
        BF(ff1.linear2_b, "layers.%d.feed_forward1.linear2.bias", i);

        BF(norm_sa_w, "layers.%d.norm_self_att.weight", i);
        BF(norm_sa_b, "layers.%d.norm_self_att.bias", i);
        BW(mha.linear_q_w, "layers.%d.self_attn.linear_q.weight", i);
        BF(mha.linear_q_b, "layers.%d.self_attn.linear_q.bias", i);
        BW(mha.linear_k_w, "layers.%d.self_attn.linear_k.weight", i);
        BF(mha.linear_k_b, "layers.%d.self_attn.linear_k.bias", i);
        BW(mha.linear_v_w, "layers.%d.self_attn.linear_v.weight", i);
        BF(mha.linear_v_b, "layers.%d.self_attn.linear_v.bias", i);
        BW(mha.linear_pos_w, "layers.%d.self_attn.linear_pos.weight", i);
        BW(mha.linear_out_w, "layers.%d.self_attn.linear_out.weight", i);
        BF(mha.linear_out_b, "layers.%d.self_attn.linear_out.bias", i);
        BF(mha.pos_bias_u, "layers.%d.self_attn.pos_bias_u", i);
        BF(mha.pos_bias_v, "layers.%d.self_attn.pos_bias_v", i);

        BF(norm_conv_w, "layers.%d.norm_conv.weight", i);
        BF(norm_conv_b, "layers.%d.norm_conv.bias", i);
        BF(conv.pw1_w, "layers.%d.conv.pointwise_conv1.weight", i);
        BF(conv.pw1_b, "layers.%d.conv.pointwise_conv1.bias", i);
        BF(conv.dw_w,  "layers.%d.conv.depthwise_conv.weight", i);
        BF(conv.dw_b,  "layers.%d.conv.depthwise_conv.bias", i);
        BF(conv.pw2_w, "layers.%d.conv.pointwise_conv2.weight", i);
        BF(conv.pw2_b, "layers.%d.conv.pointwise_conv2.bias", i);

        BF(norm_ff2_w, "layers.%d.norm_feed_forward2.weight", i);
        BF(norm_ff2_b, "layers.%d.norm_feed_forward2.bias", i);
        BW(ff2.linear1_w, "layers.%d.feed_forward2.linear1.weight", i);
        BF(ff2.linear1_b, "layers.%d.feed_forward2.linear1.bias", i);
        BW(ff2.linear2_w, "layers.%d.feed_forward2.linear2.weight", i);
        BF(ff2.linear2_b, "layers.%d.feed_forward2.linear2.bias", i);

        BF(norm_out_w, "layers.%d.norm_out.weight", i);
        BF(norm_out_b, "layers.%d.norm_out.bias", i);

#undef BF
#undef BW
    }

    /* Decoder (LSTM + embedding always fp32 in int8 manifest too). */
    F(dec.embed_w,       "decoder.prediction.embed.weight");
    F(dec.lstm0_Wi,      "decoder.lstm.0.W_i");
    F(dec.lstm0_Wr,      "decoder.lstm.0.W_r");
    F(dec.lstm0_B,       "decoder.lstm.0.B");
    F(dec.lstm1_Wi,      "decoder.lstm.1.W_i");
    F(dec.lstm1_Wr,      "decoder.lstm.1.W_r");
    F(dec.lstm1_B,       "decoder.lstm.1.B");
    F(dec.joint_enc_w,   "joint.enc.weight");
    F(dec.joint_enc_b,   "joint.enc.bias");
    F(dec.joint_pred_w,  "joint.pred.weight");
    F(dec.joint_pred_b,  "joint.pred.bias");
    F(dec.joint_out_w,   "joint.out.weight");
    F(dec.joint_out_b,   "joint.joint_net.2.bias");

#undef F
#undef W

    for (int i = 0; i < n_entries; i++) free((void *)entries[i].name);
    free(entries);

    /* ── Post-load: build fused QKV weights for each block ── */
    {
        int D = PK_D_MODEL;
        for (int i = 0; i < PK_N_BLOCKS; i++) {
            MHAWeights *mha = &m->enc.blocks[i].mha;

            /* Concatenate biases (always fp32): [Bq|Bk|Bv] → [3D] */
            float *fb = (float *)malloc(3 * D * sizeof(float));
            memcpy(fb + 0 * D, mha->linear_q_b, D * sizeof(float));
            memcpy(fb + 1 * D, mha->linear_k_b, D * sizeof(float));
            memcpy(fb + 2 * D, mha->linear_v_b, D * sizeof(float));
            mha->linear_qkv_b = fb;

            float sq = mha->linear_q_w.scale;
            float sk = mha->linear_k_w.scale;
            float sv = mha->linear_v_w.scale;

            if (sq == 0.0f && sk == 0.0f && sv == 0.0f) {
                /* fp32 path: interleave columns of Wq, Wk, Wv into [D, 3D]. */
                float *fw = (float *)malloc((size_t)D * 3 * D * sizeof(float));
                const float *wq = (const float *)mha->linear_q_w.data;
                const float *wk = (const float *)mha->linear_k_w.data;
                const float *wv = (const float *)mha->linear_v_w.data;
                for (int k = 0; k < D; k++) {
                    memcpy(fw + k * 3 * D + 0 * D, wq + k * D, D * sizeof(float));
                    memcpy(fw + k * 3 * D + 1 * D, wk + k * D, D * sizeof(float));
                    memcpy(fw + k * 3 * D + 2 * D, wv + k * D, D * sizeof(float));
                }
                mha->linear_qkv_w.data   = fw;
                mha->linear_qkv_w.scale  = 0.0f;
                mha->linear_qkv_w.scales = NULL;
                mha->linear_qkv_w.bits   = 32;
                mha->qkv_w_buf = fw;
            } else if (sq != 0.0f && sk != 0.0f && sv != 0.0f) {
                /* int8 path: requantize to common scale, then interleave. */
                float ms = sq;
                if (sk > ms) ms = sk;
                if (sv > ms) ms = sv;

                signed char *fw = (signed char *)malloc((size_t)D * 3 * D);
                const signed char *iq = (const signed char *)mha->linear_q_w.data;
                const signed char *ik = (const signed char *)mha->linear_k_w.data;
                const signed char *iv = (const signed char *)mha->linear_v_w.data;

                for (int k = 0; k < D; k++) {
                    signed char *dst = fw + k * 3 * D;
                    /* Q slice */
                    if (sq == ms) {
                        memcpy(dst, iq + k * D, D);
                    } else {
                        float r = sq / ms;
                        for (int n = 0; n < D; n++) {
                            int v = (int)roundf(iq[k * D + n] * r);
                            dst[n] = (signed char)(v < -127 ? -127 : v > 127 ? 127 : v);
                        }
                    }
                    /* K slice */
                    dst += D;
                    if (sk == ms) {
                        memcpy(dst, ik + k * D, D);
                    } else {
                        float r = sk / ms;
                        for (int n = 0; n < D; n++) {
                            int v = (int)roundf(ik[k * D + n] * r);
                            dst[n] = (signed char)(v < -127 ? -127 : v > 127 ? 127 : v);
                        }
                    }
                    /* V slice */
                    dst += D;
                    if (sv == ms) {
                        memcpy(dst, iv + k * D, D);
                    } else {
                        float r = sv / ms;
                        for (int n = 0; n < D; n++) {
                            int v = (int)roundf(iv[k * D + n] * r);
                            dst[n] = (signed char)(v < -127 ? -127 : v > 127 ? 127 : v);
                        }
                    }
                }
                mha->linear_qkv_w.data   = fw;
                mha->linear_qkv_w.scale  = ms;
                mha->linear_qkv_w.scales = NULL;
                mha->linear_qkv_w.bits   = 8;
                mha->qkv_w_buf = fw;
            } else if (mha->linear_q_w.bits == 4) {
                /* int4 path: dequantize all three to fp32, then interleave.
                 * Per-channel scales differ across Q/K/V so we fuse as fp32. */
                float *fw = (float *)calloc((size_t)D * 3 * D, sizeof(float));
                const unsigned char *dq = (const unsigned char *)mha->linear_q_w.data;
                const unsigned char *dk = (const unsigned char *)mha->linear_k_w.data;
                const unsigned char *dv = (const unsigned char *)mha->linear_v_w.data;
                const float *sq = mha->linear_q_w.scales;
                const float *sk = mha->linear_k_w.scales;
                const float *sv = mha->linear_v_w.scales;
                /* Each row k of W [K, N] is packed as N/2 bytes.
                 * Byte j holds values for columns 2j (low nibble) and 2j+1 (high nibble).
                 * Per-group scales: n_groups scales per row, group size = PK_INT4_GROUP. */
                int half_N = D / 2;
                int G = PK_INT4_GROUP;
                int n_groups = D / G;
                for (int k = 0; k < D; k++) {
                    float *dst = fw + k * 3 * D;
                    /* Q */
                    const unsigned char *src = dq + k * half_N;
                    for (int j = 0; j < half_N; j++) {
                        int col = 2 * j;
                        float s = sq[k * n_groups + col / G];
                        int lo = (int)(signed char)((src[j] & 0x0F) << 4) >> 4;
                        int hi = (int)(signed char)(src[j] & 0xF0) >> 4;
                        dst[col]     = (float)lo * s;
                        dst[col + 1] = (float)hi * s;
                    }
                    /* K */
                    dst += D;
                    src = dk + k * half_N;
                    for (int j = 0; j < half_N; j++) {
                        int col = 2 * j;
                        float s = sk[k * n_groups + col / G];
                        int lo = (int)(signed char)((src[j] & 0x0F) << 4) >> 4;
                        int hi = (int)(signed char)(src[j] & 0xF0) >> 4;
                        dst[col]     = (float)lo * s;
                        dst[col + 1] = (float)hi * s;
                    }
                    /* V */
                    dst += D;
                    src = dv + k * half_N;
                    for (int j = 0; j < half_N; j++) {
                        int col = 2 * j;
                        float s = sv[k * n_groups + col / G];
                        int lo = (int)(signed char)((src[j] & 0x0F) << 4) >> 4;
                        int hi = (int)(signed char)(src[j] & 0xF0) >> 4;
                        dst[col]     = (float)lo * s;
                        dst[col + 1] = (float)hi * s;
                    }
                }
                mha->linear_qkv_w.data   = fw;
                mha->linear_qkv_w.scale  = 0.0f;
                mha->linear_qkv_w.scales = NULL;
                mha->linear_qkv_w.bits   = 32;
                mha->qkv_w_buf = fw;
            }
        }
    }

    return 0;
}

void pk_free_weights(PkModel *m)
{
    /* Free fused QKV allocations. */
    for (int i = 0; i < PK_N_BLOCKS; i++) {
        MHAWeights *mha = &m->enc.blocks[i].mha;
        free(mha->qkv_w_buf);
        free(mha->linear_qkv_b);
    }

    if (g_mmap_ptr && g_mmap_ptr != MAP_FAILED) {
        munmap(g_mmap_ptr, g_mmap_len);
        g_mmap_ptr = NULL;
    }
}
