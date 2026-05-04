/*  sentencepiece.c – Minimal SentencePiece model loader and Unigram decoder.
 *
 *  Only does *detokenization*: given a list of token IDs, produce the string.
 *  Parses the protobuf-encoded .model file directly (no libprotobuf dep)
 *  to extract the vocabulary table (piece strings).
 *
 *  SentencePiece uses U+2581 '▁' (UTF-8: E2 96 81) as the word-boundary
 *  marker.  Decoding: concatenate pieces → replace ▁ with ' ' → strip
 *  leading space.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Tiny protobuf varint / field reader ─────────────────────────────────── */

/* Read a varint (LEB128) from buf at *pos, advance *pos. */
static unsigned long long read_varint(const unsigned char *buf, int len, int *pos)
{
    unsigned long long val = 0;
    int shift = 0;
    while (*pos < len) {
        unsigned char b = buf[(*pos)++];
        val |= (unsigned long long)(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return val;
}

/* Skip a protobuf field value based on its wire type. */
static void skip_field(const unsigned char *buf, int len, int *pos, int wire)
{
    if (wire == 0) {              /* varint */
        read_varint(buf, len, pos);
    } else if (wire == 1) {       /* 64-bit */
        *pos += 8;
    } else if (wire == 2) {       /* length-delimited */
        int flen = (int)read_varint(buf, len, pos);
        *pos += flen;
    } else if (wire == 5) {       /* 32-bit */
        *pos += 4;
    }
}

/* ── Vocab loader ────────────────────────────────────────────────────────── */

typedef struct {
    char  **pieces;   /* array of piece strings (heap-allocated) */
    int     n;        /* number of pieces */
} SPVocab;

/* Parse the inner SentencePiece message to extract the piece string.
 * Returns a malloc'd copy of the piece, or NULL. */
static char *parse_piece_msg(const unsigned char *buf, int len, int pos, int end)
{
    while (pos < end) {
        unsigned long long tag = read_varint(buf, len, &pos);
        int field = (int)(tag >> 3);
        int wire  = (int)(tag & 7);

        if (field == 1 && wire == 2) {
            /* piece string */
            int slen = (int)read_varint(buf, len, &pos);
            char *s = (char *)malloc(slen + 1);
            memcpy(s, buf + pos, slen);
            s[slen] = '\0';
            pos += slen;
            return s;
        }
        skip_field(buf, len, &pos, wire);
    }
    return NULL;
}

/* Load a SentencePiece .model file and extract the vocab table.
 * Returns 0 on success. Caller must call sp_free(). */
static int sp_load(SPVocab *v, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    int len = (int)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(len);
    fread(buf, 1, len, f);
    fclose(f);

    /* First pass: count pieces to pre-allocate. */
    int count = 0;
    {
        int pos = 0;
        while (pos < len) {
            unsigned long long tag = read_varint(buf, len, &pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);
            if (field == 1 && wire == 2) {
                int mlen = (int)read_varint(buf, len, &pos);
                pos += mlen;
                count++;
            } else {
                skip_field(buf, len, &pos, wire);
            }
        }
    }

    v->n = count;
    v->pieces = (char **)calloc(count, sizeof(char *));

    /* Second pass: extract piece strings. */
    int idx = 0, pos = 0;
    while (pos < len && idx < count) {
        unsigned long long tag = read_varint(buf, len, &pos);
        int field = (int)(tag >> 3);
        int wire  = (int)(tag & 7);
        if (field == 1 && wire == 2) {
            int mlen = (int)read_varint(buf, len, &pos);
            int end  = pos + mlen;
            v->pieces[idx] = parse_piece_msg(buf, len, pos, end);
            pos = end;
            idx++;
        } else {
            skip_field(buf, len, &pos, wire);
        }
    }

    free(buf);
    return 0;
}

static void sp_free(SPVocab *v)
{
    if (v->pieces) {
        for (int i = 0; i < v->n; i++) free(v->pieces[i]);
        free(v->pieces);
        v->pieces = NULL;
    }
}

/* ── Decode: token IDs → string ──────────────────────────────────────────── */

/* Decode a list of token IDs into a malloc'd UTF-8 string.
 * Replaces ▁ (U+2581, UTF-8 E2 96 81) with ASCII space and strips
 * the leading space if present. */
char *sp_decode(const SPVocab *v, const int *ids, int n)
{
    /* Compute total length. */
    int total = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id >= 0 && id < v->n && v->pieces[id])
            total += (int)strlen(v->pieces[id]);
    }

    char *raw = (char *)malloc(total + 1);
    raw[0] = '\0';
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id >= 0 && id < v->n && v->pieces[id]) {
            int plen = (int)strlen(v->pieces[id]);
            memcpy(raw + pos, v->pieces[id], plen);
            pos += plen;
        }
    }
    raw[pos] = '\0';

    /* Replace ▁ (E2 96 81) with ' ' in-place. Since ▁ is 3 bytes and ' '
     * is 1 byte, do a compacting pass. */
    char *out = (char *)malloc(pos + 1);
    int wp = 0;
    for (int rp = 0; rp < pos; ) {
        if (rp + 2 < pos &&
            (unsigned char)raw[rp]   == 0xE2 &&
            (unsigned char)raw[rp+1] == 0x96 &&
            (unsigned char)raw[rp+2] == 0x81) {
            out[wp++] = ' ';
            rp += 3;
        } else {
            out[wp++] = raw[rp++];
        }
    }
    out[wp] = '\0';
    free(raw);

    /* Strip leading space. */
    char *result;
    if (wp > 0 && out[0] == ' ') {
        result = (char *)malloc(wp);
        memcpy(result, out + 1, wp);  /* includes '\0' */
        free(out);
    } else {
        result = out;
    }
    return result;
}

/* ── Public API used by main.c ───────────────────────────────────────────── */

/* Opaque handle. */
typedef struct { SPVocab v; } SPModel;

SPModel *sp_model_load(const char *path)
{
    SPModel *m = (SPModel *)calloc(1, sizeof(*m));
    if (sp_load(&m->v, path) != 0) { free(m); return NULL; }
    return m;
}

void sp_model_free(SPModel *m)
{
    if (m) { sp_free(&m->v); free(m); }
}

char *sp_model_decode(const SPModel *m, const int *ids, int n)
{
    return sp_decode(&m->v, ids, n);
}
