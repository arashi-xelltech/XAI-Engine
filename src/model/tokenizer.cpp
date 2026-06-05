#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace xai {

static uint32_t tok_hash(const char *s, int len) {
    uint32_t h = 5381;
    for (int i = 0; i < len; i++) h = ((h << 5) + h) ^ (uint8_t)s[i];
    return h;
}

static void tok_build_hash(Tokenizer *t) {
    int cap = t->vocab_size * 4;
    t->hash_cap = cap;
    t->hash_keys.assign(cap, -1);
    t->hash_ids.assign(cap, -1);
    for (int i = 0; i < t->vocab_size; i++) {
        int len = (int)t->pieces[i].size();
        uint32_t h = tok_hash(t->pieces[i].c_str(), len) % cap;
        while (t->hash_keys[h] != -1) h = (h + 1) % cap;
        t->hash_keys[h] = len;
        t->hash_ids[h]  = i;
    }
}

static int tok_lookup(const Tokenizer *t, const char *s, int len) {
    uint32_t h = tok_hash(s, len) % t->hash_cap;
    while (t->hash_keys[h] != -1) {
        if (t->hash_keys[h] == len &&
            !memcmp(t->pieces[t->hash_ids[h]].c_str(), s, len))
            return t->hash_ids[h];
        h = (h + 1) % t->hash_cap;
    }
    return -1;
}

void tok_deserialize(const char *data, size_t max_size, Tokenizer *t) {
    const char *p    = data;
    const char *end  = data + max_size;

    if (p + 20 > end) { fprintf(stderr,"tok blob too small\n"); exit(1); }

    uint32_t vs, bos, eos, pad, unk;
    memcpy(&vs,  p, 4); p += 4;
    memcpy(&bos, p, 4); p += 4;
    memcpy(&eos, p, 4); p += 4;
    memcpy(&pad, p, 4); p += 4;
    memcpy(&unk, p, 4); p += 4;

    if (vs == 0 || vs > 500000) {
        fprintf(stderr, "tok: bad vocab_size=%u\n", vs); exit(1);
    }

    t->vocab_size = (int)vs;
    t->bos_id = (int)bos;
    t->eos_id = (int)eos;
    t->pad_id = (int)pad;
    t->unk_id = (int)unk;

    t->pieces.resize(vs);
    t->scores.resize(vs);
    t->types.resize(vs);

    for (uint32_t i = 0; i < vs; i++) {
        if (p + 4 > end) {
            fprintf(stderr, "tok: EOF at piece %u blen read\n", i);
            exit(1);
        }
        uint32_t blen;
        memcpy(&blen, p, 4);

        if (blen > 512) {
            bool recovered = false;
            for (int back = 1; back <= 4; back++) {
                if (p - back < data + 20) continue;
                uint32_t blen2;
                memcpy(&blen2, p - back, 4);
                if (blen2 <= 512 && p - back + 4 + blen2 <= end) {
                    fprintf(stderr,
                        "tok: piece[%u] bad blen=%u at offset %lld, "
                        "recovered with back=%d blen=%u\n",
                        i, blen, (long long)(p - data), back, blen2);
                    p -= back;
                    blen = blen2;
                    recovered = true;
                    break;
                }
            }
            if (!recovered) {
                fprintf(stderr,
                    "tok: piece[%u] unrecoverable blen=%u at offset %lld\n",
                    i, blen, (long long)(p - data));
                for (uint32_t j = i; j < vs; j++) t->pieces[j] = "";
                goto done;
            }
        }

        p += 4;
        if (p + blen > end) {
            fprintf(stderr, "tok: EOF reading piece %u data (blen=%u)\n",i,blen);
            t->pieces[i] = "";
            for (uint32_t j = i+1; j < vs; j++) t->pieces[j] = "";
            goto done;
        }
        t->pieces[i].assign(p, blen);
        p += blen;
    }

done:
    /* scores и types — если не влезают, заполним нулями */
    if (p + (int64_t)vs * 8 <= end) {
        memcpy(t->scores.data(), p, vs * 4); p += vs * 4;
        memcpy(t->types.data(),  p, vs * 4);
    } else {
        fprintf(stderr, "tok: scores/types truncated, using zeros\n");
        memset(t->scores.data(), 0, vs * sizeof(float));
        memset(t->types.data(),  0, vs * sizeof(int));
    }

    tok_build_hash(t);
}

static const char SPIECE[] = "\xe2\x96\x81";

int tok_encode(const Tokenizer *t, const char *text,
               int *ids, int max_ids, int add_bos) {
    int n = 0;
    if (add_bos && n < max_ids) ids[n++] = t->bos_id;
    int tlen = (int)strlen(text);
    int cap  = tlen * 3 + 4;
    std::vector<char> buf(cap);
    int bp = 0;
    buf[bp++] = SPIECE[0]; buf[bp++] = SPIECE[1]; buf[bp++] = SPIECE[2];
    for (int i = 0; i < tlen; i++) {
        if (text[i] == ' ') {
            buf[bp++] = SPIECE[0]; buf[bp++] = SPIECE[1]; buf[bp++] = SPIECE[2];
        } else {
            buf[bp++] = text[i];
        }
    }
    buf[bp] = 0;

    int i = 0;
    while (i < bp && n < max_ids) {
        int best_len = 0, best_id = t->unk_id;
        int max_try  = std::min(bp - i, 64);
        for (int end = max_try; end > 0; end--) {
            int id = tok_lookup(t, buf.data() + i, end);
            if (id >= 0) { best_len = end; best_id = id; break; }
        }
        if (best_len == 0) {
            unsigned char c = (unsigned char)buf[i];
            if      (c < 0x80) best_len = 1;
            else if (c < 0xE0) best_len = 2;
            else if (c < 0xF0) best_len = 3;
            else               best_len = 4;
            if (i + best_len > bp) best_len = bp - i;
        }
        ids[n++] = best_id;
        i += best_len;
    }
    return n;
}

/* ================================================================
 * BYTE TOKEN DECODING
 * ================================================================ */

// Проверяет, является ли строка байтовым токеном вида <0xNN>
static bool is_byte_token(const char *piece, int len) {
    // Паттерн: <0xHH> где HH — две hex цифры, всего 6 символов
    if (len != 6) return false;
    if (piece[0] != '<') return false;
    if (piece[1] != '0') return false;
    if (piece[2] != 'x') return false;
    if (!isxdigit((unsigned char)piece[3])) return false;
    if (!isxdigit((unsigned char)piece[4])) return false;
    if (piece[5] != '>') return false;
    return true;
}

// Декодирует <0xHH> в байт
static unsigned char decode_byte_token(const char *piece) {
    // piece гарантированно "<0xHH>"
    char hex[3] = {piece[3], piece[4], 0};
    return (unsigned char)strtol(hex, nullptr, 16);
}

/* ================================================================
 * УЛУЧШЕННОЕ ДЕКОДИРОВАНИЕ ТОКЕНОВ С ПОДДЕРЖКОЙ BYTE TOKENS
 * ================================================================ */

void tok_decode_one(const Tokenizer *t, int id, char *buf, int buf_sz) {
    if (id < 0 || id >= t->vocab_size ||
        id == t->bos_id || id == t->eos_id || id == t->pad_id) {
        buf[0] = 0; return;
    }

    const char *pc = t->pieces[id].c_str();
    int pc_len = (int)t->pieces[id].size();
    
    // Собираем "сырые" байты которые встретились в виде <0xNN>
    // и обычные символы (включая ▁ который заменяем на пробел)
    int j = 0;
    for (int i = 0; i < pc_len && j < buf_sz - 1; ) {
        // Проверяем: это байтовый токен?
        if (i + 6 <= pc_len && is_byte_token(pc + i, 6)) {
            // Декодируем <0xHH> в реальный байт
            unsigned char byte_val = decode_byte_token(pc + i);
            // Выводим как читаемый символ, если printable
            // или как escape-последовательность
            if (byte_val >= 0x20 && byte_val < 0x7F) {
                // Обычный ASCII символ
                buf[j++] = (char)byte_val;
            } else if (byte_val == '\n') {
                buf[j++] = '\n';
            } else if (byte_val == '\r') {
                buf[j++] = '\r';
            } else if (byte_val == '\t') {
                buf[j++] = '\t';
            } else if (byte_val >= 0xC0) {
                // Вероятно часть UTF-8 последовательности
                buf[j++] = (char)byte_val;
            } else if (byte_val >= 0x80) {
                // Тоже часть UTF-8
                buf[j++] = (char)byte_val;
            } else {
                // Непечатный управляющий символ — можно либо пропустить
                // либо вывести как есть (второй вариант лучше для бинарных данных)
                // Пропускаем управляющие кроме \n \r \t
                // buf[j++] = '?';  // или просто пропустить
            }
            i += 6;
        }
        // Проверяем: это маркер пробела?
        else if ((unsigned char)pc[i]   == 0xE2 &&
                 i + 2 < pc_len &&
                 (unsigned char)pc[i+1] == 0x96 &&
                 (unsigned char)pc[i+2] == 0x81) {
            buf[j++] = ' ';
            i += 3;
        }
        // Обычный символ
        else {
            buf[j++] = pc[i++];
        }
    }
    buf[j] = 0;
}

} // namespace xai