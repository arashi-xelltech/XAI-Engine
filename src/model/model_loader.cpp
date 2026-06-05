#include "model/model_loader.h"
#include "model/tensor_ops.h"
#include "model/rope.h"
#include "model/tokenizer.h"
#include "core/aligned_alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace xai {

/* ================================================================
 * MINIMAL JSON PARSER (локальный, только для загрузки)
 * ================================================================ */

namespace json {

struct JP { const char *d; int p, len; };

static void ws(JP *j) {
    while (j->p < j->len &&
           (j->d[j->p]==' '||j->d[j->p]=='\t'||
            j->d[j->p]=='\n'||j->d[j->p]=='\r')) j->p++;
}
static char peek(JP *j) { ws(j); return j->p < j->len ? j->d[j->p] : 0; }

static char *str(JP *j) {
    ws(j);
    if (j->d[j->p] != '"') return nullptr;
    j->p++;
    int start = j->p;
    while (j->p < j->len && j->d[j->p] != '"') {
        if (j->d[j->p] == '\\') j->p++;
        j->p++;
    }
    int slen = j->p - start;
    char *s  = (char*)malloc(slen + 1);
    memcpy(s, j->d + start, slen);
    s[slen] = 0;
    j->p++;
    return s;
}

static double num(JP *j) {
    ws(j);
    char buf[64]; int i = 0;
    while (j->p < j->len && i < 63) {
        char c = j->d[j->p];
        if ((c>='0'&&c<='9')||c=='-'||c=='+'||c=='.'||c=='e'||c=='E')
            buf[i++] = c, j->p++;
        else break;
    }
    buf[i] = 0;
    return atof(buf);
}

static void skip(JP *j);

static int int_arr(JP *j, int *arr, int max_n) {
    ws(j);
    if (j->d[j->p] != '[') return 0;
    j->p++;
    int n = 0;
    while (peek(j) != ']' && n < max_n) {
        arr[n++] = (int)num(j);
        ws(j);
        if (j->d[j->p] == ',') j->p++;
    }
    ws(j); j->p++;
    return n;
}

static void skip(JP *j) {
    ws(j);
    char c = j->d[j->p];
    if (c == '"') { free(str(j)); }
    else if (c == '{') {
        j->p++;
        if (peek(j) != '}') {
            while (1) {
                free(str(j)); ws(j); j->p++;
                skip(j); ws(j);
                if (j->d[j->p] == ',') j->p++; else break;
            }
        }
        ws(j); j->p++;
    }
    else if (c == '[') {
        j->p++;
        if (peek(j) != ']') {
            while (1) {
                skip(j); ws(j);
                if (j->d[j->p] == ',') j->p++; else break;
            }
        }
        ws(j); j->p++;
    }
    else num(j);
}

} // namespace json

using namespace json;

/* ================================================================
 * HEADER PARSING
 * ================================================================ */

static void parse_model_cfg(JP *j, ModelConfig *cfg) {
    ws(j); j->p++;
    while (peek(j) != '}') {
        char *k = str(j); ws(j); j->p++;
        if      (!strcmp(k,"vocab_size"))        cfg->vocab_size        = (int)num(j);
        else if (!strcmp(k,"hidden_size"))        cfg->hidden_size       = (int)num(j);
        else if (!strcmp(k,"intermediate_size"))  cfg->intermediate_size = (int)num(j);
        else if (!strcmp(k,"num_layers"))         cfg->num_layers        = (int)num(j);
        else if (!strcmp(k,"num_heads"))          cfg->num_heads         = (int)num(j);
        else if (!strcmp(k,"num_kv_heads"))       cfg->num_kv_heads      = (int)num(j);
        else if (!strcmp(k,"max_seq_len"))        cfg->max_seq_len       = (int)num(j);
        else if (!strcmp(k,"rms_norm_eps"))       cfg->rms_norm_eps      = (float)num(j);
        else if (!strcmp(k,"rope_theta"))         cfg->rope_theta        = (float)num(j);
        else skip(j);
        free(k); ws(j);
        if (j->d[j->p] == ',') j->p++;
    }
    j->p++;
    cfg->head_dim   = cfg->hidden_size / cfg->num_heads;
    cfg->kv_dim     = cfg->num_kv_heads * cfg->head_dim;
    cfg->gqa_groups = cfg->num_heads / cfg->num_kv_heads;
}

static void parse_tensor(JP *j, TensorMeta *t) {
    memset(t, 0, sizeof(*t));
    t->scale = 1.0f;
    ws(j); j->p++;
    while (peek(j) != '}') {
        char *k = str(j); ws(j); j->p++;
        if (!strcmp(k,"name")) {
            char *s = str(j); strncpy(t->name, s, MAX_NAME_LEN-1); free(s);
        }
        else if (!strcmp(k,"dtype")) {
            char *s = str(j);
            if      (!strcmp(s,"f32"))       t->dtype = DTYPE_F32;
            else if (!strcmp(s,"f16"))       t->dtype = DTYPE_F16;
            else if (!strcmp(s,"q8"))        t->dtype = DTYPE_Q8;
            else if (!strncmp(s,"tied:",5)) t->dtype = DTYPE_TIED;
            free(s);
        }
        else if (!strcmp(k,"shape"))  t->ndim  = int_arr(j, t->shape, 4);
        else if (!strcmp(k,"offset")) t->offset = (int64_t)num(j);
        else if (!strcmp(k,"size"))   t->size   = (int64_t)num(j);
        else if (!strcmp(k,"scale"))  t->scale  = (float)num(j);
        else skip(j);
        free(k); ws(j);
        if (j->d[j->p] == ',') j->p++;
    }
    j->p++;
}

static void parse_tensors(JP *j, Model *m) {
    ws(j); j->p++;
    m->num_tensors = 0;
    while (peek(j) != ']' && m->num_tensors < MAX_TENSORS) {
        parse_tensor(j, &m->tensors[m->num_tensors++]);
        ws(j);
        if (j->d[j->p] == ',') j->p++;
    }
    ws(j); j->p++;
}

static void parse_header(const char *json, int len, Model *m) {
    JP j = {json, 0, len};
    ws(&j); j.p++;
    while (peek(&j) != '}' && j.p < j.len) {
        char *k = str(&j); ws(&j); j.p++;
        if (!strcmp(k,"model_config"))       parse_model_cfg(&j, &m->cfg);
        else if (!strcmp(k,"tensors"))       parse_tensors(&j, m);
        else if (!strcmp(k,"tokenizer_blob_size"))
            m->tokenizer_blob_size = (int64_t)num(&j);
        else if (!strcmp(k,"tokenizer")) {
            ws(&j); j.p++;
            while (peek(&j) != '}') {
                char *tk = str(&j); ws(&j); j.p++;
                if      (!strcmp(tk,"vocab_size")) m->tok.vocab_size = (int)num(&j);
                else if (!strcmp(tk,"bos_id"))     m->tok.bos_id     = (int)num(&j);
                else if (!strcmp(tk,"eos_id"))     m->tok.eos_id     = (int)num(&j);
                else if (!strcmp(tk,"pad_id"))     m->tok.pad_id     = (int)num(&j);
                else if (!strcmp(tk,"unk_id"))     m->tok.unk_id     = (int)num(&j);
                else skip(&j);
                free(tk); ws(&j);
                if (j.d[j.p] == ',') j.p++;
            }
            j.p++;
        }
        else skip(&j);
        free(k); ws(&j);
        if (j.d[j.p] == ',') j.p++;
    }
}

/* ================================================================
 * MODEL LOADING
 * ================================================================ */

int load_model(const char *path, Model *m, bool quiet) {
    memset(m, 0, sizeof(Model));
    new (&m->cfg)     ModelConfig();
    new (&m->tok)     Tokenizer();
    new (&m->lm_head) WeightTensor();

    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    struct stat st; fstat(m->fd, &st);
    m->file_size = (size_t)st.st_size;

    m->file_data = (char*)mmap(nullptr, m->file_size, PROT_READ,
                               MAP_SHARED, m->fd, 0);
    if (m->file_data == MAP_FAILED) { close(m->fd); return -1; }

    madvise(m->file_data, m->file_size, MADV_SEQUENTIAL);

    char *p = m->file_data;
    if (memcmp(p, "XLLM", 4)) { fprintf(stderr, "Bad magic\n"); return -1; }
    p += 4;
    uint32_t ver; memcpy(&ver, p, 4); p += 4;
    if (ver != 1) { fprintf(stderr, "Bad version\n"); return -1; }
    uint64_t hdr_sz; memcpy(&hdr_sz, p, 8); p += 8;

    parse_header(p, (int)hdr_sz, m);
    p += hdr_sz;

    tok_deserialize(p, (size_t)m->tokenizer_blob_size, &m->tok);
    p += m->tokenizer_blob_size;

    int64_t rel = p - m->file_data;
    int pad = (int)(ALIGN_SIZE - (rel % ALIGN_SIZE));
    if (pad < ALIGN_SIZE) p += pad;
    m->tensor_data_start = p;

    if (!quiet) printf("Loading weights (zero-copy native format)...\n");
    ModelConfig *c = &m->cfg;

    madvise(m->tensor_data_start,
            m->file_size - (size_t)(m->tensor_data_start - m->file_data),
            MADV_WILLNEED);

    m->embed  = load_embedding(m, "embed.weight");
    m->layers = new LayerWeights[c->num_layers];

    for (int l = 0; l < c->num_layers; l++) {
        char nm[256];
#define LF1(field, fmt) snprintf(nm,256,fmt,l); m->layers[l].field = load_1d_f32(m,nm)
#define LWN(field, fmt) snprintf(nm,256,fmt,l); m->layers[l].field = load_weight_native(m,nm)
        LF1(attn_norm, "layers.%d.attn_norm.weight");
        LF1(ffn_norm,  "layers.%d.ffn_norm.weight");
        LWN(q_proj,    "layers.%d.attn.q_proj.weight");
        LWN(k_proj,    "layers.%d.attn.k_proj.weight");
        LWN(v_proj,    "layers.%d.attn.v_proj.weight");
        LWN(o_proj,    "layers.%d.attn.o_proj.weight");
        LWN(gate_proj, "layers.%d.ffn.gate_proj.weight");
        LWN(up_proj,   "layers.%d.ffn.up_proj.weight");
        LWN(down_proj, "layers.%d.ffn.down_proj.weight");
#undef LF1
#undef LWN
        if (!quiet) { printf("  Layer %d/%d\r", l+1, c->num_layers); fflush(stdout); }
    }
    if (!quiet) printf("  Layer %d/%d\n", c->num_layers, c->num_layers);

    m->final_norm = load_1d_f32(m, "norm.weight");

    TensorMeta *lm = find_tensor(m, "lm_head.weight");
    if (lm && lm->dtype == DTYPE_TIED) {
        m->lm_head.data = m->embed; m->lm_head.fmt = WF_F32;
        m->lm_head.rows = c->vocab_size; m->lm_head.cols = c->hidden_size;
        m->lm_head_tied = true;
    } else if (lm) {
        m->lm_head = load_weight_native(m, "lm_head.weight");
        m->lm_head_tied = false;
    } else {
        m->lm_head.data = m->embed; m->lm_head.fmt = WF_F32;
        m->lm_head.rows = c->vocab_size; m->lm_head.cols = c->hidden_size;
        m->lm_head_tied = true;
    }

    madvise(m->file_data, m->file_size, MADV_RANDOM);
    build_rope(m);

    if (!quiet) {
        printf("Model: %d layers, %d hidden, %d heads (%d kv), %d vocab\n",
               c->num_layers, c->hidden_size, c->num_heads,
               c->num_kv_heads, c->vocab_size);
        const char *wfmt =
            m->layers[0].q_proj.fmt == WF_F16 ? "F16 (zero-copy)" :
            m->layers[0].q_proj.fmt == WF_Q8  ? "Q8 (zero-copy)"  : "F32";
        printf("Weights: %s\n", wfmt);
#ifdef USE_AVX
        printf("SIMD: AVX");
#elif defined(USE_SSE)
        printf("SIMD: SSE4.2");
#else
        printf("SIMD: none");
#endif
#ifdef USE_F16C
        printf(" + F16C");
#endif
        printf("\nKV cache: keys=int8, values=float32\nThreads: %d\n",
               g_num_threads);
        if (g_sliding_window > 0) printf("Sliding window: %d\n", g_sliding_window);
    }
    return 0;
}

void free_model(Model *m) {
    free(m->embed);
    free(m->final_norm);
    free(m->rope_cos);
    free(m->rope_sin);
    if (m->layers) {
        for (int l = 0; l < m->cfg.num_layers; l++) {
            free(m->layers[l].attn_norm);
            free(m->layers[l].ffn_norm);
        }
        delete[] m->layers;
    }
    munmap(m->file_data, m->file_size);
    close(m->fd);
}

} // namespace xai