#include "model/tensor_ops.h"
#include "model/model_loader.h"   // <-- ДОБАВЛЕНО для полного определения Model
#include "core/simd_ops.h"
#include "core/aligned_alloc.h"
#include "core/vector_ops.h"
#include "core/thread_pool.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

namespace xai {

/* ================================================================
 * DOT PRODUCT BY WEIGHT FORMAT
 * ================================================================ */

float wt_dot_row(const WeightTensor *w, int row, const float *vec) {
    int cols = w->cols;
    switch (w->fmt) {
    case WF_F32: {
        const float *r = (const float*)w->data + (int64_t)row * cols;
        return simd::dot_f32u(r, vec, cols);
    }
    case WF_F16: {
        const uint16_t *r = (const uint16_t*)w->data + (int64_t)row * cols;
        return simd::dot_f16_f32(r, vec, cols);
    }
    case WF_Q8: {
        const int8_t *r = (const int8_t*)w->data + (int64_t)row * cols;
        return simd::dot_q8_f32(r, w->scale, vec, cols);
    }
    }
    return 0.0f;
}

/* ================================================================
 * MATVEC MULTIPLICATION
 * ================================================================ */

void wt_matmul_rows(float *out, const WeightTensor *w,
                    const float *vec, int r0, int r1) {
    for (int r = r0; r < r1; r++) out[r] = wt_dot_row(w, r, vec);
}

void wt_matmul(float *out, const WeightTensor *w, const float *vec) {
    if (!g_pool || w->rows < 32) {
        wt_matmul_rows(out, w, vec, 0, w->rows); return;
    }
    g_pool->parallel_for(w->rows, [&](int, int lo, int hi) {
        wt_matmul_rows(out, w, vec, lo, hi);
    });
}

void wt_matmul_dual(float *out1, float *out2,
                    const WeightTensor *w1, const WeightTensor *w2,
                    const float *vec) {
    if (w1->rows != w2->rows) {
        fprintf(stderr, "wt_matmul_dual: row count mismatch (%d vs %d)\n",
                w1->rows, w2->rows);
        exit(1);
    }
    int rows = w1->rows;
    if (!g_pool || rows < 32) {
        wt_matmul_rows(out1, w1, vec, 0, rows);
        wt_matmul_rows(out2, w2, vec, 0, rows);
        return;
    }
    g_pool->parallel_for(rows, [&](int, int lo, int hi) {
        for (int r = lo; r < hi; r++) {
            out1[r] = wt_dot_row(w1, r, vec);
            out2[r] = wt_dot_row(w2, r, vec);
        }
    });
}

void wt_matmul_qkv(float *oq, float *ok, float *ov,
                   const WeightTensor *wq, const WeightTensor *wk,
                   const WeightTensor *wv, const float *vec) {
    int rq = wq->rows, rkv = wk->rows;
    if (!g_pool || g_pool->num_workers() < 2) {
        wt_matmul(oq, wq, vec);
        wt_matmul(ok, wk, vec);
        wt_matmul(ov, wv, vec);
        return;
    }
    int total = rq + rkv + rkv;
    g_pool->parallel_for(total, [&](int, int lo, int hi) {
        for (int i = lo; i < hi; i++) {
            if (i < rq)            oq[i]       = wt_dot_row(wq, i, vec);
            else if (i < rq + rkv) ok[i - rq]  = wt_dot_row(wk, i - rq, vec);
            else                   ov[i-rq-rkv] = wt_dot_row(wv, i-rq-rkv, vec);
        }
    });
}

/* ================================================================
 * ARGMAX
 * ================================================================ */

int wt_matmul_argmax(const WeightTensor *w, const float *vec) {
    int rows = w->rows, best_idx = 0;
    float best_val = -INFINITY;

    if (!g_pool || rows < 64) {
        for (int r = 0; r < rows; r++) {
            float val = wt_dot_row(w, r, vec);
            if (val > best_val) { best_val = val; best_idx = r; }
        }
        return best_idx;
    }

    struct alignas(CACHE_LINE) LocalBest { int idx; float val; };
    int nt = std::min(g_pool->num_workers(), rows);
    std::vector<LocalBest> locals(nt);
    for (int i = 0; i < nt; i++) { locals[i].idx = 0; locals[i].val = -INFINITY; }

    g_pool->parallel_for(rows, [&](int tid, int lo, int hi) {
        int li = 0; float lv = -INFINITY;
        for (int r = lo; r < hi; r++) {
            float val = wt_dot_row(w, r, vec);
            if (val > lv) { lv = val; li = r; }
        }
        locals[tid].idx = li;
        locals[tid].val = lv;
    });

    for (int i = 0; i < nt; i++)
        if (locals[i].val > best_val) { best_val = locals[i].val; best_idx = locals[i].idx; }
    return best_idx;
}

void wt_matmul_logits(float *logits, const WeightTensor *w,
                      const float *vec) {
    wt_matmul(logits, w, vec);
}

/* ================================================================
 * WEIGHT LOADING
 * ================================================================ */

TensorMeta *find_tensor(Model *m, const char *name) {
    for (int i = 0; i < m->num_tensors; i++)
        if (!strcmp(m->tensors[i].name, name)) return &m->tensors[i];
    return nullptr;
}

float *load_1d_f32(Model *m, const char *name) {
    TensorMeta *t = find_tensor(m, name);
    if (!t) { fprintf(stderr, "Tensor not found: %s\n", name); exit(1); }
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    char *src = m->tensor_data_start + t->offset;
    switch (t->dtype) {
    case DTYPE_F32: {
        float *out = aligned_calloc_f32((size_t)n);
        memcpy(out, src, n * sizeof(float)); return out;
    }
    case DTYPE_F16: return convert_f16_to_f32((const uint16_t*)src, n);
    case DTYPE_Q8:  return convert_q8_to_f32((const int8_t*)src, t->scale, n);
    default:
        fprintf(stderr, "Cannot load 1D tensor %s\n", name); exit(1);
    }
}

WeightTensor load_weight_native(Model *m, const char *name) {
    TensorMeta *t = find_tensor(m, name);
    if (!t) { fprintf(stderr, "Weight not found: %s\n", name); exit(1); }
    WeightTensor wt;
    wt.rows = (t->ndim >= 2) ? t->shape[0] : 1;
    wt.cols = (t->ndim >= 2) ? t->shape[1] : t->shape[0];
    char *src = m->tensor_data_start + t->offset;
    switch (t->dtype) {
    case DTYPE_F32: wt.data = src; wt.fmt = WF_F32;                         break;
    case DTYPE_F16: wt.data = src; wt.fmt = WF_F16;                         break;
    case DTYPE_Q8:  wt.data = src; wt.fmt = WF_Q8; wt.scale = t->scale;    break;
    default:
        fprintf(stderr, "Unsupported dtype for %s\n", name); exit(1);
    }
    return wt;
}

float *load_embedding(Model *m, const char *name) {
    return load_1d_f32(m, name);
}

} // namespace xai