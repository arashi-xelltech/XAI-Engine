#include "inference/forward.h"
#include "model/model_loader.h"
#include "model/tensor_ops.h"
#include "core/vector_ops.h"
#include "core/simd_ops.h"        // <-- ДОБАВЛЕНО для simd::vec_add
#include "model/rope.h"
#include "core/aligned_alloc.h"
#include "core/thread_pool.h"
#include <cstring>
#include <cmath>

namespace xai {

void alloc_state(RunState *s, const ModelConfig *c) {
    s->x      = aligned_calloc_f32(c->hidden_size);
    s->xb     = aligned_calloc_f32(c->hidden_size);
    s->xb2    = aligned_calloc_f32(c->hidden_size);
    s->q      = aligned_calloc_f32((size_t)c->num_heads * c->head_dim);
    s->k      = aligned_calloc_f32(c->kv_dim);
    s->v      = aligned_calloc_f32(c->kv_dim);
    s->att    = aligned_calloc_f32((size_t)c->num_heads * c->max_seq_len);
    s->hb     = aligned_calloc_f32(c->intermediate_size);
    s->hb2    = aligned_calloc_f32(c->intermediate_size);
    s->logits = aligned_calloc_f32(c->vocab_size);
    s->kv_cache.alloc(c->num_layers, c->num_kv_heads, c->max_seq_len, c->head_dim);
    s->pos = 0;
}

void free_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->q);
    free(s->k); free(s->v); free(s->att);
    free(s->hb); free(s->hb2); free(s->logits);
    s->kv_cache.free_mem();
}

void reset_state(RunState *s, const ModelConfig *) {
    s->kv_cache.reset();
    s->pos = 0;
}

/* ================================================================
 * TRANSFORMER FORWARD
 * ================================================================ */

void forward_transformer(Model *m, RunState *s, int token, int pos) {
    ModelConfig *c = &m->cfg;
    int h    = c->hidden_size;
    int nh   = c->num_heads;
    int nkv  = c->num_kv_heads;
    int hd   = c->head_dim;
    int inter = c->intermediate_size;
    int gs   = c->gqa_groups;

    vec_copy(s->x, m->embed + (int64_t)token * h, h);

    for (int l = 0; l < c->num_layers; l++) {
        LayerWeights *lw = &m->layers[l];

        /* Attention */
        rmsnorm(s->xb, s->x, lw->attn_norm, h, c->rms_norm_eps);
        wt_matmul_qkv(s->q, s->k, s->v,
                      &lw->q_proj, &lw->k_proj, &lw->v_proj, s->xb);

        for (int head = 0; head < nh;  head++) apply_rope(s->q + head*hd, hd, pos, m->rope_cos, m->rope_sin);
        for (int head = 0; head < nkv; head++) apply_rope(s->k + head*hd, hd, pos, m->rope_cos, m->rope_sin);

        for (int head = 0; head < nkv; head++) {
            s->kv_cache.store_key  (l, head, pos, s->k + head*hd, hd);
            s->kv_cache.store_value(l, head, pos, s->v + head*hd, hd);
        }

        float scale     = 1.0f / sqrtf((float)hd);
        int att_start   = 0;
        if (g_sliding_window > 0 && pos >= g_sliding_window)
            att_start = pos - g_sliding_window + 1;
        int att_len = pos - att_start + 1;

        auto do_attention = [&](int head) {
            int kv_head = head / gs;
            float *qh  = s->q   + head * hd;
            float *att = s->att + head * c->max_seq_len;
            for (int t = att_start; t <= pos; t++)
                att[t - att_start] =
                    s->kv_cache.dot_key(l, kv_head, t, qh, hd) * scale;
            softmax_exact(att, att_len);
            float *out_h = s->xb2 + head * hd;
            memset(out_h, 0, hd * sizeof(float));
            for (int t = att_start; t <= pos; t++)
                s->kv_cache.fma_value(l, kv_head, t,
                                      att[t - att_start], out_h, hd);
        };

        if (g_pool && nh >= g_num_threads) {
            g_pool->parallel_for(nh, [&](int, int lo, int hi) {
                for (int head = lo; head < hi; head++) do_attention(head);
            });
        } else {
            for (int head = 0; head < nh; head++) do_attention(head);
        }

        wt_matmul(s->xb, &lw->o_proj, s->xb2);
        fused_residual_rmsnorm(s->x, s->xb, s->xb, lw->ffn_norm, h, c->rms_norm_eps);

        /* FFN */
        wt_matmul_dual(s->hb, s->hb2, &lw->gate_proj, &lw->up_proj, s->xb);
        silu_mul(s->hb, s->hb2, inter);
        wt_matmul(s->xb, &lw->down_proj, s->hb);
        simd::vec_add(s->x, s->xb, h);
    }

    rmsnorm(s->x, s->x, m->final_norm, h, c->rms_norm_eps);
}

void forward(Model *m, RunState *s, int token, int pos) {
    forward_transformer(m, s, token, pos);
    wt_matmul_logits(s->logits, &m->lm_head, s->x);
}

void forward_batch(Model *m, RunState *s, const int *tokens,
                   int start_pos, int batch_len) {
    for (int b = 0; b < batch_len; b++) {
        if (b == batch_len - 1) forward(m, s, tokens[b], start_pos + b);
        else                    forward_transformer(m, s, tokens[b], start_pos + b);
    }
}

} // namespace xai