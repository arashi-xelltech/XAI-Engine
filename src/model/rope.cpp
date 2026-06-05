#include "model/rope.h"
#include "model/model_loader.h"   // <-- ДОБАВЛЕНО для полного определения Model
#include "core/aligned_alloc.h"
#include "core/simd_ops.h"
#include <cmath>

namespace xai {

void build_rope(Model *m) {
    int dim    = m->cfg.head_dim;
    int max_sl = m->cfg.max_seq_len;
    float theta = m->cfg.rope_theta;
    m->rope_cos = aligned_calloc_f32((size_t)max_sl * dim);
    m->rope_sin = aligned_calloc_f32((size_t)max_sl * dim);
    for (int pos = 0; pos < max_sl; pos++) {
        for (int i = 0; i < dim / 2; i++) {
            float freq  = 1.0f / powf(theta, (float)(2*i) / dim);
            float angle = pos * freq;
            float c = cosf(angle), s = sinf(angle);
            m->rope_cos[pos*dim + i]         = c;
            m->rope_cos[pos*dim + i + dim/2] = c;
            m->rope_sin[pos*dim + i]         = s;
            m->rope_sin[pos*dim + i + dim/2] = s;
        }
    }
}

void apply_rope(float *vec, int hd, int pos,
                const float *cos_tbl, const float *sin_tbl) {
    const float *c = cos_tbl + pos * hd;
    const float *s = sin_tbl + pos * hd;
    int half = hd / 2, i = 0;
#ifdef USE_AVX
    for (; i + 8 <= half; i += 8) {
        __m256 v0 = _mm256_load_ps(vec + i);
        __m256 v1 = _mm256_load_ps(vec + i + half);
        __m256 c0 = _mm256_load_ps(c + i);
        __m256 c1 = _mm256_load_ps(c + i + half);
        __m256 s0 = _mm256_load_ps(s + i);
        __m256 s1 = _mm256_load_ps(s + i + half);
        _mm256_store_ps(vec + i,
            _mm256_sub_ps(_mm256_mul_ps(v0, c0), _mm256_mul_ps(v1, s0)));
        _mm256_store_ps(vec + i + half,
            _mm256_add_ps(_mm256_mul_ps(v1, c1), _mm256_mul_ps(v0, s1)));
    }
#endif
    for (; i < half; i++) {
        float a = vec[i], b = vec[i + half];
        vec[i]        = a * c[i]        - b * s[i];
        vec[i + half] = b * c[i + half] + a * s[i + half];
    }
}

} // namespace xai