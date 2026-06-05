#include "core/kv_cache.h"
#include "core/aligned_alloc.h"
#include "core/simd_ops.h"
#include <cstring>
#include <cmath>

namespace xai {

void KVCache::alloc(int nl, int nkv, int msl, int hd) {
    num_layers = nl; num_kv_heads = nkv; max_seq_len = msl; head_dim = hd;
    int64_t kd_sz = (int64_t)nl * nkv * msl * hd;
    int64_t ks_sz = (int64_t)nl * nkv * msl;
    int64_t vd_sz = (int64_t)nl * nkv * msl * hd;

    key_data   = (int8_t*)portable_aligned_alloc(SIMD_ALIGN, kd_sz);
    key_scales = (float*) portable_aligned_alloc(SIMD_ALIGN, ks_sz * sizeof(float));
    val_data   = (float*) portable_aligned_alloc(SIMD_ALIGN, vd_sz * sizeof(float));

    memset(key_data,   0, kd_sz);
    memset(key_scales, 0, ks_sz * sizeof(float));
    memset(val_data,   0, vd_sz * sizeof(float));
}

void KVCache::reset() {
    int64_t kd_sz = (int64_t)num_layers * num_kv_heads * max_seq_len * head_dim;
    int64_t ks_sz = (int64_t)num_layers * num_kv_heads * max_seq_len;
    memset(key_data,   0, kd_sz);
    memset(key_scales, 0, ks_sz * sizeof(float));
    memset(val_data,   0,
           (int64_t)num_layers * num_kv_heads * max_seq_len * head_dim * sizeof(float));
}

void KVCache::free_mem() {
    free(key_data); free(key_scales); free(val_data);
    key_data = nullptr; key_scales = nullptr; val_data = nullptr;
}

void KVCache::store_key(int layer, int head, int pos, const float *src, int hd) {
    int64_t base  = ((int64_t)layer * num_kv_heads + head) * max_seq_len;
    int64_t d_off = (base + pos) * (int64_t)head_dim;
    int64_t s_off = base + pos;

    float amax = 0.0f;
    for (int i = 0; i < hd; i++) { float a = fabsf(src[i]); if (a > amax) amax = a; }
    float sc  = amax / 127.0f;
    float inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    key_scales[s_off] = sc;
    int8_t *dst = key_data + d_off;
    for (int i = 0; i < hd; i++) {
        int v = (int)roundf(src[i] * inv);
        dst[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
}

void KVCache::store_value(int layer, int head, int pos, const float *src, int hd) {
    int64_t base = ((int64_t)layer * num_kv_heads + head) * max_seq_len;
    float *dst   = val_data + (base + pos) * (int64_t)head_dim;
    memcpy(dst, src, hd * sizeof(float));
}

float KVCache::dot_key(int layer, int head, int pos,
                       const float *query, int hd) const {
    int64_t base  = ((int64_t)layer * num_kv_heads + head) * max_seq_len;
    int64_t d_off = (base + pos) * (int64_t)head_dim;
    float sc      = key_scales[base + pos];
    const int8_t *k = key_data + d_off;

    float sum = 0.0f;
    int i = 0;
#ifdef USE_AVX
    __m256 vs0 = _mm256_setzero_ps();
    __m256 vs1 = _mm256_setzero_ps();
    for (; i + 16 <= hd; i += 16) {
        PREFETCH_READ(k + i + 32);
        __m128i raw0  = _mm_loadl_epi64((const __m128i*)(k + i));
        __m128i i16_0 = _mm_cvtepi8_epi16(raw0);
        __m128i lo0   = _mm_cvtepi16_epi32(i16_0);
        __m128i hi0   = _mm_cvtepi16_epi32(_mm_srli_si128(i16_0, 8));
        __m256 f0 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo0)),
            _mm_cvtepi32_ps(hi0), 1);

        __m128i raw1  = _mm_loadl_epi64((const __m128i*)(k + i + 8));
        __m128i i16_1 = _mm_cvtepi8_epi16(raw1);
        __m128i lo1   = _mm_cvtepi16_epi32(i16_1);
        __m128i hi1   = _mm_cvtepi16_epi32(_mm_srli_si128(i16_1, 8));
        __m256 f1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo1)),
            _mm_cvtepi32_ps(hi1), 1);

        vs0 = _mm256_add_ps(vs0, _mm256_mul_ps(f0, _mm256_load_ps(query + i)));
        vs1 = _mm256_add_ps(vs1, _mm256_mul_ps(f1, _mm256_load_ps(query + i + 8)));
    }
    for (; i + 8 <= hd; i += 8) {
        __m128i raw = _mm_loadl_epi64((const __m128i*)(k + i));
        __m128i i16 = _mm_cvtepi8_epi16(raw);
        __m128i lo  = _mm_cvtepi16_epi32(i16);
        __m128i hi  = _mm_cvtepi16_epi32(_mm_srli_si128(i16, 8));
        __m256 fv   = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo)),
            _mm_cvtepi32_ps(hi), 1);
        vs0 = _mm256_add_ps(vs0, _mm256_mul_ps(fv, _mm256_load_ps(query + i)));
    }
    vs0 = _mm256_add_ps(vs0, vs1);
    {
        __m128 h128 = _mm256_extractf128_ps(vs0, 1);
        __m128 l128 = _mm256_castps256_ps128(vs0);
        __m128 s128 = _mm_add_ps(h128, l128);
        s128 = _mm_hadd_ps(s128, s128);
        s128 = _mm_hadd_ps(s128, s128);
        sum  = _mm_cvtss_f32(s128);
    }
#endif
    for (; i < hd; i++) sum += query[i] * (float)k[i];
    return sum * sc;
}

void KVCache::fma_value(int layer, int head, int pos,
                        float weight, float *out, int hd) const {
    int64_t base  = ((int64_t)layer * num_kv_heads + head) * max_seq_len;
    const float *v = val_data + (base + pos) * (int64_t)head_dim;

    int i = 0;
#ifdef USE_AVX
    __m256 vw = _mm256_set1_ps(weight);
    for (; i + 16 <= hd; i += 16) {
        __m256 o0 = _mm256_load_ps(out + i);
        __m256 o1 = _mm256_load_ps(out + i + 8);
        o0 = _mm256_add_ps(o0, _mm256_mul_ps(_mm256_load_ps(v + i), vw));
        o1 = _mm256_add_ps(o1, _mm256_mul_ps(_mm256_load_ps(v + i + 8), vw));
        _mm256_store_ps(out + i,     o0);
        _mm256_store_ps(out + i + 8, o1);
    }
    for (; i + 8 <= hd; i += 8) {
        __m256 o = _mm256_load_ps(out + i);
        _mm256_store_ps(out + i,
            _mm256_add_ps(o, _mm256_mul_ps(_mm256_load_ps(v + i), vw)));
    }
#endif
    for (; i < hd; i++) out[i] += weight * v[i];
}

} // namespace xai