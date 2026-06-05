#include "core/vector_ops.h"
#include "core/simd_ops.h"
#include "core/aligned_alloc.h"
#include <cmath>
#include <cstring>

namespace xai {

void vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* ================================================================
 * RMSNORM
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *w,
             int size, float eps) {
    float ss    = simd::sum_squares(x, size);
    float scale = 1.0f / sqrtf(ss / size + eps);
    simd::vec_scale_mul(out, x, w, scale, size);
}

void fused_residual_rmsnorm(float *x, const float *residual,
                            float *out, const float *w,
                            int size, float eps) {
    float ss = 0.0f;
    int i = 0;
#ifdef USE_AVX
    __m256 vss0 = _mm256_setzero_ps(), vss1 = _mm256_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m256 vx0 = _mm256_load_ps(x + i),        vr0 = _mm256_load_ps(residual + i);
        __m256 vx1 = _mm256_load_ps(x + i + 8),    vr1 = _mm256_load_ps(residual + i + 8);
        vx0 = _mm256_add_ps(vx0, vr0);
        vx1 = _mm256_add_ps(vx1, vr1);
        _mm256_store_ps(x + i,     vx0);
        _mm256_store_ps(x + i + 8, vx1);
        vss0 = _mm256_add_ps(vss0, _mm256_mul_ps(vx0, vx0));
        vss1 = _mm256_add_ps(vss1, _mm256_mul_ps(vx1, vx1));
    }
    for (; i + 8 <= size; i += 8) {
        __m256 vx = _mm256_load_ps(x + i), vr = _mm256_load_ps(residual + i);
        vx = _mm256_add_ps(vx, vr);
        _mm256_store_ps(x + i, vx);
        vss0 = _mm256_add_ps(vss0, _mm256_mul_ps(vx, vx));
    }
    vss0 = _mm256_add_ps(vss0, vss1);
    ss   = simd::hsum_avx(vss0);
#endif
    for (; i < size; i++) { x[i] += residual[i]; ss += x[i]*x[i]; }
    float scale = 1.0f / sqrtf(ss / size + eps);
    simd::vec_scale_mul(out, x, w, scale, size);
}

/* ================================================================
 * SOFTMAX
 * ================================================================ */

void softmax(float *x, int size) {
    float max_v = simd::vec_max(x, size);
    float sum   = 0.0f;
    int i = 0;
#ifdef USE_AVX
    __m256 vmax  = _mm256_set1_ps(max_v);
    __m256 vsum0 = _mm256_setzero_ps(), vsum1 = _mm256_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m256 v0 = simd::fast_exp_avx(_mm256_sub_ps(_mm256_load_ps(x+i),   vmax));
        __m256 v1 = simd::fast_exp_avx(_mm256_sub_ps(_mm256_load_ps(x+i+8), vmax));
        _mm256_store_ps(x+i,   v0);
        _mm256_store_ps(x+i+8, v1);
        vsum0 = _mm256_add_ps(vsum0, v0);
        vsum1 = _mm256_add_ps(vsum1, v1);
    }
    for (; i + 8 <= size; i += 8) {
        __m256 v = simd::fast_exp_avx(_mm256_sub_ps(_mm256_load_ps(x+i), vmax));
        _mm256_store_ps(x+i, v);
        vsum0 = _mm256_add_ps(vsum0, v);
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    sum   = simd::hsum_avx(vsum0);
#endif
    for (; i < size; i++) { x[i] = expf(x[i] - max_v); sum += x[i]; }
    simd::vec_scale(x, 1.0f / sum, size);
}

void softmax_exact(float *x, int size) {
    float max_v = simd::vec_max(x, size), sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_v); sum += x[i]; }
    simd::vec_scale(x, 1.0f / sum, size);
}

/* ================================================================
 * SILU
 * ================================================================ */

void silu_mul(float *gate, const float *up, int size) {
    int i = 0;
#ifdef USE_AVX
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_load_ps(gate + i);
        __m256 u = _mm256_load_ps(up + i);
        _mm256_store_ps(gate + i, _mm256_mul_ps(simd::fast_silu_avx(g), u));
    }
#endif
    for (; i < size; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* ================================================================
 * F16/Q8 CONVERT
 * ================================================================ */

float *convert_f16_to_f32(const uint16_t *src, int64_t n) {
    float *out = aligned_calloc_f32((size_t)n);
    int64_t i = 0;
#ifdef USE_AVX
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m256 f  = simd::cvt_f16_to_f32x8(h);
        _mm256_store_ps(out + i, f);
    }
#endif
    for (; i < n; i++) {
        uint16_t h    = src[i];
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t expo = (h >> 10) & 0x1Fu;
        uint32_t mant = h & 0x3FFu;
        uint32_t f;
        if (expo == 0) {
            if (mant == 0) f = sign;
            else {
                expo = 1;
                while (!(mant & 0x400)) { mant <<= 1; expo--; }
                mant &= 0x3FFu;
                f = sign | ((expo + 112u) << 23) | (mant << 13);
            }
        } else if (expo == 31) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((expo + 112u) << 23) | (mant << 13);
        }
        float r; memcpy(&r, &f, 4);
        out[i] = r;
    }
    return out;
}

float *convert_q8_to_f32(const int8_t *src, float scale, int64_t n) {
    float *out = aligned_calloc_f32((size_t)n);
    int64_t i = 0;
#ifdef USE_AVX
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m128i raw = _mm_loadl_epi64((const __m128i*)(src + i));
        __m128i i16 = _mm_cvtepi8_epi16(raw);
        __m128i lo  = _mm_cvtepi16_epi32(i16);
        __m128i hi  = _mm_cvtepi16_epi32(_mm_srli_si128(i16, 8));
        __m256 fv   = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo)),
            _mm_cvtepi32_ps(hi), 1);
        _mm256_store_ps(out + i, _mm256_mul_ps(fv, vs));
    }
#endif
    for (; i < n; i++) out[i] = src[i] * scale;
    return out;
}

} // namespace xai