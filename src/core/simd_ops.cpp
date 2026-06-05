#include "simd_ops.h"
#include <cstring>
#include <cmath>

namespace xai {
namespace simd {

/* ================================================================
 * КОНВЕРТАЦИЯ F16 → F32
 * ================================================================ */

// Скалярная конвертация одного значения F16 в F32
static inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (expo == 0) {
        if (mant == 0) bits = sign;
        else {
            // Денормализованное число
            expo = 1;
            while (!(mant & 0x400)) { mant <<= 1; expo--; }
            mant &= 0x3FFu;
            bits = sign | ((expo + 112u) << 23) | (mant << 13);
        }
    } else if (expo == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((expo + 112u) << 23) | (mant << 13);
    }
    float fv;
    memcpy(&fv, &bits, 4);
    return fv;
}

/* ================================================================
 * УРОВЕНЬ 1: AVX (Sandy Bridge+, 2011+)
 * ================================================================ */

#ifdef USE_AVX

float hsum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__m256 fast_exp_avx(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));

    __m256 log2e = _mm256_set1_ps(1.44269504089f);
    __m256 t     = _mm256_mul_ps(x, log2e);
    __m256 n     = _mm256_floor_ps(t);
    __m256 f     = _mm256_sub_ps(t, n);

    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.6931472f);
    __m256 c2 = _mm256_set1_ps(0.2402265f);
    __m256 c3 = _mm256_set1_ps(0.0554960f);
    __m256 c4 = _mm256_set1_ps(0.0096760f);

    __m256 p = _mm256_add_ps(c3, _mm256_mul_ps(c4, f));
    p = _mm256_add_ps(c2, _mm256_mul_ps(p, f));
    p = _mm256_add_ps(c1, _mm256_mul_ps(p, f));
    p = _mm256_add_ps(c0, _mm256_mul_ps(p, f));

    __m256i ni    = _mm256_cvtps_epi32(n);
    __m128i ni_lo = _mm256_castsi256_si128(ni);
    __m128i ni_hi = _mm256_extractf128_si256(ni, 1);
    ni_lo = _mm_slli_epi32(_mm_add_epi32(ni_lo, _mm_set1_epi32(127)), 23);
    ni_hi = _mm_slli_epi32(_mm_add_epi32(ni_hi, _mm_set1_epi32(127)), 23);
    __m256 pow2n = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_castsi128_ps(ni_lo)),
        _mm_castsi128_ps(ni_hi), 1);

    return _mm256_mul_ps(p, pow2n);
}

__m256 soft_cvtph_ps(__m128i h8) {
    __m128i lo_u16 = h8;
    __m128i hi_u16 = _mm_srli_si128(h8, 8);

    __m128i lo32 = _mm_cvtepu16_epi32(lo_u16);
    __m128i hi32 = _mm_cvtepu16_epi32(hi_u16);

    __m128i m_sign = _mm_set1_epi32(0x8000);
    __m128i m_expo = _mm_set1_epi32(0x1F);
    __m128i m_mant = _mm_set1_epi32(0x3FF);
    __m128i bias   = _mm_set1_epi32(112);

    /* lower 4 */
    __m128i s_lo = _mm_slli_epi32(_mm_and_si128(lo32, m_sign), 16);
    __m128i e_lo = _mm_and_si128(_mm_srli_epi32(lo32, 10), m_expo);
    __m128i f_lo = _mm_and_si128(lo32, m_mant);
    __m128i b_lo = _mm_or_si128(s_lo,
                    _mm_or_si128(_mm_slli_epi32(_mm_add_epi32(e_lo, bias), 23),
                                 _mm_slli_epi32(f_lo, 13)));
    __m128i z_lo = _mm_cmpeq_epi32(e_lo, _mm_setzero_si128());
    b_lo = _mm_or_si128(_mm_andnot_si128(z_lo, b_lo),
                        _mm_and_si128(z_lo, s_lo));

    /* upper 4 */
    __m128i s_hi = _mm_slli_epi32(_mm_and_si128(hi32, m_sign), 16);
    __m128i e_hi = _mm_and_si128(_mm_srli_epi32(hi32, 10), m_expo);
    __m128i f_hi = _mm_and_si128(hi32, m_mant);
    __m128i b_hi = _mm_or_si128(s_hi,
                    _mm_or_si128(_mm_slli_epi32(_mm_add_epi32(e_hi, bias), 23),
                                 _mm_slli_epi32(f_hi, 13)));
    __m128i z_hi = _mm_cmpeq_epi32(e_hi, _mm_setzero_si128());
    b_hi = _mm_or_si128(_mm_andnot_si128(z_hi, b_hi),
                        _mm_and_si128(z_hi, s_hi));

    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_castsi128_ps(b_lo)),
        _mm_castsi128_ps(b_hi), 1);
}

__m256 cvt_f16_to_f32x8(__m128i h8) {
#ifdef USE_F16C
    return _mm256_cvtph_ps(h8);
#else
    return soft_cvtph_ps(h8);
#endif
}

__m256 fast_silu_avx(__m256 x) {
    __m256 neg_x   = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = fast_exp_avx(neg_x);
    __m256 denom   = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg);
    __m256 rcp     = _mm256_rcp_ps(denom);
    rcp = _mm256_mul_ps(rcp,
            _mm256_sub_ps(_mm256_set1_ps(2.0f),
                          _mm256_mul_ps(denom, rcp)));
    return _mm256_mul_ps(x, rcp);
}

/* ================================================================
 * AVX: DOT PRODUCTS
 * ================================================================ */

float dot_f32(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        PREFETCH_READ(a + i + 64); PREFETCH_READ(b + i + 64);
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_load_ps(a+i),    _mm256_load_ps(b+i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_load_ps(a+i+8),  _mm256_load_ps(b+i+8)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(_mm256_load_ps(a+i+16), _mm256_load_ps(b+i+16)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(_mm256_load_ps(a+i+24), _mm256_load_ps(b+i+24)));
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_load_ps(a+i), _mm256_load_ps(b+i)));
    s0 = _mm256_add_ps(_mm256_add_ps(s0,s1), _mm256_add_ps(s2,s3));
    float r = hsum_avx(s0);
    for (; i < n; i++) r += a[i]*b[i];
    return r;
}

float dot_f32u(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        PREFETCH_READ(a+i+64); PREFETCH_READ(b+i+64);
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24)));
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
    s0 = _mm256_add_ps(_mm256_add_ps(s0,s1), _mm256_add_ps(s2,s3));
    float r = hsum_avx(s0);
    for (; i < n; i++) r += a[i]*b[i];
    return r;
}

float dot_f16_f32(const uint16_t *f16, const float *f32, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        PREFETCH_READ(f16 + i + 64);
        PREFETCH_READ(f32 + i + 32);
        __m256 a0 = cvt_f16_to_f32x8(_mm_loadu_si128((const __m128i*)(f16+i)));
        __m256 a1 = cvt_f16_to_f32x8(_mm_loadu_si128((const __m128i*)(f16+i+8)));
        __m256 a2 = cvt_f16_to_f32x8(_mm_loadu_si128((const __m128i*)(f16+i+16)));
        __m256 a3 = cvt_f16_to_f32x8(_mm_loadu_si128((const __m128i*)(f16+i+24)));
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(a0, _mm256_loadu_ps(f32+i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(a1, _mm256_loadu_ps(f32+i+8)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(a2, _mm256_loadu_ps(f32+i+16)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(a3, _mm256_loadu_ps(f32+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        __m256 a = cvt_f16_to_f32x8(_mm_loadu_si128((const __m128i*)(f16+i)));
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(a, _mm256_loadu_ps(f32+i)));
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0,s1), _mm256_add_ps(s2,s3));
    float result = hsum_avx(s0);
    for (; i < n; i++) result += f16_to_f32_scalar(f16[i]) * f32[i];
    return result;
}

float dot_q8_f32(const int8_t *q8, float scale, const float *f32, int n) {
    __m256 vs0 = _mm256_setzero_ps();
    __m256 vs1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        PREFETCH_READ(q8 + i + 32);
        PREFETCH_READ(f32 + i + 16);

        __m128i raw0  = _mm_loadl_epi64((const __m128i*)(q8 + i));
        __m128i i16_0 = _mm_cvtepi8_epi16(raw0);
        __m128i lo0   = _mm_cvtepi16_epi32(i16_0);
        __m128i hi0   = _mm_cvtepi16_epi32(_mm_srli_si128(i16_0, 8));
        __m256 f0 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo0)),
            _mm_cvtepi32_ps(hi0), 1);

        __m128i raw1  = _mm_loadl_epi64((const __m128i*)(q8 + i + 8));
        __m128i i16_1 = _mm_cvtepi8_epi16(raw1);
        __m128i lo1   = _mm_cvtepi16_epi32(i16_1);
        __m128i hi1   = _mm_cvtepi16_epi32(_mm_srli_si128(i16_1, 8));
        __m256 f1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo1)),
            _mm_cvtepi32_ps(hi1), 1);

        vs0 = _mm256_add_ps(vs0, _mm256_mul_ps(f0, _mm256_loadu_ps(f32 + i)));
        vs1 = _mm256_add_ps(vs1, _mm256_mul_ps(f1, _mm256_loadu_ps(f32 + i + 8)));
    }
    for (; i + 8 <= n; i += 8) {
        __m128i raw = _mm_loadl_epi64((const __m128i*)(q8 + i));
        __m128i i16 = _mm_cvtepi8_epi16(raw);
        __m128i lo  = _mm_cvtepi16_epi32(i16);
        __m128i hi  = _mm_cvtepi16_epi32(_mm_srli_si128(i16, 8));
        __m256 fv   = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(lo)),
            _mm_cvtepi32_ps(hi), 1);
        vs0 = _mm256_add_ps(vs0, _mm256_mul_ps(fv, _mm256_loadu_ps(f32 + i)));
    }
    vs0 = _mm256_add_ps(vs0, vs1);
    float sum = hsum_avx(vs0);
    for (; i < n; i++) sum += (float)q8[i] * f32[i];
    return sum * scale;
}

float sum_squares(const float *x, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_load_ps(x + i);
        __m256 v1 = _mm256_load_ps(x + i + 8);
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(v0, v0));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(v1, v1));
    }
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_load_ps(x + i);
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(v, v));
    }
    s0 = _mm256_add_ps(s0, s1);
    float r = hsum_avx(s0);
    for (; i < n; i++) r += x[i]*x[i];
    return r;
}

void vec_scale_mul(float *out, const float *x, const float *w, float s, int n) {
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        PREFETCH_WRITE(out + i + 32);
        __m256 x0 = _mm256_load_ps(x + i);
        __m256 w0 = _mm256_load_ps(w + i);
        __m256 x1 = _mm256_load_ps(x + i + 8);
        __m256 w1 = _mm256_load_ps(w + i + 8);
        _mm256_store_ps(out + i,     _mm256_mul_ps(_mm256_mul_ps(x0, vs), w0));
        _mm256_store_ps(out + i + 8, _mm256_mul_ps(_mm256_mul_ps(x1, vs), w1));
    }
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vw = _mm256_load_ps(w + i);
        _mm256_store_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vs), vw));
    }
    for (; i < n; i++) out[i] = x[i] * s * w[i];
}

void vec_add(float *out, const float *a, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_store_ps(out+i,
            _mm256_add_ps(_mm256_load_ps(out+i), _mm256_load_ps(a+i)));
        _mm256_store_ps(out+i+8,
            _mm256_add_ps(_mm256_load_ps(out+i+8), _mm256_load_ps(a+i+8)));
    }
    for (; i + 8 <= n; i += 8)
        _mm256_store_ps(out+i,
            _mm256_add_ps(_mm256_load_ps(out+i), _mm256_load_ps(a+i)));
    for (; i < n; i++) out[i] += a[i];
}

void vec_scale(float *x, float s, int n) {
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_store_ps(x+i, _mm256_mul_ps(_mm256_load_ps(x+i), vs));
    for (; i < n; i++) x[i] *= s;
}

void vec_fma(float *out, const float *v, float w, int n) {
    __m256 vw = _mm256_set1_ps(w);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        PREFETCH_READ(v + i + 32);
        __m256 o0 = _mm256_load_ps(out + i);
        __m256 o1 = _mm256_load_ps(out + i + 8);
        _mm256_store_ps(out+i,
            _mm256_add_ps(o0, _mm256_mul_ps(_mm256_load_ps(v+i), vw)));
        _mm256_store_ps(out+i+8,
            _mm256_add_ps(o1, _mm256_mul_ps(_mm256_load_ps(v+i+8), vw)));
    }
    for (; i + 8 <= n; i += 8) {
        __m256 o = _mm256_load_ps(out + i);
        _mm256_store_ps(out+i,
            _mm256_add_ps(o, _mm256_mul_ps(_mm256_load_ps(v+i), vw)));
    }
    for (; i < n; i++) out[i] += w * v[i];
}

float vec_max(const float *x, int n) {
    if (n == 0) return -INFINITY;
    __m256 vm = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        vm = _mm256_max_ps(vm, _mm256_load_ps(x + i));
    __m128 hi   = _mm256_extractf128_ps(vm, 1);
    __m128 lo   = _mm256_castps256_ps128(vm);
    __m128 m128 = _mm_max_ps(hi, lo);
    m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128,m128,_MM_SHUFFLE(2,3,0,1)));
    m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128,m128,_MM_SHUFFLE(1,0,3,2)));
    float result = _mm_cvtss_f32(m128);
    for (; i < n; i++) if (x[i] > result) result = x[i];
    return result;
}

/* ================================================================
 * УРОВЕНЬ 2: SSE2 (Pentium 4+, Athlon 64+, Core Duo+, Mac Mini 2006+)
 * ================================================================ */

#elif defined(USE_SSE)

// SSE2: горизонтальное суммирование 4-элементного вектора
static inline float hsum_sse(__m128 v) {
    // [a, b, c, d] -> [a+c, b+d, a+c, b+d]
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    // [a+c, b+d] -> [a+b+c+d, ...]
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// SSE2: быстрая экспонента (4 значения за раз)
static inline __m128 fast_exp_sse(__m128 x) {
    x = _mm_max_ps(x, _mm_set1_ps(-88.0f));
    x = _mm_min_ps(x, _mm_set1_ps( 88.0f));

    __m128 log2e = _mm_set1_ps(1.44269504089f);
    __m128 t     = _mm_mul_ps(x, log2e);
    // floor(t) через приведение к int и обратно
    __m128i ni = _mm_cvttps_epi32(t);
    __m128 n   = _mm_cvtepi32_ps(ni);
    // Коррекция для отрицательных
    __m128 mask = _mm_cmplt_ps(t, n);
    __m128 corr = _mm_and_ps(mask, _mm_set1_ps(1.0f));
    n = _mm_sub_ps(n, corr);

    __m128 f = _mm_sub_ps(t, n);

    __m128 c0 = _mm_set1_ps(1.0f);
    __m128 c1 = _mm_set1_ps(0.6931472f);
    __m128 c2 = _mm_set1_ps(0.2402265f);
    __m128 c3 = _mm_set1_ps(0.0554960f);
    __m128 c4 = _mm_set1_ps(0.0096760f);

    __m128 p = _mm_add_ps(c3, _mm_mul_ps(c4, f));
    p = _mm_add_ps(c2, _mm_mul_ps(p, f));
    p = _mm_add_ps(c1, _mm_mul_ps(p, f));
    p = _mm_add_ps(c0, _mm_mul_ps(p, f));

    // n + 127 в экспоненту float
    __m128i ni2 = _mm_cvttps_epi32(n);
    ni2 = _mm_add_epi32(ni2, _mm_set1_epi32(127));
    ni2 = _mm_slli_epi32(ni2, 23);
    __m128 pow2n = _mm_castsi128_ps(ni2);

    return _mm_mul_ps(p, pow2n);
}

// SSE2: конвертация 4×F16 → 4×F32
static inline __m128 cvt_f16_to_f32x4(uint64_t h4) {
    // h4 содержит 4 значения f16 в нижних 64 битах
    __m128i h = _mm_set1_epi64x((long long)h4);

    // Расширяем до 32 бит
    __m128i lo16 = _mm_unpacklo_epi16(h, _mm_setzero_si128());
    __m128i hi16 = _mm_unpackhi_epi16(h, _mm_setzero_si128());
    // Берем только младшие 4 (остальные нули)
    __m128i vals = lo16;

    __m128i m_sign = _mm_set1_epi32(0x8000);
    __m128i m_expo = _mm_set1_epi32(0x1F);
    __m128i m_mant = _mm_set1_epi32(0x3FF);
    __m128i bias   = _mm_set1_epi32(112);

    __m128i s = _mm_slli_epi32(_mm_and_si128(vals, m_sign), 16);
    __m128i e = _mm_and_si128(_mm_srli_epi32(vals, 10), m_expo);
    __m128i f_mant = _mm_and_si128(vals, m_mant);

    __m128i b = _mm_or_si128(s,
                 _mm_or_si128(_mm_slli_epi32(_mm_add_epi32(e, bias), 23),
                              _mm_slli_epi32(f_mant, 13)));

    // Обнуляем если экспонента 0
    __m128i z = _mm_cmpeq_epi32(e, _mm_setzero_si128());
    b = _mm_or_si128(_mm_andnot_si128(z, b), _mm_and_si128(z, s));

    return _mm_castsi128_ps(b);
}

/* ================================================================
 * SSE2: DOT PRODUCTS
 * ================================================================ */

float dot_f32(const float *a, const float *b, int n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        PREFETCH_READ(a + i + 32); PREFETCH_READ(b + i + 32);
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_load_ps(a+i),    _mm_load_ps(b+i)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_load_ps(a+i+4),  _mm_load_ps(b+i+4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_load_ps(a+i+8),  _mm_load_ps(b+i+8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_load_ps(a+i+12), _mm_load_ps(b+i+12)));
    }
    for (; i + 4 <= n; i += 4)
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_load_ps(a+i), _mm_load_ps(b+i)));
    s0 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
    float r = hsum_sse(s0);
    for (; i < n; i++) r += a[i]*b[i];
    return r;
}

float dot_f32u(const float *a, const float *b, int n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        PREFETCH_READ(a + i + 32); PREFETCH_READ(b + i + 32);
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a+i),    _mm_loadu_ps(b+i)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(a+i+4),  _mm_loadu_ps(b+i+4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_loadu_ps(a+i+8),  _mm_loadu_ps(b+i+8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_loadu_ps(a+i+12), _mm_loadu_ps(b+i+12)));
    }
    for (; i + 4 <= n; i += 4)
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i)));
    s0 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
    float r = hsum_sse(s0);
    for (; i < n; i++) r += a[i]*b[i];
    return r;
}

float dot_f16_f32(const uint16_t *f16, const float *f32, int n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        PREFETCH_READ(f16 + i + 16);
        // Конвертируем первые 4 f16
        uint64_t h4_0;
        memcpy(&h4_0, f16 + i, 8);
        __m128 a0 = cvt_f16_to_f32x4(h4_0);
        __m128 b0 = _mm_loadu_ps(f32 + i);
        s0 = _mm_add_ps(s0, _mm_mul_ps(a0, b0));

        // Конвертируем вторые 4 f16
        uint64_t h4_1;
        memcpy(&h4_1, f16 + i + 4, 8);
        __m128 a1 = cvt_f16_to_f32x4(h4_1);
        __m128 b1 = _mm_loadu_ps(f32 + i + 4);
        s1 = _mm_add_ps(s1, _mm_mul_ps(a1, b1));
    }
    for (; i + 4 <= n; i += 4) {
        uint64_t h4;
        memcpy(&h4, f16 + i, 8);
        __m128 a = cvt_f16_to_f32x4(h4);
        s0 = _mm_add_ps(s0, _mm_mul_ps(a, _mm_loadu_ps(f32 + i)));
    }
    s0 = _mm_add_ps(s0, s1);
    float result = hsum_sse(s0);
    for (; i < n; i++) result += f16_to_f32_scalar(f16[i]) * f32[i];
    return result;
}

float dot_q8_f32(const int8_t *q8, float scale, const float *f32, int n) {
    __m128 vs0 = _mm_setzero_ps(), vs1 = _mm_setzero_ps();
    int i = 0;

    // SSE2: конвертируем 4×int8 → 4×float
    for (; i + 8 <= n; i += 8) {
        PREFETCH_READ(q8 + i + 16);

        // Загружаем 8 int8, расширяем до 16-bit
        __m128i q8v = _mm_loadl_epi64((const __m128i*)(q8 + i));
        // Расширяем со знаком: младшие 4 → int16
        __m128i i16_lo = _mm_srai_epi16(_mm_unpacklo_epi8(q8v, q8v), 8);
        __m128i i16_hi = _mm_srai_epi16(_mm_unpackhi_epi8(q8v, q8v), 8);

        // Младшие 4 int16 → int32
        __m128i i32_0 = _mm_srai_epi32(_mm_unpacklo_epi16(i16_lo, i16_lo), 16);
        __m128i i32_1 = _mm_srai_epi32(_mm_unpackhi_epi16(i16_lo, i16_lo), 16);
        __m128 f0 = _mm_cvtepi32_ps(i32_0);
        __m128 f1 = _mm_cvtepi32_ps(i32_1);

        vs0 = _mm_add_ps(vs0, _mm_mul_ps(f0, _mm_loadu_ps(f32 + i)));
        vs1 = _mm_add_ps(vs1, _mm_mul_ps(f1, _mm_loadu_ps(f32 + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        __m128i q8v = _mm_loadl_epi64((const __m128i*)(q8 + i));
        __m128i i16_lo = _mm_srai_epi16(_mm_unpacklo_epi8(q8v, q8v), 8);
        __m128i i32_0 = _mm_srai_epi32(_mm_unpacklo_epi16(i16_lo, i16_lo), 16);
        __m128 f0 = _mm_cvtepi32_ps(i32_0);
        vs0 = _mm_add_ps(vs0, _mm_mul_ps(f0, _mm_loadu_ps(f32 + i)));
    }
    vs0 = _mm_add_ps(vs0, vs1);
    float sum = hsum_sse(vs0);
    for (; i < n; i++) sum += (float)q8[i] * f32[i];
    return sum * scale;
}

float sum_squares(const float *x, int n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128 v0 = _mm_load_ps(x + i);
        __m128 v1 = _mm_load_ps(x + i + 4);
        s0 = _mm_add_ps(s0, _mm_mul_ps(v0, v0));
        s1 = _mm_add_ps(s1, _mm_mul_ps(v1, v1));
    }
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_load_ps(x + i);
        s0 = _mm_add_ps(s0, _mm_mul_ps(v, v));
    }
    s0 = _mm_add_ps(s0, s1);
    float r = hsum_sse(s0);
    for (; i < n; i++) r += x[i]*x[i];
    return r;
}

void vec_scale_mul(float *out, const float *x, const float *w, float s, int n) {
    __m128 vs = _mm_set1_ps(s);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        PREFETCH_WRITE(out + i + 16);
        __m128 x0 = _mm_load_ps(x + i);
        __m128 w0 = _mm_load_ps(w + i);
        __m128 x1 = _mm_load_ps(x + i + 4);
        __m128 w1 = _mm_load_ps(w + i + 4);
        _mm_store_ps(out + i,     _mm_mul_ps(_mm_mul_ps(x0, vs), w0));
        _mm_store_ps(out + i + 4, _mm_mul_ps(_mm_mul_ps(x1, vs), w1));
    }
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_load_ps(x + i);
        __m128 vw = _mm_load_ps(w + i);
        _mm_store_ps(out + i, _mm_mul_ps(_mm_mul_ps(vx, vs), vw));
    }
    for (; i < n; i++) out[i] = x[i] * s * w[i];
}

void vec_add(float *out, const float *a, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm_store_ps(out+i,     _mm_add_ps(_mm_load_ps(out+i),   _mm_load_ps(a+i)));
        _mm_store_ps(out+i+4,   _mm_add_ps(_mm_load_ps(out+i+4), _mm_load_ps(a+i+4)));
    }
    for (; i + 4 <= n; i += 4)
        _mm_store_ps(out+i, _mm_add_ps(_mm_load_ps(out+i), _mm_load_ps(a+i)));
    for (; i < n; i++) out[i] += a[i];
}

void vec_scale(float *x, float s, int n) {
    __m128 vs = _mm_set1_ps(s);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_store_ps(x+i, _mm_mul_ps(_mm_load_ps(x+i), vs));
    for (; i < n; i++) x[i] *= s;
}

void vec_fma(float *out, const float *v, float w, int n) {
    __m128 vw = _mm_set1_ps(w);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        PREFETCH_READ(v + i + 16);
        __m128 o0 = _mm_load_ps(out + i);
        __m128 o1 = _mm_load_ps(out + i + 4);
        _mm_store_ps(out+i,   _mm_add_ps(o0, _mm_mul_ps(_mm_load_ps(v+i), vw)));
        _mm_store_ps(out+i+4, _mm_add_ps(o1, _mm_mul_ps(_mm_load_ps(v+i+4), vw)));
    }
    for (; i + 4 <= n; i += 4) {
        __m128 o = _mm_load_ps(out + i);
        _mm_store_ps(out+i, _mm_add_ps(o, _mm_mul_ps(_mm_load_ps(v+i), vw)));
    }
    for (; i < n; i++) out[i] += w * v[i];
}

float vec_max(const float *x, int n) {
    if (n == 0) return -INFINITY;
    __m128 vm = _mm_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vm = _mm_max_ps(vm, _mm_load_ps(x + i));
    // Горизонтальный максимум
    __m128 shuf = _mm_shuffle_ps(vm, vm, _MM_SHUFFLE(2, 3, 0, 1));
    vm = _mm_max_ps(vm, shuf);
    shuf = _mm_shuffle_ps(vm, vm, _MM_SHUFFLE(1, 0, 3, 2));
    vm = _mm_max_ps(vm, shuf);
    float result = _mm_cvtss_f32(vm);
    for (; i < n; i++) if (x[i] > result) result = x[i];
    return result;
}

/* ================================================================
 * УРОВЕНЬ 3: SCALAR FALLBACK
 * ================================================================ */

#else

float dot_f32(const float *a, const float *b, int n) {
    float s = 0; for (int i=0;i<n;i++) s+=a[i]*b[i]; return s;
}
float dot_f32u(const float *a, const float *b, int n) {
    return dot_f32(a,b,n);
}
float dot_f16_f32(const uint16_t *f16, const float *f32, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) result += f16_to_f32_scalar(f16[i]) * f32[i];
    return result;
}
float dot_q8_f32(const int8_t *q8, float scale, const float *f32, int n) {
    float s = 0;
    for (int i=0;i<n;i++) s += (float)q8[i]*f32[i];
    return s*scale;
}
float sum_squares(const float *x, int n) {
    float s=0; for(int i=0;i<n;i++) s+=x[i]*x[i]; return s;
}
void vec_scale_mul(float *o, const float *x, const float *w, float s, int n) {
    for(int i=0;i<n;i++) o[i]=x[i]*s*w[i];
}
void vec_add(float *o, const float *a, int n) {
    for(int i=0;i<n;i++) o[i]+=a[i];
}
void vec_scale(float *x, float s, int n) {
    for(int i=0;i<n;i++) x[i]*=s;
}
void vec_fma(float *o, const float *v, float w, int n) {
    for(int i=0;i<n;i++) o[i]+=w*v[i];
}
float vec_max(const float *x, int n) {
    float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m) m=x[i]; return m;
}

#endif /* USE_AVX / USE_SSE */

} // namespace simd
} // namespace xai