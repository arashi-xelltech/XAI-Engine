#pragma once
#include "simd_detect.h"
#include <cstdint>
#include <cmath>

namespace xai {
namespace simd {

#ifdef USE_AVX
float hsum_avx(__m256 v);
__m256 fast_exp_avx(__m256 x);
__m256 fast_silu_avx(__m256 x);
__m256 soft_cvtph_ps(__m128i h8);
__m256 cvt_f16_to_f32x8(__m128i h8);
#endif

// DOT продукты
float dot_f32(const float *a, const float *b, int n);
float dot_f32u(const float *a, const float *b, int n);
float dot_f16_f32(const uint16_t *f16, const float *f32, int n);
float dot_q8_f32(const int8_t *q8, float scale, const float *f32, int n);

// Векторные операции
float sum_squares(const float *x, int n);
void vec_scale_mul(float *out, const float *x, const float *w, float s, int n);
void vec_add(float *out, const float *a, int n);
void vec_scale(float *x, float s, int n);
void vec_fma(float *out, const float *v, float w, int n);
float vec_max(const float *x, int n);

} // namespace simd
} // namespace xai