#pragma once
#include "core/types.h"

namespace xai {

// High-level ops that may use thread pool internally
void vec_copy(float *dst, const float *src, int n);

// RMSNorm
void rmsnorm(float *out, const float *x, const float *w, int size, float eps);
void fused_residual_rmsnorm(float *x, const float *residual,
                            float *out, const float *w, int size, float eps);

// Softmax
void softmax(float *x, int size);
void softmax_exact(float *x, int size);

// SiLU
void silu_mul(float *gate, const float *up, int size);

// F16/Q8 conversion
float *convert_f16_to_f32(const uint16_t *src, int64_t n);
float *convert_q8_to_f32(const int8_t *src, float scale, int64_t n);

} // namespace xai