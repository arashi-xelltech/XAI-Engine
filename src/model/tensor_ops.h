#pragma once
#include "core/types.h"

namespace xai {

struct Model;
class BarrierPool;

// Weight matrix-vector operations
float wt_dot_row(const WeightTensor *w, int row, const float *vec);
void wt_matmul_rows(float *out, const WeightTensor *w, const float *vec, int r0, int r1);
void wt_matmul(float *out, const WeightTensor *w, const float *vec);
void wt_matmul_dual(float *out1, float *out2,
                    const WeightTensor *w1, const WeightTensor *w2,
                    const float *vec);
void wt_matmul_qkv(float *oq, float *ok, float *ov,
                   const WeightTensor *wq, const WeightTensor *wk,
                   const WeightTensor *wv, const float *vec);
int wt_matmul_argmax(const WeightTensor *w, const float *vec);
void wt_matmul_logits(float *logits, const WeightTensor *w, const float *vec);

// Weight loading helpers
TensorMeta *find_tensor(Model *m, const char *name);
float *load_1d_f32(Model *m, const char *name);
WeightTensor load_weight_native(Model *m, const char *name);
float *load_embedding(Model *m, const char *name);

} // namespace xai