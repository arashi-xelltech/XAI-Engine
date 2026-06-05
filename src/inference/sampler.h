#pragma once
#include <cstdint>
#include <vector>

namespace xai {

struct PI { float v; int i; };

void rng_seed(uint64_t s);
float rng_f32();
void quickselect_topk(PI *arr, int n, int k);
int sample_token(float *logits, int vs, float temp, int top_k, float top_p);
void apply_repetition_penalty(float *logits, int vocab_size,
                              const int *history, int hist_len, float rep_p);

} // namespace xai