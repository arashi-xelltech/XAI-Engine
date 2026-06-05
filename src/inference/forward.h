#pragma once
#include "core/types.h"
#include "core/kv_cache.h"

namespace xai {

struct Model;
struct RunState;

struct RunState {
    float *x      = nullptr;
    float *xb     = nullptr;
    float *xb2    = nullptr;
    float *q      = nullptr;
    float *k      = nullptr;
    float *v      = nullptr;
    float *att    = nullptr;
    float *hb     = nullptr;
    float *hb2    = nullptr;
    float *logits = nullptr;

    KVCache kv_cache;
    int     pos = 0;
};

void alloc_state(RunState *s, const ModelConfig *c);
void free_state(RunState *s);
void reset_state(RunState *s, const ModelConfig *c);

void forward_transformer(Model *m, RunState *s, int token, int pos);
void forward(Model *m, RunState *s, int token, int pos);
void forward_batch(Model *m, RunState *s, const int *tokens, int start_pos, int batch_len);

} // namespace xai