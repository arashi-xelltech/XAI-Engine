#pragma once
#include "core/types.h"
#include <cstdint>

namespace xai {

struct KVCache {
    int8_t *key_data    = nullptr;
    float  *key_scales  = nullptr;
    float  *val_data    = nullptr;

    int num_layers   = 0;
    int num_kv_heads = 0;
    int max_seq_len  = 0;
    int head_dim     = 0;

    void alloc(int nl, int nkv, int msl, int hd);
    void reset();
    void free_mem();

    void store_key(int layer, int head, int pos, const float *src, int hd);
    void store_value(int layer, int head, int pos, const float *src, int hd);

    float dot_key(int layer, int head, int pos, const float *query, int hd) const;
    void fma_value(int layer, int head, int pos, float weight, float *out, int hd) const;
};

} // namespace xai