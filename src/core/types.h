#pragma once
#include "config.h"
#include <cstdint>
#include <vector>
#include <string>

namespace xai {

enum DType : int {
    DTYPE_F32  = 0,
    DTYPE_F16  = 1,
    DTYPE_Q8   = 2,
    DTYPE_TIED = 3
};

enum WeightFmt { WF_F32, WF_F16, WF_Q8 };

struct WeightTensor {
    const void *data  = nullptr;
    WeightFmt   fmt   = WF_F32;
    float       scale = 1.0f;
    int         rows  = 0;
    int         cols  = 0;
};

struct TensorMeta {
    char    name[MAX_NAME_LEN];
    DType   dtype;
    int     ndim;
    int     shape[4];
    int64_t offset;
    int64_t size;
    float   scale;
};

struct ModelConfig {
    int   vocab_size        = 0;
    int   hidden_size       = 0;
    int   intermediate_size = 0;
    int   num_layers        = 0;
    int   num_heads         = 0;
    int   num_kv_heads      = 0;
    int   max_seq_len       = 0;
    float rms_norm_eps      = 1e-5f;
    float rope_theta        = 10000.0f;
    int   head_dim          = 0;
    int   kv_dim            = 0;
    int   gqa_groups        = 1;
};

struct Tokenizer {
    int vocab_size = 0;
    int bos_id = 0, eos_id = 0, pad_id = -1, unk_id = 0;
    std::vector<std::string> pieces;
    std::vector<float>       scores;
    std::vector<int>         types;

    std::vector<int> hash_keys;
    std::vector<int> hash_ids;
    int hash_cap = 0;
};

struct LayerWeights {
    float *attn_norm  = nullptr;
    float *ffn_norm   = nullptr;

    WeightTensor q_proj;
    WeightTensor k_proj;
    WeightTensor v_proj;
    WeightTensor o_proj;
    WeightTensor gate_proj;
    WeightTensor up_proj;
    WeightTensor down_proj;
};

} // namespace xai