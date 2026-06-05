#pragma once
#include "core/types.h"

namespace xai {

struct Model {
    ModelConfig cfg;
    Tokenizer   tok;

    float        *embed      = nullptr;
    LayerWeights *layers     = nullptr;
    float        *final_norm = nullptr;

    WeightTensor  lm_head;
    bool          lm_head_tied = false;

    float *rope_cos = nullptr;
    float *rope_sin = nullptr;

    char  *file_data             = nullptr;
    size_t file_size             = 0;
    int    fd                    = -1;

    TensorMeta tensors[MAX_TENSORS];
    int        num_tensors           = 0;
    char      *tensor_data_start     = nullptr;
    int64_t    tokenizer_blob_size   = 0;
};

int load_model(const char *path, Model *m, bool quiet);
void free_model(Model *m);

} // namespace xai