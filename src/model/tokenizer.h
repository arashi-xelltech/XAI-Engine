#pragma once
#include "core/types.h"

namespace xai {

void tok_deserialize(const char *data, size_t max_size, Tokenizer *t);
int tok_encode(const Tokenizer *t, const char *text, int *ids, int max_ids, int add_bos);
void tok_decode_one(const Tokenizer *t, int id, char *buf, int buf_sz);

} // namespace xai