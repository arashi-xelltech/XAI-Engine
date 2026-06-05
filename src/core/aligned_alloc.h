#pragma once
#include <cstddef>

namespace xai {

void *portable_aligned_alloc(size_t alignment, size_t size);
float *aligned_calloc_f32(size_t count);
void aligned_free(void *ptr);

} // namespace xai