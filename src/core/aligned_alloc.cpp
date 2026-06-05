#include "core/aligned_alloc.h"
#include "core/config.h"
#include "config.h"
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>

namespace xai {

void *portable_aligned_alloc(size_t alignment, size_t size) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "posix_memalign failed for size=%zu, align=%zu: %s\n",
                size, alignment, strerror(errno));
        return nullptr;
    }
    return ptr;
}

float *aligned_calloc_f32(size_t count) {
    size_t bytes = count * sizeof(float);
    // Округление до CACHE_LINE
    bytes = (bytes + CACHE_LINE - 1) & ~(size_t)(CACHE_LINE - 1);
    float *p = (float*)portable_aligned_alloc(SIMD_ALIGN, bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

void aligned_free(void *ptr) {
    free(ptr);
}

} // namespace xai