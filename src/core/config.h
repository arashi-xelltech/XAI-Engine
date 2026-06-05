#pragma once
#include <cstdint>

namespace xai {

constexpr int MAX_SEQ_LEN    = 8192;
constexpr int MAX_NAME_LEN   = 256;
constexpr int MAX_TENSORS    = 2048;
constexpr int ALIGN_SIZE     = 64;
constexpr int SIMD_ALIGN     = 32;
constexpr int MAX_THREADS    = 64;
constexpr int CACHE_LINE     = 64;

extern int g_num_threads;
extern int g_sliding_window;

} // namespace xai