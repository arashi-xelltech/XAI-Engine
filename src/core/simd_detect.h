#pragma once

#ifdef USE_AVX
    #include <immintrin.h>
#elif defined(USE_SSE)
    // SSE2 - минимальный набор для всех SSE-уровней
    #include <emmintrin.h>   // SSE2
    #ifdef __SSE3__
        #include <pmmintrin.h>   // SSE3
    #endif
    #ifdef __SSSE3__
        #include <tmmintrin.h>   // SSSE3
    #endif
    #ifdef __SSE4_1__
        #include <smmintrin.h>   // SSE4.1
    #endif
    #ifdef __SSE4_2__
        #include <nmmintrin.h>   // SSE4.2
    #endif
#endif

#ifdef __F16C__
    #define USE_F16C 1
#endif

#define PREFETCH_READ(addr)    __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr)   __builtin_prefetch((addr), 1, 3)
#define PREFETCH_READ_L2(addr) __builtin_prefetch((addr), 0, 2)