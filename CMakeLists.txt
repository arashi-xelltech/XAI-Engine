cmake_minimum_required(VERSION 3.16)
project(xai LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ================================================================
# Определение реального SIMD уровня процессора (не компилятора)
# ================================================================

set(HOST_AVX   OFF)
set(HOST_SSE42 OFF)
set(HOST_SSE2  OFF)
set(HOST_F16C  OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Читаем флаги процессора из /proc/cpuinfo
    execute_process(
        COMMAND grep -m1 flags /proc/cpuinfo
        OUTPUT_VARIABLE CPU_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CPU_FLAGS MATCHES "avx")
        set(HOST_AVX ON)
    endif()
    if(CPU_FLAGS MATCHES "sse4_2")
        set(HOST_SSE42 ON)
    endif()
    if(CPU_FLAGS MATCHES "sse2")
        set(HOST_SSE2 ON)
    endif()
    if(CPU_FLAGS MATCHES "f16c")
        set(HOST_F16C ON)
    endif()
else()
    # На других системах — проверяем компилятор
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx" HOST_AVX)
    check_cxx_compiler_flag("-msse4.2" HOST_SSE42)
    check_cxx_compiler_flag("-msse2" HOST_SSE2)
    check_cxx_compiler_flag("-mf16c" HOST_F16C)
endif()

# ================================================================
# Установка флагов в зависимости от реального железа
# ================================================================

set(COMMON_FLAGS "-O3 -ffast-math -funroll-loops -pthread -march=native")

if(HOST_AVX)
    set(COMMON_FLAGS "${COMMON_FLAGS} -mavx -msse4.2")
    if(HOST_F16C)
        set(COMMON_FLAGS "${COMMON_FLAGS} -mf16c")
    endif()
    add_definitions(-DUSE_AVX)
    if(HOST_F16C)
        add_definitions(-DUSE_F16C)
    endif()
    message(STATUS "SIMD level: AVX (+F16C: ${HOST_F16C})")
elseif(HOST_SSE42)
    set(COMMON_FLAGS "${COMMON_FLAGS} -msse4.2")
    add_definitions(-DUSE_SSE)
    message(STATUS "SIMD level: SSE4.2")
elseif(HOST_SSE2)
    set(COMMON_FLAGS "${COMMON_FLAGS} -msse2")
    add_definitions(-DUSE_SSE)
    message(STATUS "SIMD level: SSE2")
else()
    message(STATUS "SIMD level: SCALAR (no SIMD available)")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(COMMON_FLAGS "${COMMON_FLAGS} -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

# ================================================================
# Исходники
# ================================================================

set(SOURCES
    src/core/aligned_alloc.cpp
    src/core/simd_ops.cpp
    src/core/vector_ops.cpp
    src/core/thread_pool.cpp
    src/core/kv_cache.cpp
    src/model/model_loader.cpp
    src/model/tensor_ops.cpp
    src/model/rope.cpp
    src/model/tokenizer.cpp
    src/inference/forward.cpp
    src/inference/sampler.cpp
    src/inference/generator.cpp
    src/main.cpp
)

add_executable(xai ${SOURCES})
target_include_directories(xai PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(xai m pthread)