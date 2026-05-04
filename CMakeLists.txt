cmake_minimum_required(VERSION 3.21)
project(parakeet
    VERSION 1.0.0
    DESCRIPTION "Pure-C inference engine for parakeet-unified-en-0.6b"
    HOMEPAGE_URL "https://github.com/eschmidbauer/parakeet-unified-en-0.6b-c"
    LANGUAGES C)

# Standalone builds also produce the parakeet CLI for testing/benchmarking.
# Sub-project consumers (e.g. via FetchContent) get only the static library
# unless they opt in.
option(PARAKEET_BUILD_CLI "Build the parakeet command-line binary" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PARAKEET_LIB_SOURCES
    csrc/parakeet.h
    csrc/blas.h
    csrc/blas.c
    csrc/threadpool.h
    csrc/threadpool.c
    csrc/tensor_ops.c
    csrc/encoder.c
    csrc/decoder.c
    csrc/mel.c
    csrc/weights.c
    csrc/sentencepiece.c)

add_library(parakeet STATIC ${PARAKEET_LIB_SOURCES})
add_library(parakeet::parakeet ALIAS parakeet)

target_include_directories(parakeet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>
        $<INSTALL_INTERFACE:include>)

target_compile_options(parakeet PRIVATE -O3 -ffast-math)

# weights.c performs int4 dequantization that -ffast-math can break by
# reordering 0*scale stores into NaN. Build it with -O3 only (matches the
# Makefile carve-out).
set_source_files_properties(csrc/weights.c PROPERTIES COMPILE_OPTIONS "-O3")

find_package(Threads REQUIRED)
target_link_libraries(parakeet PUBLIC Threads::Threads m)

if(PARAKEET_BUILD_CLI)
    add_executable(parakeet_cli csrc/main.c csrc/wav.c)
    set_target_properties(parakeet_cli PROPERTIES OUTPUT_NAME parakeet)
    target_link_libraries(parakeet_cli PRIVATE parakeet)
endif()
