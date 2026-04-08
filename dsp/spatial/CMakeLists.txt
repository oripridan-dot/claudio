# dsp/spatial/CMakeLists.txt — Claudio Holographic Binaural Rendering Node
cmake_minimum_required(VERSION 3.18)

add_library(claudio_spatial STATIC
    HRTFConvolutionEngine.cpp
    AcousticRaytracer.cpp
    HolographicBinauralNode.cpp
)

target_include_directories(claudio_spatial PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(claudio_spatial PUBLIC cxx_std_17)

target_compile_options(claudio_spatial PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math -Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4>
)

# ── Mutex-free enforcement (design-time reminder) ──────────────────────────────
# The SpatialLatencyGate adversary validator audits source files at test time.
# std::mutex / std::lock_guard in the audio path = CI failure.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(claudio_spatial PRIVATE CLAUDIO_MUTEX_AUDIT=1)
endif()

# ── Optional SIMD acceleration ─────────────────────────────────────────────────
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
if(HAVE_AVX2)
    target_compile_options(claudio_spatial PRIVATE -mavx2)
    message(STATUS "claudio_spatial: AVX2 SIMD enabled")
endif()

# ── Tests ──────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_subdirectory(../../tests/adversary tests_adversary
                     EXCLUDE_FROM_ALL)
endif()
