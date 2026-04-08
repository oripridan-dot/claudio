cmake_minimum_required(VERSION 3.18)
project(Claudio VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── DSP Engine ─────────────────────────────────────────────────────────────────
add_subdirectory(dsp/spatial)

# ── Tests ──────────────────────────────────────────────────────────────────────
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
endif()
