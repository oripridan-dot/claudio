# tests/adversary/CMakeLists.txt — SpatialLatencyGate Adversary Validator
cmake_minimum_required(VERSION 3.18)

add_executable(SpatialLatencyGate SpatialLatencyGate.cpp)

target_link_libraries(SpatialLatencyGate PRIVATE claudio_spatial)
target_compile_features(SpatialLatencyGate PRIVATE cxx_std_17)

target_compile_options(SpatialLatencyGate PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
)

include(CTest)
add_test(
    NAME    SpatialLatencyGate
    COMMAND SpatialLatencyGate
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# CI enforcement: non-zero exit = acoustic smearing risk = build FAIL.
set_tests_properties(SpatialLatencyGate PROPERTIES
    PASS_REGULAR_EXPRESSION "ALL TESTS PASSED"
    FAIL_REGULAR_EXPRESSION  "FAILED|FAIL \\["
    TIMEOUT 15
)
