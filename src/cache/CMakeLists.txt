# Sonder Inference module: logical KV-cache manager (docs/KV_CACHE.md, Phase 2).
#
# In-tree (root CMakeLists auto-adds src/cache): sources compile into the core
# sonder_inference library, headers are exported from src/cache/include, the
# SONDER_HAS_KV_CACHE=1 feature macro is defined, and doctest cases register
# with CTest as sonder.cache.<case>.
#
# Standalone self-test (no core sources needed; uses the root include/ and
# cmake/SonderDoctest.cmake):
#   cmake -S src/cache -B build/cache -G Ninja && cmake --build build/cache -j 4
#   ctest --test-dir build/cache
cmake_minimum_required(VERSION 3.21)

set(SONDER_CACHE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eviction_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache_manager.cpp)
set(SONDER_CACHE_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_checkpoints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_eviction_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kv_cache_manager.cpp)

if(COMMAND sonder_module_sources)
    sonder_module_sources(${SONDER_CACHE_SOURCES})
    sonder_module_include_directories(include)
    sonder_module_define(SONDER_HAS_KV_CACHE)
    sonder_add_module_tests(cache ${SONDER_CACHE_TEST_SOURCES})
    return()
endif()

# ---- standalone ---------------------------------------------------------------
project(SonderInferenceCache LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(SONDER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${SONDER_ROOT}/cmake")
include(SonderWarnings)
option(SONDER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(sonder_inference_cache STATIC ${SONDER_CACHE_SOURCES})
target_include_directories(sonder_inference_cache PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${SONDER_ROOT}/include")
target_compile_features(sonder_inference_cache PUBLIC cxx_std_20)
target_compile_definitions(sonder_inference_cache PUBLIC SONDER_HAS_KV_CACHE=1)
sonder_set_warnings(sonder_inference_cache)

enable_testing()
include(SonderDoctest)
add_executable(sonder_cache_tests ${SONDER_CACHE_TEST_SOURCES})
target_link_libraries(sonder_cache_tests PRIVATE sonder_inference_cache sonder_doctest_main)
sonder_set_warnings(sonder_cache_tests)
doctest_discover_tests(sonder_cache_tests TEST_PREFIX "sonder.cache." PROPERTIES TIMEOUT 120)
