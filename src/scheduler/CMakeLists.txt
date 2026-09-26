# Sonder inference scheduler module (Phase 2 scheduling policy).
#
# Pure policy: request queue, continuous-batching step planner, chunked
# prefill, KV admission control (through the abstract KvCapacity interface),
# preemption decisions, fairness guards, deterministic simulation and
# telemetry. No backend execution and no dependency on src/cache/.
#
# Integrated build (root CMakeLists auto-adds this directory): sources are
# compiled into sonder_inference via the SonderModules.cmake helpers and tests
# register as sonder.scheduler.<case>.
#
# Standalone build (self-test without the rest of the tree):
#   cmake -S src/scheduler -B build/scheduler -G Ninja && cmake --build build/scheduler -j 4
#   ctest --test-dir build/scheduler
cmake_minimum_required(VERSION 3.21)

set(SONDER_SCHEDULER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_capacity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stats.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulator.cpp)
set(SONDER_SCHEDULER_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/scheduler_unit_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/scheduler_sim_tests.cpp)

if(TARGET sonder_inference AND COMMAND sonder_module_sources)
    # ---- Integrated into the Sonder Inference root project ----
    sonder_module_sources(${SONDER_SCHEDULER_SOURCES})
    sonder_module_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    sonder_module_define(SONDER_HAS_SCHEDULER)
    # Convenience alias so consumers can link "the scheduler" explicitly.
    if(NOT TARGET sonder::scheduler)
        add_library(sonder::scheduler ALIAS sonder_inference)
    endif()
    sonder_add_module_tests(scheduler ${SONDER_SCHEDULER_TEST_SOURCES})
    return()
endif()

# ---- Standalone ----
project(SonderScheduler LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(SONDER_SCHEDULER_BUILD_TESTS "Build scheduler tests (fetches doctest)" ON)
option(SONDER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

function(sonder_scheduler_warnings target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus /EHsc)
        if(SONDER_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(SONDER_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(sonder_scheduler ${SONDER_SCHEDULER_SOURCES})
add_library(sonder::scheduler ALIAS sonder_scheduler)
target_include_directories(sonder_scheduler PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_compile_features(sonder_scheduler PUBLIC cxx_std_20)
sonder_scheduler_warnings(sonder_scheduler)

if(SONDER_SCHEDULER_BUILD_TESTS)
    enable_testing()
    # Same pinned doctest as the root project (cmake/SonderDoctest.cmake).
    include(FetchContent)
    FetchContent_Declare(doctest
        URL "https://github.com/doctest/doctest/archive/refs/tags/v2.5.3.tar.gz"
        URL_HASH "SHA256=174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR "sonder-no-cmake")
    FetchContent_MakeAvailable(doctest)
    include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

    add_executable(sonder_scheduler_tests
        ${SONDER_SCHEDULER_TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/doctest_main_standalone.cpp)
    target_include_directories(sonder_scheduler_tests SYSTEM PRIVATE "${doctest_SOURCE_DIR}")
    target_link_libraries(sonder_scheduler_tests PRIVATE sonder::scheduler)
    sonder_scheduler_warnings(sonder_scheduler_tests)
    doctest_discover_tests(sonder_scheduler_tests TEST_PREFIX "sonder.scheduler."
                           PROPERTIES TIMEOUT 120)
endif()
