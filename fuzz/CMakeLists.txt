# Sonder Inference fuzz targets (docs/integration/hardening.md, fuzz/README.md).
#
# Two ways to build:
#
#  1. Standalone (what .github/workflows/hardening.yml does; no root edits):
#       CC=clang CXX=clang++ cmake -S fuzz -B build/fuzz -G Ninja
#       cmake --build build/fuzz
#     This file then becomes the top-level project, pulls in the repository
#     root as a subdirectory, and instruments sonder_inference for fuzzing.
#
#  2. In-tree, once the root CMakeLists.txt has `add_subdirectory(fuzz)`:
#       cmake -S . -B build/fuzz -DSONDER_BUILD_FUZZERS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
#     Nothing is built unless SONDER_BUILD_FUZZERS is ON.
#
# SONDER_FUZZ_ENGINE:
#   libfuzzer   - clang -fsanitize=fuzzer (default with Clang)
#   standalone  - plain executables driven by standalone_main.cpp that replay
#                 the seed corpora once (default with MSVC/GCC; regression only)
# SONDER_FUZZ_SANITIZERS: comma list passed to -fsanitize= (Clang only),
#   default "address,undefined". Empty disables.
cmake_minimum_required(VERSION 3.21)

set(_sonder_fuzz_standalone OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_sonder_fuzz_standalone ON)
    project(SonderInferenceFuzz LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
    endif()
endif()

option(SONDER_BUILD_FUZZERS "Build the libFuzzer targets in fuzz/" ${_sonder_fuzz_standalone})
if(NOT SONDER_BUILD_FUZZERS)
    return()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
    set(_sonder_fuzz_default_engine libfuzzer)
else()
    set(_sonder_fuzz_default_engine standalone)
endif()
set(SONDER_FUZZ_ENGINE "${_sonder_fuzz_default_engine}" CACHE STRING "libfuzzer or standalone")
set_property(CACHE SONDER_FUZZ_ENGINE PROPERTY STRINGS libfuzzer standalone)
set(SONDER_FUZZ_SANITIZERS "address,undefined" CACHE STRING "Sanitizers for fuzz builds (Clang)")
set(SONDER_FUZZ_SECONDS "60" CACHE STRING "Default -max_total_time for the fuzz-run-* targets")

if(SONDER_FUZZ_ENGINE STREQUAL "libfuzzer" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "SONDER_FUZZ_ENGINE=libfuzzer requires Clang (got ${CMAKE_CXX_COMPILER_ID})")
endif()

set(_sonder_fuzz_cov_flags "")
set(_sonder_fuzz_san_flags "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
    if(SONDER_FUZZ_ENGINE STREQUAL "libfuzzer")
        list(APPEND _sonder_fuzz_cov_flags -fsanitize=fuzzer-no-link)
    endif()
    if(SONDER_FUZZ_SANITIZERS)
        list(APPEND _sonder_fuzz_san_flags -fsanitize=${SONDER_FUZZ_SANITIZERS} -fno-sanitize-recover=all)
    endif()
    list(APPEND _sonder_fuzz_san_flags -fno-omit-frame-pointer -g)
endif()

if(_sonder_fuzz_standalone)
    # Instrument the whole library build (it is compiled from this project).
    list(JOIN _sonder_fuzz_cov_flags " " _cov)
    list(JOIN _sonder_fuzz_san_flags " " _san)
    string(APPEND CMAKE_C_FLAGS " ${_cov} ${_san}")
    string(APPEND CMAKE_CXX_FLAGS " ${_cov} ${_san}")
    set(SONDER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SONDER_BUILD_CLI OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/sonder-inference")
elseif(TARGET sonder_inference)
    # In-tree: instrument the already-declared library for coverage and make
    # every consumer link the sanitizer runtime. Only happens when
    # SONDER_BUILD_FUZZERS=ON, so normal builds are untouched.
    target_compile_options(sonder_inference PRIVATE ${_sonder_fuzz_cov_flags} ${_sonder_fuzz_san_flags})
    if(_sonder_fuzz_san_flags)
        target_link_options(sonder_inference INTERFACE ${_sonder_fuzz_san_flags})
    endif()
else()
    message(FATAL_ERROR "fuzz/: add_subdirectory(fuzz) must come after sonder_inference and its modules")
endif()

if(NOT TARGET sonder_inference)
    message(FATAL_ERROR "fuzz/: sonder_inference target not found")
endif()

set(_sonder_fuzz_targets
    json
    ollama_stream
    bench_corpus
    bench_report
    telemetry_envelope
    http_request
    gguf)

foreach(name IN LISTS _sonder_fuzz_targets)
    set(target sonder_fuzz_${name})
    set(corpus_dir "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${name}")
    if(SONDER_FUZZ_ENGINE STREQUAL "libfuzzer")
        add_executable(${target} fuzz_${name}.cpp)
        target_compile_options(${target} PRIVATE ${_sonder_fuzz_san_flags})
        target_link_options(${target} PRIVATE -fsanitize=fuzzer ${_sonder_fuzz_san_flags})
    else()
        add_executable(${target} fuzz_${name}.cpp standalone_main.cpp)
        target_compile_options(${target} PRIVATE ${_sonder_fuzz_san_flags})
        target_link_options(${target} PRIVATE ${_sonder_fuzz_san_flags})
    endif()
    target_link_libraries(${target} PRIVATE sonder::inference)
    # src/ for internal headers (e.g. backends/ollama/ollama_protocol.hpp).
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(COMMAND sonder_set_warnings)
        sonder_set_warnings(${target})
    endif()

    # Deterministic corpus replay (runs each seed once).
    if(NOT TARGET fuzz-regress)
        add_custom_target(fuzz-regress)
    endif()
    if(SONDER_FUZZ_ENGINE STREQUAL "libfuzzer")
        add_custom_target(fuzz-regress-${name}
            COMMAND $<TARGET_FILE:${target}> -runs=0 "${corpus_dir}"
            DEPENDS ${target} VERBATIM)
        # Time-boxed exploration into a scratch corpus seeded from the repo.
        add_custom_target(fuzz-run-${name}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/work/${name}"
            COMMAND $<TARGET_FILE:${target}> -max_total_time=${SONDER_FUZZ_SECONDS} -timeout=10
                    -rss_limit_mb=2048 -max_len=65536
                    "-artifact_prefix=${CMAKE_CURRENT_BINARY_DIR}/work/crash-${name}-"
                    "${CMAKE_CURRENT_BINARY_DIR}/work/${name}" "${corpus_dir}"
            DEPENDS ${target} VERBATIM USES_TERMINAL)
    else()
        add_custom_target(fuzz-regress-${name}
            COMMAND $<TARGET_FILE:${target}> "${corpus_dir}"
            DEPENDS ${target} VERBATIM)
    endif()
    add_dependencies(fuzz-regress fuzz-regress-${name})
endforeach()
