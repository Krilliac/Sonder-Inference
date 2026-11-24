# Sonder Inference - llama.cpp / GGML backend module (docs/MODULES.md).
#
# The root CMakeLists adds this directory unconditionally once it exists, so
# the module gates itself on SONDER_WITH_LLAMA_CPP (root option, default OFF):
# the default build and CI never download or compile llama.cpp.
#
# When ON it provides:
#   sonder_backend_llamacpp  static wrapper over the llama.cpp C API
#                            (sonder/backends/llamacpp/llamacpp_backend.h)
#   make_llamacpp_backend()  sonder::inference::Backend adapter compiled into
#                            sonder_inference (sonder/inference/backends/llamacpp.hpp)
#   SONDER_HAS_LLAMACPP_BACKEND=1 on sonder_inference
#
# llama.cpp is pulled with FetchContent at a pinned release tag (b11195 by
# default; recorded in docs/LICENSE_REVIEW.md — change both together on default
# upgrades).
#
# Building against a local checkout (offline builds, newer mainline, or forks):
#   -DFETCHCONTENT_SOURCE_DIR_LLAMACPP=/path/to/llama.cpp
# Specifically:
#   - Mainline ggml-org/llama.cpp >= b11556: adds stock Q2_0 (2-bit ternary),
#     Q1_0 (1-bit Bonsai-1), NVFP4 (Blackwell FP4), and F8_E4M3 (FP8).
#   - PrismML-Eng/llama.cpp (prism branch): adds Hadamard-transformed Bonsai-2
#     ternary formats (PQ2_0, PTQ1_0, PQ2_0M) and prism.hadamard.* support.
#
# To override the fetched tarball tag instead of using a local checkout, pass
# SONDER_LLAMACPP_TAG and the matching SONDER_LLAMACPP_SHA256 (and optionally
# SONDER_LLAMACPP_URL) together; SHA-256 verification fails closed on mismatch.
#
# Opt-in GPU presets live in CMakePresets.json: windows-cuda-rtx,
# windows-vulkan-xt, linux-cuda-rtx, linux-rocm-xt, and linux-vulkan-xt.

if(NOT SONDER_WITH_LLAMA_CPP)
    return()
endif()

include(FetchContent)

# Tag and hash move together; bump both (and docs/LICENSE_REVIEW.md) on upgrade.
set(SONDER_LLAMACPP_TAG "b11195" CACHE STRING
    "Upstream llama.cpp tag for FetchContent (default pinned b11195)")
set(SONDER_LLAMACPP_URL
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${SONDER_LLAMACPP_TAG}.tar.gz"
    CACHE STRING "Source archive URL for llama.cpp FetchContent")
# SHA-256 of the GitHub source tarball for tag b11195 (commit d834d44e6436).
set(SONDER_LLAMACPP_SHA256
    "d9818b7786c8a3b063bac7eb7dc8710e275638f1e288e206720c9e22e7ca216a"
    CACHE STRING "Expected SHA-256 of SONDER_LLAMACPP_URL")

# Consume llama.cpp as a library only: no tools, server, examples, tests, or
# networking. CPU backend is always available; accelerators are opt-in by the
# caller through the usual GGML_* cache variables (e.g. -DGGML_CUDA=ON).
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL  OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL  OFF CACHE BOOL "" FORCE)
# Static linkage keeps the test executables self-contained on every platform.
# (Directory-scoped: these do not leak to the rest of the project.) PIC lets
# the static archives be linked into a shared sonder_inference.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

# CMake >= 3.28: only the llama.cpp targets we link are built by default.
set(SONDER_LLAMACPP_FETCH_EXTRA)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
    list(APPEND SONDER_LLAMACPP_FETCH_EXTRA EXCLUDE_FROM_ALL)
endif()
FetchContent_Declare(llamacpp
    URL      "${SONDER_LLAMACPP_URL}"
    URL_HASH "SHA256=${SONDER_LLAMACPP_SHA256}"
    ${SONDER_LLAMACPP_FETCH_EXTRA}
)
FetchContent_MakeAvailable(llamacpp)

# llama.cpp itself is written against C++17. The root project sets
# CMAKE_CXX_STANDARD 20, and under MSVC C++20 char8_t makes u8"" literals in
# llama-chat.cpp incompatible with const char*. Build the upstream llama
# target as C++17; Sonder's own targets stay C++20.
foreach(sonder_llamacpp_upstream_target IN ITEMS llama)
    if(TARGET ${sonder_llamacpp_upstream_target})
        set_target_properties(${sonder_llamacpp_upstream_target} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)
    endif()
endforeach()

add_library(sonder_backend_llamacpp STATIC
    llamacpp_backend.cpp
    llamacpp_util.cpp
)
add_library(sonder::backend_llamacpp ALIAS sonder_backend_llamacpp)
target_include_directories(sonder_backend_llamacpp
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_features(sonder_backend_llamacpp PUBLIC cxx_std_20)
# llama.h stays private: nothing outside this directory sees llama.cpp types.
target_link_libraries(sonder_backend_llamacpp PRIVATE llama ggml)
target_compile_definitions(sonder_backend_llamacpp
    PRIVATE SONDER_LLAMACPP_TAG_STRING="${SONDER_LLAMACPP_TAG}"
)
if(COMMAND sonder_set_warnings)
    sonder_set_warnings(sonder_backend_llamacpp)
endif()

# Core adapter: sonder::inference::Backend implementation compiled into the
# core library, which links the wrapper.
sonder_module_sources(${CMAKE_CURRENT_SOURCE_DIR}/llamacpp_core_backend.cpp)
sonder_module_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
sonder_module_define(SONDER_HAS_LLAMACPP_BACKEND)
target_link_libraries(sonder_inference PRIVATE sonder_backend_llamacpp)
target_include_directories(sonder_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# llama.cpp's bundled vocab-only GGUF fixtures (tokenizer metadata, no
# weights). Unit tests use them to exercise tokenization without downloads.
set(SONDER_LLAMACPP_VOCAB_DIR "${llamacpp_SOURCE_DIR}/models")

if(SONDER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
