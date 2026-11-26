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
option(SONDER_WITH_MTMD "Build optional llama.cpp libmtmd multimodal projector support" OFF)
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
if(SONDER_WITH_MTMD)
    set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
    set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
elseif(LLAMA_BUILD_MTMD)
    set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
endif()
# Static linkage keeps the test executables self-contained on every platform.
# (Directory-scoped: these do not leak to the rest of the project.) PIC lets
# the static archives be linked into a shared sonder_inference.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

# Optional local llama.cpp source tree override (offline builds, newer mainline,
# or forks such as PrismML-Eng/llama.cpp). Mirrors into FetchContent's standard
# override variable when that variable is not already set.
set(SONDER_LLAMACPP_SOURCE_DIR "" CACHE PATH
    "Optional local llama.cpp source directory (sets FETCHCONTENT_SOURCE_DIR_LLAMACPP if unset)")
if(SONDER_LLAMACPP_SOURCE_DIR AND NOT FETCHCONTENT_SOURCE_DIR_LLAMACPP)
    set(FETCHCONTENT_SOURCE_DIR_LLAMACPP "${SONDER_LLAMACPP_SOURCE_DIR}")
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

# Feature auto-detection against the resolved llama.cpp / ggml headers so both
# pinned mainline b11195 and newer/forked trees (e.g. PrismML-Eng/llama.cpp)
# build cleanly without breaking older APIs.
set(SONDER_LLAMACPP_FEATURE_DEFS "")
set(SONDER_LLAMACPP_PRIVATE_INCLUDES "")
set(_sonder_llama_h "${llamacpp_SOURCE_DIR}/include/llama.h")
set(_sonder_llama_ext_h "${llamacpp_SOURCE_DIR}/src/llama-ext.h")
set(_sonder_ggml_h "${llamacpp_SOURCE_DIR}/ggml/include/ggml.h")

if(EXISTS "${_sonder_llama_h}")
    file(READ "${_sonder_llama_h}" _sonder_llama_h_content)

    string(FIND "${_sonder_llama_h_content}" "kv_unified" _idx_kv_unified)
    if(NOT _idx_kv_unified EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_KV_UNIFIED=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "op_offload" _idx_op_offload)
    if(NOT _idx_op_offload EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_OP_OFFLOAD=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "llama_set_sampler" _idx_set_sampler)
    if(NOT _idx_set_sampler EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_SET_SAMPLER=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "n_outputs_max_per_seq" _idx_n_outputs_max)
    if(NOT _idx_n_outputs_max EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "llama_state_seq_get_data_ext" _idx_state_ext)
    if(NOT _idx_state_ext EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_STATE_SEQ_EXT=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "llama_model_n_embd_out" _idx_n_embd_out)
    if(NOT _idx_n_embd_out EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_N_EMBD_OUT=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "llama_n_ctx_seq" _idx_n_ctx_seq)
    if(NOT _idx_n_ctx_seq EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_N_CTX_SEQ=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "moe_cache_size" _idx_moe_cache_size)
    if(NOT _idx_moe_cache_size EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_MOE_CACHE_SIZE=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "swa_full" _idx_swa_full)
    if(NOT _idx_swa_full EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_SWA_FULL=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "llama_model_n_swa" _idx_model_n_swa)
    if(NOT _idx_model_n_swa EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_MODEL_N_SWA=1)
    endif()

    string(FIND "${_sonder_llama_h_content}" "LLAMA_CONTEXT_TYPE_MTP" _idx_ctx_mtp)
    string(FIND "${_sonder_llama_h_content}" "llama_model_n_layer_nextn" _idx_layer_nextn)
    string(FIND "${_sonder_llama_h_content}" "load_mtp" _idx_load_mtp)
    if(NOT _idx_ctx_mtp EQUAL -1
       AND NOT _idx_layer_nextn EQUAL -1
       AND NOT _idx_load_mtp EQUAL -1
       AND EXISTS "${_sonder_llama_ext_h}")
        file(READ "${_sonder_llama_ext_h}" _sonder_llama_ext_h_content)
        string(FIND "${_sonder_llama_ext_h_content}" "llama_set_embeddings_nextn" _idx_embd_nextn)
        if(NOT _idx_embd_nextn EQUAL -1)
            list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_MTP=1)
            list(APPEND SONDER_LLAMACPP_PRIVATE_INCLUDES "${llamacpp_SOURCE_DIR}/src")

            string(FIND "${_sonder_llama_ext_h_content}" "llama_set_nextn_layer_offset" _idx_nextn_offset)
            if(NOT _idx_nextn_offset EQUAL -1)
                list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET=1)
            endif()

            string(FIND "${_sonder_llama_ext_h_content}" "llama_get_ctx_other" _idx_get_ctx_other)
            string(FIND "${_sonder_llama_h_content}" "ctx_other" _idx_ctx_other)
            if(NOT _idx_get_ctx_other EQUAL -1 AND NOT _idx_ctx_other EQUAL -1)
                list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_CTX_OTHER=1)
            endif()

            string(FIND "${_sonder_llama_h_content}" "n_rs_seq" _idx_n_rs_seq)
            if(NOT _idx_n_rs_seq EQUAL -1)
                list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_N_RS_SEQ=1)
            endif()
        endif()
    endif()
endif()

if(EXISTS "${_sonder_ggml_h}")
    file(READ "${_sonder_ggml_h}" _sonder_ggml_h_content)

    string(FIND "${_sonder_ggml_h_content}" "GGML_TYPE_F8_E4M3" _idx_f8)
    if(NOT _idx_f8 EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_GGML_F8_E4M3=1)
    endif()

    string(FIND "${_sonder_ggml_h_content}" "GGML_TYPE_Q2_0" _idx_q2_0)
    if(NOT _idx_q2_0 EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_GGML_Q2_0=1)
    endif()

    string(FIND "${_sonder_ggml_h_content}" "GGML_TYPE_PQ2_0" _idx_pq2_0)
    if(NOT _idx_pq2_0 EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_GGML_PQ2_0=1)
    endif()
endif()

set(SONDER_LLAMACPP_EXTRA_LIBS "")
set(_sonder_mtmd_h "${llamacpp_SOURCE_DIR}/tools/mtmd/mtmd.h")
set(_sonder_mtmd_helper_h "${llamacpp_SOURCE_DIR}/tools/mtmd/mtmd-helper.h")
if((SONDER_WITH_MTMD OR LLAMA_BUILD_MTMD)
   AND TARGET mtmd
   AND EXISTS "${_sonder_mtmd_h}"
   AND EXISTS "${_sonder_mtmd_helper_h}")
    set_target_properties(mtmd PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)
    list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_MTMD=1)
    list(APPEND SONDER_LLAMACPP_PRIVATE_INCLUDES "${llamacpp_SOURCE_DIR}/tools/mtmd")
    list(APPEND SONDER_LLAMACPP_EXTRA_LIBS mtmd)

    file(READ "${_sonder_mtmd_helper_h}" _sonder_mtmd_helper_h_content)
    string(FIND "${_sonder_mtmd_helper_h_content}" "mtmd_helper_bitmap_wrapper" _idx_mtmd_wrapper)
    if(NOT _idx_mtmd_wrapper EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_MTMD_WRAPPER_API=1)
    endif()

    file(READ "${_sonder_mtmd_h}" _sonder_mtmd_h_content)
    string(FIND "${_sonder_mtmd_h_content}" "text_len" _idx_mtmd_text_len)
    if(NOT _idx_mtmd_text_len EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_MTMD_TEXT_LEN=1)
    endif()
    string(FIND "${_sonder_mtmd_h_content}" "mtmd_tokenize_from_parts" _idx_mtmd_parts)
    if(NOT _idx_mtmd_parts EQUAL -1)
        list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_MTMD_PARTS=1)
    endif()
elseif(SONDER_WITH_MTMD)
    message(WARNING "SONDER_WITH_MTMD=ON requested, but upstream llama.cpp tree does not expose standalone mtmd target")
endif()

if(SONDER_LLAMACPP_FEATURE_DEFS)
    message(STATUS "Sonder llamacpp detected features: ${SONDER_LLAMACPP_FEATURE_DEFS}")
endif()

add_library(sonder_backend_llamacpp STATIC
    llamacpp_backend.cpp
    llamacpp_util.cpp
)
add_library(sonder::backend_llamacpp ALIAS sonder_backend_llamacpp)
target_include_directories(sonder_backend_llamacpp
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    PRIVATE ${SONDER_LLAMACPP_PRIVATE_INCLUDES}
)
target_compile_features(sonder_backend_llamacpp PUBLIC cxx_std_20)
# llama.h stays private: nothing outside this directory sees llama.cpp types.
target_link_libraries(sonder_backend_llamacpp PRIVATE llama ggml ${SONDER_LLAMACPP_EXTRA_LIBS})
target_compile_definitions(sonder_backend_llamacpp
    PUBLIC  ${SONDER_LLAMACPP_FEATURE_DEFS}
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
