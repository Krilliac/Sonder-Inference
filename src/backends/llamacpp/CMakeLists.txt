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
# llama.cpp is pulled with FetchContent at the tag pinned in
# cmake/llamacpp-pin.json (SONDER_LLAMACPP_TRACK selects the entry). The record
# in docs/LICENSE_REVIEW.md must match: `python -I scripts/llamacpp_pin.py check`.
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
# SONDER_LLAMACPP_TAG and the matching, different SONDER_LLAMACPP_SHA256 (and
# optionally SONDER_LLAMACPP_URL) together; configure fails when the tag differs
# but the SHA-256 does not, and SHA-256 verification fails closed on mismatch.
#
# Opt-in GPU presets live in CMakePresets.json: windows-cuda-rtx,
# windows-vulkan-xt, linux-cuda-rtx, linux-rocm-xt, and linux-vulkan-xt.

if(NOT SONDER_WITH_LLAMA_CPP)
    return()
endif()

# The in-process backend never links llama.cpp's RPC client: a broken or
# hostile rpc-server aborts the whole client process (GGML_ABORT in
# ggml-rpc.cpp) and every tensor it returns would be parsed inside sonder.
# Cross-host layer split runs only in the isolated llamaserver rpc_worker child
# (docs/integration/rpc-worker.md). Checked before anything is fetched.
if(GGML_RPC)
    message(FATAL_ERROR
        "GGML_RPC=ON is refused for the in-process llamacpp backend: the RPC client must run in an "
        "isolated worker process. Use the llamaserver backend's rpc_worker (docs/integration/rpc-worker.md).")
endif()
# Pin it off in the cache as well, so a fork (FETCHCONTENT_SOURCE_DIR_LLAMACPP)
# whose ggml declares GGML_RPC with a default of ON cannot turn it back on:
# option() leaves an existing cache entry alone.
set(GGML_RPC OFF CACHE BOOL "ggml: use RPC (refused for the in-process llamacpp backend)" FORCE)

include(FetchContent)

# --- pin resolution begin (extracted by scripts/test_llamacpp_pin.py) ---
# Pin manifest: cmake/llamacpp-pin.json holds one entry per track (tag, peeled
# commit, URL, SHA-256). It is the single source of truth; CI, the docs check
# (scripts/llamacpp_pin.py check) and the test expectations all read it.
set(SONDER_LLAMACPP_TRACK "mainline" CACHE STRING
    "llama.cpp pin track from cmake/llamacpp-pin.json (the manifest has only mainline)")
set_property(CACHE SONDER_LLAMACPP_TRACK PROPERTY STRINGS mainline)
set(SONDER_LLAMACPP_PIN_FILE "${CMAKE_CURRENT_LIST_DIR}/../../../cmake/llamacpp-pin.json" CACHE FILEPATH
    "Pin manifest to read (override for experiments only)")

# Override-only variables: empty means "use the manifest value". Manifest values
# are never stored in the cache, so a manifest bump reaches existing build dirs.
set(SONDER_LLAMACPP_TAG "" CACHE STRING
    "Override llama.cpp tag (must be paired with a different SONDER_LLAMACPP_SHA256)")
set(SONDER_LLAMACPP_URL "" CACHE STRING
    "Override llama.cpp source archive URL (default: manifest URL with the tag substituted)")
set(SONDER_LLAMACPP_SHA256 "" CACHE STRING
    "Override expected SHA-256 of the llama.cpp source archive")
# Manifest values applied by the last configure of this build directory: SHA-256,
# tag and the archive URL derived from them.
# (INTERNAL cache entries imply FORCE, so they are only written at the end and
# copied into locals here.)
set(_sonder_llamacpp_applied_sha256 "${SONDER_LLAMACPP_APPLIED_SHA256}")
set(_sonder_llamacpp_applied_tag    "${SONDER_LLAMACPP_APPLIED_TAG}")
set(_sonder_llamacpp_applied_url    "${SONDER_LLAMACPP_APPLIED_URL}")
# Values that earlier revisions of this file stored in the cache as the defaults
# (b11195); a cached value equal to one of these was never an override.
set(_sonder_llamacpp_legacy_sha256
    "d9818b7786c8a3b063bac7eb7dc8710e275638f1e288e206720c9e22e7ca216a")
set(_sonder_llamacpp_legacy_tag "b11195")
set(_sonder_llamacpp_legacy_url
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11195.tar.gz")

if(NOT EXISTS "${SONDER_LLAMACPP_PIN_FILE}")
    message(FATAL_ERROR "llama.cpp pin manifest not found: ${SONDER_LLAMACPP_PIN_FILE}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SONDER_LLAMACPP_PIN_FILE}")
file(READ "${SONDER_LLAMACPP_PIN_FILE}" _sonder_llamacpp_pin_json)
string(JSON _sonder_llamacpp_entry ERROR_VARIABLE _sonder_llamacpp_err
       GET "${_sonder_llamacpp_pin_json}" tracks "${SONDER_LLAMACPP_TRACK}")
if(_sonder_llamacpp_err)
    message(FATAL_ERROR "SONDER_LLAMACPP_TRACK='${SONDER_LLAMACPP_TRACK}' is not a track in "
                        "${SONDER_LLAMACPP_PIN_FILE}: ${_sonder_llamacpp_err}")
endif()
string(JSON _sonder_llamacpp_m_tag    GET "${_sonder_llamacpp_entry}" tag)
string(JSON _sonder_llamacpp_m_url    GET "${_sonder_llamacpp_entry}" url)
string(JSON _sonder_llamacpp_m_sha256 GET "${_sonder_llamacpp_entry}" sha256)
# expected_features: SONDER_LLAMACPP_* defines that header detection must produce
# for this track's tarball (checked after detection, see "expected features").
set(SONDER_LLAMACPP_EXPECTED_FEATURES "")
string(JSON _sonder_llamacpp_ef_type ERROR_VARIABLE _sonder_llamacpp_err
       TYPE "${_sonder_llamacpp_entry}" expected_features)
if(NOT _sonder_llamacpp_err)
    if(NOT _sonder_llamacpp_ef_type STREQUAL "ARRAY")
        message(FATAL_ERROR "${SONDER_LLAMACPP_PIN_FILE}: tracks.${SONDER_LLAMACPP_TRACK}."
                            "expected_features must be an array (got ${_sonder_llamacpp_ef_type})")
    endif()
    string(JSON _sonder_llamacpp_ef_len LENGTH "${_sonder_llamacpp_entry}" expected_features)
    if(_sonder_llamacpp_ef_len GREATER 0)
        math(EXPR _sonder_llamacpp_ef_last "${_sonder_llamacpp_ef_len} - 1")
        foreach(_sonder_llamacpp_ef_i RANGE ${_sonder_llamacpp_ef_last})
            string(JSON _sonder_llamacpp_ef_name GET "${_sonder_llamacpp_entry}"
                   expected_features ${_sonder_llamacpp_ef_i})
            if(NOT _sonder_llamacpp_ef_name MATCHES "^SONDER_LLAMACPP_[A-Z0-9_]+$")
                message(FATAL_ERROR "${SONDER_LLAMACPP_PIN_FILE}: expected_features entry "
                                    "'${_sonder_llamacpp_ef_name}' is not a SONDER_LLAMACPP_* define name")
            endif()
            list(APPEND SONDER_LLAMACPP_EXPECTED_FEATURES "${_sonder_llamacpp_ef_name}")
        endforeach()
    endif()
endif()

# Stale-cache handling. A cached value equal to what an earlier configure applied
# from the manifest (or to a legacy default) was a default, not an override, and
# moves to the manifest. Each variable is judged on its own: an explicit TAG or
# URL override survives a SHA-256 that happens to equal the manifest value.
if(SONDER_LLAMACPP_SHA256 AND
   (SONDER_LLAMACPP_SHA256 STREQUAL _sonder_llamacpp_applied_sha256
    OR SONDER_LLAMACPP_SHA256 IN_LIST _sonder_llamacpp_legacy_sha256))
    if(NOT SONDER_LLAMACPP_SHA256 STREQUAL _sonder_llamacpp_m_sha256)
        message(STATUS "llama.cpp pin: cached SHA-256 ${SONDER_LLAMACPP_SHA256} was the "
                       "previous default; moving to manifest ${_sonder_llamacpp_m_sha256} "
                       "(${_sonder_llamacpp_m_tag})")
    endif()
    set(SONDER_LLAMACPP_SHA256 "" CACHE STRING "" FORCE)
endif()
if(SONDER_LLAMACPP_TAG AND
   (SONDER_LLAMACPP_TAG STREQUAL _sonder_llamacpp_applied_tag
    OR SONDER_LLAMACPP_TAG IN_LIST _sonder_llamacpp_legacy_tag))
    set(SONDER_LLAMACPP_TAG "" CACHE STRING "" FORCE)
endif()
if(SONDER_LLAMACPP_URL AND
   (SONDER_LLAMACPP_URL STREQUAL _sonder_llamacpp_applied_url
    OR SONDER_LLAMACPP_URL IN_LIST _sonder_llamacpp_legacy_url))
    set(SONDER_LLAMACPP_URL "" CACHE STRING "" FORCE)
endif()

set(SONDER_LLAMACPP_RESOLVED_TAG    "${_sonder_llamacpp_m_tag}")
set(SONDER_LLAMACPP_RESOLVED_URL    "${_sonder_llamacpp_m_url}")
set(SONDER_LLAMACPP_RESOLVED_SHA256 "${_sonder_llamacpp_m_sha256}")
if(SONDER_LLAMACPP_TAG AND NOT SONDER_LLAMACPP_TAG STREQUAL _sonder_llamacpp_m_tag)
    if(NOT SONDER_LLAMACPP_SHA256 OR SONDER_LLAMACPP_SHA256 STREQUAL _sonder_llamacpp_m_sha256)
        message(FATAL_ERROR "SONDER_LLAMACPP_TAG=${SONDER_LLAMACPP_TAG} differs from the manifest tag "
                            "${_sonder_llamacpp_m_tag}, so SONDER_LLAMACPP_SHA256 must be set to the "
                            "new tarball's SHA-256 (it is unset or still the manifest value).")
    endif()
    set(SONDER_LLAMACPP_RESOLVED_TAG "${SONDER_LLAMACPP_TAG}")
    string(REPLACE "${_sonder_llamacpp_m_tag}" "${SONDER_LLAMACPP_TAG}"
           SONDER_LLAMACPP_RESOLVED_URL "${_sonder_llamacpp_m_url}")
endif()
if(SONDER_LLAMACPP_SHA256)
    if(NOT SONDER_LLAMACPP_SHA256 STREQUAL _sonder_llamacpp_m_sha256)
        message(WARNING "llama.cpp pin override: SONDER_LLAMACPP_SHA256=${SONDER_LLAMACPP_SHA256} "
                        "replaces the manifest value ${_sonder_llamacpp_m_sha256}")
    endif()
    set(SONDER_LLAMACPP_RESOLVED_SHA256 "${SONDER_LLAMACPP_SHA256}")
endif()
if(SONDER_LLAMACPP_URL)
    set(SONDER_LLAMACPP_RESOLVED_URL "${SONDER_LLAMACPP_URL}")
endif()
set(SONDER_LLAMACPP_APPLIED_SHA256 "${_sonder_llamacpp_m_sha256}" CACHE INTERNAL "")
set(SONDER_LLAMACPP_APPLIED_TAG    "${_sonder_llamacpp_m_tag}"    CACHE INTERNAL "")
set(SONDER_LLAMACPP_APPLIED_URL    "${_sonder_llamacpp_m_url}"    CACHE INTERNAL "")
message(STATUS "llama.cpp pin: track=${SONDER_LLAMACPP_TRACK} tag=${SONDER_LLAMACPP_RESOLVED_TAG} "
               "url=${SONDER_LLAMACPP_RESOLVED_URL} sha256=${SONDER_LLAMACPP_RESOLVED_SHA256}")

# --- pin resolution end ---

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
    URL      "${SONDER_LLAMACPP_RESOLVED_URL}"
    URL_HASH "SHA256=${SONDER_LLAMACPP_RESOLVED_SHA256}"
    ${SONDER_LLAMACPP_FETCH_EXTRA}
)
FetchContent_MakeAvailable(llamacpp)
# Backstop for a tree that adds the RPC backend some other way.
if(TARGET ggml-rpc)
    message(FATAL_ERROR
        "the fetched llama.cpp defines the ggml-rpc target; the in-process llamacpp backend must not contain the "
        "RPC client (docs/integration/rpc-worker.md)")
endif()

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

# Feature auto-detection against the resolved llama.cpp / ggml headers so the
# pinned mainline tag, older trees (b11195) and newer/forked trees (e.g. PrismML-Eng/llama.cpp)
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
       AND NOT _idx_load_mtp EQUAL -1)
        # The NextN calls (llama_set_embeddings_nextn, llama_set_nextn_layer_offset,
        # llama_get_ctx_other, ...) are public in llama.h from v1.0.0; older trees
        # (b11195) declare them only in the private src/llama-ext.h, which then has
        # to be included (SONDER_LLAMACPP_MTP_VIA_EXT).
        set(_sonder_llama_has_nextn OFF)
        set(_sonder_llama_nextn_api "")
        string(FIND "${_sonder_llama_h_content}" "llama_set_embeddings_nextn" _idx_embd_nextn)
        if(NOT _idx_embd_nextn EQUAL -1)
            set(_sonder_llama_has_nextn ON)
            set(_sonder_llama_nextn_api "${_sonder_llama_h_content}")
        elseif(EXISTS "${_sonder_llama_ext_h}")
            file(READ "${_sonder_llama_ext_h}" _sonder_llama_ext_h_content)
            string(FIND "${_sonder_llama_ext_h_content}" "llama_set_embeddings_nextn" _idx_embd_nextn)
            if(NOT _idx_embd_nextn EQUAL -1)
                set(_sonder_llama_has_nextn ON)
                set(_sonder_llama_nextn_api "${_sonder_llama_ext_h_content}")
                list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_MTP_VIA_EXT=1)
                list(APPEND SONDER_LLAMACPP_PRIVATE_INCLUDES "${llamacpp_SOURCE_DIR}/src")
            endif()
        endif()
        if(_sonder_llama_has_nextn)
            list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_MTP=1)

            string(FIND "${_sonder_llama_nextn_api}" "llama_set_nextn_layer_offset" _idx_nextn_offset)
            if(NOT _idx_nextn_offset EQUAL -1)
                list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET=1)
            endif()

            string(FIND "${_sonder_llama_nextn_api}" "llama_get_ctx_other" _idx_get_ctx_other)
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

# --- expected features begin (extracted by scripts/test_llamacpp_pin.py) ---
# A header-detection regression (an upstream rename, or a detection bug like the
# NextN move from llama-ext.h to llama.h) would otherwise turn a feature off
# silently, and the tests that follow the macros would still pass. When the
# source is the manifest tarball itself (manifest SHA-256, no local source dir),
# every define the track lists in expected_features must have been detected.
if(SONDER_LLAMACPP_EXPECTED_FEATURES)
    if(FETCHCONTENT_SOURCE_DIR_LLAMACPP)
        message(STATUS "llama.cpp expected_features not enforced: local source dir "
                       "${FETCHCONTENT_SOURCE_DIR_LLAMACPP}")
    elseif(NOT SONDER_LLAMACPP_RESOLVED_SHA256 STREQUAL _sonder_llamacpp_m_sha256)
        message(STATUS "llama.cpp expected_features not enforced: SHA-256 override "
                       "${SONDER_LLAMACPP_RESOLVED_SHA256}")
    else()
        set(_sonder_llamacpp_missing_features "")
        foreach(_sonder_llamacpp_ef_name IN LISTS SONDER_LLAMACPP_EXPECTED_FEATURES)
            if(NOT "${_sonder_llamacpp_ef_name}=1" IN_LIST SONDER_LLAMACPP_FEATURE_DEFS)
                list(APPEND _sonder_llamacpp_missing_features "${_sonder_llamacpp_ef_name}")
            endif()
        endforeach()
        if(_sonder_llamacpp_missing_features)
            message(FATAL_ERROR "llama.cpp ${SONDER_LLAMACPP_RESOLVED_TAG} (track "
                                "${SONDER_LLAMACPP_TRACK}): header detection did not produce "
                                "${_sonder_llamacpp_missing_features}, which "
                                "${SONDER_LLAMACPP_PIN_FILE} lists in expected_features.")
        endif()
        message(STATUS "llama.cpp expected_features present: ${SONDER_LLAMACPP_EXPECTED_FEATURES}")
    endif()
endif()
# --- expected features end ---

add_library(sonder_backend_llamacpp STATIC
    llamacpp_backend.cpp
    llamacpp_util.cpp
    slot_store.cpp
    request_guards.cpp
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
    PRIVATE SONDER_LLAMACPP_TAG_STRING="${SONDER_LLAMACPP_RESOLVED_TAG}"
)
if(COMMAND sonder_set_warnings)
    sonder_set_warnings(sonder_backend_llamacpp)
endif()
# llama.cpp reports grammar violations by throwing std::runtime_error through its extern "C" API
# (llama_sampler_accept / llama_sampler_apply). Under MSVC /EHsc the compiler assumes extern "C"
# functions never throw, can drop our try/catch around those calls and the exception then hits a
# noexcept frame (std::terminate). /EHs (without the "c") keeps the catch. Scoped to this target.
if(MSVC)
    # /EHsc arrives from both CMAKE_CXX_FLAGS and sonder_set_warnings. Each /EH letter is independent
    # and the last instance wins, so appending /EHc- turns off only the "c" (keeping "s"). This also
    # applies to clang-cl, whose driver parses the letters the same way.
    target_compile_options(sonder_backend_llamacpp PRIVATE /EHs /EHc-)
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
