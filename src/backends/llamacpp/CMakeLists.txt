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
# llama.cpp is pulled with FetchContent at the single archive pinned in
# cmake/llamacpp-pin.json and patched from third_party/ggml-patches/ via
# cmake/ApplyGgmlPatches.cmake. Separate llama.cpp forks, multi-track pins, and
# unverified local source-tree overrides are retired. The record in
# docs/LICENSE_REVIEW.md must match: `python -I scripts/llamacpp_pin.py check`.
#
# Offline / CI cached archive builds pass a local mirror path of the pinned
# archive via `-DSONDER_LLAMACPP_URL=/path/to/llama.cpp-v1.0.0.tar.gz`;
# FetchContent still verifies the manifest SHA-256 and applies all patches.
#
# Opt-in GPU presets live in CMakePresets.json: windows-cuda-blackwell,
# windows-cuda-rtx, windows-rocm-xt, windows-vulkan-xt, linux-llamacpp,
# linux-cuda-blackwell, linux-cuda-rtx, linux-rocm-xt, and linux-vulkan-xt.

# Optional llguidance structured-output path (see the llguidance block below). OFF by default: the
# default build, CI and every existing test are unchanged. Needs the Rust toolchain (cargo) and git
# at build time, and network access to github.com the first time llguidance is fetched.
option(SONDER_WITH_LLGUIDANCE
       "Build llguidance (Rust) into the llama.cpp backend for response_format json_schema" OFF)

if(NOT SONDER_WITH_LLAMA_CPP)
    if(SONDER_WITH_LLGUIDANCE)
        message(WARNING "SONDER_WITH_LLGUIDANCE=ON has no effect without SONDER_WITH_LLAMA_CPP=ON")
    endif()
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
# Pin it off in the cache as well, so a pre-populated CMakeCache.txt cannot
# turn GGML_RPC back on: option() leaves an existing cache entry alone.
set(GGML_RPC OFF CACHE BOOL "ggml: use RPC (refused for the in-process llamacpp backend)" FORCE)

include(FetchContent)

# --- pin resolution begin (extracted by scripts/test_llamacpp_pin.py) ---
# Pin manifest: cmake/llamacpp-pin.json holds the single pinned mainline archive
# (tag, peeled commit, URL, SHA-256) plus SI's third_party/ggml-patches/ series.
# Separate forks and local checkout overrides are refused fail-closed.
if(NOT SONDER_LLAMACPP_REPO_ROOT)
    get_filename_component(SONDER_LLAMACPP_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()
set(SONDER_LLAMACPP_TRACK "mainline" CACHE STRING
    "Single llama.cpp engine track (only 'mainline' is permitted; forks are retired)")
set_property(CACHE SONDER_LLAMACPP_TRACK PROPERTY STRINGS mainline)
if(NOT SONDER_LLAMACPP_TRACK STREQUAL "" AND NOT SONDER_LLAMACPP_TRACK STREQUAL "mainline")
    message(FATAL_ERROR "SONDER_LLAMACPP_TRACK='${SONDER_LLAMACPP_TRACK}' is refused: Sonder Inference builds only "
                        "the single pinned 'mainline' engine plus third_party/ggml-patches/ (separate forks are retired).")
endif()
set(SONDER_LLAMACPP_TRACK "mainline")

if(SONDER_LLAMACPP_SOURCE_DIR OR FETCHCONTENT_SOURCE_DIR_LLAMACPP)
    message(FATAL_ERROR "Local llama.cpp checkout overrides (SONDER_LLAMACPP_SOURCE_DIR / "
                        "FETCHCONTENT_SOURCE_DIR_LLAMACPP) are retired: Sonder Inference compiles only the "
                        "SHA-pinned archive in cmake/llamacpp-pin.json with third_party/ggml-patches/. "
                        "For offline builds pass -DSONDER_LLAMACPP_URL=/path/to/pinned-archive.tar.gz.")
endif()

# Configure-hygiene seam (not an untrusted-input security boundary): unit tests
# point SONDER_LLAMACPP_PIN_FILE at temporary manifests; production builds use
# cmake/llamacpp-pin.json enforced by CI (`scripts/llamacpp_pin.py check`).
set(SONDER_LLAMACPP_PIN_FILE "${SONDER_LLAMACPP_REPO_ROOT}/cmake/llamacpp-pin.json" CACHE FILEPATH
    "Pin manifest to read (override for tests only)")
set(SONDER_LLAMACPP_URL "" CACHE STRING
    "Optional local file or mirror URL for the pinned llama.cpp archive (must still match manifest SHA-256)")
# Legacy cache entries retained only so existing build dirs migrate cleanly;
# any explicit value differing from the manifest fails closed below.
set(SONDER_LLAMACPP_TAG "" CACHE STRING "Retired (must be empty or match manifest tag)")
set(SONDER_LLAMACPP_SHA256 "" CACHE STRING "Retired (must be empty or match manifest SHA-256)")

set(_sonder_llamacpp_applied_sha256 "${SONDER_LLAMACPP_APPLIED_SHA256}")
set(_sonder_llamacpp_applied_tag    "${SONDER_LLAMACPP_APPLIED_TAG}")
set(_sonder_llamacpp_applied_url    "${SONDER_LLAMACPP_APPLIED_URL}")
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
string(JSON _sonder_llamacpp_tracks_len ERROR_VARIABLE _sonder_llamacpp_err
       LENGTH "${_sonder_llamacpp_pin_json}" tracks)
if(_sonder_llamacpp_err OR NOT _sonder_llamacpp_tracks_len EQUAL 1)
    message(FATAL_ERROR "${SONDER_LLAMACPP_PIN_FILE}: tracks must contain only 'mainline'")
endif()
string(JSON _sonder_llamacpp_entry ERROR_VARIABLE _sonder_llamacpp_err
       GET "${_sonder_llamacpp_pin_json}" tracks mainline)
if(_sonder_llamacpp_err)
    message(FATAL_ERROR "tracks.mainline missing in ${SONDER_LLAMACPP_PIN_FILE}: ${_sonder_llamacpp_err}")
endif()
string(JSON _sonder_llamacpp_m_tag    GET "${_sonder_llamacpp_entry}" tag)
string(JSON _sonder_llamacpp_m_url    GET "${_sonder_llamacpp_entry}" url)
string(JSON _sonder_llamacpp_m_sha256 GET "${_sonder_llamacpp_entry}" sha256)

# Register configure dependencies on the SI patch directory and listed patch files.
set(_sonder_llamacpp_patch_dir_rel "third_party/ggml-patches")
string(JSON _sonder_llamacpp_pd_val ERROR_VARIABLE _sonder_llamacpp_pd_err
       GET "${_sonder_llamacpp_entry}" patch_dir)
if(NOT _sonder_llamacpp_pd_err)
    set(_sonder_llamacpp_patch_dir_rel "${_sonder_llamacpp_pd_val}")
endif()
set(SONDER_LLAMACPP_PATCH_DIR "${SONDER_LLAMACPP_REPO_ROOT}/${_sonder_llamacpp_patch_dir_rel}")
if(IS_DIRECTORY "${SONDER_LLAMACPP_PATCH_DIR}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SONDER_LLAMACPP_PATCH_DIR}")
endif()
string(JSON _sonder_llamacpp_patches_type ERROR_VARIABLE _sonder_llamacpp_p_err
       TYPE "${_sonder_llamacpp_entry}" patches)
if(NOT _sonder_llamacpp_p_err)
    if(NOT _sonder_llamacpp_patches_type STREQUAL "ARRAY")
        message(FATAL_ERROR "${SONDER_LLAMACPP_PIN_FILE}: tracks.mainline.patches must be an array")
    endif()
    string(JSON _sonder_llamacpp_patches_len LENGTH "${_sonder_llamacpp_entry}" patches)
    if(_sonder_llamacpp_patches_len GREATER 0)
        math(EXPR _sonder_llamacpp_p_last "${_sonder_llamacpp_patches_len} - 1")
        foreach(_sonder_llamacpp_pi RANGE ${_sonder_llamacpp_p_last})
            string(JSON _sonder_llamacpp_p_file GET "${_sonder_llamacpp_entry}" patches ${_sonder_llamacpp_pi} file)
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                         "${SONDER_LLAMACPP_PATCH_DIR}/${_sonder_llamacpp_p_file}")
        endforeach()
    endif()
endif()

# expected_features: SONDER_LLAMACPP_* defines that header detection must produce.
set(SONDER_LLAMACPP_EXPECTED_FEATURES "")
string(JSON _sonder_llamacpp_ef_type ERROR_VARIABLE _sonder_llamacpp_err
       TYPE "${_sonder_llamacpp_entry}" expected_features)
if(NOT _sonder_llamacpp_err)
    if(NOT _sonder_llamacpp_ef_type STREQUAL "ARRAY")
        message(FATAL_ERROR "${SONDER_LLAMACPP_PIN_FILE}: tracks.mainline."
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

# Stale-cache migration for existing build directories upgrading from older pins.
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

# Refuse any explicit tag or SHA-256 override differing from the manifest.
if(SONDER_LLAMACPP_TAG AND NOT SONDER_LLAMACPP_TAG STREQUAL _sonder_llamacpp_m_tag)
    message(FATAL_ERROR "SONDER_LLAMACPP_TAG=${SONDER_LLAMACPP_TAG} is refused: Sonder Inference builds only "
                        "manifest tag ${_sonder_llamacpp_m_tag} from ${SONDER_LLAMACPP_PIN_FILE}.")
endif()
if(SONDER_LLAMACPP_SHA256 AND NOT SONDER_LLAMACPP_SHA256 STREQUAL _sonder_llamacpp_m_sha256)
    message(FATAL_ERROR "SONDER_LLAMACPP_SHA256=${SONDER_LLAMACPP_SHA256} is refused: archive SHA-256 must match "
                        "manifest ${_sonder_llamacpp_m_sha256} in ${SONDER_LLAMACPP_PIN_FILE}.")
endif()

set(SONDER_LLAMACPP_RESOLVED_TAG    "${_sonder_llamacpp_m_tag}")
set(SONDER_LLAMACPP_RESOLVED_URL    "${_sonder_llamacpp_m_url}")
set(SONDER_LLAMACPP_RESOLVED_SHA256 "${_sonder_llamacpp_m_sha256}")
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

set(_sonder_apply_patches_script "${SONDER_LLAMACPP_REPO_ROOT}/cmake/ApplyGgmlPatches.cmake")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_sonder_apply_patches_script}")

# If <FETCHCONTENT_BASE_DIR>/llamacpp-src already exists from an earlier configure,
# verify and apply any newly added patches before FetchContent evaluates llama.cpp.
if(FETCHCONTENT_BASE_DIR)
    get_filename_component(_sonder_fc_base_dir "${FETCHCONTENT_BASE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
else()
    set(_sonder_fc_base_dir "${CMAKE_BINARY_DIR}/_deps")
endif()
if(EXISTS "${_sonder_fc_base_dir}/llamacpp-src/CMakeLists.txt")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
                "-DSONDER_LLAMACPP_PIN_FILE=${SONDER_LLAMACPP_PIN_FILE}"
                "-DSONDER_LLAMACPP_REPO_ROOT=${SONDER_LLAMACPP_REPO_ROOT}"
                "-DSONDER_LLAMACPP_SOURCE_DIR_TO_PATCH=${_sonder_fc_base_dir}/llamacpp-src"
                -P "${_sonder_apply_patches_script}"
        RESULT_VARIABLE _sonder_pre_patch_rc
    )
    if(NOT _sonder_pre_patch_rc EQUAL 0)
        message(FATAL_ERROR "Failed to verify/apply ggml-patches on existing ${_sonder_fc_base_dir}/llamacpp-src")
    endif()
endif()

# CMake >= 3.28: only the llama.cpp targets we link are built by default.
set(SONDER_LLAMACPP_FETCH_EXTRA)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
    list(APPEND SONDER_LLAMACPP_FETCH_EXTRA EXCLUDE_FROM_ALL)
endif()
FetchContent_Declare(llamacpp
    URL      "${SONDER_LLAMACPP_RESOLVED_URL}"
    URL_HASH "SHA256=${SONDER_LLAMACPP_RESOLVED_SHA256}"
    PATCH_COMMAND "${CMAKE_COMMAND}"
                  "-DSONDER_LLAMACPP_PIN_FILE=${SONDER_LLAMACPP_PIN_FILE}"
                  "-DSONDER_LLAMACPP_REPO_ROOT=${SONDER_LLAMACPP_REPO_ROOT}"
                  -P "${_sonder_apply_patches_script}"
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

# Feature auto-detection against the resolved llama.cpp / ggml headers on the
# pinned mainline archive + SI's third_party/ggml-patches/ series.
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

    # Absent on unpatched v1.0.0; reachable only if a reviewed SI patch in
    # third_party/ggml-patches/ adds GGML_TYPE_PQ2_0 to ggml.h.
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

# prism.hadamard.* (Hadamard-folded Bonsai) runtime support. Mainline (>= b11556 /
# v1.0.0, PR #29600) registers the key in src/llama-arch.cpp; the PrismML-Eng fork
# reads it as a string literal in src/llama-model.cpp (and mentions it in
# src/llama-context.cpp). Absent on older builds such as b11195; the backend then
# refuses to load such models instead of silently producing garbage.
# Override: -DSONDER_LLAMACPP_FORCE_PRISM_HADAMARD=ON|OFF.
set(SONDER_LLAMACPP_FORCE_PRISM_HADAMARD "" CACHE STRING
    "Force prism.hadamard runtime detection: ON, OFF, or empty to probe the llama.cpp sources")
set_property(CACHE SONDER_LLAMACPP_FORCE_PRISM_HADAMARD PROPERTY STRINGS "" ON OFF)
set(_sonder_has_prism_hadamard FALSE)
if(SONDER_LLAMACPP_FORCE_PRISM_HADAMARD STREQUAL "")
    foreach(_sonder_prism_src IN ITEMS llama-arch.cpp llama-model.cpp llama-context.cpp)
        set(_sonder_prism_path "${llamacpp_SOURCE_DIR}/src/${_sonder_prism_src}")
        if(EXISTS "${_sonder_prism_path}")
            file(READ "${_sonder_prism_path}" _sonder_prism_content)
            string(FIND "${_sonder_prism_content}" "prism.hadamard.version" _idx_prism_hadamard)
            if(NOT _idx_prism_hadamard EQUAL -1)
                set(_sonder_has_prism_hadamard TRUE)
                break()
            endif()
        endif()
    endforeach()
elseif(SONDER_LLAMACPP_FORCE_PRISM_HADAMARD STREQUAL "ON")
    set(_sonder_has_prism_hadamard TRUE)
elseif(NOT SONDER_LLAMACPP_FORCE_PRISM_HADAMARD STREQUAL "OFF")
    message(FATAL_ERROR "SONDER_LLAMACPP_FORCE_PRISM_HADAMARD must be ON, OFF or empty, got '${SONDER_LLAMACPP_FORCE_PRISM_HADAMARD}'")
endif()
if(_sonder_has_prism_hadamard)
    list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_PRISM_HADAMARD=1)
endif()

if(SONDER_LLAMACPP_FEATURE_DEFS)
    message(STATUS "Sonder llamacpp detected features: ${SONDER_LLAMACPP_FEATURE_DEFS}")
endif()

# --- expected features begin (extracted by scripts/test_llamacpp_pin.py) ---
# A header-detection regression (an upstream rename, or a detection bug like the
# NextN move from llama-ext.h to llama.h) would otherwise turn a feature off
# silently, and the tests that follow the macros would still pass. Every define
# listed in expected_features must be detected on the pinned archive.
if(SONDER_LLAMACPP_EXPECTED_FEATURES)
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
# --- expected features end ---

# --- llguidance begin (optional: SONDER_WITH_LLGUIDANCE) ---
# llama.cpp builds llguidance only inside its `common` library (LLAMA_LLGUIDANCE), which this project
# does not build (LLAMA_BUILD_COMMON=OFF). So the Rust library is built here with the same recipe
# upstream uses (cargo build --release --package llguidance at a pinned commit) and wrapped by
# llguidance_sampler.cpp. SONDER_LLAMACPP_HAS_LLGUIDANCE=1 is defined only when this block runs.
set(SONDER_LLGUIDANCE_GIT_REPOSITORY "https://github.com/guidance-ai/llguidance")
# v1.0.1: the commit llama.cpp v1.0.0 pins in common/CMakeLists.txt.
set(SONDER_LLGUIDANCE_GIT_TAG "d795912fedc7d393de740177ea9ea761e7905774")
set(SONDER_LLGUIDANCE_LINK_LIBS "")
if(SONDER_WITH_LLGUIDANCE)
    find_program(SONDER_CARGO_EXECUTABLE cargo
                 HINTS "$ENV{CARGO_HOME}/bin" "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
    if(NOT SONDER_CARGO_EXECUTABLE)
        message(FATAL_ERROR
            "SONDER_WITH_LLGUIDANCE=ON requires the Rust toolchain, but `cargo` was not found on PATH "
            "(or in $CARGO_HOME/bin, ~/.cargo/bin). Install it from https://rustup.rs and re-run "
            "configure, set -DSONDER_CARGO_EXECUTABLE=/path/to/cargo, or configure with "
            "-DSONDER_WITH_LLGUIDANCE=OFF (the default).")
    endif()
    find_package(Git QUIET)
    if(NOT GIT_FOUND)
        message(FATAL_ERROR
            "SONDER_WITH_LLGUIDANCE=ON requires `git` to fetch llguidance "
            "${SONDER_LLGUIDANCE_GIT_REPOSITORY} at ${SONDER_LLGUIDANCE_GIT_TAG}. Install git or "
            "configure with -DSONDER_WITH_LLGUIDANCE=OFF.")
    endif()
    # Keep the pin in step with the llama.cpp tree this build uses: warn when upstream moved it.
    set(_sonder_llg_upstream_cmake "${llamacpp_SOURCE_DIR}/common/CMakeLists.txt")
    if(EXISTS "${_sonder_llg_upstream_cmake}")
        file(READ "${_sonder_llg_upstream_cmake}" _sonder_llg_upstream_text)
        if(_sonder_llg_upstream_text MATCHES "llguidance_ext[^)]*GIT_TAG[ \t]+([0-9a-f]+)")
            if(NOT CMAKE_MATCH_1 STREQUAL SONDER_LLGUIDANCE_GIT_TAG)
                message(WARNING "llama.cpp ${SONDER_LLAMACPP_RESOLVED_TAG} pins llguidance ${CMAKE_MATCH_1}, "
                                "this build uses ${SONDER_LLGUIDANCE_GIT_TAG}; review the llguidance_sampler.cpp "
                                "port and docs/LICENSE_REVIEW.md before moving the pin.")
            endif()
        endif()
    endif()

    include(ExternalProject)
    set(_sonder_llg_src "${CMAKE_BINARY_DIR}/llguidance/source")
    set(_sonder_llg_out "${_sonder_llg_src}/target/release")
    set(_sonder_llg_lib "${_sonder_llg_out}/${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}")
    ExternalProject_Add(sonder_llguidance_ext
        GIT_REPOSITORY    "${SONDER_LLGUIDANCE_GIT_REPOSITORY}"
        GIT_TAG           "${SONDER_LLGUIDANCE_GIT_TAG}"
        PREFIX            "${CMAKE_BINARY_DIR}/llguidance"
        SOURCE_DIR        "${_sonder_llg_src}"
        BUILD_IN_SOURCE   TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     "${SONDER_CARGO_EXECUTABLE}" build --release --package llguidance
        INSTALL_COMMAND   ""
        UPDATE_COMMAND    ""
        BUILD_BYPRODUCTS  "${_sonder_llg_lib}" "${_sonder_llg_out}/llguidance.h"
    )
    add_library(sonder_llguidance STATIC IMPORTED)
    set_target_properties(sonder_llguidance PROPERTIES IMPORTED_LOCATION "${_sonder_llg_lib}")
    add_dependencies(sonder_llguidance sonder_llguidance_ext)
    list(APPEND SONDER_LLAMACPP_FEATURE_DEFS SONDER_LLAMACPP_HAS_LLGUIDANCE=1)
    list(APPEND SONDER_LLAMACPP_PRIVATE_INCLUDES "${_sonder_llg_out}")
    list(APPEND SONDER_LLGUIDANCE_LINK_LIBS sonder_llguidance)
    if(WIN32)
        list(APPEND SONDER_LLGUIDANCE_LINK_LIBS ws2_32 userenv ntdll bcrypt)
    else()
        find_package(Threads REQUIRED)
        list(APPEND SONDER_LLGUIDANCE_LINK_LIBS Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
    message(STATUS "llguidance: ON (${SONDER_LLGUIDANCE_GIT_TAG}, cargo: ${SONDER_CARGO_EXECUTABLE})")
endif()
# --- llguidance end ---

add_library(sonder_backend_llamacpp STATIC
    device_health.cpp
    gguf_compat.cpp
    kv_policy.cpp
    llamacpp_backend.cpp
    llamacpp_util.cpp
    pq2_reblock.cpp
    lora_registry.cpp
    llguidance_sampler.cpp
    embeddings.cpp
    kv_telemetry.cpp
    prefix_checkpoints.cpp
    graph_telemetry.cpp
    slot_store.cpp
    spec_controller.cpp
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
target_link_libraries(sonder_backend_llamacpp PRIVATE llama ggml ${SONDER_LLAMACPP_EXTRA_LIBS} ${SONDER_LLGUIDANCE_LINK_LIBS})
if(SONDER_WITH_LLGUIDANCE)
    add_dependencies(sonder_backend_llamacpp sonder_llguidance_ext)
endif()
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
sonder_module_sources(${CMAKE_CURRENT_SOURCE_DIR}/tool_grammar_preflight.cpp)
sonder_module_sources(${CMAKE_CURRENT_SOURCE_DIR}/json_schema_normalize.cpp)
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
