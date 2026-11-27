# Configure-time verifier and applier for Sonder Inference's owned ggml/llama.cpp
# patch series (third_party/ggml-patches/, declared in cmake/llamacpp-pin.json).
#
# Inputs:
#   SONDER_LLAMACPP_PIN_FILE            path to cmake/llamacpp-pin.json
#   SONDER_LLAMACPP_REPO_ROOT           path to Sonder-Inference repo root
#   SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH path to extracted llama.cpp tree
#                                       (defaults to CMAKE_CURRENT_BINARY_DIR,
#                                       which is WORKING_DIRECTORY under
#                                       FetchContent PATCH_COMMAND)

cmake_minimum_required(VERSION 3.20)

if(NOT SONDER_LLAMACPP_PIN_FILE OR NOT EXISTS "${SONDER_LLAMACPP_PIN_FILE}")
    message(FATAL_ERROR "ApplyGgmlPatches: SONDER_LLAMACPP_PIN_FILE missing or not found: '${SONDER_LLAMACPP_PIN_FILE}'")
endif()
if(NOT SONDER_LLAMACPP_REPO_ROOT OR NOT IS_DIRECTORY "${SONDER_LLAMACPP_REPO_ROOT}")
    message(FATAL_ERROR "ApplyGgmlPatches: SONDER_LLAMACPP_REPO_ROOT is not a directory: '${SONDER_LLAMACPP_REPO_ROOT}'")
endif()
if(NOT SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH)
    set(SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH "${CMAKE_CURRENT_BINARY_DIR}")
endif()
if(NOT IS_DIRECTORY "${SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH}")
    message(FATAL_ERROR "ApplyGgmlPatches: target source dir does not exist: '${SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH}'")
endif()

file(READ "${SONDER_LLAMACPP_PIN_FILE}" _pin_json)

string(JSON _tracks_len ERROR_VARIABLE _err LENGTH "${_pin_json}" tracks)
if(_err OR NOT _tracks_len EQUAL 1)
    message(FATAL_ERROR "ApplyGgmlPatches: ${SONDER_LLAMACPP_PIN_FILE} must define exactly one track ('mainline')")
endif()
string(JSON _entry ERROR_VARIABLE _err GET "${_pin_json}" tracks mainline)
if(_err)
    message(FATAL_ERROR "ApplyGgmlPatches: missing tracks.mainline in ${SONDER_LLAMACPP_PIN_FILE}: ${_err}")
endif()

set(_patch_dir_rel "third_party/ggml-patches")
string(JSON _has_patch_dir ERROR_VARIABLE _pd_err GET "${_entry}" patch_dir)
if(NOT _pd_err)
    set(_patch_dir_rel "${_has_patch_dir}")
endif()

if("${_patch_dir_rel}" STREQUAL ""
   OR "${_patch_dir_rel}" MATCHES "^/"
   OR "${_patch_dir_rel}" MATCHES "\\\\"
   OR "${_patch_dir_rel}" MATCHES ":"
   OR "${_patch_dir_rel}" MATCHES "(^|/)\\.\\.(/|$)"
   OR "${_patch_dir_rel}" MATCHES "(^|/)\\.(/|$)")
    message(FATAL_ERROR "ApplyGgmlPatches: unsafe patch_dir '${_patch_dir_rel}' in ${SONDER_LLAMACPP_PIN_FILE}")
endif()

set(_patch_dir "${SONDER_LLAMACPP_REPO_ROOT}/${_patch_dir_rel}")
if(NOT IS_DIRECTORY "${_patch_dir}")
    message(FATAL_ERROR "ApplyGgmlPatches: patch directory not found: ${_patch_dir}")
endif()

set(_patches_type "ARRAY")
set(_patches_len 0)
string(JSON _patches_type_raw ERROR_VARIABLE _p_err TYPE "${_entry}" patches)
if(NOT _p_err)
    if(NOT _patches_type_raw STREQUAL "ARRAY")
        message(FATAL_ERROR "ApplyGgmlPatches: tracks.mainline.patches must be an array (got ${_patches_type_raw})")
    endif()
    string(JSON _patches_len LENGTH "${_entry}" patches)
endif()

function(_sonder_check_rel_patch_target path_str patch_name out_clean_var)
    if("${path_str}" STREQUAL "/dev/null")
        set(${out_clean_var} "" PARENT_SCOPE)
        return()
    endif()
    set(_clean "${path_str}")
    if(_clean MATCHES "^[ab]/(.+)$")
        set(_clean "${CMAKE_MATCH_1}")
    endif()
    if("${_clean}" STREQUAL ""
       OR "${_clean}" MATCHES "^/"
       OR "${_clean}" MATCHES "\\\\"
       OR "${_clean}" MATCHES ":"
       OR "${_clean}" MATCHES "(^|/)\\.\\.(/|$)"
       OR "${_clean}" MATCHES "(^|/)\\.(/|$)"
       OR "${_clean}" MATCHES "^\\.git(/|$)")
        message(FATAL_ERROR "ApplyGgmlPatches: unsafe target path '${path_str}' in patch ${patch_name}")
    endif()
    set(${out_clean_var} "${_clean}" PARENT_SCOPE)
endfunction()

function(_sonder_inspect_patch_file patch_path patch_name out_targets_var)
    file(READ "${patch_path}" _raw)
    if("${_raw}" STREQUAL "")
        message(FATAL_ERROR "ApplyGgmlPatches: patch ${patch_name} is empty")
    endif()
    if(_raw MATCHES "GIT binary patch")
        message(FATAL_ERROR "ApplyGgmlPatches: binary patches are forbidden (${patch_name})")
    endif()
    if(_raw MATCHES "(^|\n)(new file mode|old mode|deleted file mode) 120000")
        message(FATAL_ERROR "ApplyGgmlPatches: symlink mode 120000 is forbidden (${patch_name})")
    endif()
    string(REPLACE "\r\n" "\n" _norm "${_raw}")
    # Escape semicolons AND square brackets before splitting lines into a CMake
    # list: in CMake lists, an unbalanced '[' suppresses ';' splitting across
    # subsequent lines until a matching ']'.
    string(REPLACE ";" "__SONDER_SEMI__" _escaped "${_norm}")
    string(REPLACE "[" "__SONDER_LBRACK__" _escaped "${_escaped}")
    string(REPLACE "]" "__SONDER_RBRACK__" _escaped "${_escaped}")
    string(REPLACE "\n" ";" _lines "${_escaped}")
    set(_saw_minus OFF)
    set(_saw_plus OFF)
    set(_saw_hunk OFF)
    set(_targets "")
    foreach(_line IN LISTS _lines)
        string(REPLACE "__SONDER_RBRACK__" "]" _line "${_line}")
        string(REPLACE "__SONDER_LBRACK__" "[" _line "${_line}")
        string(REPLACE "__SONDER_SEMI__" ";" _line "${_line}")
        if(_line MATCHES "^diff --git ([^ \t]+) ([^ \t]+)$")
            _sonder_check_rel_patch_target("${CMAKE_MATCH_1}" "${patch_name}" _d1)
            _sonder_check_rel_patch_target("${CMAKE_MATCH_2}" "${patch_name}" _d2)
        elseif(_line MATCHES "^--- ([^ \t]+)")
            set(_saw_minus ON)
            _sonder_check_rel_patch_target("${CMAKE_MATCH_1}" "${patch_name}" _m_clean)
            if(NOT "${_m_clean}" STREQUAL "")
                list(APPEND _targets "${_m_clean}")
            endif()
        elseif(_line MATCHES "^\\+\\+\\+ ([^ \t]+)")
            set(_saw_plus ON)
            _sonder_check_rel_patch_target("${CMAKE_MATCH_1}" "${patch_name}" _p_clean)
            if(NOT "${_p_clean}" STREQUAL "")
                list(APPEND _targets "${_p_clean}")
            endif()
        elseif(_line MATCHES "^@@ ")
            set(_saw_hunk ON)
        elseif(_line MATCHES "^(rename|copy) (from|to) (.+)$")
            _sonder_check_rel_patch_target("${CMAKE_MATCH_3}" "${patch_name}" _rc_clean)
            if(NOT "${_rc_clean}" STREQUAL "")
                list(APPEND _targets "${_rc_clean}")
            endif()
        endif()
    endforeach()
    if(NOT _saw_minus OR NOT _saw_plus)
        message(FATAL_ERROR "ApplyGgmlPatches: patch ${patch_name} contains no unified diff headers")
    endif()
    if(_targets)
        list(REMOVE_DUPLICATES _targets)
    endif()
    if(NOT _targets OR NOT _saw_hunk)
        message(FATAL_ERROR "ApplyGgmlPatches: patch ${patch_name} touches no target files or contains no hunks")
    endif()
    set(${out_targets_var} "${_targets}" PARENT_SCOPE)
endfunction()

set(_listed_files "")
set(_patch_abs_list "")
set(_expected_stamp "")
set(_prev_seq -1)

if(_patches_len GREATER 0)
    math(EXPR _last "${_patches_len} - 1")
    foreach(_i RANGE ${_last})
        string(JSON _item_type TYPE "${_entry}" patches ${_i})
        if(NOT _item_type STREQUAL "OBJECT")
            message(FATAL_ERROR "ApplyGgmlPatches: patches[${_i}] must be an object")
        endif()
        string(JSON _p_file ERROR_VARIABLE _e1 GET "${_entry}" patches ${_i} file)
        string(JSON _p_sha  ERROR_VARIABLE _e2 GET "${_entry}" patches ${_i} sha256)
        string(JSON _p_sum  ERROR_VARIABLE _e3 GET "${_entry}" patches ${_i} summary)
        string(JSON _p_up   ERROR_VARIABLE _e4 GET "${_entry}" patches ${_i} upstream)
        if(_e1 OR _e2 OR _e3 OR _e4 OR "${_p_sum}" STREQUAL "" OR "${_p_up}" STREQUAL "")
            message(FATAL_ERROR "ApplyGgmlPatches: patches[${_i}] requires non-empty file, sha256, summary, and upstream")
        endif()
        if(NOT _p_file MATCHES "^([0-9][0-9][0-9][0-9])-[a-z0-9][a-z0-9._-]*\\.patch$")
            message(FATAL_ERROR "ApplyGgmlPatches: invalid patch filename '${_p_file}' (expected NNNN-<slug>.patch)")
        endif()
        math(EXPR _seq "1${CMAKE_MATCH_1} - 10000")
        if(NOT _seq GREATER _prev_seq)
            message(FATAL_ERROR "ApplyGgmlPatches: patch sequence numbers must strictly increase (at '${_p_file}')")
        endif()
        set(_prev_seq ${_seq})
        if(NOT _p_sha MATCHES "^[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]$")
            message(FATAL_ERROR "ApplyGgmlPatches: patches[${_i}].sha256 is not 64 lowercase hex chars: '${_p_sha}'")
        endif()

        set(_p_abs "${_patch_dir}/${_p_file}")
        if(NOT EXISTS "${_p_abs}" OR IS_DIRECTORY "${_p_abs}")
            message(FATAL_ERROR "ApplyGgmlPatches: listed patch file missing: ${_p_abs}")
        endif()
        file(SHA256 "${_p_abs}" _actual_sha)
        if(NOT _actual_sha STREQUAL _p_sha)
            message(FATAL_ERROR "ApplyGgmlPatches: SHA-256 mismatch for ${_p_file}: got ${_actual_sha}, expected ${_p_sha}")
        endif()
        _sonder_inspect_patch_file("${_p_abs}" "${_p_file}" _p_targets)

        list(APPEND _listed_files "${_p_file}")
        list(APPEND _patch_abs_list "${_p_abs}")
        set("_patch_targets_${_i}" "${_p_targets}")
        string(APPEND _expected_stamp "${_p_file}:${_p_sha}\n")
    endforeach()
endif()

# Refuse any unlisted *.patch file in third_party/ggml-patches/.
file(GLOB _disk_patches LIST_DIRECTORIES false RELATIVE "${_patch_dir}" "${_patch_dir}/*.patch")
foreach(_dp IN LISTS _disk_patches)
    if(NOT "${_dp}" IN_LIST _listed_files)
        message(FATAL_ERROR "ApplyGgmlPatches: unlisted patch file '${_dp}' in ${_patch_dir} (register it with SHA-256 in ${SONDER_LLAMACPP_PIN_FILE})")
    endif()
endforeach()

set(_stamp_file "${SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH}/.sonder-patches-applied")
if(_patches_len EQUAL 0)
    if(EXISTS "${_stamp_file}")
        message(FATAL_ERROR "ApplyGgmlPatches: target tree ${SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH} has previously applied patches, "
                            "but the manifest patch list is now empty; clean the build _deps/llamacpp-src directory.")
    endif()
    message(STATUS "llama.cpp patches: 0 patches in ${_patch_dir_rel} (clean upstream pin)")
    return()
endif()

find_program(_git_exe git REQUIRED)

if(EXISTS "${_stamp_file}")
    file(READ "${_stamp_file}" _current_stamp)
    if(_current_stamp STREQUAL _expected_stamp)
        message(STATUS "llama.cpp patches: ${_patches_len} patch(es) already applied and verified")
        return()
    endif()
    message(FATAL_ERROR "ApplyGgmlPatches: applied patch stamp in ${SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH} differs from "
                        "${SONDER_LLAMACPP_PIN_FILE}; clean _deps/llamacpp-src to re-apply from the pristine archive.")
endif()

# Isolate git apply from any enclosing git repository (e.g. when _deps/llamacpp-src
# lives inside ${sourceDir}/build/<preset>/_deps/llamacpp-src). Without a ceiling,
# git discovers the parent .git, interprets patch paths relative to the outer repo
# root, silently skips every file outside cwd, and exits 0.
get_filename_component(_target_abs "${SONDER_LLAMACPP_SOURCE_DIR_TO_PATCH}" ABSOLUTE)
get_filename_component(_ceiling_abs "${_target_abs}" DIRECTORY)
file(TO_NATIVE_PATH "${_ceiling_abs}" _ceiling_native)

function(_sonder_target_file_state full_path out_var)
    if(EXISTS "${full_path}" AND NOT IS_DIRECTORY "${full_path}")
        file(SHA256 "${full_path}" _digest)
        set(${out_var} "FILE:${_digest}" PARENT_SCOPE)
    else()
        set(${out_var} "MISSING" PARENT_SCOPE)
    endif()
endfunction()

math(EXPR _apply_last "${_patches_len} - 1")
foreach(_idx RANGE ${_apply_last})
    list(GET _patch_abs_list ${_idx} _p_abs)
    get_filename_component(_p_name "${_p_abs}" NAME)
    set(_cur_targets "${_patch_targets_${_idx}}")
    if(NOT _cur_targets)
        message(FATAL_ERROR "ApplyGgmlPatches: internal error: patch ${_p_name} has 0 target files")
    endif()

    set(_before_states "")
    foreach(_rel IN LISTS _cur_targets)
        _sonder_target_file_state("${_target_abs}/${_rel}" _st_before)
        list(APPEND _before_states "${_st_before}")
    endforeach()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${_ceiling_native}"
                "${_git_exe}" -c core.autocrlf=false -c core.safecrlf=false
                apply --check -v -p1 --whitespace=nowarn "${_p_abs}"
        WORKING_DIRECTORY "${_target_abs}"
        RESULT_VARIABLE _chk_rc
        OUTPUT_VARIABLE _chk_out
        ERROR_VARIABLE _chk_err
    )
    set(_chk_log "${_chk_out}\n${_chk_err}")
    if(NOT _chk_rc EQUAL 0)
        message(FATAL_ERROR "ApplyGgmlPatches: git apply --check failed for ${_p_name}: ${_chk_err}")
    endif()
    if(_chk_log MATCHES "Skipped patch" OR NOT _chk_log MATCHES "Checking patch ")
        message(FATAL_ERROR "ApplyGgmlPatches: git apply --check skipped or matched 0 files for ${_p_name}: ${_chk_log}")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${_ceiling_native}"
                "${_git_exe}" -c core.autocrlf=false -c core.safecrlf=false
                apply -v -p1 --whitespace=nowarn "${_p_abs}"
        WORKING_DIRECTORY "${_target_abs}"
        RESULT_VARIABLE _app_rc
        OUTPUT_VARIABLE _app_out
        ERROR_VARIABLE _app_err
    )
    set(_app_log "${_app_out}\n${_app_err}")
    if(NOT _app_rc EQUAL 0)
        message(FATAL_ERROR "ApplyGgmlPatches: git apply failed for ${_p_name}: ${_app_err}")
    endif()
    if(_app_log MATCHES "Skipped patch" OR NOT _app_log MATCHES "Applied patch ")
        message(FATAL_ERROR "ApplyGgmlPatches: git apply skipped or applied 0 files for ${_p_name}: ${_app_log}")
    endif()

    list(LENGTH _cur_targets _t_len)
    math(EXPR _t_last "${_t_len} - 1")
    foreach(_ti RANGE ${_t_last})
        list(GET _cur_targets ${_ti} _rel)
        list(GET _before_states ${_ti} _st_before)
        _sonder_target_file_state("${_target_abs}/${_rel}" _st_after)
        if("${_st_before}" STREQUAL "${_st_after}")
            message(FATAL_ERROR "ApplyGgmlPatches: patch ${_p_name} did not modify expected target '${_rel}' (fail-closed)")
        endif()
    endforeach()

    message(STATUS "llama.cpp patches: applied ${_p_name} (${_t_len} file(s))")
endforeach()

file(WRITE "${_stamp_file}" "${_expected_stamp}")
