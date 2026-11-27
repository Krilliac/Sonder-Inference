# Verifies MAIN-03 CUDA build matrix presets, configure-time CUDA / CUB-10445 /
# architecture checks (cmake/SonderCudaCheck.cmake), llamacpp CMake wiring, and
# cuobjdump SASS verification mode.
#
#   cmake -DSOURCE_DIR=<repo> -DWORK_DIR=<scratch> -P cuda_build_matrix_check.cmake
cmake_minimum_required(VERSION 3.21)

if(NOT SOURCE_DIR OR NOT WORK_DIR)
    message(FATAL_ERROR "cuda_build_matrix_check.cmake: pass -DSOURCE_DIR and -DWORK_DIR")
endif()

set(presets_file "${SOURCE_DIR}/CMakePresets.json")
set(cuda_check_module "${SOURCE_DIR}/cmake/SonderCudaCheck.cmake")
set(llamacpp_cmake "${SOURCE_DIR}/src/backends/llamacpp/CMakeLists.txt")

if(NOT EXISTS "${presets_file}")
    message(FATAL_ERROR "missing CMakePresets.json: ${presets_file}")
endif()
if(NOT EXISTS "${cuda_check_module}")
    message(FATAL_ERROR "missing cmake/SonderCudaCheck.cmake: ${cuda_check_module}")
endif()
if(NOT EXISTS "${llamacpp_cmake}")
    message(FATAL_ERROR "missing llamacpp CMakeLists.txt: ${llamacpp_cmake}")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# ---------------------------------------------------------------------------
# 1. CMakePresets.json matrix validation
# ---------------------------------------------------------------------------
file(READ "${presets_file}" presets_json)

function(find_named_preset array_field target_name out_obj)
    string(JSON len LENGTH "${presets_json}" "${array_field}")
    math(EXPR last "${len} - 1")
    foreach(i RANGE ${last})
        string(JSON item GET "${presets_json}" "${array_field}" ${i})
        string(JSON name GET "${item}" name)
        if(name STREQUAL target_name)
            set(${out_obj} "${item}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "CMakePresets.json: ${array_field} is missing '${target_name}'")
endfunction()

foreach(preset_name IN ITEMS
        windows-cuda-rtx
        windows-cuda-blackwell
        windows-cuda-blackwell-120f
        linux-cuda-rtx
        linux-cuda-blackwell
        linux-cuda-blackwell-120f)
    find_named_preset(configurePresets "${preset_name}" cfg_obj)
    find_named_preset(buildPresets "${preset_name}" bld_obj)
    find_named_preset(testPresets "${preset_name}" tst_obj)
    string(JSON bld_cfg GET "${bld_obj}" configurePreset)
    string(JSON tst_cfg GET "${tst_obj}" configurePreset)
    if(NOT bld_cfg STREQUAL preset_name OR NOT tst_cfg STREQUAL preset_name)
        message(FATAL_ERROR "build/test preset '${preset_name}' does not point to its configurePreset")
    endif()
endforeach()

foreach(bw_a IN ITEMS windows-cuda-blackwell linux-cuda-blackwell)
    find_named_preset(configurePresets "${bw_a}" obj)
    string(JSON arch GET "${obj}" cacheVariables CMAKE_CUDA_ARCHITECTURES)
    string(JSON graphs GET "${obj}" cacheVariables GGML_CUDA_GRAPHS)
    string(JSON fa GET "${obj}" cacheVariables GGML_CUDA_FA_ALL_QUANTS)
    string(JSON btype GET "${obj}" cacheVariables CMAKE_BUILD_TYPE)
    if(NOT arch STREQUAL "120a")
        message(FATAL_ERROR "${bw_a}: expected CMAKE_CUDA_ARCHITECTURES='120a', got '${arch}'")
    endif()
    if(NOT graphs STREQUAL "ON" OR NOT fa STREQUAL "ON" OR NOT btype STREQUAL "Release")
        message(FATAL_ERROR "${bw_a}: expected Release + GGML_CUDA_GRAPHS=ON + GGML_CUDA_FA_ALL_QUANTS=ON")
    endif()
endforeach()

foreach(bw_f IN ITEMS windows-cuda-blackwell-120f linux-cuda-blackwell-120f)
    find_named_preset(configurePresets "${bw_f}" obj)
    string(JSON arch GET "${obj}" cacheVariables CMAKE_CUDA_ARCHITECTURES)
    string(JSON parent GET "${obj}" inherits)
    if(NOT arch STREQUAL "120f")
        message(FATAL_ERROR "${bw_f}: expected CMAKE_CUDA_ARCHITECTURES='120f', got '${arch}'")
    endif()
    if(bw_f STREQUAL "windows-cuda-blackwell-120f" AND NOT parent STREQUAL "windows-cuda-blackwell")
        message(FATAL_ERROR "${bw_f}: expected inherits='windows-cuda-blackwell', got '${parent}'")
    endif()
    if(bw_f STREQUAL "linux-cuda-blackwell-120f" AND NOT parent STREQUAL "linux-cuda-blackwell")
        message(FATAL_ERROR "${bw_f}: expected inherits='linux-cuda-blackwell', got '${parent}'")
    endif()
endforeach()

foreach(rtx_p IN ITEMS windows-cuda-rtx linux-cuda-rtx)
    find_named_preset(configurePresets "${rtx_p}" obj)
    string(JSON arch GET "${obj}" cacheVariables CMAKE_CUDA_ARCHITECTURES)
    string(JSON graphs GET "${obj}" cacheVariables GGML_CUDA_GRAPHS)
    if(NOT arch STREQUAL "86;89;120a")
        message(FATAL_ERROR "${rtx_p}: expected CMAKE_CUDA_ARCHITECTURES='86;89;120a', got '${arch}'")
    endif()
    if(NOT graphs STREQUAL "ON")
        message(FATAL_ERROR "${rtx_p}: expected GGML_CUDA_GRAPHS='ON', got '${graphs}'")
    endif()
endforeach()

# Ensure no preset in CMakePresets.json uses bare 120/121 without a/f suffix.
string(JSON cfg_len LENGTH "${presets_json}" configurePresets)
math(EXPR cfg_last "${cfg_len} - 1")
foreach(i RANGE ${cfg_last})
    string(JSON item GET "${presets_json}" configurePresets ${i})
    string(JSON pname GET "${item}" name)
    string(JSON arch ERROR_VARIABLE arch_err GET "${item}" cacheVariables CMAKE_CUDA_ARCHITECTURES)
    if(NOT arch_err)
        foreach(tok IN LISTS arch)
            if(tok MATCHES "^12[0-9](-real|-virtual)?$")
                message(FATAL_ERROR "preset '${pname}' still contains bare Blackwell target '${tok}'")
            endif()
        endforeach()
    endif()
endforeach()

# Ensure Markdown docs agree with the 120a/120f preset matrix (no stale
# CMAKE_CUDA_ARCHITECTURES=86;89;120 or =120 without a/f suffix, and PLACEMENT.md
# documents 86;89;120a, sm_120a, -120f presets, and CUDA >= 13.4).
file(GLOB_RECURSE md_files LIST_DIRECTORIES false
    "${SOURCE_DIR}/README.md"
    "${SOURCE_DIR}/docs/*.md"
    "${SOURCE_DIR}/src/*.md")
foreach(md_file IN LISTS md_files)
    file(READ "${md_file}" md_text)
    if(md_text MATCHES "CMAKE_CUDA_ARCHITECTURES[ \t]*=[ \t]*\"?(86;89;120|120)([^0-9af]|$)")
        message(FATAL_ERROR "stale bare-120 CMAKE_CUDA_ARCHITECTURES claim in ${md_file}")
    endif()
endforeach()

set(placement_doc "${SOURCE_DIR}/docs/PLACEMENT.md")
file(READ "${placement_doc}" placement_text)
foreach(req_substr IN ITEMS
        "CMAKE_CUDA_ARCHITECTURES=86;89;120a"
        "sm_120a"
        "windows-cuda-blackwell-120f"
        "CUDA >= 13.4")
    string(FIND "${placement_text}" "${req_substr}" p_idx)
    if(p_idx EQUAL -1)
        message(FATAL_ERROR "docs/PLACEMENT.md is missing '${req_substr}'")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# 2. Mock CUDA toolkit helper + configure runner
# ---------------------------------------------------------------------------
function(write_mock_toolkit dir major minor patch build vj_ver)
    file(REMOVE_RECURSE "${dir}")
    file(MAKE_DIRECTORY "${dir}/include")
    set(hdr
        "#pragma once\n"
        "#define CUBLAS_VER_MAJOR ${major}\n"
        "#define CUBLAS_VER_MINOR ${minor}\n"
        "#define CUBLAS_VER_PATCH ${patch}\n"
        "#define CUBLAS_VER_BUILD ${build}\n")
    string(CONCAT hdr_text ${hdr})
    file(WRITE "${dir}/include/cublas_api.h" "${hdr_text}")
    if(NOT "${vj_ver}" STREQUAL "NONE")
        file(WRITE "${dir}/version.json"
            "{\n  \"libcublas\": { \"name\": \"CUDA cuBLAS\", \"version\": \"${vj_ver}\" }\n}\n")
    endif()
endfunction()

set(mini_src "${WORK_DIR}/mini_src")
file(MAKE_DIRECTORY "${mini_src}")
file(WRITE "${mini_src}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.21)\n"
    "project(sonder_cuda_check_mini LANGUAGES NONE)\n"
    "list(APPEND CMAKE_MODULE_PATH \"${SOURCE_DIR}/cmake\")\n"
    "include(SonderCudaCheck)\n"
    "sonder_check_cuda_config()\n"
    "message(STATUS \"MINI_GRAPHS=\${GGML_CUDA_GRAPHS}\")\n")

set(case_counter 0)
function(expect_configure expect_pass expected_substr)
    math(EXPR next_case "${case_counter} + 1")
    set(case_counter "${next_case}" PARENT_SCOPE)
    set(bdir "${WORK_DIR}/b_${next_case}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${mini_src}" -B "${bdir}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        TIMEOUT 60)
    set(combined "${out}\n${err}")
    string(REGEX REPLACE "[ \t\r\n]+" " " combined_norm "${combined}")
    if(expect_pass)
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR "case ${next_case} expected PASS (rc=0) with args [${ARGN}], got rc=${rc}:\n${combined}")
        endif()
    else()
        if(rc EQUAL 0)
            message(FATAL_ERROR "case ${next_case} expected FAIL (rc!=0) with args [${ARGN}], got rc=0:\n${combined}")
        endif()
    endif()
    if(NOT "${expected_substr}" STREQUAL "")
        if(NOT combined_norm MATCHES "${expected_substr}")
            message(FATAL_ERROR "case ${next_case} output did not match '${expected_substr}':\n${combined}")
        endif()
    endif()
endfunction()

set(tk_135 "${WORK_DIR}/tk_135")
write_mock_toolkit("${tk_135}" 13 10 0 0 "13.10.0.0")
set(tk_134 "${WORK_DIR}/tk_134")
write_mock_toolkit("${tk_134}" 13 7 0 27 "13.7.0.27")
set(tk_133_patched "${WORK_DIR}/tk_133_patched")
write_mock_toolkit("${tk_133_patched}" 13 6 1 5 "13.6.1.5")
set(tk_cub10445_hdr "${WORK_DIR}/tk_cub10445_hdr")
write_mock_toolkit("${tk_cub10445_hdr}" 13 6 0 2 "13.6.0.2")
set(tk_cub10445_vj "${WORK_DIR}/tk_cub10445_vj")
write_mock_toolkit("${tk_cub10445_vj}" 13 6 1 5 "13.6.0.2")

# Positive cases:
expect_configure(ON "preferred >= 13\\.5 met.*MINI_GRAPHS=ON"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DCMAKE_CUDA_ARCHITECTURES=120a")

expect_configure(ON "13\\.5\\+ preferred.*cuBLAS=13\\.7\\.0\\.27"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_134}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.1"
    "-DCMAKE_CUDA_ARCHITECTURES=86\\;89\\;120a"
    "-DGGML_CUDA_GRAPHS=ON")

expect_configure(ON "archs=120f"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_134}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.1"
    "-DSONDER_CUDA_CHECK_CMAKE_VERSION=3.31.8"
    "-DCMAKE_CUDA_ARCHITECTURES=120f")

expect_configure(ON "archs=120f"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_134}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.1"
    "-DSONDER_CUDA_CHECK_CMAKE_VERSION=4.0.2"
    "-DCMAKE_CUDA_ARCHITECTURES=120f")

expect_configure(ON "SONDER_ALLOW_LEGACY_CUDA=ON"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_133_patched}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.3.1"
    "-DSONDER_ALLOW_LEGACY_CUDA=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=120a")

# Negative cases:
expect_configure(OFF "older than required 13\\.4"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_133_patched}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.3.1"
    "-DCMAKE_CUDA_ARCHITECTURES=120a")

expect_configure(OFF "CUB-10445"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_cub10445_hdr}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.0"
    "-DCMAKE_CUDA_ARCHITECTURES=120a")

expect_configure(OFF "CUB-10445"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_cub10445_hdr}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.3.1"
    "-DSONDER_ALLOW_LEGACY_CUDA=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=120a")

expect_configure(OFF "CUB-10445"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_cub10445_vj}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.0"
    "-DCMAKE_CUDA_ARCHITECTURES=120a")

expect_configure(OFF "bare Blackwell target '120'"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DCMAKE_CUDA_ARCHITECTURES=120")

expect_configure(OFF "bare Blackwell target '120-real'"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DCMAKE_CUDA_ARCHITECTURES=120-real")

expect_configure(OFF "CMake 3\\.31\\.6 rejects"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DSONDER_CUDA_CHECK_CMAKE_VERSION=3.31.6"
    "-DCMAKE_CUDA_ARCHITECTURES=120f")

expect_configure(OFF "CMake 4\\.0\\.1 rejects"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DSONDER_CUDA_CHECK_CMAKE_VERSION=4.0.1"
    "-DCMAKE_CUDA_ARCHITECTURES=120f")

expect_configure(OFF ">= 12\\.9"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_134}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=12.8.1"
    "-DSONDER_ALLOW_LEGACY_CUDA=ON"
    "-DSONDER_CUDA_CHECK_CMAKE_VERSION=4.0.2"
    "-DCMAKE_CUDA_ARCHITECTURES=120f")

expect_configure(OFF "must not be empty"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DCMAKE_CUDA_ARCHITECTURES=")

expect_configure(OFF "invalid CUDA architecture"
    "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_135}"
    "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0"
    "-DCMAKE_CUDA_ARCHITECTURES=-1")

# ---------------------------------------------------------------------------
# 3. Verify src/backends/llamacpp/CMakeLists.txt invokes sonder_check_cuda_config()
#    before FetchContent_MakeAvailable(llamacpp)
# ---------------------------------------------------------------------------
file(READ "${llamacpp_cmake}" llamacpp_text)
string(FIND "${llamacpp_text}" "include(SonderCudaCheck)" idx_inc)
string(FIND "${llamacpp_text}" "sonder_check_cuda_config()" idx_call)
string(FIND "${llamacpp_text}" "FetchContent_MakeAvailable(llamacpp)" idx_fetch)
if(idx_inc EQUAL -1 OR idx_call EQUAL -1 OR idx_fetch EQUAL -1 OR NOT (idx_inc LESS idx_call AND idx_call LESS idx_fetch))
    message(FATAL_ERROR "src/backends/llamacpp/CMakeLists.txt must include(SonderCudaCheck) and call sonder_check_cuda_config() before FetchContent_MakeAvailable(llamacpp)")
endif()

# Extract the pre-FetchContent prefix up to `# --- pin resolution begin` and run
# it with GGML_CUDA=ON + CUB-10445 mock toolkit to prove the real CMakeLists fails closed.
string(FIND "${llamacpp_text}" "# --- pin resolution begin" idx_pin)
if(idx_pin EQUAL -1)
    message(FATAL_ERROR "pin resolution marker missing in src/backends/llamacpp/CMakeLists.txt")
endif()
string(SUBSTRING "${llamacpp_text}" 0 ${idx_pin} llamacpp_prefix)
set(hook_src "${WORK_DIR}/hook_src")
file(MAKE_DIRECTORY "${hook_src}")
file(WRITE "${hook_src}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.21)\n"
    "project(sonder_llamacpp_hook_mini LANGUAGES NONE)\n"
    "list(APPEND CMAKE_MODULE_PATH \"${SOURCE_DIR}/cmake\")\n"
    "set(SONDER_WITH_LLAMA_CPP ON)\n"
    "set(GGML_CUDA ON)\n"
    "${llamacpp_prefix}\n")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${hook_src}" -B "${WORK_DIR}/b_hook_bad"
        "-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT=${tk_cub10445_hdr}"
        "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.0"
        "-DCMAKE_CUDA_ARCHITECTURES=120a"
    RESULT_VARIABLE hook_bad_rc
    OUTPUT_VARIABLE hook_bad_out
    ERROR_VARIABLE hook_bad_err
    TIMEOUT 60)
if(hook_bad_rc EQUAL 0 OR NOT "${hook_bad_out}\n${hook_bad_err}" MATCHES "CUB-10445")
    message(FATAL_ERROR "llamacpp CMakeLists prefix failed to reject CUB-10445:\n${hook_bad_out}\n${hook_bad_err}")
endif()

# ---------------------------------------------------------------------------
# 4. SASS verification script mode (-DSONDER_CUDA_VERIFY_SASS=ON)
# ---------------------------------------------------------------------------
set(fake_bin "${WORK_DIR}/fake_target.cubin")
file(WRITE "${fake_bin}" "CUBIN")

function(write_fake_cuobjdump base_path sass_body exit_code out_exe)
    set(sass_txt "${base_path}.sass.txt")
    file(WRITE "${sass_txt}" "${sass_body}")
    if(WIN32)
        file(TO_NATIVE_PATH "${sass_txt}" sass_native)
        set(script_path "${base_path}.cmd")
        file(WRITE "${script_path}"
            "@echo off\r\n"
            "type \"${sass_native}\"\r\n"
            "exit /b ${exit_code}\r\n")
    else()
        set(script_path "${base_path}.sh")
        file(WRITE "${script_path}"
            "#!/bin/sh\n"
            "cat \"${sass_txt}\"\n"
            "exit ${exit_code}\n")
        file(CHMOD "${script_path}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    endif()
    set(${out_exe} "${script_path}" PARENT_SCOPE)
endfunction()

write_fake_cuobjdump("${WORK_DIR}/cuobj_120a"
    "\tcode for sm_120a\n\t\tFunction : mul_mat_q\n\t/*0060*/ OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X R4, R4, R2, RZ, R8, R9, URZ ;\n"
    0 cuobj_120a)
write_fake_cuobjdump("${WORK_DIR}/cuobj_120f"
    "\tcode for sm_120\n\t\tFunction : mul_mat_q\n\t/*0060*/ OMMA.SF.16864.F32.E2M1.E2M1.UE8M0.2X R4, R4, R2, RZ, R8, R9, URZ ;\n"
    0 cuobj_120f)
write_fake_cuobjdump("${WORK_DIR}/cuobj_sm86"
    "\tcode for sm_86\n\t\tFunction : mul_mat_q\n\t/*0060*/ HMMA.16816.F32 R4, R4, R2, R4 ;\n"
    0 cuobj_sm86)
write_fake_cuobjdump("${WORK_DIR}/cuobj_no_omma"
    "\tcode for sm_120a\n\t\tFunction : k_add\n\t/*0010*/ FADD R4, R4, R5 ;\n"
    0 cuobj_no_omma)

function(expect_sass expect_pass expected_substr)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -DSONDER_CUDA_VERIFY_SASS=ON ${ARGN} -P "${cuda_check_module}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        TIMEOUT 60)
    set(combined "${out}\n${err}")
    string(REGEX REPLACE "[ \t\r\n]+" " " combined_norm "${combined}")
    if(expect_pass AND NOT rc EQUAL 0)
        message(FATAL_ERROR "SASS check expected PASS with [${ARGN}], got rc=${rc}:\n${combined}")
    endif()
    if(NOT expect_pass AND rc EQUAL 0)
        message(FATAL_ERROR "SASS check expected FAIL with [${ARGN}], got rc=0:\n${combined}")
    endif()
    if(NOT "${expected_substr}" STREQUAL "" AND NOT combined_norm MATCHES "${expected_substr}")
        message(FATAL_ERROR "SASS check output did not match '${expected_substr}':\n${combined}")
    endif()
endfunction()

expect_sass(ON "native Blackwell SASS verified"
    "-DTARGET_BINARY=${fake_bin}" "-DCUOBJDUMP_EXECUTABLE=${cuobj_120a}")
expect_sass(ON "native Blackwell SASS verified"
    "-DTARGET_BINARY=${fake_bin}" "-DCUOBJDUMP_EXECUTABLE=${cuobj_120f}")
expect_sass(OFF "contains no native Blackwell SASS"
    "-DTARGET_BINARY=${fake_bin}" "-DCUOBJDUMP_EXECUTABLE=${cuobj_sm86}")
expect_sass(OFF "no block-scaled FP4 MMA instruction"
    "-DTARGET_BINARY=${fake_bin}" "-DCUOBJDUMP_EXECUTABLE=${cuobj_no_omma}")
expect_sass(ON "REQUIRE_FP4_MMA=OFF"
    "-DTARGET_BINARY=${fake_bin}" "-DCUOBJDUMP_EXECUTABLE=${cuobj_no_omma}" "-DREQUIRE_FP4_MMA=OFF")
expect_sass(OFF "TARGET_BINARY"
    "-DTARGET_BINARY=${WORK_DIR}/does_not_exist.cubin" "-DCUOBJDUMP_EXECUTABLE=${cuobj_120a}")

message(STATUS "cuda_build_matrix_check: ok")
