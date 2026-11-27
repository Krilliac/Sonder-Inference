# Sonder Inference - CUDA build matrix configure checks and SASS verifier (MAIN-03).
#
# Two modes in one module:
#
#   1. Project configure mode:
#        include(SonderCudaCheck)
#        sonder_check_cuda_config()
#      Validates when GGML_CUDA=ON (before FetchContent_MakeAvailable(llamacpp)):
#        - CUDA Toolkit >= 13.4 (13.5 preferred; override with -DSONDER_ALLOW_LEGACY_CUDA=ON,
#          subject to hard compiler/ISA floors >= 12.8 for 120a and >= 12.9 for 120f, and
#          rejecting SM < 75 removed in CUDA 13.0).
#        - CMAKE_CUDA_ARCHITECTURES requires architecture- or family-specific Blackwell
#          targets (120a or 120f) so block-scaled FP4 MMA (.kind::mxf4 / .kind::mxf4nvf4)
#          assembles; bare 120/121 targets are refused.
#        - CMake version compatibility for 'f' family-specific targets (CMake < 3.31.8 and
#          4.0.0..4.0.1 reject the 'f' suffix in CMakeCUDAArchitecturesValidate.cmake).
#        - NVIDIA bug CUB-10445 guard: refuses cuBLAS 13.6.0 (CUDA 13.3 Update 1; fixed in
#          cuBLAS patch 13.6.1 and CUDA >= 13.4) via both cublas_api.h and toolkit version.json.
#        - GGML_CUDA_GRAPHS defaults to ON when unset and warns if explicitly disabled.
#
#   2. Script mode (-P) for post-build native SASS verification:
#        cmake -DSONDER_CUDA_VERIFY_SASS=ON -DTARGET_BINARY=<cubin|obj|lib|exe> \
#              [-DCUOBJDUMP_EXECUTABLE=<path>] [-DREQUIRE_FP4_MMA=ON|OFF] \
#              -P cmake/SonderCudaCheck.cmake
#      Runs `cuobjdump --dump-sass` and verifies native Blackwell SASS ('code for sm_120a'
#      on 120a builds or 'code for sm_120' on 120f builds) plus block-scaled FP4 tensor-core
#      MMA instructions (OMMA.SF.16864.F32.E2M1.E2M1) when REQUIRE_FP4_MMA=ON (the default).

if(DEFINED SONDER_CUDA_VERIFY_SASS AND NOT "${SONDER_CUDA_VERIFY_SASS}" STREQUAL "")
    string(TOUPPER "${SONDER_CUDA_VERIFY_SASS}" _sonder_verify_sass_upper)
    if(NOT _sonder_verify_sass_upper MATCHES "^(ON|OFF|TRUE|FALSE|1|0|YES|NO|Y|N)$")
        message(FATAL_ERROR
            "Sonder SASS check: invalid boolean value '${SONDER_CUDA_VERIFY_SASS}' for SONDER_CUDA_VERIFY_SASS")
    endif()
endif()

if(SONDER_CUDA_VERIFY_SASS)
    if(NOT DEFINED TARGET_BINARY OR "${TARGET_BINARY}" STREQUAL "")
        message(FATAL_ERROR
            "Sonder SASS check: pass -DTARGET_BINARY=<path-to-cubin-obj-lib-or-exe>")
    endif()
    if(NOT EXISTS "${TARGET_BINARY}" OR IS_DIRECTORY "${TARGET_BINARY}")
        message(FATAL_ERROR
            "Sonder SASS check: TARGET_BINARY does not exist or is a directory: ${TARGET_BINARY}")
    endif()

    if(NOT DEFINED REQUIRE_FP4_MMA OR "${REQUIRE_FP4_MMA}" STREQUAL "")
        set(REQUIRE_FP4_MMA ON)
    else()
        string(TOUPPER "${REQUIRE_FP4_MMA}" _sonder_req_fp4_upper)
        if(NOT _sonder_req_fp4_upper MATCHES "^(ON|OFF|TRUE|FALSE|1|0|YES|NO|Y|N)$")
            message(FATAL_ERROR
                "Sonder SASS check: invalid boolean value '${REQUIRE_FP4_MMA}' for REQUIRE_FP4_MMA")
        endif()
    endif()

    set(_sonder_cuobjdump "")
    if(DEFINED CUOBJDUMP_EXECUTABLE AND NOT "${CUOBJDUMP_EXECUTABLE}" STREQUAL "")
        if(NOT EXISTS "${CUOBJDUMP_EXECUTABLE}" OR IS_DIRECTORY "${CUOBJDUMP_EXECUTABLE}")
            message(FATAL_ERROR
                "Sonder SASS check: CUOBJDUMP_EXECUTABLE not found or is a directory: ${CUOBJDUMP_EXECUTABLE}")
        endif()
        set(_sonder_cuobjdump "${CUOBJDUMP_EXECUTABLE}")
    else()
        set(_sonder_cuda_bin_hints "")
        if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
            list(APPEND _sonder_cuda_bin_hints "$ENV{CUDA_PATH}/bin")
        endif()
        find_program(_sonder_cuobjdump NAMES cuobjdump HINTS ${_sonder_cuda_bin_hints})
        if(NOT _sonder_cuobjdump)
            message(FATAL_ERROR
                "Sonder SASS check: cuobjdump not found (pass -DCUOBJDUMP_EXECUTABLE=<path> or set CUDA_PATH)")
        endif()
    endif()

    execute_process(
        COMMAND "${_sonder_cuobjdump}" --dump-sass "${TARGET_BINARY}"
        RESULT_VARIABLE _sonder_sass_rc
        OUTPUT_VARIABLE _sonder_sass_out
        ERROR_VARIABLE _sonder_sass_err
        TIMEOUT 120)
    if(NOT _sonder_sass_rc EQUAL 0)
        message(FATAL_ERROR
            "Sonder SASS check: cuobjdump failed (exit ${_sonder_sass_rc}) on '${TARGET_BINARY}': ${_sonder_sass_err}")
    endif()
    if("${_sonder_sass_out}" STREQUAL "")
        message(FATAL_ERROR
            "Sonder SASS check: cuobjdump produced empty SASS output for '${TARGET_BINARY}' "
            "(binary contains no embedded cubin SASS).")
    endif()
    # 120a cubins emit 'code for sm_120a'; 120f family cubins emit 'code for sm_120'.
    if(NOT _sonder_sass_out MATCHES "code for sm_12[0-9]a?([\r\n \t]|$)")
        message(FATAL_ERROR
            "Sonder SASS check: '${TARGET_BINARY}' contains no native Blackwell SASS "
            "('code for sm_120a' or 'code for sm_120'); ensure CMAKE_CUDA_ARCHITECTURES includes "
            "120a or 120f rather than PTX-only or pre-Blackwell targets.")
    endif()
    if(REQUIRE_FP4_MMA)
        if(NOT _sonder_sass_out MATCHES "OMMA\\.SF\\.16864\\.F32\\.E2M1\\.E2M1")
            message(FATAL_ERROR
                "Sonder SASS check: '${TARGET_BINARY}' contains Blackwell SASS, but no "
                "block-scaled FP4 MMA instruction (OMMA.SF.16864.F32.E2M1.E2M1) was found.")
        endif()
    endif()

    message(STATUS
        "Sonder SASS check: ok (${TARGET_BINARY}: native Blackwell SASS verified, "
        "REQUIRE_FP4_MMA=${REQUIRE_FP4_MMA})")
    return()
endif()

option(SONDER_ALLOW_LEGACY_CUDA
    "Allow CUDA Toolkits older than 13.4 when GGML_CUDA=ON (not recommended for Blackwell sm_120)"
    OFF)

function(sonder_check_cuda_config)
    # 1. Default GGML_CUDA_GRAPHS=ON when unset; warn if explicitly disabled.
    if(NOT DEFINED GGML_CUDA_GRAPHS AND NOT DEFINED CACHE{GGML_CUDA_GRAPHS})
        set(GGML_CUDA_GRAPHS ON CACHE BOOL "ggml: use CUDA graphs (llama.cpp only)" FORCE)
        set(GGML_CUDA_GRAPHS ON PARENT_SCOPE)
        set(_eff_graphs ON)
    elseif(GGML_CUDA_GRAPHS)
        set(_eff_graphs ON)
    else()
        set(_eff_graphs OFF)
        message(WARNING
            "Sonder CUDA check: GGML_CUDA_GRAPHS is OFF; disabling CUDA graph capture "
            "degrades Blackwell and Windows WDDM decode throughput.")
    endif()

    # 2. Validate CMAKE_CUDA_ARCHITECTURES before enable_language(CUDA) runs.
    set(_has_blackwell_arch OFF)
    set(_has_arch_a OFF)
    set(_has_arch_f OFF)
    set(_has_special_arch OFF)
    set(_special_name "")
    set(_numeric_sms "")

    if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
        if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "(\\.\\.|[/\\\\])")
            message(FATAL_ERROR
                "Sonder CUDA check: unsafe path characters in CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'.")
        endif()
        if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
            message(FATAL_ERROR
                "Sonder CUDA check: CMAKE_CUDA_ARCHITECTURES must not be empty when set.")
        endif()

        set(_arch_count 0)
        foreach(_raw_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if("${_raw_arch}" STREQUAL "")
                continue()
            endif()
            math(EXPR _arch_count "${_arch_count} + 1")
            string(STRIP "${_raw_arch}" _arch)
            if("${_arch}" STREQUAL "" OR NOT "${_arch}" STREQUAL "${_raw_arch}")
                message(FATAL_ERROR
                    "Sonder CUDA check: invalid CUDA architecture '${_raw_arch}' in CMAKE_CUDA_ARCHITECTURES.")
            endif()

            if(_arch MATCHES "^(native|all|all-major)$")
                set(_has_special_arch ON)
                set(_special_name "${_arch}")
            elseif(_arch MATCHES "^([1-9][0-9]*)(a|f)?(-real|-virtual)?$")
                set(_sm_num "${CMAKE_MATCH_1}")
                set(_sm_sfx "${CMAKE_MATCH_2}")
                if(_sm_num LESS 50 OR _sm_num GREATER 999)
                    message(FATAL_ERROR
                        "Sonder CUDA check: out-of-range CUDA architecture '${_arch}' (expected SM 50..999).")
                endif()
                list(APPEND _numeric_sms "${_sm_num}")

                if(_sm_num GREATER_EQUAL 120 AND _sm_num LESS 130)
                    if("${_sm_sfx}" STREQUAL "")
                        message(FATAL_ERROR
                            "Sonder CUDA check: CMAKE_CUDA_ARCHITECTURES contains bare Blackwell target '${_arch}'. "
                            "Block-scaled FP4 MMA (.kind::mxf4 / .kind::mxf4nvf4) fails to assemble on plain "
                            "sm_${_sm_num}; use architecture-specific '${_sm_num}a' (non-portable cubin) or "
                            "family-specific '${_sm_num}f' (requires nvcc >= 12.9 and CMake >= 3.31.8 or >= 4.0.2).")
                    endif()
                    set(_has_blackwell_arch ON)
                endif()

                if("${_sm_sfx}" STREQUAL "a")
                    set(_has_arch_a ON)
                elseif("${_sm_sfx}" STREQUAL "f")
                    set(_has_arch_f ON)
                endif()
            else()
                message(FATAL_ERROR
                    "Sonder CUDA check: invalid CUDA architecture '${_arch}' in CMAKE_CUDA_ARCHITECTURES.")
            endif()
        endforeach()

        if(_arch_count EQUAL 0)
            message(FATAL_ERROR
                "Sonder CUDA check: CMAKE_CUDA_ARCHITECTURES must not be empty when set.")
        endif()
        if(_has_special_arch AND _arch_count GREATER 1)
            message(FATAL_ERROR
                "Sonder CUDA check: special target '${_special_name}' cannot be combined with explicit "
                "architectures in CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'.")
        endif()

        set(_eff_cmake_ver "${CMAKE_VERSION}")
        if(DEFINED SONDER_CUDA_CHECK_CMAKE_VERSION AND NOT "${SONDER_CUDA_CHECK_CMAKE_VERSION}" STREQUAL "")
            if(NOT "${SONDER_CUDA_CHECK_CMAKE_VERSION}" MATCHES "^[0-9]+\\.[0-9]+(\\.[0-9]+)?")
                message(FATAL_ERROR
                    "Sonder CUDA check: malformed SONDER_CUDA_CHECK_CMAKE_VERSION='${SONDER_CUDA_CHECK_CMAKE_VERSION}'.")
            endif()
            set(_eff_cmake_ver "${SONDER_CUDA_CHECK_CMAKE_VERSION}")
        endif()

        if(_has_arch_f)
            if(_eff_cmake_ver VERSION_LESS "3.31.8"
               OR (_eff_cmake_ver VERSION_GREATER_EQUAL "4.0.0" AND _eff_cmake_ver VERSION_LESS "4.0.2"))
                message(FATAL_ERROR
                    "Sonder CUDA check: CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}' uses a "
                    "family-specific 'f' target, which CMake ${_eff_cmake_ver} rejects in "
                    "CMakeCUDAArchitecturesValidate.cmake. Upgrade CMake to >= 3.31.8 (< 4.0.0) or >= 4.0.2, "
                    "or use '120a' (windows-cuda-blackwell / linux-cuda-blackwell).")
            endif()
        endif()
    endif()

    # 3. Discover CUDA Toolkit (supports mock toolkit root seam for offline tests).
    set(_cuda_found OFF)
    set(_cuda_version "")
    set(_cuda_inc_dirs "")
    set(_cuda_root "")

    if(DEFINED SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT AND NOT "${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}" STREQUAL "")
        if("${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}" MATCHES "\\.\\.")
            message(FATAL_ERROR
                "Sonder CUDA check: unsafe SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT='${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}'.")
        endif()
        if(EXISTS "${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}" AND IS_DIRECTORY "${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}")
            set(_cuda_found ON)
            set(_cuda_root "${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}")
            set(_cuda_inc_dirs "${SONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT}/include")
            set(_cuda_version "${SONDER_CUDA_CHECK_MOCK_VERSION}")
        endif()
    else()
        string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _sonder_cuda_plat_lwr)
        if(MSVC AND _sonder_cuda_plat_lwr STREQUAL "arm64ec")
            find_library(CUDA_CUDART
                NAMES cudart
                HINTS ${CUDAToolkit_ROOT} ENV CUDA_PATH
                PATH_SUFFIXES lib/x64)
        endif()
        find_package(CUDAToolkit QUIET)
        set(_cuda_found "${CUDAToolkit_FOUND}")
        set(_cuda_version "${CUDAToolkit_VERSION}")
        set(_cuda_inc_dirs "${CUDAToolkit_INCLUDE_DIRS}")
        set(_cuda_root "${CUDAToolkit_LIBRARY_ROOT}")
        if("${_cuda_root}" STREQUAL "" AND _cuda_inc_dirs)
            list(GET _cuda_inc_dirs 0 _first_inc)
            get_filename_component(_cuda_root "${_first_inc}" DIRECTORY)
        endif()
    endif()

    if(NOT _cuda_found OR "${_cuda_version}" STREQUAL "")
        message(FATAL_ERROR
            "Sonder CUDA check: GGML_CUDA=ON requires the NVIDIA CUDA Toolkit (>= 13.4, 13.5 preferred), "
            "but CUDAToolkit was not found. Set CUDAToolkit_ROOT or CUDA_PATH, or build without GGML_CUDA=ON.")
    endif()
    if(NOT "${_cuda_version}" MATCHES "^[0-9]+\\.[0-9]+(\\.[0-9]+)*$")
        message(FATAL_ERROR
            "Sonder CUDA check: malformed CUDA Toolkit version '${_cuda_version}'.")
    endif()

    # 4. Enforce hard ISA/compiler floors and the >= 13.4 (13.5 preferred) policy floor.
    if(_has_arch_f AND _cuda_version VERSION_LESS "12.9")
        message(FATAL_ERROR
            "Sonder CUDA check: family-specific 'f' CUDA architectures (${CMAKE_CUDA_ARCHITECTURES}) "
            "require nvcc / CUDA Toolkit >= 12.9 (PTX ISA 8.8); found ${_cuda_version}.")
    endif()
    if((_has_blackwell_arch OR _has_arch_a) AND _cuda_version VERSION_LESS "12.8")
        message(FATAL_ERROR
            "Sonder CUDA check: Blackwell / 'a'-suffixed CUDA targets (${CMAKE_CUDA_ARCHITECTURES}) "
            "require CUDA Toolkit >= 12.8; found ${_cuda_version}.")
    endif()
    if(_cuda_version VERSION_GREATER_EQUAL "13.0")
        foreach(_sm IN LISTS _numeric_sms)
            if(_sm LESS 75)
                message(FATAL_ERROR
                    "Sonder CUDA check: offline compilation for sm_${_sm} was removed in CUDA 13 "
                    "(found CUDA ${_cuda_version}; minimum SM target in CUDA 13 is 75).")
            endif()
        endforeach()
    endif()

    if(_cuda_version VERSION_LESS "13.4")
        if(NOT SONDER_ALLOW_LEGACY_CUDA)
            message(FATAL_ERROR
                "Sonder CUDA check: CUDA Toolkit ${_cuda_version} is older than required 13.4 "
                "(13.5 preferred; cc 12.x cuBLAS BIAS-epilogue and CUDA-graph-capture fixes landed in 13.4). "
                "Upgrade the CUDA Toolkit, or pass -DSONDER_ALLOW_LEGACY_CUDA=ON on legacy hosts.")
        else()
            message(WARNING
                "Sonder CUDA check: allowing legacy CUDA Toolkit ${_cuda_version} (< 13.4) "
                "because SONDER_ALLOW_LEGACY_CUDA=ON.")
        endif()
    endif()

    # 5. Check cuBLAS version for CUB-10445 (cuBLAS 13.6.0 in CUDA 13.3 Update 1; fixed in 13.6.1+).
    set(_cublas_h "")
    foreach(_inc IN LISTS _cuda_inc_dirs)
        if(EXISTS "${_inc}/cublas_api.h" AND NOT IS_DIRECTORY "${_inc}/cublas_api.h")
            set(_cublas_h "${_inc}/cublas_api.h")
            break()
        endif()
    endforeach()
    if("${_cublas_h}" STREQUAL "")
        message(FATAL_ERROR
            "Sonder CUDA check: cublas_api.h not found in CUDA include directories (${_cuda_inc_dirs}); "
            "cannot verify cuBLAS version against CUB-10445.")
    endif()

    file(READ "${_cublas_h}" _cublas_h_text)
    string(REGEX MATCH "(^|\n)[ \t]*#[ \t]*define[ \t]+CUBLAS_VER_MAJOR[ \t]+([0-9]+)[ \t]*(\r?\n|$)"
           _m_maj "${_cublas_h_text}")
    set(_cb_major "${CMAKE_MATCH_2}")
    string(REGEX MATCH "(^|\n)[ \t]*#[ \t]*define[ \t]+CUBLAS_VER_MINOR[ \t]+([0-9]+)[ \t]*(\r?\n|$)"
           _m_min "${_cublas_h_text}")
    set(_cb_minor "${CMAKE_MATCH_2}")
    string(REGEX MATCH "(^|\n)[ \t]*#[ \t]*define[ \t]+CUBLAS_VER_PATCH[ \t]+([0-9]+)[ \t]*(\r?\n|$)"
           _m_pat "${_cublas_h_text}")
    set(_cb_patch "${CMAKE_MATCH_2}")
    string(REGEX MATCH "(^|\n)[ \t]*#[ \t]*define[ \t]+CUBLAS_VER_BUILD[ \t]+([0-9]+)[ \t]*(\r?\n|$)"
           _m_bld "${_cublas_h_text}")
    set(_cb_build "${CMAKE_MATCH_2}")

    if("${_cb_major}" STREQUAL "" OR "${_cb_minor}" STREQUAL "" OR "${_cb_patch}" STREQUAL "")
        message(FATAL_ERROR
            "Sonder CUDA check: failed to parse CUBLAS_VER_MAJOR/MINOR/PATCH from '${_cublas_h}'.")
    endif()

    set(_cublas_semver "${_cb_major}.${_cb_minor}.${_cb_patch}")
    if(NOT "${_cb_build}" STREQUAL "")
        set(_cublas_full "${_cublas_semver}.${_cb_build}")
    else()
        set(_cublas_full "${_cublas_semver}")
    endif()

    if("${_cublas_semver}" STREQUAL "13.6.0")
        message(FATAL_ERROR
            "Sonder CUDA check: cuBLAS ${_cublas_full} (from '${_cublas_h}') is affected by NVIDIA bug "
            "CUB-10445 (wrong GEMV results on Blackwell cc 12.0/12.1 when transposed, non-accumulation "
            "dim > 3145680, beta != 0; shipped in CUDA 13.3 Update 1). Install cuBLAS patch >= 13.6.1 "
            "or upgrade to CUDA >= 13.4 (13.5 preferred).")
    endif()

    # Cross-check toolkit version.json when present (catches header/DLL patch mismatches).
    set(_vj_candidates "")
    if(NOT "${_cuda_root}" STREQUAL "")
        list(APPEND _vj_candidates "${_cuda_root}/version.json")
    endif()
    get_filename_component(_inc_parent "${_cublas_h}" DIRECTORY)
    get_filename_component(_inc_grandparent "${_inc_parent}" DIRECTORY)
    if(NOT "${_inc_grandparent}" STREQUAL "")
        list(APPEND _vj_candidates "${_inc_grandparent}/version.json")
    endif()
    list(REMOVE_DUPLICATES _vj_candidates)

    foreach(_vj_path IN LISTS _vj_candidates)
        if(EXISTS "${_vj_path}" AND NOT IS_DIRECTORY "${_vj_path}")
            file(READ "${_vj_path}" _vj_text)
            string(JSON _vj_root_type ERROR_VARIABLE _vj_parse_err TYPE "${_vj_text}")
            if(_vj_parse_err)
                message(FATAL_ERROR
                    "Sonder CUDA check: malformed toolkit version.json ('${_vj_path}'): ${_vj_parse_err}")
            endif()
            string(JSON _vj_cb_ver ERROR_VARIABLE _vj_cb_err GET "${_vj_text}" libcublas version)
            if(NOT _vj_cb_err AND NOT "${_vj_cb_ver}" STREQUAL "")
                if(_vj_cb_ver MATCHES "^13\\.6\\.0(\\.|$)")
                    message(FATAL_ERROR
                        "Sonder CUDA check: toolkit version.json ('${_vj_path}') reports libcublas "
                        "${_vj_cb_ver}, which is affected by NVIDIA bug CUB-10445. Install cuBLAS patch "
                        ">= 13.6.1 or upgrade to CUDA >= 13.4 (13.5 preferred).")
                endif()
                if(NOT _vj_cb_ver MATCHES "^${_cb_major}\\.${_cb_minor}\\.${_cb_patch}(\\.|$)")
                    message(WARNING
                        "Sonder CUDA check: cublas_api.h reports ${_cublas_full} while '${_vj_path}' "
                        "reports libcublas ${_vj_cb_ver}; verify deployed cublas shared libraries.")
                endif()
            endif()
            break()
        endif()
    endforeach()

    if(_cuda_version VERSION_GREATER_EQUAL "13.5")
        set(_pref_note "preferred >= 13.5 met")
    else()
        set(_pref_note "13.5+ preferred")
    endif()

    set(SONDER_CUDA_CHECK_PASSED ON CACHE INTERNAL "Sonder CUDA configure check passed")
    set(SONDER_CUDA_DETECTED_CUBLAS "${_cublas_full}" CACHE INTERNAL "Detected cuBLAS version")
    message(STATUS
        "Sonder CUDA check: CUDA=${_cuda_version} (${_pref_note}) "
        "cuBLAS=${_cublas_full} (CUB-10445 clean) "
        "archs=${CMAKE_CUDA_ARCHITECTURES} graphs=${_eff_graphs}")
endfunction()
