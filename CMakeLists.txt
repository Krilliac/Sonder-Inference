cmake_minimum_required(VERSION 3.21)

# MSVC debug info: with a compiler launcher (sccache) the default /Zi funnels every
# compile into one shared target PDB, so parallel compiles fail with C1041, and sccache
# cannot package a separate PDB at all. /Z7 (Embedded) is what sccache can cache.
# The policy must be set here: after cmake_minimum_required(), which resets policies
# to 3.21 where CMP0141 is unset (OLD: /Zi baked into the default flags, variable
# ignored), and before project(), because CMP0141 is read when project() first enables
# the languages. CMake older than 3.25 has no CMP0141 and keeps /Zi. Non-MSVC
# toolchains ignore the variable.
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")

project(SonderInference
    VERSION 0.1.0
    DESCRIPTION "Sonder Inference runtime (policy layer over execution backends)"
    LANGUAGES C CXX)

# Release carries no debug info; only Debug and RelWithDebInfo embed it.
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SONDER_BUILD_TESTS "Build the Sonder Inference test suite" ${PROJECT_IS_TOP_LEVEL})
option(SONDER_BUILD_CLI "Build the sonder-infer CLI" ${PROJECT_IS_TOP_LEVEL})
option(SONDER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors for Sonder targets" OFF)
# Direct llama.cpp/GGML backend (src/backends/llamacpp). OFF by default: when ON,
# llama.cpp is fetched at a pinned tag and compiled (CPU; accelerators via GGML_*).
option(SONDER_WITH_LLAMA_CPP "Build the direct llama.cpp backend" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SonderWarnings)
include(SonderModules)
include(SonderBuildInfo)

add_library(sonder_inference
    src/common/error.cpp
    src/common/json.cpp
    src/common/log_queue.cpp
    src/common/diag_log.cpp
    src/common/sha256.cpp
    src/engine/sampling_config.cpp
    src/devices/device.cpp
    src/devices/coopmat_canary.cpp
    src/telemetry/telemetry.cpp
    src/sessions/session.cpp
    src/engine/engine.cpp
    src/engine/request_runtime.cpp
    src/engine/c_api.cpp
    src/backends/backend.cpp
    src/backends/mock_backend.cpp
    src/net/http_client.cpp
)
add_library(sonder::inference ALIAS sonder_inference)
# Handle-anchored no-follow file opens (src/common/safe_file.hpp), shared by the weight
# manifests (src/artifacts) and the Ollama model store (src/models).
if(WIN32)
    target_sources(sonder_inference PRIVATE src/common/safe_file_win32.cpp)
else()
    target_sources(sonder_inference PRIVATE src/common/safe_file_posix.cpp)
endif()

target_include_directories(sonder_inference
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(sonder_inference PUBLIC cxx_std_20)
target_compile_definitions(sonder_inference
    PRIVATE
        SONDER_INFERENCE_BUILDING
        SONDER_INFERENCE_VERSION="${PROJECT_VERSION}")
# The commit lives in one generated TU, not a compile definition (cmake/SonderBuildInfo.cmake).
sonder_add_build_commit(sonder_inference)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(sonder_inference PUBLIC SONDER_INFERENCE_SHARED)
endif()

find_package(Threads REQUIRED)
target_link_libraries(sonder_inference PUBLIC Threads::Threads)
if(WIN32)
    # ntdll: NtCreateFile for handle-relative opens (src/common/safe_file_win32.cpp).
    target_link_libraries(sonder_inference PRIVATE ws2_32 advapi32 ntdll)
    target_compile_definitions(sonder_inference PRIVATE _WIN32_WINNT=0x0A00)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(sonder_inference PRIVATE ${CMAKE_DL_LIBS})
endif()
sonder_set_warnings(sonder_inference)

if(SONDER_BUILD_CLI)
    add_executable(sonder-infer
        tools/sonder-infer/cli_runtime.cpp
        tools/sonder-infer/main.cpp)
    target_link_libraries(sonder-infer PRIVATE sonder::inference)
    sonder_set_warnings(sonder-infer)
endif()

if(SONDER_BUILD_TESTS)
    enable_testing()
    include(SonderDoctest)
    # Test-only seams in src/common/safe_file.* and src/models (a weak-identity switch, a
    # directory-entry counter, a hash-slot hook). They live in the library's own sources, so they
    # are compiled in whenever the test suite is built (SONDER_BUILD_TESTS defaults to ON for a
    # top-level build, release and CI presets included); a library built with
    # -DSONDER_BUILD_TESTS=OFF contains none of them. Store contents cannot reach them.
    target_compile_definitions(sonder_inference PUBLIC SONDER_INFERENCE_TEST_HOOKS=1)
endif()

# Optional modules, each owned by its own work stream (docs/MODULES.md).
# A module is enabled simply by its CMakeLists.txt existing; modules add
# sources/definitions to sonder_inference and register their own tests via
# sonder_add_module_tests(). Workers never need to edit this file.
set(SONDER_MODULE_DIRS
    src/cache
    src/scheduler
    src/sampling
    src/backends/llamacpp
    src/backends/ollama
    src/backends/llamaserver
    src/artifacts
    src/models
    src/worker
    src/server
    bench)
foreach(module_dir IN LISTS SONDER_MODULE_DIRS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${module_dir}/CMakeLists.txt")
        message(STATUS "Sonder module: ${module_dir}")
        add_subdirectory(${module_dir})
    endif()
endforeach()

# Optional TLS for the internal HTTP client (docs/integration/tls.md).
# SONDER_WITH_TLS defaults to OFF: nothing is compiled or linked then.
include(SonderTls)

# Optional in-tree libFuzzer targets (docs/integration/hardening.md). CI
# builds fuzz/ standalone instead, so this stays OFF by default.
# In standalone mode (`cmake -S fuzz`) fuzz/ is the top-level project and
# already adds this directory; adding fuzz/ again from here would recurse.
option(SONDER_BUILD_FUZZERS "Build fuzz/ targets (Clang/libFuzzer)" OFF)
if(SONDER_BUILD_FUZZERS AND NOT CMAKE_SOURCE_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/fuzz")
    add_subdirectory(fuzz)
endif()

if(SONDER_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
