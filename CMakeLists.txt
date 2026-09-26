cmake_minimum_required(VERSION 3.21)

project(SonderInference
    VERSION 0.1.0
    DESCRIPTION "Sonder Inference runtime (policy layer over execution backends)"
    LANGUAGES C CXX)

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

# Record the source commit for telemetry/benchmark provenance.
set(SONDER_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE SONDER_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT SONDER_GIT_COMMIT)
        set(SONDER_GIT_COMMIT "unknown")
    endif()
endif()

add_library(sonder_inference
    src/common/error.cpp
    src/common/json.cpp
    src/engine/sampling_config.cpp
    src/devices/device.cpp
    src/telemetry/telemetry.cpp
    src/sessions/session.cpp
    src/engine/engine.cpp
    src/engine/c_api.cpp
    src/backends/backend.cpp
    src/backends/mock_backend.cpp
    src/net/http_client.cpp
)
add_library(sonder::inference ALIAS sonder_inference)

target_include_directories(sonder_inference
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(sonder_inference PUBLIC cxx_std_20)
target_compile_definitions(sonder_inference
    PRIVATE
        SONDER_INFERENCE_BUILDING
        SONDER_INFERENCE_VERSION="${PROJECT_VERSION}"
        SONDER_INFERENCE_GIT_COMMIT="${SONDER_GIT_COMMIT}")
if(BUILD_SHARED_LIBS)
    target_compile_definitions(sonder_inference PUBLIC SONDER_INFERENCE_SHARED)
endif()

find_package(Threads REQUIRED)
target_link_libraries(sonder_inference PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(sonder_inference PRIVATE ws2_32 advapi32)
    target_compile_definitions(sonder_inference PRIVATE _WIN32_WINNT=0x0A00)
endif()
sonder_set_warnings(sonder_inference)

if(SONDER_BUILD_CLI)
    add_executable(sonder-infer tools/sonder-infer/main.cpp)
    target_link_libraries(sonder-infer PRIVATE sonder::inference)
    sonder_set_warnings(sonder-infer)
endif()

if(SONDER_BUILD_TESTS)
    enable_testing()
    include(SonderDoctest)
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
    bench)
foreach(module_dir IN LISTS SONDER_MODULE_DIRS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${module_dir}/CMakeLists.txt")
        message(STATUS "Sonder module: ${module_dir}")
        add_subdirectory(${module_dir})
    endif()
endforeach()

if(SONDER_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
