# Helpers for optional Sonder modules (docs/MODULES.md).

# Adds sources to the core library from a module directory. Relative paths
# are resolved against the calling CMakeLists.txt directory.
function(sonder_module_sources)
    target_sources(sonder_inference PRIVATE ${ARGN})
endfunction()

# Declares a module feature macro visible to the library and its consumers,
# e.g. sonder_module_define(SONDER_HAS_OLLAMA_BACKEND).
function(sonder_module_define)
    foreach(def IN LISTS ARGN)
        target_compile_definitions(sonder_inference PUBLIC ${def}=1)
    endforeach()
endfunction()

# Adds a public include directory for module-owned headers.
function(sonder_module_include_directories)
    foreach(dir IN LISTS ARGN)
        get_filename_component(abs "${dir}" ABSOLUTE)
        target_include_directories(sonder_inference PUBLIC "$<BUILD_INTERFACE:${abs}>")
    endforeach()
endfunction()

# sonder_add_module_tests(<name> <sources...>)
# Builds a doctest executable sonder_<name>_tests (main provided), links the
# core library, exposes src/ and tests/ (test_helpers.hpp) as include dirs,
# and registers every test case with CTest as "sonder.<name>.<case>".
function(sonder_add_module_tests name)
    if(NOT SONDER_BUILD_TESTS)
        return()
    endif()
    set(target sonder_${name}_tests)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE sonder::inference sonder_doctest_main)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tests")
    sonder_set_warnings(${target})
    doctest_discover_tests(${target} TEST_PREFIX "sonder.${name}." PROPERTIES TIMEOUT 120)
endfunction()
