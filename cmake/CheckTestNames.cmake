# Fails when a doctest TEST_CASE name contains ';'.
#
# doctest_discover_tests() registers one CTest test per TEST_CASE, and CMake
# treats ';' as a list separator, so a name like "a; b" is split into two CTest
# entries ("a" and " b"). Neither matches the real test case, doctest runs zero
# tests for each filter and exits 0, and CTest reports both halves as passed:
# the test never runs. Three such tests passed "green" this way until
# 2026-09-30. Usage: cmake -DSOURCE_DIR=<repo> -P CheckTestNames.cmake
if(NOT SOURCE_DIR)
    message(FATAL_ERROR "CheckTestNames.cmake: pass -DSOURCE_DIR=<repository root>")
endif()

file(GLOB_RECURSE _sources LIST_DIRECTORIES false
    "${SOURCE_DIR}/src/*.cpp" "${SOURCE_DIR}/src/*.hpp"
    "${SOURCE_DIR}/tests/*.cpp" "${SOURCE_DIR}/tests/*.hpp"
    "${SOURCE_DIR}/bench/*.cpp" "${SOURCE_DIR}/tools/*.cpp")

set(_bad "")
set(_checked 0)
foreach(_file IN LISTS _sources)
    file(READ "${_file}" _text)
    # Protect the list separator before splitting the file into lines.
    string(REPLACE ";" "<SEMICOLON>" _text "${_text}")
    string(REPLACE "\n" ";" _lines "${_text}")
    set(_lineno 0)
    foreach(_line IN LISTS _lines)
        math(EXPR _lineno "${_lineno} + 1")
        if(_line MATCHES "TEST_CASE[A-Z_]*\\(\"([^\"]*)\"")
            set(_raw "${CMAKE_MATCH_1}")
            math(EXPR _checked "${_checked} + 1")
            if(_raw MATCHES "<SEMICOLON>")
                string(REPLACE "<SEMICOLON>" ";" _name "${_raw}")
                file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_file}")
                string(APPEND _bad "  ${_rel}:${_lineno}: \"${_name}\"\n")
            endif()
        endif()
    endforeach()
endforeach()

if(_checked EQUAL 0)
    message(FATAL_ERROR "CheckTestNames.cmake: found no TEST_CASE in ${SOURCE_DIR}; the scan is not checking anything")
endif()
if(_bad)
    message(FATAL_ERROR "doctest TEST_CASE names must not contain ';' (CTest splits them and the test silently never runs):\n${_bad}")
endif()
message(STATUS "CheckTestNames: ${_checked} TEST_CASE names checked, none contain ';'")
