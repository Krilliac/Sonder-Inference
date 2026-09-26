# CLI smoke-test driver: runs one command and checks its exact exit code and
# output. Used by tests/CMakeLists.txt (sonder.cli.* tests); no network.
#
#   cmake [-D<VAR>=<value>...] -P cli_expect.cmake -- <program> [args...]
#
# Variables:
#   EXPECT_RC            required exit code (e.g. 0, 1, 2)
#   STDIN_FILE           file fed to stdin (default: an empty file, so the
#                        child never inherits ctest's stdin)
#   STDOUT_REGEX         stdout must match (CMake regex)
#   STDERR_REGEX         stderr must match
#   STDOUT_NOT_REGEX     stdout must not match
#   STDERR_NOT_REGEX     stderr must not match
#   STDOUT_JSON=ON       stdout is exactly one JSON document on one line
#   STDERR_JSON=ON       stderr is exactly one JSON document on one line
#   STDERR_JSONL=ON      every non-empty stderr line is a JSON object
#   JSONL_FILE           file whose non-empty lines must all be JSON objects
#                        with "schema": "sonder.observatory.event/1"; it is
#                        removed before the run so stale output never passes
#   JSONL_FILE_REGEX     JSONL_FILE content must match
#   JSONL_FILE_REGEX2    second pattern JSONL_FILE content must match
#   JSONL_FILE_COUNT_REGEX, JSONL_FILE_COUNT
#                        JSONL_FILE must contain exactly JSONL_FILE_COUNT
#                        matches of JSONL_FILE_COUNT_REGEX (the pattern must
#                        not match a ';')
#
# In every *_REGEX value the two characters "\n" stand for a newline (CMake
# regular expressions have no newline escape), so "[^\n]*" stays on one line.
cmake_minimum_required(VERSION 3.21)

set(command)
set(started OFF)
foreach(i RANGE ${CMAKE_ARGC})
    if(i EQUAL CMAKE_ARGC)
        break()
    endif()
    if(started)
        list(APPEND command "${CMAKE_ARGV${i}}")
    elseif(CMAKE_ARGV${i} STREQUAL "--")
        set(started ON)
    endif()
endforeach()
if(NOT command)
    message(FATAL_ERROR "cli_expect: no command after --")
endif()
if(NOT DEFINED EXPECT_RC)
    message(FATAL_ERROR "cli_expect: EXPECT_RC is required")
endif()

if(JSONL_FILE)
    file(REMOVE "${JSONL_FILE}")
endif()
foreach(var STDOUT_REGEX STDERR_REGEX STDOUT_NOT_REGEX STDERR_NOT_REGEX JSONL_FILE_REGEX JSONL_FILE_REGEX2
            JSONL_FILE_COUNT_REGEX)
    if(DEFINED ${var})
        string(REPLACE "\\n" "\n" ${var} "${${var}}")
    endif()
endforeach()

# execute_process() inherits this process's stdin unless INPUT_FILE is set;
# a unique empty file keeps parallel tests independent.
set(empty_stdin)
if(NOT STDIN_FILE)
    string(RANDOM LENGTH 16 token)
    set(empty_stdin "${CMAKE_CURRENT_BINARY_DIR}/cli_expect_stdin_${token}.txt")
    file(WRITE "${empty_stdin}" "")
    set(STDIN_FILE "${empty_stdin}")
endif()
execute_process(
    COMMAND ${command}
    INPUT_FILE "${STDIN_FILE}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    TIMEOUT 60)
if(empty_stdin)
    file(REMOVE "${empty_stdin}")
endif()

# Windows text-mode streams write CRLF; patterns are written for LF.
string(REPLACE "\r\n" "\n" out "${out}")
string(REPLACE "\r\n" "\n" err "${err}")

set(failures)
macro(fail msg)
    list(APPEND failures "${msg}")
endmacro()

if(NOT rc STREQUAL "${EXPECT_RC}")
    fail("exit code ${rc}, expected ${EXPECT_RC}")
endif()
if(DEFINED STDOUT_REGEX AND NOT out MATCHES "${STDOUT_REGEX}")
    fail("stdout does not match '${STDOUT_REGEX}'")
endif()
if(DEFINED STDERR_REGEX AND NOT err MATCHES "${STDERR_REGEX}")
    fail("stderr does not match '${STDERR_REGEX}'")
endif()
if(DEFINED STDOUT_NOT_REGEX AND out MATCHES "${STDOUT_NOT_REGEX}")
    fail("stdout matches '${STDOUT_NOT_REGEX}'")
endif()
if(DEFINED STDERR_NOT_REGEX AND err MATCHES "${STDERR_NOT_REGEX}")
    fail("stderr matches '${STDERR_NOT_REGEX}'")
endif()

# Splits `text` into lines without CMake list semantics (JSON may contain ';'
# and brackets) and checks each non-empty line with string(JSON).
function(check_json_lines text what require_single schema)
    set(pos 0)
    set(count 0)
    string(LENGTH "${text}" len)
    while(pos LESS len)
        string(SUBSTRING "${text}" ${pos} -1 rest)
        string(FIND "${rest}" "\n" nl)
        if(nl EQUAL -1)
            set(line "${rest}")
            string(LENGTH "${rest}" step)
        else()
            string(SUBSTRING "${rest}" 0 ${nl} line)
            math(EXPR step "${nl} + 1")
        endif()
        math(EXPR pos "${pos} + ${step}")
        string(REGEX REPLACE "\r$" "" line "${line}")
        if(line STREQUAL "")
            continue()
        endif()
        math(EXPR count "${count} + 1")
        if(NOT line MATCHES "^{.*}$")
            set(check_failure "${what}: line ${count} is not a JSON object: ${line}" PARENT_SCOPE)
            return()
        endif()
        string(JSON type ERROR_VARIABLE json_error TYPE "${line}")
        if(NOT json_error STREQUAL "NOTFOUND" OR NOT type STREQUAL "OBJECT")
            set(check_failure "${what}: line ${count} is not valid JSON (${json_error}): ${line}" PARENT_SCOPE)
            return()
        endif()
        if(schema)
            string(JSON value ERROR_VARIABLE json_error GET "${line}" schema)
            if(NOT value STREQUAL "${schema}")
                set(check_failure "${what}: line ${count} has schema '${value}', expected '${schema}'" PARENT_SCOPE)
                return()
            endif()
        endif()
    endwhile()
    if(count EQUAL 0)
        set(check_failure "${what}: no JSON line" PARENT_SCOPE)
    elseif(require_single AND NOT count EQUAL 1)
        set(check_failure "${what}: ${count} JSON lines, expected exactly one" PARENT_SCOPE)
    else()
        set(check_failure "" PARENT_SCOPE)
    endif()
endfunction()

if(STDOUT_JSON)
    check_json_lines("${out}" "stdout" ON "")
    if(check_failure)
        fail("${check_failure}")
    endif()
endif()
if(STDERR_JSON)
    check_json_lines("${err}" "stderr" ON "")
    if(check_failure)
        fail("${check_failure}")
    endif()
endif()
if(STDERR_JSONL)
    check_json_lines("${err}" "stderr" OFF "")
    if(check_failure)
        fail("${check_failure}")
    endif()
endif()
if(JSONL_FILE)
    if(NOT EXISTS "${JSONL_FILE}")
        fail("${JSONL_FILE} was not written")
    else()
        file(READ "${JSONL_FILE}" jsonl)
        check_json_lines("${jsonl}" "${JSONL_FILE}" OFF "sonder.observatory.event/1")
        if(check_failure)
            fail("${check_failure}")
        endif()
        if(DEFINED JSONL_FILE_REGEX AND NOT jsonl MATCHES "${JSONL_FILE_REGEX}")
            fail("${JSONL_FILE} does not match '${JSONL_FILE_REGEX}'")
        endif()
        if(DEFINED JSONL_FILE_REGEX2 AND NOT jsonl MATCHES "${JSONL_FILE_REGEX2}")
            fail("${JSONL_FILE} does not match '${JSONL_FILE_REGEX2}'")
        endif()
        if(DEFINED JSONL_FILE_COUNT_REGEX)
            string(REGEX MATCHALL "${JSONL_FILE_COUNT_REGEX}" matches "${jsonl}")
            list(LENGTH matches match_count)
            if(NOT match_count EQUAL "${JSONL_FILE_COUNT}")
                fail("${JSONL_FILE} has ${match_count} matches of '${JSONL_FILE_COUNT_REGEX}', expected ${JSONL_FILE_COUNT}")
            endif()
        endif()
    endif()
endif()

if(failures)
    string(REPLACE ";" "\n  " report "${failures}")
    message(FATAL_ERROR "cli_expect: ${command}\n  ${report}\n--- stdout ---\n${out}\n--- stderr ---\n${err}")
endif()
