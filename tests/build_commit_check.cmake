# Checks cmake/SonderBuildInfo.cmake script mode: the generated source carries
# the commit, an unchanged commit leaves the file untouched, a new one rewrites it.
#   cmake -DSOURCE_DIR=<repo> -DWORK_DIR=<scratch> -P build_commit_check.cmake
cmake_minimum_required(VERSION 3.21)
if(NOT SOURCE_DIR OR NOT WORK_DIR)
    message(FATAL_ERROR "build_commit_check.cmake: pass -DSOURCE_DIR and -DWORK_DIR")
endif()
set(out "${WORK_DIR}/build_commit.cpp")
file(REMOVE "${out}")

function(generate commit)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -DSONDER_BUILD_INFO_RUN=ON
            "-DSONDER_SOURCE_DIR=${SOURCE_DIR}"
            "-DSONDER_BUILD_COMMIT_TEMPLATE=${SOURCE_DIR}/src/engine/build_commit.cpp.in"
            "-DSONDER_BUILD_COMMIT_OUTPUT=${out}"
            "-DSONDER_GIT_COMMIT_OVERRIDE=${commit}"
            -P "${SOURCE_DIR}/cmake/SonderBuildInfo.cmake"
        RESULT_VARIABLE rc ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "generator failed (${rc}): ${err}")
    endif()
endfunction()

generate(aaaaaaaaaaaa)
file(READ "${out}" text)
if(NOT text MATCHES "return \"aaaaaaaaaaaa\";")
    message(FATAL_ERROR "generated source lacks the commit:\n${text}")
endif()
file(TIMESTAMP "${out}" t1 "%Y%m%d%H%M%S")

execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1.5)
generate(aaaaaaaaaaaa)
file(TIMESTAMP "${out}" t2 "%Y%m%d%H%M%S")
if(NOT t1 STREQUAL t2)
    message(FATAL_ERROR "unchanged commit rewrote the file (${t1} -> ${t2})")
endif()

generate(bbbbbbbbbbbb)
file(READ "${out}" text)
if(NOT text MATCHES "return \"bbbbbbbbbbbb\";" OR text MATCHES "aaaaaaaaaaaa")
    message(FATAL_ERROR "new commit not reflected:\n${text}")
endif()
message(STATUS "build_commit_check: ok")
