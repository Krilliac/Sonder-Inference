# Build provenance: the short git SHA lives in one generated translation unit
# (build_commit.cpp) instead of a compile definition. A definition changes the
# command line of every sonder_inference object on every commit, which defeats
# any compiler cache; a generated source changes one object.
#
# Two modes, one file:
#   include(SonderBuildInfo)      -> defines sonder_add_build_commit(<target>)
#   cmake -DSONDER_BUILD_INFO_RUN=ON ... -P SonderBuildInfo.cmake
#                                 -> (re)generates the source; run at build time
#                                    by the sonder_build_commit custom target.
#
# Script-mode variables: SONDER_SOURCE_DIR, SONDER_BUILD_COMMIT_TEMPLATE,
# SONDER_BUILD_COMMIT_OUTPUT, GIT_EXECUTABLE (optional), and
# SONDER_GIT_COMMIT_OVERRIDE (optional; used by the tests instead of git).

# Resolves the short SHA of SOURCE_DIR into <out_var>, "unknown" when unavailable.
function(sonder_resolve_git_commit source_dir git_executable out_var)
    set(commit "unknown")
    if(SONDER_GIT_COMMIT_OVERRIDE)
        set(commit "${SONDER_GIT_COMMIT_OVERRIDE}")
    elseif(git_executable AND EXISTS "${source_dir}/.git")
        execute_process(
            COMMAND "${git_executable}" rev-parse --short=12 HEAD
            WORKING_DIRECTORY "${source_dir}"
            OUTPUT_VARIABLE resolved
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(resolved)
            set(commit "${resolved}")
        endif()
    endif()
    set(${out_var} "${commit}" PARENT_SCOPE)
endfunction()

# Writes the generated source. configure_file leaves the file (and its mtime)
# untouched when the content is identical, so an unchanged SHA rebuilds nothing.
function(sonder_write_build_commit template output commit)
    set(SONDER_GIT_COMMIT "${commit}")
    configure_file("${template}" "${output}" @ONLY)
endfunction()

if(SONDER_BUILD_INFO_RUN)
    sonder_resolve_git_commit("${SONDER_SOURCE_DIR}" "${GIT_EXECUTABLE}" _commit)
    sonder_write_build_commit("${SONDER_BUILD_COMMIT_TEMPLATE}" "${SONDER_BUILD_COMMIT_OUTPUT}" "${_commit}")
    return()
endif()

# Adds the generated source to <target> and keeps it current at build time
# without a reconfigure.
function(sonder_add_build_commit target)
    set(template "${PROJECT_SOURCE_DIR}/src/engine/build_commit.cpp.in")
    set(output "${PROJECT_BINARY_DIR}/generated/build_commit.cpp")
    find_package(Git QUIET)
    sonder_resolve_git_commit("${PROJECT_SOURCE_DIR}" "${GIT_EXECUTABLE}" commit)
    sonder_write_build_commit("${template}" "${output}" "${commit}")
    target_sources(${target} PRIVATE "${output}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${template}")
    add_custom_target(sonder_build_commit
        COMMAND "${CMAKE_COMMAND}"
            -DSONDER_BUILD_INFO_RUN=ON
            "-DSONDER_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DSONDER_BUILD_COMMIT_TEMPLATE=${template}"
            "-DSONDER_BUILD_COMMIT_OUTPUT=${output}"
            "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
            -P "${PROJECT_SOURCE_DIR}/cmake/SonderBuildInfo.cmake"
        BYPRODUCTS "${output}"
        COMMENT "Refreshing build commit string"
        VERBATIM)
    add_dependencies(${target} sonder_build_commit)
endfunction()
