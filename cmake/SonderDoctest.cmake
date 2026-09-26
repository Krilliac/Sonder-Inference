# doctest (MIT) test framework, fetched at configure time for tests only.
# Not vendored and not linked into the library. See docs/LICENSE_REVIEW.md.
include(FetchContent)

set(SONDER_DOCTEST_VERSION "2.5.3")
set(SONDER_DOCTEST_SHA256 "174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331")

# Offline builds: point FETCHCONTENT_SOURCE_DIR_DOCTEST at an extracted
# doctest-2.5.3 tree.
FetchContent_Declare(doctest
    URL "https://github.com/doctest/doctest/archive/refs/tags/v${SONDER_DOCTEST_VERSION}.tar.gz"
    URL_HASH "SHA256=${SONDER_DOCTEST_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Only headers and the CMake discovery script are used; skip doctest's
    # own CMakeLists so its options/targets do not leak into this project.
    SOURCE_SUBDIR "sonder-no-cmake")
FetchContent_MakeAvailable(doctest)

if(NOT TARGET sonder_doctest)
    add_library(sonder_doctest INTERFACE)
    target_include_directories(sonder_doctest SYSTEM INTERFACE "${doctest_SOURCE_DIR}")
endif()
include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

# Shared doctest main for all test executables.
if(NOT TARGET sonder_doctest_main)
    add_library(sonder_doctest_main STATIC "${CMAKE_CURRENT_LIST_DIR}/doctest_main.cpp")
    target_link_libraries(sonder_doctest_main PUBLIC sonder_doctest)
    target_compile_features(sonder_doctest_main PUBLIC cxx_std_20)
endif()
