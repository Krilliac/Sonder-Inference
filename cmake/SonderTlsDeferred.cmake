# CI helper for .github/workflows/tls.yml (owner: feat/tls).
#
# Lets a TLS-ON build work whether or not the root CMakeLists.txt already has
# include(SonderTls) (docs/integration/tls.md). Pass
#
#   -DCMAKE_PROJECT_SonderInference_INCLUDE=<repo>/cmake/SonderTlsDeferred.cmake
#
# project(SonderInference) includes this file, and it defers
# include(SonderTls) to the end of the root directory, after
# add_library(sonder_inference) and the test helpers exist. SonderTls.cmake
# uses include_guard(GLOBAL), so once the root includes it this does nothing.
# Plain local builds do not need this file.
set(_SONDER_TLS_MODULE "${CMAKE_CURRENT_LIST_DIR}/SonderTls.cmake")
macro(_sonder_tls_deferred_include)
    # Subdirectories cannot be added during deferred execution, so
    # SonderTls.cmake include()s tests/tls instead of add_subdirectory().
    set(_SONDER_TLS_DEFERRED TRUE)
    include("${_SONDER_TLS_MODULE}")
endmacro()
cmake_language(DEFER CALL _sonder_tls_deferred_include)
