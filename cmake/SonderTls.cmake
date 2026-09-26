# Optional TLS for the internal HTTP client (src/net/tls*). Owner: feat/tls.
# See docs/integration/tls.md.
#
#   -DSONDER_WITH_TLS=ON            enable https:// in src/net (default OFF)
#   -DSONDER_TLS_BACKEND=auto|openssl|schannel
#        auto = schannel on Windows, openssl elsewhere.
#
# Include from the root CMakeLists.txt AFTER add_library(sonder_inference) and
# after include(SonderDoctest) (i.e. next to the module loop):
#     include(SonderTls)
# With SONDER_WITH_TLS=OFF this file only declares the options: no sources,
# no definitions, no dependencies, and https:// stays ErrorCode::unsupported.

option(SONDER_WITH_TLS "Enable TLS (https://) in the internal HTTP client" OFF)
set(SONDER_TLS_BACKEND "auto" CACHE STRING "TLS backend when SONDER_WITH_TLS=ON: auto, openssl or schannel")
set_property(CACHE SONDER_TLS_BACKEND PROPERTY STRINGS auto openssl schannel)

if(NOT SONDER_WITH_TLS)
    return()
endif()

if(NOT TARGET sonder_inference)
    message(FATAL_ERROR "SonderTls.cmake must be included after add_library(sonder_inference)")
endif()

set(_sonder_tls_backend "${SONDER_TLS_BACKEND}")
if(_sonder_tls_backend STREQUAL "auto")
    if(WIN32)
        set(_sonder_tls_backend "schannel")
    else()
        set(_sonder_tls_backend "openssl")
    endif()
endif()

set(_sonder_net_dir "${PROJECT_SOURCE_DIR}/src/net")
target_sources(sonder_inference PRIVATE "${_sonder_net_dir}/tls_common.cpp")
# SONDER_HAS_TLS is PUBLIC so tests and consumers can branch on it.
target_compile_definitions(sonder_inference PUBLIC SONDER_HAS_TLS=1)

if(_sonder_tls_backend STREQUAL "openssl")
    find_package(OpenSSL 1.1.1 REQUIRED COMPONENTS SSL Crypto)
    target_sources(sonder_inference PRIVATE "${_sonder_net_dir}/tls_openssl.cpp")
    target_link_libraries(sonder_inference PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(sonder_inference PRIVATE SONDER_TLS_BACKEND_OPENSSL=1)
elseif(_sonder_tls_backend STREQUAL "schannel")
    if(NOT WIN32)
        message(FATAL_ERROR "SONDER_TLS_BACKEND=schannel is only available on Windows")
    endif()
    target_sources(sonder_inference PRIVATE "${_sonder_net_dir}/tls_schannel.cpp")
    target_link_libraries(sonder_inference PRIVATE secur32 crypt32 bcrypt)
    target_compile_definitions(sonder_inference PRIVATE SONDER_TLS_BACKEND_SCHANNEL=1)
else()
    message(FATAL_ERROR "Unknown SONDER_TLS_BACKEND '${SONDER_TLS_BACKEND}' (expected auto, openssl or schannel)")
endif()
set(SONDER_TLS_BACKEND_RESOLVED "${_sonder_tls_backend}")
message(STATUS "Sonder TLS: ON (backend: ${_sonder_tls_backend})")

# Loopback TLS tests (tests/tls). The in-test server and certificate
# generation use OpenSSL, also when the client backend is Schannel.
if(SONDER_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/tests/tls/CMakeLists.txt")
    add_subdirectory("${PROJECT_SOURCE_DIR}/tests/tls" "${PROJECT_BINARY_DIR}/tests/tls")
endif()
