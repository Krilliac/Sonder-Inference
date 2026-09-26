# TLS transport tests (built only when SONDER_WITH_TLS=ON; see
# cmake/SonderTls.cmake). A self-signed PKI is generated at test time and a
# TLS server runs on 127.0.0.1 inside the test process: no network access and
# no checked-in keys.
sonder_add_module_tests(tls test_tls.cpp)

if(TARGET sonder_tls_tests)
    # The in-test server always uses OpenSSL (also for the Schannel client).
    find_package(OpenSSL 1.1.1 REQUIRED COMPONENTS SSL Crypto)
    target_link_libraries(sonder_tls_tests PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_include_directories(sonder_tls_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    if(WIN32)
        target_link_libraries(sonder_tls_tests PRIVATE ws2_32)
        target_compile_definitions(sonder_tls_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
    endif()
endif()
