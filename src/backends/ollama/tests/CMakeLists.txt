# Ollama backend tests: recorded NDJSON fixtures + an in-process fake Ollama
# server. No live Ollama is required.
sonder_add_module_tests(ollama
    test_ollama_adapter.cpp
    ollama_decoder_tests.cpp
    ollama_client_tests.cpp
    ollama_backend_tests.cpp)

if(TARGET sonder_ollama_tests)
    # cpp-httplib (MIT) is used ONLY by tests, to host the fake server. It is
    # fetched at a pinned release with a verified hash and never linked into
    # sonder_inference. See INTEGRATION_NOTES.md.
    include(FetchContent)
    FetchContent_Declare(sonder_test_cpp_httplib
        URL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.58.0/httplib.h
        URL_HASH SHA256=aa14e7e7bd2703694e0a6b6855af3b8c406102ab1fc56ac905fe33619b31faa5
        DOWNLOAD_NO_EXTRACT TRUE)
    FetchContent_MakeAvailable(sonder_test_cpp_httplib)
    target_include_directories(sonder_ollama_tests SYSTEM PRIVATE "${sonder_test_cpp_httplib_SOURCE_DIR}")
    target_include_directories(sonder_ollama_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(sonder_ollama_tests PRIVATE
        SONDER_OLLAMA_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
    if(WIN32)
        target_link_libraries(sonder_ollama_tests PRIVATE ws2_32)
        target_compile_definitions(sonder_ollama_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
    endif()
    if(MSVC)
        target_compile_options(sonder_ollama_tests PRIVATE /bigobj)
    endif()
endif()
