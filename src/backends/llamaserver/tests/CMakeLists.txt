sonder_add_module_tests(llamaserver test_backend.cpp test_supervisor.cpp)

if(TARGET sonder_llamaserver_tests)
    # Reuse the already reviewed and pinned test-only cpp-httplib dependency
    # declared by the Ollama tests; never fetch or link an upstream runtime.
    include(FetchContent)
    FetchContent_GetProperties(sonder_test_cpp_httplib)
    target_include_directories(sonder_llamaserver_tests SYSTEM PRIVATE "${sonder_test_cpp_httplib_SOURCE_DIR}")
    if(WIN32)
        target_link_libraries(sonder_llamaserver_tests PRIVATE ws2_32)
        target_compile_definitions(sonder_llamaserver_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
    endif()
    if(MSVC)
        target_compile_options(sonder_llamaserver_tests PRIVATE /bigobj)
    endif()
endif()
