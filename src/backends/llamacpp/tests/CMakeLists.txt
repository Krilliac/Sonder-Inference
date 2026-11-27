# Tests for the llama.cpp backend module. None of them download anything.
#  sonder.llamacpp.*          doctest cases, no model weights: pure helpers,
#                             llama.cpp error paths, tokenization against the
#                             vocab-only GGUF fixtures bundled with llama.cpp,
#                             and the core Backend adapter.
#  sonder.llamacpp.integration real model via SONDER_TEST_GGUF=<file.gguf>;
#                             reported as Skipped (exit 77) when unset.

sonder_add_module_tests(llamacpp
    test_gguf_compat.cpp
    test_llamacpp_util.cpp
    test_llamacpp_nomodel.cpp
    test_llamacpp_adapter.cpp
    test_llamacpp_chat.cpp
    test_llamacpp_frontier.cpp
    test_llamacpp_speculative_verifier.cpp
    test_pq2_reblock.cpp
    test_slot_store.cpp
    test_request_guards.cpp
)
if(TARGET sonder_llamacpp_tests)
    target_link_libraries(sonder_llamacpp_tests PRIVATE sonder_backend_llamacpp)
    target_link_libraries(sonder_llamacpp_tests PRIVATE llama ggml)
    target_include_directories(sonder_llamacpp_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_compile_definitions(sonder_llamacpp_tests PRIVATE
        SONDER_LLAMACPP_VOCAB_DIR="${SONDER_LLAMACPP_VOCAB_DIR}"
        SONDER_EXPECTED_LLAMACPP_TAG="${SONDER_LLAMACPP_RESOLVED_TAG}")
endif()

add_executable(sonder_llamacpp_integration test_llamacpp_integration.cpp)
target_link_libraries(sonder_llamacpp_integration PRIVATE
    sonder::inference sonder_backend_llamacpp sonder_doctest)
sonder_set_warnings(sonder_llamacpp_integration)
add_test(NAME sonder.llamacpp.integration COMMAND sonder_llamacpp_integration)
set_tests_properties(sonder.llamacpp.integration PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
