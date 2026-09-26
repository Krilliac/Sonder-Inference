# Core test suite. Module tests live with their modules
# (sonder_add_module_tests in cmake/SonderModules.cmake).
sonder_add_module_tests(core
    test_json.cpp
    test_sampling.cpp
    test_cancellation.cpp
    test_mock_backend.cpp
    test_session.cpp
    test_telemetry.cpp
    test_c_api.cpp
    test_kv_capacity_adapter.cpp
    test_engine_runtime.cpp
    test_chat.cpp
    test_cli_args.cpp
)

# Proves sonder_inference.h is valid C and the ABI is usable from C.
add_executable(sonder_c_abi_smoke c_abi_smoke.c)
target_link_libraries(sonder_c_abi_smoke PRIVATE sonder::inference)
set_target_properties(sonder_c_abi_smoke PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME sonder.c_abi_smoke COMMAND sonder_c_abi_smoke)
set_tests_properties(sonder.c_abi_smoke PROPERTIES TIMEOUT 60)

if(SONDER_BUILD_CLI)
    # End-to-end CLI checks against the mock backend (no network required).
    add_test(NAME sonder.cli.version COMMAND sonder-infer version)
    add_test(NAME sonder.cli.devices COMMAND sonder-infer devices)
    add_test(NAME sonder.cli.generate_mock
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt "hello sonder"
                --max-tokens 8 --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-generate.jsonl")
    add_test(NAME sonder.cli.rejects_bad_sampling
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt x --temperature -1)
    set_tests_properties(sonder.cli.rejects_bad_sampling PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(sonder.cli.version sonder.cli.devices sonder.cli.generate_mock
        sonder.cli.rejects_bad_sampling PROPERTIES TIMEOUT 60)

    # Sampling flags added with feat/sampling-config.
    add_test(NAME sonder.cli.generate_sampling_flags
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt "hello sonder" --max-tokens 4
                --typical-p 0.9 --repeat-last-n -1 --presence-penalty 0.5 --frequency-penalty 0.25
                --num-ctx 512 --logit-bias "1:-inf,2:1.5")
    add_test(NAME sonder.cli.rejects_bad_logit_bias
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt x --logit-bias 1:500)
    add_test(NAME sonder.cli.rejects_malformed_logit_bias
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt x --logit-bias oops)
    add_test(NAME sonder.cli.rejects_bad_typical_p
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt x --typical-p 0)
    set_tests_properties(sonder.cli.rejects_bad_logit_bias sonder.cli.rejects_malformed_logit_bias
        sonder.cli.rejects_bad_typical_p PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(sonder.cli.generate_sampling_flags sonder.cli.rejects_bad_logit_bias
        sonder.cli.rejects_malformed_logit_bias sonder.cli.rejects_bad_typical_p PROPERTIES TIMEOUT 60)

    # `sonder-infer chat` (docs/integration/chat-cli.md).
    add_test(NAME sonder.cli.chat_messages_mock
        COMMAND sonder-infer chat --backend mock --model mock:tiny --max-tokens 6
                --messages "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_messages.json")
    set_tests_properties(sonder.cli.chat_messages_mock PROPERTIES
        PASS_REGULAR_EXPRESSION "chat native=no messages=4 stop=max_tokens")
    add_test(NAME sonder.cli.chat_rejects_bad_role
        COMMAND sonder-infer chat --backend mock --model mock:tiny
                --messages "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_messages_bad_role.json")
    add_test(NAME sonder.cli.chat_rejects_missing_file
        COMMAND sonder-infer chat --backend mock --model mock:tiny --messages "${CMAKE_CURRENT_BINARY_DIR}/absent.json")
    set_tests_properties(sonder.cli.chat_rejects_bad_role sonder.cli.chat_rejects_missing_file
        PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(sonder.cli.chat_messages_mock sonder.cli.chat_rejects_bad_role
        sonder.cli.chat_rejects_missing_file PROPERTIES TIMEOUT 60)

    if(TARGET sonder-bench)
        # `sonder-infer bench --markdown` prints and writes the markdown summary.
        add_test(NAME sonder.cli.bench_markdown_mock
            COMMAND sonder-infer bench --backend mock --model mock:tiny --markdown
                    --corpus "${PROJECT_SOURCE_DIR}/bench/corpus/smoke.json"
                    --out "${CMAKE_CURRENT_BINARY_DIR}/cli-bench-md.json" --warmup 0 --runs 1)
        set_tests_properties(sonder.cli.bench_markdown_mock PROPERTIES TIMEOUT 60
            PASS_REGULAR_EXPRESSION "wrote [^\n]*cli-bench-md\\.md")
    endif()
endif()
