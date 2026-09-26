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
endif()
