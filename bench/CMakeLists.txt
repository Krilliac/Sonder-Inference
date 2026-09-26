# Module: benchmark harness (owner: feat/ollama-bench).
sonder_module_sources(src/benchmark.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_BENCH)

sonder_add_module_tests(bench tests/test_benchmark.cpp)
if(TARGET sonder_bench_tests)
    target_compile_definitions(sonder_bench_tests PRIVATE
        SONDER_TEST_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/corpus/smoke.json")
endif()

if(SONDER_BUILD_TESTS AND TARGET sonder-infer)
    add_test(NAME sonder.bench.cli_mock
        COMMAND sonder-infer bench --backend mock --model mock:tiny
                --corpus "${CMAKE_CURRENT_SOURCE_DIR}/corpus/smoke.json"
                --out "${CMAKE_CURRENT_BINARY_DIR}/bench-mock.json" --warmup 0 --runs 1)
    set_tests_properties(sonder.bench.cli_mock PROPERTIES TIMEOUT 60)
endif()
