# Module: benchmark harness (owner: feat/ollama-bench).
# Library code compiles into sonder_inference (used by `sonder-infer bench`);
# `sonder-bench` is a standalone runner that also writes a markdown summary.
sonder_module_sources(src/benchmark.cpp src/report.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_BENCH)

add_executable(sonder-bench tools/sonder_bench.cpp)
target_link_libraries(sonder-bench PRIVATE sonder::inference)
sonder_set_warnings(sonder-bench)

sonder_add_module_tests(bench tests/test_benchmark.cpp tests/test_benchmark_ext.cpp)
if(TARGET sonder_bench_tests)
    target_compile_definitions(sonder_bench_tests PRIVATE
        SONDER_TEST_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/corpus/smoke.json"
        SONDER_TEST_BASELINE_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/corpus/baseline.json")
endif()

if(SONDER_BUILD_TESTS AND TARGET sonder-infer)
    add_test(NAME sonder.bench.cli_mock
        COMMAND sonder-infer bench --backend mock --model mock:tiny
                --corpus "${CMAKE_CURRENT_SOURCE_DIR}/corpus/smoke.json"
                --out "${CMAKE_CURRENT_BINARY_DIR}/bench-mock.json" --warmup 0 --runs 1)
    set_tests_properties(sonder.bench.cli_mock PROPERTIES TIMEOUT 60)
endif()

if(SONDER_BUILD_TESTS)
    add_test(NAME sonder.bench.tool_mock
        COMMAND sonder-bench --backend mock --model mock:tiny
                --corpus "${CMAKE_CURRENT_SOURCE_DIR}/corpus/baseline.json"
                --out-dir "${CMAKE_CURRENT_BINARY_DIR}/bench-results" --stem tool-mock
                --warmup 0 --runs 1 --hardware "CI runner")
    set_tests_properties(sonder.bench.tool_mock PROPERTIES TIMEOUT 60)
endif()
