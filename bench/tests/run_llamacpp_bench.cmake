# Exercise the real standalone CLI, omitting --backend to check the default.
# Match the native integration suite's opt-in SONDER_TEST_GGUF contract.
# CTest's SKIP_REGULAR_EXPRESSION reports absent prerequisites as skipped.
if(NOT NATIVE_ENABLED)
    message("SKIP: native bench requires SONDER_WITH_LLAMA_CPP=ON")
    return()
endif()
if("$ENV{SONDER_TEST_GGUF}" STREQUAL "")
    message("SKIP: native bench SONDER_TEST_GGUF not set")
    return()
endif()

execute_process(
    COMMAND "${BENCH_EXE}" --model "$ENV{SONDER_TEST_GGUF}"
        --ctx-size 512 --threads 1 --gpu-layers 0
        --corpus "${CORPUS}" --warmup 0 --runs 1 --budget-seconds 0
        --no-gpu-snapshot --out-dir "${OUT_DIR}" --stem native-smoke
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors
    TIMEOUT 150)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "native bench failed (${result}):\n${output}\n${errors}")
endif()
file(READ "${OUT_DIR}/native-smoke.json" report)
string(JSON backend GET "${report}" backend name)
string(JSON failures GET "${report}" summary failures)
string(JSON cancelled GET "${report}" summary cancelled)
string(JSON tokens GET "${report}" summary total_completion_tokens)
string(JSON rows LENGTH "${report}" runs)
if(NOT backend STREQUAL "llamacpp" OR NOT failures EQUAL 0 OR
   NOT cancelled EQUAL 0 OR tokens LESS 1 OR NOT rows EQUAL 3)
    message(FATAL_ERROR
        "native bench returned invalid results: backend=${backend}, failures=${failures}, cancelled=${cancelled}, tokens=${tokens}, rows=${rows}")
endif()
message(STATUS "native bench completed ${rows} request rows with no failures")
