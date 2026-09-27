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
    test_runtime_tracking.cpp
    test_chat.cpp
    test_cli_args.cpp
    test_cli_spec.cpp
    test_cli_repl.cpp
)

# Proves sonder_inference.h is valid C and the ABI is usable from C.
add_executable(sonder_c_abi_smoke c_abi_smoke.c)
target_link_libraries(sonder_c_abi_smoke PRIVATE sonder::inference)
set_target_properties(sonder_c_abi_smoke PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME sonder.c_abi_smoke COMMAND sonder_c_abi_smoke)
set_tests_properties(sonder.c_abi_smoke PROPERTIES TIMEOUT 60)

if(SONDER_BUILD_CLI)
    # Optional modules compiled into this build (their feature macros).
    get_target_property(sonder_defs sonder_inference INTERFACE_COMPILE_DEFINITIONS)
    set(sonder_cli_has_server OFF)
    set(sonder_cli_has_ollama OFF)
    set(sonder_cli_has_bench OFF)
    if("SONDER_HAS_SERVER=1" IN_LIST sonder_defs)
        set(sonder_cli_has_server ON)
    endif()
    if("SONDER_HAS_OLLAMA_BACKEND=1" IN_LIST sonder_defs)
        set(sonder_cli_has_ollama ON)
    endif()
    if("SONDER_HAS_BENCH=1" IN_LIST sonder_defs)
        set(sonder_cli_has_bench ON)
    endif()

    # Patterns must not contain an unbalanced '[' (CMake list syntax); use '.'.
    # sonder_cli_test(<name> RC <code> COMMAND <exe> <args>... [ENV VAR=value...]
    #                 [STDIN file] [STDOUT_REGEX re] [STDERR_REGEX re] [STDOUT_NOT_REGEX re]
    #                 [STDERR_NOT_REGEX re] [STDOUT_JSON] [STDERR_JSON] [STDERR_JSONL]
    #                 [JSONL_FILE path [JSONL_FILE_REGEX re] [JSONL_FILE_REGEX2 re]
    #                  [JSONL_FILE_COUNT_REGEX re JSONL_FILE_COUNT n]])
    # Runs the command through cli_expect.cmake, which checks the exact exit
    # code (docs/CLI.md "Exit codes") and the output. The Sonder environment
    # defaults are cleared unless ENV sets them, so a developer's shell never
    # changes the result.
    function(sonder_cli_test name)
        cmake_parse_arguments(T "STDOUT_JSON;STDERR_JSON;STDERR_JSONL"
            "RC;STDIN;STDOUT_REGEX;STDERR_REGEX;STDOUT_NOT_REGEX;STDERR_NOT_REGEX;JSONL_FILE;JSONL_FILE_REGEX;JSONL_FILE_REGEX2;JSONL_FILE_COUNT_REGEX;JSONL_FILE_COUNT"
            "ENV;COMMAND" ${ARGN})
        set(defs "-DEXPECT_RC=${T_RC}")
        if(T_STDIN)
            list(APPEND defs "-DSTDIN_FILE=${T_STDIN}")
        endif()
        foreach(var STDOUT_REGEX STDERR_REGEX STDOUT_NOT_REGEX STDERR_NOT_REGEX JSONL_FILE JSONL_FILE_REGEX
                    JSONL_FILE_REGEX2 JSONL_FILE_COUNT_REGEX JSONL_FILE_COUNT)
            if(DEFINED T_${var})
                list(APPEND defs "-D${var}=${T_${var}}")
            endif()
        endforeach()
        foreach(var STDOUT_JSON STDERR_JSON STDERR_JSONL)
            if(T_${var})
                list(APPEND defs "-D${var}=ON")
            endif()
        endforeach()
        add_test(NAME sonder.cli.${name}
            COMMAND "${CMAKE_COMMAND}" -E env
                --unset=SONDER_INFER_BACKEND --unset=SONDER_INFER_MODEL --unset=SONDER_OLLAMA_URL
                --unset=OLLAMA_HOST ${T_ENV}
                "${CMAKE_COMMAND}" ${defs} -P "${CMAKE_CURRENT_SOURCE_DIR}/cli_expect.cmake" -- ${T_COMMAND})
        set_tests_properties(sonder.cli.${name} PROPERTIES TIMEOUT 90)
    endfunction()

    set(infer $<TARGET_FILE:sonder-infer>)
    set(chat_fixture "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_messages.json")
    set(mock --backend mock --model mock:tiny)

    # End-to-end CLI checks against the mock backend (no network required).
    add_test(NAME sonder.cli.version COMMAND sonder-infer version)
    add_test(NAME sonder.cli.devices COMMAND sonder-infer devices)
    add_test(NAME sonder.cli.generate_mock
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt "hello sonder"
                --max-tokens 8 --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-generate.jsonl")
    set_tests_properties(sonder.cli.version sonder.cli.devices sonder.cli.generate_mock PROPERTIES TIMEOUT 60)
    sonder_cli_test(rejects_bad_sampling RC 2 STDERR_REGEX "temperature must be finite"
        COMMAND ${infer} generate ${mock} --prompt x --temperature -1)

    # Sampling flags added with feat/sampling-config.
    add_test(NAME sonder.cli.generate_sampling_flags
        COMMAND sonder-infer generate --backend mock --model mock:tiny --prompt "hello sonder" --max-tokens 4
                --typical-p 0.9 --repeat-last-n -1 --presence-penalty 0.5 --frequency-penalty 0.25
                --num-ctx 512 --logit-bias "1:-inf,2:1.5")
    set_tests_properties(sonder.cli.generate_sampling_flags PROPERTIES TIMEOUT 60)
    sonder_cli_test(rejects_bad_logit_bias RC 2 STDERR_REGEX "logit_bias values must be within"
        COMMAND ${infer} generate ${mock} --prompt x --logit-bias 1:500)
    sonder_cli_test(rejects_malformed_logit_bias RC 2 STDERR_REGEX "invalid --logit-bias entry 'oops'"
        COMMAND ${infer} generate ${mock} --prompt x --logit-bias oops)
    sonder_cli_test(rejects_bad_typical_p RC 2 STDERR_REGEX "typical_p must be within"
        COMMAND ${infer} generate ${mock} --prompt x --typical-p 0)

    # `sonder-infer chat` (docs/integration/chat-cli.md).
    add_test(NAME sonder.cli.chat_messages_mock
        COMMAND sonder-infer chat --backend mock --model mock:tiny --max-tokens 6 --messages "${chat_fixture}")
    set_tests_properties(sonder.cli.chat_messages_mock PROPERTIES TIMEOUT 60
        PASS_REGULAR_EXPRESSION "chat native=no messages=4 outcome=completed stop=max_tokens")
    sonder_cli_test(chat_rejects_bad_role RC 2 STDERR_REGEX "unknown role"
        COMMAND ${infer} chat ${mock} --messages "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_messages_bad_role.json")
    sonder_cli_test(chat_rejects_missing_file RC 2 STDERR_REGEX "messages: cannot open"
        COMMAND ${infer} chat ${mock} --messages "${CMAKE_CURRENT_BINARY_DIR}/absent.json")

    if(TARGET sonder-bench)
        # `sonder-infer bench --markdown` prints and writes the markdown summary.
        add_test(NAME sonder.cli.bench_markdown_mock
            COMMAND sonder-infer bench --backend mock --model mock:tiny --markdown
                    --corpus "${PROJECT_SOURCE_DIR}/bench/corpus/smoke.json"
                    --out "${CMAKE_CURRENT_BINARY_DIR}/cli-bench-md.json" --warmup 0 --runs 1)
        set_tests_properties(sonder.cli.bench_markdown_mock PROPERTIES TIMEOUT 60
            PASS_REGULAR_EXPRESSION "wrote [^\n]*cli-bench-md\\.md")
    endif()

    # --- Help and usage (docs/CLI.md) --------------------------------------
    sonder_cli_test(bare_usage RC 2 STDERR_REGEX "Commands:.*generate.*help <command>" COMMAND ${infer})
    sonder_cli_test(help_overview RC 0 STDOUT_REGEX "Commands:.*130 cancelled" COMMAND ${infer} help)
    sonder_cli_test(help_flag_overview RC 0 STDOUT_REGEX "Commands:" COMMAND ${infer} --help)
    sonder_cli_test(unknown_command RC 2 STDERR_REGEX "unknown command 'genrate' \\(did you mean 'generate'\\?\\)"
        COMMAND ${infer} genrate)
    sonder_cli_test(help_unknown_command RC 2 STDERR_REGEX "unknown command 'nope'" COMMAND ${infer} help nope)
    set(help_commands version devices backends models generate chat bench help)
    if(sonder_cli_has_server)
        list(APPEND help_commands serve)
    endif()
    foreach(cmd IN LISTS help_commands)
        sonder_cli_test(help_${cmd} RC 0 STDOUT_REGEX "Usage:\\n  sonder-infer ${cmd}" COMMAND ${infer} help ${cmd})
        sonder_cli_test(${cmd}_dash_help RC 0 STDOUT_REGEX "Usage:\\n  sonder-infer ${cmd}"
            COMMAND ${infer} ${cmd} --help)
    endforeach()
    sonder_cli_test(generate_short_help RC 0 STDOUT_REGEX "Examples:" COMMAND ${infer} generate -h)
    # Help wins over other (even invalid) arguments.
    sonder_cli_test(help_with_bad_args RC 0 STDOUT_REGEX "Usage:" COMMAND ${infer} generate --bogus -h)
    sonder_cli_test(help_mock_rule RC 0
        STDOUT_REGEX "'mock' and 'mock:<anything>' and stops after about 16 tokens by design"
        COMMAND ${infer} help generate)

    # --- Unknown options and checked values (exit 2) -----------------------
    sonder_cli_test(unknown_option RC 2 STDERR_REGEX "error: generate: unknown option --bogus\\n"
        COMMAND ${infer} generate ${mock} --prompt hi --bogus 1)
    sonder_cli_test(did_you_mean RC 2 STDERR_REGEX "unknown option --max-token \\(did you mean --max-tokens\\?\\)"
        COMMAND ${infer} generate ${mock} --prompt hi --max-token 5)
    sonder_cli_test(bad_float RC 2 STDERR_REGEX "invalid value 'abc' for --top-p"
        COMMAND ${infer} generate ${mock} --prompt hi --top-p abc)
    sonder_cli_test(bad_integer RC 2 STDERR_REGEX "invalid value '12x' for --max-tokens"
        COMMAND ${infer} generate ${mock} --prompt hi --max-tokens 12x)
    sonder_cli_test(bad_seed RC 2 STDERR_REGEX "invalid value '-3' for --seed"
        COMMAND ${infer} generate ${mock} --prompt hi --seed -3)
    sonder_cli_test(bad_mock_delay RC 2 STDERR_REGEX "value '70000' for --mock-delay-ms is out of range"
        COMMAND ${infer} generate ${mock} --prompt hi --mock-delay-ms 70000)
    sonder_cli_test(bad_priority RC 2 STDERR_REGEX "value '17' for --priority is out of range"
        COMMAND ${infer} generate ${mock} --prompt hi --priority 17)
    sonder_cli_test(bad_workload RC 2 STDERR_REGEX "invalid value 'fast' for --workload"
        COMMAND ${infer} chat ${mock} --messages "${chat_fixture}" --workload fast)
    sonder_cli_test(bad_run_id RC 2 STDERR_REGEX "invalid value 'a b' for --run-id"
        COMMAND ${infer} generate ${mock} --prompt hi --run-id "a b")
    sonder_cli_test(bad_stats_mode RC 2 STDERR_REGEX "invalid value 'yaml' for --stats"
        COMMAND ${infer} generate ${mock} --prompt hi --stats yaml)
    sonder_cli_test(bad_telemetry_level RC 2 STDERR_REGEX "--telemetry-level"
        COMMAND ${infer} generate ${mock} --prompt hi --telemetry-level loud)
    sonder_cli_test(duplicate_option RC 2 STDERR_REGEX "--prompt given more than once"
        COMMAND ${infer} generate ${mock} --prompt a --prompt b)
    sonder_cli_test(positional_argument RC 2 STDERR_REGEX "unexpected argument 'extra'"
        COMMAND ${infer} generate ${mock} --prompt a extra)
    sonder_cli_test(missing_prompt RC 2 STDERR_REGEX "--prompt is required" COMMAND ${infer} generate ${mock})
    sonder_cli_test(missing_backend RC 2 STDERR_REGEX "--backend is required" COMMAND ${infer} generate --prompt hi)
    sonder_cli_test(unknown_backend RC 2 STDERR_REGEX "unknown backend 'mokc' \\(did you mean 'mock'\\?\\)"
        COMMAND ${infer} generate --backend mokc --prompt hi)
    sonder_cli_test(models_unknown_backend RC 2 STDERR_REGEX "unknown backend 'nope'"
        COMMAND ${infer} models --backend nope)
    if(sonder_cli_has_ollama)
        sonder_cli_test(missing_model_hint RC 2 STDERR_REGEX "sonder-infer models --backend ollama"
            COMMAND ${infer} generate --backend ollama --prompt hi)
    endif()
    if(TARGET sonder-bench)
        sonder_cli_test(bench_bad_runs RC 2 STDERR_REGEX "value '0' for --runs is out of range"
            COMMAND ${infer} bench ${mock} --corpus "${PROJECT_SOURCE_DIR}/bench/corpus/smoke.json"
                    --out "${CMAKE_CURRENT_BINARY_DIR}/unused.json" --runs 0)
        sonder_cli_test(bench_missing_corpus RC 2 STDERR_REGEX "error: bench:"
            COMMAND ${infer} bench ${mock} --corpus "${CMAKE_CURRENT_BINARY_DIR}/absent-corpus.json"
                    --out "${CMAKE_CURRENT_BINARY_DIR}/unused.json")
    endif()

    # --- Runtime failures (exit 1) -----------------------------------------
    sonder_cli_test(model_not_found RC 1 STDERR_REGEX "not_found"
        COMMAND ${infer} generate --backend mock --model notmock --prompt hi)
    sonder_cli_test(telemetry_unwritable RC 1 STDERR_REGEX "cannot open --telemetry"
        COMMAND ${infer} generate ${mock} --prompt hi
                --telemetry "${CMAKE_CURRENT_BINARY_DIR}/no-such-dir/sub/t.jsonl")

    # --- Machine-readable output -------------------------------------------
    sonder_cli_test(version_json RC 0 STDOUT_JSON STDOUT_REGEX "\"abi_version\":1" COMMAND ${infer} version --json)
    sonder_cli_test(devices_json RC 0 STDOUT_JSON STDOUT_REGEX "\"platform\":\"[^\"]+\",\"devices\":" COMMAND ${infer} devices --json)
    # The mock backend is always available; an absent Ollama is reported, not fatal.
    sonder_cli_test(backends_json RC 0 STDOUT_JSON
        STDOUT_REGEX "\"name\":\"mock\",\"available\":true"
        COMMAND ${infer} backends --json --ollama-url http://127.0.0.1:9)
    sonder_cli_test(models_mock_json RC 0 STDOUT_JSON
        STDOUT_REGEX "\"backend\":\"mock\",\"synthetic\":true,\"models\":..\"name\":\"mock:tiny\""
        COMMAND ${infer} models --backend mock --json)
    sonder_cli_test(stats_json RC 0 STDERR_JSON
        STDERR_REGEX "\"command\":\"generate\",\"backend\":\"mock\",\"model\":\"mock:tiny\",\"synthetic\":true"
        JSONL_FILE "${CMAKE_CURRENT_BINARY_DIR}/cli-stats.jsonl" JSONL_FILE_REGEX "request.completed"
        COMMAND ${infer} generate ${mock} --prompt hi --quiet --stats json
                --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-stats.jsonl")
    # --quiet with --telemetry -: stderr carries only JSON lines.
    sonder_cli_test(quiet_telemetry_jsonl RC 0 STDERR_JSONL STDERR_REGEX "\"event_type\":\"request.completed\""
        STDOUT_REGEX "[a-z]" COMMAND ${infer} generate ${mock} --prompt hi --quiet --telemetry -)
    sonder_cli_test(quiet_stats_jsonl RC 0 STDERR_JSONL STDERR_REGEX "\"command\":\"generate\""
        COMMAND ${infer} generate ${mock} --prompt hi --quiet --stats json --telemetry -)
    sonder_cli_test(stats_none RC 0 STDERR_NOT_REGEX "outcome="
        COMMAND ${infer} generate ${mock} --prompt hi --stats none)
    sonder_cli_test(mock_banner RC 0 STDERR_REGEX "MOCK BACKEND - synthetic output"
        COMMAND ${infer} generate ${mock} --prompt hi)

    # --- Environment defaults (flags win) ----------------------------------
    sonder_cli_test(env_defaults RC 0 STDERR_REGEX "outcome=completed"
        ENV SONDER_INFER_BACKEND=mock SONDER_INFER_MODEL=mock:tiny
        COMMAND ${infer} generate --prompt hi)
    sonder_cli_test(flags_override_env RC 0 STDERR_REGEX "outcome=completed"
        ENV SONDER_INFER_BACKEND=bogus SONDER_INFER_MODEL=bogus
        COMMAND ${infer} generate ${mock} --prompt hi)
    sonder_cli_test(env_bad_backend RC 2 STDERR_REGEX "unknown backend 'bogus'"
        ENV SONDER_INFER_BACKEND=bogus COMMAND ${infer} generate --prompt hi)
    sonder_cli_test(env_model_from_env RC 1 STDERR_REGEX "only serves models named mock"
        ENV SONDER_INFER_MODEL=not-a-mock-model COMMAND ${infer} generate --backend mock --prompt hi)
    sonder_cli_test(mock_default_model RC 0 STDERR_REGEX "\"model\":\"mock\""
        COMMAND ${infer} generate --backend mock --prompt hi --stats json)
    if(sonder_cli_has_ollama)
        # OLLAMA_HOST host:port form reaches the backend (nothing listens on port 9).
        sonder_cli_test(env_ollama_host RC 0 STDOUT_REGEX "at http://127.0.0.1:9"
            ENV OLLAMA_HOST=127.0.0.1:9 COMMAND ${infer} backends)
        sonder_cli_test(env_ollama_url_flag_wins RC 0 STDOUT_REGEX "at http://127.0.0.1:8"
            ENV OLLAMA_HOST=127.0.0.1:9 COMMAND ${infer} backends --ollama-url http://127.0.0.1:8)
        sonder_cli_test(ollama_unreachable_hint RC 1 STDERR_REGEX "hint: Ollama is not reachable"
            COMMAND ${infer} models --backend ollama --ollama-url http://127.0.0.1:9)
        # --ollama-allow-remote never permits plain http:// to a remote host
        # (192.0.2.1 is TEST-NET-1; the check fires before any connection).
        sonder_cli_test(ollama_remote_plain_http RC 2
            STDERR_REGEX "--ollama-allow-remote refuses plain http:// to a non-loopback host"
            COMMAND ${infer} models --backend ollama --ollama-url http://192.0.2.1:11434 --ollama-allow-remote)
        sonder_cli_test(ollama_remote_plain_http_env RC 2 STDERR_REGEX "refuses plain http://"
            ENV OLLAMA_HOST=192.0.2.1:11434 COMMAND ${infer} models --backend ollama --ollama-allow-remote)
        # A loopback URL is still allowed with the flag (warning, then the
        # usual unreachable error).
        sonder_cli_test(ollama_allow_remote_loopback RC 1
            STDERR_REGEX "warning: --ollama-allow-remote.*https:// only.*hint: Ollama is not reachable"
            COMMAND ${infer} models --backend ollama --ollama-url http://127.0.0.1:9 --ollama-allow-remote)
    endif()
    sonder_cli_test(allow_remote_plain_http_any_backend RC 2 STDERR_REGEX "error: generate: --ollama-allow-remote refuses"
        COMMAND ${infer} generate ${mock} --prompt hi --ollama-url http://192.0.2.1 --ollama-allow-remote)

    # --- Ctrl-C: exit 130 (POSIX shells only; SIGINT from a script) -----------
    # The stats line proves the handler cancelled the request: a process
    # killed by the default SIGINT action would also show status 130.
    if(UNIX)
        add_test(NAME sonder.cli.generate_sigint_exit_130
            COMMAND sh -c "\"$1\" generate --backend mock --model mock:tiny --prompt hi --max-tokens 16 \
--mock-delay-ms 500 >/dev/null 2>\"$2\" & p=$!; sleep 1; kill -INT $p; wait $p; rc=$?; \
test $rc -eq 130 || { echo \"exit $rc, expected 130\"; exit 1; }; \
grep -q 'outcome=cancelled' \"$2\" || { echo 'no outcome=cancelled stats line'; cat \"$2\"; exit 1; }"
                sh $<TARGET_FILE:sonder-infer> "${CMAKE_CURRENT_BINARY_DIR}/cli-sigint.err")
        set_tests_properties(sonder.cli.generate_sigint_exit_130 PROPERTIES TIMEOUT 60)
    endif()

    # --- Chat through Session::chat (request telemetry) ---------------------
    sonder_cli_test(chat_telemetry RC 0
        JSONL_FILE "${CMAKE_CURRENT_BINARY_DIR}/cli-chat.jsonl"
        JSONL_FILE_REGEX "\"event_type\":\"request.queued\"[^\\n]*\"attributes\":\\{\"kind\":\"chat\""
        JSONL_FILE_REGEX2 "\"event_type\":\"request.completed\""
        COMMAND ${infer} chat ${mock} --messages "${chat_fixture}" --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-chat.jsonl")
    # The fixture sends two turns ("hello sonder" and "//not a command", sent
    # as "/not a command"), then /exit; the "never sent" line after /exit
    # would be a third request.completed. Piped stdin gets no prompt, so the
    # dangling-prompt fix is covered by tests/test_cli_repl.cpp.
    sonder_cli_test(chat_repl RC 0 STDIN "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_repl_input.txt"
        STDOUT_REGEX "turns=1 failed=0 cancelled=0.*/stats +last turn and session totals"
        JSONL_FILE "${CMAKE_CURRENT_BINARY_DIR}/cli-chat-repl.jsonl"
        JSONL_FILE_REGEX "\"event_type\":\"request.queued\"[^\\n]*\"kind\":\"chat\""
        JSONL_FILE_REGEX2 "\"event_type\":\"request.completed\""
        JSONL_FILE_COUNT_REGEX "\"event_type\":\"request.completed\"" JSONL_FILE_COUNT 2
        COMMAND ${infer} chat ${mock} --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-chat-repl.jsonl")
    sonder_cli_test(chat_repl_stats_json RC 0 STDIN "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_repl_input.txt"
        STDERR_JSONL STDERR_REGEX "\"turn\":2"
        COMMAND ${infer} chat ${mock} --quiet --stats json)

    # --- Correlation and scheduling flags ----------------------------------
    sonder_cli_test(generate_correlation RC 0
        JSONL_FILE "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-generate.jsonl"
        JSONL_FILE_REGEX "\"run_id\":\"cli-run-1\",\"request_id\":\"req-[0-9a-f]+\",\"agent_id\":\"agent-7\",\"task_id\":\"task-9\""
        JSONL_FILE_REGEX2 "\"kind\":\"generate\",\"priority\":-3,\"workload\":\"research_worker\""
        COMMAND ${infer} generate ${mock} --prompt hi --run-id cli-run-1 --agent-id agent-7 --task-id task-9
                --workload research_worker --priority -3 --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-generate.jsonl")
    sonder_cli_test(chat_correlation RC 0
        JSONL_FILE "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-chat.jsonl"
        JSONL_FILE_REGEX "\"run_id\":\"chat.run:2\",\"request_id\":\"req-[0-9a-f]+\",\"agent_id\":\"a_1\",\"task_id\":\"t-1\""
        JSONL_FILE_REGEX2 "\"kind\":\"chat\",\"priority\":16,\"workload\":\"interactive_user\""
        COMMAND ${infer} chat ${mock} --messages "${chat_fixture}" --run-id chat.run:2 --agent-id a_1 --task-id t-1
                --workload interactive_user --priority 16 --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-chat.jsonl")
    if(TARGET sonder-bench)
        sonder_cli_test(bench_correlation RC 0
            JSONL_FILE "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-bench.jsonl"
            JSONL_FILE_REGEX "\"run_id\":\"bench-run-1\",\"request_id\":\"req-[0-9a-f]+\",\"agent_id\":\"bench-agent\",\"task_id\":null"
            JSONL_FILE_REGEX2 "\"priority\":4,\"workload\":\"background_indexing\""
            COMMAND ${infer} bench ${mock} --corpus "${PROJECT_SOURCE_DIR}/bench/corpus/smoke.json"
                    --out "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-bench.json" --warmup 0 --runs 1 --quiet
                    --run-id bench-run-1 --agent-id bench-agent --workload background_indexing --priority 4
                    --telemetry "${CMAKE_CURRENT_BINARY_DIR}/cli-corr-bench.jsonl")
    endif()

    # --- serve dispatch ------------------------------------------------------
    if(sonder_cli_has_server)
        sonder_cli_test(serve_help RC 0 STDOUT_REGEX "sonder-infer serve - local HTTP API.*--ready-file"
            COMMAND ${infer} serve --help)
        sonder_cli_test(serve_unknown_option RC 2 STDERR_REGEX "unknown option --prot \\(did you mean --port\\?\\)"
            COMMAND ${infer} serve --backend mock --prot 1)
        sonder_cli_test(serve_bad_port RC 2 STDERR_REGEX "--port must be an integer"
            COMMAND ${infer} serve --backend mock --port 99999)
        sonder_cli_test(serve_remote_needs_token RC 2 STDERR_REGEX "token"
            COMMAND ${infer} serve --backend mock --host 0.0.0.0 --port 0)
        sonder_cli_test(serve_help_wins RC 0 STDOUT_REGEX "sonder-infer serve - local HTTP API"
            COMMAND ${infer} serve --bogus --help)
        sonder_cli_test(serve_remote_plain_http RC 2 STDERR_REGEX "error: serve: --ollama-allow-remote refuses plain http://"
            COMMAND ${infer} serve --backend mock --port 0 --ollama-url=http://192.0.2.1 --ollama-allow-remote)
        sonder_cli_test(serve_remote_plain_http_env RC 2 STDERR_REGEX "refuses plain http://"
            ENV SONDER_OLLAMA_URL=http://ollama.example:11434
            COMMAND ${infer} serve --backend mock --port 0 --ollama-allow-remote)
    else()
        sonder_cli_test(serve_not_built RC 2 STDERR_REGEX "does not include the server module"
            COMMAND ${infer} serve --backend mock)
    endif()

    # --- sonder-bench -------------------------------------------------------
    if(TARGET sonder-bench)
        set(bench_tool $<TARGET_FILE:sonder-bench>)
        sonder_cli_test(bench_tool_no_args RC 2 STDERR_REGEX "Usage:\\n  sonder-bench --model NAME"
            COMMAND ${bench_tool})
        sonder_cli_test(bench_tool_help RC 0 STDOUT_REGEX "MOCK BACKEND banner.*Exit status: 0 ok"
            COMMAND ${bench_tool} --help)
        sonder_cli_test(bench_tool_unknown_option RC 2 STDERR_REGEX "unknown option --run \\(did you mean --runs\\?\\)"
            COMMAND ${bench_tool} --backend mock --model mock:tiny --run 1)
        sonder_cli_test(bench_tool_bad_budget RC 2 STDERR_REGEX "invalid value 'soon' for --budget-seconds"
            COMMAND ${bench_tool} --backend mock --model mock:tiny --budget-seconds soon)
    endif()
endif()
