# Reliability harness: supervisor fault injection against a real child process,
# plus the self-test of scripts/reliability_run.py. No network beyond loopback
# and no model weights.

# Fake upstream used as the supervised child (not a model).
add_executable(sonder_fault_child fault_child.cpp)
target_link_libraries(sonder_fault_child PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(sonder_fault_child PRIVATE ws2_32)
endif()
sonder_set_warnings(sonder_fault_child)

# The supervisor lives in the llamaserver module; skip when it is not built.
if(TARGET sonder_llamaserver_tests)
    # Same steps as sonder_add_module_tests, inline so RUN_SERIAL can be set:
    # these cases assert wall-clock bounds and port/pid state, and the
    # hardening workflow runs ctest -j 4 under sanitizers.
    if(SONDER_BUILD_TESTS)
        add_executable(sonder_fault_tests test_fault_injection.cpp)
        target_link_libraries(sonder_fault_tests PRIVATE sonder::inference sonder_doctest_main)
        target_include_directories(sonder_fault_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
        sonder_set_warnings(sonder_fault_tests)
        doctest_discover_tests(sonder_fault_tests TEST_PREFIX "sonder.fault."
            PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
    endif()
    if(TARGET sonder_fault_tests)
        add_dependencies(sonder_fault_tests sonder_fault_child)
        target_compile_definitions(sonder_fault_tests PRIVATE
            "SONDER_FAULT_CHILD_PATH=\"$<TARGET_FILE:sonder_fault_child>\"")
        if(WIN32)
            target_link_libraries(sonder_fault_tests PRIVATE ws2_32)
            target_compile_definitions(sonder_fault_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
        endif()
    endif()
else()
    message(STATUS "sonder_llamaserver_tests not built: sonder.fault.* tests not registered")
endif()

# Self-test of the reliability driver against its built-in fake server.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME sonder.reliability_run.self_test
        COMMAND "${Python3_EXECUTABLE}" -I "${PROJECT_SOURCE_DIR}/scripts/reliability_run.py" --self-test)
    set_tests_properties(sonder.reliability_run.self_test PROPERTIES
        TIMEOUT 900 RUN_SERIAL TRUE LABELS "stress;synthetic")
else()
    message(STATUS "Python3 not found: sonder.reliability_run.self_test not registered")
endif()
