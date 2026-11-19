# Module: weight-content digest manifests (docs/integration/weight-manifest.md).
#
# Streaming SHA-256 manifests for multi-file model directories, a verifier with
# deterministic sampling, and the `sonder-infer weights` subcommand body. Uses the
# in-house SHA-256 from src/server (no third-party code), so it is only built when
# that module is present. Consumers test SONDER_HAS_WEIGHT_MANIFEST.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../server/src/sha256.cpp")
    message(STATUS "Sonder module src/artifacts skipped: needs src/server (sha256)")
    return()
endif()
sonder_module_sources(
    src/weight_manifest.cpp
    src/weights_cli.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_WEIGHT_MANIFEST)

sonder_add_module_tests(artifacts
    tests/test_weight_manifest.cpp)
