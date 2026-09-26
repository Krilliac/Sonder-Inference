# Sonder sampling module: composable sampler chain over logits.
# Picked up automatically by the root CMakeLists (SONDER_MODULE_DIRS); adds
# its sources to sonder_inference and registers doctest cases as
# sonder.sampling.<case>. See src/sampling/README.md.

sonder_module_sources(
    src/types.cpp
    src/rng.cpp
    src/config.cpp
    src/constraint.cpp
    src/samplers.cpp
    src/chain.cpp
    src/stop.cpp
    src/core_bridge.cpp
)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_SAMPLER_CHAIN)

sonder_add_module_tests(sampling
    tests/test_config.cpp
    tests/test_rng.cpp
    tests/test_samplers.cpp
    tests/test_chain.cpp
    tests/test_stop.cpp
    tests/test_constraint.cpp
    tests/test_core_bridge.cpp
)
