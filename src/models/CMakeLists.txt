# Module: SI model store that serves an existing Ollama model library in place
# (docs/integration/model-store.md). Read-only manifest/blob reader, name
# resolution, params/template/system records, opt-in digest verification, a
# ModelRegistry for backends, and `sonder-infer models import-ollama`.
# Core-only dependencies (JSON, SHA-256); consumers test SONDER_HAS_MODEL_STORE.
sonder_module_sources(
    src/model_store.cpp
    src/models_cli.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_MODEL_STORE)

sonder_add_module_tests(model_store
    tests/test_model_store.cpp)
