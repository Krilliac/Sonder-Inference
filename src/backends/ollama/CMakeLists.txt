# Module: Ollama compatibility backend (owner: feat/ollama-bench).
sonder_module_sources(ollama_backend.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_OLLAMA_BACKEND)

sonder_add_module_tests(ollama tests/test_ollama_adapter.cpp)
