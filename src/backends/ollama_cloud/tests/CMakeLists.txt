# Ollama Cloud tests: a fake in-process transport replaces HTTPS, so no
# network, TLS endpoint or API key is needed.
sonder_add_module_tests(ollama_cloud
    ollama_cloud_tests.cpp)
