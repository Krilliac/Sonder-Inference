#pragma once

#include <string>

#include "fake_ollama_server.hpp"
#include "sonder/inference/backends/ollama.hpp"

namespace sonder_test {

inline std::string fixture(const char* name) {
    return read_file(std::string(SONDER_OLLAMA_FIXTURE_DIR) + "/" + name);
}

inline sonder::inference::ollama::OllamaConfig config_for(const FakeOllamaServer& s) {
    sonder::inference::ollama::OllamaConfig c;
    c.base_url = s.base_url();
    c.connect_timeout = std::chrono::milliseconds(1000);
    c.request_timeout = std::chrono::milliseconds(8000);
    return c;
}

}  // namespace sonder_test
