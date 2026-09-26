// Ollama wire-format helpers (internal; unit-tested without a server).
#pragma once

#include <string>
#include <string_view>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::ollama {

// Body for POST /api/generate with stream=true and sampling mapped to options.
json::Object build_generate_body(const std::string& model, const GenerateRequest& request,
                                 const std::string& keep_alive);

struct StreamLine {
    std::string piece;  // generated text in this line (may be empty)
    bool done = false;
};

// Parses one NDJSON line of a streaming /api/generate response, updating
// stats from the final (done=true) line. Server-side errors become
// backend_error; malformed JSON becomes protocol_error.
Result<StreamLine> parse_stream_line(std::string_view line, GenerateStats& stats);

// Maps one entry of GET /api/tags "models" to a descriptor.
ModelDescriptor descriptor_from_tag(const json::Value& entry);

}  // namespace sonder::inference::ollama
