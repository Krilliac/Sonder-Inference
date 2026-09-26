// Ollama wire-format helpers declared in ollama_protocol.hpp (internal API,
// unit-tested without a server).
#include "ollama_protocol.hpp"

#include "sonder/inference/backends/ollama.hpp"

namespace sonder::inference::ollama {

json::Object build_generate_body(const std::string& model, const GenerateRequest& request,
                                 const std::string& keep_alive) {
    GenerateParams p;
    p.model = model;
    p.prompt = request.prompt;
    p.keep_alive = keep_alive;
    p.options = sampling_to_options(request.sampling);
    return build_generate_body(p).as_object();
}

Result<StreamLine> parse_stream_line(std::string_view line, GenerateStats& stats) {
    StreamLine out;
    if (line.find_first_not_of(" \t\r") == std::string_view::npos) {
        return out;
    }
    auto parsed = json::parse(line);
    if (!parsed.ok()) {
        return Status(ErrorCode::protocol_error, "ollama: malformed NDJSON line: " + parsed.status().message());
    }
    const json::Value& v = parsed.value();
    // Same rules as StreamDecoder::on_line: each line must be an object, and
    // only a non-null "error" field is a server-side error.
    if (!v.is_object()) {
        return Status(ErrorCode::protocol_error, "ollama: malformed NDJSON line: expected a JSON object");
    }
    if (const json::Value* err = v.find("error"); err != nullptr && !err->is_null()) {
        return Status(ErrorCode::backend_error, "ollama: " + (err->is_string() ? err->as_string() : err->dump()));
    }
    if (const json::Value* r = v.find("response")) {
        out.piece = r->as_string();
    }
    if (const json::Value* d = v.find("done")) {
        out.done = d->as_bool();
    }
    if (out.done) {
        auto u64 = [&](const char* key) -> std::uint64_t {
            const json::Value* f = v.find(key);
            const auto n = f ? f->as_int(0) : 0;
            return n > 0 ? static_cast<std::uint64_t>(n) : 0;
        };
        if (v.find("eval_count") || v.find("prompt_eval_count")) {
            stats.token_counts_from_backend = true;
        }
        stats.prompt_tokens = u64("prompt_eval_count");
        stats.completion_tokens = u64("eval_count");
        stats.load_ns = u64("load_duration");
        stats.prompt_eval_ns = u64("prompt_eval_duration");
        stats.eval_ns = u64("eval_duration");
        const json::Value* reason = v.find("done_reason");
        const std::string r = reason ? reason->as_string() : std::string();
        stats.stop_reason = r == "length" ? StopReason::max_tokens : StopReason::end_of_sequence;
    }
    return out;
}

ModelDescriptor descriptor_from_tag(const json::Value& entry) {
    ModelDescriptor d;
    d.name = entry.find("name") ? entry.find("name")->as_string() : std::string();
    if (d.name.empty() && entry.find("model")) {
        d.name = entry.find("model")->as_string();
    }
    d.backend = kOllamaBackendName;
    if (const json::Value* size = entry.find("size")) {
        const auto n = size->as_int(0);
        d.size_bytes = n > 0 ? static_cast<std::uint64_t>(n) : 0;
    }
    if (const json::Value* details = entry.find("details")) {
        auto str = [&](const char* key) {
            const json::Value* f = details->find(key);
            return f ? f->as_string() : std::string();
        };
        d.format = str("format");
        d.family = str("family");
        d.parameter_size = str("parameter_size");
        d.quantization = str("quantization_level");
    }
    return d;
}

}  // namespace sonder::inference::ollama
