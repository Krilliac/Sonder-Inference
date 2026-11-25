// MOCK BACKEND - deterministic output for tests and harness development only.
// It performs no inference and must never be used for quality/perf claims.
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <thread>

#include "engine/accounting_tokens.hpp"
#include "sonder/inference/backends.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference {
namespace {

constexpr const char* kVocabulary[] = {
    "sonder", "engine", "session", "token",  "cache",   "prefix", "decode", "prefill",
    "batch",  "device", "model",   "stream", "policy",  "memory", "sample", "adapter",
    "kernel", "block",  "page",    "queue",  "latency", "budget", "node",   "context",
};
constexpr std::size_t kVocabularySize = sizeof(kVocabulary) / sizeof(kVocabulary[0]);

std::uint64_t fnv1a(std::string_view s) {
    std::uint64_t h = 1469598103934665603ull;
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return h;
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

constexpr std::array<std::uint8_t, 8> kMockCheckpointMagic = {'S', 'N', 'D', 'R', 'C', 'K', 'P', '1'};
constexpr std::size_t kMockCheckpointBytes = 64;

void write_u64_le(std::uint8_t* dst, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu);
    }
}

std::uint64_t read_u64_le(const std::uint8_t* src) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(src[i]) << (i * 8);
    }
    return v;
}

std::uint64_t fnv1a_bytes(std::span<const std::uint8_t> data) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    for (const std::uint8_t b : data) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

std::uint64_t hash_token_prefix(std::span<const TokenId> tokens) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    for (const TokenId t : tokens) {
        const auto u = static_cast<std::uint32_t>(t);
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<std::uint8_t>((u >> (b * 8)) & 0xFFu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

std::uint64_t count_words(std::string_view s) {
    std::uint64_t n = 0;
    bool in_word = false;
    for (const char c : s) {
        const bool space = c == ' ' || c == '\t' || c == '\n' || c == '\r';
        if (!space && !in_word) {
            ++n;
        }
        in_word = !space;
    }
    return n;
}

void attach_mock_logprobs(TokenChunk& chunk, std::string_view token_text, std::uint64_t state,
                          const SamplingConfig& sampling) {
    if (!sampling.logprobs) {
        return;
    }
    const float primary_lp = -0.05f - static_cast<float>(state % 120u) * 0.01f;
    chunk.logprob = primary_lp;
    if (sampling.top_logprobs > 0) {
        const std::size_t k = static_cast<std::size_t>(sampling.top_logprobs);
        chunk.top_logprobs.reserve(k);
        chunk.top_logprobs.push_back(TopLogprob{std::string(token_text), primary_lp});
        std::size_t alt_idx = static_cast<std::size_t>(splitmix64(state) % kVocabularySize);
        for (std::size_t j = 1; j < k; ++j) {
            std::string alt_tok = kVocabulary[(alt_idx + j) % kVocabularySize];
            if (alt_tok == token_text) {
                alt_tok = kVocabulary[(alt_idx + j + 1) % kVocabularySize];
            }
            const float alt_lp = primary_lp - 0.35f * static_cast<float>(j);
            chunk.top_logprobs.push_back(TopLogprob{std::move(alt_tok), alt_lp});
        }
    }
}

std::string lower_ascii(std::string_view sv) {
    std::string out(sv);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

std::string build_mock_tool_arguments(const ToolDefinition& tool) {
    if (tool.parameters_json.empty()) {
        return "{}";
    }
    auto parsed = json::parse(tool.parameters_json);
    if (!parsed.ok() || !parsed.value().is_object()) {
        return "{}";
    }
    const json::Value* props = parsed.value().find("properties");
    if (!props || !props->is_object() || props->as_object().empty()) {
        return "{}";
    }
    std::vector<std::string> wanted;
    if (const json::Value* req = parsed.value().find("required"); req && req->is_array()) {
        for (const auto& item : req->as_array()) {
            if (item.is_string()) {
                wanted.push_back(item.as_string());
            }
        }
    }
    if (wanted.empty()) {
        wanted.push_back(props->as_object().begin()->first);
    }
    json::Object args;
    for (const auto& key : wanted) {
        const json::Value* prop = props->find(key);
        if (!prop || !prop->is_object()) {
            args.set(key, "mock");
            continue;
        }
        if (const json::Value* en = prop->find("enum"); en && en->is_array() && !en->as_array().empty()) {
            args.set(key, en->as_array().front());
            continue;
        }
        if (const json::Value* c = prop->find("const"); c != nullptr) {
            args.set(key, *c);
            continue;
        }
        const json::Value* t = prop->find("type");
        const std::string type_name = (t && t->is_string()) ? t->as_string() : std::string("string");
        if (type_name == "integer") {
            args.set(key, 42);
        } else if (type_name == "number") {
            args.set(key, 3.14);
        } else if (type_name == "boolean") {
            args.set(key, true);
        } else if (type_name == "array") {
            args.set(key, json::Array{});
        } else if (type_name == "object") {
            args.set(key, json::Object{});
        } else {
            args.set(key, "mock_" + key);
        }
    }
    return json::Value(std::move(args)).dump();
}

// Sleeps for `d` while polling the token; returns false if cancelled.
bool cancellable_sleep(std::chrono::microseconds d, const CancellationToken& cancel) {
    const auto deadline = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < deadline) {
        if (cancel.cancelled()) {
            return false;
        }
        const auto remaining = deadline - std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::min<std::chrono::steady_clock::duration>(remaining, std::chrono::milliseconds(1)));
    }
    return !cancel.cancelled();
}

// Token-level mock: vocabulary = kVocabulary plus an end-of-sequence token.
// Logits are a pure function of (prompt, model, position, previous token), so
// the output depends only on the engine's sampler and seed.
class MockTokenStream final : public TokenStream {
public:
    static constexpr TokenId kEos = static_cast<TokenId>(kVocabularySize);

    MockTokenStream(const GenerateRequest& request, const std::string& model_name, MockBackendOptions options)
        : prompt_(detail::accounting_tokenize(request.prompt)),
          state_(fnv1a(request.prompt) ^ fnv1a(model_name)),
          options_(options),
          logits_(kVocabularySize + 1, 0.0f) {}

    std::size_t vocab_size() const override { return logits_.size(); }
    const std::vector<TokenId>& prompt_tokens() const override { return prompt_; }

    Result<std::span<const float>> next_logits(const CancellationToken& cancel) override {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }
        if (options_.token_delay.count() > 0 && !cancellable_sleep(options_.token_delay, cancel)) {
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }
        if (options_.fail_after_tokens >= 0 && position_ == static_cast<std::uint64_t>(options_.fail_after_tokens)) {
            return Status(ErrorCode::backend_error,
                          "mock backend injected failure at token " + std::to_string(position_));
        }
        const std::uint64_t step = splitmix64(state_ + position_ * 0x9E37ull);
        for (std::size_t i = 0; i < kVocabularySize; ++i) {
            const std::uint64_t r = splitmix64(step ^ (static_cast<std::uint64_t>(i) * 0xD6E8FEB86659FD93ull));
            // Uniform in [-4, 4).
            logits_[i] = static_cast<float>(static_cast<double>(r >> 40) / static_cast<double>(1ull << 24) * 8.0 - 4.0);
        }
        const auto natural = static_cast<std::uint64_t>(std::max<std::int32_t>(1, options_.default_completion_tokens));
        logits_[kVocabularySize] = position_ + 1 >= natural ? 16.0f : -8.0f;
        if (options_.ban_all_tokens) {
            std::fill(logits_.begin(), logits_.end(), -std::numeric_limits<float>::infinity());
        }
        return std::span<const float>(logits_);
    }

    Status accept(TokenId token, std::string& piece) override {
        if (token < 0 || static_cast<std::size_t>(token) >= logits_.size()) {
            return Status(ErrorCode::invalid_argument, "token id out of range: " + std::to_string(token));
        }
        piece.clear();
        if (token != kEos) {
            if (position_ > 0) {
                piece.push_back(' ');
            }
            piece += kVocabulary[static_cast<std::size_t>(token)];
        }
        state_ = splitmix64(state_ ^ (static_cast<std::uint64_t>(token) + 1));
        ++position_;
        return Status::success();
    }

    bool is_end_of_generation(TokenId token) const override { return token == kEos; }

private:
    std::vector<TokenId> prompt_;
    std::uint64_t state_;
    MockBackendOptions options_;
    std::vector<float> logits_;
    std::uint64_t position_ = 0;
};

class MockModel final : public BackendModel {
public:
    MockModel(ModelDescriptor d, MockBackendOptions o) : descriptor_(std::move(d)), options_(o) {}

    const ModelDescriptor& descriptor() const override { return descriptor_; }

    Result<std::vector<TokenId>> tokenize(std::string_view text) override { return detail::accounting_tokenize(text); }

    Result<std::string> detokenize(std::span<const TokenId> tokens) override {
        std::string out;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) out.push_back(' ');
            const auto idx = static_cast<std::size_t>(tokens[i] < 0 ? -tokens[i] : tokens[i]) % kVocabularySize;
            out += kVocabulary[idx];
        }
        return out;
    }

    Result<EmbedResult> embed(const EmbedRequest& request, const CancellationToken& cancel) override {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }
        if (request.inputs.empty()) {
            return Status(ErrorCode::invalid_argument, "embed inputs must be non-empty");
        }
        const auto t0 = std::chrono::steady_clock::now();
        constexpr std::size_t kMockDims = 16;
        EmbedResult result;
        result.embeddings.reserve(request.inputs.size());
        for (const std::string& input : request.inputs) {
            if (cancel.cancelled()) {
                return Status(ErrorCode::cancelled, "cancelled by caller");
            }
            const auto toks = detail::accounting_tokenize(input);
            result.prompt_tokens += toks.size();
            std::uint64_t seed = fnv1a(input) ^ fnv1a(descriptor_.name);
            std::vector<float> vec(kMockDims, 0.0f);
            double norm_sq = 0.0;
            for (std::size_t i = 0; i < kMockDims; ++i) {
                seed = splitmix64(seed + i * 0x9E3779B97F4A7C15ull);
                const float v = static_cast<float>(static_cast<double>(seed >> 40) / static_cast<double>(1ull << 24) * 2.0 - 1.0);
                vec[i] = v;
                norm_sq += static_cast<double>(v) * static_cast<double>(v);
            }
            if (request.normalize && norm_sq > 0.0) {
                const float inv = static_cast<float>(1.0 / std::sqrt(norm_sq));
                for (float& v : vec) v *= inv;
            }
            result.embeddings.push_back(std::move(vec));
        }
        result.eval_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        return result;
    }

    Result<std::unique_ptr<TokenStream>> open_token_stream(const GenerateRequest& request) override {
        if (!options_.token_logits) {
            return Status(ErrorCode::unsupported, "mock backend created without token_logits");
        }
        apply_checkpoint_exchange(request);
        return std::unique_ptr<TokenStream>(std::make_unique<MockTokenStream>(request, descriptor_.name, options_));
    }

    Result<RecurrentStateSnapshot> capture_recurrent_state(std::uint64_t token_position) override {
        if (!options_.enable_recurrent_checkpoints) {
            return Status(ErrorCode::unsupported, "mock backend created without enable_recurrent_checkpoints");
        }
        std::uint64_t prefix_hash = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            prefix_hash = last_prompt_hash_;
        }
        auto built = build_checkpoint_bytes(token_position, prefix_hash);
        if (!built.ok()) {
            return built.status();
        }
        RecurrentStateSnapshot snapshot;
        snapshot.token_position = token_position;
        snapshot.bytes = std::move(built).value();
        return snapshot;
    }

    Status restore_recurrent_state(std::span<const std::uint8_t> bytes) override {
        if (!options_.enable_recurrent_checkpoints) {
            return Status(ErrorCode::unsupported, "mock backend created without enable_recurrent_checkpoints");
        }
        std::uint64_t restored_pos = 0;
        std::uint64_t restored_prefix_hash = 0;
        if (Status st = validate_checkpoint_bytes(bytes, restored_pos, restored_prefix_hash); !st.ok()) {
            return st;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_prompt_hash_ = restored_prefix_hash;
        }
        return Status::success();
    }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        GenerateStats stats;
        stats.prompt_tokens = count_words(request.prompt);
        stats.token_counts_from_backend = true;
        const auto t0 = std::chrono::steady_clock::now();

        if (cancel.cancelled()) {
            stats.stop_reason = StopReason::cancelled;
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }

        apply_checkpoint_exchange(request);
        if (request.checkpoint != nullptr && request.checkpoint->restored) {
            stats.cached_tokens = request.checkpoint->restore_token_position;
        }

        // Greedy decoding ignores the seed, like a real sampler would.
        std::uint64_t state = fnv1a(request.prompt) ^ fnv1a(descriptor_.name);
        if (request.sampling.temperature > 0.0f) {
            state ^= splitmix64(request.sampling.seed.value_or(0));
        }

        const std::int32_t natural = std::max<std::int32_t>(1, options_.default_completion_tokens);
        const std::int32_t limit = std::min(natural, request.sampling.max_tokens);
        std::string emitted;
        std::string piece;

        for (std::int32_t i = 0; i < limit; ++i) {
            if (cancel.cancelled()) {
                stats.stop_reason = StopReason::cancelled;
                return Status(ErrorCode::cancelled, "cancelled by caller");
            }
            if (options_.token_delay.count() > 0 && !cancellable_sleep(options_.token_delay, cancel)) {
                return Status(ErrorCode::cancelled, "cancelled by caller");
            }
            if (options_.fail_after_tokens >= 0 && i == options_.fail_after_tokens) {
                return Status(ErrorCode::backend_error, "mock backend injected failure at token " + std::to_string(i));
            }
            state = splitmix64(state + static_cast<std::uint64_t>(i));
            piece.clear();
            if (i > 0) {
                piece.push_back(' ');
            }
            piece += kVocabulary[state % kVocabularySize];

            // Stop-sequence handling over the accumulated text.
            bool hit_stop = false;
            if (!request.sampling.stop.empty()) {
                const std::string combined = emitted + piece;
                std::size_t earliest = std::string::npos;
                const std::string* matched = nullptr;
                for (const auto& s : request.sampling.stop) {
                    const auto p = combined.find(s);
                    if (p != std::string::npos &&
                        (p < earliest || (p == earliest && matched && s.size() > matched->size()))) {
                        earliest = std::min(earliest, p);
                        matched = &s;
                    }
                }
                if (earliest != std::string::npos) {
                    hit_stop = true;
                    if (matched) {
                        stats.matched_stop = *matched;
                    }
                    piece = earliest > emitted.size() ? combined.substr(emitted.size(), earliest - emitted.size())
                                                      : std::string();
                }
            }
            if (!piece.empty()) {
                emitted += piece;
                ++stats.completion_tokens;
                TokenChunk chunk{piece, stats.chunks++};
                attach_mock_logprobs(chunk, piece, state, request.sampling);
                if (on_chunk && !on_chunk(chunk)) {
                    stats.stop_reason = StopReason::callback;
                    break;
                }
            }
            if (hit_stop) {
                stats.stop_reason = StopReason::stop_sequence;
                break;
            }
        }
        if (stats.stop_reason == StopReason::none) {
            stats.stop_reason = limit < natural ? StopReason::max_tokens : StopReason::end_of_sequence;
        }
        stats.eval_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        return stats;
    }

    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        if (Status st = validate_chat_messages(request.messages); !st.ok()) {
            return st;
        }
        if (request.tools.empty()) {
            return BackendModel::chat(request, cancel, on_chunk);
        }
        const ToolChoiceMode mode = request.tool_choice ? request.tool_choice->mode : ToolChoiceMode::auto_choice;
        if (mode == ToolChoiceMode::none) {
            return BackendModel::chat(request, cancel, on_chunk);
        }

        const ToolDefinition* chosen = nullptr;
        const ChatMessage& last_msg = request.messages.back();
        const std::string last_lower = lower_ascii(last_msg.content);

        if (mode == ToolChoiceMode::function) {
            for (const auto& t : request.tools) {
                if (t.name == request.tool_choice->function_name) {
                    chosen = &t;
                    break;
                }
            }
            if (chosen == nullptr) {
                return Status(ErrorCode::invalid_argument,
                              "mock: requested tool_choice function '" + request.tool_choice->function_name +
                                  "' is not in tools");
            }
        } else if (mode == ToolChoiceMode::required) {
            for (const auto& t : request.tools) {
                if (last_lower.find(lower_ascii(t.name)) != std::string::npos) {
                    chosen = &t;
                    break;
                }
            }
            if (chosen == nullptr) {
                chosen = &request.tools.front();
            }
        } else if (last_msg.role == "user") {
            // Auto mode: trigger tool call deterministically when user prompt
            // names a tool or asks to use/call a tool.
            for (const auto& t : request.tools) {
                if (last_lower.find(lower_ascii(t.name)) != std::string::npos) {
                    chosen = &t;
                    break;
                }
            }
            if (chosen == nullptr && (last_lower.find("tool") != std::string::npos ||
                                      last_lower.find("function") != std::string::npos ||
                                      last_lower.find("weather") != std::string::npos ||
                                      last_lower.find("calculate") != std::string::npos)) {
                chosen = &request.tools.front();
            }
        }

        if (chosen == nullptr) {
            return BackendModel::chat(request, cancel, on_chunk);
        }

        const auto t0 = std::chrono::steady_clock::now();
        GenerateStats stats;
        const std::string prompt = format_chat_prompt(request.messages);
        stats.prompt_tokens = count_words(prompt);
        stats.token_counts_from_backend = true;
        if (cancel.cancelled()) {
            stats.stop_reason = StopReason::cancelled;
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }

        const std::string args_json = build_mock_tool_arguments(*chosen);
        const std::uint64_t seed = fnv1a(prompt) ^ fnv1a(chosen->name);
        const std::string call_id = "call_mock_" + std::to_string(seed);
        // Emit native structured tool_call_deltas across two chunks (header then
        // arguments) matching MockBackend's advertised Capability::tool_calling.
        std::uint64_t state = seed;
        for (std::size_t step = 0; step < 2; ++step) {
            if (cancel.cancelled()) {
                stats.stop_reason = StopReason::cancelled;
                return Status(ErrorCode::cancelled, "cancelled by caller");
            }
            state = splitmix64(state);
            ++stats.completion_tokens;
            TokenChunk chunk{"", stats.chunks++};
            ToolCallDelta tcd;
            tcd.index = 0;
            if (step == 0) {
                tcd.id = call_id;
                tcd.name = chosen->name;
            } else {
                tcd.arguments_delta = args_json;
            }
            chunk.tool_call_deltas.push_back(std::move(tcd));
            if (on_chunk && !on_chunk(chunk)) {
                stats.stop_reason = StopReason::callback;
                break;
            }
        }
        if (stats.stop_reason == StopReason::none) {
            stats.stop_reason = StopReason::end_of_sequence;
        }
        stats.eval_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        return stats;
    }

private:
    [[nodiscard]] std::size_t effective_checkpoint_limit() const noexcept {
        return std::min(options_.max_checkpoint_bytes, kMaxRecurrentCheckpointBytes);
    }

    [[nodiscard]] std::uint64_t checkpoint_model_hash() const noexcept {
        return fnv1a(descriptor_.name) ^ splitmix64(static_cast<std::uint64_t>(descriptor_.architecture));
    }

    Result<std::vector<std::uint8_t>> build_checkpoint_bytes(std::uint64_t token_position,
                                                             std::uint64_t prefix_hash) const {
        if (token_position == 0) {
            return Status(ErrorCode::invalid_argument, "recurrent checkpoint token_position must be > 0");
        }
        const std::size_t limit = effective_checkpoint_limit();
        if (limit == 0 || kMockCheckpointBytes > limit) {
            return Status(ErrorCode::invalid_argument,
                          "recurrent checkpoint snapshot exceeds max_checkpoint_bytes (" + std::to_string(limit) +
                              " bytes)");
        }
        const std::uint64_t model_hash = checkpoint_model_hash();
        std::vector<std::uint8_t> bytes(kMockCheckpointBytes, 0);
        std::memcpy(bytes.data(), kMockCheckpointMagic.data(), kMockCheckpointMagic.size());
        write_u64_le(bytes.data() + 8, model_hash);
        write_u64_le(bytes.data() + 16, token_position);
        write_u64_le(bytes.data() + 24, prefix_hash);
        std::uint64_t state = splitmix64(model_hash ^ token_position ^ prefix_hash);
        for (std::size_t off = 32; off < 56; off += 8) {
            state = splitmix64(state);
            write_u64_le(bytes.data() + off, state);
        }
        const std::uint64_t checksum = fnv1a_bytes(std::span<const std::uint8_t>(bytes.data(), 56));
        write_u64_le(bytes.data() + 56, checksum);
        return bytes;
    }

    Status validate_checkpoint_bytes(std::span<const std::uint8_t> bytes, std::uint64_t& out_pos,
                                     std::uint64_t& out_prefix_hash) const {
        if (bytes.empty()) {
            return Status(ErrorCode::invalid_argument, "recurrent checkpoint snapshot is empty");
        }
        const std::size_t limit = effective_checkpoint_limit();
        if (limit == 0 || bytes.size() > limit) {
            return Status(ErrorCode::invalid_argument,
                          "recurrent checkpoint snapshot exceeds max_checkpoint_bytes (" + std::to_string(limit) +
                              " bytes)");
        }
        if (bytes.size() != kMockCheckpointBytes ||
            std::memcmp(bytes.data(), kMockCheckpointMagic.data(), kMockCheckpointMagic.size()) != 0) {
            return Status(ErrorCode::invalid_argument, "invalid mock recurrent checkpoint snapshot");
        }
        const std::uint64_t expected_checksum = fnv1a_bytes(bytes.subspan(0, 56));
        if (read_u64_le(bytes.data() + 56) != expected_checksum) {
            return Status(ErrorCode::invalid_argument, "mock recurrent checkpoint checksum mismatch");
        }
        if (read_u64_le(bytes.data() + 8) != checkpoint_model_hash()) {
            return Status(ErrorCode::invalid_argument, "mock recurrent checkpoint model mismatch");
        }
        const std::uint64_t pos = read_u64_le(bytes.data() + 16);
        if (pos == 0) {
            return Status(ErrorCode::invalid_argument, "mock recurrent checkpoint has zero token_position");
        }
        out_pos = pos;
        out_prefix_hash = read_u64_le(bytes.data() + 24);
        return Status::success();
    }

    // Performs atomic restore + prefill-boundary capture under state_mutex_
    // before releasing the mutex to invoke caller callbacks.
    void apply_checkpoint_exchange(const GenerateRequest& request) {
        if (!options_.enable_recurrent_checkpoints) {
            return;
        }
        const std::vector<TokenId> prompt_tokens = detail::accounting_tokenize(request.prompt);
        if (request.checkpoint == nullptr) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_prompt_hash_ = hash_token_prefix(prompt_tokens);
            return;
        }

        RecurrentCheckpointExchange& cp = *request.checkpoint;
        std::vector<std::uint8_t> captured_vec;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!cp.restore_bytes.empty() && cp.restore_token_position > 0 &&
                cp.restore_token_position <= prompt_tokens.size()) {
                std::uint64_t restored_pos = 0;
                std::uint64_t restored_hash = 0;
                Status rst = validate_checkpoint_bytes(cp.restore_bytes, restored_pos, restored_hash);
                if (rst.ok() && restored_pos != cp.restore_token_position) {
                    rst = Status(ErrorCode::invalid_argument, "mock recurrent checkpoint token_position mismatch");
                }
                if (rst.ok()) {
                    const std::uint64_t expected_prefix_hash = hash_token_prefix(
                        std::span<const TokenId>(prompt_tokens.data(), static_cast<std::size_t>(restored_pos)));
                    if (restored_hash != expected_prefix_hash) {
                        rst = Status(ErrorCode::invalid_argument, "mock recurrent checkpoint prefix hash mismatch");
                    }
                }
                if (rst.ok()) {
                    last_prompt_hash_ = restored_hash;
                    cp.restored = true;
                } else {
                    cp.restored = false;
                    cp.restore_error = std::move(rst);
                }
            }

            if (cp.capture_at_token_position > 0 && cp.capture_at_token_position <= prompt_tokens.size()) {
                const std::uint64_t prefix_hash = hash_token_prefix(std::span<const TokenId>(
                    prompt_tokens.data(), static_cast<std::size_t>(cp.capture_at_token_position)));
                last_prompt_hash_ = prefix_hash;
                auto built = build_checkpoint_bytes(cp.capture_at_token_position, prefix_hash);
                if (built.ok()) {
                    captured_vec = std::move(built).value();
                    cp.captured = true;
                } else {
                    cp.capture_error = built.status();
                }
            } else {
                last_prompt_hash_ = hash_token_prefix(prompt_tokens);
            }
        }

        if (cp.restored && cp.on_restored) {
            cp.on_restored();
        }
        if (cp.captured) {
            if (cp.on_captured) {
                cp.on_captured(cp.capture_at_token_position, std::move(captured_vec));
            } else {
                cp.captured_bytes = std::move(captured_vec);
            }
        }
    }

    ModelDescriptor descriptor_;
    MockBackendOptions options_;
    mutable std::mutex state_mutex_;
    std::uint64_t last_prompt_hash_ = 0;
};

class MockBackend final : public Backend {
public:
    explicit MockBackend(MockBackendOptions o) : options_(o) {}

    std::string name() const override { return kMockBackendName; }
    std::string description() const override {
        return "MOCK deterministic backend for tests only; performs no inference";
    }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::tokenization)
            .add(Capability::streaming)
            .add(Capability::deterministic)
            .add(Capability::embeddings)
            .add(Capability::tool_calling)
            .add(Capability::multimodal)
            .add(Capability::logprobs);
        if (options_.token_logits) {
            c.add(Capability::token_logits);
        }
        if (options_.enable_recurrent_checkpoints) {
            c.add(Capability::recurrent_checkpoint);
        }
        return c;
    }
    Result<std::string> probe() override { return std::string("mock-1"); }
    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{describe("mock:tiny")}; }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        if (options.model.rfind("mock", 0) != 0) {
            return Status(ErrorCode::not_found, "mock backend only serves models named mock or mock:*; got " + options.model);
        }
        return std::shared_ptr<BackendModel>(std::make_shared<MockModel>(describe(options.model), options_));
    }

private:
    ModelDescriptor describe(const std::string& name) const {
        ModelDescriptor d;
        d.name = name;
        d.backend = kMockBackendName;
        d.format = "mock";
        d.family = "mock-deterministic";
        d.parameter_size = "0";
        d.quantization = "none";
        d.context_length = 4096;
        d.architecture = options_.default_architecture;
        return d;
    }
    MockBackendOptions options_;
};

}  // namespace

std::shared_ptr<Backend> make_mock_backend(MockBackendOptions options) { return std::make_shared<MockBackend>(options); }

}  // namespace sonder::inference
