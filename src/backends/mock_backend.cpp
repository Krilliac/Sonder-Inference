// MOCK BACKEND - deterministic output for tests and harness development only.
// It performs no inference and must never be used for quality/perf claims.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

#include "engine/accounting_tokens.hpp"
#include "sonder/inference/backends.hpp"

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

    Result<std::unique_ptr<TokenStream>> open_token_stream(const GenerateRequest& request) override {
        if (!options_.token_logits) {
            return Status(ErrorCode::unsupported, "mock backend created without token_logits");
        }
        return std::unique_ptr<TokenStream>(std::make_unique<MockTokenStream>(request, descriptor_.name, options_));
    }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        GenerateStats stats;
        stats.prompt_tokens = count_words(request.prompt);
        stats.token_counts_from_backend = true;
        const auto t0 = std::chrono::steady_clock::now();

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
                const TokenChunk chunk{piece, stats.chunks++};
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

private:
    ModelDescriptor descriptor_;
    MockBackendOptions options_;
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
        c.add(Capability::tokenization).add(Capability::streaming).add(Capability::deterministic);
        if (options_.token_logits) {
            c.add(Capability::token_logits);
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
    static ModelDescriptor describe(const std::string& name) {
        ModelDescriptor d;
        d.name = name;
        d.backend = kMockBackendName;
        d.format = "mock";
        d.family = "mock-deterministic";
        d.parameter_size = "0";
        d.quantization = "none";
        d.context_length = 4096;
        return d;
    }
    MockBackendOptions options_;
};

}  // namespace

std::shared_ptr<Backend> make_mock_backend(MockBackendOptions options) { return std::make_shared<MockBackend>(options); }

}  // namespace sonder::inference
