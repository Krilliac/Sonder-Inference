// MOCK BACKEND - deterministic output for tests and harness development only.
// It performs no inference and must never be used for quality/perf claims.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

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

class MockModel final : public BackendModel {
public:
    MockModel(ModelDescriptor d, MockBackendOptions o) : descriptor_(std::move(d)), options_(o) {}

    const ModelDescriptor& descriptor() const override { return descriptor_; }

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
                for (const auto& s : request.sampling.stop) {
                    const auto p = combined.find(s);
                    if (p != std::string::npos) {
                        earliest = std::min(earliest, p);
                    }
                }
                if (earliest != std::string::npos) {
                    hit_stop = true;
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
