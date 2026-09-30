#include <doctest/doctest.h>

#include <memory>
#include <string>

#include "test_helpers.hpp"

using namespace sonder::inference;

namespace {
class ReasoningModel final : public BackendModel {
  public:
    ReasoningModel() {
        descriptor_.name = "reasoning:test";
        descriptor_.backend = "reasoning-test";
        descriptor_.format = "test";
    }
    const ModelDescriptor &descriptor() const override { return descriptor_; }
    Result<GenerateStats> generate(const GenerateRequest &, const CancellationToken &,
                                   const TokenCallback &) override {
        return Status(ErrorCode::internal, "reasoning test model requires chat");
    }
    Result<GenerateStats> chat(const ChatRequest &request, const CancellationToken &,
                               const TokenCallback &callback) override {
        GenerateStats stats;
        stats.stop_reason = StopReason::end_of_sequence;
        if (callback && request.separate_reasoning && !callback(TokenChunk{"", 0, "hidden thought"}))
            stats.stop_reason = StopReason::callback;
        if (stats.stop_reason != StopReason::callback && callback &&
            !callback(TokenChunk{"visible answer", 1}))
            stats.stop_reason = StopReason::callback;
        stats.chunks = stats.stop_reason == StopReason::callback ? 1 : 2;
        return stats;
    }
    bool has_native_chat() const override { return true; }

  private:
    ModelDescriptor descriptor_;
};

class ReasoningBackend final : public Backend {
  public:
    std::string name() const override { return "reasoning-test"; }
    std::string description() const override { return "reasoning transport test backend"; }
    BackendCapabilities capabilities() const override {
        return BackendCapabilities{}.add(Capability::streaming);
    }
    Result<std::string> probe() override { return std::string("test"); }
    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions &) override {
        return std::shared_ptr<BackendModel>(std::make_shared<ReasoningModel>());
    }
};
} // namespace

TEST_CASE("Session preserves reasoning-only chunks separately from visible text") {
    Engine engine;
    REQUIRE(engine.register_backend(std::make_shared<ReasoningBackend>()).ok());
    auto model = engine.load_model("reasoning-test", {"reasoning:test", "cpu:0"});
    REQUIRE(model.ok());
    auto session = engine.create_session(model.value(), {});
    REQUIRE(session.ok());

    RequestOptions options;
    options.separate_reasoning = true;
    std::string callback_text;
    std::string callback_reasoning;
    auto result = session.value()->chat(
        {{"user", "hello"}},
        [&](const TokenChunk &chunk) {
            callback_text += chunk.text;
            callback_reasoning += chunk.reasoning;
            return true;
        },
        std::nullopt, options);
    REQUIRE(result.ok());
    CHECK(result->text == "visible answer");
    CHECK(result->reasoning == "hidden thought");
    CHECK(callback_text == result->text);
    CHECK(callback_reasoning == result->reasoning);
}
