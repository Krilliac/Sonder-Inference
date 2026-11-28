// POST /v1/embeddings (OpenAI shape): input forms (string, string array, token
// array, array of token arrays), encoding_format float|base64, per-request
// limits, usage counts. Parser and response-shaping units plus HTTP tests
// against a fake embedding engine (a si::Backend that records the request it
// is handed) and the mock backend.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "server_test_support.hpp"
#include "src/embeddings_route.hpp"
#include "src/openai.hpp"
#include "src/tool_call_parser.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

// Backend whose model records every EmbedRequest and answers with a canned result.
struct FakeEmbedEngine final : si::Backend {
    struct State {
        std::mutex mu;
        std::vector<si::EmbedRequest> seen;
        std::vector<std::vector<float>> vectors{{0.0f, 0.5f, 3.0f}, {1.0f, -2.5f, 0.25f}};
        std::uint64_t prompt_tokens = 7;
        bool reject_non_float = false;
    };
    struct Model final : si::BackendModel {
        si::ModelDescriptor desc;
        std::shared_ptr<State> state;
        explicit Model(std::shared_ptr<State> s) : state(std::move(s)) {
            desc.name = "fake-embed:test";
            desc.backend = "fake-embed";
            desc.context_length = 4096;
        }
        const si::ModelDescriptor& descriptor() const override { return desc; }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                               const si::TokenCallback&) override {
            return si::GenerateStats{};
        }
        si::Result<si::EmbedResult> embed(const si::EmbedRequest& request, const si::CancellationToken&) override {
            std::lock_guard<std::mutex> lock(state->mu);
            state->seen.push_back(request);
            if (state->reject_non_float && !request.encoding_format.empty() && request.encoding_format != "float") {
                return si::Status(si::ErrorCode::protocol_error, "missing embedding vector");
            }
            si::EmbedResult r;
            const std::size_t n = request.inputs.size() + request.token_inputs.size();
            for (std::size_t i = 0; i < n; ++i) r.embeddings.push_back(state->vectors[i % state->vectors.size()]);
            r.prompt_tokens = state->prompt_tokens;
            r.dimensions = r.embeddings.empty() ? 0 : r.embeddings.front().size();
            r.pooling_type = si::EmbedPoolingType::last;
            r.encoding_format = request.encoding_format;
            return r;
        }
    };
    std::shared_ptr<State> state = std::make_shared<State>();
    std::shared_ptr<Model> model = std::make_shared<Model>(state);
    std::string name() const override { return "fake-embed"; }
    std::string description() const override { return "records embed requests"; }
    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::embeddings);
    }
    si::Result<std::string> probe() override { return std::string("fake-embed-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

srv::ServerOptions fake_engine_options(const std::shared_ptr<si::Backend>& backend) {
    auto o = Fixture::defaults();
    o.models = {"fake-embed:test"};
    o.backend.backend.clear();
    o.backend_instance = backend;
    return o;
}

std::vector<std::uint8_t> decode_b64(const std::string& s) {
    auto d = det::decode_strict_base64(s, 1u << 22, "embedding");
    REQUIRE(std::holds_alternative<std::vector<std::uint8_t>>(d));
    return std::get<std::vector<std::uint8_t>>(d);
}

std::vector<float> floats_from_le_bytes(const std::vector<std::uint8_t>& bytes) {
    REQUIRE(bytes.size() % 4u == 0u);
    std::vector<float> out;
    for (std::size_t i = 0; i < bytes.size(); i += 4) {
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[i]) | (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
                                   (static_cast<std::uint32_t>(bytes[i + 2]) << 16) |
                                   (static_cast<std::uint32_t>(bytes[i + 3]) << 24);
        float f = 0.0f;
        static_assert(sizeof(f) == sizeof(bits));
        std::memcpy(&f, &bits, sizeof f);
        out.push_back(f);
    }
    return out;
}

const det::ApiError& expect_parse_error(const std::variant<det::EmbedJob, det::ApiError>& parsed) {
    REQUIRE(std::holds_alternative<det::ApiError>(parsed));
    return std::get<det::ApiError>(parsed);
}

const det::EmbedJob& expect_job(const std::variant<det::EmbedJob, det::ApiError>& parsed) {
    REQUIRE_MESSAGE(std::holds_alternative<det::EmbedJob>(parsed),
                    (std::holds_alternative<det::ApiError>(parsed) ? std::get<det::ApiError>(parsed).message
                                                                   : std::string()));
    return std::get<det::EmbedJob>(parsed);
}

det::EmbedRouteLimits tiny_limits() {
    det::EmbedRouteLimits l;
    l.max_inputs = 3;
    l.max_input_bytes = 8;
    l.max_total_bytes = 12;
    l.max_input_tokens = 4;
    l.max_total_tokens = 6;
    return l;
}

}  // namespace

// ============================================================== input forms

TEST_CASE("embeddings parser: input accepts a string, a string array, a token array and an array of token arrays") {
    {
        const auto parsed = det::parse_embed_request(R"({"input":"hello"})");
        const auto& job = expect_job(parsed);
        REQUIRE(job.inputs.size() == 1u);
        CHECK(job.inputs[0] == "hello");
        CHECK(job.token_inputs.empty());
    }
    {
        const auto parsed = det::parse_embed_request(R"({"input":["a","b","c"]})");
        const auto& job = expect_job(parsed);
        CHECK(job.inputs.size() == 3u);
        CHECK(job.token_inputs.empty());
    }
    {
        // A flat integer array is ONE tokenized input (OpenAI semantics).
        const auto parsed = det::parse_embed_request(R"({"input":[1,2,3]})");
        const auto& job = expect_job(parsed);
        CHECK(job.inputs.empty());
        REQUIRE(job.token_inputs.size() == 1u);
        CHECK(job.token_inputs[0] == std::vector<si::TokenId>{1, 2, 3});
    }
    {
        const auto parsed = det::parse_embed_request(R"({"input":[[1,2],[3],[4,5,6]]})");
        const auto& job = expect_job(parsed);
        CHECK(job.inputs.empty());
        REQUIRE(job.token_inputs.size() == 3u);
        CHECK(job.token_inputs[0] == std::vector<si::TokenId>{1, 2});
        CHECK(job.token_inputs[1] == std::vector<si::TokenId>{3});
        CHECK(job.token_inputs[2] == std::vector<si::TokenId>{4, 5, 6});
    }
}

TEST_CASE("embeddings parser: rejects malformed input shapes with param 'input'") {
    for (const char* body : {R"({})", R"({"input":null})", R"({"input":42})", R"({"input":true})", R"({"input":{}})",
                             R"({"input":[]})", R"({"input":["a",1]})", R"({"input":[1,"a"]})",
                             R"({"input":[[1],"a"]})", R"({"input":["a",[1]]})", R"({"input":[[]]})",
                             R"({"input":[[1],[]]})", R"({"input":[-1]})", R"({"input":[[1,-5]]})",
                             R"({"input":[1.5]})", R"({"input":[[2147483648]]})", R"({"input":[[1,"x"]]})",
                             R"({"input":[[[1]]]})", R"({"input":[null]})"}) {
        CAPTURE(body);
        const auto parsed = det::parse_embed_request(body);
        const auto& err = expect_parse_error(parsed);
        CHECK(err.status == 400);
        CHECK(err.code == "invalid_request");
        CHECK(err.param == std::optional<std::string>("input"));
    }
}

TEST_CASE("embeddings parser: every per-request limit is enforced and named in the message") {
    const auto limits = tiny_limits();
    const auto ok = [&](std::string_view body) { return std::holds_alternative<det::EmbedJob>(det::parse_embed_request(body, limits)); };
    const auto err_of = [&](std::string_view body) {
        const auto parsed = det::parse_embed_request(body, limits);
        REQUIRE(std::holds_alternative<det::ApiError>(parsed));
        return std::get<det::ApiError>(parsed);
    };

    // At the limits: accepted.
    CHECK(ok(R"({"input":["12345678","1234"]})"));              // 8 + 4 bytes == max_total_bytes
    CHECK(ok(R"({"input":["a","b","c"]})"));                    // 3 inputs == max_inputs
    CHECK(ok(R"({"input":[1,2,3,4]})"));                        // 4 tokens == max_input_tokens
    CHECK(ok(R"({"input":[[1,2,3],[4,5,6]]})"));                // 6 tokens == max_total_tokens

    // Number of inputs.
    {
        const auto e = err_of(R"({"input":["a","b","c","d"]})");
        CHECK(e.status == 400);
        CHECK(e.param == std::optional<std::string>("input"));
        CHECK(e.message.find("3") != std::string::npos);
        CHECK(e.message.find("items") != std::string::npos);
    }
    {
        const auto e = err_of(R"({"input":[[1],[2],[3],[4]]})");
        CHECK(e.message.find("items") != std::string::npos);
    }
    // One text input too long (bytes, not characters: 3 x U+00E9 is 6 bytes, 5 of them is 10).
    {
        const auto e = err_of(R"({"input":["123456789"]})");
        CHECK(e.status == 400);
        CHECK(e.message.find("input[0]") != std::string::npos);
        CHECK(e.message.find("8 bytes") != std::string::npos);
    }
    CHECK_FALSE(ok("{\"input\":[\"\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\"]}"));
    CHECK(ok("{\"input\":[\"\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\"]}"));
    // Total text bytes across inputs.
    {
        const auto e = err_of(R"({"input":["12345678","12345"]})");
        CHECK(e.message.find("12 bytes") != std::string::npos);
    }
    // One token array too long.
    {
        const auto e = err_of(R"({"input":[1,2,3,4,5]})");
        CHECK(e.message.find("4 tokens") != std::string::npos);
    }
    {
        const auto e = err_of(R"({"input":[[1,2],[1,2,3,4,5]]})");
        CHECK(e.message.find("input[1]") != std::string::npos);
    }
    // Total tokens across arrays.
    {
        const auto e = err_of(R"({"input":[[1,2,3,4],[5,6,7]]})");
        CHECK(e.message.find("6 tokens") != std::string::npos);
    }
}

TEST_CASE("embeddings parser: the default limits admit a maximal token-array request") {
    // 256 inputs x 1024 ids == 262144 == the default total-token limit.
    std::string body = R"({"input":[)";
    for (int i = 0; i < 256; ++i) {
        if (i > 0) body += ',';
        body += '[';
        for (int t = 0; t < 1024; ++t) {
            if (t > 0) body += ',';
            body += std::to_string(t);
        }
        body += ']';
    }
    body += "]}";
    const auto parsed = det::parse_embed_request(body);
    const auto& job = expect_job(parsed);
    CHECK(job.token_inputs.size() == 256u);

    // One id more than the total limit is refused (not silently truncated).
    const std::string over = body.substr(0, body.size() - 3) + ",7]]}";
    const auto parsed_over = det::parse_embed_request(over);
    const auto& err = expect_parse_error(parsed_over);
    CHECK(err.status == 400);
    CHECK(err.param == std::optional<std::string>("input"));
}

// ============================================================== encoding_format

TEST_CASE("embeddings parser: encoding_format accepts float and base64 only") {
    for (const char* fmt : {"float", "base64"}) {
        const std::string body = std::string(R"({"input":"x","encoding_format":")") + fmt + "\"}";
        CAPTURE(body);
        const auto parsed = det::parse_embed_request(body);
        CHECK(expect_job(parsed).encoding_format == fmt);
    }
    CHECK(expect_job(det::parse_embed_request(R"({"input":"x"})")).encoding_format == "float");
    CHECK(expect_job(det::parse_embed_request(R"({"input":"x","encoding_format":null})")).encoding_format == "float");
    for (const char* bad : {R"({"input":"x","encoding_format":"int8"})", R"({"input":"x","encoding_format":"BASE64"})",
                            R"({"input":"x","encoding_format":""})", R"({"input":"x","encoding_format":7})",
                            R"({"input":"x","encoding_format":["float"]})"}) {
        CAPTURE(bad);
        const auto parsed = det::parse_embed_request(bad);
        const auto& err = expect_parse_error(parsed);
        CHECK(err.status == 400);
        CHECK(err.code == "unsupported_parameter");
        CHECK(err.param == std::optional<std::string>("encoding_format"));
    }
}

TEST_CASE("embeddings base64: little-endian float32 bytes, known vectors and round trip") {
    const std::vector<float> a{1.0f, -2.5f};
    CHECK(det::encode_embedding_base64(a) == "AACAPwAAIMA=");
    const std::vector<float> b{0.0f, 0.5f, 3.0f};
    CHECK(det::encode_embedding_base64(b) == "AAAAAAAAAD8AAEBA");
    CHECK(det::encode_embedding_base64(std::vector<float>{}).empty());

    std::vector<float> many;
    for (int i = 0; i < 1000; ++i) many.push_back(std::sin(static_cast<float>(i)) * 1e3f - 17.25f);
    const auto back = floats_from_le_bytes(decode_b64(det::encode_embedding_base64(many)));
    REQUIRE(back.size() == many.size());
    for (std::size_t i = 0; i < many.size(); ++i) CHECK(back[i] == many[i]);  // bit-exact
}

// ============================================================== response shaping

TEST_CASE("embeddings response: float and base64 data, ordered index, usage counts") {
    si::EmbedResult r;
    r.embeddings = {{0.0f, 0.5f, 3.0f}, {1.0f, -2.5f, 0.25f}};
    r.prompt_tokens = 11;
    r.dimensions = 3;
    r.pooling_type = si::EmbedPoolingType::mean;

    det::EmbedJob job;
    {
        job.encoding_format = "float";
        const json::Object doc = det::build_embeddings_response(job, r, "m", "native", "req_1", false);
        CHECK(doc.find("object")->as_string() == "list");
        CHECK(doc.find("model")->as_string() == "m");
        const auto& data = doc.find("data")->as_array();
        REQUIRE(data.size() == 2u);
        CHECK(data[0].find("object")->as_string() == "embedding");
        CHECK(data[1].find("index")->as_uint() == 1u);
        const auto& v1 = data[1].find("embedding")->as_array();
        REQUIRE(v1.size() == 3u);
        CHECK(v1[1].as_double() == doctest::Approx(-2.5));
        const auto* usage = doc.find("usage");
        REQUIRE(usage != nullptr);
        CHECK(usage->find("prompt_tokens")->as_uint() == 11u);
        CHECK(usage->find("total_tokens")->as_uint() == 11u);
        const auto* meta = doc.find("sonder");
        REQUIRE(meta != nullptr);
        CHECK(meta->find("encoding_format")->as_string() == "float");
        CHECK(meta->find("dimensions")->as_uint() == 3u);
        CHECK(meta->find("pooling_type")->as_string() == "mean");
        CHECK(meta->find("backend")->as_string() == "native");
        CHECK(meta->find("request_id")->as_string() == "req_1");
    }
    {
        job.encoding_format = "base64";
        const json::Object doc = det::build_embeddings_response(job, r, "m", "native", "req_2", false);
        const auto& data = doc.find("data")->as_array();
        REQUIRE(data.size() == 2u);
        REQUIRE(data[0].find("embedding")->is_string());
        CHECK(data[0].find("embedding")->as_string() == "AAAAAAAAAD8AAEBA");
        const auto back = floats_from_le_bytes(decode_b64(data[1].find("embedding")->as_string()));
        CHECK(back == std::vector<float>{1.0f, -2.5f, 0.25f});
        CHECK(doc.find("usage")->find("total_tokens")->as_uint() == 11u);
        CHECK(doc.find("sonder")->find("encoding_format")->as_string() == "base64");
    }
}

// ============================================================== HTTP, fake engine

TEST_CASE("http /v1/embeddings: forwards text inputs and options to the engine; usage comes from the engine") {
    auto fake = std::make_shared<FakeEmbedEngine>();
    Fixture f(fake_engine_options(fake));
    const Reply r = post(f.port, "/v1/embeddings",
                         R"({"model":"fake-embed:test","input":["alpha","beta"],"normalize":false,"pooling_type":"cls"})");
    REQUIRE_MESSAGE(r.status == 200, r.body);
    {
        std::lock_guard<std::mutex> lock(fake->state->mu);
        REQUIRE(fake->state->seen.size() == 1u);
        const auto& seen = fake->state->seen[0];
        CHECK(seen.inputs == std::vector<std::string>{"alpha", "beta"});
        CHECK(seen.token_inputs.empty());
        CHECK_FALSE(seen.normalize);
        CHECK(seen.pooling_type == si::EmbedPoolingType::cls);
        CHECK(seen.encoding_format == "float");
    }
    const auto doc = r.json();
    CHECK(doc.find("usage")->find("prompt_tokens")->as_uint() == 7u);
    CHECK(doc.find("usage")->find("total_tokens")->as_uint() == 7u);
    const auto& data = doc.find("data")->as_array();
    REQUIRE(data.size() == 2u);
    CHECK(data[0].find("embedding")->as_array()[1].as_double() == doctest::Approx(0.5));
}

TEST_CASE("http /v1/embeddings: token arrays reach the engine as token ids, base64 comes back as a string") {
    auto fake = std::make_shared<FakeEmbedEngine>();
    Fixture f(fake_engine_options(fake));
    const Reply r = post(f.port, "/v1/embeddings",
                         R"({"input":[[101,2023,102],[101,102]],"encoding_format":"base64"})");
    REQUIRE_MESSAGE(r.status == 200, r.body);
    {
        std::lock_guard<std::mutex> lock(fake->state->mu);
        REQUIRE(fake->state->seen.size() == 1u);
        const auto& seen = fake->state->seen[0];
        CHECK(seen.inputs.empty());
        REQUIRE(seen.token_inputs.size() == 2u);
        CHECK(seen.token_inputs[0] == std::vector<si::TokenId>{101, 2023, 102});
        CHECK(seen.token_inputs[1] == std::vector<si::TokenId>{101, 102});
        // Server owns base64 wire encoding; the backend must always be asked for float.
        CHECK(seen.encoding_format == "float");
    }
    const auto doc = r.json();
    const auto& data = doc.find("data")->as_array();
    REQUIRE(data.size() == 2u);
    REQUIRE(data[0].find("embedding")->is_string());
    CHECK(data[0].find("embedding")->as_string() == "AAAAAAAAAD8AAEBA");
    CHECK(floats_from_le_bytes(decode_b64(data[1].find("embedding")->as_string())) ==
          std::vector<float>{1.0f, -2.5f, 0.25f});
    CHECK(doc.find("sonder")->find("encoding_format")->as_string() == "base64");
}

TEST_CASE("http /v1/embeddings: base64 wire format is encoded by server and never forwarded to backend") {
    // Models float-only backends such as llamaserver whose response parser expects float arrays.
    auto fake = std::make_shared<FakeEmbedEngine>();
    fake->state->reject_non_float = true;
    Fixture f(fake_engine_options(fake));
    const Reply r = post(f.port, "/v1/embeddings",
                         R"({"model":"fake-embed:test","input":"hello","encoding_format":"base64"})");
    REQUIRE_MESSAGE(r.status == 200, r.body);
    {
        std::lock_guard<std::mutex> lock(fake->state->mu);
        REQUIRE(fake->state->seen.size() == 1u);
        CHECK(fake->state->seen[0].encoding_format == "float");
    }
    const auto doc = r.json();
    const auto& data = doc.find("data")->as_array();
    REQUIRE(data.size() == 1u);
    REQUIRE(data[0].find("embedding")->is_string());
    CHECK(data[0].find("embedding")->as_string() == "AAAAAAAAAD8AAEBA");
    CHECK(doc.find("sonder")->find("encoding_format")->as_string() == "base64");
}

TEST_CASE("http /v1/embeddings: ingress limits answer 400 before the engine is touched") {
    auto fake = std::make_shared<FakeEmbedEngine>();
    Fixture f(fake_engine_options(fake));

    std::string too_many = R"({"input":[)";
    for (int i = 0; i < 257; ++i) too_many += (i ? ",\"x\"" : "\"x\"");
    too_many += "]}";
    const Reply many = post(f.port, "/v1/embeddings", too_many);
    CHECK(many.status == 400);
    CHECK(many.json().find("error")->find("param")->as_string() == "input");

    const std::string big_text = std::string(256u * 1024u + 1u, 'a');
    const Reply text = post(f.port, "/v1/embeddings", R"({"input":")" + big_text + "\"}");
    CHECK(text.status == 400);
    CHECK(text.json().find("error")->find("param")->as_string() == "input");

    std::string big_tokens = R"({"input":[)";
    for (int i = 0; i < 65537; ++i) big_tokens += (i ? ",1" : "1");
    big_tokens += "]}";
    const Reply tokens = post(f.port, "/v1/embeddings", big_tokens);
    CHECK(tokens.status == 400);
    CHECK(tokens.json().find("error")->find("param")->as_string() == "input");

    CHECK(post(f.port, "/v1/embeddings", R"({"input":"x","encoding_format":"int8"})").status == 400);

    std::lock_guard<std::mutex> lock(fake->state->mu);
    CHECK(fake->state->seen.empty());
}

// ============================================================== HTTP, mock backend

TEST_CASE("http /v1/embeddings (mock): base64 decodes to the same vector the float form returns") {
    Fixture f;
    const Reply fl = post(f.port, "/v1/embeddings", R"({"model":"mock:tiny","input":"same text"})");
    const Reply b64 = post(f.port, "/v1/embeddings",
                           R"({"model":"mock:tiny","input":"same text","encoding_format":"base64"})");
    REQUIRE_MESSAGE(fl.status == 200, fl.body);
    REQUIRE_MESSAGE(b64.status == 200, b64.body);
    const auto fdoc = fl.json();
    const auto bdoc = b64.json();
    const auto& fvec = fdoc.find("data")->as_array()[0].find("embedding")->as_array();
    const auto bvec = floats_from_le_bytes(decode_b64(bdoc.find("data")->as_array()[0].find("embedding")->as_string()));
    REQUIRE(bvec.size() == fvec.size());
    REQUIRE(bvec.size() == 16u);
    for (std::size_t i = 0; i < bvec.size(); ++i) CHECK(static_cast<double>(bvec[i]) == doctest::Approx(fvec[i].as_double()).epsilon(1e-6));
    CHECK(bdoc.find("usage")->find("prompt_tokens")->as_uint() == fdoc.find("usage")->find("prompt_tokens")->as_uint());
    CHECK(bdoc.find("usage")->find("prompt_tokens")->as_uint() > 0u);
}

TEST_CASE("http /v1/embeddings (mock): token arrays are served and usage counts exactly their ids") {
    Fixture f;
    const Reply r = post(f.port, "/v1/embeddings", R"({"model":"mock:tiny","input":[[5,6,7],[8,9]]})");
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto doc = r.json();
    REQUIRE(doc.find("data")->as_array().size() == 2u);
    CHECK(doc.find("usage")->find("prompt_tokens")->as_uint() == 5u);
    CHECK(doc.find("usage")->find("total_tokens")->as_uint() == 5u);
    // Different ids -> different vectors; same ids -> same vector (deterministic).
    const Reply again = post(f.port, "/v1/embeddings", R"({"model":"mock:tiny","input":[[5,6,7]]})");
    REQUIRE(again.status == 200);
    CHECK(again.json().find("data")->as_array()[0].find("embedding")->as_array()[0].as_double() ==
          doctest::Approx(doc.find("data")->as_array()[0].find("embedding")->as_array()[0].as_double()));
    CHECK(doc.find("data")->as_array()[0].find("embedding")->as_array()[0].as_double() !=
          doctest::Approx(doc.find("data")->as_array()[1].find("embedding")->as_array()[0].as_double()));
}

TEST_CASE("http /v1/embeddings: flat token array returns 200 with one embedding (fake engine and mock)") {
    {
        auto fake = std::make_shared<FakeEmbedEngine>();
        Fixture f(fake_engine_options(fake));
        const Reply r = post(f.port, "/v1/embeddings", R"({"input":[101,2023,102]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        {
            std::lock_guard<std::mutex> lock(fake->state->mu);
            REQUIRE(fake->state->seen.size() == 1u);
            const auto& seen = fake->state->seen[0];
            CHECK(seen.inputs.empty());
            REQUIRE(seen.token_inputs.size() == 1u);
            CHECK(seen.token_inputs[0] == std::vector<si::TokenId>{101, 2023, 102});
        }
        const auto doc = r.json();
        const auto& data = doc.find("data")->as_array();
        REQUIRE(data.size() == 1u);
        CHECK(data[0].find("index")->as_uint() == 0u);
        const auto& emb = data[0].find("embedding")->as_array();
        REQUIRE(emb.size() == 3u);
        CHECK(emb[0].as_double() == doctest::Approx(0.0));
        CHECK(emb[1].as_double() == doctest::Approx(0.5));
        CHECK(emb[2].as_double() == doctest::Approx(3.0));
        CHECK(doc.find("usage")->find("prompt_tokens")->as_uint() == 7u);
        CHECK(doc.find("usage")->find("total_tokens")->as_uint() == 7u);
    }
    {
        Fixture f;
        const Reply flat = post(f.port, "/v1/embeddings", R"({"model":"mock:tiny","input":[5,6,7]})");
        const Reply nested = post(f.port, "/v1/embeddings", R"({"model":"mock:tiny","input":[[5,6,7]]})");
        REQUIRE_MESSAGE(flat.status == 200, flat.body);
        REQUIRE_MESSAGE(nested.status == 200, nested.body);
        const auto fdoc = flat.json();
        const auto ndoc = nested.json();
        const auto& fdata = fdoc.find("data")->as_array();
        const auto& ndata = ndoc.find("data")->as_array();
        REQUIRE(fdata.size() == 1u);
        REQUIRE(ndata.size() == 1u);
        CHECK(fdata[0].find("index")->as_uint() == 0u);
        CHECK(fdoc.find("usage")->find("prompt_tokens")->as_uint() == 3u);
        CHECK(fdoc.find("usage")->find("total_tokens")->as_uint() == 3u);
        const auto& fvec = fdata[0].find("embedding")->as_array();
        const auto& nvec = ndata[0].find("embedding")->as_array();
        REQUIRE(fvec.size() == 16u);
        REQUIRE(nvec.size() == 16u);
        for (std::size_t i = 0; i < fvec.size(); ++i) {
            CHECK(fvec[i].as_double() == doctest::Approx(nvec[i].as_double()));
        }
    }
}
