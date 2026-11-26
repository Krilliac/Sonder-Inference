// End-to-end HTTP and unit tests for Wave 5 Slice 12:
//   - POST /v1/rerank (sorting, original index preservation, top_n, return_documents,
//     validation, priority admission, cancellation, non-reranker 400/501)
//   - POST /v1/score (input pair order preservation, pairs & text_1/text_2 forms,
//     consistency with /v1/rerank, validation, non-reranker 400/501)
//   - Extended POST /v1/embeddings (dimensions fail-closed validation, opt-in truncation,
//     pooling_type/encoding_format/model metadata)
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "server_test_support.hpp"
#include "src/openai.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

// Minimal backend that does NOT override rerank() or score() (default ErrorCode::unsupported -> 501).
struct NoRerankBackend final : si::Backend {
    struct Model final : si::BackendModel {
        si::ModelDescriptor desc;
        Model() {
            desc.name = "norerank:test";
            desc.backend = "norerank-test";
            desc.context_length = 4096;
        }
        const si::ModelDescriptor& descriptor() const override { return desc; }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                               const si::TokenCallback&) override {
            return si::GenerateStats{};
        }
    };
    std::shared_ptr<Model> model = std::make_shared<Model>();
    std::string name() const override { return "norerank-test"; }
    std::string description() const override { return "backend without rerank or score"; }
    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming);
    }
    si::Result<std::string> probe() override { return std::string("norerank-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

// Backend advertising Capability::reranking whose loaded model explicitly refuses
// reranking as a non-reranker model (ErrorCode::invalid_argument -> HTTP 400).
struct NonRerankerModelBackend final : si::Backend {
    struct Model final : si::BackendModel {
        si::ModelDescriptor desc;
        Model() {
            desc.name = "causal-only:test";
            desc.backend = "nonreranker-test";
            desc.context_length = 4096;
        }
        const si::ModelDescriptor& descriptor() const override { return desc; }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                               const si::TokenCallback&) override {
            return si::GenerateStats{};
        }
        si::Result<si::RerankResult> rerank(const si::RerankRequest&, const si::CancellationToken&) override {
            return si::Status(si::ErrorCode::invalid_argument,
                              "model 'causal-only:test' does not support reranking (RANK pooling required)");
        }
        si::Result<si::ScoreResult> score(const si::ScoreRequest&, const si::CancellationToken&) override {
            return si::Status(si::ErrorCode::invalid_argument,
                              "model 'causal-only:test' does not support reranking (RANK pooling required)");
        }
    };
    std::shared_ptr<Model> model = std::make_shared<Model>();
    std::string name() const override { return "nonreranker-test"; }
    std::string description() const override { return "rerank-capable backend with a non-reranker model"; }
    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::reranking);
    }
    si::Result<std::string> probe() override { return std::string("nonreranker-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

srv::ServerOptions custom_backend_options(const std::shared_ptr<si::Backend>& backend, const std::string& model_id) {
    auto o = Fixture::defaults();
    o.models = {model_id};
    o.backend.backend.clear();
    o.backend_instance = backend;
    return o;
}

}  // namespace

// ================================================================= Pure parser unit tests

TEST_CASE("rerank parser: validates query, documents, top_n, return_documents, and bounds") {
    {
        const auto parsed = det::parse_rerank_request(
            R"({"model":"bge-reranker","query":"memory leak","documents":["alpha","beta"],"top_n":1,"return_documents":true,"priority":"subagent","deadline_ms":2500})");
        REQUIRE(std::holds_alternative<det::RerankJob>(parsed));
        const auto& job = std::get<det::RerankJob>(parsed);
        CHECK(job.model == "bge-reranker");
        CHECK(job.query == "memory leak");
        REQUIRE(job.documents.size() == 2u);
        CHECK(job.documents[0] == "alpha");
        CHECK(job.documents[1] == "beta");
        CHECK(job.top_n == std::optional<std::size_t>(1u));
        CHECK(job.return_documents);
        CHECK(job.priority_class == std::optional<si::RequestPriority>(si::RequestPriority::subagent));
        CHECK(job.deadline_ms == std::optional<std::uint64_t>(2500u));
    }

    auto expect_err = [](std::string_view body, const char* expected_code, const char* expected_param = nullptr) {
        CAPTURE(body);
        const auto parsed = det::parse_rerank_request(body);
        REQUIRE(std::holds_alternative<det::ApiError>(parsed));
        const auto& err = std::get<det::ApiError>(parsed);
        CHECK(err.status == 400);
        CHECK(err.code == expected_code);
        if (expected_param != nullptr) {
            CHECK(err.param == std::optional<std::string>(expected_param));
        }
    };

    expect_err("{bad", "invalid_json");
    expect_err("[]", "invalid_json");
    expect_err(R"({"documents":["a"]})", "invalid_request", "query");
    expect_err(R"({"query":"","documents":["a"]})", "invalid_request", "query");
    expect_err(R"({"query":123,"documents":["a"]})", "invalid_request", "query");
    expect_err(R"({"query":"q","documents":[]})", "invalid_request", "documents");
    expect_err(R"({"query":"q","documents":"not-array"})", "invalid_request", "documents");
    expect_err(R"({"query":"q","documents":["ok",42]})", "invalid_request", "documents");
    expect_err(R"({"query":"q","documents":["a"],"top_n":0})", "invalid_request", "top_n");
    expect_err(R"({"query":"q","documents":["a"],"top_n":-2})", "invalid_request", "top_n");
    expect_err(R"({"query":"q","documents":["a"],"top_n":1.5})", "invalid_request", "top_n");
    expect_err(R"({"query":"q","documents":["a"],"top_n":257})", "invalid_request", "top_n");
    expect_err(R"({"query":"q","documents":["a"],"return_documents":"yes"})", "invalid_request", "return_documents");
    expect_err(R"({"model":"","query":"q","documents":["a"]})", "invalid_request", "model");

    // > kMaxRerankDocuments (256) rejected.
    std::string too_many = R"({"query":"q","documents":[)";
    for (std::size_t i = 0; i < det::kMaxRerankDocuments + 1; ++i) {
        if (i > 0) too_many += ',';
        too_many += "\"d\"";
    }
    too_many += "]}";
    expect_err(too_many, "invalid_request", "documents");
}

TEST_CASE("score parser: supports pairs objects, pairs tuples, text_1/text_2 broadcast, and rejects malformed inputs") {
    // 1. pairs array of {text_1, text_2} objects
    {
        const auto parsed = det::parse_score_request(
            R"({"pairs":[{"text_1":"q1","text_2":"d1"},{"text_1":"q2","text_2":"d2"}]})");
        REQUIRE(std::holds_alternative<det::ScoreJob>(parsed));
        const auto& job = std::get<det::ScoreJob>(parsed);
        REQUIRE(job.pairs.size() == 2u);
        CHECK(job.pairs[0].text_1 == "q1");
        CHECK(job.pairs[0].text_2 == "d1");
        CHECK(job.pairs[1].text_1 == "q2");
        CHECK(job.pairs[1].text_2 == "d2");
    }
    // 2. pairs array of [text_1, text_2] tuples
    {
        const auto parsed = det::parse_score_request(R"({"pairs":[["a","b"],["c","d"]]})");
        REQUIRE(std::holds_alternative<det::ScoreJob>(parsed));
        const auto& job = std::get<det::ScoreJob>(parsed);
        REQUIRE(job.pairs.size() == 2u);
        CHECK(job.pairs[1].text_1 == "c");
        CHECK(job.pairs[1].text_2 == "d");
    }
    // 3. text_1 string broadcast against text_2 array
    {
        const auto parsed = det::parse_score_request(R"({"text_1":"query","text_2":["doc0","doc1","doc2"]})");
        REQUIRE(std::holds_alternative<det::ScoreJob>(parsed));
        const auto& job = std::get<det::ScoreJob>(parsed);
        REQUIRE(job.pairs.size() == 3u);
        CHECK(job.pairs[0].text_1 == "query");
        CHECK(job.pairs[2].text_1 == "query");
        CHECK(job.pairs[2].text_2 == "doc2");
    }
    // 4. text_1 array broadcast against text_2 string
    {
        const auto parsed = det::parse_score_request(R"({"text_1":["q0","q1"],"text_2":"doc"})");
        REQUIRE(std::holds_alternative<det::ScoreJob>(parsed));
        const auto& job = std::get<det::ScoreJob>(parsed);
        REQUIRE(job.pairs.size() == 2u);
        CHECK(job.pairs[0].text_2 == "doc");
        CHECK(job.pairs[1].text_1 == "q1");
    }
    // 5. parallel arrays of equal length
    {
        const auto parsed = det::parse_score_request(R"({"text_1":["q0","q1"],"text_2":["d0","d1"]})");
        REQUIRE(std::holds_alternative<det::ScoreJob>(parsed));
        const auto& job = std::get<det::ScoreJob>(parsed);
        REQUIRE(job.pairs.size() == 2u);
        CHECK(job.pairs[0].text_1 == "q0");
        CHECK(job.pairs[0].text_2 == "d0");
        CHECK(job.pairs[1].text_1 == "q1");
        CHECK(job.pairs[1].text_2 == "d1");
    }
    // Validation errors
    CHECK(std::holds_alternative<det::ApiError>(det::parse_score_request(R"({})")));
    CHECK(std::holds_alternative<det::ApiError>(
        det::parse_score_request(R"({"pairs":[["a","b"]],"text_1":"a","text_2":"b"})")));
    CHECK(std::holds_alternative<det::ApiError>(det::parse_score_request(R"({"pairs":[]})")));
    CHECK(std::holds_alternative<det::ApiError>(det::parse_score_request(R"({"pairs":[["only_one"]]})")));
    CHECK(std::holds_alternative<det::ApiError>(det::parse_score_request(R"({"text_1":["a","b"],"text_2":["x","y","z"]})")));
    CHECK(std::holds_alternative<det::ApiError>(det::parse_score_request(R"({"text_1":"a"})")));
}

TEST_CASE("embed parser: validates dimensions, pooling_type, normalize, and encoding_format") {
    {
        const auto parsed = det::parse_embed_request(
            R"({"input":["hello"],"dimensions":16,"pooling_type":"mean","encoding_format":"float","normalize":false})");
        REQUIRE(std::holds_alternative<det::EmbedJob>(parsed));
        const auto& job = std::get<det::EmbedJob>(parsed);
        CHECK(job.dimensions == std::optional<std::size_t>(16u));
        CHECK(job.pooling_type == si::EmbedPoolingType::mean);
        CHECK(job.encoding_format == "float");
        CHECK_FALSE(job.normalize);
    }
    // Invalid dimensions
    for (const char* bad_dims :
         {R"({"input":"x","dimensions":0})", R"({"input":"x","dimensions":-8})", R"({"input":"x","dimensions":8.5})",
          R"({"input":"x","dimensions":"16"})", R"({"input":"x","dimensions":70000})"}) {
        CAPTURE(bad_dims);
        const auto p = det::parse_embed_request(bad_dims);
        REQUIRE(std::holds_alternative<det::ApiError>(p));
        CHECK(std::get<det::ApiError>(p).status == 400);
        CHECK(std::get<det::ApiError>(p).param == std::optional<std::string>("dimensions"));
    }
    // Rank pooling rejected explicitly on /v1/embeddings
    {
        const auto p = det::parse_embed_request(R"({"input":"x","pooling_type":"rank"})");
        REQUIRE(std::holds_alternative<det::ApiError>(p));
        CHECK(std::get<det::ApiError>(p).code == "unsupported_parameter");
        CHECK(std::get<det::ApiError>(p).param == std::optional<std::string>("pooling_type"));
    }
    // Unknown pooling rejected
    {
        const auto p = det::parse_embed_request(R"({"input":"x","pooling_type":"bogus"})");
        REQUIRE(std::holds_alternative<det::ApiError>(p));
        CHECK(std::get<det::ApiError>(p).code == "invalid_request");
        CHECK(std::get<det::ApiError>(p).param == std::optional<std::string>("pooling_type"));
    }
    // Bad normalize type rejected
    {
        const auto p = det::parse_embed_request(R"({"input":"x","normalize":"yes"})");
        REQUIRE(std::holds_alternative<det::ApiError>(p));
        CHECK(std::get<det::ApiError>(p).param == std::optional<std::string>("normalize"));
    }
}

// ================================================================= HTTP /v1/rerank

TEST_CASE("http /v1/rerank: sorts descending, preserves original indices, honors top_n and return_documents") {
    Fixture f;
    const std::string body = R"({
        "model": "mock:tiny",
        "query": "sonder kv cache prefix reuse",
        "documents": [
            "completely unrelated banana recipe",
            "partial match about cache management",
            "exact sonder kv cache prefix reuse engine guide",
            "sonder prefix scheduling notes"
        ],
        "return_documents": true
    })";
    const Reply r = post(f.port, "/v1/rerank", body);
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto doc = r.json();
    CHECK(doc.find("object")->as_string() == "list");
    CHECK(doc.find("model")->as_string() == "mock:tiny");
    CHECK(doc.find("usage")->find("prompt_tokens")->as_uint() > 0u);
    CHECK(doc.find("usage")->find("total_tokens")->as_uint() ==
          doc.find("usage")->find("prompt_tokens")->as_uint());
    CHECK(doc.find("sonder")->find("api_version")->as_int() == 1);
    CHECK(doc.find("sonder")->find("backend")->as_string() == "mock");
    CHECK(doc.find("sonder")->find("model")->as_string() == "mock:tiny");
    CHECK(doc.find("sonder")->find("synthetic")->as_bool());

    const auto& results = doc.find("results")->as_array();
    REQUIRE(results.size() == 4u);
    // Document 2 contains the exact full query -> must rank #1 while preserving index == 2!
    CHECK(results[0].find("index")->as_int() == 2);
    // Document 0 is completely unrelated -> must rank last with index == 0!
    CHECK(results[3].find("index")->as_int() == 0);
    // Strictly non-increasing relevance scores and matching echoed document text.
    const std::vector<std::string> orig_docs = {
        "completely unrelated banana recipe",
        "partial match about cache management",
        "exact sonder kv cache prefix reuse engine guide",
        "sonder prefix scheduling notes",
    };
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto idx = static_cast<std::size_t>(results[i].find("index")->as_int());
        REQUIRE(idx < orig_docs.size());
        const auto* doc_obj = results[i].find("document");
        REQUIRE((doc_obj != nullptr && doc_obj->is_object()));
        CHECK(doc_obj->find("text")->as_string() == orig_docs[idx]);
        if (i > 0) {
            CHECK(results[i - 1].find("relevance_score")->as_double() >=
                  results[i].find("relevance_score")->as_double());
        }
    }

    // top_n=2 with return_documents omitted (false) returns top 2 without document echo.
    {
        const Reply r_top = post(f.port, "/v1/rerank", R"({
            "query": "sonder kv cache prefix reuse",
            "documents": [
                "completely unrelated banana recipe",
                "partial match about cache management",
                "exact sonder kv cache prefix reuse engine guide",
                "sonder prefix scheduling notes"
            ],
            "top_n": 2
        })");
        REQUIRE_MESSAGE(r_top.status == 200, r_top.body);
        const auto top_doc = r_top.json();
        const auto& top = top_doc.find("results")->as_array();
        REQUIRE(top.size() == 2u);
        CHECK(top[0].find("index")->as_int() == 2);
        CHECK(top[0].find("document") == nullptr);
        CHECK(top[0].find("relevance_score")->as_double() ==
              doctest::Approx(results[0].find("relevance_score")->as_double()));
        CHECK(top[1].find("index")->as_int() == results[1].find("index")->as_int());
    }

    // top_n > documents.size() clamps cleanly to documents.size().
    {
        const Reply r_clamp = post(f.port, "/v1/rerank",
                                   R"({"query":"alpha","documents":["alpha","beta"],"top_n":10})");
        REQUIRE_MESSAGE(r_clamp.status == 200, r_clamp.body);
        CHECK(r_clamp.json().find("results")->as_array().size() == 2u);
    }

    // Validation & routing errors
    CHECK(post(f.port, "/v1/rerank", R"({"query":"","documents":["a"]})").status == 400);
    CHECK(post(f.port, "/v1/rerank", R"({"query":"a","documents":[]})").status == 400);
    CHECK(post(f.port, "/v1/rerank", R"({"query":"a","documents":["x"],"top_n":0})").status == 400);
    CHECK(post(f.port, "/v1/rerank", R"({"model":"missing","query":"a","documents":["x"]})").status == 404);
    const Reply wrong_method = get(f.port, "/v1/rerank");
    CHECK(wrong_method.status == 405);
    CHECK(wrong_method.header("allow") == "POST, OPTIONS");
}

// ================================================================= HTTP /v1/score

TEST_CASE("http /v1/score: preserves exact input pair order and matches /v1/rerank scores") {
    Fixture f;
    // Deliberately put low-relevance pair first and high-relevance pair second.
    const Reply score_r = post(f.port, "/v1/score", R"({
        "model": "mock:tiny",
        "pairs": [
            {"text_1": "sonder kv cache", "text_2": "banana smoothie"},
            {"text_1": "sonder kv cache", "text_2": "sonder kv cache prefix reuse"},
            {"text_1": "sonder kv cache", "text_2": "cache policy"}
        ]
    })");
    REQUIRE_MESSAGE(score_r.status == 200, score_r.body);
    const auto score_doc = score_r.json();
    CHECK(score_doc.find("object")->as_string() == "list");
    CHECK(score_doc.find("model")->as_string() == "mock:tiny");
    const auto& data = score_doc.find("data")->as_array();
    REQUIRE(data.size() == 3u);
    // Exact original order preserved (indices 0, 1, 2 — NOT sorted by score!).
    CHECK(data[0].find("object")->as_string() == "score");
    CHECK(data[0].find("index")->as_int() == 0);
    CHECK(data[1].find("index")->as_int() == 1);
    CHECK(data[2].find("index")->as_int() == 2);
    const double s0 = data[0].find("score")->as_double();
    const double s1 = data[1].find("score")->as_double();
    const double s2 = data[2].find("score")->as_double();
    CHECK(s1 > s2);
    CHECK(s2 > s0);

    // Same query + docs through /v1/rerank must produce the exact same scores, sorted!
    const Reply rerank_r = post(f.port, "/v1/rerank", R"({
        "model": "mock:tiny",
        "query": "sonder kv cache",
        "documents": ["banana smoothie", "sonder kv cache prefix reuse", "cache policy"]
    })");
    REQUIRE_MESSAGE(rerank_r.status == 200, rerank_r.body);
    const auto rerank_doc = rerank_r.json();
    const auto& rr = rerank_doc.find("results")->as_array();
    REQUIRE(rr.size() == 3u);
    CHECK(rr[0].find("index")->as_int() == 1);
    CHECK(rr[0].find("relevance_score")->as_double() == doctest::Approx(s1));
    CHECK(rr[1].find("index")->as_int() == 2);
    CHECK(rr[1].find("relevance_score")->as_double() == doctest::Approx(s2));
    CHECK(rr[2].find("index")->as_int() == 0);
    CHECK(rr[2].find("relevance_score")->as_double() == doctest::Approx(s0));

    // text_1 / text_2 broadcast form works identically.
    const Reply broadcast_r = post(f.port, "/v1/score", R"({
        "text_1": "sonder kv cache",
        "text_2": ["banana smoothie", "sonder kv cache prefix reuse", "cache policy"]
    })");
    REQUIRE_MESSAGE(broadcast_r.status == 200, broadcast_r.body);
    const auto broadcast_doc = broadcast_r.json();
    const auto& bdata = broadcast_doc.find("data")->as_array();
    REQUIRE(bdata.size() == 3u);
    CHECK(bdata[0].find("score")->as_double() == doctest::Approx(s0));
    CHECK(bdata[1].find("score")->as_double() == doctest::Approx(s1));
    CHECK(bdata[2].find("score")->as_double() == doctest::Approx(s2));

    // Validation & routing errors
    CHECK(post(f.port, "/v1/score", R"({})").status == 400);
    CHECK(post(f.port, "/v1/score", R"({"text_1":["a","b"],"text_2":["x"]})").status == 200);
    CHECK(post(f.port, "/v1/score", R"({"text_1":["a","b"],"text_2":["x","y","z"]})").status == 400);
    const Reply wrong_method = get(f.port, "/v1/score");
    CHECK(wrong_method.status == 405);
    CHECK(wrong_method.header("allow") == "POST, OPTIONS");
}

// ================================================================= Non-reranker explicit 400/501 & capabilities

TEST_CASE("http /v1/rerank and /v1/score: explicit 501 on unsupported backend and 400 on non-reranker model") {
    // 1. Backend without rerank/score overrides -> 501 not_implemented + capabilities.reranking == false.
    {
        auto backend = std::make_shared<NoRerankBackend>();
        Fixture f(custom_backend_options(backend, "norerank:test"));
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE(h.status == 200);
        const auto hj = h.json();
        CHECK_FALSE(hj.find("capabilities")->find("reranking")->as_bool());
        const auto* cap_status = hj.find("diagnostics")->find("capability_status")->find("reranking");
        REQUIRE(cap_status != nullptr);
        CHECK(cap_status->find("adapter_supports")->as_string() == "no");
        CHECK(cap_status->find("observed_active")->as_string() == "no");

        const Reply rr = post(f.port, "/v1/rerank", R"({"query":"q","documents":["d"]})");
        CHECK(rr.status == 501);
        CHECK(rr.json().find("error")->find("code")->as_string() == "not_implemented");

        const Reply sc = post(f.port, "/v1/score", R"({"text_1":"q","text_2":"d"})");
        CHECK(sc.status == 501);
        CHECK(sc.json().find("error")->find("code")->as_string() == "not_implemented");
    }

    // 2. MockBackend with enable_reranking=false -> 501 not_implemented + capabilities.reranking == false.
    {
        si::MockBackendOptions mo;
        mo.enable_reranking = false;
        auto backend = si::make_mock_backend(mo);
        Fixture f(custom_backend_options(backend, "mock:tiny"));
        CHECK_FALSE(get(f.port, "/v1/sonder/health").json().find("capabilities")->find("reranking")->as_bool());
        CHECK(post(f.port, "/v1/rerank", R"({"query":"q","documents":["d"]})").status == 501);
        CHECK(post(f.port, "/v1/score", R"({"text_1":"q","text_2":"d"})").status == 501);
    }

    // 3. Rerank-capable backend serving a non-reranker model -> explicit 400 unsupported_parameter.
    {
        auto backend = std::make_shared<NonRerankerModelBackend>();
        Fixture f(custom_backend_options(backend, "causal-only:test"));
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE(h.status == 200);
        const auto hj = h.json();
        CHECK(hj.find("capabilities")->find("reranking")->as_bool());
        const auto* cap_status = hj.find("diagnostics")->find("capability_status")->find("reranking");
        REQUIRE(cap_status != nullptr);
        CHECK(cap_status->find("adapter_supports")->as_string() == "yes");

        const Reply rr = post(f.port, "/v1/rerank", R"({"query":"q","documents":["d"]})");
        CHECK(rr.status == 400);
        CHECK(rr.json().find("error")->find("code")->as_string() == "unsupported_parameter");

        const Reply sc = post(f.port, "/v1/score", R"({"text_1":"q","text_2":"d"})");
        CHECK(sc.status == 400);
        CHECK(sc.json().find("error")->find("code")->as_string() == "unsupported_parameter");
    }
}

// ================================================================= Extended /v1/embeddings dimensions fail-closed

TEST_CASE("http /v1/embeddings: dimensions fail-closed validation and metadata contract") {
    // 1. Default mock backend (native dim 16, truncation disabled -> fail closed).
    {
        Fixture f;
        // Exact native dimension succeeds and emits rich metadata.
        const Reply exact = post(f.port, "/v1/embeddings",
                                 R"({"model":"mock:tiny","input":["a","b"],"dimensions":16,"pooling_type":"mean","encoding_format":"float"})");
        REQUIRE_MESSAGE(exact.status == 200, exact.body);
        const auto doc = exact.json();
        CHECK(doc.find("model")->as_string() == "mock:tiny");
        REQUIRE(doc.find("data")->as_array().size() == 2u);
        CHECK(doc.find("data")->as_array()[0].find("embedding")->as_array().size() == 16u);
        const auto* meta = doc.find("sonder");
        REQUIRE(meta != nullptr);
        CHECK(meta->find("model")->as_string() == "mock:tiny");
        CHECK(meta->find("dimensions")->as_uint() == 16u);
        CHECK(meta->find("pooling_type")->as_string() == "mean");
        CHECK(meta->find("encoding_format")->as_string() == "float");

        // Smaller dimension (< 16) MUST fail closed (never silently truncate arbitrary models!).
        const Reply smaller = post(f.port, "/v1/embeddings", R"({"input":"a","dimensions":8})");
        CHECK(smaller.status == 400);
        CHECK(smaller.json().find("error")->find("code")->as_string() == "unsupported_parameter");
        CHECK(smaller.json().find("error")->find("param")->as_string() == "dimensions");

        // Larger dimension (> 16) MUST fail closed.
        const Reply larger = post(f.port, "/v1/embeddings", R"({"input":"a","dimensions":32})");
        CHECK(larger.status == 400);
        CHECK(larger.json().find("error")->find("code")->as_string() == "invalid_request");

        // Zero / negative / rank pooling fail closed at ingress.
        CHECK(post(f.port, "/v1/embeddings", R"({"input":"a","dimensions":0})").status == 400);
        const Reply rank_err = post(f.port, "/v1/embeddings", R"({"input":"a","pooling_type":"rank"})");
        CHECK(rank_err.status == 400);
        CHECK(rank_err.json().find("error")->find("code")->as_string() == "unsupported_parameter");
    }

    // 2. Opt-in truncation-capable model supports reduced dimensions with L2 re-normalization.
    {
        si::MockBackendOptions mo;
        mo.allow_embed_dimension_truncation = true;
        auto backend = si::make_mock_backend(mo);
        Fixture f(custom_backend_options(backend, "mock:matryoshka"));
        const Reply r = post(f.port, "/v1/embeddings", R"({"input":"hello","dimensions":8,"normalize":true})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(doc.find("sonder")->find("dimensions")->as_uint() == 8u);
        const auto& vec = doc.find("data")->as_array()[0].find("embedding")->as_array();
        REQUIRE(vec.size() == 8u);
        double sq = 0.0;
        for (const auto& v : vec) {
            const double d = v.as_double();
            sq += d * d;
        }
        CHECK(std::sqrt(sq) == doctest::Approx(1.0).epsilon(1e-4));
        // Still rejects > native dimension (16).
        CHECK(post(f.port, "/v1/embeddings", R"({"input":"hello","dimensions":17})").status == 400);
    }
}

// ================================================================= Priority admission & cancellation on /v1/rerank & /v1/score

TEST_CASE("http /v1/rerank and /v1/score: respect priority admission capacity and deadline expiration") {
    struct GatedRerankBackend final : si::Backend {
        struct Model final : si::BackendModel {
            si::ModelDescriptor desc;
            std::atomic<bool> chat_entered{false};
            std::atomic<bool> release_chat{false};
            std::atomic<bool> chat_active{false};
            std::atomic<bool> rerank_overlapped{false};
            std::atomic<bool> score_overlapped{false};

            Model() {
                desc.name = "gated:rerank";
                desc.backend = "gated-rerank";
                desc.context_length = 4096;
            }
            const si::ModelDescriptor& descriptor() const override { return desc; }
            bool has_native_chat() const override { return true; }
            si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                                   const si::TokenCallback&) override {
                return si::GenerateStats{};
            }
            si::Result<si::GenerateStats> chat(const si::ChatRequest&, const si::CancellationToken& cancel,
                                               const si::TokenCallback& on_chunk) override {
                chat_active.store(true);
                chat_entered.store(true);
                while (!release_chat.load() && !cancel.cancelled()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                chat_active.store(false);
                if (on_chunk) (void)on_chunk(si::TokenChunk{"ok", 0});
                si::GenerateStats st;
                st.prompt_tokens = 2;
                st.completion_tokens = 1;
                st.chunks = 1;
                st.token_counts_from_backend = true;
                st.stop_reason = si::StopReason::end_of_sequence;
                return st;
            }
            si::Result<si::RerankResult> rerank(const si::RerankRequest& req,
                                                const si::CancellationToken&) override {
                if (chat_active.load()) rerank_overlapped.store(true);
                si::RerankResult out;
                out.model = desc.name;
                out.prompt_tokens = static_cast<std::uint64_t>(req.documents.size());
                for (std::size_t i = 0; i < req.documents.size(); ++i) {
                    out.results.push_back(si::RerankItem{i, static_cast<double>(i), std::nullopt});
                }
                si::finalize_rerank_items(out.results, req.top_n);
                return out;
            }
            si::Result<si::ScoreResult> score(const si::ScoreRequest& req,
                                              const si::CancellationToken&) override {
                if (chat_active.load()) score_overlapped.store(true);
                si::ScoreResult out;
                out.model = desc.name;
                out.prompt_tokens = static_cast<std::uint64_t>(req.pairs.size());
                for (std::size_t i = 0; i < req.pairs.size(); ++i) {
                    out.scores.push_back(si::ScoreItem{i, 0.5 + static_cast<double>(i)});
                }
                return out;
            }
        };
        std::shared_ptr<Model> model = std::make_shared<Model>();
        std::string name() const override { return "gated-rerank"; }
        std::string description() const override { return "gated rerank admission test backend"; }
        si::BackendCapabilities capabilities() const override {
            return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::reranking);
        }
        std::size_t max_concurrent_requests() const override { return 1; }
        si::Result<std::string> probe() override { return std::string("gated-rerank-1"); }
        si::Result<std::vector<si::ModelDescriptor>> list_models() override {
            return std::vector<si::ModelDescriptor>{model->descriptor()};
        }
        si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
            return std::static_pointer_cast<si::BackendModel>(model);
        }
    };

    auto be = std::make_shared<GatedRerankBackend>();
    auto o = custom_backend_options(be, "gated:rerank");
    o.priority_admission = srv::PriorityAdmissionPolicy::on;
    o.backend_capacity = 1;
    Fixture f(o);

    Reply chat_reply;
    std::thread t_chat([&] {
        chat_reply = post(f.port, "/v1/chat/completions", R"({"messages":[{"role":"user","content":"hold"}]})");
    });
    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!be->model->chat_entered.load() && std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(be->model->chat_entered.load());

    // 1. Queued rerank with tight deadline expires with 504 deadline_exceeded without running.
    const Reply expired_rerank = post(
        f.port, "/v1/rerank", R"({"query":"q","documents":["a","b"],"priority":"background","deadline_ms":60})");
    CHECK(expired_rerank.status == 504);
    CHECK(expired_rerank.json().find("error")->find("code")->as_string() == "deadline_exceeded");

    // 2. Queued rerank + score wait for permit, never overlapping the active chat.
    Reply rerank_reply;
    Reply score_reply;
    std::thread t_rerank([&] { rerank_reply = post(f.port, "/v1/rerank", R"({"query":"q","documents":["a","b"]})"); });
    std::thread t_score([&] { score_reply = post(f.port, "/v1/score", R"({"text_1":"q","text_2":["a","b"]})"); });
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CHECK_FALSE(be->model->rerank_overlapped.load());
    CHECK_FALSE(be->model->score_overlapped.load());

    be->model->release_chat.store(true);
    t_chat.join();
    t_rerank.join();
    t_score.join();

    REQUIRE_MESSAGE(chat_reply.status == 200, chat_reply.body);
    REQUIRE_MESSAGE(rerank_reply.status == 200, rerank_reply.body);
    REQUIRE_MESSAGE(score_reply.status == 200, score_reply.body);
    CHECK_FALSE(be->model->rerank_overlapped.load());
    CHECK_FALSE(be->model->score_overlapped.load());
}
