// Tests for the Ollama model store reader. Every test builds a synthetic
// manifests/blobs tree (tiny fake GGUF headers) in a temp directory; nothing
// here needs Ollama, a network or model weights.
#include <doctest/doctest.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common/sha256.hpp"
#include "sonder/inference/model_store.hpp"

namespace fs = std::filesystem;
namespace m = sonder::inference::models;
namespace si = sonder::inference;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("sonder-modelstore-test-" + std::to_string(tick) + "-" + std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

void write_file(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string le32(std::uint32_t v) {
    std::string s(4, '\0');
    for (int i = 0; i < 4; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
    return s;
}

std::string le64(std::uint64_t v) {
    std::string s(8, '\0');
    for (int i = 0; i < 8; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
    return s;
}

// A tiny fake GGUF: valid header (version 3), then `filler` bytes.
std::string fake_gguf(std::uint64_t tensors, std::uint64_t kvs, std::size_t filler, char fill = 'x') {
    return std::string("GGUF") + le32(3) + le64(tensors) + le64(kvs) + std::string(filler, fill);
}

struct Layer {
    std::string media_type;
    std::string digest;  // "sha256:<hex>"
    std::uint64_t size = 0;
    std::string from;
};

std::string layer_json(const Layer& l) {
    std::string s = "{\"mediaType\":\"" + l.media_type + "\",\"digest\":\"" + l.digest + "\",\"size\":" + std::to_string(l.size);
    if (!l.from.empty()) s += ",\"from\":\"" + l.from + "\"";
    return s + "}";
}

constexpr const char* kModel = "application/vnd.ollama.image.model";
constexpr const char* kProjector = "application/vnd.ollama.image.projector";
constexpr const char* kAdapter = "application/vnd.ollama.image.adapter";
constexpr const char* kTemplate = "application/vnd.ollama.image.template";
constexpr const char* kSystem = "application/vnd.ollama.image.system";
constexpr const char* kParams = "application/vnd.ollama.image.params";
constexpr const char* kLicense = "application/vnd.ollama.image.license";
constexpr const char* kTensor = "application/vnd.ollama.image.tensor";
constexpr const char* kConfigType = "application/vnd.docker.container.image.v1+json";
constexpr const char* kManifestType = "application/vnd.docker.distribution.manifest.v2+json";
constexpr const char* kListType = "application/vnd.ollama.manifest.list.v2+json";

class Store {
public:
    TempDir tmp;

    // Writes a blob named after the SHA-256 of `content`; returns its layer.
    Layer blob(const std::string& media_type, const std::string& content, const std::string& from = "") {
        const std::string hex = si::detail::sha256_hex(content);
        write_file(tmp.path / "blobs" / ("sha256-" + hex), content);
        return Layer{media_type, "sha256:" + hex, content.size(), from};
    }

    // A blob whose file name does NOT match its content (corruption).
    Layer corrupt_blob(const std::string& media_type, const std::string& claimed_content, const std::string& real_content) {
        const std::string hex = si::detail::sha256_hex(claimed_content);
        write_file(tmp.path / "blobs" / ("sha256-" + hex), real_content);
        return Layer{media_type, "sha256:" + hex, claimed_content.size(), ""};
    }

    // A layer for content that is NOT written (missing blob).
    static Layer absent(const std::string& media_type, const std::string& content) {
        return Layer{media_type, "sha256:" + si::detail::sha256_hex(content), content.size(), ""};
    }

    std::string manifest(const Layer& config, const std::vector<Layer>& layers, const std::string& extra = "") {
        std::string s = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kManifestType + "\",\"config\":" + layer_json(config) +
                        ",\"layers\":[";
        for (std::size_t i = 0; i < layers.size(); ++i) s += (i ? "," : "") + layer_json(layers[i]);
        return s + "]" + extra + "}";
    }

    Layer config_blob(const std::string& json) { return blob(kConfigType, json); }

    void put_v2(const std::string& host_dir, const std::string& ns, const std::string& model, const std::string& tag,
                const std::string& manifest_json) {
        write_file(tmp.path / "manifests-v2" / host_dir / ns / model / tag, manifest_json);
    }
    void put_legacy(const std::string& host_dir, const std::string& ns, const std::string& model, const std::string& tag,
                    const std::string& manifest_json) {
        write_file(tmp.path / "manifests" / host_dir / ns / model / tag, manifest_json);
    }

    fs::path blob_file(const Layer& l) const { return tmp.path / "blobs" / ("sha256-" + l.digest.substr(7)); }
};

const m::ModelRecord* find(const m::ScanResult& scan, const std::string& display) {
    for (const auto& r : scan.records) {
        if (r.name.display() == display) return &r;
    }
    return nullptr;
}

// path -> size for everything under `root` (proves a scan/verify writes nothing).
std::map<std::string, std::uint64_t> snapshot(const fs::path& root) {
    std::map<std::string, std::uint64_t> out;
    for (const auto& e : fs::recursive_directory_iterator(root)) {
        out[e.path().generic_string()] = e.is_regular_file() ? static_cast<std::uint64_t>(e.file_size()) : 0;
    }
    return out;
}

// One fully featured v2 model: model, projector, adapter, template, system, params, license.
void make_full_model(Store& s) {
    const auto cfg = s.config_blob(
        R"({"model_format":"gguf","model_family":"qwen2","model_families":["qwen2"],"model_type":"7.6B","file_type":"Q4_K_M","renderer":"r1","parser":"p1","requires":"0.30.0","capabilities":["completion","tools"]})");
    const auto model = s.blob(kModel, fake_gguf(291, 26, 4096), "qwen2.5-coder:7b");
    const auto proj = s.blob(kProjector, fake_gguf(10, 4, 512, 'p'));
    const auto adapter = s.blob(kAdapter, fake_gguf(2, 1, 64, 'a'));
    const auto tmpl = s.blob(kTemplate, "{{ .Prompt }}");
    const auto sys = s.blob(kSystem, "You are Sonder.");
    const auto params = s.blob(
        kParams,
        R"({"num_ctx":8192,"temperature":0.2,"top_p":0.9,"top_k":40,"seed":7.0,"stop":["<|im_end|>","<|im_start|>"],"use_mmap":true,"bogus":"x","min_p":"high"})");
    const auto lic = s.blob(kLicense, "Apache-2.0 license text");
    s.put_v2("ollama.com", "library", "coder", "7b", s.manifest(cfg, {model, proj, adapter, tmpl, sys, params, lic}, ",\"runner\":\"llamacpp\",\"format\":\"gguf\""));
}

}  // namespace

TEST_CASE("model store: name parsing applies Ollama defaults and validation") {
    auto n = m::parse_ollama_name("llama3");
    REQUIRE(n.ok());
    CHECK(n.value().full() == "registry.ollama.ai/library/llama3:latest");
    CHECK(n.value().display() == "llama3:latest");

    n = m::parse_ollama_name("qwen3.5:4b");
    REQUIRE(n.ok());
    CHECK(n.value().model == "qwen3.5");
    CHECK(n.value().tag == "4b");

    n = m::parse_ollama_name("R4C3R/qwen3-0.6b-heretic:q4_k_m");
    REQUIRE(n.ok());
    CHECK(n.value().ns == "R4C3R");
    CHECK(n.value().display() == "R4C3R/qwen3-0.6b-heretic:q4_k_m");

    n = m::parse_ollama_name("hf.co/bartowski/LLaMA-Mesh-GGUF:Q8_0");
    REQUIRE(n.ok());
    CHECK(n.value().host == "hf.co");
    CHECK(n.value().display() == "hf.co/bartowski/LLaMA-Mesh-GGUF:Q8_0");

    n = m::parse_ollama_name("https://localhost:5000/team/model:v1");
    REQUIRE(n.ok());
    CHECK(n.value().host == "localhost:5000");

    // ollama.com is the same registry as registry.ollama.ai.
    n = m::parse_ollama_name("ollama.com/library/x:1");
    REQUIRE(n.ok());
    CHECK(n.value().host == "registry.ollama.ai");
    CHECK(n.value().key() == m::parse_ollama_name("x:1").value().key());

    for (const char* bad : {"", "a b", "../x", "x:", ":tag", "/x", "h//x", "a/b/c/d:t", "ns.dot/model", "-lead", "x@sha256:abc", "m:t/x"}) {
        CAPTURE(bad);
        auto r = m::parse_ollama_name(bad);
        CHECK_FALSE(r.ok());
        if (!r.ok()) CHECK(r.status().code() == si::ErrorCode::invalid_argument);
    }
    CHECK_FALSE(m::parse_ollama_name(std::string(81, 'a')).ok());
}

TEST_CASE("model store: GGUF header reader") {
    TempDir t;
    write_file(t.path / "ok.gguf", fake_gguf(291, 26, 16));
    auto h = m::read_gguf_header(t.path / "ok.gguf");
    REQUIRE(h.ok());
    CHECK(h.value().version == 3);
    CHECK(h.value().tensor_count == 291);
    CHECK(h.value().kv_count == 26);

    write_file(t.path / "v1.gguf", std::string("GGUF") + le32(1) + le64(1) + le64(1));
    auto v1 = m::read_gguf_header(t.path / "v1.gguf");
    CHECK_FALSE(v1.ok());
    CHECK(v1.status().code() == si::ErrorCode::unsupported);

    write_file(t.path / "v9.gguf", std::string("GGUF") + le32(9) + le64(1) + le64(1));
    CHECK_FALSE(m::read_gguf_header(t.path / "v9.gguf").ok());
    write_file(t.path / "magic.gguf", std::string("GGUX") + le32(3) + le64(1) + le64(1));
    CHECK_FALSE(m::read_gguf_header(t.path / "magic.gguf").ok());
    write_file(t.path / "short.gguf", "GGUF");
    CHECK_FALSE(m::read_gguf_header(t.path / "short.gguf").ok());
    CHECK_FALSE(m::read_gguf_header(t.path / "absent.gguf").ok());
}

TEST_CASE("model store: open rejects directories that are not an Ollama store") {
    TempDir t;
    auto missing = m::OllamaStore::open(t.path / "nope");
    CHECK_FALSE(missing.ok());
    CHECK(missing.status().code() == si::ErrorCode::not_found);
    auto empty = m::OllamaStore::open(t.path);
    CHECK_FALSE(empty.ok());
    CHECK(empty.status().code() == si::ErrorCode::invalid_argument);
}

TEST_CASE("model store: a full v2 entry resolves to in-place paths and typed records") {
    Store s;
    make_full_model(s);
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto rec = store.value().resolve("coder:7b");
    REQUIRE(rec.ok());
    const m::ModelRecord& r = rec.value();
    CHECK(r.ready());
    CHECK(r.reason == m::UnsupportedReason::none);
    CHECK(r.layout == m::ManifestLayout::v2);
    CHECK(r.runner == "llamacpp");
    CHECK(r.format == "gguf");
    CHECK(r.parent == "qwen2.5-coder:7b");
    CHECK(r.name.display() == "coder:7b");

    // Paths point into the store's blobs/ directory (no copy).
    REQUIRE_FALSE(r.model_path.empty());
    CHECK(r.model_path.parent_path().filename() == "blobs");
    CHECK(fs::exists(r.model_path));
    CHECK(fs::file_size(r.model_path) == 24 + 4096);
    REQUIRE(r.projector_paths.size() == 1);
    REQUIRE(r.adapter_paths.size() == 1);
    CHECK(r.shard_paths.empty());
    REQUIRE(r.gguf.has_value());
    CHECK(r.gguf->tensor_count == 291);
    CHECK(r.gguf->kv_count == 26);

    CHECK(r.config.model_family == "qwen2");
    CHECK(r.config.parameter_size == "7.6B");
    CHECK(r.config.quantization == "Q4_K_M");
    CHECK(r.config.renderer == "r1");
    CHECK(r.config.requires_version == "0.30.0");
    REQUIRE(r.config.capabilities.size() == 2);

    REQUIRE(r.template_text.has_value());
    CHECK(*r.template_text == "{{ .Prompt }}");
    REQUIRE(r.system_prompt.has_value());
    CHECK(*r.system_prompt == "You are Sonder.");
    CHECK(r.license_count == 1);

    CHECK(r.params.num_ctx == 8192);
    CHECK(r.params.top_k == 40);
    CHECK(r.params.seed == 7);  // integral double accepted like Ollama's float64 -> int
    REQUIRE(r.params.temperature.has_value());
    CHECK(*r.params.temperature == doctest::Approx(0.2));
    CHECK(*r.params.top_p == doctest::Approx(0.9));
    REQUIRE(r.params.stop.size() == 2);
    CHECK(r.params.stop[0] == "<|im_end|>");
    CHECK(r.params.extra.contains("use_mmap"));
    CHECK(r.params.extra.contains("bogus"));
    CHECK_FALSE(r.params.min_p.has_value());  // "high" is not a number
    bool warned = false;
    for (const auto& w : r.warnings) warned = warned || w.find("min_p") != std::string::npos;
    CHECK(warned);

    CHECK(r.weights_bytes == 24 + 4096 + 24 + 512 + 24 + 64);
    CHECK(r.referenced_bytes > r.weights_bytes);
    CHECK(r.blobs.size() == 8);  // config + 7 layers
    CHECK(r.blobs.front().role == m::BlobRole::config);
}

TEST_CASE("model store: scanning and verifying never write to the store") {
    Store s;
    make_full_model(s);
    const auto before = snapshot(s.tmp.path);
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto scan = store.value().scan();
    REQUIRE(scan.records.size() == 1);
    (void)m::verify_blobs(scan.records[0]);
    (void)store.value().resolve("coder:7b");
    CHECK(snapshot(s.tmp.path) == before);
}

TEST_CASE("model store: legacy tree, hf.co names and host:port encoding") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf","model_family":"llama","model_type":"8B","file_type":"Q8_0"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 100));
    s.put_legacy("hf.co", "bartowski", "LLaMA-Mesh-GGUF", "Q8_0", s.manifest(cfg, {model}));
    s.put_legacy("registry.ollama.ai", "library", "bge-m3", "latest", s.manifest(cfg, {model}));
    s.put_v2("localhost%3A5000", "team", "private", "v1", s.manifest(cfg, {model}));
    // A part that fails Ollama's name rules is skipped and noted, not served.
    s.put_legacy("registry.ollama.ai", "library", "bad name", "latest", s.manifest(cfg, {model}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto scan = store.value().scan();
    CHECK(scan.records.size() == 3);
    const auto* mesh = find(scan, "hf.co/bartowski/LLaMA-Mesh-GGUF:Q8_0");
    REQUIRE(mesh != nullptr);
    CHECK(mesh->layout == m::ManifestLayout::legacy);
    CHECK(mesh->ready());
    CHECK(find(scan, "bge-m3:latest") != nullptr);
    const auto* priv = find(scan, "localhost:5000/team/private:v1");
    REQUIRE(priv != nullptr);
    CHECK(priv->layout == m::ManifestLayout::v2);
    CHECK(scan.notes.size() == 1);

    auto r = store.value().resolve("localhost:5000/team/private:v1");
    REQUIRE(r.ok());
    CHECK(r.value().layout == m::ManifestLayout::v2);
    CHECK(store.value().resolve("hf.co/bartowski/LLaMA-Mesh-GGUF:Q8_0").ok());
    // Default-host names resolve through the legacy registry.ollama.ai tree.
    CHECK(store.value().resolve("bge-m3").ok());
    CHECK(store.value().resolve("library/bge-m3:latest").ok());
    CHECK(store.value().resolve("BGE-M3").ok());  // case-insensitive fallback
}

TEST_CASE("model store: v2 wins over a legacy duplicate and the two public hosts are one registry") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf","file_type":"Q4_K_M"})");
    const auto old_model = s.blob(kModel, fake_gguf(1, 1, 10, 'o'));
    const auto new_model = s.blob(kModel, fake_gguf(2, 2, 20, 'n'));
    s.put_legacy("registry.ollama.ai", "library", "dup", "latest", s.manifest(cfg, {old_model}));
    s.put_v2("ollama.com", "library", "dup", "latest", s.manifest(cfg, {new_model}));
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto scan = store.value().scan();
    REQUIRE(scan.records.size() == 1);
    CHECK(scan.records[0].layout == m::ManifestLayout::v2);
    CHECK(scan.records[0].gguf->tensor_count == 2);
    auto r = store.value().resolve("dup");
    REQUIRE(r.ok());
    CHECK(r.value().layout == m::ManifestLayout::v2);
}

TEST_CASE("model store: unsupported entries are reported with a reason") {
    Store s;
    const auto gguf_cfg = s.config_blob(R"({"model_format":"gguf","file_type":"Q4_K_M"})");

    // Ollama Cloud stub: config only, remote_host set, no layers.
    const auto cloud_cfg = s.config_blob(R"({"remote_host":"https://ollama.com","remote_model":"glm-5.2","capabilities":["thinking"]})");
    s.put_v2("ollama.com", "library", "glm-5.2", "cloud", s.manifest(cloud_cfg, {}));

    // safetensors / MLX: tensor layers and a safetensors config.
    const auto st_cfg = s.config_blob(R"({"model_format":"safetensors","model_family":"gemma"})");
    const auto tensor = s.blob(kTensor, "tensor-bytes");
    s.put_v2("ollama.com", "library", "mlxmodel", "latest", s.manifest(st_cfg, {tensor}, ",\"runner\":\"mlx\",\"format\":\"safetensors\""));

    // Missing blob.
    s.put_v2("ollama.com", "library", "missing", "latest",
             s.manifest(gguf_cfg, {Store::absent(kModel, fake_gguf(1, 1, 300, 'm'))}));

    // Truncated blob (smaller than the manifest says).
    const auto full = fake_gguf(1, 1, 300, 't');
    const auto trunc = s.corrupt_blob(kModel, full, full.substr(0, 100));
    s.put_v2("ollama.com", "library", "trunc", "latest", s.manifest(gguf_cfg, {trunc}));

    // Not a GGUF file.
    const auto junk = s.blob(kModel, std::string(200, 'j'));
    s.put_v2("ollama.com", "library", "junk", "latest", s.manifest(gguf_cfg, {junk}));

    // No model layer.
    const auto sys = s.blob(kSystem, "just a system prompt");
    s.put_v2("ollama.com", "library", "nolayer", "latest", s.manifest(gguf_cfg, {sys}));

    // Broken manifest JSON, and a layer with a malformed digest.
    s.put_v2("ollama.com", "library", "broken", "latest", "{not json");
    s.put_v2("ollama.com", "library", "baddigest", "latest",
             s.manifest(gguf_cfg, {Layer{kModel, "sha256:../../etc/passwd", 10, ""}}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto scan = store.value().scan();
    REQUIRE(scan.records.size() == 8);
    auto expect = [&](const char* name, m::UnsupportedReason reason) {
        CAPTURE(name);
        const auto* r = find(scan, name);
        REQUIRE(r != nullptr);
        CHECK(r->status == m::EntryStatus::unsupported);
        CHECK(r->reason == reason);
        CHECK_FALSE(r->detail.empty());
    };
    expect("glm-5.2:cloud", m::UnsupportedReason::cloud_stub);
    expect("mlxmodel:latest", m::UnsupportedReason::safetensors);
    expect("missing:latest", m::UnsupportedReason::missing_blob);
    expect("trunc:latest", m::UnsupportedReason::blob_size_mismatch);
    expect("junk:latest", m::UnsupportedReason::invalid_gguf);
    expect("nolayer:latest", m::UnsupportedReason::no_model_layer);
    expect("broken:latest", m::UnsupportedReason::manifest_invalid);
    expect("baddigest:latest", m::UnsupportedReason::manifest_invalid);
    CHECK(find(scan, "glm-5.2:cloud")->detail.find("glm-5.2") != std::string::npos);
    CHECK(find(scan, "glm-5.2:cloud")->config.capabilities.size() == 1);

    // Real stores hold cloud stubs whose manifest has "layers": null (Go nil slice).
    const auto null_cfg = s.config_blob(R"({"model_format":"","remote_host":"https://ollama.com:443","remote_model":"gpt-oss:120b"})");
    s.put_legacy("registry.ollama.ai", "library", "gpt-oss", "120b-cloud",
                 std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kManifestType + "\",\"config\":" + layer_json(null_cfg) +
                     ",\"layers\":null}");
    scan = store.value().scan();
    REQUIRE(scan.records.size() == 9);
    CHECK(find(scan, "gpt-oss:120b-cloud")->reason == m::UnsupportedReason::cloud_stub);

    // Resolving an unsupported entry yields the record (the CLI explains it) ...
    auto cloud = store.value().resolve("glm-5.2:cloud");
    REQUIRE(cloud.ok());
    CHECK_FALSE(cloud.value().ready());
}

TEST_CASE("model store: manifest lists select a GGUF child and reject MLX-only lists") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf","file_type":"Q4_K_M"})");
    const auto model = s.blob(kModel, fake_gguf(5, 5, 50));
    const std::string child = s.manifest(cfg, {model});
    const std::string child_hex = si::detail::sha256_hex(child);
    write_file(s.tmp.path / "blobs" / ("sha256-" + child_hex), child);

    const std::string mlx_inline = std::string("{\"mediaType\":\"") + kManifestType +
                                   "\",\"runner\":\"mlx\",\"format\":\"safetensors\",\"config\":" + layer_json(cfg) + ",\"layers\":[]}";
    const std::string list = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[" + mlx_inline +
                             ",{\"mediaType\":\"" + kManifestType + "\",\"digest\":\"sha256:" + child_hex +
                             "\",\"runner\":\"llamacpp\",\"format\":\"gguf\"}]}";
    s.put_v2("ollama.com", "library", "multi", "latest", list);

    const std::string mlx_only = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[" + mlx_inline + "]}";
    s.put_v2("ollama.com", "library", "mlxonly", "latest", mlx_only);
    const std::string none = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[]}";
    s.put_v2("ollama.com", "library", "emptylist", "latest", none);

    // A reference whose blob content does not match its digest is rejected.
    write_file(s.tmp.path / "blobs" / ("sha256-" + std::string(64, 'a')), child);
    const std::string tampered = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[{\"mediaType\":\"" +
                                 kManifestType + "\",\"digest\":\"sha256:" + std::string(64, 'a') +
                                 "\",\"runner\":\"llamacpp\",\"format\":\"gguf\"}]}";
    s.put_v2("ollama.com", "library", "tampered", "latest", tampered);

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto multi = store.value().resolve("multi");
    REQUIRE(multi.ok());
    CHECK(multi.value().ready());
    CHECK(multi.value().runner == "llamacpp");
    CHECK(multi.value().gguf->tensor_count == 5);
    CHECK(store.value().resolve("mlxonly").value().reason == m::UnsupportedReason::safetensors);
    CHECK(store.value().resolve("emptylist").value().reason == m::UnsupportedReason::no_compatible_manifest);
    CHECK(store.value().resolve("tampered").value().reason == m::UnsupportedReason::manifest_invalid);
}

TEST_CASE("model store: resolve rejects bad names and reports unknown ones") {
    Store s;
    make_full_model(s);
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto bad = store.value().resolve("../../etc/passwd");
    CHECK_FALSE(bad.ok());
    CHECK(bad.status().code() == si::ErrorCode::invalid_argument);
    auto unknown = store.value().resolve("nothing:here");
    CHECK_FALSE(unknown.ok());
    CHECK(unknown.status().code() == si::ErrorCode::not_found);
    CHECK(store.value().resolve("coder:7b").ok());
    CHECK(store.value().resolve("library/coder:7b").ok());
    CHECK(store.value().resolve("registry.ollama.ai/library/coder:7b").ok());
    CHECK_FALSE(store.value().resolve("coder").ok());  // default tag is latest, not 7b
}

namespace {

// Changes the process working directory for one scope (doctest runs cases one at a time).
struct CwdGuard {
    fs::path old;
    explicit CwdGuard(const fs::path& to) : old(fs::current_path()) { fs::current_path(to); }
    ~CwdGuard() {
        std::error_code ec;
        fs::current_path(old, ec);
    }
    CwdGuard(const CwdGuard&) = delete;
    CwdGuard& operator=(const CwdGuard&) = delete;
};

}  // namespace

TEST_CASE("model store: a drive-shaped host cannot make resolve read outside the store") {
    Store s;
    make_full_model(s);
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(3, 3, 32));

    // A decoy manifest outside the store, placed where the drive-relative path "<d>:5000\ns\m\t"
    // lands when the working directory is `outside` (Windows: operator/ drops the store root
    // for a right-hand side that has a different root-name).
    TempDir outside;
    std::string drive = outside.path.root_name().string();
    drive = drive.empty() ? std::string("c")
                          : std::string(1, static_cast<char>(std::tolower(static_cast<unsigned char>(drive[0]))));
    const std::string hostile = drive + ":5000/ns/m:t";
    write_file(outside.path / "5000" / "ns" / "m" / "t", s.manifest(cfg, {model}));
    CwdGuard cwd(outside.path);

#if defined(_WIN32)
    // The decoy really is reachable through the naive legacy path, so the checks below are not vacuous.
    CHECK(fs::is_regular_file(s.tmp.path / "manifests" / (drive + ":5000") / "ns" / "m" / "t"));
#endif

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto r = store.value().resolve(hostile);
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() == si::ErrorCode::not_found);

    m::ModelRegistry reg;
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    CHECK(reg.resolve(hostile).status().code() == si::ErrorCode::not_found);

    // Same through the CLI that takes request-style names; nothing is printed as a record.
    std::ostringstream out, err;
    CHECK(m::models_main({"import-ollama", s.tmp.path.string(), "--name", hostile, "--json"}, out, err) == 1);
    CHECK(out.str().empty());
    CHECK(err.str().find("not in") != std::string::npos);

    // A host that ends in dots after the port is not even a valid name (a part may not end in '.').
    CHECK(store.value().resolve(drive + ":5000../ns/m:t").status().code() == si::ErrorCode::invalid_argument);

    // The v2 spelling of a port host still works.
    s.put_v2(drive + "%3A5000", "ns", "m", "t", s.manifest(cfg, {model}));
    auto v2 = store.value().resolve(hostile);
    REQUIRE(v2.ok());
    CHECK(v2.value().layout == m::ManifestLayout::v2);
    CHECK(fs::equivalent(v2.value().manifest_path, s.tmp.path / "manifests-v2" / (drive + "%3A5000") / "ns" / "m" / "t"));
}

TEST_CASE("model store: shards, drafts, messages, legacy prompt and Go-style option conversion") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto part1 = s.blob(kModel, fake_gguf(2, 2, 40, '1'));
    const auto part2 = s.blob(kModel, fake_gguf(2, 2, 41, '2'));
    const auto draft1 = s.blob("application/vnd.ollama.image.draft", fake_gguf(1, 1, 20, 'd'));
    const auto draft2 = s.blob("application/vnd.ollama.image.draft", fake_gguf(1, 1, 21, 'e'));
    const auto prompt = s.blob("application/vnd.ollama.image.prompt", "legacy {{ .Prompt }}");
    const auto messages = s.blob("application/vnd.ollama.image.messages", R"([{"role":"user","content":"hi"}])");
    const auto params = s.blob(kParams, R"({"num_ctx":4096.9,"top_k":-3.7,"num_predict":"many"})");
    // (A system layer over the 4 MiB cap makes the entry unsupported: see "text and config blobs must
    // match their declared size and digest".)
    s.put_v2("ollama.com", "library", "layers", "latest",
             s.manifest(cfg, {part1, part2, draft1, draft2, prompt, messages, params}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto rec = store.value().resolve("layers");
    REQUIRE(rec.ok());
    const m::ModelRecord& r = rec.value();
    CHECK(r.ready());
    CHECK(r.model_path == s.blob_file(part1));
    REQUIRE(r.shard_paths.size() == 1);
    CHECK(r.shard_paths[0] == s.blob_file(part2));
    REQUIRE(r.draft_paths.size() == 2);  // first is the draft model, the second its shard
    CHECK(r.draft_paths[0] == s.blob_file(draft1));
    CHECK(r.draft_paths[1] == s.blob_file(draft2));
    REQUIRE(r.template_text.has_value());
    CHECK(*r.template_text == "legacy {{ .Prompt }}");
    REQUIRE(r.messages_json.has_value());
    CHECK(r.messages_json->find("\"hi\"") != std::string::npos);
    CHECK(r.params.num_ctx == 4096);  // float64 -> int truncates, like Ollama
    CHECK(r.params.top_k == -3);
    CHECK_FALSE(r.params.num_predict.has_value());  // a string is still rejected
    CHECK_FALSE(r.system_prompt.has_value());
}

TEST_CASE("model store: manifest list falls back past a reference whose blob is absent") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(7, 7, 60));
    const std::string ggml_child = std::string("{\"mediaType\":\"") + kManifestType +
                                   "\",\"runner\":\"ggml\",\"format\":\"gguf\",\"config\":" + layer_json(cfg) +
                                   ",\"layers\":[" + layer_json(model) + "]}";
    const std::string missing_ref = std::string("{\"mediaType\":\"") + kManifestType + "\",\"digest\":\"sha256:" +
                                    std::string(64, 'c') + "\",\"runner\":\"llamacpp\",\"format\":\"gguf\",\"layers\":null}";
    const std::string list = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[" +
                             missing_ref + "," + ggml_child + "]}";
    s.put_v2("ollama.com", "library", "fallback", "latest", list);
    const std::string only_missing = std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[" +
                                     missing_ref + "]}";
    s.put_v2("ollama.com", "library", "onlymissing", "latest", only_missing);

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto fb = store.value().resolve("fallback");
    REQUIRE(fb.ok());
    CHECK(fb.value().ready());
    CHECK(fb.value().runner == "ggml");
    CHECK(fb.value().gguf->tensor_count == 7);
    auto om = store.value().resolve("onlymissing");
    REQUIRE(om.ok());
    CHECK(om.value().reason == m::UnsupportedReason::missing_blob);

    // A reference that carries "layers":null is still a reference (Ollama isReference), not an
    // inline child with no model layer.
    const std::string child = s.manifest(cfg, {model});
    const std::string child_hex = si::detail::sha256_hex(child);
    write_file(s.tmp.path / "blobs" / ("sha256-" + child_hex), child);
    const std::string null_ref = std::string("{\"mediaType\":\"") + kManifestType + "\",\"digest\":\"sha256:" + child_hex +
                                 "\",\"runner\":\"llamacpp\",\"format\":\"gguf\",\"layers\":null}";
    s.put_v2("ollama.com", "library", "nullref", "latest",
             std::string("{\"schemaVersion\":2,\"mediaType\":\"") + kListType + "\",\"manifests\":[" + null_ref + "]}");
    auto nr = store.value().resolve("nullref");
    REQUIRE(nr.ok());
    CHECK(nr.value().ready());
    CHECK(nr.value().runner == "llamacpp");
}

TEST_CASE("model store: the CLI counts a weights file shared by two entries once") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 1000));
    s.put_v2("ollama.com", "library", "alpha", "latest", s.manifest(cfg, {model}));
    s.put_legacy("registry.ollama.ai", "library", "beta", "latest", s.manifest(cfg, {model}));
    std::ostringstream out, err;
    REQUIRE(m::models_main({"import-ollama", s.tmp.path.string()}, out, err) == 0);
    const std::string text = out.str();
    CAPTURE(text);
    CHECK(text.find("2 entries: 2 ready (1.0 KiB of distinct weight files") != std::string::npos);
    CHECK(text.find("same weights file as alpha:latest") != std::string::npos);
    CHECK(text.find("1 entries share a weights file") != std::string::npos);
}

TEST_CASE("model store: digest verification is opt-in and catches corruption") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto good = s.blob(kModel, fake_gguf(1, 1, 1000, 'g'));
    const auto sys = s.blob(kSystem, "hello");
    s.put_v2("ollama.com", "library", "good", "latest", s.manifest(cfg, {good, sys}));

    // Same size and valid GGUF header, different tail: scan accepts it, verify must not.
    const auto claimed = fake_gguf(1, 1, 1000, 'c');
    const auto bad = s.corrupt_blob(kModel, claimed, fake_gguf(1, 1, 1000, 'X'));
    s.put_v2("ollama.com", "library", "bad", "latest", s.manifest(cfg, {bad}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto scan = store.value().scan();
    REQUIRE(find(scan, "good:latest")->ready());
    REQUIRE(find(scan, "bad:latest")->ready());  // not hashed during the scan

    std::vector<std::string> started;
    m::VerifyOptions vo;
    vo.on_blob_start = [&](const m::BlobRef& b) { started.push_back(b.digest); };
    auto ok = m::verify_blobs(*find(scan, "good:latest"), vo);
    CHECK(ok.ok());
    CHECK(ok.blobs.size() == 3);  // config + model + system
    CHECK(started.size() == 3);
    for (const auto& b : ok.blobs) CHECK(b.actual_sha256 == b.blob.digest.substr(7));

    m::VerifyOptions weights;
    weights.weights_only = true;
    auto only = m::verify_blobs(*find(scan, "good:latest"), weights);
    CHECK(only.ok());
    REQUIRE(only.blobs.size() == 1);
    CHECK(only.blobs[0].blob.role == m::BlobRole::model);

    auto fail = m::verify_blobs(*find(scan, "bad:latest"));
    CHECK_FALSE(fail.ok());
    bool saw_mismatch = false;
    for (const auto& b : fail.blobs) {
        if (b.verdict == m::BlobVerdict::mismatch) {
            saw_mismatch = true;
            CHECK(b.detail.find("sha256:") != std::string::npos);
        }
    }
    CHECK(saw_mismatch);

    // A blob deleted after the scan is reported missing.
    fs::remove(s.blob_file(good));
    auto gone = m::verify_blobs(*find(scan, "good:latest"));
    CHECK_FALSE(gone.ok());
    bool saw_missing = false;
    for (const auto& b : gone.blobs) saw_missing = saw_missing || b.verdict == m::BlobVerdict::missing;
    CHECK(saw_missing);

    // Cancellation stops before hashing anything.
    si::CancellationSource src;
    src.cancel();
    m::VerifyOptions cancelled;
    cancelled.cancel = src.token();
    auto stopped = m::verify_blobs(*find(scan, "bad:latest"), cancelled);
    CHECK_FALSE(stopped.ok());
    REQUIRE_FALSE(stopped.blobs.empty());
    CHECK(stopped.blobs[0].verdict == m::BlobVerdict::cancelled);
}

TEST_CASE("model store: registry resolves ready models by name, explains the rest") {
    Store s;
    make_full_model(s);
    const auto cloud_cfg = s.config_blob(R"({"remote_host":"https://ollama.com","remote_model":"kimi-k3"})");
    s.put_v2("ollama.com", "library", "kimi-k3", "cloud", s.manifest(cloud_cfg, {}));

    m::ModelRegistry reg;
    CHECK_FALSE(reg.add_ollama_dir(s.tmp.path / "absent").ok());
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());  // duplicate root ignored
    CHECK(reg.list().size() == 2);

    auto hit = reg.resolve("coder:7b");
    REQUIRE(hit.ok());
    CHECK(hit.value().model_path.filename().string().rfind("sha256-", 0) == 0);

    auto cloud = reg.resolve("kimi-k3:cloud");
    CHECK_FALSE(cloud.ok());
    CHECK(cloud.status().code() == si::ErrorCode::unsupported);
    CHECK(cloud.status().message().find("cloud_stub") != std::string::npos);

    auto miss = reg.resolve("ghost:1");
    CHECK(miss.status().code() == si::ErrorCode::not_found);
    CHECK(reg.resolve("bad name").status().code() == si::ErrorCode::invalid_argument);

    // A model pulled after the scan is found through the live fallback, and by refresh().
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    s.put_v2("ollama.com", "library", "late", "latest", s.manifest(cfg, {s.blob(kModel, fake_gguf(1, 1, 8))}));
    CHECK(reg.resolve("late").ok());
    CHECK(reg.list().size() == 2);
    reg.refresh();
    CHECK(reg.list().size() == 3);

    // The first registered store wins for a shared name.
    Store second;
    const auto cfg2 = second.config_blob(R"({"model_format":"gguf"})");
    second.put_v2("ollama.com", "library", "coder", "7b", second.manifest(cfg2, {second.blob(kModel, fake_gguf(99, 1, 8))}));
    REQUIRE(reg.add_ollama_dir(second.tmp.path).ok());
    CHECK(reg.resolve("coder:7b").value().gguf->tensor_count == 291);
}

TEST_CASE("model store: JSON document shape") {
    Store s;
    make_full_model(s);
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    const auto doc = m::to_json(store.value().scan(), store.value().root());
    auto parsed = si::json::parse(doc.dump());
    REQUIRE(parsed.ok());
    CHECK(parsed.value().find("schema")->as_string() == "sonder.inference.model_store/1");
    CHECK(parsed.value().find("source")->find("kind")->as_string() == "ollama");
    const auto& models = parsed.value().find("models")->as_array();
    REQUIRE(models.size() == 1);
    const auto& rec = models[0];
    CHECK(rec.find("status")->as_string() == "ready");
    CHECK(rec.find("reason")->is_null());
    CHECK(rec.find("files")->find("model")->as_string().find("sha256-") != std::string::npos);
    CHECK(rec.find("params")->find("num_ctx")->as_int() == 8192);
    CHECK(rec.find("params")->find("stop")->as_array().size() == 2);
    CHECK(rec.find("template")->as_string() == "{{ .Prompt }}");
    CHECK(rec.find("gguf")->find("tensor_count")->as_int() == 291);
}

TEST_CASE("model store: import-ollama CLI lists, filters, verifies and rejects bad usage") {
    Store s;
    make_full_model(s);
    const auto cloud_cfg = s.config_blob(R"({"remote_host":"https://ollama.com","remote_model":"kimi-k3"})");
    s.put_v2("ollama.com", "library", "kimi-k3", "cloud", s.manifest(cloud_cfg, {}));
    const std::string dir = s.tmp.path.string();

    auto run = [&](std::vector<std::string> args, std::string& out_text, std::string& err_text) {
        std::ostringstream out, err;
        const int rc = m::models_main(args, out, err);
        out_text = out.str();
        err_text = err.str();
        return rc;
    };
    std::string out, err;

    CHECK(run({"import-ollama", dir}, out, err) == 0);
    CHECK(out.find("coder:7b") != std::string::npos);
    CHECK(out.find("ready") != std::string::npos);
    CHECK(out.find("unsupported") != std::string::npos);
    CHECK(out.find("cloud_stub") != std::string::npos);
    CHECK(out.find("1 ready") != std::string::npos);
    CHECK(out.find("digests not verified") != std::string::npos);

    CHECK(run({"import-ollama", dir, "--json"}, out, err) == 0);
    auto doc = si::json::parse(out);
    REQUIRE(doc.ok());
    CHECK(doc.value().find("models")->as_array().size() == 2);

    CHECK(run({"import-ollama", dir, "--name", "coder:7b", "--json", "--verify", "--weights-only"}, out, err) == 0);
    auto verified = si::json::parse(out);
    REQUIRE(verified.ok());
    const auto& one = verified.value().find("models")->as_array();
    REQUIRE(one.size() == 1);
    CHECK(one[0].find("verification")->find("ok")->as_bool());
    CHECK(one[0].find("verification")->find("blobs")->as_array().size() == 3);  // model, projector, adapter
    CHECK(err.find("verifying coder:7b model") != std::string::npos);

    // Corrupt the model blob in place (same size): scan stays ready, --verify fails with exit 1.
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    const auto rec = store.value().resolve("coder:7b").value();
    write_file(rec.model_path, fake_gguf(291, 26, 4096, 'Z'));
    CHECK(run({"import-ollama", dir, "--name", "coder:7b", "--verify"}, out, err) == 1);
    CHECK(out.find("FAILED") != std::string::npos);
    CHECK(run({"import-ollama", dir, "--name", "coder:7b"}, out, err) == 0);

    CHECK(run({"import-ollama", dir, "--name", "ghost"}, out, err) == 1);
    CHECK(err.find("not in") != std::string::npos);
    CHECK(run({"import-ollama", (s.tmp.path / "nope").string()}, out, err) == 1);
    CHECK(run({"import-ollama", dir, "--bogus"}, out, err) == 2);
    CHECK(run({"import-ollama", dir, "--weights-only"}, out, err) == 2);
    CHECK(run({"import-ollama", dir, "extra"}, out, err) == 2);
    CHECK(run({"import-ollama", dir, "--name"}, out, err) == 2);
    CHECK(run({"import-ollama", dir, "--name", "bad name"}, out, err) == 2);
    CHECK(run({}, out, err) == 2);
    CHECK(run({"export"}, out, err) == 2);
    CHECK(run({"import-ollama", "--help"}, out, err) == 0);
    CHECK(out.find("usage: sonder-infer models import-ollama") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Security review of PR 135 (RET-04): links, verification, amplification, schema, names, messages.
// ---------------------------------------------------------------------------
namespace {

// Creates a directory link at `link` pointing to `target`: a symlink first; on Windows, where that needs a
// privilege, a junction (`mklink /J`), which any user can create (same helper as test_weight_manifest.cpp).
bool make_dir_link(const fs::path& link, const fs::path& target) {
    std::error_code ec;
    fs::create_directories(link.parent_path(), ec);
    fs::create_directory_symlink(target, link, ec);
    if (!ec) return true;
#if defined(_WIN32)
    const std::string cmd = "mklink /J \"" + link.string() + "\" \"" + target.string() + "\" >nul 2>nul";
    if (std::system(cmd.c_str()) == 0) {
        std::error_code probe;
        return fs::exists(link, probe);
    }
#endif
    return false;
}

// Removes a directory link without touching its target.
void remove_dir_link(const fs::path& link) {
    std::error_code ec;
    fs::remove(link, ec);
}

bool any_contains(const std::vector<std::string>& v, const std::string& needle) {
    for (const auto& s : v) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

std::string scan_json(const fs::path& root) {
    auto store = m::OllamaStore::open(root);
    if (!store.ok()) return {};
    return m::to_json(store.value().scan(), store.value().root()).dump();
}

}  // namespace

TEST_CASE("model store: a text layer hardlinked to a file outside the store is never read") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    TempDir outside;
    const std::string secret = "TOP-SECRET-hardlink-payload";
    write_file(outside.path / "secret.txt", secret);

    // The digest names the secret's real hash and size, so only the link count gives it away.
    const std::string hex = si::detail::sha256_hex(secret);
    std::error_code ec;
    fs::create_hard_link(outside.path / "secret.txt", s.tmp.path / "blobs" / ("sha256-" + hex), ec);
    REQUIRE_MESSAGE(!ec, "hard link creation failed: " << ec.message());
    s.put_v2("ollama.com", "library", "evil", "latest",
             s.manifest(cfg, {model, Layer{kTemplate, "sha256:" + hex, secret.size(), ""}}));

    // The reviewer's form: a different claimed digest and size 1.
    const std::string fake_hex(64, 'a');
    fs::create_hard_link(outside.path / "secret.txt", s.tmp.path / "blobs" / ("sha256-" + fake_hex), ec);
    REQUIRE(!ec);
    s.put_v2("ollama.com", "library", "evil2", "latest",
             s.manifest(cfg, {model, Layer{kSystem, "sha256:" + fake_hex, 1, ""}}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    for (const char* name : {"evil", "evil2"}) {
        CAPTURE(name);
        auto r = store.value().resolve(name);
        REQUIRE(r.ok());
        CHECK_FALSE(r.value().ready());
        CHECK(r.value().reason == m::UnsupportedReason::unsafe_path);
        CHECK(r.value().detail.find("hard links") != std::string::npos);
        CHECK_FALSE(r.value().template_text.has_value());
        CHECK_FALSE(r.value().system_prompt.has_value());
    }
    CHECK(scan_json(s.tmp.path).find("TOP-SECRET") == std::string::npos);
    m::ModelRegistry reg;
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    CHECK_FALSE(reg.resolve("evil").ok());
    CHECK_FALSE(reg.resolve("evil2").ok());
}

TEST_CASE("model store: a blob that is a symlink to a file outside the store is never read") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    TempDir outside;
    const std::string secret = "TOP-SECRET-symlink-payload";
    write_file(outside.path / "secret.txt", secret);
    const std::string hex = si::detail::sha256_hex(secret);
    std::error_code ec;
    fs::create_symlink(outside.path / "secret.txt", s.tmp.path / "blobs" / ("sha256-" + hex), ec);
    if (ec) {
#if defined(_WIN32)
        MESSAGE("file symlinks need a privilege here (" << ec.message() << "); covered by the junction cases");
        return;
#else
        FAIL_CHECK("file symlink creation failed: " << ec.message());
        return;
#endif
    }
    s.put_v2("ollama.com", "library", "evil", "latest",
             s.manifest(cfg, {model, Layer{kSystem, "sha256:" + hex, secret.size(), ""}}));
    // A weight blob symlinked to a GGUF outside the store is not served either.
    TempDir outside_gguf;
    const std::string ext = fake_gguf(4, 4, 16, 'e');
    write_file(outside_gguf.path / "ext.gguf", ext);
    const std::string ext_hex = si::detail::sha256_hex(ext);
    fs::create_symlink(outside_gguf.path / "ext.gguf", s.tmp.path / "blobs" / ("sha256-" + ext_hex), ec);
    REQUIRE(!ec);
    s.put_v2("ollama.com", "library", "extgguf", "latest",
             s.manifest(cfg, {Layer{kModel, "sha256:" + ext_hex, ext.size(), ""}}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto r = store.value().resolve("evil");
    REQUIRE(r.ok());
    CHECK_FALSE(r.value().ready());
    CHECK(r.value().reason == m::UnsupportedReason::unsafe_path);
    CHECK_FALSE(r.value().system_prompt.has_value());
    auto g = store.value().resolve("extgguf");
    REQUIRE(g.ok());
    CHECK_FALSE(g.value().ready());
    CHECK(g.value().reason == m::UnsupportedReason::unsafe_path);
    // read_gguf_header refuses the link too.
    CHECK_FALSE(m::read_gguf_header(s.tmp.path / "blobs" / ("sha256-" + ext_hex)).ok());
    CHECK(scan_json(s.tmp.path).find("TOP-SECRET") == std::string::npos);
}

TEST_CASE("model store: directory links (junction or symlink) under the store are refused") {
    // Case 1: manifests-v2/ollama.com/library is a link to a directory outside the store.
    {
        Store s;
        const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
        const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
        TempDir outside;
        write_file(outside.path / "evil" / "latest", s.manifest(cfg, {model}));
        fs::create_directories(s.tmp.path / "manifests-v2" / "ollama.com");
        if (!make_dir_link(s.tmp.path / "manifests-v2" / "ollama.com" / "library", outside.path)) {
            FAIL_CHECK("no directory link (symlink or junction) could be created here; link refusal NOT exercised");
            return;
        }
        auto store = m::OllamaStore::open(s.tmp.path);
        REQUIRE(store.ok());
        auto scan = store.value().scan();
        CHECK(find(scan, "evil:latest") == nullptr);
        CHECK(any_contains(scan.notes, "manifests-v2/ollama.com/library: a link or reparse point"));
        auto r = store.value().resolve("evil");
        REQUIRE(r.ok());
        CHECK_FALSE(r.value().ready());
        CHECK(r.value().reason == m::UnsupportedReason::unsafe_path);
        remove_dir_link(s.tmp.path / "manifests-v2" / "ollama.com" / "library");
    }
    // Case 2: blobs/ itself is a link to a directory outside the store.
    {
        Store s;
        TempDir outside;
        const std::string cfg_text = R"({"model_format":"gguf"})";
        const std::string gguf = fake_gguf(1, 1, 8);
        const Layer cfg{kConfigType, "sha256:" + si::detail::sha256_hex(cfg_text), cfg_text.size(), ""};
        const Layer model{kModel, "sha256:" + si::detail::sha256_hex(gguf), gguf.size(), ""};
        write_file(outside.path / ("sha256-" + cfg.digest.substr(7)), cfg_text);
        write_file(outside.path / ("sha256-" + model.digest.substr(7)), gguf);
        s.put_v2("ollama.com", "library", "linked", "latest", s.manifest(cfg, {model}));
        if (!make_dir_link(s.tmp.path / "blobs", outside.path)) {
            FAIL_CHECK("no directory link could be created here; link refusal NOT exercised");
            return;
        }
        auto store = m::OllamaStore::open(s.tmp.path);
        REQUIRE(store.ok());
        auto r = store.value().resolve("linked");
        REQUIRE(r.ok());
        CHECK_FALSE(r.value().ready());
        CHECK(r.value().reason == m::UnsupportedReason::unsafe_path);
        remove_dir_link(s.tmp.path / "blobs");
    }
}

#if defined(_WIN32)
TEST_CASE("model store: NTFS junctions for a manifest directory or blobs are refused") {
    // make_dir_link() prefers a symlink, which Developer Mode allows; this case always uses a junction.
    auto junction = [](const fs::path& link, const fs::path& target) {
        std::error_code ec;
        fs::create_directories(link.parent_path(), ec);
        const std::string cmd = "mklink /J \"" + link.string() + "\" \"" + target.string() + "\" >nul 2>nul";
        return std::system(cmd.c_str()) == 0 && fs::exists(link, ec);
    };
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    TempDir outside;
    write_file(outside.path / "lib" / "evil" / "latest", s.manifest(cfg, {model}));
    s.put_v2("ollama.com", "library", "fine", "latest", s.manifest(cfg, {model}));
    REQUIRE_MESSAGE(junction(s.tmp.path / "manifests-v2" / "ollama.com" / "team", outside.path / "lib"),
                    "mklink /J failed; junction refusal NOT exercised");
    {
        std::error_code ec;
        const auto st = fs::symlink_status(s.tmp.path / "manifests-v2" / "ollama.com" / "team", ec);
        CHECK(st.type() != fs::file_type::symlink);  // really a junction, not a symlink
    }
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto scan = store.value().scan();
    CHECK(find(scan, "team/evil:latest") == nullptr);
    CHECK(any_contains(scan.notes, "manifests-v2/ollama.com/team: a link or reparse point"));
    auto r = store.value().resolve("team/evil");
    REQUIRE(r.ok());
    CHECK(r.value().reason == m::UnsupportedReason::unsafe_path);
    CHECK(store.value().resolve("fine").value().ready());
    remove_dir_link(s.tmp.path / "manifests-v2" / "ollama.com" / "team");

    // blobs/ as a junction to a directory holding the same files.
    TempDir other;
    fs::create_directories(other.path / "b");
    for (const auto& e : fs::directory_iterator(s.tmp.path / "blobs")) fs::copy_file(e.path(), other.path / "b" / e.path().filename());
    fs::remove_all(s.tmp.path / "blobs");
    REQUIRE(junction(s.tmp.path / "blobs", other.path / "b"));
    auto linked = store.value().resolve("fine");
    REQUIRE(linked.ok());
    CHECK(linked.value().reason == m::UnsupportedReason::unsafe_path);
    remove_dir_link(s.tmp.path / "blobs");
}
#endif

TEST_CASE("model store: a manifest leaf symlink is followed only into a matching blob") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    const std::string manifest = s.manifest(cfg, {model});
    const auto mblob = s.blob(kManifestType, manifest);  // Ollama keeps the manifest as a blob too
    const fs::path leaf_dir = s.tmp.path / "manifests-v2" / "ollama.com" / "library" / "linked";
    fs::create_directories(leaf_dir);
    std::error_code ec;
    // Relative link the way Ollama writes it (manifest/manifest.go: os.Symlink(rel, ...)).
    fs::create_symlink(fs::path("../../../../blobs") / ("sha256-" + mblob.digest.substr(7)), leaf_dir / "latest", ec);
    if (ec) {
#if defined(_WIN32)
        MESSAGE("file symlinks need a privilege here (" << ec.message() << "); manifest-link case not exercised");
        return;
#else
        FAIL_CHECK("file symlink creation failed: " << ec.message());
        return;
#endif
    }
    // A link whose target has a blob-shaped name but lies outside the store.
    TempDir outside;
    write_file(outside.path / ("sha256-" + mblob.digest.substr(7)), manifest);
    const fs::path out_dir = s.tmp.path / "manifests-v2" / "ollama.com" / "library" / "outlink";
    fs::create_directories(out_dir);
    fs::create_symlink(outside.path / ("sha256-" + mblob.digest.substr(7)), out_dir / "latest", ec);
    REQUIRE(!ec);
    // A link into blobs/ whose content does not hash to the blob name.
    const std::string other = s.manifest(cfg, {model}, ",\"runner\":\"llamacpp\"");
    write_file(s.tmp.path / "blobs" / ("sha256-" + std::string(64, 'b')), other);
    const fs::path bad_dir = s.tmp.path / "manifests-v2" / "ollama.com" / "library" / "badlink";
    fs::create_directories(bad_dir);
    fs::create_symlink(fs::path("../../../../blobs") / ("sha256-" + std::string(64, 'b')), bad_dir / "latest", ec);
    REQUIRE(!ec);

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto out = store.value().resolve("outlink");
    REQUIRE(out.ok());
    CHECK_FALSE(out.value().ready());
    CHECK(out.value().reason == m::UnsupportedReason::unsafe_path);
    auto bad = store.value().resolve("badlink");
    REQUIRE(bad.ok());
    CHECK_FALSE(bad.value().ready());
#if defined(_WIN32)
    CHECK(bad.value().reason == m::UnsupportedReason::unsafe_path);
    // Windows: Ollama writes copies there (manifest.go "Avoid symlinks on windows"); a link is refused.
    auto win = store.value().resolve("linked");
    REQUIRE(win.ok());
    CHECK_FALSE(win.value().ready());
    CHECK(win.value().reason == m::UnsupportedReason::unsafe_path);
#else
    CHECK(bad.value().reason == m::UnsupportedReason::manifest_invalid);
    auto ok = store.value().resolve("linked");
    REQUIRE(ok.ok());
    CHECK(ok.value().ready());
    // The link itself is a source, so retargeting it makes a cached record stale.
    bool has_link_source = false;
    for (const auto& src : ok.value().sources) has_link_source = has_link_source || src.link;
    CHECK(has_link_source);
    auto scan = store.value().scan();
    REQUIRE(find(scan, "linked:latest") != nullptr);
    CHECK(find(scan, "linked:latest")->ready());
#endif
}

TEST_CASE("model store: text and config blobs must match their declared size and digest") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    // Same size, different bytes (a swapped system prompt).
    const auto swapped = s.corrupt_blob(kSystem, "You are Sonder.", "You are Evil!!!");
    s.put_v2("ollama.com", "library", "swapped", "latest", s.manifest(cfg, {model, swapped}));
    // Declared size differs from the file.
    auto tmpl = s.blob(kTemplate, "{{ .Prompt }}");
    tmpl.size = 3;
    s.put_v2("ollama.com", "library", "tsize", "latest", s.manifest(cfg, {model, tmpl}));
    // A config whose content does not match its digest.
    const auto bad_cfg = s.corrupt_blob(kConfigType, R"({"model_format":"gguf","n":1})", R"({"model_format":"ggml","n":1})");
    s.put_v2("ollama.com", "library", "badcfg", "latest", s.manifest(bad_cfg, {model}));
    // A config with a malformed digest.
    s.put_v2("ollama.com", "library", "cfgdigest", "latest",
             s.manifest(Layer{kConfigType, "sha256:bad", 10, ""}, {model}));
    // A params layer whose bytes are not what the digest names.
    const auto bad_params = s.corrupt_blob(kParams, R"({"num_ctx":2048})", R"({"num_ctx":9999})");
    s.put_v2("ollama.com", "library", "badparams", "latest", s.manifest(cfg, {model, bad_params}));
    // A system layer above the 4 MiB cap is not silently dropped.
    const auto big = s.blob(kSystem, std::string((4u << 20) + 1, 's'));
    s.put_v2("ollama.com", "library", "bigsys", "latest", s.manifest(cfg, {model, big}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    const std::pair<const char*, m::UnsupportedReason> cases[] = {
        {"swapped", m::UnsupportedReason::digest_mismatch},   {"tsize", m::UnsupportedReason::blob_size_mismatch},
        {"badcfg", m::UnsupportedReason::digest_mismatch},    {"cfgdigest", m::UnsupportedReason::manifest_invalid},
        {"badparams", m::UnsupportedReason::digest_mismatch}, {"bigsys", m::UnsupportedReason::manifest_invalid}};
    for (const auto& [name, reason] : cases) {
        CAPTURE(name);
        auto r = store.value().resolve(name);
        REQUIRE(r.ok());
        CHECK_FALSE(r.value().ready());
        CHECK(r.value().reason == reason);
        CHECK_FALSE(r.value().detail.empty());
    }
    CHECK_FALSE(store.value().resolve("swapped").value().system_prompt.has_value());
    CHECK_FALSE(store.value().resolve("badparams").value().params.num_ctx.has_value());
}

TEST_CASE("model store: resolve does not serve a weight blob whose content does not match its digest") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto claimed = fake_gguf(1, 1, 1000, 'A');
    const auto bad = s.corrupt_blob(kModel, claimed, fake_gguf(1, 1, 1000, 'B'));
    s.put_v2("ollama.com", "library", "bad", "latest", s.manifest(cfg, {bad}));
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto r = store.value().resolve("bad");
    REQUIRE(r.ok());
    CHECK_FALSE(r.value().ready());
    CHECK(r.value().reason == m::UnsupportedReason::digest_mismatch);
    m::ModelRegistry reg;
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    auto via = reg.resolve("bad");
    CHECK_FALSE(via.ok());
    CHECK(via.status().code() == si::ErrorCode::unsupported);
    CHECK(via.status().message().find("digest_mismatch") != std::string::npos);
}

TEST_CASE("model store: a cached registry record is revalidated before it is returned") {
    Store s;
    make_full_model(s);
    m::ModelRegistry reg;
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    auto first = reg.resolve("coder:7b");
    REQUIRE(first.ok());
    const fs::path model = first.value().model_path;

    // Same size, valid header, different content after the scan. The write time is moved on
    // explicitly: a rewrite inside the file system's timestamp granularity would otherwise keep
    // it (the documented limit of the identity check).
    write_file(model, fake_gguf(291, 26, 4096, 'Z'));
    fs::last_write_time(model, fs::last_write_time(model) + std::chrono::seconds(5));
    auto swapped = reg.resolve("coder:7b");
    CHECK_FALSE(swapped.ok());
    CHECK(swapped.status().message().find("digest_mismatch") != std::string::npos);

    // Truncated after the scan.
    write_file(model, fake_gguf(291, 26, 10));
    auto truncated = reg.resolve("coder:7b");
    CHECK_FALSE(truncated.ok());
}

TEST_CASE("model store: manifest layer count and text bytes are bounded") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    // 65 content layers (one blob referenced again and again) exceed the 64-layer cap.
    std::vector<Layer> many(65, model);
    s.put_v2("ollama.com", "library", "bomb", "latest", s.manifest(cfg, many));
    // 64 is still accepted.
    std::vector<Layer> max(64, model);
    s.put_v2("ollama.com", "library", "max", "latest", s.manifest(cfg, max));

    // More than 16 MiB of text layers in one manifest.
    const std::size_t big = (4u << 20) - 64;
    const auto cfg_big = s.config_blob(std::string(R"({"model_format":"gguf","pad":")") + std::string(1u << 20, 'p') + "\"}");
    const auto t = s.blob(kTemplate, std::string(big, 't'));
    const auto sy = s.blob(kSystem, std::string(big, 'y'));
    const auto msg = s.blob("application/vnd.ollama.image.messages", "[\"" + std::string(big, 'm') + "\"]");
    const auto par = s.blob(kParams, "{\"x\":\"" + std::string(big, 'q') + "\"}");
    s.put_v2("ollama.com", "library", "fat", "latest", s.manifest(cfg_big, {model, t, sy, msg, par}));

    // Stop strings: at most 64 entries, each bounded.
    std::string stops = "{\"stop\":[";
    for (int i = 0; i < 70; ++i) stops += (i ? ",\"s" : "\"s") + std::to_string(i) + "\"";
    stops += ",\"" + std::string(5000, 'L') + "\"]}";
    const auto stop_params = s.blob(kParams, stops);
    s.put_v2("ollama.com", "library", "stops", "latest", s.manifest(cfg, {model, stop_params}));

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto bomb = store.value().resolve("bomb");
    REQUIRE(bomb.ok());
    CHECK_FALSE(bomb.value().ready());
    CHECK(bomb.value().reason == m::UnsupportedReason::manifest_invalid);
    CHECK(bomb.value().shard_paths.size() < 64);
    auto mx = store.value().resolve("max");
    REQUIRE(mx.ok());
    CHECK(mx.value().ready());
    auto fat = store.value().resolve("fat");
    REQUIRE(fat.ok());
    CHECK_FALSE(fat.value().ready());
    CHECK(fat.value().reason == m::UnsupportedReason::manifest_invalid);
    CHECK(fat.value().detail.find("past") != std::string::npos);
    auto st = store.value().resolve("stops");
    REQUIRE(st.ok());
    CHECK(st.value().params.stop.size() == 64);
    for (const auto& x : st.value().params.stop) CHECK(x.size() <= 1024);
    CHECK(any_contains(st.value().warnings, "more than 64 entries"));
}

TEST_CASE("model store: declared byte totals cannot wrap") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    // License blobs are only counted, never read, so a huge declared size reaches the sum.
    const Layer lic{kLicense, "sha256:" + std::string(64, 'c'), 18446744073709551600ull, ""};
    s.put_v2("ollama.com", "library", "wrap", "latest", s.manifest(cfg, {model, lic}));
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto r = store.value().resolve("wrap");
    REQUIRE(r.ok());
    CHECK_FALSE(r.value().ready());
    CHECK(r.value().reason == m::UnsupportedReason::manifest_invalid);
    CHECK(r.value().referenced_bytes >= r.value().weights_bytes);
}

TEST_CASE("model store: manifest schema is strict about keys sizes and digests") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 8));
    const std::string good = s.manifest(cfg, {model});
    // Duplicate key: the second "layers" would silently win.
    std::string dup = good;
    dup.insert(dup.size() - 1, ",\"layers\":[]");
    s.put_v2("ollama.com", "library", "dupkey", "latest", dup);
    // Fractional, negative and string sizes.
    auto frac = good;
    const std::string sz = "\"size\":" + std::to_string(model.size);
    frac.replace(frac.find(sz), sz.size(), "\"size\":" + std::to_string(model.size) + ".9");
    s.put_v2("ollama.com", "library", "frac", "latest", frac);
    auto neg = good;
    neg.replace(neg.find(sz), sz.size(), "\"size\":-1");
    s.put_v2("ollama.com", "library", "neg", "latest", neg);
    auto str = good;
    str.replace(str.find(sz), sz.size(), "\"size\":\"" + std::to_string(model.size) + "\"");
    s.put_v2("ollama.com", "library", "strsize", "latest", str);
    // The blob-file spelling of a digest is not a manifest digest.
    auto dash = good;
    const std::string dg = "\"digest\":\"" + model.digest + "\"";
    dash.replace(dash.find(dg), dg.size(), "\"digest\":\"sha256-" + model.digest.substr(7) + "\"");
    s.put_v2("ollama.com", "library", "dash", "latest", dash);

    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    for (const char* name : {"dupkey", "frac", "neg", "strsize", "dash"}) {
        CAPTURE(name);
        auto r = store.value().resolve(name);
        REQUIRE(r.ok());
        CHECK_FALSE(r.value().ready());
        CHECK(r.value().reason == m::UnsupportedReason::manifest_invalid);
    }
}

TEST_CASE("model store: name parts with trailing dots or DOS device names are rejected") {
    for (const char* bad : {"coder.:7b", "coder:7b.", "coder...:7b...", "CON:latest", "con", "x:nul", "lpt1.txt:1",
                            "ns/aux:1", "registry.ollama.ai/nul/x:1", "com7", "Prn.foo", "host./ns/m:t"}) {
        CAPTURE(bad);
        auto r = m::parse_ollama_name(bad);
        CHECK_FALSE(r.ok());
        if (!r.ok()) CHECK(r.status().code() == si::ErrorCode::invalid_argument);
    }
    // Names that merely contain those words stay valid.
    for (const char* ok : {"console:1", "nullx", "com10:1", "lpt:1", "auxiliary/con-model:1", "a.b:c.d"}) {
        CAPTURE(ok);
        CHECK(m::parse_ollama_name(ok).ok());
    }
    Store s;
    make_full_model(s);
    m::ModelRegistry reg;
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    CHECK(reg.resolve("coder...:7b...").status().code() == si::ErrorCode::invalid_argument);
}

#if defined(_WIN32)
TEST_CASE("model store: names with unpaired UTF-16 surrogates are skipped without throwing") {
    Store s;
    make_full_model(s);
    const std::wstring bad_name{L'b', L'a', L'd', static_cast<wchar_t>(0xD800)};
    const fs::path lib = s.tmp.path / "manifests-v2" / "ollama.com" / "library";
    fs::create_directories(lib / fs::path(bad_name));
    write_file(lib / fs::path(bad_name) / "latest", "{}");
    fs::create_directories(s.tmp.path / "manifests-v2" / fs::path(bad_name) / "library" / "m");
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    m::ScanResult scan;
    CHECK_NOTHROW(scan = store.value().scan());
    CHECK(find(scan, "coder:7b") != nullptr);
    CHECK(any_contains(scan.notes, "not representable"));
    CHECK_NOTHROW((void)store.value().resolve("unknown:latest"));
    CHECK_NOTHROW((void)scan_json(s.tmp.path));
}
#endif

TEST_CASE("model store: status messages and notes carry no root path or raw control bytes") {
    Store s;
    make_full_model(s);
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    s.put_legacy("registry.ollama.ai", "library", "bad name", "latest", s.manifest(cfg, {}));
    const auto cloud_cfg =
        s.config_blob(R"({"remote_host":"https://user:hunter2@internal.example:8443/x","remote_model":"kimi\u001b[31m"})");
    s.put_v2("ollama.com", "library", "stub", "cloud", s.manifest(cloud_cfg, {}));
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    const std::string root_text = store.value().root().generic_string();
    const std::string root_native = store.value().root().string();

    auto miss = store.value().resolve("ghost:1");
    REQUIRE_FALSE(miss.ok());
    CHECK(miss.status().message().find(root_text) == std::string::npos);
    CHECK(miss.status().message().find(root_native) == std::string::npos);

    const std::string hostile = "bad\r\n forged-log\x1b[31m" + std::string(10000, 'A');
    auto bad = m::parse_ollama_name(hostile);
    REQUIRE_FALSE(bad.ok());
    const std::string msg = bad.status().message();  // a copy: status() returns by value
    CHECK(msg.size() < 400);
    CHECK(msg.find('\r') == std::string::npos);
    CHECK(msg.find('\n') == std::string::npos);
    CHECK(msg.find('\x1b') == std::string::npos);

    m::ModelRegistry reg;
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    auto stub = reg.resolve("stub:cloud");
    REQUIRE_FALSE(stub.ok());
    CHECK(stub.status().message().find("hunter2") == std::string::npos);
    CHECK(stub.status().message().find("internal.example") == std::string::npos);
    CHECK(stub.status().message().find('\x1b') == std::string::npos);

    auto scan = store.value().scan();
    REQUIRE_FALSE(scan.notes.empty());
    for (const auto& n : scan.notes) {
        CAPTURE(n);
        CHECK(n.find(root_text) == std::string::npos);
        CHECK(n.find(root_native) == std::string::npos);
    }
}

TEST_CASE("model store: weight verification policy first_use caches by file identity") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto good = s.blob(kModel, fake_gguf(1, 1, 2000, 'g'));
    // One blob referenced three times (shards) is opened and hashed once.
    s.put_v2("ollama.com", "library", "good", "latest", s.manifest(cfg, {good, good, good}));
    const auto bad = s.corrupt_blob(kModel, fake_gguf(1, 1, 500, 'A'), fake_gguf(1, 1, 500, 'B'));
    s.put_v2("ollama.com", "library", "bad", "latest", s.manifest(cfg, {bad}));

    int hashed = 0;
    m::StoreOptions first;
    first.on_weight_hash = [&](const m::BlobRef&) { ++hashed; };
    REQUIRE(first.verify_weights == m::WeightVerification::first_use);  // the default
    auto store = m::OllamaStore::open(s.tmp.path, first);
    REQUIRE(store.ok());

    auto scan = store.value().scan();  // first_use: the scan does not hash weights
    CHECK(hashed == 0);
    REQUIRE(find(scan, "good:latest") != nullptr);
    CHECK(find(scan, "good:latest")->ready());
    CHECK_FALSE(find(scan, "good:latest")->weights_verified);

    auto r1 = store.value().resolve("good");
    REQUIRE(r1.ok());
    CHECK(r1.value().ready());
    CHECK(r1.value().weights_verified);
    CHECK(hashed == 1);
    auto r2 = store.value().resolve("good");
    REQUIRE(r2.ok());
    CHECK(r2.value().weights_verified);
    CHECK(hashed == 1);  // cached: same file id, size, write and change time
    // A copy of the store shares the cache.
    const m::OllamaStore copy = store.value();
    CHECK(copy.resolve("good").value().weights_verified);
    CHECK(hashed == 1);

    // A mismatch is cached too, so a corrupt multi-GB file is not re-read on every request.
    CHECK(store.value().resolve("bad").value().reason == m::UnsupportedReason::digest_mismatch);
    CHECK(hashed == 2);
    CHECK(store.value().resolve("bad").value().reason == m::UnsupportedReason::digest_mismatch);
    CHECK(hashed == 2);

    // A changed file is hashed again.
    write_file(s.blob_file(good), fake_gguf(1, 1, 2000, 'h'));
    fs::last_write_time(s.blob_file(good), fs::last_write_time(s.blob_file(good)) + std::chrono::seconds(5));
    auto r3 = store.value().resolve("good");
    REQUIRE(r3.ok());
    CHECK(hashed == 3);
    CHECK(r3.value().reason == m::UnsupportedReason::digest_mismatch);
    CHECK_FALSE(r3.value().weights_verified);

    // revalidate(): a record from before the change is stale.
    CHECK_FALSE(store.value().revalidate(r1.value()));
    auto fresh = store.value().resolve("bad");
    REQUIRE(fresh.ok());
    auto again = fresh.value();
    CHECK(store.value().revalidate(again));
    CHECK(again.reason == m::UnsupportedReason::digest_mismatch);
}

TEST_CASE("model store: weight verification policies always and off") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto good = s.blob(kModel, fake_gguf(1, 1, 300, 'g'));
    s.put_v2("ollama.com", "library", "good", "latest", s.manifest(cfg, {good}));
    const auto bad = s.corrupt_blob(kModel, fake_gguf(1, 1, 300, 'A'), fake_gguf(1, 1, 300, 'B'));
    s.put_v2("ollama.com", "library", "bad", "latest", s.manifest(cfg, {bad}));

    int hashed = 0;
    m::StoreOptions always;
    always.verify_weights = m::WeightVerification::always;
    always.on_weight_hash = [&](const m::BlobRef&) { ++hashed; };
    auto a = m::OllamaStore::open(s.tmp.path, always);
    REQUIRE(a.ok());
    CHECK(a.value().resolve("good").value().weights_verified);
    CHECK(a.value().resolve("good").value().weights_verified);
    CHECK(hashed == 2);  // nothing cached
    auto scan = a.value().scan();
    CHECK(hashed == 4);  // the scan hashes both entries
    CHECK(find(scan, "good:latest")->weights_verified);
    CHECK(find(scan, "bad:latest")->reason == m::UnsupportedReason::digest_mismatch);

    m::StoreOptions off;
    off.verify_weights = m::WeightVerification::off;
    off.on_weight_hash = [&](const m::BlobRef&) { ++hashed; };
    auto o = m::OllamaStore::open(s.tmp.path, off);
    REQUIRE(o.ok());
    auto unverified = o.value().resolve("bad");
    REQUIRE(unverified.ok());
    CHECK(unverified.value().ready());  // size and header only, by request
    CHECK_FALSE(unverified.value().weights_verified);
    CHECK(hashed == 4);

    m::ModelRegistry reg(off);
    REQUIRE(reg.add_ollama_dir(s.tmp.path).ok());
    CHECK(reg.resolve("bad").ok());
    m::ModelRegistry strict;  // default first_use
    REQUIRE(strict.add_ollama_dir(s.tmp.path).ok());
    CHECK(strict.resolve("bad").status().code() == si::ErrorCode::unsupported);
    CHECK(strict.resolve("good").ok());
}

TEST_CASE("model store: a hard-linked weight blob is served only after its digest matches") {
    // Ollama hard-links split-model shards into blobs/.ollama-split-*/ while it runs one
    // (llm/model_split.go), so a weight with two links is normal; its content must still match.
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 64));
    s.put_v2("ollama.com", "library", "split", "latest", s.manifest(cfg, {model}));
    std::error_code ec;
    fs::create_directories(s.tmp.path / "blobs" / ".ollama-split-abc");
    fs::create_hard_link(s.blob_file(model), s.tmp.path / "blobs" / ".ollama-split-abc" / "model-00001-of-00001.gguf", ec);
    REQUIRE_MESSAGE(!ec, "hard link creation failed: " << ec.message());
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto r = store.value().resolve("split");
    REQUIRE(r.ok());
    CHECK(r.value().ready());
    CHECK(r.value().weights_verified);
    CHECK(any_contains(r.value().warnings, "hard links"));
}

TEST_CASE("model store: verify_blobs reads through no link and reports each digest once") {
    Store s;
    const auto cfg = s.config_blob(R"({"model_format":"gguf"})");
    const auto model = s.blob(kModel, fake_gguf(1, 1, 64));
    s.put_v2("ollama.com", "library", "twice", "latest", s.manifest(cfg, {model, model}));
    auto store = m::OllamaStore::open(s.tmp.path);
    REQUIRE(store.ok());
    auto r = store.value().resolve("twice");
    REQUIRE(r.ok());
    REQUIRE(r.value().ready());
    auto report = m::verify_blobs(r.value());
    CHECK(report.ok());
    CHECK(report.blobs.size() == 2);  // config + one model, not two

    // Replace the model blob with a link to an identical file outside the store.
    TempDir outside;
    write_file(outside.path / "copy.gguf", fake_gguf(1, 1, 64));
    fs::remove(s.blob_file(model));
    std::error_code ec;
    fs::create_symlink(outside.path / "copy.gguf", s.blob_file(model), ec);
    if (ec) {
#if defined(_WIN32)
        MESSAGE("file symlinks need a privilege here (" << ec.message() << "); verify link case not exercised");
        return;
#else
        FAIL_CHECK("file symlink creation failed: " << ec.message());
        return;
#endif
    }
    auto linked = m::verify_blobs(r.value());
    CHECK_FALSE(linked.ok());
    bool saw_unsafe = false;
    for (const auto& b : linked.blobs) saw_unsafe = saw_unsafe || b.verdict == m::BlobVerdict::unsafe;
    CHECK(saw_unsafe);
}

TEST_CASE("model store: the JSON document redacts URL credentials in remote_host") {
    Store s;
    const auto cloud_cfg = s.config_blob(R"({"remote_host":"https://user:hunter2@internal.example:8443/x","remote_model":"m1"})");
    s.put_v2("ollama.com", "library", "stub", "cloud", s.manifest(cloud_cfg, {}));
    const std::string doc = scan_json(s.tmp.path);
    CHECK(doc.find("hunter2") == std::string::npos);
    CHECK(doc.find("https://internal.example:8443/x") != std::string::npos);
}
