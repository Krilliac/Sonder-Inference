// Tests for the Ollama model store reader. Every test builds a synthetic
// manifests/blobs tree (tiny fake GGUF headers) in a temp directory; nothing
// here needs Ollama, a network or model weights.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
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
