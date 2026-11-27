// Unit tests for LoRA adapter registry, path/scale/metadata validation,
// per-alias & per-request application, and clean unloading.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lora_registry.hpp"
#include "sonder/inference/backends/llamacpp.hpp"

using namespace sonder::backends::llamacpp;

namespace {

namespace fs = std::filesystem;

std::string PathToUtf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

fs::path PathFromUtf8(std::string_view s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}

class TempDir {
public:
    TempDir() {
        static std::atomic<std::uint64_t> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
        root_ = fs::temp_directory_path() /
                ("sonder_lora_test_" + std::to_string(stamp) + "_" + std::to_string(id));
        fs::create_directories(root_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    [[nodiscard]] const fs::path& path() const noexcept { return root_; }
    [[nodiscard]] std::string utf8() const { return PathToUtf8(root_); }

    fs::path CreateFile(const fs::path& rel, std::string_view content = "fake-lora") const {
        const fs::path full = root_ / rel;
        fs::create_directories(full.parent_path());
        std::ofstream ofs(full, std::ios::binary);
        ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
        ofs.close();
        return full;
    }

private:
    fs::path root_;
};

class FakeLoraRuntimeOps final : public LoraRuntimeOps {
public:
    LoraModelMetadata base_meta{"base-qwen.gguf", "qwen2", "Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct"};
    std::unordered_map<std::string, LoraAdapterMetadata> pre_meta_by_path;
    std::unordered_map<std::string, LoraAdapterMetadata> post_meta_by_path;
    std::unordered_set<std::string> fail_init_paths;
    bool fail_set_context = false;

    std::vector<std::string> events;
    std::unordered_map<LoraAdapterHandle, std::string> live_handles;
    std::vector<std::pair<std::string, float>> active_on_ctx;
    int init_calls = 0;
    int init_successes = 0;
    int free_calls = 0;
    int set_ctx_calls = 0;
    std::unordered_set<std::string> fail_post_meta_paths;

    Status ReadBaseModelMetadata(LoraModelMetadata& out) const override {
        out = base_meta;
        return Status::Ok();
    }

    Status ReadAdapterMetadata(const std::string& canonical_path, LoraAdapterMetadata& out) const override {
        const auto it = pre_meta_by_path.find(canonical_path);
        if (it == pre_meta_by_path.end()) {
            return Status::Error(ErrorCode::kLoadFailed, "no fake metadata for: " + canonical_path);
        }
        out = it->second;
        return Status::Ok();
    }

    Status InitAdapter(const std::string& canonical_path, LoraAdapterHandle& out_handle) override {
        ++init_calls;
        events.push_back("init:" + canonical_path);
        if (fail_init_paths.count(canonical_path) != 0) {
            out_handle = nullptr;
            return Status::Error(ErrorCode::kLoadFailed, "fake llama_adapter_lora_init failed: " + canonical_path);
        }
        const std::uintptr_t id = next_handle_id_++;
        out_handle = reinterpret_cast<LoraAdapterHandle>(id);
        live_handles[out_handle] = canonical_path;
        ++init_successes;
        return Status::Ok();
    }

    Status ReadLoadedAdapterMetadata(LoraAdapterHandle handle, LoraAdapterMetadata& out) const override {
        const auto hit = live_handles.find(handle);
        if (hit == live_handles.end()) {
            return Status::Error(ErrorCode::kInvalidArgument, "unknown adapter handle");
        }
        if (fail_post_meta_paths.count(hit->second) != 0) {
            return Status::Error(ErrorCode::kLoadFailed, "fake ReadLoadedAdapterMetadata failure");
        }
        if (const auto pit = post_meta_by_path.find(hit->second); pit != post_meta_by_path.end()) {
            out = pit->second;
            return Status::Ok();
        }
        return ReadAdapterMetadata(hit->second, out);
    }

    void FreeAdapter(LoraAdapterHandle handle) noexcept override {
        ++free_calls;
        const auto it = live_handles.find(handle);
        if (it != live_handles.end()) {
            events.push_back("free:" + it->second);
            live_handles.erase(it);
        } else {
            events.push_back("free:UNKNOWN");
        }
    }

    Status SetContextAdapters(std::span<const ActiveLoraBinding> bindings) override {
        ++set_ctx_calls;
        if (bindings.empty()) {
            events.push_back("set_ctx:clear");
        } else {
            events.push_back("set_ctx:" + std::to_string(bindings.size()));
        }
        if (fail_set_context) {
            return Status::Error(ErrorCode::kDecodeFailed, "fake llama_set_adapters_lora failed");
        }
        // Verify every handle passed to context is live (no dangling pointers!).
        std::vector<std::pair<std::string, float>> next;
        next.reserve(bindings.size());
        for (const auto& b : bindings) {
            const auto it = live_handles.find(b.handle);
            if (it == live_handles.end()) {
                return Status::Error(ErrorCode::kInvalidArgument, "dangling adapter handle passed to context");
            }
            next.emplace_back(it->second, b.scale);
        }
        active_on_ctx = std::move(next);
        return Status::Ok();
    }

private:
    std::uintptr_t next_handle_id_ = 0x1000;
};

LoraAdapterMetadata MakeValidMeta(std::string arch = "qwen2", std::string base_name = "Qwen2.5-7B-Instruct") {
    LoraAdapterMetadata m;
    m.general_type = "adapter";
    m.adapter_type = "lora";
    m.architecture = std::move(arch);
    m.base_model_name = std::move(base_name);
    m.alpha = 16.0F;
    return m;
}

}  // namespace

TEST_CASE("lora: scale validation accepts bounded finite values and rejects invalid ones") {
    CHECK(ValidateLoraScale(0.0F).ok());
    CHECK(ValidateLoraScale(0.5F).ok());
    CHECK(ValidateLoraScale(1.0F).ok());
    CHECK(ValidateLoraScale(kMaxLoraScale).ok());

    CHECK(ValidateLoraScale(-0.01F).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraScale(-1.0F).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraScale(kMaxLoraScale + 0.01F).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraScale(100.0F).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraScale(std::numeric_limits<float>::quiet_NaN()).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraScale(std::numeric_limits<float>::infinity()).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraScale(-std::numeric_limits<float>::infinity()).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("lora: path validation enforces existence, regular file, root containment, and no '..' escapes") {
    TempDir tmp;
    const fs::path root1 = tmp.path() / "models";
    const fs::path root2 = tmp.path() / "extra_models";
    const fs::path sibling_prefix = tmp.path() / "models_evil";
    const fs::path outside = tmp.path() / "outside";
    fs::create_directories(root1);
    fs::create_directories(root2);
    fs::create_directories(sibling_prefix);
    fs::create_directories(outside);

    const fs::path good1 = tmp.CreateFile("models/adapters/coder.gguf");
    const fs::path good2 = tmp.CreateFile("extra_models/math.gguf");
    const fs::path evil_sibling = tmp.CreateFile("models_evil/bad.gguf");
    const fs::path outside_file = tmp.CreateFile("outside/escape.gguf");

    const std::vector<std::string> roots = {PathToUtf8(root1), PathToUtf8(root2)};

    // Valid relative path under root1.
    std::string canonical;
    REQUIRE(ValidateLoraAdapterPath("adapters/coder.gguf", roots, canonical).ok());
    CHECK(canonical == PathToUtf8(fs::canonical(good1)));

    // Valid relative path under root2.
    REQUIRE(ValidateLoraAdapterPath("math.gguf", roots, canonical).ok());
    CHECK(canonical == PathToUtf8(fs::canonical(good2)));

    // Valid absolute path inside root1.
    REQUIRE(ValidateLoraAdapterPath(PathToUtf8(good1), roots, canonical).ok());
    CHECK(canonical == PathToUtf8(fs::canonical(good1)));

    // Empty and embedded NUL rejected.
    CHECK(ValidateLoraAdapterPath("", roots, canonical).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath(std::string_view("a\0b.gguf", 8), roots, canonical).code ==
          ErrorCode::kInvalidArgument);

    // Malformed UTF-8 bytes fail closed without throwing std::system_error.
    const std::string bad_utf8("\xFF\xFE\xFD\x80.gguf", 9);
    const std::string overlong_utf8("\xC0\xAF.gguf", 7);
    const std::string surrogate_utf8("\xED\xA0\x80.gguf", 8);
    CHECK(ValidateLoraPathLexical(bad_utf8).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraPathLexical(overlong_utf8).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraPathLexical(surrogate_utf8).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath(bad_utf8, roots, canonical).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath("adapters/coder.gguf", {bad_utf8, PathToUtf8(root1)}, canonical).ok());
    CHECK(canonical == PathToUtf8(fs::canonical(good1)));
    CHECK(ValidateLoraAdapterPath("adapters/coder.gguf", {bad_utf8}, canonical).code == ErrorCode::kInvalidArgument);
    CHECK_FALSE(BaseModelMatches(bad_utf8, "ok.gguf"));
    CHECK_FALSE(BaseModelMatches("ok.gguf", bad_utf8));
    CHECK(ValidateLoraAliasSyntax({bad_utf8, "base.gguf", {{"adapters/coder.gguf", 1.0F}}}).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAliasSyntax({"ok", bad_utf8, {{"adapters/coder.gguf", 1.0F}}}).code ==
          ErrorCode::kInvalidArgument);

    // No model roots configured fails closed.
    CHECK(ValidateLoraAdapterPath(PathToUtf8(good1), {}, canonical).code == ErrorCode::kInvalidArgument);

    // Non-existent file rejected with kFileNotFound.
    CHECK(ValidateLoraAdapterPath("adapters/missing.gguf", roots, canonical).code == ErrorCode::kFileNotFound);

    // Directory is not a regular file.
    CHECK(ValidateLoraAdapterPath("adapters", roots, canonical).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath(PathToUtf8(root1 / "adapters"), roots, canonical).code ==
          ErrorCode::kInvalidArgument);

    // Lexical '..' traversal is ALWAYS rejected (even if normalized target stays inside root).
    CHECK(ValidateLoraAdapterPath("../outside/escape.gguf", roots, canonical).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath("adapters/../adapters/coder.gguf", roots, canonical).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath("adapters\\..\\..\\outside\\escape.gguf", roots, canonical).code ==
          ErrorCode::kInvalidArgument);

    // Absolute path outside configured roots rejected.
    CHECK(ValidateLoraAdapterPath(PathToUtf8(outside_file), roots, canonical).code == ErrorCode::kInvalidArgument);

    // Sibling prefix-collision directory ('models_evil' vs 'models') must NOT match 'models'.
    CHECK(ValidateLoraAdapterPath(PathToUtf8(evil_sibling), roots, canonical).code == ErrorCode::kInvalidArgument);

#if defined(_WIN32)
    // Drive-relative path and NTFS alternate data stream syntax must be rejected.
    CHECK(ValidateLoraAdapterPath("C:coder.gguf", roots, canonical).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLoraAdapterPath("adapters/coder.gguf:hidden", roots, canonical).code ==
          ErrorCode::kInvalidArgument);
#endif

    // Symlink inside root1 pointing outside all roots must be refused (when platform permits symlinks).
    std::error_code sym_ec;
    const fs::path escape_link = root1 / "link_escape.gguf";
    fs::create_symlink(outside_file, escape_link, sym_ec);
    if (!sym_ec) {
        CHECK(ValidateLoraAdapterPath("link_escape.gguf", roots, canonical).code == ErrorCode::kInvalidArgument);
        CHECK(ValidateLoraAdapterPath(PathToUtf8(escape_link), roots, canonical).code == ErrorCode::kInvalidArgument);
    }
}

TEST_CASE("lora: registry alias registration, resolution, listing, and error checks") {
    TempDir tmp;
    const fs::path root = tmp.path() / "models";
    const fs::path a1 = tmp.CreateFile("models/lora_a.gguf");
    const fs::path a2 = tmp.CreateFile("models/lora_b.gguf");
    const fs::path outside = tmp.CreateFile("outside.gguf");

    LoraRegistry reg({PathToUtf8(root)});

    LoraAliasEntry entry;
    entry.alias = "sonder-coder:v1";
    entry.base_model = "qwen2.5-7b.gguf";
    entry.adapters = {{"lora_a.gguf", 0.75F}, {PathToUtf8(a2), 0.25F}};
    REQUIRE(reg.RegisterAlias(entry).ok());
    CHECK(reg.HasAlias("sonder-coder:v1"));

    // Duplicate alias rejected.
    CHECK(reg.RegisterAlias(entry).code == ErrorCode::kInvalidArgument);

    // Resolve returns canonical adapter paths and scales.
    LoraAliasEntry resolved;
    REQUIRE(reg.ResolveAlias("sonder-coder:v1", resolved).ok());
    CHECK(resolved.alias == "sonder-coder:v1");
    CHECK(resolved.base_model == "qwen2.5-7b.gguf");
    REQUIRE(resolved.adapters.size() == 2u);
    CHECK(resolved.adapters[0].path == PathToUtf8(fs::canonical(a1)));
    CHECK(resolved.adapters[0].scale == doctest::Approx(0.75F));
    CHECK(resolved.adapters[1].path == PathToUtf8(fs::canonical(a2)));
    CHECK(resolved.adapters[1].scale == doctest::Approx(0.25F));

    // Unknown alias fails cleanly.
    CHECK(reg.ResolveAlias("no-such-alias", resolved).code == ErrorCode::kFileNotFound);

    // List aliases returns sorted entries.
    LoraAliasEntry second;
    second.alias = "alpha-lora";
    second.base_model = "qwen2.5-7b.gguf";
    second.adapters = {{"lora_a.gguf", 1.0F}};
    REQUIRE(reg.RegisterAlias(second).ok());
    const auto all = reg.ListAliases();
    REQUIRE(all.size() == 2u);
    CHECK(all[0].alias == "alpha-lora");
    CHECK(all[1].alias == "sonder-coder:v1");

    // Unregister removes cleanly.
    CHECK(reg.UnregisterAlias("alpha-lora"));
    CHECK_FALSE(reg.UnregisterAlias("alpha-lora"));
    CHECK_FALSE(reg.HasAlias("alpha-lora"));

    // Bad registrations rejected: empty alias, empty base, empty adapters, duplicate path, bad scale, outside path.
    CHECK(reg.RegisterAlias({"", "base.gguf", {{"lora_a.gguf", 1.0F}}}).code == ErrorCode::kInvalidArgument);
    CHECK(reg.RegisterAlias({"x", "", {{"lora_a.gguf", 1.0F}}}).code == ErrorCode::kInvalidArgument);
    CHECK(reg.RegisterAlias({"x", "base.gguf", {}}).code == ErrorCode::kInvalidArgument);
    CHECK(reg.RegisterAlias({"dup", "base.gguf", {{"lora_a.gguf", 0.5F}, {PathToUtf8(a1), 0.5F}}}).code ==
          ErrorCode::kInvalidArgument);
    CHECK(reg.RegisterAlias({"badscale", "base.gguf", {{"lora_a.gguf", -0.5F}}}).code == ErrorCode::kInvalidArgument);
    CHECK(reg.RegisterAlias({"escape", "base.gguf", {{PathToUtf8(outside), 1.0F}}}).code ==
          ErrorCode::kInvalidArgument);
}

TEST_CASE("lora: compatibility check refuses architecture, type, and base-model mismatches") {
    LoraModelMetadata base;
    base.model_id = "qwen2.5-7b.gguf";
    base.architecture = "qwen2";
    base.name = "Qwen2.5-7B-Instruct";
    base.basename = "Qwen2.5-7B";
    base.base_model_name = "Qwen2.5-7B-Instruct";

    CHECK(ValidateLoraCompatibility(base, MakeValidMeta("qwen2", "Qwen2.5-7B-Instruct")).ok());
    // Case-insensitive base model name match works.
    CHECK(ValidateLoraCompatibility(base, MakeValidMeta("qwen2", "qwen2.5-7b-instruct")).ok());
    // Missing optional base_model_name on adapter is allowed when architecture matches.
    CHECK(ValidateLoraCompatibility(base, MakeValidMeta("qwen2", "")).ok());

    // Wrong general.architecture refused.
    auto st = ValidateLoraCompatibility(base, MakeValidMeta("llama", ""));
    CHECK(st.code == ErrorCode::kInvalidArgument);
    CHECK(st.message.find("architecture") != std::string::npos);

    // Missing general.architecture on adapter or base refused.
    CHECK(ValidateLoraCompatibility(base, MakeValidMeta("", "")).code == ErrorCode::kInvalidArgument);
    LoraModelMetadata no_arch = base;
    no_arch.architecture.clear();
    CHECK(ValidateLoraCompatibility(no_arch, MakeValidMeta("qwen2", "")).code == ErrorCode::kInvalidArgument);

    // Wrong general.type refused.
    auto wrong_general = MakeValidMeta("qwen2", "");
    wrong_general.general_type = "model";
    CHECK(ValidateLoraCompatibility(base, wrong_general).code == ErrorCode::kInvalidArgument);

    // Wrong adapter.type refused.
    auto wrong_adapter = MakeValidMeta("qwen2", "");
    wrong_adapter.adapter_type = "control_vector";
    CHECK(ValidateLoraCompatibility(base, wrong_adapter).code == ErrorCode::kInvalidArgument);

    // Same architecture, incompatible explicit base_model.0.name refused.
    st = ValidateLoraCompatibility(base, MakeValidMeta("qwen2", "Qwen2.5-14B-Instruct"));
    CHECK(st.code == ErrorCode::kInvalidArgument);
    CHECK(st.message.find("base model") != std::string::npos);

    // Invalid alpha refused.
    auto bad_alpha = MakeValidMeta("qwen2", "");
    bad_alpha.alpha = -1.0F;
    CHECK(ValidateLoraCompatibility(base, bad_alpha).code == ErrorCode::kInvalidArgument);
    bad_alpha.alpha = std::numeric_limits<float>::quiet_NaN();
    CHECK(ValidateLoraCompatibility(base, bad_alpha).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("lora: adapter manager applies per-alias and per-request, caches, refuses mismatches, and unloads cleanly") {
    TempDir tmp;
    const fs::path root = tmp.path() / "models";
    const fs::path base_file = tmp.CreateFile("models/base-qwen.gguf");
    const fs::path a1 = tmp.CreateFile("models/a1.gguf");
    const fs::path a2 = tmp.CreateFile("models/a2.gguf");
    const fs::path wrong_arch = tmp.CreateFile("models/wrong_arch.gguf");
    const fs::path post_mismatch = tmp.CreateFile("models/post_mismatch.gguf");

    const std::string c_base = PathToUtf8(fs::canonical(base_file));
    const std::string c_a1 = PathToUtf8(fs::canonical(a1));
    const std::string c_a2 = PathToUtf8(fs::canonical(a2));
    const std::string c_wrong = PathToUtf8(fs::canonical(wrong_arch));
    const std::string c_post = PathToUtf8(fs::canonical(post_mismatch));

    FakeLoraRuntimeOps ops;
    ops.base_meta.model_id = c_base;
    ops.pre_meta_by_path[c_a1] = MakeValidMeta("qwen2", "Qwen2.5-7B-Instruct");
    ops.pre_meta_by_path[c_a2] = MakeValidMeta("qwen2", "Qwen2.5-7B-Instruct");
    ops.pre_meta_by_path[c_wrong] = MakeValidMeta("llama", "Llama-3-8B");
    ops.pre_meta_by_path[c_post] = MakeValidMeta("qwen2", "Qwen2.5-7B-Instruct");
    ops.post_meta_by_path[c_post] = MakeValidMeta("mistral", "Mistral-7B");

    LoraRegistry reg({PathToUtf8(root)});
    REQUIRE(reg.RegisterAlias({"coder", "base-qwen.gguf", {{"a1.gguf", 0.8F}}}).ok());
    REQUIRE(reg.RegisterAlias({"other-base", "different-model.gguf", {{"a1.gguf", 1.0F}}}).ok());

    {
        LoraAdapterManager mgr(&ops, {PathToUtf8(root)});

        // Alias targeting a different base model is refused before touching llama.cpp.
        CHECK(mgr.ApplyAlias(reg, "other-base").code == ErrorCode::kInvalidArgument);
        CHECK(ops.init_calls == 0);
        CHECK(mgr.LoadedCount() == 0u);

        // Apply alias loads adapter once and sets it on context.
        REQUIRE(mgr.ApplyAlias(reg, "coder").ok());
        CHECK(ops.init_calls == 1);
        CHECK(mgr.LoadedCount() == 1u);
        CHECK(mgr.IsAdapterLoaded(c_a1));
        REQUIRE(ops.active_on_ctx.size() == 1u);
        CHECK(ops.active_on_ctx[0].first == c_a1);
        CHECK(ops.active_on_ctx[0].second == doctest::Approx(0.8F));

        // Per-request scoped override applies a1 + a2 with new scales, then restores coder (a1 @ 0.8).
        {
            Status scope_st;
            ScopedLoraApplication scoped(mgr, {{"a1.gguf", 0.4F}, {"a2.gguf", 0.6F}}, scope_st);
            REQUIRE(scope_st.ok());
            CHECK(scoped.applied());
            CHECK(ops.init_calls == 2);  // a1 reused from cache, only a2 newly initialized
            REQUIRE(ops.active_on_ctx.size() == 2u);
            CHECK(ops.active_on_ctx[0].first == c_a1);
            CHECK(ops.active_on_ctx[0].second == doctest::Approx(0.4F));
            CHECK(ops.active_on_ctx[1].first == c_a2);
            CHECK(ops.active_on_ctx[1].second == doctest::Approx(0.6F));
        }
        // Restored back to coder alias after scope exit.
        REQUIRE(ops.active_on_ctx.size() == 1u);
        CHECK(ops.active_on_ctx[0].first == c_a1);
        CHECK(ops.active_on_ctx[0].second == doctest::Approx(0.8F));

        // Mismatched adapter in multi-adapter request fails atomically without altering active set or leaking.
        const int init_before = ops.init_calls;
        CHECK(mgr.ApplyAdapters({{"a1.gguf", 0.5F}, {"wrong_arch.gguf", 1.0F}}).code == ErrorCode::kInvalidArgument);
        CHECK(ops.init_calls == init_before);  // pre-check refused before llama_adapter_lora_init
        REQUIRE(ops.active_on_ctx.size() == 1u);
        CHECK(ops.active_on_ctx[0].second == doctest::Approx(0.8F));

        // Post-init metadata mismatch frees newly initialized handle cleanly (zero leaks).
        const std::size_t live_before = ops.live_handles.size();
        CHECK(mgr.ApplyAdapters({{"post_mismatch.gguf", 1.0F}}).code == ErrorCode::kInvalidArgument);
        CHECK(ops.live_handles.size() == live_before);
        CHECK_FALSE(mgr.IsAdapterLoaded(c_post));

        // UnloadAdapter removes active adapter from context BEFORE freeing handle.
        ops.events.clear();
        REQUIRE(mgr.UnloadAdapter("a1.gguf").ok());
        CHECK_FALSE(mgr.IsAdapterLoaded(c_a1));
        CHECK(ops.active_on_ctx.empty());
        REQUIRE(ops.events.size() >= 2u);
        CHECK(ops.events[0] == "set_ctx:clear");
        CHECK(ops.events[1] == "free:" + c_a1);

        // ClearActive clears context while keeping cached handles loaded.
        REQUIRE(mgr.ApplyAdapters({{"a2.gguf", 0.9F}}).ok());
        CHECK(ops.active_on_ctx.size() == 1u);
        REQUIRE(mgr.ClearActive().ok());
        CHECK(ops.active_on_ctx.empty());
        CHECK(mgr.IsAdapterLoaded(c_a2));

        // InitAdapter failure on adapter #2 rolls back newly initialized adapter #1 cleanly.
        const fs::path a3 = tmp.CreateFile("models/a3.gguf");
        const fs::path a4 = tmp.CreateFile("models/a4.gguf");
        const std::string c_a3 = PathToUtf8(fs::canonical(a3));
        const std::string c_a4 = PathToUtf8(fs::canonical(a4));
        ops.pre_meta_by_path[c_a3] = MakeValidMeta("qwen2", "Qwen2.5-7B-Instruct");
        ops.pre_meta_by_path[c_a4] = MakeValidMeta("qwen2", "Qwen2.5-7B-Instruct");
        ops.fail_init_paths.insert(c_a4);
        const std::size_t live_pre_fail = ops.live_handles.size();
        CHECK(mgr.ApplyAdapters({{"a3.gguf", 0.5F}, {"a4.gguf", 0.5F}}).code == ErrorCode::kLoadFailed);
        CHECK(ops.live_handles.size() == live_pre_fail);
        CHECK_FALSE(mgr.IsAdapterLoaded(c_a3));
        CHECK_FALSE(mgr.IsAdapterLoaded(c_a4));
        ops.fail_init_paths.erase(c_a4);

        // SetContextAdapters failure rolls back newly initialized handles and keeps active set intact.
        REQUIRE(mgr.ApplyAdapters({{"a2.gguf", 0.7F}}).ok());
        ops.fail_set_context = true;
        CHECK(mgr.ApplyAdapters({{"a3.gguf", 0.5F}}).code == ErrorCode::kDecodeFailed);
        CHECK_FALSE(mgr.IsAdapterLoaded(c_a3));
        REQUIRE(mgr.ActiveAdapters().size() == 1u);
        CHECK(mgr.ActiveAdapters()[0].path == c_a2);
        CHECK(mgr.ActiveAdapters()[0].scale == doctest::Approx(0.7F));
        ops.fail_set_context = false;

        // ReadLoadedAdapterMetadata error fails closed and frees newly initialized handle.
        ops.fail_post_meta_paths.insert(c_a3);
        const std::size_t live_before_post_err = ops.live_handles.size();
        CHECK(mgr.ApplyAdapters({{"a3.gguf", 0.5F}}).code == ErrorCode::kLoadFailed);
        CHECK(ops.live_handles.size() == live_before_post_err);
        CHECK_FALSE(mgr.IsAdapterLoaded(c_a3));
        ops.fail_post_meta_paths.erase(c_a3);

        // Exceeding kMaxLoraAdaptersPerRequest is rejected cleanly.
        std::vector<LoraAdapterSpec> too_many(kMaxLoraAdaptersPerRequest + 1, LoraAdapterSpec{"a2.gguf", 0.1F});
        CHECK(mgr.ApplyAdapters(too_many).code == ErrorCode::kInvalidArgument);
    }
    // Destructor must have cleared context and freed all remaining loaded adapters (zero leaks).
    CHECK(ops.live_handles.empty());
    CHECK(ops.free_calls == ops.init_successes);
    CHECK(ops.init_calls == ops.init_successes + 1);  // 1 failed InitAdapter for a4.gguf
}

TEST_CASE("lora: core backend options validation, capability advertisement, and real GGUF metadata guard") {
    namespace si = sonder::inference;

    // Capability::lora is advertised on LlamaCppCoreBackend.
    auto be = si::make_llamacpp_backend();
    REQUIRE(be != nullptr);
    CHECK(be->capabilities().has(si::Capability::lora));
    bool found_lora_cap = false;
    for (const auto& [cap_name, st] : be->capability_statuses()) {
        if (cap_name == "lora") {
            found_lora_cap = true;
            CHECK(st.adapter_supports == si::CapabilityState::yes);
            CHECK(st.build_supports == si::CapabilityState::yes);
            CHECK(st.configured == si::CapabilityState::no);
        }
    }
    CHECK(found_lora_cap);

    // Valid LoRA alias option passes pure startup validation and shows up in list_models().
    TempDir tmp;
    const fs::path root = tmp.path() / "models";
    const fs::path outside_adapter = tmp.CreateFile("outside.gguf");
    fs::create_directories(root);
    si::LlamaCppBackendOptions opts;
    opts.model_dirs = {PathToUtf8(root)};
    opts.lora_aliases.push_back(
        si::LlamaCppLoraAliasOption{"sonder-coder", "qwen2.5-7b.gguf", {{"adapters/coder.gguf", 0.8f}}});
    CHECK(si::validate_llamacpp_options(opts).ok());

    auto be_with_lora = si::make_llamacpp_backend(opts);
    const auto models = be_with_lora->list_models();
    REQUIRE(models.ok());
    bool found_alias = false;
    for (const auto& m : models.value()) {
        if (m.name == "sonder-coder") {
            found_alias = true;
            CHECK(m.format == "gguf");
        }
    }
    CHECK(found_alias);

    // Loading an alias whose adapter escapes configured roots fails fast with invalid_argument.
    si::LlamaCppBackendOptions escape_opts;
    escape_opts.model_dirs = {PathToUtf8(root)};
    escape_opts.lora_aliases.push_back(
        si::LlamaCppLoraAliasOption{"bad-alias", "missing-base.gguf", {{PathToUtf8(outside_adapter), 1.0f}}});
    auto be_escape = si::make_llamacpp_backend(escape_opts);
    si::ModelLoadOptions ml;
    ml.model = "bad-alias";
    CHECK(be_escape->load_model(ml).status().code() == si::ErrorCode::invalid_argument);

    // Invalid scale and '..' traversal fail startup validation immediately.
    si::LlamaCppBackendOptions bad_scale = opts;
    bad_scale.lora_aliases[0].adapters[0].scale = -1.0f;
    CHECK(si::validate_llamacpp_options(bad_scale).code() == si::ErrorCode::invalid_argument);

    si::LlamaCppBackendOptions bad_path = opts;
    bad_path.lora_aliases[0].adapters[0].path = "../escape.gguf";
    CHECK(si::validate_llamacpp_options(bad_path).code() == si::ErrorCode::invalid_argument);

    // Unloaded LlamaCppBackend rejects ApplyLoraAdapters cleanly.
    LlamaCppBackend raw_be;
    CHECK(ApplyLoraAdapters(raw_be, {{"a.gguf", 1.0F}}, {}).code == ErrorCode::kNotLoaded);
    CHECK(ClearLoraAdapters(raw_be).code == ErrorCode::kNotLoaded);
    UnloadAllLoraAdapters(raw_be);
    CHECK(ActiveLoraAdapters(raw_be).empty());

    // Vocab-only load with bundled non-adapter GGUF exercises real GGUF metadata check refusal
    // and verifies move-constructed LlamaCppBackend preserves LoRA hook registration cleanly.
    LoadOptions lo;
    lo.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    lo.vocab_only = true;
    REQUIRE(raw_be.Load(lo).ok());
    LlamaCppBackend moved_be(std::move(raw_be));
    const Status non_adapter_st =
        ApplyLoraAdapters(moved_be, {{"ggml-vocab-llama-spm.gguf", 1.0F}}, {SONDER_LLAMACPP_VOCAB_DIR});
    CHECK_FALSE(non_adapter_st.ok());
    CHECK(non_adapter_st.code == ErrorCode::kInvalidArgument);
    moved_be.Unload();
}
