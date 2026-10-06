#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>

#include "src/gguf_artifact_observation.hpp"
#include "src/sha256.hpp"

namespace si = sonder::inference;
namespace sd = sonder::inference::server::detail;
namespace {

void u64(std::string& out, std::uint64_t n, unsigned bytes = 8) {
    for (unsigned i = 0; i < bytes; ++i) out.push_back(static_cast<char>(n >> (8 * i)));
}
void str(std::string& out, std::string_view value) { u64(out, value.size()); out += value; }
void integer(std::string& out, std::string_view key, std::uint32_t value) {
    str(out, key); u64(out, 4, 4); u64(out, value, 4);
}
std::string header() {
    std::string out = "GGUF";
    u64(out, 3, 4); u64(out, 1); u64(out, 4);
    str(out, "general.architecture"); u64(out, 8, 4); str(out, "qwen3.8");
    integer(out, "qwen3.8.block_count", 2);
    integer(out, "qwen3.8.expert_count", 64);
    integer(out, "qwen3.8.expert_used_count", 4);
    str(out, "blk.0.ffn.weight"); u64(out, 1, 4); u64(out, 2); u64(out, 0, 4); u64(out, 32);
    return out;
}
std::string artifact(char revision = 'A') {
    auto out = header(); out.append(17, '\0'); out += "payload-";
    out.push_back(revision); out.push_back(static_cast<char>(0xff)); return out;
}
// Independently generated with Python hashlib over the little-endian fixture;
// these expectations never use the implementation being qualified.
constexpr auto kA = "fea7a52b506970b6a4c0c0e9d98050be6e13c42d03ffa8687aa14ca2a4bdec38";
constexpr auto kB = "893ed756220ce4eb5e8322f996b79506993afe18f566d7b08b9a5de0382d2ada";
constexpr auto kHeader = "891a2b75e87aeae948fc9f14267663377644caa6d7f26fb6281190ba70ecd2be";

sd::GgufObservationOptions options() {
    return {1024 * 1024, 4096, std::chrono::steady_clock::now() + std::chrono::seconds(10), 65536};
}

// No get area: all consumed bytes go through xsgetn. Short transfers are
// deliberate, seek attempts are counted, EOF/throw/no-progress are distinct.
class Source : public std::streambuf {
public:
    explicit Source(std::string bytes) : bytes_(std::move(bytes)) {}
    std::size_t position = 0;
    std::size_t chunk = 65536;
    std::size_t max_request = 0;
    unsigned seeks = 0;
    bool zero_progress = false;
    bool throw_read = false;
    bool throw_eof = false;
    std::function<void()> on_read;
    std::function<void()> on_eof;
protected:
    std::streamsize xsgetn(char* out, std::streamsize n) override {
        max_request = std::max(max_request, static_cast<std::size_t>(n));
        if (throw_read) throw std::runtime_error("private-source-secret");
        if (zero_progress) return 0;
        const auto count = std::min({static_cast<std::size_t>(n), chunk, bytes_.size() - position});
        std::memcpy(out, bytes_.data() + position, count); position += count;
        if (on_read) on_read();
        return static_cast<std::streamsize>(count);
    }
    int_type underflow() override {
        if (position < bytes_.size()) return traits_type::to_int_type(bytes_[position]);
        if (on_eof) on_eof();
        if (throw_eof) throw std::runtime_error("private-EOF-secret");
        return traits_type::eof();
    }
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
        ++seeks; return pos_type(off_type(-1));
    }
    pos_type seekpos(pos_type, std::ios_base::openmode) override { ++seeks; return pos_type(off_type(-1)); }
private:
    std::string bytes_;
};

void error(const si::Result<sd::GgufArtifactObservation>& result, si::ErrorCode code) {
    REQUIRE_FALSE(result.ok()); CHECK(result.status().code() == code);
    CHECK(result.status().message().find("private") == std::string::npos);
    CHECK(result.status().message().find(kA) == std::string::npos);
}

}  // namespace

TEST_CASE("GGUF artifact observation binds same-stream header and full tail") {
    for (const auto chunk : {1u, 7u, 65536u}) {
        Source source(artifact()); source.chunk = 3;
        std::istream in(&source); in.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
        auto opt = options(); opt.read_chunk_bytes = chunk;
        const auto result = sd::observe_gguf_artifact(in, kA, opt);
        REQUIRE(result.ok()); CHECK(result->sha256 == kA); CHECK(result->byte_length == 258);
        CHECK(result->architecture == "qwen3.8"); REQUIRE(result->routed_experts.has_value());
        CHECK(result->routed_experts->total == 64); CHECK(result->routed_experts->active == 4);
        CHECK(source.position == 258); CHECK(source.seeks == 0); CHECK(source.max_request <= chunk);
        CHECK(in.exceptions() == (std::ios::failbit | std::ios::badbit | std::ios::eofbit));
    }
}

TEST_CASE("GGUF artifact observation distinguishes revisions with identical headers") {
    for (const auto revision : {'A', 'B'}) {
        std::istringstream in(artifact(revision));
        auto result = sd::observe_gguf_artifact(in, revision == 'A' ? kA : kB, options());
        REQUIRE(result.ok()); CHECK(result->architecture == "qwen3.8");
        std::istringstream wrong(artifact(revision));
        error(sd::observe_gguf_artifact(wrong, revision == 'A' ? kB : kA, options()), si::ErrorCode::invalid_argument);
    }
}

TEST_CASE("GGUF artifact observation rejects original digest after tail truncation") {
    auto bytes = artifact(); bytes.pop_back(); std::istringstream in(bytes);
    error(sd::observe_gguf_artifact(in, kA, options()), si::ErrorCode::invalid_argument);
    // A matching header-only digest binds inspected content, never loadability.
    std::istringstream only_header(header());
    REQUIRE(sd::observe_gguf_artifact(only_header, kHeader, options()).ok());
}

TEST_CASE("GGUF artifact observation enforces strict full and inclusive header budgets") {
    for (const auto cap : {257u, 258u, 259u}) {
        Source source(artifact()); std::istream in(&source); auto opt = options(); opt.max_read_bytes = cap;
        opt.max_header_bytes = 231;
        auto result = sd::observe_gguf_artifact(in, kA, opt);
        if (cap == 259) REQUIRE(result.ok()); else error(result, si::ErrorCode::invalid_argument);
        CHECK(source.position <= cap);
    }
    Source source(artifact()); std::istream in(&source); auto opt = options(); opt.max_header_bytes = 230;
    error(sd::observe_gguf_artifact(in, kA, opt), si::ErrorCode::invalid_argument);
    CHECK(source.position <= 230);
}

TEST_CASE("GGUF artifact observation validates options without consuming input") {
    for (unsigned variant = 0; variant < 10; ++variant) {
        Source source(artifact()); std::istream in(&source); auto opt = options(); std::string digest = kA;
        switch (variant) {
            case 0: digest.clear(); break;
            case 1: digest[0] = 'F'; break;
            case 2: digest[0] = 'g'; break;
            case 3: opt.max_read_bytes = 1; break;
            case 4: opt.max_read_bytes = UINT64_MAX; break;
            case 5: opt.max_header_bytes = 0; break;
            case 6: opt.max_header_bytes = opt.max_read_bytes; break;
            case 7: opt.read_chunk_bytes = 0; break;
            case 8: opt.read_chunk_bytes = 65537; break;
            case 9: opt.deadline = std::chrono::steady_clock::time_point::max(); break;
        }
        error(sd::observe_gguf_artifact(in, digest, opt), si::ErrorCode::invalid_argument);
        CHECK(source.position == 0);
    }
}

TEST_CASE("GGUF artifact observation checks declared string allowance before allocation or read") {
    std::string bytes = "GGUF"; u64(bytes, 3, 4); u64(bytes, 0); u64(bytes, 1); u64(bytes, 1u << 24);
    Source source(bytes); std::istream in(&source); auto opt = options(); opt.max_header_bytes = 64;
    error(sd::observe_gguf_artifact(in, kA, opt), si::ErrorCode::invalid_argument);
    CHECK(source.position == 32); CHECK(source.max_request <= 8);
}

TEST_CASE("GGUF artifact observation rejects initial and midstream failures without raw diagnostics") {
    for (unsigned variant = 0; variant < 6; ++variant) {
        Source source(artifact()); std::istream in(&source);
        switch (variant) {
            case 0: in.setstate(std::ios::badbit); break;
            case 1: in.setstate(std::ios::failbit); break;
            case 2: source.throw_read = true; break;
            case 3: source.zero_progress = true; break;
            case 4: source.throw_eof = true; break;
            case 5: source.on_read = [&] { if (source.position >= 24) in.setstate(std::ios::badbit); }; break;
        }
        error(sd::observe_gguf_artifact(in, kA, options()), si::ErrorCode::io_error);
    }
    std::istream no_buffer(nullptr);
    error(sd::observe_gguf_artifact(no_buffer, kA, options()), si::ErrorCode::io_error);
}

TEST_CASE("GGUF artifact observation cancels before reads within reads and at EOF admission") {
    for (unsigned variant = 0; variant < 3; ++variant) {
        si::CancellationSource cancel; Source source(artifact()); std::istream in(&source);
        if (variant == 0) cancel.cancel();
        if (variant == 1) source.on_read = [&] { if (source.position >= 24) cancel.cancel(); };
        if (variant == 2) source.on_eof = [&] { cancel.cancel(); };
        error(sd::observe_gguf_artifact(in, kA, options(), cancel.token()), si::ErrorCode::cancelled);
        if (variant == 0) CHECK(source.position == 0);
    }
}

TEST_CASE("GGUF artifact observation cooperatively observes expired deadlines") {
    Source source(artifact()); std::istream in(&source); auto opt = options(); opt.deadline = {};
    error(sd::observe_gguf_artifact(in, kA, opt), si::ErrorCode::timeout); CHECK(source.position == 0);
    // A source may spend arbitrary time in I/O. No hard deadline is claimed;
    // expiry is detected after it returns. The reader snapshots options so
    // caller source code cannot change the admitted byte/deadline budgets.
    auto during = options(); during.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    source.on_read = [&] { std::this_thread::sleep_until(during.deadline); };
    error(sd::observe_gguf_artifact(in, kA, during), si::ErrorCode::timeout);
    si::CancellationSource cancel; cancel.cancel();
    error(sd::observe_gguf_artifact(in, kA, opt, cancel.token()), si::ErrorCode::cancelled);
}

TEST_CASE("GGUF artifact observation snapshots budgets before caller source code runs") {
    Source source(artifact()); std::istream in(&source); auto opt = options(); opt.max_header_bytes = 230;
    source.on_read = [&] { opt.max_header_bytes = 4096; opt.max_read_bytes = 1024 * 1024; };
    error(sd::observe_gguf_artifact(in, kA, opt), si::ErrorCode::invalid_argument);
    CHECK(source.position <= 230);
}

TEST_CASE("GGUF artifact observation snapshots trusted expectation and refuses buffer replacement") {
    std::string expected = kA; Source revision_b(artifact('B')); std::istream in(&revision_b);
    revision_b.on_read = [&] { expected.assign(kB); };
    error(sd::observe_gguf_artifact(in, expected, options()), si::ErrorCode::invalid_argument);
    Source original(artifact()), replacement(artifact()); std::istream replaced(&original);
    original.on_read = [&] { replaced.rdbuf(&replacement); };
    error(sd::observe_gguf_artifact(replaced, kA, options()), si::ErrorCode::io_error);
    CHECK(original.position == 4); CHECK(replacement.position == 0);
}

TEST_CASE("GGUF artifact observation preserves unknown counts and ordinary header interpretation") {
    for (const auto replacement : {"qwen3_8", "Qwen3.8"}) {
        auto bytes = artifact(); bytes.replace(bytes.find("qwen3.8", 24), 7, replacement);
        // Keep the required block-count key in the new literal namespace;
        // only expert keys retain the old namespace and must stay unknown.
        bytes.replace(bytes.find("qwen3.8.block_count"), 7, replacement);
        auto ordinary = si::parse_gguf_model_info(bytes);
        REQUIRE_MESSAGE(ordinary.ok(), ordinary.status().message()); CHECK_FALSE(ordinary->routed_experts);
        std::istringstream in(bytes); auto result = sd::observe_gguf_artifact(in, sd::sha256_hex(bytes), options());
        REQUIRE(result.ok()); CHECK(result->architecture == ordinary->architecture); CHECK_FALSE(result->routed_experts);
    }
    auto bytes = artifact(); bytes[0] = 'X'; std::istringstream malformed(bytes);
    error(sd::observe_gguf_artifact(malformed, kA, options()), si::ErrorCode::invalid_argument);
    REQUIRE(si::parse_gguf_model_info(header()).ok());
}

TEST_CASE("GGUF artifact observation handles multiple tail chunks and fresh instances") {
    auto bytes = artifact(); bytes.append(65536 - bytes.size(), 'X');
    constexpr auto expected = "3ac5941895fe366b44d7f935dab90d764cbb1ce58baf15ad6f6a6a7196e31409";
    for (unsigned i = 0; i < 8; ++i) {
        Source source(bytes); source.chunk = 257; std::istream in(&source); auto opt = options(); opt.read_chunk_bytes = 1024;
        auto result = sd::observe_gguf_artifact(in, expected, opt);
        REQUIRE(result.ok()); CHECK(result->byte_length == 65536); CHECK(source.position == 65536);
    }
}

TEST_CASE("GGUF artifact incremental SHA agrees with independent padding boundary vectors") {
    const std::array<std::pair<std::size_t, const char*>, 7> vectors{{
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {1, "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"},
        {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
        {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
        {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
        {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
        {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
    }};
    for (const auto& [length, expected] : vectors) {
        const std::string bytes(length, 'a'); CHECK(sd::sha256_hex(bytes) == expected);
        for (const auto split : {1u, 7u, 64u}) {
            sd::Sha256 hash;
            for (std::size_t at = 0; at < bytes.size(); at += split) hash.update(std::string_view(bytes).substr(at, split));
            const auto digest = hash.finish(); std::string hex;
            for (const auto byte : digest) { hex.push_back("0123456789abcdef"[byte >> 4]); hex.push_back("0123456789abcdef"[byte & 15]); }
            CHECK(hex == expected);
        }
    }
}
