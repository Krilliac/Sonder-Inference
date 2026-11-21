#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "sonder/inference/weight_manifest.hpp"

namespace fs = std::filesystem;
namespace w = sonder::inference::weights;

namespace {

constexpr const char* kAbcSha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("sonder-weights-test-" + std::to_string(tick) + "-" + std::to_string(counter++));
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

std::string pattern(std::size_t n, unsigned salt) {
    std::string s(n, '\0');
    for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<char>((i * 31u + salt * 7u + (i >> 8)) & 0xff);
    return s;
}

// A small "model": two shards in the root, one nested file, an empty file.
void make_model(const fs::path& dir) {
    write_file(dir / "model-00001-of-00002.safetensors", pattern(10000, 1));
    write_file(dir / "model-00002-of-00002.safetensors", pattern(5000, 2));
    write_file(dir / "tokenizer" / "tokenizer.json", "{\"v\":1}");
    write_file(dir / "empty.bin", "");
}

struct CliResult {
    int rc;
    std::string out;
    std::string err;
};

CliResult run_cli(std::vector<std::string> args) {
    std::ostringstream out, err;
    const int rc = w::weights_main(args, out, err);
    return {rc, out.str(), err.str()};
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void flip_byte(const fs::path& p, std::size_t offset) {
    std::string data = slurp(p);
    REQUIRE(offset < data.size());
    data[offset] = static_cast<char>(data[offset] ^ 0x01);
    write_file(p, data);
}

}  // namespace

TEST_CASE("weights: SHA-256 streaming matches the known vector and is independent of buffer size") {
    TempDir t;
    write_file(t.path / "abc", "abc");
    auto digest = w::hash_file(t.path / "abc", "abc", {});
    REQUIRE(digest.ok());
    CHECK(digest.value() == kAbcSha);

    const std::string big = pattern(100000, 3);
    write_file(t.path / "big", big);
    w::HashOptions small;
    small.buffer_bytes = 1;  // clamped to 4 KiB, not a 1 byte loop
    std::uint64_t chunks = 0;
    small.progress = [&](std::string_view, std::uint64_t, std::uint64_t) { ++chunks; };
    std::uint64_t n = 0;
    auto a = w::hash_file(t.path / "big", "big", small, &n);
    auto b = w::hash_file(t.path / "big", "big", {});
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    CHECK(a.value() == b.value());
    CHECK(n == big.size());
    CHECK(chunks == (big.size() + 4095) / 4096);

    auto empty = w::hash_file(t.path / "missing", "missing", {});
    CHECK_FALSE(empty.ok());
}

TEST_CASE("weights: manifest and verify round trip through the CLI exit 0") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path manifest = t.path / "m.json";
    auto made = run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"});
    CHECK(made.rc == 0);
    CHECK(made.out.find("4 files") != std::string::npos);

    auto parsed = w::parse_manifest(slurp(manifest));
    REQUIRE(parsed.ok());
    CHECK(parsed.value().files.size() == 4);
    CHECK(parsed.value().model_dir == "model");
    CHECK_FALSE(parsed.value().created_at.empty());
    CHECK_FALSE(parsed.value().tool_version.empty());
    CHECK(parsed.value().files.front().path == "empty.bin");  // sorted by path

    auto verified = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(verified.rc == 0);
    CHECK(verified.out.find("verify: OK") != std::string::npos);
    CHECK(verified.out.find("4 hashed") != std::string::npos);

    // Default mode: names and sizes only, nothing hashed.
    auto cheap = run_cli({"verify", model.string(), manifest.string(), "--quiet"});
    CHECK(cheap.rc == 0);
    CHECK(cheap.out.find("0 hashed") != std::string::npos);
}

TEST_CASE("weights: a manifest stored inside the model directory is not an extra file") {
    TempDir t;
    make_model(t.path);
    const fs::path manifest = t.path / "weights.manifest.json";
    REQUIRE(run_cli({"manifest", t.path.string(), "-o", manifest.string(), "--quiet"}).rc == 0);
    auto parsed = w::parse_manifest(slurp(manifest));
    REQUIRE(parsed.ok());
    CHECK(parsed.value().files.size() == 4);
    CHECK(run_cli({"verify", t.path.string(), manifest.string(), "--full", "--quiet"}).rc == 0);

    // Regenerating over the existing manifest is accepted (it is a valid manifest) and still ignored.
    REQUIRE(run_cli({"manifest", t.path.string(), "-o", manifest.string(), "--quiet"}).rc == 0);
    auto again = w::parse_manifest(slurp(manifest));
    REQUIRE(again.ok());
    CHECK(again.value().files.size() == 4);
    CHECK(run_cli({"verify", t.path.string(), manifest.string(), "--full", "--quiet"}).rc == 0);
}

TEST_CASE("weights: one flipped byte is a sha256 mismatch with exit 2") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);
    flip_byte(model / "model-00001-of-00002.safetensors", 5000);

    auto full = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(full.rc == 2);
    CHECK(full.out.find("mismatch: model-00001-of-00002.safetensors (sha256") != std::string::npos);
    CHECK(full.out.find("verify: FAILED") != std::string::npos);

    // A sample at least as large as the file count covers the corrupt file too.
    auto sampled = run_cli({"verify", model.string(), manifest.string(), "--sample", "4", "--seed", "1", "--quiet"});
    CHECK(sampled.rc == 2);

    // Cheap mode cannot see a same-size flip; that is documented behaviour.
    CHECK(run_cli({"verify", model.string(), manifest.string(), "--quiet"}).rc == 0);

    // A size change is caught even without hashing.
    write_file(model / "model-00002-of-00002.safetensors", pattern(4999, 2));
    auto cheap = run_cli({"verify", model.string(), manifest.string(), "--quiet"});
    CHECK(cheap.rc == 2);
    CHECK(cheap.out.find("(size expected 5000, actual 4999)") != std::string::npos);
}

TEST_CASE("weights: missing file is exit 3 and an extra file is reported") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);

    fs::remove(model / "tokenizer" / "tokenizer.json");
    auto missing = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(missing.rc == 3);
    CHECK(missing.out.find("missing: tokenizer/tokenizer.json") != std::string::npos);

    write_file(model / "tokenizer" / "tokenizer.json", "{\"v\":1}");
    write_file(model / "rogue.bin", "x");
    auto extra = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(extra.rc == 3);
    CHECK(extra.out.find("extra: rogue.bin") != std::string::npos);

    // A content mismatch outranks a missing or extra file.
    flip_byte(model / "model-00002-of-00002.safetensors", 0);
    CHECK(run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"}).rc == 2);
}

TEST_CASE("weights: escaping and absolute manifest paths are refused") {
    const std::string sha(64, 'a');
    auto doc = [&](const std::string& path) {
        return "{\"format\":\"sonder.weight_manifest\",\"version\":1,\"files\":[{\"path\":\"" + path +
               "\",\"size\":1,\"sha256\":\"" + sha + "\"}]}";
    };
    CHECK(w::parse_manifest(doc("ok/file.bin")).ok());
    for (const char* bad : {"../escape.bin", "a/../../escape.bin", "/etc/passwd", "a//b", "./a", "a/./b", "C:/x.bin",
                            "a\\\\b.bin", "a/", "dir/..", "name.", "x:stream", ""}) {
        CAPTURE(bad);
        CHECK_FALSE(w::parse_manifest(doc(bad)).ok());
        CHECK_FALSE(w::validate_relative_path(bad).ok());
    }

    TempDir t;
    make_model(t.path / "model");
    write_file(t.path / "bad.json", doc("../outside.bin"));
    auto r = run_cli({"verify", (t.path / "model").string(), (t.path / "bad.json").string(), "--quiet"});
    CHECK(r.rc == 4);
    CHECK(r.err.find("unsafe path") != std::string::npos);
}

TEST_CASE("weights: manifest parsing rejects bad versions, hashes and duplicates") {
    const std::string sha(64, 'b');
    auto one = [&](const std::string& version, const std::string& digest) {
        return "{\"format\":\"sonder.weight_manifest\",\"version\":" + version +
               ",\"files\":[{\"path\":\"a\",\"size\":1,\"sha256\":\"" + digest + "\"}]}";
    };
    CHECK(w::parse_manifest(one("1", sha)).ok());
    CHECK_FALSE(w::parse_manifest(one("2", sha)).ok());
    CHECK_FALSE(w::parse_manifest(one("1", std::string(63, 'b'))).ok());
    CHECK_FALSE(w::parse_manifest(one("1", std::string(64, 'B'))).ok());
    CHECK_FALSE(w::parse_manifest("{\"format\":\"other\",\"version\":1,\"files\":[]}").ok());
    CHECK_FALSE(w::parse_manifest("not json").ok());
    const std::string dup = "{\"format\":\"sonder.weight_manifest\",\"version\":1,\"files\":["
                            "{\"path\":\"a\",\"size\":1,\"sha256\":\"" + sha + "\"},"
                            "{\"path\":\"a\",\"size\":1,\"sha256\":\"" + sha + "\"}]}";
    CHECK_FALSE(w::parse_manifest(dup).ok());
}

// Creates a directory link at `link` pointing to `target`. Tries a symlink first; on Windows, where that
// needs a privilege, falls back to a junction (`mklink /J`), which any user can create.
bool make_dir_link(const fs::path& link, const fs::path& target) {
    std::error_code ec;
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

TEST_CASE("weights: directory links (symlink or junction) inside the model directory are refused") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path outside = t.path / "outside";
    write_file(outside / "tokenizer.json", "{\"v\":1}");
    const fs::path manifest = t.path / "m.json";

    // Case 1: a link to a directory sitting in the model tree blocks manifest creation.
    const fs::path link = model / "linked";
    if (!make_dir_link(link, outside)) {
        FAIL_CHECK("no directory link (symlink or junction) could be created here; link refusal NOT exercised");
        return;
    }
    auto built = w::build_manifest(model, {});
    CHECK_FALSE(built.ok());
    CHECK(built.status().code() == sonder::inference::ErrorCode::invalid_argument);
    auto cli = run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"});
    CHECK(cli.rc == 4);
    CHECK_FALSE(fs::exists(manifest));
    fs::remove(link);

    // Case 2: a real manifest, then the "tokenizer" directory is swapped for a link to a directory
    // holding an identical file. Verify must report it unsafe (exit 3) and never read through it.
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);
    fs::remove_all(model / "tokenizer");
    REQUIRE(make_dir_link(model / "tokenizer", outside));
    auto verified = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(verified.rc == 3);
    CHECK(verified.out.find("unsafe: tokenizer/tokenizer.json") != std::string::npos);
    fs::remove(model / "tokenizer");
}

TEST_CASE("weights: file symlinks inside the model directory are refused") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    write_file(t.path / "outside.bin", "secret");
    std::error_code ec;
    fs::create_symlink(t.path / "outside.bin", model / "link.bin", ec);
    if (ec) {
        // Unprivileged Windows cannot create file symlinks; the directory-link case above covers the
        // reparse-point branch there. Reported as a failed check so it never reads as a silent pass
        // on a platform that could have run it (POSIX always can).
#if defined(_WIN32)
        MESSAGE("file symlinks need a privilege here (" << ec.message() << "); covered by the directory-link case");
        return;
#else
        FAIL_CHECK("file symlink creation failed: " << ec.message());
        return;
#endif
    }
    auto built = w::build_manifest(model, {});
    CHECK_FALSE(built.ok());
    CHECK(built.status().code() == sonder::inference::ErrorCode::invalid_argument);
    auto cli = run_cli({"manifest", model.string(), "-o", (t.path / "m.json").string(), "--quiet"});
    CHECK(cli.rc == 4);
    CHECK_FALSE(fs::exists(t.path / "m.json"));

    fs::remove(model / "link.bin");
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);
    fs::remove(model / "empty.bin");
    fs::create_symlink(t.path / "outside.bin", model / "empty.bin", ec);
    REQUIRE_FALSE(ec);
    auto verified = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(verified.rc == 3);
    CHECK(verified.out.find("unsafe: empty.bin") != std::string::npos);
}

TEST_CASE("weights: ill-formed UTF-8 names are refused, not written as U+FFFD") {
    // Strict validator, platform independent.
    CHECK(w::validate_relative_path("caf\xc3\xa9/ok.bin").ok());
    for (const char* bad : {"bad\xff.bin", "a\x80", "\xc0\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80", "x\xe2\x82"}) {
        CAPTURE(bad);
        CHECK_FALSE(w::validate_relative_path(bad).ok());
    }
#if !defined(_WIN32)
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path odd = model / std::string("bad\xff.bin");
    {
        std::ofstream f(odd, std::ios::binary);
        if (!f) {
            FAIL_CHECK("could not create a non-UTF-8 file name on this filesystem; refusal NOT exercised");
            return;
        }
        f << "x";
    }
    auto listing = w::list_directory(model, {});
    REQUIRE(listing.ok());
    CHECK(listing.value().unsafe.size() == 1);
    CHECK(std::find(listing.value().files.begin(), listing.value().files.end(), "bad\xff.bin") ==
          listing.value().files.end());
    auto built = w::build_manifest(model, {});
    CHECK_FALSE(built.ok());
    CHECK(built.status().code() == sonder::inference::ErrorCode::invalid_argument);
    const fs::path manifest = t.path / "m.json";
    CHECK(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 4);
    CHECK_FALSE(fs::exists(manifest));
#endif
}

TEST_CASE("weights: an unreadable subdirectory fails the listing instead of returning a truncated one") {
#if defined(_WIN32)
    MESSAGE("POSIX permission bits are needed to make a subdirectory unreadable; not exercised on Windows");
#else
    TempDir t;
    const fs::path model = t.path / "model";
    fs::create_directories(model / "locked");
    { std::ofstream(model / "a.bin", std::ios::binary) << "x"; }
    { std::ofstream(model / "locked" / "b.bin", std::ios::binary) << "y"; }
    fs::permissions(model / "locked", fs::perms::none);
    std::error_code probe;
    const bool readable = fs::directory_iterator(model / "locked", probe) != fs::directory_iterator() || !probe;
    if (!probe && readable) {
        fs::permissions(model / "locked", fs::perms::owner_all);
        MESSAGE("running with permission checks bypassed (root); not exercised");
        return;
    }
    auto listing = w::list_directory(model, {});
    fs::permissions(model / "locked", fs::perms::owner_all);
    CHECK_FALSE(listing.ok());
    if (!listing.ok()) CHECK(listing.status().code() == sonder::inference::ErrorCode::io_error);
#endif
}

TEST_CASE("weights: printable_path escapes control, C1, bidi and ill-formed bytes and keeps ordinary names") {
    CHECK(w::printable_path("a\nverify: OK") == "a\\x0averify: OK");
    CHECK(w::printable_path("\x1b[2J") == "\\x1b[2J");
    CHECK(w::printable_path("a\x7f") == "a\\x7f");
    CHECK(w::printable_path("\xc2\x9b") == "\\u009b");
    CHECK(w::printable_path("bad\xff") == "bad\\xff");
    CHECK(w::printable_path("a\\b") == "a\\\\b");
    CHECK(w::printable_path("\xe2\x80\xae") == "\\u202e");
    CHECK(w::printable_path("\xe2\x80\xa8") == "\\u2028");
    CHECK(w::printable_path("\xe2\x81\xa6") == "\\u2066");
    CHECK(w::printable_path("x\xe2\x82") == "x\\xe2\\x82");  // truncated sequence: every byte escaped
    CHECK(w::printable_path("caf\xc3\xa9/x.bin") == "caf\xc3\xa9/x.bin");
    CHECK(w::printable_path("tokenizer/tokenizer.json") == "tokenizer/tokenizer.json");
    CHECK(w::printable_path("").empty());

    CHECK_FALSE(w::validate_relative_path("a\xc2\x9b").ok());
    CHECK_FALSE(w::validate_relative_path("\xc2\x80/x").ok());
    CHECK(w::validate_relative_path("caf\xc3\xa9").ok());

    const std::string sha(64, 'a');
    const std::string doc = "{\"format\":\"sonder.weight_manifest\",\"version\":1,\"files\":[{\"path\":\"x\\nverify: OK\","
                            "\"size\":-1,\"sha256\":\"" + sha + "\"}]}";
    auto parsed = w::parse_manifest(doc);
    REQUIRE_FALSE(parsed.ok());
    CHECK(parsed.status().message().find('\n') == std::string::npos);
    CHECK(parsed.status().message().find("x\\x0averify: OK") != std::string::npos);
}

TEST_CASE("weights: a C1 control character in a file name is escaped on stdout and stderr") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);

    // Built from bytes (U+009B is C2 9B) so the test does not depend on how the compiler reads a literal.
    const std::u8string name{u8'x', static_cast<char8_t>(0xC2), static_cast<char8_t>(0x9B)};
    {
        const fs::path odd = model / fs::path(name + u8".bin");
        std::ofstream f(odd, std::ios::binary);
        if (!f) {
            FAIL_CHECK("could not create a file name holding U+009B here; C1 escaping NOT exercised");
            return;
        }
        f << "x";
    }
    const std::string raw = "\xc2\x9b";
    auto verified = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(verified.rc == 3);
    CHECK(verified.out.find(raw) == std::string::npos);
    CHECK(verified.err.find(raw) == std::string::npos);
    CHECK(verified.out.find("unsafe: x\\u009b.bin") != std::string::npos);

    auto made = run_cli({"manifest", model.string(), "-o", (t.path / "m2.json").string(), "--quiet"});
    CHECK(made.rc == 4);
    CHECK(made.err.find(raw) == std::string::npos);
    CHECK(made.err.find("x\\u009b.bin") != std::string::npos);
}

#if !defined(_WIN32)
TEST_CASE("weights: newline and escape characters in file names cannot forge output lines") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);

    for (const char* name : {"x\nverify: OK (4 files checked, 4 hashed)", "e\x1b[1A.bin"}) {
        std::ofstream f(model / std::string(name), std::ios::binary);
        if (!f) {
            FAIL_CHECK("could not create a file name with control characters here; escaping NOT exercised");
            return;
        }
        f << "x";
    }
    auto verified = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(verified.rc == 3);
    std::istringstream lines(verified.out);
    for (std::string line; std::getline(lines, line);) {
        CAPTURE(line);
        CHECK(line.rfind("verify: OK", 0) == std::string::npos);
    }
    CHECK(verified.out.find('\x1b') == std::string::npos);
    CHECK(verified.out.find("x\\x0averify: OK") != std::string::npos);

    auto made = run_cli({"manifest", model.string(), "-o", (t.path / "m2.json").string(), "--quiet"});
    CHECK(made.rc == 4);
    CHECK(made.err.find('\x1b') == std::string::npos);
    CHECK(std::count(made.err.begin(), made.err.end(), '\n') == 1);
}
#endif

TEST_CASE("weights: -o refuses to overwrite a weight shard, a non-manifest file, a link or a directory") {
    TempDir t;
    const fs::path model = t.path / "model";
    make_model(model);
    const fs::path shard = model / "model-00001-of-00002.safetensors";
    const std::string shard_bytes = pattern(10000, 1);

    // Inside the model directory: the shard must neither be skipped nor truncated.
    auto in_dir = run_cli({"manifest", model.string(), "-o", shard.string(), "--quiet"});
    CHECK(in_dir.rc == 4);
    CHECK(in_dir.err.find("refusing to overwrite") != std::string::npos);
    CHECK(slurp(shard) == shard_bytes);

    // An existing non-manifest file elsewhere.
    write_file(t.path / "notes.txt", "hello");
    auto notes = run_cli({"manifest", model.string(), "-o", (t.path / "notes.txt").string(), "--quiet"});
    CHECK(notes.rc == 4);
    CHECK(slurp(t.path / "notes.txt") == "hello");

    // A hard link to a shard (no privilege needed on NTFS or POSIX).
    {
        std::error_code ec;
        fs::create_hard_link(shard, t.path / "hl.json", ec);
        if (ec) {
            FAIL_CHECK("hard link creation failed: " << ec.message() << "; hard-link refusal NOT exercised");
        } else {
            auto linked = run_cli({"manifest", model.string(), "-o", (t.path / "hl.json").string(), "--quiet"});
            CHECK(linked.rc == 4);
            CHECK(slurp(shard) == shard_bytes);
        }
    }

    // A symlink pointing at a shard is not written through.
    {
        std::error_code ec;
        fs::create_symlink(shard, t.path / "sl.json", ec);
        if (ec) {
#if defined(_WIN32)
            MESSAGE("file symlinks need a privilege here (" << ec.message() << "); symlink -o refusal not exercised");
#else
            FAIL_CHECK("symlink creation failed: " << ec.message());
#endif
        } else {
            auto sym = run_cli({"manifest", model.string(), "-o", (t.path / "sl.json").string(), "--quiet"});
            CHECK(sym.rc == 4);
            CHECK(slurp(shard) == shard_bytes);
        }
    }

    // A directory.
    fs::create_directories(t.path / "outdir");
    CHECK(run_cli({"manifest", model.string(), "-o", (t.path / "outdir").string(), "--quiet"}).rc == 4);

    // The shard is still intact and nothing was written into the model by the refusals.
    CHECK(slurp(shard) == shard_bytes);
}

TEST_CASE("weights: expected hashes from a saved Hugging Face tree listing are enforced") {
    TempDir t;
    const fs::path model = t.path / "model";
    write_file(model / "a.safetensors", "abc");
    write_file(model / "config.json", "{}");
    const std::string wrong(64, '0');
    auto tree = [&](const std::string& oid_a) {
        return "[{\"type\":\"file\",\"oid\":\"1111111111111111111111111111111111111111\",\"size\":3,\"path\":\"a.safetensors\","
               "\"lfs\":{\"oid\":\"" + oid_a + "\",\"size\":3,\"pointerSize\":134}},"
               "{\"type\":\"file\",\"oid\":\"2222222222222222222222222222222222222222\",\"size\":2,\"path\":\"config.json\"},"
               "{\"type\":\"directory\",\"oid\":\"3333\",\"path\":\"onnx\"}]";
    };

    auto expected = w::parse_expected_json(tree(kAbcSha));
    REQUIRE(expected.ok());
    CHECK(expected.value().size() == 1);  // git blob oid (SHA-1) of config.json is not a content digest
    CHECK(expected.value().at("a.safetensors") == kAbcSha);
    CHECK(w::parse_expected_json("{\"siblings\":[{\"rfilename\":\"a.safetensors\",\"lfs\":{\"sha256\":\"" +
                                 std::string(kAbcSha) + "\"}}]}").ok());
    CHECK_FALSE(w::parse_expected_json("{\"x\":1}").ok());
    CHECK_FALSE(w::parse_expected_json(tree("zz")).ok());
    CHECK_FALSE(w::parse_expected_json("[{\"path\":\"../x\",\"sha256\":\"" + std::string(kAbcSha) + "\"}]").ok());

    write_file(t.path / "good.json", tree(kAbcSha));
    const fs::path manifest = t.path / "m.json";
    auto ok = run_cli({"manifest", model.string(), "-o", manifest.string(), "--expected", (t.path / "good.json").string(), "--quiet"});
    CHECK(ok.rc == 0);
    auto parsed = w::parse_manifest(slurp(manifest));
    REQUIRE(parsed.ok());
    CHECK(parsed.value().files[0].path == "a.safetensors");
    CHECK(parsed.value().files[0].expected_sha256 == std::optional<std::string>(kAbcSha));
    CHECK_FALSE(parsed.value().files[1].expected_sha256.has_value());

    write_file(t.path / "bad.json", tree(wrong));
    const fs::path bad_manifest = t.path / "bad-m.json";
    auto bad = run_cli({"manifest", model.string(), "-o", bad_manifest.string(), "--expected", (t.path / "bad.json").string(), "--quiet"});
    CHECK(bad.rc == 2);
    CHECK(bad.out.find("expected-mismatch: a.safetensors") != std::string::npos);
    // The manifest still records what was found, so verify keeps reporting the disagreement.
    auto again = run_cli({"verify", model.string(), bad_manifest.string(), "--quiet"});
    CHECK(again.rc == 2);
    CHECK(again.out.find("expected_sha256") != std::string::npos);

    // Listed upstream but absent locally.
    write_file(t.path / "missing.json",
               "[{\"type\":\"file\",\"path\":\"b.safetensors\",\"lfs\":{\"oid\":\"" + std::string(kAbcSha) + "\"}}]");
    auto gone = run_cli({"manifest", model.string(), "-o", (t.path / "m3.json").string(), "--expected",
                         (t.path / "missing.json").string(), "--quiet"});
    CHECK(gone.rc == 3);
    CHECK(gone.out.find("missing: b.safetensors") != std::string::npos);
}

TEST_CASE("weights: sample selection is deterministic for a seed and distinct") {
    const auto a = w::select_sample(100, 10, 42);
    const auto b = w::select_sample(100, 10, 42);
    CHECK(a == b);
    CHECK(a.size() == 10);
    CHECK(std::set<std::size_t>(a.begin(), a.end()).size() == 10);
    for (std::size_t i = 1; i < a.size(); ++i) CHECK(a[i - 1] < a[i]);
    for (std::size_t v : a) CHECK(v < 100);

    bool differs = false;
    for (std::uint64_t seed = 43; seed < 60 && !differs; ++seed) differs = w::select_sample(100, 10, seed) != a;
    CHECK(differs);

    CHECK(w::select_sample(5, 9, 1).size() == 5);
    CHECK(w::select_sample(5, 0, 1).empty());
    CHECK(w::select_sample(0, 3, 1).empty());
}

TEST_CASE("weights: sample verify hashes the same files for a seed and reports the seed") {
    TempDir t;
    const fs::path model = t.path / "model";
    for (int i = 0; i < 12; ++i) write_file(model / ("shard-" + std::to_string(i) + ".bin"), pattern(300 + i, static_cast<unsigned>(i)));
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);

    auto first = run_cli({"verify", model.string(), manifest.string(), "--sample", "3", "--seed", "99", "--quiet"});
    auto second = run_cli({"verify", model.string(), manifest.string(), "--sample", "3", "--seed", "99", "--quiet"});
    CHECK(first.rc == 0);
    CHECK(first.out == second.out);
    CHECK(first.out.find("sample: seed 99, 3 of 12 files hashed") != std::string::npos);

    // Without --seed a seed is drawn and printed so the run can be repeated.
    auto random = run_cli({"verify", model.string(), manifest.string(), "--sample", "2", "--quiet"});
    CHECK(random.rc == 0);
    const std::string marker = "sample: seed ";
    const auto at = random.out.find(marker);
    REQUIRE(at != std::string::npos);
    const std::size_t digits_at = at + marker.size();
    const std::size_t digits_end = random.out.find(',', digits_at);
    REQUIRE(digits_end != std::string::npos);
    const std::string drawn = random.out.substr(digits_at, digits_end - digits_at);
    CHECK_FALSE(drawn.empty());

    // Replaying the printed seed gives the same sampled files.
    auto replay = run_cli({"verify", model.string(), manifest.string(), "--sample", "2", "--seed", drawn, "--quiet"});
    CHECK(replay.rc == 0);
    auto sampled_lines = [](const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream in(text);
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("sampled: ", 0) == 0) lines.push_back(line);
        }
        return lines;
    };
    CHECK(sampled_lines(random.out).size() == 2);
    CHECK(sampled_lines(random.out) == sampled_lines(replay.out));
}

TEST_CASE("weights: sample mode hashes only the chosen files, full mode hashes all") {
    TempDir t;
    const fs::path model = t.path / "model";
    for (int i = 0; i < 12; ++i) write_file(model / ("shard-" + std::to_string(i) + ".bin"), pattern(300 + i, static_cast<unsigned>(i)));
    const fs::path manifest = t.path / "m.json";
    REQUIRE(run_cli({"manifest", model.string(), "-o", manifest.string(), "--quiet"}).rc == 0);
    auto parsed = w::parse_manifest(slurp(manifest));
    REQUIRE(parsed.ok());
    const auto& files = parsed.value().files;
    REQUIRE(files.size() == 12);

    const auto chosen = w::select_sample(files.size(), 3, 7);
    REQUIRE(chosen.size() == 3);
    std::size_t outside = 0;
    while (std::find(chosen.begin(), chosen.end(), outside) != chosen.end()) ++outside;

    // Same-size corruption of a file the sample does not cover is invisible to the sample...
    flip_byte(model / files[outside].path, 10);
    auto missed = run_cli({"verify", model.string(), manifest.string(), "--sample", "3", "--seed", "7", "--quiet"});
    CHECK(missed.rc == 0);
    CHECK(missed.out.find("sampled: " + files[outside].path) == std::string::npos);
    // ...but --full finds it.
    auto full = run_cli({"verify", model.string(), manifest.string(), "--full", "--quiet"});
    CHECK(full.rc == 2);
    CHECK(full.out.find("mismatch: " + files[outside].path) != std::string::npos);

    // Corruption of a chosen file is found by the sample with that seed.
    flip_byte(model / files[outside].path, 10);  // restore
    flip_byte(model / files[chosen[1]].path, 10);
    auto hit = run_cli({"verify", model.string(), manifest.string(), "--sample", "3", "--seed", "7", "--quiet"});
    CHECK(hit.rc == 2);
    CHECK(hit.out.find("mismatch: " + files[chosen[1]].path) != std::string::npos);
}

TEST_CASE("weights: command line usage errors exit 4") {
    TempDir t;
    make_model(t.path / "model");
    const std::string dir = (t.path / "model").string();
    const std::string m = (t.path / "m.json").string();
    CHECK(run_cli({}).rc == 4);
    CHECK(run_cli({"frobnicate"}).rc == 4);
    CHECK(run_cli({"manifest", dir}).rc == 4);                      // no -o
    CHECK(run_cli({"manifest", "-o", m}).rc == 4);                  // no dir
    CHECK(run_cli({"manifest", dir, "-o", m, "--sample", "1"}).rc == 4);
    CHECK(run_cli({"verify", dir}).rc == 4);                        // no manifest
    CHECK(run_cli({"verify", dir, m, "--full", "--sample", "2"}).rc == 4);
    CHECK(run_cli({"verify", dir, m, "--sample", "x"}).rc == 4);
    CHECK(run_cli({"verify", dir, m, "--seed", "5"}).rc == 4);
    CHECK(run_cli({"verify", dir, m, "--bogus"}).rc == 4);
    CHECK(run_cli({"manifest", "--help"}).rc == 0);
    // An unreadable manifest file is an I/O failure, not a usage error.
    CHECK(run_cli({"verify", dir, m, "--quiet"}).rc == 1);
    CHECK(run_cli({"manifest", (t.path / "nope").string(), "-o", m, "--quiet"}).rc == 1);
}

TEST_CASE("weights: cancellation stops hashing") {
    TempDir t;
    write_file(t.path / "f", pattern(50000, 1));
    sonder::inference::CancellationSource source;
    source.cancel();
    w::HashOptions options;
    options.cancel = source.token();
    auto r = w::hash_file(t.path / "f", "f", options);
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() == sonder::inference::ErrorCode::cancelled);
}
