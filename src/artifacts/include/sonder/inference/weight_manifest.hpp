// Sonder Inference: weight-content digest manifests.
//
// A manifest records, for every file of a model directory (multi-file
// safetensors or GGUF sets), the relative path, byte size and SHA-256, plus an
// optional expected SHA-256 (e.g. a Hugging Face LFS oid saved locally). It is
// evidence about bytes at rest in a directory at verification time. It does NOT
// bind those bytes to what a backend later loads (ROADMAP "associate inspected
// content with loaded bytes" stays open), it proves no provenance, and a file
// can change between the hash and the load (TOCTOU). Network is never used.
//
// Thread affinity: every function is reentrant; nothing holds global state.
// Progress callbacks run on the calling thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference::weights {

inline constexpr int kManifestVersion = 1;
inline constexpr const char* kManifestFormat = "sonder.weight_manifest";

struct ManifestFile {
    std::string path;    // relative, '/'-separated, validated by validate_relative_path()
    std::uint64_t size = 0;
    std::string sha256;  // 64 lowercase hex
    std::optional<std::string> expected_sha256;
};

struct WeightManifest {
    int version = kManifestVersion;
    std::string model_dir;  // leaf name of the directory at creation; informational only
    std::string created_at;  // UTC, e.g. 2026-11-19T08:00:00Z
    std::string tool_version;
    std::vector<ManifestFile> files;  // sorted by path (bytewise), unique
};

// relative path -> expected lowercase sha256 hex.
using ExpectedHashes = std::map<std::string, std::string>;

using ProgressFn = std::function<void(std::string_view rel_path, std::uint64_t done, std::uint64_t total)>;

struct HashOptions {
    std::size_t buffer_bytes = 1u << 20;  // clamped to [4 KiB, 64 MiB]; the only per-file allocation
    ProgressFn progress;                  // called after each chunk
    CancellationToken cancel;
};

// Accepts only relative, '/'-separated paths of non-empty segments: no
// absolute paths, drive letters or ':' , backslashes, '.', '..', control
// characters, or segments ending in '.' or ' '.
// Also rejects C1 controls (U+0080-U+009F).
Status validate_relative_path(std::string_view path);

// Renders a name for a terminal or log line; never throws. Ordinary names come back unchanged.
// Lossless and unambiguous: '\' -> "\\", bytes < 0x20 and 0x7F -> "\xNN", each byte of an ill-formed
// UTF-8 sequence -> "\xNN", and C1 controls, U+2028/U+2029 and bidi controls (U+202A-U+202E,
// U+2066-U+2069) -> a backslash, u and four hex digits. Names that were not validated must go through this before being printed.
std::string printable_path(std::string_view s);
bool is_sha256_hex(std::string_view text);  // exactly 64 lowercase hex digits

// Streams `file` through SHA-256 with a bounded buffer. `display` is only used for the
// progress callback and error messages. `bytes_read` receives the number of bytes hashed.
Result<std::string> hash_file(const std::filesystem::path& file, std::string_view display,
                              const HashOptions& options, std::uint64_t* bytes_read = nullptr);

struct DirListing {
    std::vector<std::string> files;   // safe regular files, relative paths, sorted
    // symlinks/reparse points, special files, names that are not valid UTF-8 or have no UTF-8 form.
    // Raw names (machine data for a supervisor, may hold control characters): print through printable_path().
    std::vector<std::string> unsafe;
};
// Walks `dir` without following links. Entries equivalent to any `ignore` path are skipped.
Result<DirListing> list_directory(const std::filesystem::path& dir,
                                  const std::vector<std::filesystem::path>& ignore = {});

struct BuildOptions {
    ExpectedHashes expected;
    std::vector<std::filesystem::path> ignore;
    HashOptions hash;
};
struct BuildResult {
    WeightManifest manifest;
    std::vector<std::string> expected_mismatch;  // computed sha256 != expected
    std::vector<std::string> expected_missing;   // expected path not present in the directory
};
// Refuses (invalid_argument) a directory holding any unsafe entry.
Result<BuildResult> build_manifest(const std::filesystem::path& dir, const BuildOptions& options);

std::string manifest_to_json(const WeightManifest& manifest);  // compact JSON plus trailing '\n'
Result<WeightManifest> parse_manifest(std::string_view json_text);
// Accepts a Hugging Face tree-API array (entries with "path" and "lfs":{"oid"}),
// a model-API object with "siblings" (entries with "rfilename"/"path"), or either
// with a plain "sha256" member. Entries without a sha256 are ignored: a non-LFS git
// blob oid is a SHA-1, not a content digest.
Result<ExpectedHashes> parse_expected_json(std::string_view json_text);

struct VerifyOptions {
    bool full = false;                   // hash every file
    // Hash up to this many randomly chosen files. The choice is drawn from all manifest entries
    // (so a seed selects the same indices whatever is on disk); entries already reported missing
    // or size-mismatched are not hashed, so fewer than N may be.
    std::optional<std::size_t> sample;
    std::uint64_t seed = 0;              // for sample; reported back
    std::vector<std::filesystem::path> ignore;
    HashOptions hash;
};
struct Mismatch {
    std::string path;
    std::string kind;  // "size" | "sha256" | "expected_sha256"
    std::string expected;
    std::string actual;
};
struct VerifyReport {
    std::size_t files_total = 0;
    std::uint64_t seed = 0;
    std::vector<std::string> hashed;
    std::vector<Mismatch> mismatches;
    std::vector<std::string> missing;
    std::vector<std::string> extra;
    std::vector<std::string> unsafe;  // raw names; print through printable_path()
    // 0 ok, 2 content/size mismatch (takes precedence), 3 missing/extra/unsafe entries.
    [[nodiscard]] int exit_code() const noexcept;
};
// Always checks, without hashing: presence, byte size, extra files, symlinks, and the
// manifest's own expected_sha256 against its sha256. Hashes per `options`. I/O
// failures and cancellation return an error Status instead of a report.
Result<VerifyReport> verify_manifest(const std::filesystem::path& dir, const WeightManifest& manifest,
                                     const VerifyOptions& options);

// Deterministic across platforms and standard libraries: `n` distinct indices of
// [0, count), sorted ascending (all of them when n >= count).
std::vector<std::size_t> select_sample(std::size_t count, std::size_t n, std::uint64_t seed);

// `sonder-infer weights manifest|verify ...`. Exit codes: 0 ok, 1 I/O failure,
// 2 mismatch, 3 missing/extra file, 4 usage or invalid manifest, 130 cancelled.
int weights_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace sonder::inference::weights
