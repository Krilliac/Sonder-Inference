// Unit tests for versioned sequence KV slot persistence (slot_store).
// Uses a fake sequence-state buffer and tiny synthetic/vocab-only GGUF headers
// so no model weights are required.
#include "slot_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "common/sha256.hpp"
#include <doctest/doctest.h>

namespace fs = std::filesystem;
using namespace sonder::backends::llamacpp;

namespace {

struct ScopedTempDir {
    fs::path dir;

    ScopedTempDir() {
        static std::atomic<std::uint64_t> seq{0};
        const auto n = ++seq;
        dir = fs::temp_directory_path() / ("sonder_slot_store_test_" + std::to_string(n) + "_" +
                                           std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
    }

    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

class FakeSlotStateBackend final : public SlotStateBackend {
public:
    std::map<std::int32_t, std::vector<std::uint8_t>> slots;
    bool fail_get_size = false;
    bool fail_get_data = false;
    bool short_get_data = false;
    bool fail_set_data = false;
    mutable std::size_t get_size_calls = 0;
    mutable std::size_t get_data_calls = 0;
    std::size_t set_data_calls = 0;

    SlotStoreStatus GetSequenceStateSize(std::int32_t seq_id, std::size_t& out_size) const override {
        ++get_size_calls;
        out_size = 0;
        if (seq_id < 0) {
            return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "negative seq_id");
        }
        if (fail_get_size) {
            return SlotStoreStatus::Error(SlotStoreError::kBackendError, "injected get_size failure");
        }
        const auto it = slots.find(seq_id);
        if (it == slots.end() || it->second.empty()) {
            return SlotStoreStatus::Error(SlotStoreError::kBackendError, "empty sequence state");
        }
        out_size = it->second.size();
        return SlotStoreStatus::Ok();
    }

    SlotStoreStatus GetSequenceStateData(std::int32_t seq_id, std::span<std::uint8_t> dst,
                                         std::size_t& out_written) const override {
        ++get_data_calls;
        out_written = 0;
        if (seq_id < 0) {
            return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "negative seq_id");
        }
        if (fail_get_data) {
            return SlotStoreStatus::Error(SlotStoreError::kBackendError, "injected get_data failure");
        }
        const auto it = slots.find(seq_id);
        if (it == slots.end() || dst.size() < it->second.size()) {
            return SlotStoreStatus::Error(SlotStoreError::kBackendError, "buffer too small or missing seq");
        }
        if (short_get_data) {
            return SlotStoreStatus::Error(SlotStoreError::kBackendError, "injected zero-byte write");
        }
        std::memcpy(dst.data(), it->second.data(), it->second.size());
        out_written = it->second.size();
        return SlotStoreStatus::Ok();
    }

    SlotStoreStatus SetSequenceStateData(std::int32_t seq_id,
                                         std::span<const std::uint8_t> src) override {
        ++set_data_calls;
        if (seq_id < 0 || src.empty()) {
            return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "invalid set_data args");
        }
        if (fail_set_data) {
            return SlotStoreStatus::Error(SlotStoreError::kBackendError, "injected set_data failure");
        }
        slots[seq_id].assign(src.begin(), src.end());
        return SlotStoreStatus::Ok();
    }
};

void AppendLe32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFu));
}

void AppendLe64(std::vector<std::uint8_t>& buf, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFu));
}

void AppendGgufStr(std::vector<std::uint8_t>& buf, std::string_view s) {
    AppendLe64(buf, static_cast<std::uint64_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

void WriteLe32At(std::span<std::uint8_t> buf, std::size_t off, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) buf[off + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFu);
}

void WriteLe64At(std::span<std::uint8_t> buf, std::size_t off, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) buf[off + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFu);
}

// Builds a valid minimal GGUF v3 file with 1 string KV ("general.architecture" = arch),
// 1 uint32 KV ("llama.context_length" = n_ctx), and 1 1D F32 tensor ("weight", 64 floats = 256 bytes)
// padded to 32-byte alignment, returning the exact header/data-offset boundary.
std::size_t BuildSyntheticGgufWithWeights(const fs::path& path, std::string_view arch,
                                          std::uint32_t n_ctx, std::uint8_t weight_fill) {
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    AppendLe32(bytes, 3u);   // GGUF v3
    AppendLe64(bytes, 1u);   // n_tensors = 1
    AppendLe64(bytes, 2u);   // n_kv = 2

    // KV 0: general.architecture (GGUF_TYPE_STRING = 8)
    AppendGgufStr(bytes, "general.architecture");
    AppendLe32(bytes, 8u);
    AppendGgufStr(bytes, arch);

    // KV 1: llama.context_length (GGUF_TYPE_UINT32 = 4)
    AppendGgufStr(bytes, "llama.context_length");
    AppendLe32(bytes, 4u);
    AppendLe32(bytes, n_ctx);

    // Tensor 0: "weight", 1D [64], GGML_TYPE_F32 (= 0), offset = 0
    AppendGgufStr(bytes, "weight");
    AppendLe32(bytes, 1u);     // n_dims = 1
    AppendLe64(bytes, 64u);    // ne[0] = 64
    AppendLe32(bytes, 0u);     // GGML_TYPE_F32 = 0
    AppendLe64(bytes, 0u);     // tensor data offset = 0

    // Pad to GGUF_DEFAULT_ALIGNMENT (32)
    while ((bytes.size() % 32u) != 0u) {
        bytes.push_back(0u);
    }
    const std::size_t data_offset = bytes.size();
    bytes.insert(bytes.end(), 64u * sizeof(float), weight_fill);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(out.good());
    return data_offset;
}

std::vector<std::uint8_t> ReadAllBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in.good());
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteAllBytes(const fs::path& p, std::span<const std::uint8_t> bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    REQUIRE(out.good());
}

std::size_t CountTempFiles(const fs::path& dir) {
    std::size_t count = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".tmp") {
            ++count;
        }
    }
    return count;
}

SlotModelIdentity SampleModelIdentity() {
    SlotModelIdentity id;
    id.file_size = 17'179'869'184ULL + 4096ULL;  // > 16 GiB to exercise full 64-bit LE fields
    for (std::size_t i = 0; i < id.header_sha256.size(); ++i) {
        id.header_sha256[i] = static_cast<std::uint8_t>(0xA5u ^ (i * 7u));
    }
    return id;
}

}  // namespace

TEST_CASE("slot_store: header serialization round-trip and pre-allocation length validation") {
    const KvCacheType all_types[] = {
        KvCacheType::kF16,     KvCacheType::kF32,  KvCacheType::kBF16,  KvCacheType::kQ8_0,
        KvCacheType::kQ5_1,    KvCacheType::kQ5_0, KvCacheType::kQ4_1,  KvCacheType::kQ4_0,
        KvCacheType::kIQ4_NL,  KvCacheType::kF8_E4M3, KvCacheType::kQ2_0, KvCacheType::kPQ2_0,
    };

    for (const KvCacheType tk : all_types) {
        const KvCacheType tv = (tk == KvCacheType::kF16) ? KvCacheType::kQ8_0 : KvCacheType::kF16;
        SlotFileHeader hdr;
        hdr.model_identity = SampleModelIdentity();
        hdr.type_k = tk;
        hdr.type_v = tv;
        hdr.n_ctx = 32768u;
        hdr.token_count = 512u;
        hdr.token_table_bytes = 512u * sizeof(Token);
        hdr.state_bytes = 65536u;
        for (std::size_t i = 0; i < hdr.payload_sha256.size(); ++i) {
            hdr.payload_sha256[i] = static_cast<std::uint8_t>(i * 3u + 1u);
        }

        std::array<std::uint8_t, kSlotFileHeaderBytes> wire{};
        const SlotStoreStatus ser = SerializeSlotHeader(hdr, wire);
        REQUIRE_MESSAGE(ser.ok(), ser.message);
        CHECK( std::memcmp(wire.data(), kSlotFileMagic.data(), kSlotFileMagic.size()) == 0 );

        SlotFileHeader decoded{};
        const SlotStoreStatus des = DeserializeSlotHeader(wire, decoded);
        REQUIRE_MESSAGE(des.ok(), des.message);
        CHECK(decoded == hdr);
    }

    // Also round-trips cleanly when token_table_bytes == 0 (count-only metadata).
    SlotFileHeader base;
    base.model_identity = SampleModelIdentity();
    base.type_k = KvCacheType::kQ8_0;
    base.type_v = KvCacheType::kQ8_0;
    base.n_ctx = 4096u;
    base.token_count = 17u;
    base.token_table_bytes = 0u;
    base.state_bytes = 1024u;
    std::array<std::uint8_t, kSlotFileHeaderBytes> valid_wire{};
    REQUIRE(SerializeSlotHeader(base, valid_wire).ok());

    SlotFileHeader out{};
    // Truncated header buffer (< 128 bytes).
    CHECK(DeserializeSlotHeader(std::span<const std::uint8_t>(valid_wire.data(), 64), out).code ==
          SlotStoreError::kTruncatedFile);

    // Corrupted magic.
    auto bad_magic = valid_wire;
    bad_magic[0] = 'X';
    CHECK(DeserializeSlotHeader(bad_magic, out).code == SlotStoreError::kBadMagic);

    // Unsupported format_version (0, 2, UINT32_MAX).
    for (const std::uint32_t bad_ver : {0u, 2u, 0xFFFFFFFFu}) {
        auto bad = valid_wire;
        WriteLe32At(bad, 8, bad_ver);
        CHECK(DeserializeSlotHeader(bad, out).code == SlotStoreError::kUnsupportedVersion);
    }

    // Wrong header_bytes field.
    {
        auto bad = valid_wire;
        WriteLe32At(bad, 12, 64u);
        CHECK(DeserializeSlotHeader(bad, out).code == SlotStoreError::kInvalidHeader);
    }

    // Non-zero reserved field.
    {
        auto bad = valid_wire;
        WriteLe32At(bad, 68, 1u);
        CHECK(DeserializeSlotHeader(bad, out).code == SlotStoreError::kInvalidHeader);
    }

    // Unknown K/V wire codes.
    {
        auto bad_k = valid_wire;
        WriteLe32At(bad_k, 56, 999u);
        CHECK(DeserializeSlotHeader(bad_k, out).code == SlotStoreError::kInvalidHeader);

        auto bad_v = valid_wire;
        WriteLe32At(bad_v, 60, 0xFFFFFFFFu);
        CHECK(DeserializeSlotHeader(bad_v, out).code == SlotStoreError::kInvalidHeader);
    }

    // Zero model_file_size or invalid n_ctx or token_count > n_ctx.
    {
        auto zero_model = valid_wire;
        WriteLe64At(zero_model, 16, 0u);
        CHECK(DeserializeSlotHeader(zero_model, out).code == SlotStoreError::kInvalidHeader);

        auto zero_ctx = valid_wire;
        WriteLe32At(zero_ctx, 64, 0u);
        CHECK(DeserializeSlotHeader(zero_ctx, out).code == SlotStoreError::kInvalidHeader);

        auto huge_ctx = valid_wire;
        WriteLe32At(huge_ctx, 64, kMaxSlotContextTokens + 1u);
        CHECK(DeserializeSlotHeader(huge_ctx, out).code == SlotStoreError::kInvalidHeader);

        auto tok_over_ctx = valid_wire;
        WriteLe64At(tok_over_ctx, 72, 4097u);  // > n_ctx (4096)
        CHECK(DeserializeSlotHeader(tok_over_ctx, out).code == SlotStoreError::kInvalidHeader);

        auto tok_max = valid_wire;
        WriteLe64At(tok_max, 72, std::numeric_limits<std::uint64_t>::max());
        CHECK(DeserializeSlotHeader(tok_max, out).code == SlotStoreError::kInvalidHeader);
    }

    // Mismatched or overflowed token_table_bytes and state_bytes.
    {
        auto bad_tok_bytes = valid_wire;
        WriteLe64At(bad_tok_bytes, 80, 17u * sizeof(Token) + 4u);
        CHECK(DeserializeSlotHeader(bad_tok_bytes, out).code == SlotStoreError::kInvalidHeader);

        auto zero_state = valid_wire;
        WriteLe64At(zero_state, 88, 0u);
        CHECK(DeserializeSlotHeader(zero_state, out).code == SlotStoreError::kInvalidHeader);

        auto max_state = valid_wire;
        WriteLe64At(max_state, 88, std::numeric_limits<std::uint64_t>::max());
        CHECK(DeserializeSlotHeader(max_state, out).code == SlotStoreError::kFileTooLarge);

        SlotStoreLimits tight;
        tight.max_payload_bytes = 512u;
        tight.max_file_bytes = kSlotFileHeaderBytes + 512u;
        CHECK(DeserializeSlotHeader(valid_wire, out, tight).code == SlotStoreError::kFileTooLarge);
    }
}

TEST_CASE("slot_store: GGUF model identity hashes header/metadata region only, not weight tail") {
    ScopedTempDir tmp;
    const fs::path gguf_a = tmp.dir / "model_a.gguf";
    const fs::path gguf_b = tmp.dir / "model_b_diff_weights.gguf";
    const fs::path gguf_c = tmp.dir / "model_c_diff_meta.gguf";

    const std::size_t expected_hdr_len = BuildSyntheticGgufWithWeights(gguf_a, "llama", 4096u, 0x11u);
    CHECK(BuildSyntheticGgufWithWeights(gguf_b, "llama", 4096u, 0xEEu) == expected_hdr_len);
    (void)BuildSyntheticGgufWithWeights(gguf_c, "llama", 8192u, 0x11u);

    SlotModelIdentity id_a{};
    std::uint64_t hashed_bytes_a = 0;
    const SlotStoreStatus st_a = ComputeGgufModelIdentity(gguf_a, id_a, kDefaultMaxGgufHeaderBytes, &hashed_bytes_a);
    REQUIRE_MESSAGE(st_a.ok(), st_a.message);
    CHECK(hashed_bytes_a == expected_hdr_len);

    const auto raw_a = ReadAllBytes(gguf_a);
    CHECK(id_a.file_size == raw_a.size());
    CHECK(hashed_bytes_a < id_a.file_size);

    // Cross-check SHA-256 of [0, expected_hdr_len) against Sonder's reference Sha256.
    sonder::inference::detail::Sha256 ref_sha;
    ref_sha.update(std::string_view(reinterpret_cast<const char*>(raw_a.data()), expected_hdr_len));
    CHECK(id_a.header_sha256 == ref_sha.finish());

    // Changing ONLY the tensor weight bytes (keeping file_size and GGUF header identical)
    // leaves header_sha256 unchanged, proving the whole file was NOT hashed.
    SlotModelIdentity id_b{};
    REQUIRE(ComputeGgufModelIdentity(gguf_b, id_b).ok());
    CHECK(id_b == id_a);

    // Changing metadata ("llama.context_length" 4096 -> 8192) with the exact same file_size
    // changes header_sha256.
    SlotModelIdentity id_c{};
    REQUIRE(ComputeGgufModelIdentity(gguf_c, id_c).ok());
    CHECK(id_c.file_size == id_a.file_size);
    CHECK(id_c.header_sha256 != id_a.header_sha256);
    CHECK(id_c != id_a);

    // Changing total file_size (appending weight padding) changes model identity.
    auto padded = raw_a;
    padded.insert(padded.end(), 32u, 0x00u);
    const fs::path gguf_padded = tmp.dir / "model_padded.gguf";
    WriteAllBytes(gguf_padded, padded);
    SlotModelIdentity id_padded{};
    REQUIRE(ComputeGgufModelIdentity(gguf_padded, id_padded).ok());
    CHECK(id_padded.header_sha256 == id_a.header_sha256);
    CHECK(id_padded.file_size != id_a.file_size);
    CHECK(id_padded != id_a);

#ifdef SONDER_LLAMACPP_VOCAB_DIR
    // Bundled vocab-only GGUF fixture also produces a valid identity.
    const fs::path vocab_gguf = fs::path(SONDER_LLAMACPP_VOCAB_DIR) / "ggml-vocab-llama-spm.gguf";
    SlotModelIdentity vocab_id{};
    std::uint64_t vocab_hdr_bytes = 0;
    const SlotStoreStatus vst =
        ComputeGgufModelIdentity(vocab_gguf, vocab_id, kDefaultMaxGgufHeaderBytes, &vocab_hdr_bytes);
    REQUIRE_MESSAGE(vst.ok(), vst.message);
    CHECK(vocab_id.file_size > 0u);
    CHECK(vocab_hdr_bytes == vocab_id.file_size);
#endif

    // Invalid / missing / non-GGUF / oversized-header files fail closed.
    SlotModelIdentity dummy{};
    CHECK(ComputeGgufModelIdentity(tmp.dir / "missing.gguf", dummy).code == SlotStoreError::kInvalidGguf);
    CHECK(ComputeGgufModelIdentity(tmp.dir, dummy).code == SlotStoreError::kInvalidPath);

    const fs::path not_gguf = tmp.dir / "bad.gguf";
    const std::array<std::uint8_t, 32> bad_bytes{'N', 'O', 'P', 'E'};
    WriteAllBytes(not_gguf, bad_bytes);
    CHECK(ComputeGgufModelIdentity(not_gguf, dummy).code == SlotStoreError::kInvalidGguf);

    // Header cap smaller than actual GGUF header refuses cleanly.
    CHECK(ComputeGgufModelIdentity(gguf_a, dummy, /*max_header_bytes=*/32u).code == SlotStoreError::kInvalidGguf);
}

TEST_CASE("slot_store: save and restore round-trip, atomic overwrite, and failed-save preservation") {
    ScopedTempDir tmp;
    const fs::path slot_path = tmp.dir / "slot_0.bin";

    FakeSlotStateBackend src_backend;
    src_backend.slots[0] = {0x10, 0x20, 0x30, 0x40, 0x55, 0xAA, 0xFE, 0x01, 0x99};
    src_backend.slots[2] = {0xDE, 0xAD, 0xBE, 0xEF};

    SlotSaveMetadata save_meta;
    save_meta.model_identity = SampleModelIdentity();
    save_meta.type_k = KvCacheType::kQ8_0;
    save_meta.type_v = KvCacheType::kQ4_0;
    save_meta.n_ctx = 8192u;
    save_meta.tokens = {1, 42, 1024, 0, 31999};
    save_meta.token_count = save_meta.tokens.size();

    SlotFileHeader written_hdr{};
    const SlotStoreStatus sv = SaveSlotFile(slot_path, 0, save_meta, src_backend, {}, &written_hdr);
    REQUIRE_MESSAGE(sv.ok(), sv.message);
    CHECK(fs::exists(slot_path));
    CHECK(CountTempFiles(tmp.dir) == 0u);
    CHECK(written_hdr.token_count == 5u);
    CHECK(written_hdr.state_bytes == src_backend.slots[0].size());

    FakeSlotStateBackend dst_backend;
    dst_backend.slots[2] = {0x77, 0x88};  // unrelated sequence must stay untouched

    SlotRestoreExpectation exp;
    exp.model_identity = save_meta.model_identity;
    exp.type_k = save_meta.type_k;
    exp.type_v = save_meta.type_v;
    exp.n_ctx = save_meta.n_ctx;
    exp.expected_token_count = 5u;

    SlotRestoreResult restored{};
    const SlotStoreStatus rs = RestoreSlotFile(slot_path, /*dest_seq_id=*/1, exp, dst_backend, &restored);
    REQUIRE_MESSAGE(rs.ok(), rs.message);
    CHECK(restored.header == written_hdr);
    CHECK(restored.tokens == save_meta.tokens);
    CHECK(restored.restored_state_bytes == src_backend.slots[0].size());
    CHECK(dst_backend.slots[1] == src_backend.slots[0]);
    CHECK(dst_backend.slots[2] == std::vector<std::uint8_t>{0x77, 0x88});

    // Atomic overwrite onto the same path with count-only metadata (no token ID table).
    SlotSaveMetadata save_meta2 = save_meta;
    save_meta2.tokens.clear();
    save_meta2.token_count = 64u;
    const SlotStoreStatus sv2 = SaveSlotFile(slot_path, 2, save_meta2, src_backend);
    REQUIRE_MESSAGE(sv2.ok(), sv2.message);
    CHECK(CountTempFiles(tmp.dir) == 0u);

    exp.expected_token_count = std::nullopt;  // accept any valid token_count <= n_ctx
    SlotRestoreResult restored2{};
    REQUIRE(RestoreSlotFile(slot_path, 0, exp, dst_backend, &restored2).ok());
    CHECK(restored2.header.token_count == 64u);
    CHECK(restored2.tokens.empty());
    CHECK(dst_backend.slots[0] == src_backend.slots[2]);

    // A failed save (backend error or payload cap exceeded) must NOT corrupt the existing file
    // and must leave zero temporary files behind.
    src_backend.fail_get_data = true;
    CHECK(SaveSlotFile(slot_path, 0, save_meta, src_backend).code == SlotStoreError::kBackendError);
    src_backend.fail_get_data = false;
    CHECK(CountTempFiles(tmp.dir) == 0u);

    SlotStoreLimits tiny_limit;
    tiny_limit.max_payload_bytes = 2u;
    tiny_limit.max_file_bytes = kSlotFileHeaderBytes + 2u;
    CHECK(SaveSlotFile(slot_path, 0, save_meta, src_backend, tiny_limit).code == SlotStoreError::kFileTooLarge);
    CHECK(CountTempFiles(tmp.dir) == 0u);

    // Previous file at `slot_path` still restores cleanly after failed overwrites.
    SlotRestoreResult after_failed_overwrite{};
    REQUIRE(RestoreSlotFile(slot_path, 3, exp, dst_backend, &after_failed_overwrite).ok());
    CHECK(dst_backend.slots[3] == src_backend.slots[2]);
}

TEST_CASE("slot_store: every mismatch is refused with a typed error before touching backend state") {
    ScopedTempDir tmp;
    const fs::path slot_path = tmp.dir / "valid.slot";

    FakeSlotStateBackend backend;
    const std::vector<std::uint8_t> original_slot0{0x01, 0x02, 0x03, 0x04};
    backend.slots[0] = {0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78};

    SlotSaveMetadata meta;
    meta.model_identity = SampleModelIdentity();
    meta.type_k = KvCacheType::kQ8_0;
    meta.type_v = KvCacheType::kF16;
    meta.n_ctx = 4096u;
    meta.tokens = {10, 20, 30};

    REQUIRE(SaveSlotFile(slot_path, 0, meta, backend).ok());
    const auto pristine_file = ReadAllBytes(slot_path);

    backend.slots[0] = original_slot0;
    backend.set_data_calls = 0;

    SlotRestoreExpectation base_exp;
    base_exp.model_identity = meta.model_identity;
    base_exp.type_k = meta.type_k;
    base_exp.type_v = meta.type_v;
    base_exp.n_ctx = meta.n_ctx;
    base_exp.expected_token_count = 3u;

    // 1. Magic mismatch on disk.
    {
        auto corrupted = pristine_file;
        corrupted[0] ^= 0xFFu;
        const fs::path p = tmp.dir / "bad_magic.slot";
        WriteAllBytes(p, corrupted);
        CHECK(RestoreSlotFile(p, 0, base_exp, backend).code == SlotStoreError::kBadMagic);
        CHECK(backend.set_data_calls == 0u);
    }

    // 2. Format version mismatch on disk.
    {
        auto corrupted = pristine_file;
        WriteLe32At(corrupted, 8, kSlotFormatVersion + 1u);
        const fs::path p = tmp.dir / "bad_ver.slot";
        WriteAllBytes(p, corrupted);
        CHECK(RestoreSlotFile(p, 0, base_exp, backend).code == SlotStoreError::kUnsupportedVersion);
        CHECK(backend.set_data_calls == 0u);
    }

    // 3. Model file size mismatch.
    {
        SlotRestoreExpectation wrong = base_exp;
        wrong.model_identity.file_size += 1u;
        CHECK(RestoreSlotFile(slot_path, 0, wrong, backend).code == SlotStoreError::kModelMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    // 4. Model GGUF header SHA-256 mismatch.
    {
        SlotRestoreExpectation wrong = base_exp;
        wrong.model_identity.header_sha256[31] ^= 0x01u;
        CHECK(RestoreSlotFile(slot_path, 0, wrong, backend).code == SlotStoreError::kModelMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    // 5. K cache dtype mismatch.
    {
        SlotRestoreExpectation wrong = base_exp;
        wrong.type_k = KvCacheType::kF16;
        CHECK(RestoreSlotFile(slot_path, 0, wrong, backend).code == SlotStoreError::kKvTypeMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    // 6. V cache dtype mismatch.
    {
        SlotRestoreExpectation wrong = base_exp;
        wrong.type_v = KvCacheType::kQ8_0;
        CHECK(RestoreSlotFile(slot_path, 0, wrong, backend).code == SlotStoreError::kKvTypeMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    // 7. Context length (n_ctx) mismatch.
    {
        SlotRestoreExpectation wrong = base_exp;
        wrong.n_ctx = 8192u;
        CHECK(RestoreSlotFile(slot_path, 0, wrong, backend).code == SlotStoreError::kContextLengthMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    // 8. Token count mismatch (when expected_token_count is specified).
    {
        SlotRestoreExpectation wrong = base_exp;
        wrong.expected_token_count = 4u;
        CHECK(RestoreSlotFile(slot_path, 0, wrong, backend).code == SlotStoreError::kTokenCountMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    CHECK(backend.slots[0] == original_slot0);
}

TEST_CASE("slot_store: truncated, oversized, trailing-garbage, and corrupted-checksum files refused") {
    ScopedTempDir tmp;
    const fs::path valid_path = tmp.dir / "valid.slot";

    FakeSlotStateBackend backend;
    backend.slots[0].resize(512);
    for (std::size_t i = 0; i < backend.slots[0].size(); ++i) {
        backend.slots[0][i] = static_cast<std::uint8_t>((i * 13u + 7u) & 0xFFu);
    }

    SlotSaveMetadata meta;
    meta.model_identity = SampleModelIdentity();
    meta.type_k = KvCacheType::kF16;
    meta.type_v = KvCacheType::kF16;
    meta.n_ctx = 2048u;
    meta.tokens = {7, 8, 9, 10};
    REQUIRE(SaveSlotFile(valid_path, 0, meta, backend).ok());

    const auto pristine = ReadAllBytes(valid_path);
    const std::vector<std::uint8_t> sentinel{0xAA, 0xBB};
    backend.slots[0] = sentinel;
    backend.set_data_calls = 0;

    SlotRestoreExpectation exp;
    exp.model_identity = meta.model_identity;
    exp.type_k = meta.type_k;
    exp.type_v = meta.type_v;
    exp.n_ctx = meta.n_ctx;

    // Empty file, mid-header truncation, and 1-byte-short payload truncation.
    const std::array<std::size_t, 5> trunc_lens{
        0u, 17u, 127u, static_cast<std::size_t>(kSlotFileHeaderBytes), pristine.size() - 1u};
    for (const std::size_t trunc_len : trunc_lens) {
        CAPTURE(trunc_len);
        const fs::path p = tmp.dir / ("trunc_" + std::to_string(trunc_len) + ".slot");
        WriteAllBytes(p, std::span<const std::uint8_t>(pristine.data(), trunc_len));
        CHECK(RestoreSlotFile(p, 0, exp, backend).code == SlotStoreError::kTruncatedFile);
        CHECK(backend.set_data_calls == 0u);
    }

    // Oversized file on disk vs configured max_file_bytes / max_payload_bytes.
    {
        SlotStoreLimits small_file_cap;
        small_file_cap.max_file_bytes = pristine.size() - 1u;
        CHECK(RestoreSlotFile(valid_path, 0, exp, backend, nullptr, small_file_cap).code ==
              SlotStoreError::kFileTooLarge);

        SlotStoreLimits small_payload_cap;
        small_payload_cap.max_payload_bytes = 256u;
        CHECK(RestoreSlotFile(valid_path, 0, exp, backend, nullptr, small_payload_cap).code ==
              SlotStoreError::kFileTooLarge);
        CHECK(backend.set_data_calls == 0u);
    }

    // Hostile 128-byte header declaring huge 1 EiB state_bytes (must fail before allocating).
    {
        std::vector<std::uint8_t> hostile_hdr(pristine.begin(), pristine.begin() + kSlotFileHeaderBytes);
        WriteLe64At(hostile_hdr, 88, std::uint64_t{1} << 60);
        const fs::path p = tmp.dir / "hostile_huge_len.slot";
        WriteAllBytes(p, hostile_hdr);
        CHECK(RestoreSlotFile(p, 0, exp, backend).code == SlotStoreError::kFileTooLarge);
        CHECK(backend.set_data_calls == 0u);
    }

    // Trailing unauthenticated byte appended past declared payload.
    {
        auto trailing = pristine;
        trailing.push_back(0x00u);
        const fs::path p = tmp.dir / "trailing.slot";
        WriteAllBytes(p, trailing);
        CHECK(RestoreSlotFile(p, 0, exp, backend).code == SlotStoreError::kInvalidHeader);
        CHECK(backend.set_data_calls == 0u);
    }

    // Corrupted checksum cases: flip bit in token table, first state byte, last state byte, and header digest.
    for (const std::size_t flip_offset :
         {std::size_t{kSlotFileHeaderBytes},                                  // token table byte
          std::size_t{kSlotFileHeaderBytes + meta.tokens.size() * 4u},        // first state byte
          pristine.size() - 1u,                                               // final state byte
          std::size_t{96}}) {                                                 // payload_sha256 in header
        CAPTURE(flip_offset);
        auto bad = pristine;
        bad[flip_offset] ^= 0x80u;
        const fs::path p = tmp.dir / ("bad_crc_" + std::to_string(flip_offset) + ".slot");
        WriteAllBytes(p, bad);
        CHECK(RestoreSlotFile(p, 0, exp, backend).code == SlotStoreError::kChecksumMismatch);
        CHECK(backend.set_data_calls == 0u);
    }

    CHECK(backend.slots[0] == sentinel);
}

TEST_CASE("slot_store: path confinement and llama.cpp v1.0.0 adapter preconditions") {
    ScopedTempDir tmp;

    CHECK(ValidateSlotFilePath(fs::path{}).code == SlotStoreError::kInvalidPath);
    CHECK(ValidateSlotFilePath(tmp.dir / ".." / "escape.slot").code == SlotStoreError::kInvalidPath);
    CHECK(ValidateSlotFilePath(tmp.dir).code == SlotStoreError::kInvalidPath);  // directory

    fs::path utf8_out;
    const std::string with_nul("bad\0name.slot", 13);
    CHECK(ValidateSlotFilePathUtf8(with_nul, utf8_out).code == SlotStoreError::kInvalidPath);

    fs::path resolved;
    REQUIRE(ResolveSlotFilePath(tmp.dir, "session_a.slot", resolved).ok());
    CHECK(resolved.parent_path() == fs::weakly_canonical(tmp.dir));
    for (std::string_view bad_name :
         {std::string_view{}, std::string_view{"."}, std::string_view{".."},
          std::string_view{"../escape.slot"}, std::string_view{"sub/a.slot"},
          std::string_view{"sub\\a.slot"}, std::string_view{"C:escape.slot"}}) {
        CAPTURE(bad_name);
        CHECK(ResolveSlotFilePath(tmp.dir, bad_name, resolved).code == SlotStoreError::kInvalidPath);
    }

    // LlamaCppSlotStateBackend rejects null context and negative seq_id cleanly.
    LlamaCppSlotStateBackend null_be(nullptr);
    std::size_t sz = 99u;
    CHECK(null_be.GetSequenceStateSize(0, sz).code == SlotStoreError::kInvalidArgument);
    CHECK(sz == 0u);
    std::array<std::uint8_t, 8> buf{};
    std::size_t wrote = 99u;
    CHECK(null_be.GetSequenceStateData(0, buf, wrote).code == SlotStoreError::kInvalidArgument);
    CHECK(wrote == 0u);
    CHECK(null_be.SetSequenceStateData(0, buf).code == SlotStoreError::kInvalidArgument);

    // Negative seq_id, negative token IDs, and extreme limits are rejected before I/O.
    FakeSlotStateBackend fake_be;
    fake_be.slots[0] = {0x11, 0x22, 0x33, 0x44};
    SlotSaveMetadata bad_tok_meta;
    bad_tok_meta.model_identity = SampleModelIdentity();
    bad_tok_meta.n_ctx = 1024u;
    bad_tok_meta.tokens = {1, -1, 2};
    CHECK(SaveSlotFile(tmp.dir / "neg_tok.slot", 0, bad_tok_meta, fake_be).code ==
          SlotStoreError::kInvalidArgument);
    CHECK(SaveSlotFile(tmp.dir / "neg_seq.slot", -1, bad_tok_meta, fake_be).code ==
          SlotStoreError::kInvalidArgument);
    SlotStoreLimits overflow_lim;
    overflow_lim.max_payload_bytes = std::numeric_limits<std::size_t>::max();
    CHECK(SaveSlotFile(tmp.dir / "bad_lim.slot", 0, bad_tok_meta, fake_be, overflow_lim).code ==
          SlotStoreError::kInvalidArgument);

    // Forged slot file with valid SHA-256 checksum over a negative token ID is refused before set_data.
    {
        SlotSaveMetadata ok_meta;
        ok_meta.model_identity = SampleModelIdentity();
        ok_meta.n_ctx = 1024u;
        ok_meta.tokens = {1, 2, 3};
        const fs::path forged_path = tmp.dir / "forged_neg_tok.slot";
        REQUIRE(SaveSlotFile(forged_path, 0, ok_meta, fake_be).ok());
        auto forged = ReadAllBytes(forged_path);
        WriteLe32At(forged, kSlotFileHeaderBytes + 4u, 0xFFFFFFFFu);  // token[1] = -1
        sonder::inference::detail::Sha256 sha;
        sha.update(std::string_view(reinterpret_cast<const char*>(forged.data() + kSlotFileHeaderBytes),
                                    forged.size() - kSlotFileHeaderBytes));
        const auto digest = sha.finish();
        std::memcpy(forged.data() + 96u, digest.data(), 32u);
        WriteAllBytes(forged_path, forged);

        SlotRestoreExpectation exp_ok;
        exp_ok.model_identity = ok_meta.model_identity;
        exp_ok.n_ctx = ok_meta.n_ctx;
        fake_be.set_data_calls = 0;
        CHECK(RestoreSlotFile(forged_path, 0, exp_ok, fake_be).code == SlotStoreError::kInvalidHeader);
        CHECK(fake_be.set_data_calls == 0u);
    }

    SlotSaveMetadata meta;
    meta.model_identity = SampleModelIdentity();
    meta.n_ctx = 1024u;
    meta.token_count = 4u;
    CHECK(SaveLlamaSequenceSlot(tmp.dir / "null_ctx.slot", nullptr, 0, meta).code ==
          SlotStoreError::kInvalidArgument);

    SlotRestoreExpectation exp;
    exp.model_identity = meta.model_identity;
    exp.n_ctx = 1024u;
    CHECK(RestoreLlamaSequenceSlot(tmp.dir / "null_ctx.slot", nullptr, 0, exp).code ==
          SlotStoreError::kInvalidArgument);
}

TEST_CASE("slot_store: malformed UTF-8, control chars, trailing dots/spaces, and reserved Win32 device names fail closed") {
    ScopedTempDir tmp;

    // 1. Malformed UTF-8 byte sequences (lone continuations, overlongs, UTF-16 surrogates,
    //    > U+10FFFF, truncated multi-byte) must return kInvalidPath without throwing.
    const std::string bad_utf8_cases[] = {
        std::string("\xFF\xFE\xFD\x80.slot", 9),
        std::string("\x80.slot", 6),
        std::string("\xC0\xAF.slot", 7),              // overlong '/'
        std::string("\xE0\x80\xAF.slot", 8),          // 3-byte overlong
        std::string("\xED\xA0\x80.slot", 8),          // UTF-16 high surrogate U+D800
        std::string("\xED\xBF\xBF.slot", 8),          // UTF-16 low surrogate U+DFFF
        std::string("\xF4\x90\x80\x80.slot", 9),      // > U+10FFFF
        std::string("\xE2\x82", 2),                   // truncated 3-byte sequence
        std::string("\xF0\x9F\x98", 3),               // truncated 4-byte sequence
    };
    for (const std::string& bad : bad_utf8_cases) {
        fs::path out_p;
        SlotStoreStatus s1 = SlotStoreStatus::Ok();
        SlotStoreStatus s2 = SlotStoreStatus::Ok();
        CHECK_NOTHROW(s1 = ValidateSlotFilePathUtf8(bad, out_p));
        CHECK(s1.code == SlotStoreError::kInvalidPath);
        CHECK(out_p.empty());

        CHECK_NOTHROW(s2 = ResolveSlotFilePath(tmp.dir, bad, out_p));
        CHECK(s2.code == SlotStoreError::kInvalidPath);
        CHECK(out_p.empty());
    }

    // Valid non-ASCII UTF-8 filename succeeds cleanly in both helpers.
    {
        fs::path out_p;
        const std::string valid_utf8 = "caf\xC3\xA9_\xE6\x97\xA5\xE6\x9C\xAC_\xF0\x9F\x9A\x80.slot";
        SlotStoreStatus r1 = SlotStoreStatus::Error(SlotStoreError::kInvalidPath, "init");
        CHECK_NOTHROW(r1 = ResolveSlotFilePath(tmp.dir, valid_utf8, out_p));
        REQUIRE_MESSAGE(r1.ok(), r1.message);

        fs::path out_p2;
        const std::u8string u8_dir = tmp.dir.u8string();
        const std::string dir_utf8(u8_dir.begin(), u8_dir.end());
        SlotStoreStatus r2 = SlotStoreStatus::Error(SlotStoreError::kInvalidPath, "init");
        CHECK_NOTHROW(r2 = ValidateSlotFilePathUtf8(dir_utf8 + "/" + valid_utf8, out_p2));
        REQUIRE_MESSAGE(r2.ok(), r2.message);
    }

    // 2. ASCII control chars (< 0x20 and 0x7F), NTFS ADS colons, wildcards, and trailing dots/spaces.
    const std::string bad_component_cases[] = {
        std::string("tab\tname.slot"),
        std::string("nl\nname.slot"),
        std::string("cr\rname.slot"),
        std::string("ctrl\x01name.slot", 14),
        std::string("del\x7Fname.slot", 12),
        std::string("slot.bin."),
        std::string("slot.bin "),
        std::string("slot.bin.."),
        std::string("slot.bin ."),
        std::string("slot<bad>.slot"),
        std::string("slot|bad.slot"),
        std::string("slot?bad.slot"),
        std::string("slot*bad.slot"),
        std::string("slot\"bad.slot"),
        std::string("slot.bin:ads"),
    };
    for (const std::string& bad : bad_component_cases) {
        CAPTURE(bad);
        fs::path out_p;
        CHECK(ResolveSlotFilePath(tmp.dir, bad, out_p).code == SlotStoreError::kInvalidPath);
        CHECK(ValidateSlotFilePath(tmp.dir / bad).code == SlotStoreError::kInvalidPath);
    }

    // 3. Windows reserved device names (case-insensitive, bare, with extension, or trailing space before dot).
    const char* reserved_names[] = {
        "CON", "con", "PRN", "prn", "AUX", "aux", "NUL", "nul",
        "COM1", "com1", "COM9", "LPT1", "lpt1", "LPT9", "COM0", "LPT0",
        "CON.slot", "nul.bin", "Aux.tar.gz", "com4.slot", "LPT2.txt",
        "CON .slot", "NUL  .slot",
        "COM\xC2\xB9.slot", "LPT\xC2\xB2", "com\xC2\xB3",
    };
    for (const char* dev : reserved_names) {
        CAPTURE(dev);
        fs::path out_p;
        CHECK(ResolveSlotFilePath(tmp.dir, dev, out_p).code == SlotStoreError::kInvalidPath);
        const std::string_view sv(dev);
        const std::u8string u8_dev(sv.begin(), sv.end());
        CHECK(ValidateSlotFilePath(tmp.dir / fs::path(u8_dev)).code == SlotStoreError::kInvalidPath);
    }

    // Non-reserved prefixes ("CONSOLE.slot", "NULLIFY.slot", "COM10.slot") remain allowed.
    for (const char* allowed : {"CONSOLE.slot", "NULLIFY.slot", "COM10.slot", "LPT10.slot"}) {
        CAPTURE(allowed);
        fs::path out_p;
        CHECK(ResolveSlotFilePath(tmp.dir, allowed, out_p).ok());
        CHECK(ValidateSlotFilePath(tmp.dir / allowed).ok());
    }

#if defined(_WIN32)
    // Unpaired UTF-16 surrogate in raw fs::path must also return kInvalidPath without throwing.
    {
        const std::wstring bad_wide = {L'b', L'a', L'd', static_cast<wchar_t>(0xD800), L'.', L's'};
        const fs::path bad_wpath = tmp.dir / fs::path(bad_wide);
        SlotStoreStatus ws = SlotStoreStatus::Ok();
        CHECK_NOTHROW(ws = ValidateSlotFilePath(bad_wpath));
        CHECK(ws.code == SlotStoreError::kInvalidPath);
    }
#endif
}
