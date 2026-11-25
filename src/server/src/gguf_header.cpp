// GGUF header reader (versions 2 and 3), sliding-window architecture
// resolution, and bounded artifact observation. Reads only the header:
// metadata and the tensor table, never tensor data.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <map>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gguf_artifact_observation.hpp"
#include "sha256.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "vram_internal.hpp"

namespace sonder::inference {

namespace {

using detail::excerpt;

constexpr std::uint64_t kMaxCount = 1ull << 24;        // KV pairs, tensors, array items
constexpr std::uint64_t kMaxStringBytes = 1ull << 24;  // one string (chat templates are ~20 KiB)
// general.architecture is a short identifier ("qwen35", "llama"); it is kept
// and quoted in messages, so a longer or non-printable one is refused.
constexpr std::size_t kMaxArchitectureBytes = 256;
// Recurrent (ssm.*) sizes from the file. Real models are far below these
// (Qwen3.8-27B: conv 4, inner 6144, state 128, groups 16); within them the
// state size cannot overflow, and a row stays under 2^28 f32 values (1 GiB).
constexpr std::uint64_t kMaxSsmConvKernel = 1u << 10;
constexpr std::uint64_t kMaxSsmSize = 1u << 20;  // inner, state, groups
constexpr std::uint64_t kMaxRecurrentStateValues = 1ull << 28;

Status malformed(const std::string& what) { return Status(ErrorCode::invalid_argument, "GGUF header: " + what); }

bool checked_add(std::uint64_t& dst, std::uint64_t add) noexcept {
    if (add > std::numeric_limits<std::uint64_t>::max() - dst) return false;
    dst += add;
    return true;
}

bool printable(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](char ch) { return ch >= 0x20 && ch < 0x7f; });
}

// ggml tensor types: (elements per block, bytes per block). ggml.h / ggml.c
// (upstream ggml-org/llama.cpp + PrismML-Eng/llama.cpp `prism` branch).
bool ggml_type_size(std::uint32_t type, std::uint64_t& block_elems, std::uint64_t& block_bytes) {
    struct Entry {
        std::uint32_t type;
        std::uint64_t elems;
        std::uint64_t bytes;
    };
    static constexpr Entry kTypes[] = {
        {0, 1, 4},      {1, 1, 2},      {2, 32, 18},    {3, 32, 20},    {6, 32, 22},    {7, 32, 24},
        {8, 32, 34},    {9, 32, 36},    {10, 256, 84},  {11, 256, 110}, {12, 256, 144}, {13, 256, 176},
        {14, 256, 210}, {15, 256, 292}, {16, 256, 66},  {17, 256, 74},  {18, 256, 98},  {19, 256, 50},
        {20, 32, 18},   {21, 256, 110}, {22, 256, 82},  {23, 256, 136}, {24, 1, 1},     {25, 1, 2},
        {26, 1, 4},     {27, 1, 8},     {28, 1, 8},     {29, 256, 56},  {30, 1, 2},     {34, 256, 54},
        {35, 256, 66},  {39, 32, 17},
        // Mainline llama.cpp FP4/1-bit/2-bit-ternary/FP8 types:
        {40, 64, 36},   // GGML_TYPE_NVFP4   (Blackwell FP4: 64 elems, 4x UE4M3 scales + 32 B data = 36 B)
        {41, 128, 18},  // GGML_TYPE_Q1_0    (1-bit Bonsai-1: 128 elems, fp16 scale + 16 B data = 18 B)
        {42, 64, 18},   // GGML_TYPE_Q2_0    (mainline 2-bit ternary: 64 elems, fp16 scale + 16 B data = 18 B)
        {43, 1, 1},     // GGML_TYPE_F8_E4M3 (Ada/Blackwell/RDNA4 FP8: 1 elem, 1 B)
        // PrismML-Eng/llama.cpp (`prism` branch) Bonsai-2 ternary & KV types:
        {142, 128, 34}, // GGML_TYPE_PQ2_0   (PrismML Bonsai-2: 128 elems, fp16 scale + 32 B data = 34 B)
        {143, 128, 28}, // GGML_TYPE_PTQ1_0  (PrismML Bonsai-2 dense ternary: 128 elems, fp16 scale + 26 B = 28 B)
        {145, 128, 36}, // GGML_TYPE_PQ2_0M  (PrismML KV-only: 128 elems, fp16 scale + fp16 min + 32 B = 36 B)
    };
    for (const auto& e : kTypes) {
        if (e.type == type) {
            block_elems = e.elems;
            block_bytes = e.bytes;
            return true;
        }
    }
    return false;
}

std::string digest_to_hex(const std::array<std::uint8_t, 32>& bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const std::uint8_t byte : bytes) {
        out.push_back(kHex[byte >> 4]);
        out.push_back(kHex[byte & 0x0Fu]);
    }
    return out;
}

std::filesystem::path path_from_utf8(std::string_view s) {
    std::u8string u;
    u.reserve(s.size());
    for (const char ch : s) {
        u.push_back(static_cast<char8_t>(static_cast<unsigned char>(ch)));
    }
    return std::filesystem::path(std::move(u));
}

// general.file_type (llama_ftype from llama.h) to canonical quantization label.
std::string_view gguf_file_type_name(std::uint32_t ftype) noexcept {
    switch (ftype) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 7: return "Q8_0";
        case 8: return "Q5_0";
        case 9: return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 19: return "IQ2_XXS";
        case 20: return "IQ2_XS";
        case 21: return "Q2_K_S";
        case 22: return "IQ3_XS";
        case 23: return "IQ3_XXS";
        case 24: return "IQ1_S";
        case 25: return "IQ4_NL";
        case 26: return "IQ3_S";
        case 27: return "IQ3_M";
        case 28: return "IQ2_S";
        case 29: return "IQ2_M";
        case 30: return "IQ4_XS";
        case 31: return "IQ1_M";
        case 32: return "BF16";
        case 36: return "TQ1_0";
        case 37: return "TQ2_0";
        case 38: return "MXFP4_MOE";
        case 39: return "NVFP4";
        case 40: return "Q1_0";
        case 41: return "Q2_0";
        case 42: return "F8_E4M3";
        case 141: return "PQ2_0";
        case 142: return "PTQ1_0";
        default: return {};
    }
}

// ggml tensor type id to canonical quantization label (fallback when
// general.file_type is absent or unrecognized).
std::string_view ggml_type_name(std::uint32_t type) noexcept {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        case 39: return "MXFP4";
        case 40: return "NVFP4";
        case 41: return "Q1_0";
        case 42: return "Q2_0";
        case 43: return "F8_E4M3";
        case 142: return "PQ2_0";
        case 143: return "PTQ1_0";
        case 145: return "PQ2_0M";
        default: return {};
    }
}

// Canonical sorted GGUF tokenizer metadata keys hashed into tokenizer_digest.
constexpr std::array<std::string_view, 12> kTokenizerKeys = {
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pad_token_id",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.unk_token_id",
};

constexpr bool tokenizer_keys_sorted() noexcept {
    for (std::size_t idx = 1; idx < kTokenizerKeys.size(); ++idx) {
        if (!(kTokenizerKeys[idx - 1] < kTokenizerKeys[idx])) return false;
    }
    return true;
}
static_assert(tokenizer_keys_sorted(), "kTokenizerKeys must be strictly sorted");

int tokenizer_key_index(std::string_view key) noexcept {
    for (std::size_t idx = 0; idx < kTokenizerKeys.size(); ++idx) {
        if (kTokenizerKeys[idx] == key) return static_cast<int>(idx);
    }
    return -1;
}

struct TokenizerKeyHash {
    bool present = false;
    std::array<std::uint8_t, 32> digest{};
};

// Optional observation state: ordinary header readers retain their original
// istream behavior. Direct streambuf reads permit positive short transfers
// without istream::read incorrectly treating every short transfer as EOF.
class ObservationReader {
public:
    ObservationReader(std::istream& input, const server::detail::GgufObservationOptions& options,
                      CancellationToken cancel)
        : input_(input), buffer_(input.rdbuf()), options_(options), cancel_(std::move(cancel)) {}

    bool poll() {
        if (code_ != ErrorCode::ok) return false;
        if (cancel_.cancelled()) return fail(ErrorCode::cancelled, "GGUF observation cancelled");
        if (std::chrono::steady_clock::now() >= options_.deadline) {
            return fail(ErrorCode::timeout, "GGUF observation deadline expired");
        }
        if (!input_.good() || !buffer_ || input_.rdbuf() != buffer_) {
            return fail(ErrorCode::io_error, "GGUF observation input failure");
        }
        return true;
    }
    bool header_allow(std::uint64_t n) {
        if (!poll()) return false;
        if (n > options_.max_header_bytes - total_) {
            return fail(ErrorCode::invalid_argument, "GGUF observation header budget exceeded");
        }
        if (n >= options_.max_read_bytes - total_) {
            return fail(ErrorCode::invalid_argument, "GGUF observation read budget reached");
        }
        return true;
    }
    std::size_t read(char* out, std::size_t n) {
        std::size_t done = 0;
        while (done < n && !eof_ && poll()) {
            const auto step = static_cast<std::streamsize>(std::min<std::uint64_t>(
                std::min(n - done, options_.read_chunk_bytes), options_.max_read_bytes - total_));
            try {
                if (!input_.good() || !input_.rdbuf()) {
                    fail(ErrorCode::io_error, "GGUF observation input failure");
                    break;
                }
                const auto got = buffer_->sgetn(out + done, step);
                if (got < 0 || got > step || !input_.good() || input_.rdbuf() != buffer_) {
                    fail(ErrorCode::io_error, "GGUF observation input failure");
                    break;
                }
                if (got == 0) {
                    // A zero transfer with data still available is failure,
                    // never an unbounded retry. This may cause source prefetch.
                    const auto next = buffer_->sgetc();
                    if (!input_.good() || input_.rdbuf() != buffer_ || next != std::char_traits<char>::eof()) {
                        fail(ErrorCode::io_error, "GGUF observation input made no progress");
                    } else {
                        eof_ = true;
                    }
                    poll();
                    break;
                }
                hash_.update(std::string_view(out + done, static_cast<std::size_t>(got)));
                done += static_cast<std::size_t>(got);
                total_ += static_cast<std::uint64_t>(got);
                if (total_ == options_.max_read_bytes) {
                    fail(ErrorCode::invalid_argument, "GGUF observation read budget reached");
                }
            } catch (const std::bad_alloc&) {
                throw;
            } catch (...) {
                fail(ErrorCode::io_error, "GGUF observation input failure");
            }
            if (!poll()) break;
        }
        return done;
    }
    Status status() const { return {code_, message_}; }
    std::uint64_t total() const { return total_; }
    bool eof() const { return eof_; }
    server::detail::Sha256& hash() { return hash_; }

private:
    bool fail(ErrorCode code, const char* message) {
        if (code_ == ErrorCode::ok) {
            code_ = code;
            message_ = message;
        }
        return false;
    }
    std::istream& input_;
    std::streambuf* const buffer_;
    const server::detail::GgufObservationOptions options_;
    CancellationToken cancel_;
    server::detail::Sha256 hash_;
    std::uint64_t total_ = 0;
    bool eof_ = false;
    ErrorCode code_ = ErrorCode::ok;
    const char* message_ = "";
};

class Cursor {
public:
    class TapGuard {
    public:
        TapGuard(Cursor& cursor, server::detail::Sha256* hasher) noexcept : cursor_(cursor) {
            cursor_.set_tap(hasher);
        }
        ~TapGuard() { cursor_.set_tap(nullptr); }
        TapGuard(const TapGuard&) = delete;
        TapGuard& operator=(const TapGuard&) = delete;

    private:
        Cursor& cursor_;
    };

    explicit Cursor(std::istream& in, ObservationReader* observer = nullptr) : in_(in), observer_(observer) {}

    void set_tap(server::detail::Sha256* tap) noexcept { tap_ = tap; }
    bool poll() { return !observer_ || observer_->poll(); }
    bool allow(std::uint64_t n) { return !observer_ || observer_->header_allow(n); }

    bool bytes(void* out, std::size_t n) {
        bool ok = false;
        if (observer_) {
            ok = allow(n) && observer_->read(static_cast<char*>(out), n) == n && poll();
        } else {
            in_.read(static_cast<char*>(out), static_cast<std::streamsize>(n));
            ok = static_cast<std::size_t>(in_.gcount()) == n;
        }
        if (ok && tap_ && n > 0) {
            tap_->update(std::string_view(static_cast<const char*>(out), n));
        }
        return ok;
    }
    bool skip(std::uint64_t n) {
        if (!allow(n)) return false;
        char buf[4096];
        while (n > 0) {
            const std::size_t step = static_cast<std::size_t>(std::min<std::uint64_t>(n, sizeof(buf)));
            if (!bytes(buf, step)) return false;
            n -= step;
        }
        return true;
    }
    template <class T>
    bool pod(T& out) {
        unsigned char raw[sizeof(T)];
        if (!bytes(raw, sizeof(T))) return false;
        // GGUF is little-endian.
        std::uint64_t v = 0;
        for (std::size_t i = sizeof(T); i-- > 0;) v = (v << 8) | raw[i];
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (sizeof(T) == 4) {
                const auto bits = static_cast<std::uint32_t>(v);
                std::memcpy(&out, &bits, sizeof(T));
            } else {
                std::memcpy(&out, &v, sizeof(T));
            }
        } else {
            out = static_cast<T>(v);
        }
        return true;
    }
    bool string(std::string& out) {
        std::uint64_t n = 0;
        if (!pod(n) || n > kMaxStringBytes || !allow(n)) return false;
        out.resize(static_cast<std::size_t>(n));
        return n == 0 || bytes(out.data(), out.size());
    }
    bool skip_string() {
        std::uint64_t n = 0;
        return pod(n) && n <= kMaxStringBytes && skip(n);
    }

private:
    std::istream& in_;
    ObservationReader* observer_;
    server::detail::Sha256* tap_ = nullptr;
};

std::uint64_t scalar_size(std::uint32_t type) {
    switch (type) {
        case 0:
        case 1:
        case 7: return 1;
        case 2:
        case 3: return 2;
        case 4:
        case 5:
        case 6: return 4;
        case 10:
        case 11:
        case 12: return 8;
        default: return 0;
    }
}

// A scalar value as text (integers exactly, floats with default formatting).
bool read_scalar(Cursor& c, std::uint32_t type, std::string& text, std::int64_t& as_int, bool& is_int) {
    is_int = true;
    switch (type) {
        case 0: { std::uint8_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 1: { std::int8_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 2: { std::uint16_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 3: { std::int16_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 4: { std::uint32_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 5: { std::int32_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 7: {
            std::uint8_t v;
            if (!c.pod(v)) return false;
            as_int = v != 0;
            is_int = false;
            text = v ? "true" : "false";
            return true;
        }
        case 10: {
            std::uint64_t v;
            if (!c.pod(v)) return false;
            as_int = static_cast<std::int64_t>(
                std::min<std::uint64_t>(v, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
            break;
        }
        case 11: { std::int64_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 6: {
            float v;
            if (!c.pod(v)) return false;
            is_int = false;
            as_int = 0;
            std::ostringstream s;
            s << v;
            text = s.str();
            return true;
        }
        case 12: {
            double v;
            if (!c.pod(v)) return false;
            is_int = false;
            as_int = 0;
            std::ostringstream s;
            s << v;
            text = s.str();
            return true;
        }
        default: return false;
    }
    text = std::to_string(as_int);
    return true;
}

// An integer metadata value the estimate may need: a scalar (value >= 0), or
// the items of an integer or BOOL array (per_layer; BOOL items are 0 or 1).
struct Numbers {
    std::int64_t value = -1;
    std::vector<std::int64_t> per_layer;
};

// Retain original scalar validity and every relevant key occurrence. The
// string metadata view can contain array maxima and is not count evidence.
// general.architecture may appear after its qualified keys in a GGUF header.
struct ExpertCountObservation {
    std::string key;
    std::optional<std::uint32_t> value;
};

bool expert_count_key(std::string_view key) {
    return key.size() <= kMaxArchitectureBytes + std::string_view(".expert_used_count").size() &&
           (key.ends_with(".expert_count") || key.ends_with(".expert_used_count"));
}

bool expert_namespace(std::string_view architecture) {
    const auto initial = [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
    };
    return !architecture.empty() && initial(architecture.front()) &&
           std::all_of(architecture.begin(), architecture.end(),
                       [&](char ch) { return initial(ch) || ch == '.' || ch == '-'; });
}

std::optional<RoutedExpertCounts> resolve_routed_experts(
    std::string_view architecture, std::size_t architecture_occurrences,
    const std::vector<ExpertCountObservation>& observations, ObservationReader* observer = nullptr) {
    if (architecture_occurrences != 1 || !expert_namespace(architecture)) return std::nullopt;
    const std::string prefix = std::string(architecture) + ".";
    const std::string total_key = prefix + "expert_count";
    const std::string active_key = prefix + "expert_used_count";
    std::size_t total_occurrences = 0;
    std::size_t active_occurrences = 0;
    std::optional<std::uint32_t> total;
    std::optional<std::uint32_t> active;
    for (const auto& observation : observations) {
        if (observer && !observer->poll()) return std::nullopt;
        if (observation.key == total_key) {
            ++total_occurrences;
            total = observation.value;
        } else if (observation.key == active_key) {
            ++active_occurrences;
            active = observation.value;
        }
    }
    if (total_occurrences != 1 || active_occurrences != 1 || !total || !active || *active > *total) {
        return std::nullopt;
    }
    return RoutedExpertCounts{*total, *active};
}

// ------------------------------------------------- sliding-window attention
//
// llama.cpp keeps the KV of sliding-window (SWA) layers in a second cache of
//   size_swa = GGML_PAD(min(n_ctx_seq, n_swa * (unified ? n_seq_max : 1) + n_ubatch), 256)
// cells per stream, where full-attention layers keep n_ctx_seq
// (llama_kv_cache_iswa, src/llama-kv-cache-iswa.cpp:73; llama.cpp 7fe450e,
// b11195 and the bundled llama-server's 161755f29 agree). Which layers slide,
// and the window, are set by each architecture's loader
// (src/models/<arch>.cpp, load_arch_hparams), so the estimate models SWA only
// for the architectures below. Each was checked in 7fe450e, and each gets
// llama_kv_cache_iswa (lfm2: llama_memory_hybrid_iswa) from
// llama_model::create_memory. Any other architecture counts every layer at
// full context.
enum class SwaWhen : std::uint8_t {
    always,           // the loader sets an SWA type unconditionally
    window_positive,  // only when <arch>.attention.sliding_window is present and > 0
    window_not_zero,  // unless sliding_window is present and 0 (llama4)
    blocks_64,        // only the 64-layer model (exaone4)
    never,            // the loader turns SWA off whatever the file says (phi3)
};

struct SwaArchitecture {
    std::string_view name;  // general.architecture
    SwaWhen when;
    // set_swa_pattern's period when the file has no per-layer
    // sliding_window_pattern array (llama_model_base::load_swa_pattern; a
    // scalar sliding_window_pattern in the file replaces it). 0 = the loader
    // requires the array.
    std::uint32_t period;
    bool dense_first;       // the full layer opens each period instead of closing it
    std::uint64_t window;   // the window without the key; 0 = the loader requires the key
    bool window_fixed;      // the loader replaces the file's window with `window`
    bool attention_layers;  // every layer with KV heads slides; no pattern (lfm2)
};

constexpr SwaArchitecture kSwaArchitectures[] = {
    // name          when                      period dense_first window fixed attention_layers
    {"afmoe",        SwaWhen::window_positive, 4, false, 0, false, false},
    {"cohere2",      SwaWhen::always,          4, false, 0, false, false},
    {"cohere2moe",   SwaWhen::always,          4, true, 0, false, false},
    {"exaone-moe",   SwaWhen::always,          4, false, 0, false, false},
    {"exaone4",      SwaWhen::blocks_64,       4, false, 4096, false, false},
    {"gemma2",       SwaWhen::always,          2, false, 4096, false, false},
    {"gemma3",       SwaWhen::window_positive, 6, false, 0, false, false},
    {"gemma3n",      SwaWhen::always,          5, false, 0, false, false},
    {"gemma4",       SwaWhen::always,          0, false, 0, false, false},
    {"gpt-oss",      SwaWhen::always,          2, false, 0, false, false},
    {"granite_swa",  SwaWhen::always,          0, false, 0, false, false},
    {"laguna",       SwaWhen::window_positive, 4, true, 0, false, false},
    {"lfm2",         SwaWhen::window_positive, 0, false, 0, false, true},
    {"llama4",       SwaWhen::window_not_zero, 4, false, 8192, true, false},  // chunked attention
    {"maple",        SwaWhen::always,          0, false, 0, false, false},
    {"mellum",       SwaWhen::window_positive, 4, false, 0, false, false},
    {"mimo2",        SwaWhen::always,          0, false, 0, false, false},
    {"muse-glimmer", SwaWhen::always,          4, false, 0, false, false},
    {"olmo2",        SwaWhen::window_positive, 4, false, 0, false, false},
    {"phi3",         SwaWhen::never,           0, false, 0, false, false},
    {"plamo3",       SwaWhen::window_positive, 8, false, 0, false, false},
    {"smallthinker", SwaWhen::window_positive, 4, true, 4096, true, false},
    {"spark2_5",     SwaWhen::always,          0, false, 0, false, false},
    {"step35",       SwaWhen::always,          0, false, 0, false, false},
};

// Sets info.swa_layers and info.sliding_window as llama.cpp's loader does for
// info.architecture. `window` and `pattern` are <arch>.attention.sliding_window
// and sliding_window_pattern (null when absent). Needs the final kv_heads and
// nextn_layers.
void resolve_sliding_window(GgufModelInfo& info, const Numbers* window, const Numbers* pattern,
                            ObservationReader* observer = nullptr) {
    const bool has_window = window && window->value >= 0;
    const std::uint64_t file_window = has_window ? static_cast<std::uint64_t>(window->value) : 0;
    const SwaArchitecture* arch = nullptr;
    for (const auto& candidate : kSwaArchitectures) {
        if (candidate.name == info.architecture) arch = &candidate;
    }
    if (!arch) {
        info.sliding_window_unmodelled = file_window > 0;
        return;
    }
    const std::uint32_t n_layer = info.block_count - info.nextn_layers;  // llama_hparams::n_layer()
    bool enabled = false;
    switch (arch->when) {
        case SwaWhen::always: enabled = true; break;
        case SwaWhen::window_positive: enabled = file_window > 0; break;
        case SwaWhen::window_not_zero: enabled = !has_window || file_window > 0; break;
        case SwaWhen::blocks_64: enabled = n_layer == 64; break;
        case SwaWhen::never: break;
    }
    // A required key that is missing makes llama.cpp refuse the model, so
    // nothing is capped then either.
    if (!enabled || (!arch->window_fixed && !has_window && arch->window == 0)) return;
    if (observer && !observer->poll()) return;
    std::vector<bool> swa(info.block_count, false);
    if (arch->attention_layers) {
        for (std::uint32_t i = 0; i < n_layer; ++i) {
            if (observer && !observer->poll()) return;
            swa[i] = info.kv_heads[i] > 0;
        }
    } else if (pattern && !pattern->per_layer.empty()) {
        // The per-layer array wins (get_arr); blocks past its end stay full.
        for (std::size_t i = 0; i < swa.size() && i < pattern->per_layer.size(); ++i) {
            if (observer && !observer->poll()) return;
            swa[i] = pattern->per_layer[i] != 0;
        }
    } else if (arch->period > 0) {
        // set_swa_pattern(n, dense_first); MTP blocks (past n_layer) stay full.
        const std::int64_t n = pattern && pattern->value >= 0 ? pattern->value : arch->period;
        for (std::uint32_t i = 0; i < n_layer; ++i) {
            if (observer && !observer->poll()) return;
            swa[i] = n == 0 || (arch->dense_first ? i % n != 0 : i % n < n - 1);
        }
    } else {
        return;  // the loader requires the per-layer array
    }
    if (std::none_of(swa.begin(), swa.end(), [](bool b) { return b; })) return;
    info.swa_layers = std::move(swa);
    info.sliding_window = (arch->window_fixed || !has_window) ? arch->window : file_window;
}

Result<GgufModelInfo> parse(std::istream& in, ObservationReader* observer = nullptr) {
    Cursor c(in, observer);
    char magic[4];
    if (!c.bytes(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) return malformed("not a GGUF file");
    std::uint32_t version = 0;
    std::uint64_t n_tensors = 0;
    std::uint64_t n_kv = 0;
    if (!c.pod(version) || (version != 2 && version != 3)) return malformed("unsupported version");
    if (!c.pod(n_tensors) || !c.pod(n_kv) || n_tensors > kMaxCount || n_kv > kMaxCount) {
        return malformed("implausible tensor or key count");
    }
    GgufModelInfo info;
    std::vector<std::pair<std::string, Numbers>> numbers;  // keys we may need
    std::vector<ExpertCountObservation> expert_observations;
    std::array<TokenizerKeyHash, kTokenizerKeys.size()> tokenizer_hashes{};
    std::size_t architecture_occurrences = 0;
    std::uint64_t token_count = 0;
    bool prism_tied_output = false;
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        if (!c.poll()) return malformed("observation interrupted");
        std::string key;
        std::uint32_t type = 0;
        if (!c.string(key) || !c.pod(type)) return malformed("truncated metadata");
        if (key == "general.architecture") ++architecture_occurrences;
        const bool is_prism_hadamard = key.rfind("prism.hadamard.", 0) == 0;
        if (is_prism_hadamard) info.prism_hadamard = true;
        ExpertCountObservation* expert = nullptr;
        if (expert_count_key(key)) {
            expert_observations.push_back({key, std::nullopt});
            expert = &expert_observations.back();
        }
        const int tok_idx = tokenizer_key_index(key);
        server::detail::Sha256 tok_hash;
        if (tok_idx >= 0) {
            const unsigned char tok_type_le[4] = {
                static_cast<unsigned char>(type & 0xFFu),
                static_cast<unsigned char>((type >> 8) & 0xFFu),
                static_cast<unsigned char>((type >> 16) & 0xFFu),
                static_cast<unsigned char>((type >> 24) & 0xFFu),
            };
            tok_hash.update(std::string_view(reinterpret_cast<const char*>(tok_type_le), 4));
        }
        Cursor::TapGuard tap_guard(c, tok_idx >= 0 ? &tok_hash : nullptr);
        if (type == 8) {
            std::string value;
            if (!c.string(value)) return malformed("truncated string value");
            if (key == "general.architecture") {
                if (value.size() > kMaxArchitectureBytes || !printable(value)) {
                    return malformed("general.architecture must be at most 256 printable ASCII characters");
                }
                info.architecture = value;
            } else if (key == "prism.hadamard.transform" && value.size() <= 256) {
                info.prism_hadamard_transform = value;
            } else if (key == "tokenizer.chat_template") {
                info.template_digest = value.empty() ? std::string() : server::detail::sha256_hex(value);
            }
            if (value.size() <= 256) info.metadata.emplace_back(key, value);
        } else if (type == 9) {
            std::uint32_t item_type = 0;
            std::uint64_t n = 0;
            if (!c.pod(item_type) || !c.pod(n) || n > kMaxCount) return malformed("truncated array");
            if (key == "tokenizer.ggml.tokens") token_count = n;
            if (item_type == 8) {
                for (std::uint64_t k = 0; k < n; ++k) {
                    if (!c.skip_string()) return malformed("truncated string array");
                }
            } else {
                const std::uint64_t size = scalar_size(item_type);
                if (size == 0) return malformed("unsupported array item type in '" + excerpt(key) + "'");
                // PrismML Hadamard arrays (e.g. prism.hadamard.sign_values[28672])
                // and tokenizer arrays (scores/token_type) are skipped without
                // allocating or bloating info.metadata (tokenizer arrays are
                // hashed incrementally through Cursor's tap).
                if (is_prism_hadamard || tok_idx >= 0) {
                    if (!c.skip(size * n)) return malformed("truncated array");
                } else if (item_type != 6 && item_type != 12 && n <= 4096) {
                    // Integer and BOOL arrays are kept per item: per-block head
                    // counts, a sliding_window_pattern (BOOL, or INT32/UINT32). A
                    // BOOL array stays out of `metadata`, which the classification
                    // reads and which never held one.
                    Numbers num;
                    std::int64_t max_value = 0;
                    for (std::uint64_t k = 0; k < n; ++k) {
                        std::string text;
                        std::int64_t v = 0;
                        bool is_int = false;
                        if (!read_scalar(c, item_type, text, v, is_int)) return malformed("truncated array");
                        num.per_layer.push_back(v);
                        max_value = std::max(max_value, v);
                    }
                    if (item_type != 7) info.metadata.emplace_back(key, std::to_string(max_value));
                    numbers.emplace_back(key, std::move(num));
                } else {
                    if (!c.skip(size * n)) return malformed("truncated array");
                }
            }
        } else {
            std::string text;
            std::int64_t v = 0;
            bool is_int = false;
            if (!read_scalar(c, type, text, v, is_int)) {
                return malformed("unsupported value type in '" + excerpt(key) + "'");
            }
            info.metadata.emplace_back(key, text);
            if (is_int) numbers.emplace_back(key, Numbers{v, {}});
            if (key == "prism.hadamard.block_size" && is_int && v > 0 &&
                v <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
                info.prism_hadamard_block_size = static_cast<std::uint32_t>(v);
            }
            if (key == "prism.hadamard.tied_output" && v != 0) {
                prism_tied_output = true;
            }
            if (expert && is_int && v > 0 && v <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
                expert->value = static_cast<std::uint32_t>(v);
            }
        }
        if (tok_idx >= 0) {
            c.set_tap(nullptr);
            tokenizer_hashes[static_cast<std::size_t>(tok_idx)].present = true;
            tokenizer_hashes[static_cast<std::size_t>(tok_idx)].digest = tok_hash.finish();
        }
    }
    if (info.architecture.empty()) return malformed("general.architecture is missing");
    if (!c.poll()) return malformed("observation interrupted");
    bool any_tokenizer_key = false;
    for (const auto& entry : tokenizer_hashes) {
        if (entry.present) {
            any_tokenizer_key = true;
            break;
        }
    }
    if (any_tokenizer_key) {
        server::detail::Sha256 tok_combined;
        tok_combined.update("sonder-gguf-tokenizer-v1\n");
        for (std::size_t idx = 0; idx < kTokenizerKeys.size(); ++idx) {
            if (!tokenizer_hashes[idx].present) continue;
            const std::string_view tok_k = kTokenizerKeys[idx];
            const std::uint64_t tok_klen = static_cast<std::uint64_t>(tok_k.size());
            unsigned char tok_len_le[8];
            for (std::size_t tok_b = 0; tok_b < 8; ++tok_b) {
                tok_len_le[tok_b] = static_cast<unsigned char>((tok_klen >> (8 * tok_b)) & 0xFFu);
            }
            tok_combined.update(std::string_view(reinterpret_cast<const char*>(tok_len_le), 8));
            tok_combined.update(tok_k);
            tok_combined.update(std::string_view(
                reinterpret_cast<const char*>(tokenizer_hashes[idx].digest.data()),
                tokenizer_hashes[idx].digest.size()));
        }
        info.tokenizer_digest = digest_to_hex(tok_combined.finish());
    }
    info.routed_experts =
        resolve_routed_experts(info.architecture, architecture_occurrences, expert_observations, observer);
    const std::string a = info.architecture + ".";
    const auto number = [&](const std::string& key) -> const Numbers* {
        for (const auto& [k, n] : numbers) {
            if (!c.poll()) return nullptr;
            if (k == key) return &n;
        }
        return nullptr;
    };
    const auto scalar = [&](const std::string& key, std::int64_t fallback) -> std::int64_t {
        const Numbers* n = number(key);
        return n && n->value >= 0 ? n->value : fallback;
    };
    const std::int64_t blocks = scalar(a + "block_count", -1);
    if (blocks <= 0 || blocks > 100000) return malformed(a + "block_count is missing");
    info.block_count = static_cast<std::uint32_t>(blocks);
    const std::int64_t ctx_len = scalar(a + "context_length", 0);
    info.training_context_length = ctx_len > 0 ? static_cast<std::uint64_t>(ctx_len) : 0;
    info.embedding_length = static_cast<std::uint64_t>(scalar(a + "embedding_length", 0));
    info.nextn_layers =
        static_cast<std::uint32_t>(std::clamp<std::int64_t>(scalar(a + "nextn_predict_layers", 0), 0, blocks));
    const std::int64_t heads = scalar(a + "attention.head_count", 0);
    const std::uint64_t head_dim =
        heads > 0 && info.embedding_length > 0 ? info.embedding_length / static_cast<std::uint64_t>(heads) : 0;
    info.key_length = static_cast<std::uint64_t>(scalar(a + "attention.key_length", static_cast<std::int64_t>(head_dim)));
    info.value_length =
        static_cast<std::uint64_t>(scalar(a + "attention.value_length", static_cast<std::int64_t>(head_dim)));
    // SWA blocks' head sizes default to the full-attention ones (llama_model::load_hparams).
    info.key_length_swa =
        static_cast<std::uint64_t>(scalar(a + "attention.key_length_swa", static_cast<std::int64_t>(info.key_length)));
    info.value_length_swa = static_cast<std::uint64_t>(
        scalar(a + "attention.value_length_swa", static_cast<std::int64_t>(info.value_length)));
    if (!c.poll()) return malformed("observation interrupted");
    info.kv_heads.assign(info.block_count, 0);
    if (const Numbers* kv = number(a + "attention.head_count_kv"); kv && !kv->per_layer.empty()) {
        for (std::size_t i = 0; i < info.kv_heads.size() && i < kv->per_layer.size(); ++i) {
            if (!c.poll()) return malformed("observation interrupted");
            info.kv_heads[i] = static_cast<std::uint32_t>(std::max<std::int64_t>(0, kv->per_layer[i]));
        }
    } else {
        const std::int64_t kv_heads = scalar(a + "attention.head_count_kv", heads);
        std::fill(info.kv_heads.begin(), info.kv_heads.end(),
                  static_cast<std::uint32_t>(std::max<std::int64_t>(0, kv_heads)));
    }
    // Recurrent state (Mamba 1/2, delta-net): conv (kernel - 1) x channels,
    // channels = inner + 2 x groups x state, plus inner x state. scalar()
    // returns only non-negative values; the caps keep the unsigned products
    // below 2^53, and a larger state than any real model's is refused.
    const auto conv = static_cast<std::uint64_t>(scalar(a + "ssm.conv_kernel", 0));
    const auto inner = static_cast<std::uint64_t>(scalar(a + "ssm.inner_size", 0));
    const auto state = static_cast<std::uint64_t>(scalar(a + "ssm.state_size", 0));
    const auto groups = static_cast<std::uint64_t>(scalar(a + "ssm.group_count", 0));
    if (conv > 0 && inner > 0 && state > 0) {
        if (conv > kMaxSsmConvKernel || inner > kMaxSsmSize || state > kMaxSsmSize || groups > kMaxSsmSize) {
            return malformed("implausible " + a + "ssm.* sizes");
        }
        const std::uint64_t values = (conv - 1) * (inner + 2 * groups * state) + inner * state;
        if (values > kMaxRecurrentStateValues) return malformed("implausible " + a + "ssm.* sizes");
        info.recurrent_state_values = values;
    }

    // Tensor table.
    if (!c.poll()) return malformed("observation interrupted");
    std::vector<bool> attention_tensor(info.block_count, false);
    bool any_attention_tensor = false;
    info.block_bytes.assign(info.block_count, 0);
    std::map<std::uint32_t, std::uint64_t> tensor_bytes_by_type;
    std::uint64_t output_vocab = 0;
    std::uint64_t embd_vocab = 0;
    std::uint64_t token_embd_weight_bytes = 0;
    bool has_output_weight = false;
    bool has_output_norm = false;
    for (std::uint64_t t = 0; t < n_tensors; ++t) {
        if (!c.poll()) return malformed("observation interrupted");
        std::string name;
        std::uint32_t dims = 0;
        if (!c.string(name) || !c.pod(dims) || dims == 0 || dims > 8) return malformed("truncated tensor table");
        std::uint64_t elements = 1;
        std::uint64_t second = 0;
        for (std::uint32_t d = 0; d < dims; ++d) {
            std::uint64_t n = 0;
            if (!c.pod(n)) return malformed("truncated tensor table");
            if (d == 1) second = n;
            if (n != 0 && elements > std::numeric_limits<std::uint64_t>::max() / n) return malformed("tensor too large");
            elements *= n;
        }
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
        if (!c.pod(type) || !c.pod(offset)) return malformed("truncated tensor table");
        std::uint64_t block_elems = 0;
        std::uint64_t block_bytes = 0;
        if (!ggml_type_size(type, block_elems, block_bytes)) {
            return malformed("tensor " + excerpt(name) + " has an unknown ggml type " + std::to_string(type));
        }
        if (type == 34 || type == 35 || type == 42 || type == 142 || type == 143 || type == 145) {
            info.has_ternary_tensors = true;
        }
        if (type == 41) {
            info.has_one_bit_tensors = true;
        }
        if (type == 142 || type == 143 || type == 145) {
            info.has_prism_private_tensors = true;
        }
        if (type == 39) {
            info.has_mxfp4_tensors = true;
            info.has_fp4_or_fp8_tensors = true;
        }
        if (type == 40) {
            info.has_nvfp4_tensors = true;
            info.has_fp4_or_fp8_tensors = true;
        }
        if (type == 43) {
            info.has_fp8_tensors = true;
            info.has_fp4_or_fp8_tensors = true;
        }
        const std::uint64_t num_blocks = elements / block_elems + (elements % block_elems != 0 ? 1u : 0u);
        if (num_blocks > std::numeric_limits<std::uint64_t>::max() / block_bytes) {
            return malformed("tensor too large");
        }
        const std::uint64_t bytes = num_blocks * block_bytes;
        std::uint64_t& type_bytes_acc = tensor_bytes_by_type[type];
        type_bytes_acc = detail::saturating_add(type_bytes_acc, bytes);
        std::uint32_t tensor_block = kNoBlock;
        if (name.rfind("blk.", 0) == 0) {
            const auto dot = name.find('.', 4);
            std::uint64_t layer = 0;
            bool ok = dot != std::string::npos && dot > 4;
            for (std::size_t k = 4; ok && k < dot; ++k) {
                ok = name[k] >= '0' && name[k] <= '9' && layer < 1000000;
                layer = layer * 10 + static_cast<std::uint64_t>(name[k] - '0');
            }
            if (!ok || layer >= info.block_count) return malformed("tensor " + excerpt(name) + " names an unknown block");
            if (!checked_add(info.block_bytes[layer], bytes)) return malformed("tensor sizes overflow");
            tensor_block = static_cast<std::uint32_t>(layer);
            const std::string suffix = name.substr(dot + 1);
            if (suffix == "attn_k.weight" || suffix == "attn_kv_a_mqa.weight") {
                attention_tensor[layer] = true;
                any_attention_tensor = true;
            }
        } else if (name == "token_embd.weight") {
            if (!checked_add(info.embedding_bytes, bytes)) return malformed("tensor sizes overflow");
            token_embd_weight_bytes = bytes;
            embd_vocab = second;
        } else if (name.rfind("output", 0) == 0) {
            if (!checked_add(info.output_bytes, bytes)) return malformed("tensor sizes overflow");
            if (name == "output.weight") {
                has_output_weight = true;
                output_vocab = second;
            } else if (name == "output_norm.weight") {
                has_output_norm = true;
            }
        } else {
            if (!checked_add(info.other_bytes, bytes)) return malformed("tensor sizes overflow");
        }
        if (info.tensors.size() < kMaxStoredTensors && name.size() <= kMaxStoredTensorName) {
            GgufTensorInfo entry;
            entry.expert = tensor_block != kNoBlock &&
                           (name.find("_exps.") != std::string::npos || name.find("_chexps.") != std::string::npos);
            entry.name = std::move(name);
            entry.type = type;
            entry.bytes = bytes;
            entry.block = tensor_block;
            info.tensors.push_back(std::move(entry));
        } else {
            info.tensors_truncated = true;
        }
    }
    // Tied LM head (e.g. Ternary-Bonsai-2-4B prism.hadamard.tied_output=1, Qwen3-4B,
    // Llama-3.2, Gemma): llama.cpp duplicates token_embd.weight onto LAYER_OUTPUT
    // (TENSOR_DUPLICATED), so full GPU offload allocates it in VRAM as well as host input.
    if (!has_output_weight && token_embd_weight_bytes > 0 && (has_output_norm || prism_tied_output)) {
        if (!checked_add(info.output_bytes, token_embd_weight_bytes)) return malformed("tensor sizes overflow");
    }
    const Numbers* file_type_num = number("general.file_type");
    if (file_type_num && file_type_num->value >= 0 &&
        file_type_num->value <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        const std::string_view ftype_name = gguf_file_type_name(static_cast<std::uint32_t>(file_type_num->value));
        if (!ftype_name.empty()) info.quantization = std::string(ftype_name);
    }
    if (info.quantization.empty() && !tensor_bytes_by_type.empty()) {
        std::uint32_t dominant_type = 0;
        std::uint64_t dominant_bytes = 0;
        bool has_dominant = false;
        for (const auto& [tb_type, tb_bytes] : tensor_bytes_by_type) {
            if (!has_dominant || tb_bytes > dominant_bytes) {
                dominant_type = tb_type;
                dominant_bytes = tb_bytes;
                has_dominant = true;
            }
        }
        if (has_dominant) {
            const std::string_view dom_name = ggml_type_name(dominant_type);
            if (!dom_name.empty()) info.quantization = std::string(dom_name);
        }
    }
    const std::uint64_t inferred_vocab = output_vocab > 0 ? output_vocab : embd_vocab;
    info.vocab_size =
        token_count > 0 ? token_count
                        : static_cast<std::uint64_t>(scalar(a + "vocab_size", static_cast<std::int64_t>(inferred_vocab)));
    info.kind = classify_model_architecture(info.metadata);

    // Which blocks keep attention KV. Per-block head counts (already set)
    // are authoritative when the metadata had an array; otherwise hybrids
    // use their attn_k tensors, then full_attention_interval (trailing NextN/MTP
    // blocks are always dense attention: qwen35.cpp load_arch_hparams).
    const Numbers* kv_array = number(a + "attention.head_count_kv");
    const bool per_layer_heads = kv_array && !kv_array->per_layer.empty();
    if (!per_layer_heads && info.kind != ModelArchitecture::attention_only) {
        const std::int64_t interval = scalar(a + "full_attention_interval", 0);
        const std::uint32_t trunk_blocks = info.block_count - info.nextn_layers;
        for (std::uint32_t i = 0; i < info.block_count; ++i) {
            if (!c.poll()) return malformed("observation interrupted");
            bool attention = false;
            if (any_attention_tensor) {
                attention = attention_tensor[i];
            } else if (i >= trunk_blocks) {
                attention = true;
            } else if (interval > 0) {
                attention = (static_cast<std::int64_t>(i) + 1) % interval == 0;
            }
            if (!attention) info.kv_heads[i] = 0;
        }
    }
    resolve_sliding_window(info, number(a + "attention.sliding_window"), number(a + "attention.sliding_window_pattern"),
                           observer);
    if (!c.poll()) return malformed("observation interrupted");
    return info;
}

}  // namespace

namespace server::detail {

Result<GgufArtifactObservation> observe_gguf_artifact(std::istream& input, std::string_view expected_sha256,
                                                      const GgufObservationOptions& options, CancellationToken cancel) {
    if (expected_sha256.size() != 64 || !std::all_of(expected_sha256.begin(), expected_sha256.end(), [](char ch) {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        }) ||
        options.max_read_bytes < 2 || options.max_read_bytes > std::numeric_limits<std::uint64_t>::max() / 8 ||
        options.max_header_bytes == 0 || options.max_header_bytes >= options.max_read_bytes ||
        options.read_chunk_bytes == 0 || options.read_chunk_bytes > 65536 ||
        options.deadline == std::chrono::steady_clock::time_point::max()) {
        return Status(ErrorCode::invalid_argument, "invalid GGUF observation options");
    }
    try {
        const std::string trusted_sha256(expected_sha256);
        ObservationReader reader(input, options, std::move(cancel));
        if (!reader.poll()) return reader.status();
        if (!input.good() || !input.rdbuf()) return Status(ErrorCode::io_error, "GGUF observation input failure");
        GgufArtifactObservation observation;
        {
            auto info = parse(input, &reader);
            if (!reader.poll()) return reader.status();
            if (!info.ok()) return Status(ErrorCode::invalid_argument, "invalid GGUF observation header");
            observation.architecture = std::move(info->architecture);
            observation.routed_experts = info->routed_experts;
        }  // Discard estimate-only metadata/vectors before draining payload.
        char buffer[65536];
        while (!reader.eof() && reader.poll()) reader.read(buffer, sizeof(buffer));
        if (!reader.poll()) return reader.status();
        if (!reader.eof()) return Status(ErrorCode::io_error, "GGUF observation input failure");
        const auto bytes = reader.hash().finish();
        static constexpr char kHex[] = "0123456789abcdef";
        observation.sha256.reserve(64);
        for (const auto byte : bytes) {
            observation.sha256.push_back(kHex[byte >> 4]);
            observation.sha256.push_back(kHex[byte & 15]);
        }
        if (!reader.poll()) return reader.status();
        if (observation.sha256 != trusted_sha256) {
            return Status(ErrorCode::invalid_argument, "GGUF observation digest mismatch");
        }
        observation.byte_length = reader.total();
        if (!reader.poll()) return reader.status();
        return observation;
    } catch (const std::bad_alloc&) {
        return Status(ErrorCode::unavailable, "GGUF observation allocation failure");
    }
}

}  // namespace server::detail

Result<GgufModelInfo> parse_gguf_model_info(std::string_view header_bytes) {
    std::istringstream in{std::string(header_bytes)};
    return parse(in);
}

Result<GgufModelInfo> read_gguf_model_info(const std::string& path) {
    std::ifstream in(path_from_utf8(path), std::ios::binary);
    if (!in) return Status(ErrorCode::not_found, "cannot read GGUF " + path);
    auto info = parse(in);
    if (!info.ok()) return Status(info.status().code(), info.status().message() + " (" + path + ")");
    return info;
}

std::string redact_model_path(const Status& error, const std::string& path) {
    std::string text = error.message();
    if (path.empty()) return text;
    static constexpr std::string_view kPlaceholder = "<model path>";
    for (std::size_t at = text.find(path); at != std::string::npos; at = text.find(path, at + kPlaceholder.size())) {
        text.replace(at, path.size(), kPlaceholder);
    }
    return text;
}

}  // namespace sonder::inference
