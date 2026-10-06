// GGUF header reader and the VRAM fit estimate for launch profiles
// (launch_profile.hpp). Reads only the header: metadata and the tensor
// table, never tensor data.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <type_traits>

#include "sonder/inference/launch_profile.hpp"
#include "gguf_artifact_observation.hpp"
#include "sha256.hpp"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  if defined(_MSC_VER)
#    include <dxgi.h>
#    pragma comment(lib, "dxgi.lib")
#  endif
#endif

namespace sonder::inference {

namespace {

constexpr std::uint64_t kMaxCount = 1ull << 24;       // KV pairs, tensors, array items
constexpr std::uint64_t kMaxStringBytes = 1ull << 24;  // one string (chat templates are ~20 KiB)
constexpr std::uint64_t kMiB = 1ull << 20;
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

bool printable(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](char ch) { return ch >= 0x20 && ch < 0x7f; });
}

// File text (a key or tensor name, up to 16 MiB) as quoted in an error: at
// most 64 bytes, non-printable bytes as '?'. Errors reach /v1/models
// (`estimate_error`) and the operator's log.
// (Not named `quoted`: ADL on a std::string argument would pick std::quoted.)
std::string excerpt(std::string_view text) {
    std::string out;
    for (const char ch : text.substr(0, 64)) out += (ch >= 0x20 && ch < 0x7f) ? ch : '?';
    if (text.size() > 64) out += "...";
    return out;
}

// `v` rounded to the nearest integer and saturated to [0, 2^64 - 1]: casting
// an out-of-range double is undefined behaviour, and crafted GGUF metadata
// can push the KV arithmetic (done in double) past 2^64.
std::uint64_t saturating_u64(double v) {
    const double r = std::floor(v + 0.5);
    if (!(r > 0.0)) return 0;  // also NaN
    if (r >= 18446744073709551616.0) return std::numeric_limits<std::uint64_t>::max();  // 2^64
    return static_cast<std::uint64_t>(r);
}

// a x b and a + b, saturated at 2^64 - 1 (the window comes from the file).
std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    return b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b ? std::numeric_limits<std::uint64_t>::max()
                                                                         : a * b;
}
std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

// Speculative decoding that rolls back recurrent state keeps one state
// snapshot per draft position: llama.cpp allocates max(1, n_seq_max) x
// (1 + n_rs_seq) recurrent rows per layer, with n_rs_seq = --spec-draft-n-max
// for these types (common_params_speculative::need_n_rs_seq), clamped to 0 on
// architectures without recurrent rollback (llm_arch_supports_rs_rollback;
// llama-memory-recurrent.cpp; llama.cpp source at 7fe450e). On the bundled
// llama-server (161755f29) each +1 of --spec-draft-n-max measured +150 MiB
// with Qwen3.8-27B, whose snapshot is 149.6 MiB.
constexpr std::string_view kRollbackSpeculativeTypes[] = {"draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark"};
constexpr std::string_view kRecurrentRollbackArchitectures[] = {
    "qwen35", "qwen35moe", "qwen4exp", "deepseek4", "nemotron_h", "nemotron_h_moe", "bailingmoe3", "lfm2", "lfm2moe",
    "kimi-k3"};
constexpr std::uint32_t kDefaultDraftNMax = 3;  // llama.cpp common_params_speculative_draft::n_max

// n_rs_seq for `profile` on `model`: the extra recurrent snapshots per sequence.
std::uint32_t recurrent_rollback_snapshots(const GgufModelInfo& model, const LaunchProfile& profile) {
    if (!profile.speculative) return 0;
    const auto& types = profile.speculative->types;
    const bool rolls_back = std::any_of(types.begin(), types.end(), [](const std::string& t) {
        return std::find(std::begin(kRollbackSpeculativeTypes), std::end(kRollbackSpeculativeTypes), t) !=
               std::end(kRollbackSpeculativeTypes);
    });
    const bool supported = std::find(std::begin(kRecurrentRollbackArchitectures), std::end(kRecurrentRollbackArchitectures),
                                     model.architecture) != std::end(kRecurrentRollbackArchitectures);
    return rolls_back && supported ? profile.speculative->draft_n_max.value_or(kDefaultDraftNMax) : 0;
}

// ggml tensor types: (elements per block, bytes per block). ggml.h / ggml.c.
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
        if (code_ == ErrorCode::ok) { code_ = code; message_ = message; }
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
    explicit Cursor(std::istream& in, ObservationReader* observer = nullptr) : in_(in), observer_(observer) {}

    bool poll() { return !observer_ || observer_->poll(); }
    bool allow(std::uint64_t n) { return !observer_ || observer_->header_allow(n); }

    bool bytes(void* out, std::size_t n) {
        if (observer_) return allow(n) && observer_->read(static_cast<char*>(out), n) == n && poll();
        in_.read(static_cast<char*>(out), static_cast<std::streamsize>(n));
        return static_cast<std::size_t>(in_.gcount()) == n;
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
};

std::uint64_t scalar_size(std::uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
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
        case 7: { std::uint8_t v; if (!c.pod(v)) return false; as_int = v != 0; is_int = false; text = v ? "true" : "false"; return true; }
        case 10: { std::uint64_t v; if (!c.pod(v)) return false; as_int = static_cast<std::int64_t>(std::min<std::uint64_t>(v, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))); break; }
        case 11: { std::int64_t v; if (!c.pod(v)) return false; as_int = v; break; }
        case 6: { float v; if (!c.pod(v)) return false; is_int = false; as_int = 0; std::ostringstream s; s << v; text = s.str(); return true; }
        case 12: { double v; if (!c.pod(v)) return false; is_int = false; as_int = 0; std::ostringstream s; s << v; text = s.str(); return true; }
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
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '_';
    };
    return !architecture.empty() && initial(architecture.front()) &&
           std::all_of(architecture.begin(), architecture.end(), [&](char ch) {
               return initial(ch) || ch == '.' || ch == '-';
           });
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
    bool dense_first;          // the full layer opens each period instead of closing it
    std::uint64_t window;      // the window without the key; 0 = the loader requires the key
    bool window_fixed;         // the loader replaces the file's window with `window`
    bool attention_layers;     // every layer with KV heads slides; no pattern (lfm2)
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
    std::size_t architecture_occurrences = 0;
    std::uint64_t token_count = 0;
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        if (!c.poll()) return malformed("observation interrupted");
        std::string key;
        std::uint32_t type = 0;
        if (!c.string(key) || !c.pod(type)) return malformed("truncated metadata");
        if (key == "general.architecture") ++architecture_occurrences;
        ExpertCountObservation* expert = nullptr;
        if (expert_count_key(key)) {
            expert_observations.push_back({key, std::nullopt});
            expert = &expert_observations.back();
        }
        if (type == 8) {
            std::string value;
            if (!c.string(value)) return malformed("truncated string value");
            if (key == "general.architecture") {
                if (value.size() > kMaxArchitectureBytes || !printable(value)) {
                    return malformed("general.architecture must be at most 256 printable ASCII characters");
                }
                info.architecture = value;
            }
            if (value.size() <= 256) info.metadata.emplace_back(key, value);
            continue;
        }
        if (type == 9) {
            std::uint32_t item_type = 0;
            std::uint64_t n = 0;
            if (!c.pod(item_type) || !c.pod(n) || n > kMaxCount) return malformed("truncated array");
            if (key == "tokenizer.ggml.tokens") token_count = n;
            if (item_type == 8) {
                for (std::uint64_t k = 0; k < n; ++k) {
                    if (!c.skip_string()) return malformed("truncated string array");
                }
                continue;
            }
            const std::uint64_t size = scalar_size(item_type);
            if (size == 0) return malformed("unsupported array item type in '" + excerpt(key) + "'");
            // Integer and BOOL arrays are kept per item: per-block head
            // counts, a sliding_window_pattern (BOOL, or INT32/UINT32). A
            // BOOL array stays out of `metadata`, which the classification
            // reads and which never held one.
            if (item_type != 6 && item_type != 12 && n <= 4096) {
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
                continue;
            }
            if (!c.skip(size * n)) return malformed("truncated array");
            continue;
        }
        std::string text;
        std::int64_t v = 0;
        bool is_int = false;
        if (!read_scalar(c, type, text, v, is_int)) return malformed("unsupported value type in '" + excerpt(key) + "'");
        info.metadata.emplace_back(key, text);
        if (is_int) numbers.emplace_back(key, Numbers{v, {}});
        if (expert && is_int && v > 0 && v <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
            expert->value = static_cast<std::uint32_t>(v);
        }
    }
    if (info.architecture.empty()) return malformed("general.architecture is missing");
    if (!c.poll()) return malformed("observation interrupted");
    info.routed_experts = resolve_routed_experts(info.architecture, architecture_occurrences, expert_observations, observer);
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
    info.embedding_length = static_cast<std::uint64_t>(scalar(a + "embedding_length", 0));
    info.nextn_layers = static_cast<std::uint32_t>(std::clamp<std::int64_t>(scalar(a + "nextn_predict_layers", 0), 0, blocks));
    const std::int64_t heads = scalar(a + "attention.head_count", 0);
    const std::uint64_t head_dim = heads > 0 && info.embedding_length > 0 ? info.embedding_length / static_cast<std::uint64_t>(heads) : 0;
    info.key_length = static_cast<std::uint64_t>(scalar(a + "attention.key_length", static_cast<std::int64_t>(head_dim)));
    info.value_length = static_cast<std::uint64_t>(scalar(a + "attention.value_length", static_cast<std::int64_t>(head_dim)));
    // SWA blocks' head sizes default to the full-attention ones (llama_model::load_hparams).
    info.key_length_swa =
        static_cast<std::uint64_t>(scalar(a + "attention.key_length_swa", static_cast<std::int64_t>(info.key_length)));
    info.value_length_swa =
        static_cast<std::uint64_t>(scalar(a + "attention.value_length_swa", static_cast<std::int64_t>(info.value_length)));
    if (!c.poll()) return malformed("observation interrupted");
    info.kv_heads.assign(info.block_count, 0);
    if (const Numbers* kv = number(a + "attention.head_count_kv"); kv && !kv->per_layer.empty()) {
        for (std::size_t i = 0; i < info.kv_heads.size() && i < kv->per_layer.size(); ++i) {
            if (!c.poll()) return malformed("observation interrupted");
            info.kv_heads[i] = static_cast<std::uint32_t>(std::max<std::int64_t>(0, kv->per_layer[i]));
        }
    } else {
        const std::int64_t kv_heads = scalar(a + "attention.head_count_kv", heads);
        std::fill(info.kv_heads.begin(), info.kv_heads.end(), static_cast<std::uint32_t>(std::max<std::int64_t>(0, kv_heads)));
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
    std::uint64_t output_vocab = 0;
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
        const std::uint64_t bytes = (elements + block_elems - 1) / block_elems * block_bytes;
        if (name.rfind("blk.", 0) == 0) {
            const auto dot = name.find('.', 4);
            std::uint64_t layer = 0;
            bool ok = dot != std::string::npos && dot > 4;
            for (std::size_t k = 4; ok && k < dot; ++k) {
                ok = name[k] >= '0' && name[k] <= '9' && layer < 1000000;
                layer = layer * 10 + static_cast<std::uint64_t>(name[k] - '0');
            }
            if (!ok || layer >= info.block_count) return malformed("tensor " + excerpt(name) + " names an unknown block");
            info.block_bytes[layer] += bytes;
            const std::string suffix = name.substr(dot + 1);
            if (suffix == "attn_k.weight" || suffix == "attn_kv_a_mqa.weight") {
                attention_tensor[layer] = true;
                any_attention_tensor = true;
            }
        } else if (name == "token_embd.weight") {
            info.embedding_bytes += bytes;
        } else if (name.rfind("output", 0) == 0) {
            info.output_bytes += bytes;
            if (name == "output.weight") output_vocab = second;
        } else {
            info.other_bytes += bytes;
        }
    }
    info.vocab_size = token_count > 0 ? token_count : static_cast<std::uint64_t>(scalar(a + "vocab_size", static_cast<std::int64_t>(output_vocab)));
    info.kind = classify_model_architecture(info.metadata);

    // Which blocks keep attention KV. Per-block head counts (already set)
    // are authoritative when the metadata had an array; otherwise hybrids
    // use their attn_k tensors, then full_attention_interval.
    const Numbers* kv_array = number(a + "attention.head_count_kv");
    const bool per_layer_heads = kv_array && !kv_array->per_layer.empty();
    if (!per_layer_heads && info.kind != ModelArchitecture::attention_only) {
        const std::int64_t interval = scalar(a + "full_attention_interval", 0);
        for (std::uint32_t i = 0; i < info.block_count; ++i) {
            if (!c.poll()) return malformed("observation interrupted");
            bool attention = false;
            if (any_attention_tensor) {
                attention = attention_tensor[i];
            } else if (interval > 0) {
                attention = (static_cast<std::int64_t>(i) + 1) % interval == 0;
            }
            if (!attention) info.kv_heads[i] = 0;
        }
    }
    resolve_sliding_window(info, number(a + "attention.sliding_window"), number(a + "attention.sliding_window_pattern"), observer);
    if (!c.poll()) return malformed("observation interrupted");
    return info;
}

}  // namespace

namespace server::detail {

Result<GgufArtifactObservation> observe_gguf_artifact(
    std::istream& input, std::string_view expected_sha256,
    const GgufObservationOptions& options, CancellationToken cancel) {
    if (expected_sha256.size() != 64 || !std::all_of(expected_sha256.begin(), expected_sha256.end(), [](char ch) {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        }) || options.max_read_bytes < 2 || options.max_read_bytes > std::numeric_limits<std::uint64_t>::max() / 8 ||
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
    std::ifstream in(path, std::ios::binary);
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

VramEstimate estimate_vram(const GgufModelInfo& m, const LaunchProfile& p, const VramEstimateInputs& inputs) {
    VramEstimate e;
    const std::uint32_t n_layer = m.block_count;
    const bool mtp = p.speculative &&
                     std::find(p.speculative->types.begin(), p.speculative->types.end(), "draft-mtp") != p.speculative->types.end();
    // llama.cpp: -1/all, or more layers than blocks, also offloads the output.
    const std::int64_t requested = p.n_gpu_layers.value_or(-1);
    const bool all = requested < 0 || requested > static_cast<std::int64_t>(n_layer);
    const std::uint32_t offloaded = all ? n_layer : static_cast<std::uint32_t>(requested);
    const std::uint32_t first_gpu = n_layer - offloaded;
    const std::uint32_t first_nextn = n_layer - std::min(m.nextn_layers, n_layer);

    const std::uint64_t ctx_raw = p.ctx_size && *p.ctx_size > 0 ? *p.ctx_size : 0;
    std::uint64_t ctx = ctx_raw;
    if (ctx == 0) {
        for (const auto& [k, v] : m.metadata) {
            if (k == m.architecture + ".context_length") {
                try {
                    ctx = std::stoull(v);
                } catch (...) {
                    ctx = 0;
                }
            }
        }
        e.notes.emplace_back("ctx_size unset: the model's training context is assumed");
    }
    ctx = (ctx + 255) / 256 * 256;  // llama.cpp pads the context to 256
    e.context = ctx;
    // Without `parallel`, llama-server runs its automatic default: 4 slots
    // (n_seq_max 4, so 4 recurrent states) with a unified KV cache
    // (kLlamaServerAutoParallel). The direct llamacpp backend decodes one
    // sequence and refuses `parallel`.
    const bool auto_parallel = !p.parallel && p.backend == kLaunchProfileBackendLlamaServer;
    const std::uint32_t slots = p.parallel ? *p.parallel : (auto_parallel ? kLlamaServerAutoParallel : 1u);
    const bool unified = auto_parallel || p.kv_unified.value_or(!p.parallel.has_value());
    e.sequences = unified ? 1 : slots;
    e.recurrent_sequences = slots;
    // --ctx-size is the total KV size. A non-unified cache splits it into one
    // stream per sequence of n_ctx_seq = pad(n_ctx / n_seq_max, 256) cells
    // (llama_context's constructor, llama.cpp b11195), so the pool is
    // n_ctx_seq x parallel, not ctx x parallel. Unified: one pool of ctx.
    std::uint64_t ctx_per_sequence = ctx;
    if (e.sequences > 1) {
        ctx_per_sequence = (ctx / e.sequences + 255) / 256 * 256;
    }
    e.context_per_sequence = ctx_per_sequence;

    const double bk = kv_cache_type_bytes(p.cache_type_k.value_or("f16"));
    const double bv = kv_cache_type_bytes(p.cache_type_v.value_or("f16"));
    // Micro-batch: --ubatch-size, at most --batch-size (llama_context). The
    // defaults are llama-server's 2048/512 and the direct backend's 512/512.
    const std::uint64_t batch = p.batch_size.value_or(p.backend == kLaunchProfileBackendLlamaCpp ? 512u : 2048u);
    const std::uint64_t ubatch = std::min<std::uint64_t>(p.ubatch_size.value_or(512), batch);
    // Recurrent rows per layer: one state per sequence, times (1 + n_rs_seq)
    // snapshots when the speculative types roll recurrent state back.
    e.recurrent_snapshots = 1 + recurrent_rollback_snapshots(m, p);
    const std::uint64_t recurrent_rows = static_cast<std::uint64_t>(slots) * e.recurrent_snapshots;
    const auto sliding = [&m](std::uint32_t i) { return i < m.swa_layers.size() && m.swa_layers[i]; };
    // With flash attention off llama.cpp stores V transposed and gives every
    // layer the widest layer's V row (n_embd_v_gqa_max over all blocks;
    // llama_kv_cache, [TAG_V_CACHE_VARIABLE]). `auto` and `on` allocate the
    // cache before flash attention is resolved, with each layer's own row.
    const bool v_padded = p.flash_attn && (*p.flash_attn == "off" || *p.flash_attn == "disabled");
    double v_row_max = 0.0;
    if (v_padded) {
        for (std::uint32_t i = 0; i < n_layer; ++i) {
            const double v = static_cast<double>(sliding(i) ? m.value_length_swa : m.value_length);
            v_row_max = std::max(v_row_max, static_cast<double>(m.kv_heads[i]) * v);
        }
    }
    double kv_per_token = 0.0;      // full-attention layers on the GPU
    double swa_kv_per_token = 0.0;  // sliding-window layers on the GPU
    for (std::uint32_t i = first_gpu; i < n_layer; ++i) {
        if (i >= first_nextn && !mtp) continue;  // MTP blocks load only for draft-mtp
        e.weights_bytes += m.block_bytes[i];
        ++e.gpu_layers;
        if (m.kv_heads[i] > 0) {
            ++e.attention_layers;
            const bool swa = sliding(i);
            const double heads = static_cast<double>(m.kv_heads[i]);
            const double k = static_cast<double>(swa ? m.key_length_swa : m.key_length);
            const double v = static_cast<double>(swa ? m.value_length_swa : m.value_length);
            const double bytes = v_padded ? heads * k * bk + v_row_max * bv : heads * (k * bk + v * bv);
            if (swa) {
                ++e.swa_attention_layers;
                swa_kv_per_token += bytes;
            } else {
                kv_per_token += bytes;
            }
        } else if (m.kind != ModelArchitecture::attention_only) {
            e.recurrent_bytes += m.recurrent_state_values * 4 * recurrent_rows;
        }
    }
    if (all) e.weights_bytes += m.output_bytes;
    e.kv_bytes_per_token = saturating_u64(kv_per_token);
    e.swa_kv_bytes_per_token = saturating_u64(swa_kv_per_token);
    double kv = kv_per_token * static_cast<double>(ctx_per_sequence) * e.sequences;
    if (e.swa_attention_layers > 0) {
        // Full size with --swa-full (judged under the name llama-server
        // reads), a caller-supplied inherited setting, and on the direct backend, which keeps the llama.cpp
        // library's default swa_full = true (llama_context_default_params,
        // b11195). llama-server's own default is false.
        bool swa_full_flag = false;
        for (const auto& arg : p.extra_args) swa_full_flag = swa_full_flag || llamaserver_flag_name(arg) == "--swa-full";
        const bool direct = p.backend == kLaunchProfileBackendLlamaCpp;
        if (direct || swa_full_flag || inputs.inherited_swa_full) {
            e.swa_context_per_sequence = ctx_per_sequence;
            e.notes.emplace_back(direct ? "the llamacpp backend allocates full-size sliding-window caches (llama.cpp's "
                                          "default swa_full); sliding-window layers are counted at full context"
                                        : swa_full_flag ? "--swa-full: sliding-window layers are counted at full context"
                                                        : "LLAMA_ARG_SWA_FULL is set: sliding-window layers are counted at full context");
        } else {
            // size_swa: window x n_seq_max for one unified stream, the window
            // for each stream of a split cache, plus the micro-batch. Below
            // ctx_per_sequence, a multiple of 256, so the padding cannot wrap.
            const std::uint64_t window = saturating_add(saturating_mul(m.sliding_window, unified ? slots : 1u), ubatch);
            const std::uint64_t cells = std::min(ctx_per_sequence, window);
            e.swa_context_per_sequence = (cells + 255) / 256 * 256;
        }
        kv += swa_kv_per_token * static_cast<double>(e.swa_context_per_sequence) * e.sequences;
    }
    e.kv_bytes = saturating_u64(kv);
    if (m.sliding_window_unmodelled && e.attention_layers > 0) {
        e.notes.emplace_back(m.architecture +
                             ".attention.sliding_window is set, but sliding-window caches are modelled only for "
                             "architectures checked against llama.cpp; every layer is counted at full context");
    }

    if (!p.mmproj.empty() && p.mmproj_offload.value_or(true)) {
        e.weights_bytes += inputs.mmproj_bytes;
        e.notes.emplace_back("vision encoder compute buffers are not included");
    }
    if (p.speculative && !p.speculative->draft_model.empty()) {
        e.weights_bytes += inputs.draft_model_bytes;
        e.notes.emplace_back("the draft model's KV cache is not included");
    }
    if (mtp && m.nextn_layers > 0) {
        e.notes.emplace_back("the MTP draft context's compute buffer, which grows with the context, is not included");
    }
    if (offloaded > 0) {
        e.compute_bytes = m.vocab_size * ubatch * 4 + 4 * m.embedding_length * ubatch * 4 + 256 * kMiB;
    }
    for (const auto& arg : p.extra_args) {
        // The name llama-server reads (`--override_tensor` is `--override-tensor`).
        const std::string flag = llamaserver_flag_name(arg);
        if (flag == "-ot" || flag == "--override-tensor" || flag == "-cmoe" || flag == "--cpu-moe" ||
            flag == "-ncmoe" || flag == "--n-cpu-moe" || flag == "-ts" || flag == "--tensor-split" ||
            flag == "-sm" || flag == "--split-mode" || flag == "-dev" || flag == "--device") {
            e.notes.emplace_back("placement flags in extra_args (" + flag +
                                 ") are ignored: the estimate assumes every offloaded block is on one GPU");
            break;
        }
    }
    if (m.kind != ModelArchitecture::attention_only && m.recurrent_state_values == 0) {
        e.notes.emplace_back("recurrent state size unknown for this architecture");
    }
    if (auto_parallel && e.recurrent_bytes > 0) {
        e.notes.emplace_back("parallel unset: llama-server's automatic 4 slots keep 4 recurrent states");
    }
    e.total_bytes = e.weights_bytes + e.kv_bytes + e.recurrent_bytes + e.compute_bytes;
    return e;
}

std::optional<std::uint64_t> detect_device_vram_mib() {
#if defined(_WIN32) && defined(_MSC_VER)
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory) {
        return std::nullopt;
    }
    std::uint64_t best = 0;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)) && adapter != nullptr; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            best = std::max<std::uint64_t>(best, static_cast<std::uint64_t>(desc.DedicatedVideoMemory));
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    if (best == 0) return std::nullopt;
    return best / kMiB;
#else
    return std::nullopt;
#endif
}

}  // namespace sonder::inference
