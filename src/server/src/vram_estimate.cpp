// GGUF header reader and the VRAM fit estimate for launch profiles
// (launch_profile.hpp). Reads only the header: metadata and the tensor
// table, never tensor data.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include "sonder/inference/launch_profile.hpp"

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

Status malformed(const std::string& what) { return Status(ErrorCode::invalid_argument, "GGUF header: " + what); }

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

class Cursor {
public:
    explicit Cursor(std::istream& in) : in_(in) {}

    bool bytes(void* out, std::size_t n) {
        in_.read(static_cast<char*>(out), static_cast<std::streamsize>(n));
        return static_cast<std::size_t>(in_.gcount()) == n;
    }
    bool skip(std::uint64_t n) {
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
        if (!pod(n) || n > kMaxStringBytes) return false;
        out.resize(static_cast<std::size_t>(n));
        return n == 0 || bytes(out.data(), out.size());
    }
    bool skip_string() {
        std::uint64_t n = 0;
        return pod(n) && n <= kMaxStringBytes && skip(n);
    }

private:
    std::istream& in_;
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

Result<GgufModelInfo> parse(std::istream& in) {
    Cursor c(in);
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
    struct Numbers {
        std::int64_t value = -1;
        std::vector<std::int64_t> per_layer;
    };
    std::vector<std::pair<std::string, Numbers>> numbers;  // keys we may need
    std::uint64_t token_count = 0;
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        std::uint32_t type = 0;
        if (!c.string(key) || !c.pod(type)) return malformed("truncated metadata");
        if (type == 8) {
            std::string value;
            if (!c.string(value)) return malformed("truncated string value");
            if (key == "general.architecture") info.architecture = value;
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
            if (size == 0) return malformed("unsupported array item type in '" + key + "'");
            if (item_type != 6 && item_type != 12 && item_type != 7 && n <= 4096) {
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
                info.metadata.emplace_back(key, std::to_string(max_value));
                numbers.emplace_back(key, std::move(num));
                continue;
            }
            if (!c.skip(size * n)) return malformed("truncated array");
            continue;
        }
        std::string text;
        std::int64_t v = 0;
        bool is_int = false;
        if (!read_scalar(c, type, text, v, is_int)) return malformed("unsupported value type in '" + key + "'");
        info.metadata.emplace_back(key, text);
        if (is_int) numbers.emplace_back(key, Numbers{v, {}});
    }
    if (info.architecture.empty()) return malformed("general.architecture is missing");
    const std::string a = info.architecture + ".";
    const auto number = [&](const std::string& key) -> const Numbers* {
        for (const auto& [k, n] : numbers) {
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
    info.kv_heads.assign(info.block_count, 0);
    if (const Numbers* kv = number(a + "attention.head_count_kv"); kv && !kv->per_layer.empty()) {
        for (std::size_t i = 0; i < info.kv_heads.size() && i < kv->per_layer.size(); ++i) {
            info.kv_heads[i] = static_cast<std::uint32_t>(std::max<std::int64_t>(0, kv->per_layer[i]));
        }
    } else {
        const std::int64_t kv_heads = scalar(a + "attention.head_count_kv", heads);
        std::fill(info.kv_heads.begin(), info.kv_heads.end(), static_cast<std::uint32_t>(std::max<std::int64_t>(0, kv_heads)));
    }
    // Recurrent state (Mamba 1/2, delta-net): conv (kernel - 1) x channels,
    // channels = inner + 2 x groups x state, plus inner x state.
    const std::int64_t conv = scalar(a + "ssm.conv_kernel", 0);
    const std::int64_t inner = scalar(a + "ssm.inner_size", 0);
    const std::int64_t state = scalar(a + "ssm.state_size", 0);
    const std::int64_t groups = scalar(a + "ssm.group_count", 0);
    if (conv > 0 && inner > 0 && state > 0) {
        info.recurrent_state_values = static_cast<std::uint64_t>((conv - 1) * (inner + 2 * groups * state) + inner * state);
    }

    // Tensor table.
    std::vector<bool> attention_tensor(info.block_count, false);
    bool any_attention_tensor = false;
    info.block_bytes.assign(info.block_count, 0);
    std::uint64_t output_vocab = 0;
    for (std::uint64_t t = 0; t < n_tensors; ++t) {
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
            return malformed("tensor " + name + " has an unknown ggml type " + std::to_string(type));
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
            if (!ok || layer >= info.block_count) return malformed("tensor " + name + " names an unknown block");
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
            bool attention = false;
            if (any_attention_tensor) {
                attention = attention_tensor[i];
            } else if (interval > 0) {
                attention = (static_cast<std::int64_t>(i) + 1) % interval == 0;
            }
            if (!attention) info.kv_heads[i] = 0;
        }
    }
    return info;
}

}  // namespace

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
    const bool unified = p.kv_unified.value_or(!p.parallel.has_value());
    const std::uint32_t slots = p.parallel.value_or(1);
    e.sequences = unified ? 1 : slots;
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
    double kv_per_token = 0.0;
    for (std::uint32_t i = first_gpu; i < n_layer; ++i) {
        if (i >= first_nextn && !mtp) continue;  // MTP blocks load only for draft-mtp
        e.weights_bytes += m.block_bytes[i];
        ++e.gpu_layers;
        if (m.kv_heads[i] > 0) {
            ++e.attention_layers;
            kv_per_token += static_cast<double>(m.kv_heads[i]) *
                            (static_cast<double>(m.key_length) * bk + static_cast<double>(m.value_length) * bv);
        } else if (m.kind != ModelArchitecture::attention_only) {
            e.recurrent_bytes += m.recurrent_state_values * 4 * slots;
        }
    }
    if (all) e.weights_bytes += m.output_bytes;
    e.kv_bytes_per_token = static_cast<std::uint64_t>(kv_per_token + 0.5);
    e.kv_bytes =
        static_cast<std::uint64_t>(kv_per_token * static_cast<double>(ctx_per_sequence) * e.sequences + 0.5);

    if (!p.mmproj.empty() && p.mmproj_offload.value_or(true)) {
        e.weights_bytes += inputs.mmproj_bytes;
        e.notes.emplace_back("vision encoder compute buffers are not included");
    }
    if (p.speculative && !p.speculative->draft_model.empty()) {
        e.weights_bytes += inputs.draft_model_bytes;
        e.notes.emplace_back("the draft model's KV cache is not included");
    }
    if (offloaded > 0) {
        const std::uint64_t batch = p.batch_size.value_or(p.backend == kLaunchProfileBackendLlamaCpp ? 512u : 2048u);
        const std::uint64_t ubatch = std::min<std::uint64_t>(p.ubatch_size.value_or(512), batch);
        e.compute_bytes = m.vocab_size * ubatch * 4 + 4 * m.embedding_length * ubatch * 4 + 256 * kMiB;
    }
    for (const auto& arg : p.extra_args) {
        const std::string_view flag = std::string_view(arg).substr(0, arg.find('='));
        if (flag == "-ot" || flag == "--override-tensor" || flag == "-cmoe" || flag == "--cpu-moe" ||
            flag == "-ncmoe" || flag == "--n-cpu-moe" || flag == "-ts" || flag == "--tensor-split" ||
            flag == "-sm" || flag == "--split-mode" || flag == "-dev" || flag == "--device") {
            e.notes.emplace_back("placement flags in extra_args (" + std::string(flag) +
                                 ") are ignored: the estimate assumes every offloaded block is on one GPU");
            break;
        }
    }
    if (m.kind != ModelArchitecture::attention_only && m.recurrent_state_values == 0) {
        e.notes.emplace_back("recurrent state size unknown for this architecture");
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
