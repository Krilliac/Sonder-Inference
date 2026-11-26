#include "tool_call_parser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "json_schema_gbnf.hpp"

namespace sonder::inference::server::detail {

namespace {

ApiError bad_req(std::string code, std::string message, std::optional<std::string> param = std::nullopt) {
    return make_error(400, std::move(code), std::move(message), std::move(param));
}

bool is_fence_open_boundary(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '{' || c == '[';
}

std::string_view trim_ws(std::string_view s) noexcept {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r')) {
        s.remove_suffix(1);
    }
    return s;
}

std::string lower_ascii(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

constexpr std::string_view kXmlOpen = "<tool_call>";
constexpr std::string_view kXmlClose = "</tool_call>";
constexpr std::string_view kFenceOpen = "```tool_call";
constexpr std::string_view kFenceClose = "```";

// Longest suffix of `s` that is a proper prefix of either opening tag.
std::size_t trailing_partial_open_len(std::string_view s) noexcept {
    std::size_t best = 0;
    for (const std::string_view tag : {kXmlOpen, kFenceOpen}) {
        const std::size_t max_k = std::min(s.size(), tag.size() - 1);
        for (std::size_t k = max_k; k > best; --k) {
            if (s.compare(s.size() - k, k, tag.data(), k) == 0) {
                best = k;
                break;
            }
        }
    }
    return best;
}

// Parses one tool call JSON object {"name": "...", "arguments": ...} or array of them.
bool parse_tool_call_objects(std::string_view payload, std::vector<std::pair<std::string, std::string>>& out) {
    const std::string_view trimmed = trim_ws(payload);
    if (trimmed.empty()) {
        return false;
    }
    auto parsed = json::parse(trimmed, json::ParseLimits{64, 16384});
    if (!parsed.ok()) {
        return false;
    }
    auto extract_one = [&](const json::Value& obj) -> bool {
        if (!obj.is_object()) return false;
        const json::Value* name = obj.find("name");
        if (name == nullptr || !name->is_string() || !is_valid_tool_name(name->as_string())) return false;
        const json::Value* args = obj.find("arguments");
        std::string args_str = "{}";
        if (args != nullptr && !args->is_null()) {
            if (args->is_string()) {
                auto parsed_args = json::parse(args->as_string(), json::ParseLimits{64, 16384});
                if (!parsed_args.ok() || !parsed_args.value().is_object()) {
                    return false;
                }
                args_str = args->as_string();
            } else if (args->is_object()) {
                args_str = args->dump();
            } else {
                return false;
            }
        }
        out.emplace_back(name->as_string(), std::move(args_str));
        return true;
    };
    if (parsed.value().is_object()) {
        return extract_one(parsed.value());
    }
    if (parsed.value().is_array() && !parsed.value().as_array().empty() &&
        parsed.value().as_array().size() <= kMaxToolCallsPerMessage) {
        for (const auto& item : parsed.value().as_array()) {
            if (!extract_one(item)) {
                out.clear();
                return false;
            }
        }
        return true;
    }
    return false;
}

json::Array utf8_bytes_json(std::string_view s) {
    json::Array bytes;
    bytes.reserve(s.size());
    for (const char c : s) {
        bytes.emplace_back(static_cast<int>(static_cast<unsigned char>(c)));
    }
    return bytes;
}

}  // namespace

bool is_valid_tool_name(std::string_view name) noexcept {
    if (name.empty() || name.size() > 128) {
        return false;
    }
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-' || c == '.';
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool is_valid_tool_call_id(std::string_view id) noexcept {
    if (id.empty() || id.size() > 128) {
        return false;
    }
    for (const char c : id) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20u || u == 0x7Fu) {
            return false;
        }
    }
    return true;
}

bool is_supported_image_mime(std::string_view mime) noexcept {
    return mime == "image/png" || mime == "image/jpeg" || mime == "image/webp" || mime == "image/gif";
}

std::variant<std::vector<std::uint8_t>, ApiError> decode_strict_base64(std::string_view b64,
                                                                       std::size_t max_decoded_bytes,
                                                                       std::string_view param) {
    if (b64.empty()) {
        return bad_req("invalid_request", "base64 image data must not be empty", std::string(param));
    }
    if ((b64.size() % 4u) != 0u) {
        return bad_req("invalid_request", "invalid base64 image data: length must be a multiple of 4",
                       std::string(param));
    }
    const std::size_t pad = (b64.back() == '=' ? 1u : 0u) + (b64[b64.size() - 2] == '=' ? 1u : 0u);
    const std::size_t decoded_len = (b64.size() / 4u) * 3u - pad;
    if (decoded_len == 0u) {
        return bad_req("invalid_request", "base64 image data decoded to 0 bytes", std::string(param));
    }
    if (decoded_len > max_decoded_bytes) {
        return bad_req("invalid_request",
                       "decoded image exceeds maximum size of " + std::to_string(max_decoded_bytes) + " bytes",
                       std::string(param));
    }

    static constexpr auto kTable = []() {
        std::array<std::int8_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; ++c) t[static_cast<std::size_t>(c)] = static_cast<std::int8_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; ++c) t[static_cast<std::size_t>(c)] = static_cast<std::int8_t>(c - 'a' + 26);
        for (int c = '0'; c <= '9'; ++c) t[static_cast<std::size_t>(c)] = static_cast<std::int8_t>(c - '0' + 52);
        t[static_cast<std::size_t>('+')] = 62;
        t[static_cast<std::size_t>('/')] = 63;
        return t;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(decoded_len);
    for (std::size_t i = 0; i < b64.size(); i += 4) {
        const bool last_quad = (i + 4 == b64.size());
        const unsigned char c0 = static_cast<unsigned char>(b64[i]);
        const unsigned char c1 = static_cast<unsigned char>(b64[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(b64[i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(b64[i + 3]);

        const int v0 = kTable[c0];
        const int v1 = kTable[c1];
        if (v0 < 0 || v1 < 0) {
            return bad_req("invalid_request", "invalid base64 character in image data", std::string(param));
        }

        const bool p2 = (c2 == '=');
        const bool p3 = (c3 == '=');
        if ((p2 || p3) && !last_quad) {
            return bad_req("invalid_request", "unexpected '=' padding before end of base64 image data",
                           std::string(param));
        }
        if (p2 && !p3) {
            return bad_req("invalid_request", "malformed base64 padding in image data", std::string(param));
        }
        const int v2 = p2 ? 0 : kTable[c2];
        const int v3 = p3 ? 0 : kTable[c3];
        if ((!p2 && v2 < 0) || (!p3 && v3 < 0)) {
            return bad_req("invalid_request", "invalid base64 character in image data", std::string(param));
        }
        // Strict canonical padding bit check (unused trailing bits must be 0).
        if (p2 && (v1 & 0x0F) != 0) {
            return bad_req("invalid_request", "non-canonical base64 padding bits in image data", std::string(param));
        }
        if (!p2 && p3 && (v2 & 0x03) != 0) {
            return bad_req("invalid_request", "non-canonical base64 padding bits in image data", std::string(param));
        }

        const std::uint32_t triple = (static_cast<std::uint32_t>(v0) << 18) | (static_cast<std::uint32_t>(v1) << 12) |
                                     (static_cast<std::uint32_t>(v2) << 6) | static_cast<std::uint32_t>(v3);
        out.push_back(static_cast<std::uint8_t>((triple >> 16) & 0xFFu));
        if (!p2) out.push_back(static_cast<std::uint8_t>((triple >> 8) & 0xFFu));
        if (!p3) out.push_back(static_cast<std::uint8_t>(triple & 0xFFu));
    }
    return out;
}

std::string encode_base64(std::span<const std::uint8_t> bytes) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((bytes.size() + 2u) / 3u) * 4u);
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const std::uint32_t b0 = bytes[i];
        const bool has1 = (i + 1 < bytes.size());
        const bool has2 = (i + 2 < bytes.size());
        const std::uint32_t b1 = has1 ? bytes[i + 1] : 0u;
        const std::uint32_t b2 = has2 ? bytes[i + 2] : 0u;
        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(triple >> 18) & 0x3Fu]);
        out.push_back(kAlphabet[(triple >> 12) & 0x3Fu]);
        out.push_back(has1 ? kAlphabet[(triple >> 6) & 0x3Fu] : '=');
        out.push_back(has2 ? kAlphabet[triple & 0x3Fu] : '=');
    }
    return out;
}

std::variant<MediaContent, ApiError> parse_data_image_uri(std::string_view url, std::string_view param) {
    const std::string lower_prefix = lower_ascii(url.substr(0, std::min<std::size_t>(url.size(), 16)));
    if (lower_prefix.rfind("http://", 0) == 0 || lower_prefix.rfind("https://", 0) == 0 ||
        lower_prefix.rfind("file://", 0) == 0 || lower_prefix.rfind("ftp://", 0) == 0) {
        return bad_req("invalid_request",
                       "remote or local file image URLs are forbidden (SSRF protection); only inline "
                       "data:image/<png|jpeg|webp|gif>;base64,... URIs are accepted",
                       std::string(param));
    }
    if (url.rfind("data:", 0) != 0) {
        return bad_req("invalid_request",
                       "image_url.url must be an inline data:image/<png|jpeg|webp|gif>;base64,... URI",
                       std::string(param));
    }
    const std::size_t semi = url.find(";base64,", 5);
    if (semi == std::string_view::npos) {
        return bad_req("invalid_request",
                       "image_url.url must use base64 encoding (data:image/<png|jpeg|webp|gif>;base64,...)",
                       std::string(param));
    }
    const std::string_view mime = url.substr(5, semi - 5);
    if (!is_supported_image_mime(mime)) {
        return bad_req("invalid_request",
                       "unsupported image MIME type '" + std::string(mime) +
                           "' (allowed: image/png, image/jpeg, image/webp, image/gif)",
                       std::string(param));
    }
    const std::string_view payload = url.substr(semi + 8);
    auto decoded = decode_strict_base64(payload, kMaxImageDecodedBytes, param);
    if (auto* err = std::get_if<ApiError>(&decoded)) {
        return *err;
    }
    MediaContent media;
    media.mime_type = std::string(mime);
    media.bytes = std::move(std::get<std::vector<std::uint8_t>>(decoded));
    return media;
}

std::variant<MediaContent, ApiError> parse_anthropic_image_block(const json::Object& block, std::string_view param) {
    const json::Value* src = block.find("source");
    if (src == nullptr || !src->is_object()) {
        return bad_req("invalid_request", std::string(param) + ".source must be an object", std::string(param));
    }
    const json::Value* src_type = src->find("type");
    if (src_type == nullptr || !src_type->is_string()) {
        return bad_req("invalid_request", std::string(param) + ".source.type must be a string",
                       std::string(param) + ".source.type");
    }
    if (src_type->as_string() != "base64") {
        return bad_req("invalid_request",
                       "only inline base64 image sources are accepted (remote/url image sources are forbidden for "
                       "SSRF protection)",
                       std::string(param) + ".source.type");
    }
    const json::Value* media_type = src->find("media_type");
    if (media_type == nullptr || !media_type->is_string() || !is_supported_image_mime(media_type->as_string())) {
        return bad_req("invalid_request",
                       std::string(param) +
                           ".source.media_type must be image/png, image/jpeg, image/webp, or image/gif",
                       std::string(param) + ".source.media_type");
    }
    const json::Value* data = src->find("data");
    if (data == nullptr || !data->is_string()) {
        return bad_req("invalid_request", std::string(param) + ".source.data must be a base64 string",
                       std::string(param) + ".source.data");
    }
    auto decoded = decode_strict_base64(data->as_string(), kMaxImageDecodedBytes, std::string(param) + ".source.data");
    if (auto* err = std::get_if<ApiError>(&decoded)) {
        return *err;
    }
    MediaContent media;
    media.mime_type = media_type->as_string();
    media.bytes = std::move(std::get<std::vector<std::uint8_t>>(decoded));
    return media;
}

std::optional<ApiError> parse_openai_tools(const json::Value& value, std::vector<ToolDefinition>& out,
                                           bool allow_flat_responses_format) {
    if (!value.is_array()) {
        return bad_req("invalid_request", "tools must be an array", "tools");
    }
    const auto& arr = value.as_array();
    if (arr.size() > kMaxToolsPerRequest) {
        return bad_req("invalid_request",
                       "tools exceeds maximum of " + std::to_string(kMaxToolsPerRequest) + " definitions", "tools");
    }
    out.clear();
    out.reserve(arr.size());
    std::set<std::string> seen_names;
    for (std::size_t i = 0; i < arr.size(); ++i) {
        const std::string where = "tools[" + std::to_string(i) + "]";
        if (!arr[i].is_object()) {
            return bad_req("invalid_request", where + " must be an object", where);
        }
        const json::Value* type = arr[i].find("type");
        if (type == nullptr || !type->is_string() || type->as_string() != "function") {
            return bad_req("invalid_request", where + ".type must be 'function'", where + ".type");
        }
        const json::Value* fn_obj = arr[i].find("function");
        const json::Object* spec = nullptr;
        std::string spec_prefix = where + ".function";
        if (fn_obj != nullptr && !fn_obj->is_null()) {
            if (!fn_obj->is_object()) {
                return bad_req("invalid_request", where + ".function must be an object", where + ".function");
            }
            spec = &fn_obj->as_object();
        } else if (allow_flat_responses_format) {
            spec = &arr[i].as_object();
            spec_prefix = where;
        } else {
            return bad_req("invalid_request", where + ".function is required", where + ".function");
        }
        const json::Value* name = spec->find("name");
        if (name == nullptr || !name->is_string() || !is_valid_tool_name(name->as_string())) {
            return bad_req("invalid_request",
                           spec_prefix + ".name must be 1 to 128 characters of [A-Za-z0-9_.-]", spec_prefix + ".name");
        }
        if (!seen_names.insert(name->as_string()).second) {
            return bad_req("invalid_request", "duplicate tool name '" + name->as_string() + "'", spec_prefix + ".name");
        }
        ToolDefinition def;
        def.name = name->as_string();
        if (const json::Value* desc = spec->find("description"); desc != nullptr && !desc->is_null()) {
            if (!desc->is_string()) {
                return bad_req("invalid_request", spec_prefix + ".description must be a string",
                               spec_prefix + ".description");
            }
            def.description = desc->as_string();
        }
        if (const json::Value* params = spec->find("parameters"); params != nullptr && !params->is_null()) {
            if (!params->is_object()) {
                return bad_req("invalid_request", spec_prefix + ".parameters must be a JSON Schema object",
                               spec_prefix + ".parameters");
            }
            def.parameters_json = params->dump();
        }
        out.push_back(std::move(def));
    }
    return std::nullopt;
}

std::optional<ApiError> parse_openai_tool_choice(const json::Value& value, const std::vector<ToolDefinition>& tools,
                                                 ToolChoice& out, bool allow_flat_responses_format) {
    out = ToolChoice{};
    if (value.is_string()) {
        const std::string& mode = value.as_string();
        if (mode == "auto") {
            out.mode = ToolChoiceMode::auto_choice;
            return std::nullopt;
        }
        if (mode == "none") {
            out.mode = ToolChoiceMode::none;
            return std::nullopt;
        }
        if (mode == "required") {
            if (tools.empty()) {
                return bad_req("invalid_request", "tool_choice 'required' requires at least one tool", "tool_choice");
            }
            out.mode = ToolChoiceMode::required;
            return std::nullopt;
        }
        return bad_req("invalid_request", "tool_choice must be 'auto', 'none', 'required', or a function object",
                       "tool_choice");
    }
    if (value.is_object()) {
        const json::Value* type = value.find("type");
        if (type == nullptr || !type->is_string() || type->as_string() != "function") {
            return bad_req("invalid_request", "tool_choice.type must be 'function'", "tool_choice.type");
        }
        const json::Value* name = nullptr;
        if (const json::Value* fn = value.find("function"); fn != nullptr && !fn->is_null()) {
            if (!fn->is_object()) {
                return bad_req("invalid_request", "tool_choice.function must be an object", "tool_choice.function");
            }
            name = fn->find("name");
        } else if (allow_flat_responses_format) {
            name = value.find("name");
        }
        if (name == nullptr || !name->is_string() || name->as_string().empty()) {
            return bad_req("invalid_request", "tool_choice function name must be a non-empty string", "tool_choice");
        }
        const bool exists = std::any_of(tools.begin(), tools.end(),
                                        [&](const ToolDefinition& t) { return t.name == name->as_string(); });
        if (!exists) {
            return bad_req("invalid_request",
                           "tool_choice function '" + name->as_string() + "' is not defined in tools", "tool_choice");
        }
        out.mode = ToolChoiceMode::function;
        out.function_name = name->as_string();
        return std::nullopt;
    }
    return bad_req("invalid_request", "tool_choice must be a string or object", "tool_choice");
}

std::optional<ApiError> parse_openai_message_tool_calls(const json::Value& value, std::vector<ToolCall>& out,
                                                        std::string_view param) {
    if (!value.is_array()) {
        return bad_req("invalid_messages", std::string(param) + " must be an array", std::string(param));
    }
    if (value.as_array().size() > kMaxToolCallsPerMessage) {
        return bad_req("invalid_messages",
                       std::string(param) + " exceeds maximum of " + std::to_string(kMaxToolCallsPerMessage) +
                           " tool calls",
                       std::string(param));
    }
    out.clear();
    out.reserve(value.as_array().size());
    for (std::size_t i = 0; i < value.as_array().size(); ++i) {
        const std::string where = std::string(param) + "[" + std::to_string(i) + "]";
        const json::Value& item = value.as_array()[i];
        if (!item.is_object()) {
            return bad_req("invalid_messages", where + " must be an object", where);
        }
        const json::Value* id = item.find("id");
        if (id == nullptr || !id->is_string() || !is_valid_tool_call_id(id->as_string())) {
            return bad_req("invalid_messages",
                           where + ".id must be a non-empty string of at most 128 bytes without control characters",
                           where + ".id");
        }
        const json::Value* type = item.find("type");
        if (type != nullptr && !type->is_null() && (!type->is_string() || type->as_string() != "function")) {
            return bad_req("invalid_messages", where + ".type must be 'function'", where + ".type");
        }
        const json::Value* fn = item.find("function");
        if (fn == nullptr || !fn->is_object()) {
            return bad_req("invalid_messages", where + ".function must be an object", where + ".function");
        }
        const json::Value* name = fn->find("name");
        if (name == nullptr || !name->is_string() || !is_valid_tool_name(name->as_string())) {
            return bad_req("invalid_messages",
                           where + ".function.name must be 1 to 128 characters of [A-Za-z0-9_.-]",
                           where + ".function.name");
        }
        const json::Value* args = fn->find("arguments");
        std::string args_str = "{}";
        if (args != nullptr && !args->is_null()) {
            if (!args->is_string()) {
                return bad_req("invalid_messages", where + ".function.arguments must be a JSON string",
                               where + ".function.arguments");
            }
            if (args->as_string().size() > kMaxToolCallPayloadBytes) {
                return bad_req("invalid_messages",
                               where + ".function.arguments exceeds maximum size of " +
                                   std::to_string(kMaxToolCallPayloadBytes) + " bytes",
                               where + ".function.arguments");
            }
            args_str = args->as_string();
        }
        out.push_back(ToolCall{id->as_string(), name->as_string(), std::move(args_str)});
    }
    return std::nullopt;
}

std::optional<ApiError> parse_anthropic_tools(const json::Value& value, std::vector<ToolDefinition>& out) {
    if (!value.is_array()) {
        return bad_req("invalid_request", "tools must be an array", "tools");
    }
    const auto& arr = value.as_array();
    if (arr.size() > kMaxToolsPerRequest) {
        return bad_req("invalid_request",
                       "tools exceeds maximum of " + std::to_string(kMaxToolsPerRequest) + " definitions", "tools");
    }
    out.clear();
    out.reserve(arr.size());
    std::set<std::string> seen_names;
    for (std::size_t i = 0; i < arr.size(); ++i) {
        const std::string where = "tools[" + std::to_string(i) + "]";
        if (!arr[i].is_object()) {
            return bad_req("invalid_request", where + " must be an object", where);
        }
        const json::Value* name = arr[i].find("name");
        if (name == nullptr || !name->is_string() || !is_valid_tool_name(name->as_string())) {
            return bad_req("invalid_request", where + ".name must be 1 to 128 characters of [A-Za-z0-9_.-]",
                           where + ".name");
        }
        if (!seen_names.insert(name->as_string()).second) {
            return bad_req("invalid_request", "duplicate tool name '" + name->as_string() + "'", where + ".name");
        }
        ToolDefinition def;
        def.name = name->as_string();
        if (const json::Value* desc = arr[i].find("description"); desc != nullptr && !desc->is_null()) {
            if (!desc->is_string()) {
                return bad_req("invalid_request", where + ".description must be a string", where + ".description");
            }
            def.description = desc->as_string();
        }
        if (const json::Value* schema = arr[i].find("input_schema"); schema != nullptr && !schema->is_null()) {
            if (!schema->is_object()) {
                return bad_req("invalid_request", where + ".input_schema must be an object", where + ".input_schema");
            }
            def.parameters_json = schema->dump();
        }
        out.push_back(std::move(def));
    }
    return std::nullopt;
}

std::optional<ApiError> parse_anthropic_tool_choice(const json::Value& value, const std::vector<ToolDefinition>& tools,
                                                    ToolChoice& out) {
    if (!value.is_object()) {
        return bad_req("invalid_request", "tool_choice must be an object", "tool_choice");
    }
    const json::Value* type = value.find("type");
    if (type == nullptr || !type->is_string()) {
        return bad_req("invalid_request", "tool_choice.type must be auto, none, any, or tool", "tool_choice.type");
    }
    const std::string& t = type->as_string();
    out = ToolChoice{};
    if (t == "auto") {
        out.mode = ToolChoiceMode::auto_choice;
        return std::nullopt;
    }
    if (t == "none") {
        out.mode = ToolChoiceMode::none;
        return std::nullopt;
    }
    if (t == "any") {
        if (tools.empty()) {
            return bad_req("invalid_request", "tool_choice 'any' requires at least one tool", "tool_choice");
        }
        out.mode = ToolChoiceMode::required;
        return std::nullopt;
    }
    if (t == "tool") {
        const json::Value* name = value.find("name");
        if (name == nullptr || !name->is_string() || name->as_string().empty()) {
            return bad_req("invalid_request", "tool_choice.name must be a non-empty string", "tool_choice.name");
        }
        const bool exists = std::any_of(tools.begin(), tools.end(),
                                        [&](const ToolDefinition& d) { return d.name == name->as_string(); });
        if (!exists) {
            return bad_req("invalid_request", "tool_choice.name '" + name->as_string() + "' is not defined in tools",
                           "tool_choice.name");
        }
        out.mode = ToolChoiceMode::function;
        out.function_name = name->as_string();
        return std::nullopt;
    }
    return bad_req("invalid_request", "tool_choice.type must be auto, none, any, or tool", "tool_choice.type");
}

std::string format_tools_system_prompt(const std::vector<ToolDefinition>& tools,
                                       const std::optional<ToolChoice>& choice) {
    if (tools.empty()) {
        return {};
    }
    std::string out = "You may call the following functions when helpful. To invoke a function, respond with:\n"
                      "<tool_call>{\"name\": \"<function_name>\", \"arguments\": {<json_arguments>}}</tool_call>\n\n"
                      "Available tools:\n";
    for (const auto& t : tools) {
        out += "- ";
        out += t.name;
        if (!t.description.empty()) {
            out += ": ";
            out += t.description;
        }
        if (!t.parameters_json.empty()) {
            out += "\n  parameters: ";
            out += t.parameters_json;
        }
        out += "\n";
    }
    if (choice) {
        switch (choice->mode) {
            case ToolChoiceMode::none:
                out += "\nDo not call any tool in this response.\n";
                break;
            case ToolChoiceMode::required:
                out += "\nYou MUST call at least one of the available tools in this response.\n";
                break;
            case ToolChoiceMode::function:
                out += "\nYou MUST call the function \"" + choice->function_name + "\" in this response.\n";
                break;
            case ToolChoiceMode::auto_choice:
                break;
        }
    }
    return out;
}

namespace {

std::string canonical_tool_parameters(const ToolDefinition& tool) {
    if (tool.parameters_json.empty()) {
        return R"({"type":"object"})";
    }
    auto parsed = json::parse(tool.parameters_json);
    if (parsed.ok()) {
        return canonical_json_dump(parsed.value());
    }
    return tool.parameters_json;
}

std::string build_tool_call_cache_key(const std::vector<ToolDefinition>& tools, const ToolChoice& choice) {
    json::Array tool_entries;
    if (choice.mode == ToolChoiceMode::function) {
        for (const auto& t : tools) {
            if (t.name == choice.function_name) {
                tool_entries.emplace_back(json::Object{
                    {"name", t.name},
                    {"parameters", canonical_tool_parameters(t)},
                });
                break;
            }
        }
    } else {
        tool_entries.reserve(tools.size());
        for (const auto& t : tools) {
            tool_entries.emplace_back(json::Object{
                {"name", t.name},
                {"parameters", canonical_tool_parameters(t)},
            });
        }
    }
    json::Object key_doc{
        {"mode", choice.mode == ToolChoiceMode::function ? "function" : "required"},
        {"function_name", choice.function_name},
        {"tools", std::move(tool_entries)},
    };
    return canonical_json_dump(json::Value(std::move(key_doc)));
}

Result<CompiledGrammar> compile_tool_call_grammar_uncached(const std::vector<ToolDefinition>& tools,
                                                           const ToolChoice& choice,
                                                           std::string_view canonical_key,
                                                           std::string_view tokenizer_identity,
                                                           const StructuredGrammarEngine& engine) {
    if (!engine.capabilities().active || !engine.capabilities().supports_json_schema) {
        return Status(ErrorCode::unsupported, engine.capabilities().detail);
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool relaxed_any = false;

    auto make_call_schema = [&](const ToolDefinition& tool, bool strict_params) -> json::Value {
        json::Value params_schema = json::Object{{"type", "object"}};
        if (strict_params && !tool.parameters_json.empty()) {
            auto parsed = json::parse(tool.parameters_json);
            if (parsed.ok() && parsed.value().is_object()) {
                // Verify the parameter schema itself converts cleanly; if not, caller retries non-strict.
                if (engine.compile_json_schema(parsed.value()).ok()) {
                    params_schema = std::move(parsed).value();
                } else {
                    relaxed_any = true;
                }
            }
        } else if (!strict_params && !tool.parameters_json.empty()) {
            relaxed_any = true;
        }
        return json::Object{{"type", "object"},
                            {"properties", json::Object{{"name", json::Object{{"const", tool.name}}},
                                                        {"arguments", std::move(params_schema)}}},
                            {"required", json::Array{"name", "arguments"}},
                            {"additionalProperties", false}};
    };

    auto build_call_schema = [&](bool strict_params) -> Result<json::Value> {
        if (choice.mode == ToolChoiceMode::function) {
            const ToolDefinition* found = nullptr;
            for (const auto& t : tools) {
                if (t.name == choice.function_name) {
                    found = &t;
                    break;
                }
            }
            if (found == nullptr) {
                return Status(ErrorCode::invalid_argument,
                              "tool_choice function '" + choice.function_name + "' not found");
            }
            return make_call_schema(*found, strict_params);
        }
        json::Array branches;
        branches.reserve(tools.size());
        for (const auto& t : tools) {
            branches.push_back(make_call_schema(t, strict_params));
        }
        return branches.size() == 1 ? std::move(branches.front())
                                    : json::Value(json::Object{{"anyOf", std::move(branches)}});
    };

    auto call_schema = build_call_schema(true);
    if (!call_schema.ok()) {
        return call_schema.status();
    }
    auto base = engine.compile_json_schema(call_schema.value());
    if (!base.ok()) {
        // Retry with non-strict object arguments (e.g., if nested parameter schema
        // contained root-relative $defs/$ref) while still strictly enforcing the
        // forced `<tool_call>{"name":...,"arguments":{...}}</tool_call>` wrapper.
        auto relaxed_schema = build_call_schema(false);
        if (!relaxed_schema.ok()) {
            return relaxed_schema.status();
        }
        base = engine.compile_json_schema(relaxed_schema.value());
        if (!base.ok()) {
            return base.status();
        }
        relaxed_any = true;
    }

    CompiledGrammar out = std::move(base).value();
    const std::size_t nl = out.gbnf.find('\n');
    if (nl == std::string::npos || out.gbnf.rfind("root ::= ", 0) != 0) {
        return Status(ErrorCode::internal, "unexpected GBNF root layout");
    }
    const std::string root_rhs = out.gbnf.substr(9, nl - 9);
    out.gbnf = std::string("root ::= \"<tool_call>\" space (") + root_rhs + ") \"</tool_call>\" space\n" +
               out.gbnf.substr(nl + 1);
    out.payload = out.gbnf;
    out.constraint_kind = GrammarConstraintKind::tool_call;
    out.normalized_schema_key = std::string(canonical_key);
    out.normalized_schema_hash = normalized_schema_hash(GrammarConstraintKind::tool_call, canonical_key);
    out.tokenizer_identity =
        tokenizer_identity.empty() ? std::string("default") : std::string(tokenizer_identity);
    out.capabilities.constraint_kind = GrammarConstraintKind::tool_call;
    out.capabilities.enforces_strict_schema = !relaxed_any;
    out.capabilities.uses_fallback_relaxation = relaxed_any;
    out.compile_duration = std::chrono::steady_clock::now() - t0;
    return out;
}

}  // namespace

Result<std::shared_ptr<const CompiledGrammar>> build_tool_call_grammar(
    const std::vector<ToolDefinition>& tools, const ToolChoice& choice,
    CompiledGrammarCache* cache, std::string_view tokenizer_identity,
    const StructuredGrammarEngine& engine) {
    if (tools.empty()) {
        return Status(ErrorCode::invalid_argument, "cannot build tool call grammar without tools");
    }
    if (choice.mode != ToolChoiceMode::required && choice.mode != ToolChoiceMode::function) {
        return Status(ErrorCode::invalid_argument,
                      "tool call grammar is only built for required or named function tool_choice");
    }
    const std::string canonical_key = build_tool_call_cache_key(tools, choice);
    const std::string tok_id =
        tokenizer_identity.empty() ? std::string("default") : std::string(tokenizer_identity);
    if (cache != nullptr) {
        CompiledGrammarCacheKey key;
        key.engine = engine.kind();
        key.constraint_kind = GrammarConstraintKind::tool_call;
        key.tokenizer_identity = tok_id;
        key.normalized_schema_key = canonical_key;
        key.options_key = GrammarCompileOptions{}.cache_key_fragment();
        return cache->get_or_compile(key, [&]() {
            return compile_tool_call_grammar_uncached(tools, choice, canonical_key, tok_id, engine);
        });
    }
    auto compiled = compile_tool_call_grammar_uncached(tools, choice, canonical_key, tok_id, engine);
    if (!compiled.ok()) {
        return compiled.status();
    }
    return std::make_shared<const CompiledGrammar>(std::move(compiled).value());
}

Result<std::string> build_tool_call_gbnf(const std::vector<ToolDefinition>& tools, const ToolChoice& choice) {
    auto compiled = build_tool_call_grammar(tools, choice);
    if (!compiled.ok()) {
        return compiled.status();
    }
    return compiled.value()->gbnf;
}

Result<PreparedStructuredGrammar> resolve_structured_grammar(
    std::string_view schema_param_prefix,
    std::string_view explicit_grammar,
    const std::optional<std::string>& response_format_type,
    const std::optional<json::Value>& pending_response_schema,
    const std::optional<std::string>& response_format_schema,
    const std::vector<ToolDefinition>& tools,
    const std::optional<ToolChoice>& tool_choice,
    bool backend_supports_structured_output,
    bool backend_native_tools,
    std::string_view backend_name,
    std::string_view tokenizer_identity,
    CompiledGrammarCache* cache,
    const StructuredGrammarEngine& engine) {
    PreparedStructuredGrammar out;
    const std::string tok_id =
        tokenizer_identity.empty() ? std::string("default") : std::string(tokenizer_identity);

    if (response_format_type && *response_format_type == "json_object") {
        if (cache != nullptr) {
            auto res = cache->get_or_compile_json_object(engine, tok_id);
            if (!res.ok()) {
                return res.status();
            }
            out.compiled = std::move(res).value();
        } else {
            auto res = engine.compile_json_object();
            if (!res.ok()) {
                return res.status();
            }
            res.value().tokenizer_identity = tok_id;
            out.compiled = std::make_shared<const CompiledGrammar>(std::move(res).value());
        }
        out.grammar = out.compiled->gbnf;
        return out;
    }

    if (response_format_type && *response_format_type == "json_schema") {
        std::optional<json::Value> parsed_storage;
        const json::Value* schema_ptr = pending_response_schema ? &*pending_response_schema : nullptr;
        if (schema_ptr == nullptr && response_format_schema) {
            auto parsed = json::parse(*response_format_schema);
            if (!parsed.ok() || !parsed.value().is_object()) {
                return Status(ErrorCode::invalid_argument,
                              "invalid " + std::string(schema_param_prefix) + ": schema must be a JSON object");
            }
            parsed_storage = std::move(parsed).value();
            schema_ptr = &*parsed_storage;
        }
        if (schema_ptr == nullptr) {
            return Status(ErrorCode::invalid_argument,
                          "invalid " + std::string(schema_param_prefix) + ": schema is missing");
        }
        if (cache != nullptr) {
            auto res = cache->get_or_compile_schema(engine, *schema_ptr, tok_id);
            if (!res.ok()) {
                return res.status();
            }
            out.compiled = std::move(res).value();
        } else {
            auto res = engine.compile_json_schema(*schema_ptr);
            if (!res.ok()) {
                return res.status();
            }
            res.value().tokenizer_identity = tok_id;
            out.compiled = std::make_shared<const CompiledGrammar>(std::move(res).value());
        }
        out.grammar = out.compiled->gbnf;
        return out;
    }

    if (!explicit_grammar.empty()) {
        out.grammar.assign(explicit_grammar.data(), explicit_grammar.size());
        return out;
    }

    if (!tools.empty() && tool_choice &&
        (tool_choice->mode == ToolChoiceMode::required || tool_choice->mode == ToolChoiceMode::function) &&
        backend_supports_structured_output && !backend_native_tools &&
        backend_name != "llamaserver" && backend_name != "ollama") {
        if (auto compiled = build_tool_call_grammar(tools, *tool_choice, cache, tok_id, engine); compiled.ok()) {
            out.compiled = std::move(compiled).value();
            out.grammar = out.compiled->gbnf;
        }
    }
    return out;
}

// ================================================================= ToolCallStreamParser

ToolCallStreamParser::ToolCallStreamParser(std::string id_prefix) : id_prefix_(std::move(id_prefix)) {}

std::string ToolCallStreamParser::next_call_id() {
    return id_prefix_ + std::to_string(next_index_);
}

void ToolCallStreamParser::try_emit_payload(std::string_view payload, std::string_view raw_close,
                                            ToolCallStreamOutput& out) {
    std::vector<std::pair<std::string, std::string>> calls;
    if (!parse_tool_call_objects(payload, calls)) {
        // Fail-safe: never drop model output that failed JSON tool-call validation.
        out.visible_text += raw_open_;
        out.visible_text += payload;
        out.visible_text += raw_close;
        return;
    }
    for (auto& [name, args] : calls) {
        ParsedToolEvent ev;
        ev.index = next_index_;
        ev.id = next_call_id();
        ++next_index_;
        ev.name = std::move(name);
        ev.arguments_json = std::move(args);
        completed_.push_back(ToolCall{ev.id, ev.name, ev.arguments_json});
        out.tool_calls.push_back(std::move(ev));
    }
}

ToolCallStreamOutput ToolCallStreamParser::feed(std::string_view chunk) {
    ToolCallStreamOutput out;
    if (chunk.empty()) {
        return out;
    }
    buf_.append(chunk.data(), chunk.size());

    for (;;) {
        if (state_ == State::text) {
            const std::size_t xml_pos = buf_.find(kXmlOpen);
            const std::size_t fence_pos = buf_.find(kFenceOpen);
            std::size_t open_pos = std::string::npos;
            bool is_xml = false;
            if (xml_pos != std::string::npos && (fence_pos == std::string::npos || xml_pos <= fence_pos)) {
                open_pos = xml_pos;
                is_xml = true;
            } else if (fence_pos != std::string::npos) {
                open_pos = fence_pos;
                is_xml = false;
            }

            if (open_pos == std::string::npos) {
                const std::size_t keep = trailing_partial_open_len(buf_);
                if (buf_.size() > keep) {
                    out.visible_text.append(buf_.data(), buf_.size() - keep);
                    buf_.erase(0, buf_.size() - keep);
                }
                break;
            }

            if (!is_xml) {
                const std::size_t after_fence = open_pos + kFenceOpen.size();
                if (after_fence == buf_.size()) {
                    // Wait for 1 more byte to distinguish ```tool_call\n from ```tool_call_notes.
                    if (open_pos > 0) {
                        out.visible_text.append(buf_.data(), open_pos);
                        buf_.erase(0, open_pos);
                    }
                    break;
                }
                if (!is_fence_open_boundary(buf_[after_fence])) {
                    // Ordinary markdown fence like ```tool_call_example: emit as plain text.
                    out.visible_text.append(buf_.data(), after_fence);
                    buf_.erase(0, after_fence);
                    continue;
                }
            }

            if (open_pos > 0) {
                out.visible_text.append(buf_.data(), open_pos);
            }
            if (is_xml) {
                raw_open_ = std::string(kXmlOpen);
                buf_.erase(0, open_pos + kXmlOpen.size());
                state_ = State::in_xml;
            } else {
                raw_open_ = std::string(kFenceOpen);
                buf_.erase(0, open_pos + kFenceOpen.size());
                state_ = State::in_fence;
            }
            reset_scan_state();
            continue;
        }

        // Incremental O(N) scan for closing tag (`</tool_call>` or `` ``` ``)
        // outside JSON double-quoted string literals so embedded closing markers
        // inside tool arguments do not prematurely truncate the payload and
        // byte-by-byte streaming stays strictly linear time.
        const std::string_view close_tag = (state_ == State::in_xml) ? kXmlClose : kFenceClose;
        const char lead = close_tag.front();
        std::size_t close_pos = std::string::npos;
        const std::size_t scan_limit = std::min(buf_.size(), kMaxToolCallPayloadBytes + 1u);
        while (scan_pos_ < scan_limit) {
            const char c = buf_[scan_pos_];
            if (in_str_) {
                if (esc_) {
                    esc_ = false;
                } else if (c == '\\') {
                    esc_ = true;
                } else if (c == '"') {
                    in_str_ = false;
                }
                ++scan_pos_;
                continue;
            }
            if (c == '"') {
                in_str_ = true;
                ++scan_pos_;
                continue;
            }
            if (c == lead) {
                const std::size_t rem = buf_.size() - scan_pos_;
                if (rem < close_tag.size()) {
                    if (close_tag.compare(0, rem, buf_.data() + scan_pos_, rem) == 0) {
                        // Partial closing marker right at the chunk boundary: pause here.
                        break;
                    }
                } else if (std::string_view(buf_).substr(scan_pos_, close_tag.size()) == close_tag) {
                    close_pos = scan_pos_;
                    break;
                }
            }
            ++scan_pos_;
        }

        if (close_pos == std::string::npos) {
            if (scan_pos_ > kMaxToolCallPayloadBytes) {
                // Payload exceeded kMaxToolCallPayloadBytes without closing: flush only
                // the bounded overflow prefix as plain text and re-enter text mode on
                // the remainder so subsequent tool calls in the same chunk still parse.
                out.visible_text += raw_open_;
                out.visible_text.append(buf_.data(), kMaxToolCallPayloadBytes);
                buf_.erase(0, kMaxToolCallPayloadBytes);
                raw_open_.clear();
                reset_scan_state();
                state_ = State::text;
                continue;
            }
            break;
        }
        const std::string payload = buf_.substr(0, close_pos);
        buf_.erase(0, close_pos + close_tag.size());
        try_emit_payload(payload, close_tag, out);
        raw_open_.clear();
        reset_scan_state();
        state_ = State::text;
    }

    return out;
}

ToolCallStreamOutput ToolCallStreamParser::flush() {
    ToolCallStreamOutput out;
    if (state_ == State::text) {
        if (!buf_.empty()) {
            out.visible_text += buf_;
            buf_.clear();
        }
    } else {
        if (!buf_.empty()) {
            try_emit_payload(buf_, "", out);
            buf_.clear();
        } else if (!raw_open_.empty()) {
            out.visible_text += raw_open_;
        }
        raw_open_.clear();
        reset_scan_state();
        state_ = State::text;
    }
    return out;
}

ExtractedToolOutput extract_tool_calls(std::string_view full_text, std::string id_prefix) {
    ToolCallStreamParser parser(std::move(id_prefix));
    const ToolCallStreamOutput part1 = parser.feed(full_text);
    const ToolCallStreamOutput part2 = parser.flush();
    ExtractedToolOutput out;
    out.visible_text = part1.visible_text + part2.visible_text;
    out.tool_calls = parser.completed_calls();
    return out;
}

// ================================================================= Logprobs JSON formatters

json::Object chat_token_logprob_json(const TokenLogprob& lp) {
    json::Array tops;
    tops.reserve(lp.top_logprobs.size());
    for (const auto& t : lp.top_logprobs) {
        tops.emplace_back(json::Object{{"token", t.token},
                                       {"logprob", static_cast<double>(t.logprob)},
                                       {"bytes", utf8_bytes_json(t.token)}});
    }
    return json::Object{{"token", lp.token},
                        {"logprob", static_cast<double>(lp.logprob)},
                        {"bytes", utf8_bytes_json(lp.token)},
                        {"top_logprobs", std::move(tops)}};
}

json::Object chat_logprobs_json(const std::vector<TokenLogprob>& logprobs) {
    json::Array content;
    content.reserve(logprobs.size());
    for (const auto& lp : logprobs) {
        content.emplace_back(chat_token_logprob_json(lp));
    }
    return json::Object{{"content", std::move(content)}};
}

json::Object completion_logprobs_json(const std::vector<TokenLogprob>& logprobs, std::size_t base_offset) {
    json::Array tokens;
    json::Array token_logprobs;
    json::Array top_logprobs;
    json::Array text_offset;
    tokens.reserve(logprobs.size());
    token_logprobs.reserve(logprobs.size());
    top_logprobs.reserve(logprobs.size());
    text_offset.reserve(logprobs.size());
    std::size_t offset = base_offset;
    for (const auto& lp : logprobs) {
        tokens.emplace_back(lp.token);
        token_logprobs.emplace_back(static_cast<double>(lp.logprob));
        text_offset.emplace_back(static_cast<std::uint64_t>(offset));
        offset += lp.token.size();
        json::Object top_map;
        if (!lp.top_logprobs.empty()) {
            for (const auto& t : lp.top_logprobs) {
                top_map.set(t.token, static_cast<double>(t.logprob));
            }
        } else {
            top_map.set(lp.token, static_cast<double>(lp.logprob));
        }
        top_logprobs.emplace_back(std::move(top_map));
    }
    return json::Object{{"tokens", std::move(tokens)},
                        {"token_logprobs", std::move(token_logprobs)},
                        {"top_logprobs", std::move(top_logprobs)},
                        {"text_offset", std::move(text_offset)}};
}

}  // namespace sonder::inference::server::detail
