#include "json_schema_gbnf.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::inference::server::detail {

namespace {

constexpr std::size_t kMaxSchemaDepth = 32;
constexpr std::size_t kMaxGrammarBytes = 256u * 1024u;
constexpr std::size_t kMaxRules = 1024;
constexpr std::size_t kMaxUnrolledBounds = 32;
constexpr std::size_t kMaxSchemaNodes = 2048;

bool is_supported_schema_keyword(std::string_view key) noexcept {
    // Functional keywords enforced by the converter:
    if (key == "$ref" || key == "$defs" || key == "definitions" || key == "type" || key == "const" || key == "enum" ||
        key == "anyOf" || key == "properties" || key == "required" || key == "additionalProperties" || key == "items" ||
        key == "minLength" || key == "maxLength" || key == "minItems" || key == "maxItems") {
        return true;
    }
    // Non-validating JSON Schema annotation/metadata keywords:
    if (key == "$schema" || key == "$id" || key == "$anchor" || key == "$comment" || key == "title" ||
        key == "description" || key == "default" || key == "examples" || key == "readOnly" || key == "writeOnly" ||
        key == "deprecated") {
        return true;
    }
    return false;
}

const char* const kCommonRules =
    "space ::= | \" \" | \"\\n\" [ \\t]{0,20}\n"
    "string ::= \"\\\"\" ([^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4}))* \"\\\"\" space\n"
    "char ::= [^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n"
    "integer ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) space\n"
    "number ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)? space\n"
    "boolean ::= (\"true\" | \"false\") space\n"
    "null ::= \"null\" space\n"
    "value ::= object | array | string | number | boolean | null\n"
    "object ::= \"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space\n"
    "array ::= \"[\" space ( value (\",\" space value)* )? \"]\" space\n";

// Escapes raw bytes as a GBNF double-quoted string literal (including quotes).
std::string gbnf_literal(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 4);
    out.push_back('"');
    for (const char ch : text) {
        const auto u = static_cast<unsigned char>(ch);
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (u < 0x20 || u == 0x7F) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\x";
                    out.push_back(kHex[(u >> 4) & 0x0F]);
                    out.push_back(kHex[u & 0x0F]);
                } else {
                    out.push_back(ch);
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

std::string sanitize_rule_name(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (const char ch : raw) {
        const auto u = static_cast<unsigned char>(ch);
        if (std::isalnum(u) != 0) {
            out.push_back(static_cast<char>(std::tolower(u)));
        } else if (ch == '-' || ch == '_') {
            out.push_back('-');
        } else {
            out.push_back('-');
        }
        if (out.size() >= 40) {
            break;
        }
    }
    while (!out.empty() && out.front() == '-') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    if (out.empty() || !std::isalpha(static_cast<unsigned char>(out.front()))) {
        out = "r-" + out;
    }
    return out;
}

class SchemaConverter {
public:
    explicit SchemaConverter(const json::Value& root_schema) : root_schema_(root_schema) {}

    Result<std::string> convert() {
        if (root_schema_.is_bool()) {
            if (!root_schema_.as_bool()) {
                return Status(ErrorCode::invalid_argument, "JSON schema 'false' allows no valid values");
            }
            return std::string("root ::= value\n") + kCommonRules;
        }
        if (!root_schema_.is_object()) {
            return Status(ErrorCode::invalid_argument, "JSON schema must be an object");
        }
        auto root_expr = build_node(root_schema_, "root", 0);
        if (!root_expr.ok()) {
            return root_expr.status();
        }
        std::string out;
        out.reserve(4096);
        out += "root ::= " + root_expr.value() + "\n";
        for (const auto& [name, expr] : rules_) {
            out += name;
            out += " ::= ";
            out += expr;
            out += "\n";
            if (out.size() > kMaxGrammarBytes) {
                return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
            }
        }
        out += kCommonRules;
        if (out.size() > kMaxGrammarBytes) {
            return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
        }
        return out;
    }

private:
    std::string alloc_rule_name(std::string_view hint) {
        const std::string_view short_hint = hint.size() > 24 ? hint.substr(0, 24) : hint;
        std::string base = sanitize_rule_name(short_hint);
        if (base == "root" || base == "space" || base == "string" || base == "char" || base == "integer" ||
            base == "number" || base == "boolean" || base == "null" || base == "value" || base == "object" ||
            base == "array") {
            base = "s-" + base;
        }
        std::string candidate = base;
        while (used_names_.count(candidate) != 0) {
            candidate = base + "-" + std::to_string(++counter_);
        }
        used_names_.insert(candidate);
        return candidate;
    }

    std::string next_hint(std::string_view tag) {
        return std::string(tag) + "-" + std::to_string(++hint_counter_);
    }

    Result<std::string> add_rule(std::string_view hint, std::string expr) {
        if (rules_.size() + reserved_rules_ >= kMaxRules) {
            return Status(ErrorCode::invalid_argument, "JSON schema produces too many grammar rules");
        }
        std::string name = alloc_rule_name(hint);
        grammar_bytes_ += name.size() + expr.size() + 6;
        if (grammar_bytes_ > kMaxGrammarBytes) {
            return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
        }
        rules_.emplace_back(name, std::move(expr));
        return name;
    }

    static std::string unescape_json_pointer_token(std::string_view token) {
        std::string out;
        out.reserve(token.size());
        for (std::size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '~' && i + 1 < token.size()) {
                if (token[i + 1] == '1') {
                    out.push_back('/');
                    ++i;
                    continue;
                }
                if (token[i + 1] == '0') {
                    out.push_back('~');
                    ++i;
                    continue;
                }
            }
            out.push_back(token[i]);
        }
        return out;
    }

    Result<const json::Value*> resolve_ref(std::string_view ref) const {
        constexpr std::string_view kDefsPrefix = "#/$defs/";
        constexpr std::string_view kLegacyPrefix = "#/definitions/";
        const json::Value* container = nullptr;
        std::string_view raw_key;
        if (ref.rfind(kDefsPrefix, 0) == 0) {
            container = root_schema_.find("$defs");
            raw_key = ref.substr(kDefsPrefix.size());
        } else if (ref.rfind(kLegacyPrefix, 0) == 0) {
            container = root_schema_.find("definitions");
            raw_key = ref.substr(kLegacyPrefix.size());
        } else {
            return Status(ErrorCode::invalid_argument,
                          "unsupported $ref '" + std::string(ref) + "' (only local #/$defs/... or #/definitions/... supported)");
        }
        if (container == nullptr || !container->is_object()) {
            return Status(ErrorCode::invalid_argument, "unresolved $ref '" + std::string(ref) + "'");
        }
        const std::string key = unescape_json_pointer_token(raw_key);
        const json::Value* target = container->find(key);
        if (target == nullptr) {
            return Status(ErrorCode::invalid_argument, "unresolved $ref '" + std::string(ref) + "'");
        }
        return target;
    }

    Result<std::string> build_node(const json::Value& schema, std::string_view hint, std::size_t depth) {
        if (++node_visits_ > kMaxSchemaNodes) {
            return Status(ErrorCode::invalid_argument, "JSON schema traversal exceeds maximum complexity budget");
        }
        if (depth > kMaxSchemaDepth) {
            return Status(ErrorCode::invalid_argument, "JSON schema nesting exceeds maximum depth");
        }
        if (schema.is_bool()) {
            if (!schema.as_bool()) {
                return Status(ErrorCode::invalid_argument, "JSON schema 'false' allows no valid values");
            }
            return std::string("value");
        }
        if (!schema.is_object()) {
            return Status(ErrorCode::invalid_argument, "JSON schema node must be an object or boolean");
        }

        // Fail closed on any validation keyword this GBNF converter cannot faithfully enforce.
        for (const auto& [k, v] : schema.as_object()) {
            (void)v;
            if (!is_supported_schema_keyword(k)) {
                return Status(ErrorCode::invalid_argument,
                              "unsupported JSON schema validation keyword '" + k + "'");
            }
        }

        if (const json::Value* ref = schema.find("$ref"); ref != nullptr) {
            if (!ref->is_string()) {
                return Status(ErrorCode::invalid_argument, "$ref must be a string");
            }
            const std::string& ref_str = ref->as_string();
            if (active_refs_.count(ref_str) != 0) {
                return Status(ErrorCode::invalid_argument,
                              "direct or left-recursive $ref cycle is not supported: '" + ref_str + "'");
            }
            if (auto it = ref_rules_.find(ref_str); it != ref_rules_.end()) {
                return it->second;
            }
            auto target = resolve_ref(ref_str);
            if (!target.ok()) {
                return target.status();
            }
            if (rules_.size() + reserved_rules_ >= kMaxRules) {
                return Status(ErrorCode::invalid_argument, "JSON schema produces too many grammar rules");
            }
            // Reserve rule slot up front to prevent recursion overshoot past kMaxRules.
            const std::string rule_name = alloc_rule_name(ref_str);
            ref_rules_[ref_str] = rule_name;
            active_refs_.insert(ref_str);
            ++reserved_rules_;
            auto body = build_node(*target.value(), rule_name, depth + 1);
            --reserved_rules_;
            active_refs_.erase(ref_str);
            if (!body.ok()) {
                return body.status();
            }
            grammar_bytes_ += rule_name.size() + body.value().size() + 6;
            if (grammar_bytes_ > kMaxGrammarBytes) {
                return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
            }
            rules_.emplace_back(rule_name, std::move(body).value());
            return rule_name;
        }

        if (const json::Value* c = schema.find("const"); c != nullptr) {
            return gbnf_literal(c->dump()) + " space";
        }

        if (const json::Value* en = schema.find("enum"); en != nullptr) {
            if (!en->is_array() || en->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "enum must be a non-empty array");
            }
            std::string expr = "(";
            bool first = true;
            for (const auto& item : en->as_array()) {
                if (!first) expr += " | ";
                first = false;
                expr += gbnf_literal(item.dump());
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ") space";
            return expr;
        }

        if (const json::Value* alts = schema.find("anyOf"); alts != nullptr) {
            if (!alts->is_array() || alts->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "anyOf must be a non-empty array");
            }
            std::string expr = "(";
            bool first = true;
            for (const auto& branch : alts->as_array()) {
                auto built = build_node(branch, next_hint("alt"), depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                if (!first) expr += " | ";
                first = false;
                expr += "(" + built.value() + ")";
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ")";
            return expr;
        }

        const json::Value* type_v = schema.find("type");
        if (type_v != nullptr && type_v->is_array()) {
            if (type_v->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "type array must not be empty");
            }
            std::string expr = "(";
            bool first = true;
            std::set<std::string> seen_types;
            for (const auto& t : type_v->as_array()) {
                if (!t.is_string()) {
                    return Status(ErrorCode::invalid_argument, "type array entries must be strings");
                }
                if (!seen_types.insert(t.as_string()).second) {
                    continue;
                }
                auto built = build_for_type(schema, t.as_string(), hint, depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                if (!first) expr += " | ";
                first = false;
                expr += "(" + built.value() + ")";
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ")";
            return expr;
        }

        std::string type_name;
        if (type_v != nullptr) {
            if (!type_v->is_string()) {
                return Status(ErrorCode::invalid_argument, "type must be a string or array of strings");
            }
            type_name = type_v->as_string();
        } else if (schema.find("properties") != nullptr || schema.find("additionalProperties") != nullptr ||
                   schema.find("required") != nullptr) {
            type_name = "object";
        } else if (schema.find("items") != nullptr || schema.find("minItems") != nullptr ||
                   schema.find("maxItems") != nullptr) {
            type_name = "array";
        } else if (schema.find("minLength") != nullptr || schema.find("maxLength") != nullptr) {
            type_name = "string";
        } else {
            return std::string("value");
        }
        return build_for_type(schema, type_name, hint, depth);
    }

    Result<std::string> build_for_type(const json::Value& schema, std::string_view type_name, std::string_view hint,
                                       std::size_t depth) {
        if (type_name == "string") {
            return build_string(schema);
        }
        if (type_name == "integer") {
            return std::string("integer");
        }
        if (type_name == "number") {
            return std::string("number");
        }
        if (type_name == "boolean") {
            return std::string("boolean");
        }
        if (type_name == "null") {
            return std::string("null");
        }
        if (type_name == "array") {
            return build_array(schema, hint, depth);
        }
        if (type_name == "object") {
            return build_object(schema, hint, depth);
        }
        return Status(ErrorCode::invalid_argument, "unsupported JSON schema type '" + std::string(type_name) + "'");
    }

    Result<std::string> build_string(const json::Value& schema) {
        std::size_t min_len = 0;
        std::optional<std::size_t> max_len;
        if (const json::Value* mn = schema.find("minLength"); mn != nullptr) {
            if (!mn->is_integer() || mn->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "minLength must be a non-negative integer");
            }
            min_len = static_cast<std::size_t>(mn->as_uint());
        }
        if (const json::Value* mx = schema.find("maxLength"); mx != nullptr) {
            if (!mx->is_integer() || mx->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "maxLength must be a non-negative integer");
            }
            max_len = static_cast<std::size_t>(mx->as_uint());
        }
        if (max_len && *max_len < min_len) {
            return Status(ErrorCode::invalid_argument, "maxLength must be >= minLength");
        }
        if (min_len == 0 && !max_len) {
            return std::string("string");
        }
        if (min_len <= kMaxUnrolledBounds && (!max_len || *max_len <= kMaxUnrolledBounds)) {
            std::string rep = "char{" + std::to_string(min_len) + ",";
            if (max_len) {
                rep += std::to_string(*max_len);
            }
            rep += "}";
            return "\"\\\"\" " + rep + " \"\\\"\" space";
        }
        return Status(ErrorCode::invalid_argument,
                      "minLength/maxLength above " + std::to_string(kMaxUnrolledBounds) +
                          " is not supported by GBNF conversion");
    }

    Result<std::string> build_array(const json::Value& schema, std::string_view hint, std::size_t depth) {
        (void)hint;
        std::string item_rule = "value";
        if (const json::Value* items = schema.find("items"); items != nullptr) {
            const std::string item_hint = next_hint("item");
            auto built = build_node(*items, item_hint, depth + 1);
            if (!built.ok()) {
                return built.status();
            }
            auto named = add_rule(item_hint, std::move(built).value());
            if (!named.ok()) {
                return named.status();
            }
            item_rule = std::move(named).value();
        }
        std::size_t min_items = 0;
        std::optional<std::size_t> max_items;
        if (const json::Value* mn = schema.find("minItems"); mn != nullptr) {
            if (!mn->is_integer() || mn->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "minItems must be a non-negative integer");
            }
            min_items = static_cast<std::size_t>(mn->as_uint());
        }
        if (const json::Value* mx = schema.find("maxItems"); mx != nullptr) {
            if (!mx->is_integer() || mx->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "maxItems must be a non-negative integer");
            }
            max_items = static_cast<std::size_t>(mx->as_uint());
        }
        if (max_items && *max_items < min_items) {
            return Status(ErrorCode::invalid_argument, "maxItems must be >= minItems");
        }
        if (min_items == 0 && !max_items) {
            return "\"[\" space ( " + item_rule + " (\",\" space " + item_rule + ")* )? \"]\" space";
        }
        if (max_items && *max_items == 0) {
            return std::string("\"[\" space \"]\" space");
        }
        if (min_items <= kMaxUnrolledBounds && (!max_items || *max_items <= kMaxUnrolledBounds)) {
            std::string body;
            if (min_items > 0) {
                body += item_rule;
                for (std::size_t i = 1; i < min_items; ++i) {
                    body += " \",\" space " + item_rule;
                }
                if (!max_items) {
                    body += " (\",\" space " + item_rule + ")*";
                } else {
                    for (std::size_t i = min_items; i < *max_items; ++i) {
                        body += " (\",\" space " + item_rule + ")?";
                    }
                }
            } else {
                // min_items == 0, max_items >= 1
                body = "( " + item_rule;
                for (std::size_t i = 1; i < *max_items; ++i) {
                    body += " (\",\" space " + item_rule + ")?";
                }
                body += " )?";
            }
            return "\"[\" space " + body + " \"]\" space";
        }
        return Status(ErrorCode::invalid_argument,
                      "minItems/maxItems above " + std::to_string(kMaxUnrolledBounds) +
                          " is not supported by GBNF conversion");
    }

    Result<std::string> build_object(const json::Value& schema, std::string_view hint, std::size_t depth) {
        (void)hint;
        const json::Value* props_v = schema.find("properties");
        if (props_v != nullptr && !props_v->is_object()) {
            return Status(ErrorCode::invalid_argument, "properties must be an object");
        }
        const bool has_declared_props = (props_v != nullptr && !props_v->as_object().empty());
        std::set<std::string> required_set;
        if (const json::Value* req_v = schema.find("required"); req_v != nullptr) {
            if (!req_v->is_array()) {
                return Status(ErrorCode::invalid_argument, "required must be an array of strings");
            }
            for (const auto& item : req_v->as_array()) {
                if (!item.is_string()) {
                    return Status(ErrorCode::invalid_argument, "required entries must be strings");
                }
                required_set.insert(item.as_string());
            }
        }

        // additionalProperties: default true only when no properties are declared.
        bool allow_additional = !has_declared_props;
        std::string additional_val_rule = "value";
        if (const json::Value* ap = schema.find("additionalProperties"); ap != nullptr) {
            if (ap->is_bool()) {
                allow_additional = ap->as_bool();
            } else if (ap->is_object()) {
                allow_additional = true;
                const std::string extra_hint = next_hint("extra");
                auto built = build_node(*ap, extra_hint, depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                auto named = add_rule(extra_hint, std::move(built).value());
                if (!named.ok()) {
                    return named.status();
                }
                additional_val_rule = std::move(named).value();
            } else {
                return Status(ErrorCode::invalid_argument, "additionalProperties must be a boolean or schema object");
            }
        }

        // GBNF cannot express "any string EXCEPT declared property names" cleanly without
        // exponential negation; fail closed instead of letting generic additionalProperties
        // bypass a declared property's schema.
        if (has_declared_props && allow_additional) {
            return Status(ErrorCode::invalid_argument,
                          "combining declared properties with permissive additionalProperties is not supported by "
                          "GBNF conversion; set additionalProperties to false");
        }

        if (!has_declared_props && required_set.empty()) {
            if (!allow_additional) {
                return std::string("\"{\" space \"}\" space");
            }
            if (additional_val_rule == "value") {
                return std::string("object");
            }
            const std::string kv = "string \":\" space " + additional_val_rule;
            return "\"{\" space ( " + kv + " (\",\" space " + kv + ")* )? \"}\" space";
        }

        std::vector<std::string> req_rules;
        std::vector<std::string> opt_rules;
        std::set<std::string> seen_props;
        if (has_declared_props) {
            for (const auto& [prop_name, prop_schema] : props_v->as_object()) {
                seen_props.insert(prop_name);
                const std::string prop_hint = next_hint("prop");
                auto val_expr = build_node(prop_schema, prop_hint, depth + 1);
                if (!val_expr.ok()) {
                    return val_expr.status();
                }
                const std::string key_json = json::Value(prop_name).dump();
                const std::string kv_expr = gbnf_literal(key_json) + " space \":\" space (" + val_expr.value() + ")";
                auto kv_rule = add_rule(next_hint("kv"), kv_expr);
                if (!kv_rule.ok()) {
                    return kv_rule.status();
                }
                if (required_set.count(prop_name) != 0) {
                    req_rules.push_back(std::move(kv_rule).value());
                } else {
                    opt_rules.push_back(std::move(kv_rule).value());
                }
            }
        }
        // Honor any `required` keys not explicitly listed under `properties`.
        for (const std::string& req_name : required_set) {
            if (seen_props.count(req_name) != 0) continue;
            if (!allow_additional) {
                return Status(ErrorCode::invalid_argument,
                              "required property '" + req_name + "' is absent from properties while additionalProperties is false");
            }
            const std::string key_json = json::Value(req_name).dump();
            const std::string kv_expr = gbnf_literal(key_json) + " space \":\" space (" + additional_val_rule + ")";
            auto kv_rule = add_rule(next_hint("kv"), kv_expr);
            if (!kv_rule.ok()) {
                return kv_rule.status();
            }
            req_rules.push_back(std::move(kv_rule).value());
        }

        std::string extra_kv_rule;
        if (allow_additional) {
            auto kv = add_rule(next_hint("extra-kv"), "string \":\" space " + additional_val_rule);
            if (!kv.ok()) {
                return kv.status();
            }
            extra_kv_rule = std::move(kv).value();
        }

        std::string obj = "\"{\" space ";
        if (!req_rules.empty()) {
            for (std::size_t i = 0; i < req_rules.size(); ++i) {
                if (i > 0) obj += " \",\" space ";
                obj += req_rules[i];
            }
            for (const auto& opt : opt_rules) {
                obj += " (\",\" space " + opt + ")?";
            }
            if (!extra_kv_rule.empty()) {
                obj += " (\",\" space " + extra_kv_rule + ")*";
            }
        } else {
            // All properties are optional: build exact comma-safe alternatives so any
            // subset (including empty {}) is valid JSON with no leading/trailing commas.
            std::vector<std::string> branches;
            std::size_t total_branch_bytes = 0;
            for (std::size_t i = 0; i < opt_rules.size(); ++i) {
                std::string branch = opt_rules[i];
                for (std::size_t j = i + 1; j < opt_rules.size(); ++j) {
                    branch += " (\",\" space " + opt_rules[j] + ")?";
                }
                if (!extra_kv_rule.empty()) {
                    branch += " (\",\" space " + extra_kv_rule + ")*";
                }
                total_branch_bytes += branch.size() + 5;
                if (grammar_bytes_ + total_branch_bytes > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
                branches.push_back(std::move(branch));
            }
            if (!extra_kv_rule.empty()) {
                branches.push_back(extra_kv_rule + " (\",\" space " + extra_kv_rule + ")*");
            }
            if (!branches.empty()) {
                obj += "( ";
                for (std::size_t i = 0; i < branches.size(); ++i) {
                    if (i > 0) obj += " | ";
                    obj += "(" + branches[i] + ")";
                }
                obj += " )?";
            }
        }
        obj += " \"}\" space";
        auto named = add_rule(next_hint("obj"), std::move(obj));
        if (!named.ok()) {
            return named.status();
        }
        return named.value();
    }

    const json::Value& root_schema_;
    std::vector<std::pair<std::string, std::string>> rules_;
    std::set<std::string> used_names_;
    std::map<std::string, std::string> ref_rules_;
    std::set<std::string> active_refs_;
    std::size_t reserved_rules_ = 0;
    std::size_t grammar_bytes_ = 0;
    std::size_t counter_ = 0;
    std::size_t hint_counter_ = 0;
    std::size_t node_visits_ = 0;
};

}  // namespace

std::string json_object_gbnf() {
    return std::string("root ::= object\n") + kCommonRules;
}

Result<std::string> json_schema_to_gbnf(const json::Value& schema) {
    SchemaConverter converter(schema);
    return converter.convert();
}

}  // namespace sonder::inference::server::detail
