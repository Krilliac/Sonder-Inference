#include "warmup_config.hpp"

#include <algorithm>
#include <string_view>

namespace sonder::inference::detail {

Status parse_warmup_config(const json::Value& value, BackendSetup& setup) {
    if (!value.is_object())
        return Status(ErrorCode::invalid_argument, "llamaserver config: 'warmup' must be an object");
    static constexpr std::string_view keys[] = {
        "messages_file", "chat_template_kwargs", "slots", "on_restart", "max_prefix_chars"};
    for (const auto& member : value.as_object()) {
        if (std::find(std::begin(keys), std::end(keys), member.first) == std::end(keys))
            return Status(ErrorCode::invalid_argument, "llamaserver config: unknown warmup field '" + member.first + "'");
    }
    if (const auto* v = value.find("messages_file")) {
        if (!v->is_string())
            return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.messages_file must be a string");
        setup.llamaserver_warmup_messages_file = v->as_string();
        if (setup.llamaserver_warmup_messages_file.find('\0') != std::string::npos)
            return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.messages_file must not contain NUL");
    }
    if (const auto* v = value.find("chat_template_kwargs")) {
        if (!v->is_object())
            return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.chat_template_kwargs must be an object");
        setup.llamaserver_warmup_chat_template_kwargs = v->as_object();
    }
    if (const auto* v = value.find("on_restart")) {
        if (!v->is_bool())
            return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.on_restart must be boolean");
        setup.llamaserver_warmup_on_restart = v->as_bool();
    }
    if (const auto* v = value.find("max_prefix_chars")) {
        if (!v->is_integer() || v->as_int(-1) < 1 || v->as_uint() > 16777216ull)
            return Status(ErrorCode::invalid_argument,
                          "llamaserver config: warmup.max_prefix_chars must be an integer in [1, 16777216]");
        setup.llamaserver_warmup_max_prefix_chars = static_cast<std::size_t>(v->as_uint());
    }
    if (const auto* v = value.find("slots")) {
        if (v->is_string()) {
            if (v->as_string() != "all")
                return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.slots must be 'all' or an array");
            setup.llamaserver_warmup_all_slots = true;
            setup.llamaserver_warmup_slots.clear();
        } else if (v->is_array()) {
            setup.llamaserver_warmup_all_slots = false;
            setup.llamaserver_warmup_slots.clear();
            for (const auto& item : v->as_array()) {
                if (!item.is_integer() || item.as_int(-1) < 0 || item.as_uint() > 1023ull)
                    return Status(ErrorCode::invalid_argument,
                                  "llamaserver config: warmup.slots entries must be integers in [0, 1023]");
                const auto id = static_cast<std::uint32_t>(item.as_uint());
                if (std::find(setup.llamaserver_warmup_slots.begin(), setup.llamaserver_warmup_slots.end(), id) !=
                    setup.llamaserver_warmup_slots.end())
                    return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.slots entries must be unique");
                setup.llamaserver_warmup_slots.push_back(id);
            }
        } else {
            return Status(ErrorCode::invalid_argument, "llamaserver config: warmup.slots must be 'all' or an array");
        }
    }
    return {};
}

}  // namespace sonder::inference::detail
