#include "process_options.hpp"
#include <algorithm>
#include <map>
#include <string_view>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
extern char **environ;
#endif

namespace sonder::inference::llamaserver {
Status validate_process_environment(const std::vector<std::pair<std::string, std::string>> &env) {
    std::size_t bytes = 0;
    for (const auto &[key, value] : env) {
        if (key.empty() || key.find('=') != std::string::npos || key.find('\0') != std::string::npos ||
            value.find('\0') != std::string::npos)
            return {ErrorCode::invalid_argument, "invalid child environment entry"};
        bytes += key.size() + value.size() + 2;
        if (bytes > 32760 || env.size() > 128)
            return {ErrorCode::invalid_argument, "child environment overrides exceed bounds"};
    }
    return {};
}
#if defined(_WIN32)
namespace {
Result<std::wstring> utf16(const std::string &text) {
    if (text.empty()) return std::wstring{};
    if (text.size() > 32766) return Status(ErrorCode::invalid_argument, "launch option too long");
    const auto n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (n <= 0) return Status(ErrorCode::invalid_argument, "launch option is not UTF-8");
    std::wstring result(static_cast<std::size_t>(n), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), result.data(), n) != n)
        return Status(ErrorCode::invalid_argument, "launch option conversion failed");
    return result;
}
struct IgnoreCase {
    bool operator()(const std::wstring &a, const std::wstring &b) const {
        return CompareStringOrdinal(a.data(), static_cast<int>(a.size()), b.data(), static_cast<int>(b.size()), TRUE) == CSTR_LESS_THAN;
    }
};
} // namespace
ProcessOptions::~ProcessOptions() {
    if (startup.lpAttributeList) DeleteProcThreadAttributeList(startup.lpAttributeList);
    if (output_ != INVALID_HANDLE_VALUE) CloseHandle(output_);
    if (input_ != INVALID_HANDLE_VALUE) CloseHandle(input_);
}
Status ProcessOptions::prepare(const ProcessSpec &spec) {
    if (auto st = validate_process_environment(spec.environment); !st.ok()) return st;
    startup.StartupInfo.cb = sizeof(STARTUPINFOW);
    if (!spec.environment.empty()) {
        std::map<std::wstring, std::wstring, IgnoreCase> entries;
        auto *block = GetEnvironmentStringsW();
        if (!block) return {ErrorCode::io_error, "cannot read inherited environment"};
        for (const wchar_t *p = block; *p;) {
            const std::wstring entry(p);
            const auto split = entry.find(L'=', 1); // preserves Windows '=C:' drive entries
            if (split != std::wstring::npos) entries[entry.substr(0, split)] = entry.substr(split + 1);
            p += entry.size() + 1;
        }
        FreeEnvironmentStringsW(block);
        for (const auto &[key, value] : spec.environment) {
            auto k = utf16(key), v = utf16(value);
            if (!k.ok()) return k.status();
            if (!v.ok()) return v.status();
            entries[k.value()] = v.value();
        }
        for (const auto &[key, value] : entries) {
            environment.insert(environment.end(), key.begin(), key.end());
            environment.push_back(L'=');
            environment.insert(environment.end(), value.begin(), value.end());
            environment.push_back(L'\0');
        }
        environment.push_back(L'\0');
        if (environment.size() > 32767) return {ErrorCode::invalid_argument, "merged child environment too large"};
    }
    if (spec.output_file.empty()) return {};
    if (spec.output_file.find('\0') != std::string::npos) return {ErrorCode::invalid_argument, "capture path contains NUL"};
    auto path = utf16(spec.output_file);
    if (!path.ok()) return path.status();
    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    output_ = CreateFileW(path.value().c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                          &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    input_ = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                         &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (output_ == INVALID_HANDLE_VALUE || input_ == INVALID_HANDLE_VALUE)
        return {ErrorCode::io_error, "cannot open child capture handles"};
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    attributes_.resize((size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
    auto *list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes_.data());
    if (!InitializeProcThreadAttributeList(list, 1, 0, &size))
        return {ErrorCode::io_error, "cannot initialize child handle list"};
    startup.lpAttributeList = list;
    inherited_ = {output_, input_};
    if (!UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_.data(), sizeof(inherited_), nullptr, nullptr))
        return {ErrorCode::io_error, "cannot restrict child handle inheritance"};
    startup.StartupInfo.cb = sizeof(STARTUPINFOEXW);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdOutput = output_;
    startup.StartupInfo.hStdError = output_;
    startup.StartupInfo.hStdInput = input_;
    redirect = true;
    return {};
}
#else
ProcessOptions::~ProcessOptions() {
    if (redirect) posix_spawn_file_actions_destroy(&actions);
}
Status ProcessOptions::prepare(const ProcessSpec &spec) {
    if (auto st = validate_process_environment(spec.environment); !st.ok()) return st;
    if (!spec.environment.empty()) {
        std::map<std::string, std::string> entries;
        for (char **p = environ; p && *p; ++p) {
            const std::string entry(*p);
            const auto split = entry.find('=');
            if (split != std::string::npos) entries[entry.substr(0, split)] = entry.substr(split + 1);
        }
        for (const auto &[key, value] : spec.environment) entries[key] = value;
        for (const auto &[key, value] : entries) environment.push_back(key + '=' + value);
        for (auto &entry : environment) envp.push_back(entry.data());
        envp.push_back(nullptr);
    }
    if (spec.output_file.empty()) return {};
    if (spec.output_file.find('\0') != std::string::npos) return {ErrorCode::invalid_argument, "capture path contains NUL"};
    if (posix_spawn_file_actions_init(&actions) != 0) return {ErrorCode::io_error, "cannot initialize child capture"};
    redirect = true;
    if (posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, spec.output_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600) != 0 ||
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO) != 0 ||
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0) != 0)
        return {ErrorCode::io_error, "cannot configure child capture"};
    return {};
}
#endif
} // namespace sonder::inference::llamaserver
