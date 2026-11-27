// A separate process proves the launch environment, rather than inspecting
// only the launcher's proposed environment block.
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
extern char** environ;
#endif

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    // Inspect before opening output, so that output cannot reuse the parent's
    // numeric descriptor and make a correctly closed descriptor look inherited.
    bool inherited = false;
    if (argc == 3) {
#if defined(_WIN32)
        const auto handle = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(std::stoull(argv[2])));
        DWORD flags = 0;
        inherited = GetHandleInformation(handle, &flags) != FALSE;
        // Numeric HANDLE values can be reused by startup. Only signalling the
        // parent's original event proves identity; it observes that separately.
        if (inherited) (void)SetEvent(handle);
#else
        inherited = fcntl(std::stoi(argv[2]), F_GETFD) != -1;
#endif
    }
    std::ofstream output(argv[1], std::ios::binary);
    if (!output) return 3;
    output << std::filesystem::current_path().generic_string() << '\n';
    if (argc == 3) output << "inherited=" << (inherited ? 1 : 0) << '\n';
#if defined(_WIN32)
    const auto entries = GetEnvironmentStringsW();
    if (!entries) return 4;
    for (const wchar_t* entry = entries; *entry; entry += std::wstring(entry).size() + 1) {
        const int size = WideCharToMultiByte(CP_UTF8, 0, entry, -1, nullptr, 0, nullptr, nullptr);
        if (size <= 0) {
            FreeEnvironmentStringsW(entries);
            return 5;
        }
        std::string text(static_cast<std::size_t>(size), '\0');
        WideCharToMultiByte(CP_UTF8, 0, entry, -1, text.data(), size, nullptr, nullptr);
        text.pop_back();
        output << text << '\n';
    }
    FreeEnvironmentStringsW(entries);
#else
    for (char** entry = environ; *entry; ++entry) output << *entry << '\n';
#endif
    output.flush();
    return output.good() ? 0 : 6;
}
