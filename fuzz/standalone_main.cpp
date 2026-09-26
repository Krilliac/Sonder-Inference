// Minimal driver for toolchains without libFuzzer (MSVC, GCC): runs each file
// (or every file in each directory) given on the command line through
// LLVMFuzzerTestOneInput once. Used as a deterministic corpus regression test.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace {
int run_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", p.string().c_str());
        return 1;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    int failures = 0;
    int ran = 0;
    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path arg(argv[i]);
        if (std::filesystem::is_directory(arg)) {
            for (const auto& e : std::filesystem::directory_iterator(arg)) {
                if (e.is_regular_file()) {
                    failures += run_file(e.path());
                    ++ran;
                }
            }
        } else {
            failures += run_file(arg);
            ++ran;
        }
    }
    std::printf("standalone fuzz driver: %d input(s), %d error(s)\n", ran, failures);
    return failures == 0 ? 0 : 1;
}
