// Deliberately skips C++ cleanup to exercise kernel parent-death containment.
#include <cstdlib>
#include <fstream>
#include <memory>

#include "worker/remote_backend.hpp"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    sonder::inference::worker::RemoteBackendOptions options;
    options.executable = argv[1];
    auto backend = std::make_shared<sonder::inference::worker::RemoteBackend>(options);
    if (!backend->probe().ok()) return 3;
    {
        std::ofstream output(argv[2], std::ios::binary);
        output << backend->worker_pid() << '\n';
        output.flush();
        if (!output.good()) return 4;
    }
    std::_Exit(0);
}
