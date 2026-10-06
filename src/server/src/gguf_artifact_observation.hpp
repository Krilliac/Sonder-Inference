// Internal, opt-in inspected-content evidence. Not a loaded-backend identity.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <optional>
#include <string>
#include <string_view>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/launch_profile.hpp"

namespace sonder::inference::server::detail {

struct GgufObservationOptions {
    // One byte is reserved within this budget for boundary detection:
    // admitted byte_length is strictly less than max_read_bytes.
    std::uint64_t max_read_bytes = 0;
    // Inclusive encoded header/table budget, checked before string resize.
    std::uint64_t max_header_bytes = 0;
    std::chrono::steady_clock::time_point deadline{};
    std::size_t read_chunk_bytes = 65536;
};

struct GgufArtifactObservation {
    std::string sha256;
    std::uint64_t byte_length = 0;
    std::string architecture;
    std::optional<RoutedExpertCounts> routed_experts;
};

// The caller supplies a trusted full-artifact digest (exact lowercase64hex)
// and an already-open binary stream positioned at its beginning. Reads the
// same streambuf sequentially, hashes each consumed byte, and admits only at
// clean EOF with an equal digest. Does not seek, reopen, close, or change the
// stream's exception mask; EOF flags need not be set by streambuf reads.
// Options and expectation are snapshotted before source code runs; replacing
// the captured streambuf causes failure. The caller keeps that buffer alive.
// Cancellation/deadline checks are cooperative: blocked source I/O cannot
// be interrupted here. Budgets bound logical reads, not OS prefetch or heap.
// Native header interpretation is retained; tensor payload validity, file
// immutability, provenance/authenticity and loaded-content binding are NOT
// certified. No partial observation or input-derived diagnostic on failure.
Result<GgufArtifactObservation> observe_gguf_artifact(
    std::istream& input, std::string_view expected_sha256,
    const GgufObservationOptions& options, CancellationToken cancel = {});

}  // namespace sonder::inference::server::detail
