#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace sonder::inference::sampling {

/// Result of feeding one detokenized piece into a StopSequenceMatcher.
struct StopCheck {
    bool stopped = false;          ///< A stop sequence completed.
    std::size_t stop_index = 0;    ///< Index into the stop list (valid if stopped).
    std::string emit;              ///< Text that is now safe to stream to the client.
};

/// Incremental, text-level stop-sequence detector (like llama.cpp's
/// reverse prompts / server `stop`). Text that could still be the start of a
/// stop sequence is held back so a client never sees a partial stop string.
/// On a match, `emit` contains everything before the stop sequence and the
/// stop sequence itself is not emitted. Matching is byte-wise on UTF-8.
/// When several sequences complete at once, the one that starts earliest
/// wins (ties: the longest).
class StopSequenceMatcher {
public:
    explicit StopSequenceMatcher(std::vector<std::string> stops);

    StopCheck feed(std::string_view piece);

    /// Release any held-back text (call when generation ends without a stop).
    std::string flush();

    void reset();

    [[nodiscard]] bool stopped() const noexcept { return stopped_; }
    [[nodiscard]] const std::vector<std::string>& stops() const noexcept { return stops_; }

private:
    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_ = false;
};

}  // namespace sonder::inference::sampling
