// Internal: CLI flag spec, usage text, synchronized log wrapper, and
// ServerOptions assembly (including --profile / G0050 / B1 / G0094 security
// checks) for `sonder-infer serve`.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/server.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::server {

// Multi-profile overload of profile_catalog() (see sonder/inference/server.hpp
// for the single-profile entry point): wires every profile in `active_profiles`
// into `options` (models, profile_bindings, catalog "served": true) and runs
// both per-profile and aggregate co-residency VRAM fit checks.
json::Array profile_catalog(const std::vector<LaunchProfile>& profiles,
                            const std::vector<const LaunchProfile*>& active_profiles, ServerOptions& options,
                            std::optional<std::uint64_t> budget_flag, bool allow_overcommit,
                            std::vector<std::string>& notes, std::string& refusal);

}  // namespace sonder::inference::server

namespace sonder::inference::server::detail {

extern const char* const kServeUsage;

// Serializes writes to the shared error stream (access log, diagnostics and
// the optional '--telemetry -' sink come from different threads).
class LockedLog {
public:
    explicit LockedLog(std::ostream& stream) : stream_(stream) {}
    void line(std::string_view text) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        stream_.write(text.data(), static_cast<std::streamsize>(text.size()));
        stream_.put('\n');
        stream_.flush();
    }
    // Waits for a write in progress, then turns every later line() into a
    // no-op: once this returns, nothing touches the stream again.
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }

private:
    std::mutex mutex_;
    std::ostream& stream_;
    bool closed_ = false;  // guarded by mutex_
};

// Closes the log when serve_main() returns, on every path. Declared before the
// Server so that it runs after the Server is destroyed.
struct LockedLogCloser {
    std::shared_ptr<LockedLog> log;
    ~LockedLogCloser() { log->close(); }
};

class LockedLogSink final : public TelemetrySink {
public:
    explicit LockedLogSink(std::shared_ptr<LockedLog> log) : log_(std::move(log)) {}
    void write(std::string_view json_line) override { log_->line(json_line); }

private:
    std::shared_ptr<LockedLog> log_;
};

struct ParsedServeArgs {
    std::vector<std::pair<std::string, std::string>> values;  // in order
    std::set<std::string> flags;
    bool help = false;

    [[nodiscard]] std::optional<std::string> get(const std::string& key) const {
        for (const auto& [k, v] : values) {
            if (k == key) {
                return v;
            }
        }
        return std::nullopt;
    }
    [[nodiscard]] std::vector<std::string> all(const std::string& key) const {
        std::vector<std::string> out;
        for (const auto& [k, v] : values) {
            if (k == key) {
                out.push_back(v);
            }
        }
        return out;
    }
};

bool parse_serve_args(const std::vector<std::string>& args, ParsedServeArgs& out, std::string& error);

struct ServeCliConfig {
    ServerOptions options;
    std::string ready_path;
    std::vector<std::string> profile_notes;
};

// Assembles and validates ServerOptions from parsed CLI arguments, enforcing
// all launch-profile, token-file, and backend constraints. Returns 0 on
// success, 1 on a runtime error (e.g. unwritable --telemetry path), or 2 on a
// usage error (after logging the error via `log`).
int build_serve_options(const ParsedServeArgs& a, const std::shared_ptr<LockedLog>& log, ServeCliConfig& out);

}  // namespace sonder::inference::server::detail
