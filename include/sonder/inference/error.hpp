// Sonder Inference: error and result types.
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace sonder::inference {

// Stable error codes. Numeric values are mirrored by sonder_status in the C ABI
// header (sonder_inference.h); never renumber an existing entry.
enum class ErrorCode : int {
    ok = 0,
    invalid_argument = 1,
    invalid_state = 2,
    not_found = 3,
    unavailable = 4,
    cancelled = 5,
    timeout = 6,
    backend_error = 7,
    protocol_error = 8,
    io_error = 9,
    unsupported = 10,
    internal = 11,
};

const char* to_string(ErrorCode code) noexcept;

class Status {
public:
    Status() = default;
    Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    static Status success() { return {}; }

    [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::ok; }
    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    [[nodiscard]] std::string to_string() const;

private:
    ErrorCode code_ = ErrorCode::ok;
    std::string message_;
};

// Minimal expected-like carrier (std::expected is C++23).
template <class T>
class Result {
public:
    Result(T value) : storage_(std::in_place_index<0>, std::move(value)) {}  // NOLINT(google-explicit-constructor)
    Result(Status status) : storage_(std::in_place_index<1>, std::move(status)) {}  // NOLINT(google-explicit-constructor)

    [[nodiscard]] bool ok() const noexcept { return storage_.index() == 0; }
    explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] T& value() & { return std::get<0>(storage_); }
    [[nodiscard]] const T& value() const& { return std::get<0>(storage_); }
    [[nodiscard]] T&& value() && { return std::get<0>(std::move(storage_)); }

    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }

    [[nodiscard]] Status status() const {
        if (ok()) {
            return Status::success();
        }
        return std::get<1>(storage_);
    }

private:
    std::variant<T, Status> storage_;
};

}  // namespace sonder::inference
