// Errors without exceptions: every fallible function of ours returns a Result.
// SPDX-License-Identifier: MIT
#pragma once
#include <cerrno>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

namespace dlsslop {

// What went wrong, in words for the log and the channel's status. A rejected
// request was at fault, not the daemon: the frame fails and serving goes on.
// Anything else ends the daemon.
struct Error {
    std::string what;
    bool rejected = false;
};

template <class T = void>
using Result = std::expected<T, Error>;

[[nodiscard]] inline std::unexpected<Error> fail(std::string what) { return std::unexpected(Error{std::move(what)}); }
[[nodiscard]] inline std::unexpected<Error> reject(std::string what)
{
    return std::unexpected(Error{std::move(what), true});
}
// ACTION: strerror(errno).
[[nodiscard]] inline std::unexpected<Error> fail_errno(const char* action)
{
    return fail(std::string(action) + ": " + std::strerror(errno));
}

} // namespace dlsslop

// The value of a Result, or its error returned from the enclosing function.
#define DLSSLOP_TRY(...)                                                                                \
    __extension__({                                                                                     \
        auto&& dlsslop_try_ = (__VA_ARGS__);                                                            \
        if (!dlsslop_try_) return std::unexpected(std::move(dlsslop_try_).error());                     \
        *std::move(dlsslop_try_);                                                                       \
    })
