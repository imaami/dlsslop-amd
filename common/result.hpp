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

// An error on its way into the Result its function returns. The Result is
// made out of line, where the error's words become a string: each return of an
// error is one call, and the path to it is kept cold, out of the hot code.
// Return one at once: its words may be a temporary that ends with the return
// statement, so a lambda that returns one states its Result return type.
template <class Words>
struct Failure {
    Words what;
    bool rejected;
    template <class T>
    [[gnu::cold, gnu::noinline]] operator Result<T>() &&
    {
        return std::unexpected(Error{std::string(std::forward<Words>(what)), rejected});
    }
};

[[nodiscard]] inline Failure<const char*> fail(const char* what) { return {what, false}; }
[[nodiscard]] inline Failure<std::string&&> fail(std::string&& what) { return {std::move(what), false}; }
[[nodiscard]] inline Failure<const char*> reject(const char* what) { return {what, true}; }
// ACTION: strerror(errno).
[[nodiscard, gnu::cold, gnu::noinline]] inline Failure<std::string> fail_errno(const char* action)
{
    const int error = errno;
    return {std::string(action) + ": " + std::strerror(error), false};
}

// An error passed on to the enclosing function's Result, out of line like a Failure.
template <class E>
struct Forward {
    E&& error;
    template <class T>
    [[gnu::cold, gnu::noinline]] operator Result<T>() &&
    {
        return std::unexpected(Error(std::forward<E>(error)));
    }
};
template <class E> Forward<E> forward(E&& error) { return {std::forward<E>(error)}; }

} // namespace dlsslop

// The value of a Result, or its error returned from the enclosing function.
#define DLSSLOP_TRY(...)                                                                                \
    __extension__({                                                                                     \
        auto&& dlsslop_try_ = (__VA_ARGS__);                                                            \
        if (!dlsslop_try_) return ::dlsslop::forward(std::move(dlsslop_try_).error());                  \
        *std::move(dlsslop_try_);                                                                       \
    })
