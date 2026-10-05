// Errors without exceptions: every fallible function of the GUI returns a Result.
// SPDX-License-Identifier: MIT
#pragma once
#include <cerrno>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

namespace dlsslop {

// What went wrong, in words for the user.
struct Error {
    std::string what;
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
    template <class T>
    [[gnu::cold, gnu::noinline]] operator Result<T>() &&
    {
        return std::unexpected(Error{std::string(std::forward<Words>(what))});
    }
};

[[nodiscard]] inline Failure<const char*> fail(const char* what) { return {what}; }
// ACTION: strerror(errno).
[[nodiscard, gnu::cold, gnu::noinline]] inline Failure<std::string> fail_errno(const char* action)
{
    const int error = errno;
    return {std::string(action) + ": " + std::strerror(error)};
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
