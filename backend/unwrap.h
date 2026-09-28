// Until the HIP engine, its self-test and the serving loop return Results: a
// Result's value, or its error thrown as the exception they expect.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"

#include <stdexcept>

namespace dlsslop {
template <class T>
T unwrap(Result<T> result)
{
    if (!result) {
        if (result.error().rejected) throw std::range_error(result.error().what);
        throw std::runtime_error(result.error().what);
    }
    if constexpr (!std::is_void_v<T>) return std::move(*result);
}
} // namespace dlsslop
