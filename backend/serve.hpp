// Serving shared-memory requests.
// SPDX-License-Identifier: MIT
#pragma once
#include "options.h"
#include "result.hpp"

namespace dlsslop {
// Serves the channel until stopped, told to quit or idle. Fails when serving
// ends in a fault, which the channel then reports.
Result<void> run_worker(struct options& o);
} // namespace dlsslop
