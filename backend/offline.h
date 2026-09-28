// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"

namespace dlsslop {
Result<void> run_offline(const Options& o, Backend& engine);
} // namespace dlsslop
