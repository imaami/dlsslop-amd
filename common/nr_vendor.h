// The boundary to the vendored Vulkan network runtime (vulkan-nr), which throws:
// nr_vendor.cpp, built with exceptions, catches what it throws and returns
// Results. Its header, nr_runtime.hpp, throws nothing and is safe to include.
// SPDX-License-Identifier: MIT
#pragma once
#include "nr_runtime.hpp"
#include "result.h"

#include <memory>

namespace dlsslop::nr_vendor {
Result<std::unique_ptr<nr::Runtime>> build(const nr::HostDevice& host, const nr::RuntimeConfig& config,
                                           const nr::TemporalConfig& temporal);
Result<void> record(nr::Runtime& runtime, VkCommandBuffer cmd, const nr::ColourFrame& colour,
                    const nr::Controls& controls);
Result<void> record_temporal(nr::Runtime& runtime, VkCommandBuffer cmd, const nr::ColourFrame& colour,
                             const nr::Controls& controls, const nr::TemporalFrame& temporal);
} // namespace dlsslop::nr_vendor
