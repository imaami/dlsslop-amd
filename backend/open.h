// Which backend runs the network.
// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"
#include "vulkan_network.h"

#include <memory>

namespace dlsslop {
// Where the Vulkan network loads from; the pipeline cache is a convenience.
VulkanPaths vulkan_paths(const Options& o);
// The backend --backend selects. auto takes the Vulkan network when its model
// is there, a device can run it, it builds and no option needs HIP, and says
// why not before taking HIP. The Vulkan network comes prepared; the HIP one is
// prepared by the caller.
Result<std::unique_ptr<Backend>> open_backend(Options& o, unsigned tier);
} // namespace dlsslop
