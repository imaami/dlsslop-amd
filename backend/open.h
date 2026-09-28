// Which backend runs the network.
// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"
#include "vulkan_network.h"

#include <memory>

namespace dlsslop {
// Where the Vulkan network loads from; the pipeline cache is a convenience.
dlsslop::VulkanPaths vulkan_paths(const Options& o);
// The backend --backend selects, prepared. auto takes the Vulkan network when its
// model is there, a device can run it, it builds and no option needs HIP, and
// says why not before taking HIP.
std::unique_ptr<Backend> open_backend(Options& o, unsigned tier);
} // namespace dlsslop
