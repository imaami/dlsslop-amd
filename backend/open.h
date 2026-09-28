// Which engine runs the network: the one choice made at run time.
// SPDX-License-Identifier: MIT
#pragma once
#include "hip_engine.h"
#include "identity_engine.h"
#include "vulkan_engine.h"

#include <cstdio>

namespace dlsslop {
// Where the Vulkan network loads from; the pipeline cache is a convenience.
VulkanPaths vulkan_paths(const Options& o);
// The Vulkan engine, prepared, or why not.
Result<VulkanEngine> open_vulkan(const Options& o, unsigned tier);

// RUN with the engine --backend selects. Everything RUN does is compiled for
// each engine; which one runs is decided here, once. auto takes the Vulkan
// network when its model is there, a device can run it, it builds and no
// option needs HIP, and says why not before taking HIP. The Vulkan engine comes
// prepared; RUN prepares the others.
template <class Run>
Result<void> with_engine(Options& o, unsigned tier, Run&& run)
{
    if (o.test_identity) {
        IdentityEngine engine(tier);
        return run(engine);
    }
    if (const char* hip = hip_only(o); hip && o.backend == "auto") {
        std::fprintf(stderr, "%s needs the HIP network; using HIP\n", hip);
    } else if (o.backend != "hip") {
        auto vulkan = open_vulkan(o, tier);
        if (vulkan) return run(*vulkan);
        if (o.backend == "vulkan") return std::unexpected(std::move(vulkan).error());
        std::fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", vulkan.error().what.c_str());
    }
    const auto hip = DLSSLOP_TRY(HipEngine::open(o, tier));
    return run(*hip);
}
} // namespace dlsslop
