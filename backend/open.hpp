// Which engine runs the network: the one choice made at run time.
// SPDX-License-Identifier: MIT
#pragma once
#include "hip_engine.hpp"
#include "identity_engine.hpp"
#include "vulkan_engine.hpp"

#include <optional>

namespace dlsslop {
// The engine --backend selects, opened: the Vulkan engine, prepared, or the
// HIP runtime for a HipEngine, which cannot move and is made where it runs;
// neither for --test-identity. auto takes the Vulkan network when its model is
// there, a device can run it, it builds and no option needs HIP, and says why
// not before taking HIP.
struct OpenedEngine {
    std::optional<VulkanEngine> vulkan;
    std::optional<struct hip_api> hip;
};
Result<OpenedEngine> open_engine(struct options& o, unsigned tier);

// RUN with the engine --backend selects. Everything RUN does is compiled for
// each engine; which one runs is decided once, in open_engine(). RUN prepares
// the HIP and identity engines.
template <class Run>
Result<void> with_engine(struct options& o, unsigned tier, Run&& run)
{
    auto opened = DLSSLOP_TRY(open_engine(o, tier));
    if (opened.vulkan) return run(*opened.vulkan);
    if (opened.hip) {
        HipEngine engine(o, tier, *opened.hip);
        return run(engine);
    }
    IdentityEngine engine(tier);
    return run(engine);
}
} // namespace dlsslop
