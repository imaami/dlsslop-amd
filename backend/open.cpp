// SPDX-License-Identifier: MIT
#include "open.hpp"
#include "paths.hpp"

#include <cstdio>

namespace dlsslop {
namespace {
// The Vulkan engine, prepared, or why not. The pipeline cache is a convenience.
Result<VulkanEngine> open_vulkan(const Options& o, unsigned tier)
{
    DLSSLOP_TRY(require_vulkan_model(o.vulkan_model));
    auto network = DLSSLOP_TRY(VulkanNetwork::create({o.vulkan_model, vulkan_shaders(), vulkan_cache()}, o.device));
    std::fprintf(stderr, "Vulkan network on %s\n", network.device_name().c_str());
    VulkanEngine engine(std::move(network), tier);
    DLSSLOP_TRY(engine.prepare());
    return engine;
}
} // namespace

Result<OpenedEngine> open_engine(Options& o, unsigned tier)
{
    OpenedEngine opened;
    if (o.test_identity) return opened;
    if (const char* hip = hip_only(o); hip && o.backend == "auto") {
        std::fprintf(stderr, "%s needs the HIP network; using HIP\n", hip);
    } else if (o.backend != "hip") {
        auto vulkan = open_vulkan(o, tier);
        if (vulkan) {
            opened.vulkan.emplace(*std::move(vulkan));
            return opened;
        }
        if (o.backend == "vulkan") return forward(std::move(vulkan).error());
        std::fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", vulkan.error().what.c_str());
    }
    opened.hip = DLSSLOP_TRY(open_hip(o));
    return opened;
}
} // namespace dlsslop
