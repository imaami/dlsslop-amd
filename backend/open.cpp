// SPDX-License-Identifier: MIT
#include "open.h"
#include "files.h"
#include "hip_engine.h"
#include "paths.h"
#include "vulkan_engine.h"

#include <cstdio>

namespace dlsslop {
VulkanPaths vulkan_paths(const Options& o)
{
    std::string cache = xdg_path("XDG_CACHE_HOME", ".cache", "dlsslop-amd/vulkan-pipelines.cache");
    if (!cache.empty() && !make_directories(parent_path(cache))) cache.clear();
    return {o.vulkan_model, vulkan_shaders(), cache};
}

namespace {
// The Vulkan network, prepared, or why not.
Result<std::unique_ptr<Backend>> open_vulkan(const Options& o, unsigned tier)
{
    if (!is_regular_file(o.vulkan_model))
        return fail("no model at " + o.vulkan_model + " (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)");
    std::unique_ptr<VulkanNetwork> network;
    DLSSLOP_TRY(guarded([&] { network = std::make_unique<VulkanNetwork>(vulkan_paths(o), o.device); }));
    std::fprintf(stderr, "Vulkan network on %s\n", network->device_name().c_str());
    auto engine = std::make_unique<VulkanEngine>(std::move(network), tier);
    DLSSLOP_TRY(engine->prepare());
    return engine;
}
} // namespace

Result<std::unique_ptr<Backend>> open_backend(Options& o, unsigned tier)
{
    if (const char* hip = hip_only(o); hip && o.backend == "auto") {
        std::fprintf(stderr, "%s needs the HIP network; using HIP\n", hip);
    } else if (!o.test_identity && o.backend != "hip") {
        auto vulkan = open_vulkan(o, tier);
        if (vulkan || o.backend == "vulkan") return vulkan;
        std::fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", vulkan.error().what.c_str());
    }
    if (o.test_identity) return std::make_unique<HipEngine>(o, tier, hip::Api{});
    const auto api = DLSSLOP_TRY(hip::load());
    o.device = DLSSLOP_TRY(select_device(api, o.device));
    return std::make_unique<HipEngine>(o, tier, api);
}
} // namespace dlsslop
