// SPDX-License-Identifier: MIT
#include "open.h"
#include "files.h"
#include "paths.h"

namespace dlsslop {
VulkanPaths vulkan_paths(const Options& o)
{
    std::string cache = xdg_path("XDG_CACHE_HOME", ".cache", "dlsslop-amd/vulkan-pipelines.cache");
    if (!cache.empty() && !make_directories(parent_path(cache))) cache.clear();
    return {o.vulkan_model, vulkan_shaders(), cache};
}

Result<VulkanEngine> open_vulkan(const Options& o, unsigned tier)
{
    if (!is_regular_file(o.vulkan_model))
        return fail("no model at " + o.vulkan_model + " (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)");
    auto network = DLSSLOP_TRY(VulkanNetwork::create(vulkan_paths(o), o.device));
    std::fprintf(stderr, "Vulkan network on %s\n", network.device_name().c_str());
    VulkanEngine engine(std::move(network), tier);
    DLSSLOP_TRY(engine.prepare());
    return engine;
}
} // namespace dlsslop
