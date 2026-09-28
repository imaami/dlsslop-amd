// SPDX-License-Identifier: MIT
#include "open.h"
#include "hip_engine.h"
#include "paths.h"
#include "vulkan_engine.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace dlsslop {
dlsslop::VulkanPaths vulkan_paths(const Options& o)
{
    std::string cache = xdg_path("XDG_CACHE_HOME", ".cache", "dlsslop-amd/vulkan-pipelines.cache");
    std::error_code error;
    if (!cache.empty() && !std::filesystem::create_directories(std::filesystem::path(cache).parent_path(), error) && error)
        cache.clear();
    return {o.vulkan_model, vulkan_shaders(), cache};
}

std::unique_ptr<Backend> open_backend(Options& o, unsigned tier)
{
    if (const char* hip = hip_only(o); hip && o.backend == "auto")
        std::fprintf(stderr, "%s needs the HIP network; using HIP\n", hip);
    else if (!o.test_identity && o.backend != "hip") {
        try {
            std::error_code error;
            if (!std::filesystem::is_regular_file(o.vulkan_model, error))
                throw std::runtime_error("no model at " + o.vulkan_model +
                                         " (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)");
            auto network = std::make_unique<dlsslop::VulkanNetwork>(vulkan_paths(o), o.device);
            std::fprintf(stderr, "Vulkan network on %s\n", network->device_name().c_str());
            auto engine = std::make_unique<VulkanEngine>(std::move(network), tier);
            engine->prepare();
            return engine;
        } catch (const std::exception& e) {
            if (o.backend == "vulkan") throw;
            std::fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", e.what());
        }
    }
    if (!o.test_identity) o.device = select_device(o.device);
    return std::make_unique<Engine>(o, tier);
}
} // namespace dlsslop
