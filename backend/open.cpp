// SPDX-License-Identifier: MIT
#include "open.hpp"
#include "paths.h"
#include "vulkan_runtime.h"

#include <cstdio>
#include <cstdlib>

namespace dlsslop {
namespace {
// The Vulkan engine, prepared, or why not. The pipeline cache is a convenience.
Result<VulkanEngine> open_vulkan(const struct options& o, unsigned tier)
{
    struct error e;
    if (const enum error_code code = paths_require_vulkan_model(o.vulkan_model, &e)) return forward_c(code, e);
    const struct paths_home home = paths_home();
    char* cache;
    size_t cache_length;
    if (const enum error_code code = paths_vulkan_cache(&cache, &cache_length, &home, &e)) return forward_c(code, e);
    // The network copies the paths, which borrow these strings.
    const struct vulkan_paths paths{.model = o.vulkan_model,
                                    .shaders = o.shaders,
                                    .cache = cache ? cache : "",
                                    .model_length = o.vulkan_model_length,
                                    .shaders_length = o.shaders_length,
                                    .cache_length = cache_length};
    struct vulkan_network* network;
    const enum error_code code = vulkan_network_create(&network, &paths, o.device, &e);
    std::free(cache);
    if (code) return forward_c(code, e);
    std::fprintf(stderr, "Vulkan network on %s\n", vulkan_network_device_name(network));
    VulkanEngine engine(network, tier);
    DLSSLOP_TRY(engine.prepare());
    return engine;
}
} // namespace

Result<OpenedEngine> open_engine(struct options& o, unsigned tier)
{
    OpenedEngine opened;
    if (o.test_identity) return opened;
    if (const char* hip = options_hip_only(&o); hip && o.backend == OPTIONS_BACKEND_AUTO) {
        std::fprintf(stderr, "%s needs the HIP network; using HIP\n", hip);
    } else if (o.backend != OPTIONS_BACKEND_HIP) {
        auto vulkan = open_vulkan(o, tier);
        if (vulkan) {
            opened.vulkan.emplace(*std::move(vulkan));
            return opened;
        }
        if (o.backend == OPTIONS_BACKEND_VULKAN) return forward(std::move(vulkan).error());
        std::fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", vulkan.error().what.c_str());
    }
    opened.hip = DLSSLOP_TRY(open_hip(o));
    return opened;
}
} // namespace dlsslop
