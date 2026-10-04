// libdlsslop-network.so: the in-layer network (network_module.h).
// SPDX-License-Identifier: MIT
#include "network_module.h"
#include "files.hpp"
#include "network_recorder.h"
#include "options.hpp"
#include "paths.hpp"
#include "processing.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <new>
#include <pthread.h>
#include <string>
#include <utility>

namespace {

// Where the module finds the network's shaders, beside the module, installed or in a source build;
// and the user's pipeline cache.
struct ModulePaths {
    std::string shaders, cache;
};
ModulePaths module_paths()
{
    Dl_info self{};
    dladdr(reinterpret_cast<void*>(&dlsslop_network_open), &self);
    const std::string directory = self.dli_fname ? dlsslop::parent_path(dlsslop::absolute(self.dli_fname)) : std::string();
    return {dlsslop::vulkan_shaders(dlsslop::parent_path(dlsslop::parent_path(directory)), directory),
            dlsslop::vulkan_cache()};
}

// The game's device, with the next layer's physical-device functions the build queries, looked
// up now: a lookup on the build thread would go through the loader, which takes the loader's
// lock, and vkDestroyDevice holds that lock while the layer waits for the build to end.
struct vulkan_device network_device(const dlsslop_network_device& d)
{
    const auto find = [&d](const char* name) { return d.physical_dispatch(d.instance, name); };
    struct vulkan_device device{};
    device.instance = d.instance;
    device.physical = d.physical;
    device.device = d.device;
    device.queue = d.queue;
    device.family = d.family;
    device.lock = d.lock_queue;
    device.unlock = d.unlock_queue;
    device.context = d.context;
    device.memory = d.memory;
    device.functions = {
        reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            find("vkGetPhysicalDeviceQueueFamilyProperties")),
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(find("vkGetPhysicalDeviceProperties2")),
        reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(find("vkGetPhysicalDeviceFormatProperties2"))};
    device.log = d.log;
    return device;
}
}  // namespace

struct DlsslopNetwork {
    // The model dlsslopd loads, as its config file names it; or why that is unknown, which fails
    // the network.
    dlsslop::Result<std::string> model;
    // Empty when it could not be made, which fails the network.
    struct network_recorder recorder{};
    struct vulkan_frame prepared = VULKAN_FRAME_DEFAULTS;
    // A build in the background: its frame, and whether it still runs. The
    // builder sets failed and error before it clears building.
    pthread_t builder{};
    bool joinable = false;
    std::atomic<bool> building = false;
    struct vulkan_frame target = VULKAN_FRAME_DEFAULTS;
    bool failed = false;
    std::string error;

    DlsslopNetwork(const dlsslop_network_device& d) : model(dlsslop::configured_vulkan_model())
    {
        // The recorder copies the paths, which borrow these strings.
        const std::string path = model.value_or(std::string());
        const ModulePaths found = module_paths();
        const struct vulkan_device device = network_device(d);
        const struct vulkan_paths paths{.model = path.c_str(),
                                        .shaders = found.shaders.c_str(),
                                        .cache = found.cache.c_str(),
                                        .model_length = path.size(),
                                        .shaders_length = found.shaders.size(),
                                        .cache_length = found.cache.size()};
        struct error e;
        if (network_recorder_init(&recorder, &device, &paths, true, &e) != ERROR_NONE) fail(e.what);
        if (!model) fail(model.error().what);
    }
    DlsslopNetwork(const DlsslopNetwork&) = delete;
    ~DlsslopNetwork() { network_recorder_fini(&recorder); }

    // The network cannot run: WHAT says why.
    dlsslop_network_state fail(std::string what)
    {
        error = std::move(what);
        failed = true;
        return DLSSLOP_NETWORK_FAILED;
    }

    static void* build(void* self)
    {
        auto& n = *static_cast<DlsslopNetwork*>(self);
        bool built = false;
        struct error e;
        if (network_recorder_shape(&n.recorder, &n.target, nullptr, &built, &e) != ERROR_NONE) n.fail(e.what);
        n.building.store(false, std::memory_order_release);
        return nullptr;
    }
};

extern "C" {

const uint64_t dlsslop_network_interface = DLSSLOP_NETWORK_INTERFACE;

DlsslopNetwork* dlsslop_network_open(const dlsslop_network_device* device)
{
    // A layer of another interface describes its device otherwise: an older one's begins with its
    // VkInstance, which never equals the interface.
    if (device->interface != DLSSLOP_NETWORK_INTERFACE) return nullptr;
    return new (std::nothrow) DlsslopNetwork(*device);
}

dlsslop_network_state dlsslop_network_prepare(DlsslopNetwork* n, const ShmHeader* channel,
                                              const dlsslop_network_images* images)
{
    if (n->building.load(std::memory_order_acquire)) return DLSSLOP_NETWORK_BUILDING;
    if (n->joinable) {
        pthread_join(n->builder, nullptr);
        n->joinable = false;
    }
    if (n->failed) return DLSSLOP_NETWORK_FAILED;
    // A channel of another protocol has its settings elsewhere.
    if (const uint32_t version = channel->version.load(std::memory_order_relaxed); version != kShmVersion) {
        char what[80];
        std::snprintf(what, sizeof what, "the channel is of protocol v%u, the network of v%u", version,
                      unsigned(kShmVersion));
        return n->fail(what);
    }
    // The frame's format: the composition's proxy, RGBA8 or the float16 proxy.
    const bool fp16 = images->format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!fp16 && images->format != VK_FORMAT_R8G8B8A8_UNORM)
        return n->fail("the network takes frames of RGBA8 or RGBA16F, not of format " +
                       std::to_string(int(images->format)));
    if (!images->generation) return n->fail("the composition's images have no generation");
    auto settings = dlsslop::read_settings(channel);
    if (!settings) {
        n->error = std::move(settings).error().what;
        return DLSSLOP_NETWORK_REJECTED;
    }
    settings->fp16 = fp16;
    const unsigned passes = std::min(ShmPasses(channel), NETWORK_RECORDER_MAX_PASSES);
    const struct vulkan_frame frame = dlsslop::vulkan_frame(images->width, images->height, passes, *settings);
    // A frame of the network's extent: of its shape, or of another that it is reshaped for here,
    // between frames, with no GPU work, and in images that are bound here when they are new.
    if (network_recorder_has_extent(&n->recorder, &frame)) {
        const struct vulkan_frame_images surfaces{images->generation, images->input_view, images->answer,
                                                  images->answer_view};
        bool shaped = false;
        struct error e;
        if (network_recorder_shape(&n->recorder, &frame, &surfaces, &shaped, &e) != ERROR_NONE) return n->fail(e.what);
        n->prepared = frame;
        return DLSSLOP_NETWORK_READY;
    }
    if (auto model = dlsslop::require_vulkan_model(*n->model); !model) return n->fail(std::move(model).error().what);
    // A new extent is planned here, so that one the network does not take on the device is
    // refused without a build; the build thread builds from that plan.
    struct error e;
    if (const enum error_code code = network_recorder_plan(&n->recorder, &frame, &e)) {
        if (code == ERROR_FAILED) return n->fail(e.what);
        n->error = e.what;
        return DLSSLOP_NETWORK_REJECTED;
    }
    n->target = frame;
    n->building.store(true, std::memory_order_relaxed);
    if (const int error = pthread_create(&n->builder, nullptr, DlsslopNetwork::build, n)) {
        n->building.store(false, std::memory_order_relaxed);
        return n->fail(std::string("start the network's build: ") + std::strerror(error));
    }
    n->joinable = true;
    return DLSSLOP_NETWORK_BUILDING;
}

dlsslop_network_state dlsslop_network_record(DlsslopNetwork* n, VkCommandBuffer cmd)
{
    network_recorder_record_images(&n->recorder, cmd, &n->prepared);
    return DLSSLOP_NETWORK_READY;
}

void dlsslop_network_submitted(DlsslopNetwork* n) { network_recorder_submitted(&n->recorder); }

const char* dlsslop_network_error(const DlsslopNetwork* n) { return n->error.c_str(); }

void dlsslop_network_close(DlsslopNetwork* n)
{
    if (!n) return;
    if (n->joinable) pthread_join(n->builder, nullptr);
    delete n;
}

}  // extern "C"
