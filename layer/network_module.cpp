// libdlsslop-network.so: the in-layer network (network_module.h).
// SPDX-License-Identifier: MIT
#include "network_module.h"
#include "files.h"
#include "network_recorder.h"
#include "options.h"
#include "paths.h"
#include "processing.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>
#include <pthread.h>
#include <string>
#include <utility>

namespace {
using dlsslop::NetworkRecorder;

// Where the module finds the network: MODEL, its shaders beside the module,
// installed or in a source build, and the user's pipeline cache.
dlsslop::VulkanPaths module_paths(const std::string& model)
{
    Dl_info self{};
    dladdr(reinterpret_cast<void*>(&dlsslop_network_open), &self);
    const std::string directory = self.dli_fname ? dlsslop::parent_path(dlsslop::absolute(self.dli_fname)) : std::string();
    return {model, dlsslop::vulkan_shaders(dlsslop::parent_path(dlsslop::parent_path(directory)), directory),
            dlsslop::vulkan_cache()};
}

// The game's device, with the next layer's physical-device functions the build queries, looked
// up now: a lookup on the build thread would go through the loader, which takes the loader's
// lock, and vkDestroyDevice holds that lock while the layer waits for the build to end.
dlsslop::vulkan::Device network_device(const DlsslopNetworkDevice& d)
{
    const auto find = [&d](const char* name) { return d.physicalDispatch(d.instance, name); };
    dlsslop::vulkan::Device device{};
    device.instance = d.instance;
    device.physical = d.physical;
    device.device = d.device;
    device.queue = d.queue;
    device.family = d.family;
    device.lock = d.lockQueue;
    device.unlock = d.unlockQueue;
    device.context = d.context;
    device.memory = d.memory;
    device.functions = {
        reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            find("vkGetPhysicalDeviceQueueFamilyProperties")),
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(find("vkGetPhysicalDeviceProperties2")),
        reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(find("vkGetPhysicalDeviceFormatProperties2")),
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            find("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"))};
    device.log = d.log;
    return device;
}
}  // namespace

struct DlsslopNetwork {
    // The model dlsslopd loads, as its config file names it; or why that is unknown, which fails
    // the network.
    dlsslop::Result<std::string> model;
    NetworkRecorder recorder;
    dlsslop::VulkanFrame prepared;
    // A build in the background: its frame, and whether it still runs. The
    // builder sets failed and error before it clears building.
    pthread_t builder{};
    bool joinable = false;
    std::atomic<bool> building = false;
    dlsslop::VulkanFrame target;
    bool failed = false;
    std::string error;

    DlsslopNetwork(const DlsslopNetworkDevice& d)
        : model(dlsslop::configured_vulkan_model()),
          recorder(network_device(d), module_paths(model.value_or(std::string())))
    {
        if (!model) fail(model.error().what);
    }

    // The network cannot run: WHAT says why.
    int fail(std::string what)
    {
        error = std::move(what);
        failed = true;
        return kDlsslopNetworkFailed;
    }

    static void* build(void* self)
    {
        auto& n = *static_cast<DlsslopNetwork*>(self);
        if (auto built = n.recorder.shape(n.target); !built) n.fail(std::move(built).error().what);
        n.building.store(false, std::memory_order_release);
        return nullptr;
    }
};

extern "C" {

DlsslopNetwork* dlsslop_network_open(const DlsslopNetworkDevice* device)
{
    return new (std::nothrow) DlsslopNetwork(*device);
}

int dlsslop_network_prepare(DlsslopNetwork* n, const ShmHeader* channel, uint32_t width, uint32_t height, int fp16)
{
    if (n->building.load(std::memory_order_acquire)) return kDlsslopNetworkBuilding;
    if (n->joinable) {
        pthread_join(n->builder, nullptr);
        n->joinable = false;
    }
    if (n->failed) return kDlsslopNetworkFailed;
    auto settings = dlsslop::read_settings(channel);
    if (!settings) {
        n->error = std::move(settings).error().what;
        return kDlsslopNetworkRejected;
    }
    settings->fp16 = fp16 != 0;
    const unsigned passes = std::min(ShmPasses(channel), NetworkRecorder::kMaxPasses);
    const auto frame = dlsslop::vulkan_frame(width, height, passes, *settings);
    if (!n->recorder.shape_differs(frame)) {
        n->prepared = frame;
        return kDlsslopNetworkReady;
    }
    if (auto model = dlsslop::require_vulkan_model(*n->model); !model) return n->fail(std::move(model).error().what);
    // A new extent is planned here, so that one the network does not take on the device is
    // refused without a build; the build thread builds from that plan.
    if (auto planned = n->recorder.plan(frame); !planned) {
        if (!planned.error().rejected) return n->fail(std::move(planned).error().what);
        n->error = std::move(planned).error().what;
        return kDlsslopNetworkRejected;
    }
    n->target = frame;
    n->building.store(true, std::memory_order_relaxed);
    if (const int error = pthread_create(&n->builder, nullptr, DlsslopNetwork::build, n)) {
        n->building.store(false, std::memory_order_relaxed);
        return n->fail(std::string("start the network's build: ") + std::strerror(error));
    }
    n->joinable = true;
    return kDlsslopNetworkBuilding;
}

int dlsslop_network_record(DlsslopNetwork* n, VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, uint32_t family,
                           int exported)
{
    n->recorder.record(cmd, proxy, answer, n->prepared, family, exported != 0);
    return kDlsslopNetworkReady;
}

const char* dlsslop_network_error(const DlsslopNetwork* n) { return n->error.c_str(); }

void dlsslop_network_close(DlsslopNetwork* n)
{
    if (!n) return;
    if (n->joinable) pthread_join(n->builder, nullptr);
    delete n;
}

}  // extern "C"
