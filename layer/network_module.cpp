// libdlsslop-network.so: the in-layer network (network_module.h).
// SPDX-License-Identifier: MIT
#include "network_module.h"
#include "files.h"
#include "network_recorder.h"
#include "nr_log.hpp"
#include "paths.h"
#include "processing.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <new>
#include <pthread.h>
#include <string>

namespace {
using dlsslop::NetworkRecorder;

// Where the module finds the network: its shaders beside the module, installed
// or in a source build, and the user's model and pipeline cache.
dlsslop::VulkanPaths module_paths()
{
    Dl_info self{};
    dladdr(reinterpret_cast<void*>(&dlsslop_network_open), &self);
    const std::string directory = self.dli_fname ? dlsslop::parent_path(dlsslop::absolute(self.dli_fname)) : std::string();
    return {dlsslop::default_vulkan_model(), dlsslop::vulkan_shaders(dlsslop::parent_path(dlsslop::parent_path(directory)), directory),
            dlsslop::vulkan_cache()};
}

nr::HostDevice host_device(const DlsslopNetworkDevice& d)
{
    nr::HostDevice host;
    host.instance = d.instance;
    host.physical = d.physical;
    host.device = d.device;
    host.queue = d.queue;
    host.queue_family = d.family;
    host.physical_dispatch = d.physicalDispatch;
    host.queue_lock = [lock = d.lockQueue, context = d.context] { lock(context); };
    host.queue_unlock = [unlock = d.unlockQueue, context = d.context] { unlock(context); };
    return host;
}
}  // namespace

struct DlsslopNetwork {
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

    DlsslopNetwork(const DlsslopNetworkDevice& d) : recorder(host_device(d), d.memory, module_paths()) {}

    static void* build(void* self)
    {
        auto& n = *static_cast<DlsslopNetwork*>(self);
        if (auto built = n.recorder.shape(n.target); !built) {
            n.error = std::move(built).error().what;
            n.failed = true;
        }
        n.building.store(false, std::memory_order_release);
        return nullptr;
    }
};

extern "C" {

DlsslopNetwork* dlsslop_network_open(const DlsslopNetworkDevice* device)
{
    if (device->log) nr::set_log_sink(device->log);
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
    const std::string& model = dlsslop::default_vulkan_model();
    if (!dlsslop::is_regular_file(model)) {
        n->error = "no model at " + model + " (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)";
        n->failed = true;
        return kDlsslopNetworkFailed;
    }
    n->target = frame;
    n->building.store(true, std::memory_order_relaxed);
    if (const int error = pthread_create(&n->builder, nullptr, DlsslopNetwork::build, n)) {
        n->building.store(false, std::memory_order_relaxed);
        n->error = std::string("start the network's build: ") + std::strerror(error);
        n->failed = true;
        return kDlsslopNetworkFailed;
    }
    n->joinable = true;
    return kDlsslopNetworkBuilding;
}

int dlsslop_network_record(DlsslopNetwork* n, VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, uint32_t family,
                           int exported)
{
    // The composition's copies wrote the proxy and read the last answer. The
    // pair belongs to VK_QUEUE_FAMILY_EXTERNAL between uses when exported:
    // taken for the network's copies and given back after them.
    const auto hand = [&](bool take, VkAccessFlags proxy_access, VkAccessFlags answer_access) {
        VkBufferMemoryBarrier b[2]{};
        const VkBuffer buffers[2] = {proxy, answer};
        const VkAccessFlags other[2] = {VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        const VkAccessFlags own[2] = {proxy_access, answer_access};
        for (unsigned i = 0; i < 2; ++i) {
            b[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            b[i].srcAccessMask = take ? other[i] : own[i];
            b[i].dstAccessMask = take ? own[i] : other[i];
            b[i].srcQueueFamilyIndex = !exported ? VK_QUEUE_FAMILY_IGNORED : take ? VK_QUEUE_FAMILY_EXTERNAL : family;
            b[i].dstQueueFamilyIndex = !exported ? VK_QUEUE_FAMILY_IGNORED : take ? family : VK_QUEUE_FAMILY_EXTERNAL;
            b[i].buffer = buffers[i];
            b[i].size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, b, 0,
                             nullptr);
    };
    hand(true, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    if (auto recorded = n->recorder.record(cmd, proxy, answer, n->prepared); !recorded) {
        n->error = std::move(recorded).error().what;
        n->failed = true;
        return kDlsslopNetworkFailed;
    }
    hand(false, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
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
