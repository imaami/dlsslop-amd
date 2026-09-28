// The in-layer network's module, libdlsslop-network.so, beside the layer. The
// layer loads it for a device whose ledger enabled the network, and reaches it
// through these C functions only: vulkan-nr, its exceptions and its iostreams
// stay out of the layer. C++17, like the layer.
// SPDX-License-Identifier: MIT
#pragma once

#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <cstdint>
#include <string>

struct ShmHeader;
struct DlsslopNetwork;

extern "C" {

// The game's device, as the layer knows it.
struct DlsslopNetworkDevice {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    // The queue that takes the network's uploads while it builds, from a
    // thread of the module's own: LOCK_QUEUE and UNLOCK_QUEUE, with CONTEXT,
    // are called around each of those submits.
    VkQueue queue;
    uint32_t family;
    void (*lockQueue)(void* context);
    void (*unlockQueue)(void* context);
    void* context;
    // The next layer's, for physical-device queries through the layer's handles.
    PFN_vkGetInstanceProcAddr physicalDispatch;
    VkPhysicalDeviceMemoryProperties memory;
    // Where the network's own log lines go.
    void (*log)(const char* line);
};

// What prepare() says of the next frame.
enum DlsslopNetworkState {
    kDlsslopNetworkReady,     // record() records it
    kDlsslopNetworkBuilding,  // the network is being built for it in the background
    kDlsslopNetworkRejected,  // its settings are out of range; error() says which
    kDlsslopNetworkFailed,    // the network cannot run; error() says why
};

// The network on DEVICE, not yet built; null without memory.
DlsslopNetwork* dlsslop_network_open(const DlsslopNetworkDevice* device);
// Readies the network for the next frame: WIDTH x HEIGHT, RGBA8 or with FP16
// RGBA16F, with the channel's settings. A frame of another shape starts a
// build, which must not overlap the network's recorded work: the caller has
// waited for its last frame.
int dlsslop_network_prepare(DlsslopNetwork* network, const ShmHeader* channel, uint32_t width, uint32_t height,
                            int fp16);
// Records the frame prepare() readied: the proxy the composition captured
// into PROXY, through the network, into ANSWER for the composition. Both are
// the composition's transfer buffers; EXPORTED says they belong to
// VK_QUEUE_FAMILY_EXTERNAL between uses, and FAMILY is the queue's.
// kDlsslopNetworkReady or kDlsslopNetworkFailed; failed, CMD still holds
// valid commands, which give the pair back as they found it.
int dlsslop_network_record(DlsslopNetwork* network, VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer,
                           uint32_t family, int exported);
const char* dlsslop_network_error(const DlsslopNetwork* network);
// Waits for a build to end. The device must have finished the network's work.
void dlsslop_network_close(DlsslopNetwork* network);

}  // extern "C"

namespace dlssnr {

// The module's functions, as the layer finds them.
struct NetworkModule {
    void* library = nullptr;
    decltype(&dlsslop_network_open) open = nullptr;
    decltype(&dlsslop_network_prepare) prepare = nullptr;
    decltype(&dlsslop_network_record) record = nullptr;
    decltype(&dlsslop_network_error) error = nullptr;
    decltype(&dlsslop_network_close) close = nullptr;
    // Why Load failed.
    std::string failure;

    // Loads the module at PATH.
    bool Load(const char* path) {
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (library) {
#define FIND(field, name) field = reinterpret_cast<decltype(field)>(dlsym(library, #name))
            FIND(open, dlsslop_network_open);
            FIND(prepare, dlsslop_network_prepare);
            FIND(record, dlsslop_network_record);
            FIND(error, dlsslop_network_error);
            FIND(close, dlsslop_network_close);
#undef FIND
            if (open && prepare && record && error && close) return true;
        }
        const char* why = dlerror();
        failure = why ? why : path;
        return false;
    }
};

}  // namespace dlssnr
