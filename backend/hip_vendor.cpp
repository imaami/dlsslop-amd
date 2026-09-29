// The boundary to the vendored HIP network, which throws: every call into it is
// caught here, and only here, and returned as its failure. Built with
// exceptions, as C++20 (see hip_vendor.h).
// SPDX-License-Identifier: MIT
#include "hip_vendor.h"
#include "vendor/LmxxfProductionOptions.h"

#include <dlfcn.h>
#include <exception>

namespace dlsslop::hip {

class Network : public hip_reference::Network {
public:
    using hip_reference::Network::Network;
};

void NetworkDeleter::operator()(Network* network) const noexcept { delete network; }

Handle stream(Network& network) { return network.Stream(); }

namespace vendor {
namespace {
// Nothing if F returns, else what it threw.
template <class F>
std::string caught(F&& f)
{
    try {
        f();
        return {};
    } catch (const std::exception& error) {
        return *error.what() ? error.what() : "unnamed failure in the HIP network";
    } catch (...) {
        return "unknown failure in the HIP network";
    }
}
} // namespace

std::string load(Entries& api)
{
    return caught([&] {
        const hip_probe::Api vendor; // Loads and keeps the runtime for the process.
        api.hipGetDevicePropertiesR0600 = vendor.hipGetDevicePropertiesR0600;
        api.hipInit = vendor.hipInit;
        api.hipRuntimeGetVersion = vendor.hipRuntimeGetVersion;
        api.hipGetDeviceCount = vendor.hipGetDeviceCount;
        api.hipSetDevice = vendor.hipSetDevice;
        api.hipMalloc = vendor.hipMalloc;
        api.hipFree = vendor.hipFree;
        api.hipHostMalloc = vendor.hipHostMalloc;
        api.hipMemcpy = vendor.hipMemcpy;
        api.hipMemcpyAsync = vendor.hipMemcpyAsync;
        api.hipMemsetAsync = vendor.hipMemsetAsync;
        api.hipEventCreate = vendor.hipEventCreate;
        api.hipEventRecord = vendor.hipEventRecord;
        api.hipEventElapsedTime = vendor.hipEventElapsedTime;
        api.hipEventDestroy = vendor.hipEventDestroy;
        api.hipEventSynchronize = vendor.hipEventSynchronize;
        api.hipStreamCreate = vendor.hipStreamCreate;
        api.hipStreamSynchronize = vendor.hipStreamSynchronize;
        api.hipStreamDestroy = vendor.hipStreamDestroy;
        api.hipImportExternalMemory = reinterpret_cast<decltype(api.hipImportExternalMemory)>(vendor.hipImportExternalMemory);
        api.hipExternalMemoryGetMappedBuffer =
            reinterpret_cast<decltype(api.hipExternalMemoryGetMappedBuffer)>(vendor.hipExternalMemoryGetMappedBuffer);
        api.hipDestroyExternalMemory = vendor.hipDestroyExternalMemory;
        api.hipModuleLoadData = vendor.hipModuleLoadData;
        api.hipModuleGetFunction = vendor.hipModuleGetFunction;
        api.hipModuleLaunchKernel = vendor.hipModuleLaunchKernel;
        api.hipModuleUnload = vendor.hipModuleUnload;
        api.hipGetErrorName = vendor.hipGetErrorName;
        // Resolved here so the vendored loader stays as upstream adapted it.
        api.hipHostFree = reinterpret_cast<decltype(api.hipHostFree)>(dlsym(vendor.dll, "hipHostFree"));
        api.hipSetDeviceFlags = reinterpret_cast<decltype(api.hipSetDeviceFlags)>(dlsym(vendor.dll, "hipSetDeviceFlags"));
        api.hipHostRegister = reinterpret_cast<decltype(api.hipHostRegister)>(dlsym(vendor.dll, "hipHostRegister"));
        api.hipHostUnregister = reinterpret_cast<decltype(api.hipHostUnregister)>(dlsym(vendor.dll, "hipHostUnregister"));
    });
}

std::string network(const NetworkOptions& options, NetworkHandle& network)
{
    return caught([&] {
        auto opt = LmxxfProductionOptions(options.width, options.height, options.modules, options.assets);
        opt.device = options.device;
        if (!options.performance) opt.skip_blocks.clear();
        // Production launches nothing from the WMMA, tiled, wave and fused-C32
        // modules these select; do not load them.
        opt.wmma = opt.tiled = opt.wave = opt.fused_c32 = false;
        // scripts/build-kernels.py has no recipe yet for the modules these
        // select (c32-wave1, c64-wave2, c512-m32-*, vit-*, wave-pointwise).
        opt.wave_owned = opt.c512_m32 = opt.vit_proj_n64 = opt.pdl = false;
        network.reset(new Network(opt));
        network->SetNoise({}); // Fast prefix uses procedural noise, not noise.f32.
    });
}

std::string enqueue(Network& network, void* rgba, void* history, void* rgb)
{
    return caught([&] { network.Enqueue(rgba, history, rgb, 0); });
}

std::string synchronize(Network& network)
{
    return caught([&] { network.Synchronize(); });
}

std::string print_memory(Network& network)
{
    return caught([&] { network.PrintMemory(); });
}

} // namespace vendor
} // namespace dlsslop::hip
