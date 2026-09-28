// The boundary to the vendored HIP network, which throws: every call into it is
// caught here, and only here, and returned as a Result. Built with exceptions.
// SPDX-License-Identifier: MIT
#include "hip.h"
#include "vendor/LmxxfProductionOptions.h"

#include <dlfcn.h>
#include <exception>

namespace dlsslop::hip {

class Network : public hip_reference::Network {
public:
    using hip_reference::Network::Network;
};

void NetworkDeleter::operator()(Network* network) const noexcept { delete network; }

namespace {
// What F returns, or the exception it threw as an Error.
template <class F>
auto caught(F&& f) -> Result<decltype(f())>
{
    try {
        if constexpr (std::is_void_v<decltype(f())>) {
            f();
            return {};
        } else {
            return f();
        }
    } catch (const std::exception& error) {
        return fail(error.what());
    } catch (...) {
        return fail("unknown failure in the HIP network");
    }
}
} // namespace

Result<Api> load()
{
    return caught([] {
        const hip_probe::Api vendor; // Loads and keeps the runtime for the process.
        Api api{};
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
        api.hipHostRegister = reinterpret_cast<decltype(api.hipHostRegister)>(dlsym(vendor.dll, "hipHostRegister"));
        api.hipHostUnregister = reinterpret_cast<decltype(api.hipHostUnregister)>(dlsym(vendor.dll, "hipHostUnregister"));
        return api;
    });
}

Result<NetworkHandle> network(const NetworkOptions& options)
{
    return caught([&] {
        auto opt = LmxxfProductionOptions(options.width, options.height, options.modules, options.assets);
        opt.device = options.device;
        if (!options.performance) opt.skip_blocks.clear();
        // Upstream's shipped HIP configurations (scripts/hip-*-flags.txt) add
        // these bit-exact byte residual and fragment paths to the snapshot.
        opt.mh_feature_byte = opt.mh_proj_diag_fb = opt.mh_byte_stream = opt.decoder_byte = opt.mh_ffn_frag256 = true;
        // Production launches nothing from the WMMA, tiled, wave and fused-C32
        // modules these select; do not load them.
        opt.wmma = opt.tiled = opt.wave = opt.fused_c32 = false;
        NetworkHandle network(new Network(opt));
        network->SetNoise({}); // Fast prefix uses procedural noise, not noise.f32.
        return network;
    });
}

Handle stream(Network& network) { return network.Stream(); }

Result<void> enqueue(Network& network, void* rgba, void* history, void* rgb)
{
    return caught([&] { network.Enqueue(rgba, history, rgb, 0); });
}

Result<void> synchronize(Network& network)
{
    return caught([&] { network.Synchronize(); });
}

Result<void> print_memory(Network& network)
{
    return caught([&] { network.PrintMemory(); });
}

} // namespace dlsslop::hip
