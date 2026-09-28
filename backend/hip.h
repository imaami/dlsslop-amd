// HIP for the daemon's own code: the runtime's entry points as a table, and the
// vendored lmxxf network behind functions that return Results. The vendored code
// throws; hip_vendor.cpp, the only file that includes it, is built with
// exceptions to catch them.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"
#include "vendor/hip_device_properties.h"

#include <cstddef>
#include <memory>
#include <string>

namespace dlsslop::hip {

using Handle = void*;
using DeviceProperties = hip_probe::DevicePropertiesR0600;

// hipExternalMemoryHandleDesc and hipExternalMemoryBufferDesc, as ROCm 7 lays them out.
struct MemoryDesc {
    int type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* object;
    } handle;
    unsigned long long size;
    unsigned flags;
    unsigned reserved[16];
};
struct BufferDesc {
    unsigned long long offset, size;
    unsigned flags;
    unsigned reserved[16];
};
static_assert(sizeof(MemoryDesc) == 104 && sizeof(BufferDesc) == 88, "HIP external memory ABI");

// The runtime entry points dlsslopd calls itself.
struct Api {
    int (*hipGetDevicePropertiesR0600)(DeviceProperties*, int);
    int (*hipInit)(unsigned);
    int (*hipRuntimeGetVersion)(int*);
    int (*hipGetDeviceCount)(int*);
    int (*hipSetDevice)(int);
    int (*hipMalloc)(void**, size_t);
    int (*hipFree)(void*);
    int (*hipHostMalloc)(void**, size_t, unsigned);
    // Null when the runtime lacks it; the GPU codec needs it.
    int (*hipHostFree)(void*);
    // Null when the runtime lacks them: shared memory then stays pageable.
    int (*hipHostRegister)(void*, size_t, unsigned);
    int (*hipHostUnregister)(void*);
    int (*hipMemcpy)(void*, const void*, size_t, int);
    int (*hipMemcpyAsync)(void*, const void*, size_t, int, Handle);
    int (*hipMemsetAsync)(void*, int, size_t, Handle);
    int (*hipEventCreate)(Handle*);
    int (*hipEventRecord)(Handle, Handle);
    int (*hipEventElapsedTime)(float*, Handle, Handle);
    int (*hipEventDestroy)(Handle);
    int (*hipEventSynchronize)(Handle);
    int (*hipStreamCreate)(Handle*);
    int (*hipStreamSynchronize)(Handle);
    int (*hipStreamDestroy)(Handle);
    int (*hipImportExternalMemory)(Handle*, const MemoryDesc*);
    int (*hipExternalMemoryGetMappedBuffer)(void**, Handle, const BufferDesc*);
    int (*hipDestroyExternalMemory)(Handle);
    int (*hipModuleLoadData)(Handle*, const void*);
    int (*hipModuleGetFunction)(Handle*, Handle, const char*);
    int (*hipModuleLaunchKernel)(Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Handle,
                                 void**, void**);
    int (*hipModuleUnload)(Handle);
    const char* (*hipGetErrorName)(int);

    // Nothing unless RESULT is an error: then "WHAT: <its name> (RESULT)".
    Result<void> check(int result, const char* what) const
    {
        if (!result) return {};
        return fail(std::string(what) + ": " + hipGetErrorName(result) + " (" + std::to_string(result) + ")");
    }
};

// The runtime, found as the vendored network finds it: DLSSLOP_HIP_LIBRARY, or
// the libamdhip64 sonames, then /opt/rocm/lib. On failure, every candidate's
// loader error.
Result<Api> load();

// hip_reference::Network, opaque outside hip_vendor.cpp.
class Network;
struct NetworkDeleter {
    void operator()(Network* network) const noexcept;
};
using NetworkHandle = std::unique_ptr<Network, NetworkDeleter>;

struct NetworkOptions {
    unsigned width, height, device;
    std::string modules, assets;
    bool performance; // Skip upstream's performance-preset blocks 42, 43 and 46.
};
// The pinned network with dlsslopd's production options, on the device it makes
// current for this thread.
Result<NetworkHandle> network(const NetworkOptions& options);
Handle stream(Network& network);
// One evaluation, queued on the network's stream.
Result<void> enqueue(Network& network, void* rgba, void* history, void* rgb);
Result<void> synchronize(Network& network);
// The network's memory use, to stdout.
Result<void> print_memory(Network& network);

} // namespace dlsslop::hip
