// The boundary to the vendored lmxxf network, which throws: hip_vendor.cpp,
// the only file that includes the network, catches everything it throws. It is
// built with exceptions and as C++20, which gives the vendored code a smaller
// binary than C++23, so this header needs neither. Each function returns why
// it failed, or nothing.
// SPDX-License-Identifier: MIT
#pragma once
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
struct Entries {
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
};

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
Handle stream(Network& network);

namespace vendor {
std::string load(Entries& api);
std::string network(const NetworkOptions& options, NetworkHandle& network);
std::string enqueue(Network& network, void* rgba, void* history, void* rgb);
std::string synchronize(Network& network);
std::string print_memory(Network& network);
} // namespace vendor

} // namespace dlsslop::hip
