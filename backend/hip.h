// The HIP runtime as dlsslopd calls it: the few ROCm 7 types it passes, its
// entry points, loaded with dlopen, and code objects loaded from files. The
// build needs no ROCm headers or GPU; static_asserts pin the layouts.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"

#include <cstddef>
#include <string>

namespace dlsslop::hip {

using Handle = void*;

// hipDeviceProp_tR0600, as ROCm 6 and 7 lay it out, with only the fields
// dlsslopd reads named; the runtime writes all of it.
struct DeviceProperties {
    char name[256];
    unsigned char reserved0[32];
    size_t totalGlobalMem;
    unsigned char reserved1[288];
    int pciBusID, pciDeviceID, pciDomainID;
    unsigned char reserved2[564];
    char gcnArchName[256];
    unsigned char reserved3[56];
};
static_assert(offsetof(DeviceProperties, totalGlobalMem) == 288 && offsetof(DeviceProperties, pciBusID) == 584 &&
                  offsetof(DeviceProperties, pciDeviceID) == 588 && offsetof(DeviceProperties, pciDomainID) == 592 &&
                  offsetof(DeviceProperties, gcnArchName) == 1160 && sizeof(DeviceProperties) == 1472,
              "HIP R0600 x64 ABI");

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

// The runtime entry points dlsslopd calls, as X(name, result, parameters,
// required). The runtime must export every required one; an optional one is
// null when it does not.
#define DLSSLOP_HIP_ENTRIES(X)                                                                                        \
    X(hipGetDevicePropertiesR0600, int, (DeviceProperties*, int), true)                                               \
    X(hipInit, int, (unsigned), true)                                                                                 \
    X(hipRuntimeGetVersion, int, (int*), true)                                                                        \
    X(hipGetDeviceCount, int, (int*), true)                                                                           \
    X(hipSetDevice, int, (int), true)                                                                                 \
    /* Without it, waiting threads spin. */                                                                           \
    X(hipSetDeviceFlags, int, (unsigned), false)                                                                      \
    X(hipMalloc, int, (void**, size_t), true)                                                                         \
    X(hipFree, int, (void*), true)                                                                                    \
    X(hipHostMalloc, int, (void**, size_t, unsigned), true)                                                           \
    /* The GPU codec needs it. */                                                                                     \
    X(hipHostFree, int, (void*), false)                                                                               \
    /* Without them, shared memory stays pageable. */                                                                 \
    X(hipHostRegister, int, (void*, size_t, unsigned), false)                                                         \
    X(hipHostUnregister, int, (void*), false)                                                                         \
    X(hipMemcpy, int, (void*, const void*, size_t, int), true)                                                        \
    X(hipMemcpyAsync, int, (void*, const void*, size_t, int, Handle), true)                                           \
    X(hipMemsetAsync, int, (void*, int, size_t, Handle), true)                                                        \
    /* For the network's memory report. */                                                                            \
    X(hipMemGetInfo, int, (size_t*, size_t*), true)                                                                   \
    X(hipEventCreate, int, (Handle*), true)                                                                           \
    X(hipEventRecord, int, (Handle, Handle), true)                                                                    \
    X(hipEventElapsedTime, int, (float*, Handle, Handle), true)                                                       \
    X(hipEventDestroy, int, (Handle), true)                                                                           \
    X(hipEventSynchronize, int, (Handle), true)                                                                       \
    X(hipStreamCreate, int, (Handle*), true)                                                                          \
    X(hipStreamSynchronize, int, (Handle), true)                                                                      \
    X(hipStreamDestroy, int, (Handle), true)                                                                          \
    X(hipImportExternalMemory, int, (Handle*, const MemoryDesc*), true)                                               \
    X(hipExternalMemoryGetMappedBuffer, int, (void**, Handle, const BufferDesc*), true)                               \
    X(hipDestroyExternalMemory, int, (Handle), true)                                                                  \
    X(hipModuleLoadData, int, (Handle*, const void*), true)                                                           \
    X(hipModuleGetFunction, int, (Handle*, Handle, const char*), true)                                                \
    X(hipModuleLaunchKernel, int,                                                                                     \
      (Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Handle, void**, void**), true)   \
    X(hipModuleUnload, int, (Handle), true)                                                                           \
    X(hipGetErrorName, const char*, (int), true)

// The loaded runtime. It stays loaded for the process: the driver's threads
// may outlive every user.
struct Api {
#define DLSSLOP_HIP_ENTRY(name, result, parameters, required) result(*name) parameters;
    DLSSLOP_HIP_ENTRIES(DLSSLOP_HIP_ENTRY)
#undef DLSSLOP_HIP_ENTRY

    // Nothing unless RESULT is an error: then "WHAT: <its name> (RESULT)".
    Result<void> check(int result, const char* what) const
    {
        if (!result) return {};
        return fail(std::string(what) + ": " + hipGetErrorName(result) + " (" + std::to_string(result) + ")");
    }
};

// The runtime: DLSSLOP_HIP_LIBRARY when set, else the libamdhip64 sonames,
// then /opt/rocm/lib. On failure, every candidate's loader error.
Result<Api> load();
// The code object at PATH, loaded into the current device's context.
Result<Handle> load_module(const Api& api, const std::string& path);

} // namespace dlsslop::hip
