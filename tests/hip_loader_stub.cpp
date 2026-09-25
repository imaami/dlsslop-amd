// HIP runtime stand-ins for the worker's loader and device-selection tests.
// libhip-loader-stub.so needs libhip-loader-stub-dependency.so, which the test
// leaves unreachable. libhip-fake-runtime.so loads and reports one device per
// architecture that HIP_FAKE_ARCHS lists (comma separated; unset or empty:
// none); every call the device selection does not make fails.
#if defined(HIP_LOADER_STUB_DEPENDENCY)
int hip_loader_stub_dependency() { return 0; }
#elif defined(HIP_FAKE_RUNTIME)
#include "../backend/vendor/hip_device_properties.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> archs()
{
    const char* text = std::getenv("HIP_FAKE_ARCHS");
    std::istringstream list(text ? text : "");
    std::vector<std::string> result;
    for (std::string arch; std::getline(list, arch, ',');) result.push_back(arch);
    return result;
}
}

extern "C" {
int hipInit(unsigned) { return 0; }
int hipRuntimeGetVersion(int* version) { *version = 60443000; return 0; }
int hipGetDeviceCount(int* count) { *count = static_cast<int>(archs().size()); return 0; }
int hipGetDevicePropertiesR0600(hip_probe::DevicePropertiesR0600* p, int device)
{
    const auto list = archs();
    if (device < 0 || static_cast<size_t>(device) >= list.size()) return 1;
    std::snprintf(p->name, sizeof p->name, "Fake device %d", device);
    std::snprintf(p->gcnArchName, sizeof p->gcnArchName, "%s", list[device].c_str());
    return 0;
}
const char* hipGetErrorName(int) { return "hipErrorInvalidValue"; }
// The worker resolves these when it loads the runtime.
#define HIP_FAKE_FAILS(name) int name() { return 1; }
HIP_FAKE_FAILS(hipModuleLoadData) HIP_FAKE_FAILS(hipEventCreate) HIP_FAKE_FAILS(hipEventRecord)
HIP_FAKE_FAILS(hipEventElapsedTime) HIP_FAKE_FAILS(hipEventDestroy) HIP_FAKE_FAILS(hipEventSynchronize)
HIP_FAKE_FAILS(hipHostMalloc) HIP_FAKE_FAILS(hipDeviceGetName) HIP_FAKE_FAILS(hipSetDevice)
HIP_FAKE_FAILS(hipMemGetInfo) HIP_FAKE_FAILS(hipMalloc) HIP_FAKE_FAILS(hipFree) HIP_FAKE_FAILS(hipMemcpy)
HIP_FAKE_FAILS(hipMemcpyAsync) HIP_FAKE_FAILS(hipMemsetAsync) HIP_FAKE_FAILS(hipDeviceSynchronize)
HIP_FAKE_FAILS(hipStreamCreate) HIP_FAKE_FAILS(hipStreamSynchronize) HIP_FAKE_FAILS(hipStreamDestroy)
HIP_FAKE_FAILS(hipImportExternalMemory) HIP_FAKE_FAILS(hipExternalMemoryGetMappedBuffer)
HIP_FAKE_FAILS(hipDestroyExternalMemory) HIP_FAKE_FAILS(hipImportExternalSemaphore)
HIP_FAKE_FAILS(hipSignalExternalSemaphoresAsync) HIP_FAKE_FAILS(hipWaitExternalSemaphoresAsync)
HIP_FAKE_FAILS(hipDestroyExternalSemaphore) HIP_FAKE_FAILS(hipModuleLoad)
HIP_FAKE_FAILS(hipModuleGetFunction) HIP_FAKE_FAILS(hipModuleLaunchKernel) HIP_FAKE_FAILS(hipModuleUnload)
}
#else
int hip_loader_stub_dependency();
int hip_loader_stub() { return hip_loader_stub_dependency(); }
#endif
