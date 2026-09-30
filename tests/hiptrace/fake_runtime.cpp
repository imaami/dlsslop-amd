// SPDX-License-Identifier: MIT
// A HIP runtime stand-in for the tracing runtime's offline test
// (hiptrace-test.py). Device and pinned host memory are zeroed host memory, so
// copies, memsets and the tracing runtime's hashes work on them; modules,
// functions, streams and events are distinct addresses; every call it exports
// succeeds. It exports no graph entry points: the tracing runtime reports them
// missing and fails their calls.
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace {
char handles[256];
size_t next_handle = 0;
void* handle() { return &handles[next_handle++ % sizeof handles]; }
} // namespace

extern "C" {
int hipModuleLoadData(void** module, const void*)
{
    *module = handle();
    return 0;
}
int hipModuleGetFunction(void** function, void*, const char*)
{
    *function = handle();
    return 0;
}
int hipModuleLaunchKernel(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**,
                          void**)
{
    return 0;
}
int hipExtModuleLaunchKernel(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, size_t, void*, void**,
                             void**, void*, void*, unsigned)
{
    return 0;
}
int hipStreamCreate(void** stream)
{
    *stream = handle();
    return 0;
}
int hipStreamSynchronize(void*) { return 0; }
int hipEventCreate(void** event)
{
    *event = handle();
    return 0;
}
int hipEventRecord(void*, void*) { return 0; }
int hipEventElapsedTime(float* ms, void*, void*)
{
    *ms = 0;
    return 0;
}
int hipEventDestroy(void*) { return 0; }
int hipMalloc(void** pointer, size_t bytes)
{
    *pointer = std::calloc(1, bytes);
    return *pointer ? 0 : 2; // hipErrorOutOfMemory
}
int hipFree(void* pointer)
{
    std::free(pointer);
    return 0;
}
int hipHostMalloc(void** pointer, size_t bytes, unsigned)
{
    *pointer = std::calloc(1, bytes);
    return *pointer ? 0 : 2; // hipErrorOutOfMemory
}
int hipHostFree(void* pointer)
{
    std::free(pointer);
    return 0;
}
int hipHostRegister(void*, size_t, unsigned) { return 0; }
int hipHostUnregister(void*) { return 0; }
int hipMemcpy(void* dst, const void* src, size_t bytes, int)
{
    std::memcpy(dst, src, bytes);
    return 0;
}
int hipMemcpyAsync(void* dst, const void* src, size_t bytes, int, void*)
{
    std::memcpy(dst, src, bytes);
    return 0;
}
int hipMemsetAsync(void* dst, int value, size_t bytes, void*)
{
    std::memset(dst, value, bytes);
    return 0;
}
const char* hipGetErrorName(int) { return "hipErrorFake"; }
}
