// SPDX-License-Identifier: MIT
#include "hip.hpp"
#include "files.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <span>

namespace dlsslop::hip {

Result<Api> load()
{
    static constexpr const char* kLibraries[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so.6", "/opt/rocm/lib/libamdhip64.so"};
    // A set DLSSLOP_HIP_LIBRARY is the only candidate; an empty one is unset.
    const char* library = std::getenv("DLSSLOP_HIP_LIBRARY");
    std::span<const char* const> candidates = kLibraries;
    if (library && *library) candidates = {&library, 1};
    // Each candidate's error, so that a runtime which is installed but cannot
    // load is not blamed on an absent one.
    std::string errors;
    void* dll = nullptr;
    for (const char* name : candidates) {
        if ((dll = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
        errors += "\n  ";
        errors += dlerror();
    }
    if (!dll)
        return fail("Linux HIP runtime not found; install gfx1201-capable ROCm userspace (libamdhip64), or set "
                    "DLSSLOP_HIP_LIBRARY:" + errors);
    // Each entry point's name, its place in an Api and whether the runtime must export it.
    static constexpr struct Entry {
        const char* name;
        uint16_t offset;
        bool required;
    } kEntries[] = {
#define DLSSLOP_HIP_ENTRY(name, result, parameters, required) {#name, offsetof(Api, name), required},
        DLSSLOP_HIP_ENTRIES(DLSSLOP_HIP_ENTRY)
#undef DLSSLOP_HIP_ENTRY
    };
    static_assert(sizeof(Api) == sizeof kEntries / sizeof *kEntries * sizeof(void*), "an Api is its entry points");
    Api api{};
    for (const Entry& entry : kEntries) {
        void* const symbol = dlsym(dll, entry.name);
        if (!symbol && entry.required) return fail("missing HIP export " + std::string(entry.name));
        // POSIX makes the address dlsym gives for a function a valid pointer to it.
        std::memcpy(reinterpret_cast<char*>(&api) + entry.offset, &symbol, sizeof symbol);
    }
    return api;
}

Result<Handle> load_module(const Api& api, const std::string& path)
{
    const auto image = read_file(path);
    if (!image) return fail("cannot read module " + path + ": " + image.error().what);
    if (image->empty()) return fail("empty module " + path);
    Handle module = nullptr;
    if (const int error = api.hipModuleLoadData(&module, image->data()))
        return forward(api.check(error, ("load module " + path).c_str()).error());
    return module;
}

} // namespace dlsslop::hip
