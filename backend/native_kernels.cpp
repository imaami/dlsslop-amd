// SPDX-License-Identifier: MIT
#include "native_kernels.h"
#include "files.h"

namespace dlsslop {

Result<void> NativeKernels::load(const std::string& path)
{
    const auto image = read_file(path);
    if (!image) return fail("module file missing: " + path);
    if (image->empty()) return fail("empty module");
    DLSSLOP_TRY(api.check(api.hipModuleLoadData(&module_, image->data()), "load native module"));
    for (unsigned k = 0; k < kKernelCount; ++k)
        if (const int error = api.hipModuleGetFunction(&kernels_[k], module_, kNames[k])) {
            api.hipModuleUnload(module_);
            module_ = nullptr;
            return api.check(error, kNames[k]);
        }
    return {};
}

} // namespace dlsslop
