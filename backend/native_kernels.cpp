// SPDX-License-Identifier: MIT
#include "native_kernels.hpp"

namespace dlsslop {

Result<void> NativeKernels::load(const std::string& path)
{
    module_ = DLSSLOP_TRY(hip::load_module(api, path));
    for (unsigned k = 0; k < kKernelCount; ++k)
        if (const int error = api.hipModuleGetFunction(&kernels_[k], module_, kNames[k])) {
            api.hipModuleUnload(module_);
            module_ = nullptr;
            return api.check(error, kNames[k]);
        }
    return {};
}

} // namespace dlsslop
