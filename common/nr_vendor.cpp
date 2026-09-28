// SPDX-License-Identifier: MIT
#include "nr_vendor.h"

#include <exception>

namespace dlsslop::nr_vendor {
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
        return fail("unknown failure in the Vulkan network runtime");
    }
}
} // namespace

Result<std::unique_ptr<nr::Runtime>> build(const nr::HostDevice& host, const nr::RuntimeConfig& config,
                                           const nr::TemporalConfig& temporal)
{
    return caught([&] { return std::make_unique<nr::Runtime>(host, config, nr::ControlMaskConfig{}, temporal); });
}

Result<void> record(nr::Runtime& runtime, VkCommandBuffer cmd, const nr::ColourFrame& colour,
                    const nr::Controls& controls)
{
    return caught([&] { runtime.record(cmd, colour, controls); });
}

Result<void> record_temporal(nr::Runtime& runtime, VkCommandBuffer cmd, const nr::ColourFrame& colour,
                             const nr::Controls& controls, const nr::TemporalFrame& temporal)
{
    return caught([&] { runtime.record_temporal(cmd, colour, controls, temporal); });
}
} // namespace dlsslop::nr_vendor
