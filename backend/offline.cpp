// SPDX-License-Identifier: MIT
#include "offline.h"
#include "files.h"

#include <cstdio>
#include <vector>

namespace dlsslop {
Result<void> run_offline(const Options& o, Backend& engine)
{
    const unsigned passes = std::min(o.passes.value_or(kNativeDefaultPasses), engine.max_passes);
    const size_t bytes = size_t(o.width) * o.height * 4;
    const auto input = read_file(o.input);
    if (!input || input->size() != bytes) return fail("offline input size must equal width * height * 4");
    std::vector<uint8_t> result(bytes);
    DLSSLOP_TRY(engine.infer({reinterpret_cast<const uint8_t*>(input->data()), result.data()}, o.width, o.height, passes));
    if (!write_file(o.output, {reinterpret_cast<const char*>(result.data()), result.size()}))
        return fail("write offline output");
    std::fprintf(stderr, "offline: passes=%u upload=%.2f ms, network=%.2f ms, readback=%.2f ms\n",
                 passes, engine.upload_ms, engine.inference_ms, engine.readback_ms);
    return {};
}
} // namespace dlsslop
