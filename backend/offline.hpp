// Offline mode: one RGBA8 image from a file to a file.
// SPDX-License-Identifier: MIT
#pragma once
#include "engine.hpp"
#include "files.hpp"

#include <cstdio>
#include <vector>

namespace dlsslop {
template <Engine E>
Result<void> run_offline(const struct options& o, E& engine)
{
    const unsigned passes = std::min(o.passes ? o.passes : kNativeDefaultPasses, E::max_passes);
    const size_t bytes = size_t(o.width) * o.height * 4;
    const auto input = read_file(std::string(o.input, o.input_length));
    if (!input || input->size() != bytes) return fail("offline input size must equal width * height * 4");
    std::vector<uint8_t> result(bytes);
    DLSSLOP_TRY(engine.infer({reinterpret_cast<const uint8_t*>(input->data()), result.data()}, o.width, o.height, passes));
    const std::string output(o.output, o.output_length);
    if (!write_file(output, {reinterpret_cast<const char*>(result.data()), result.size()}))
        return fail("write offline output");
    std::fprintf(stderr, "offline: passes=%u upload=%.2f ms, network=%.2f ms, readback=%.2f ms\n",
                 passes, engine.upload_ms, engine.inference_ms, engine.readback_ms);
    return {};
}
} // namespace dlsslop
