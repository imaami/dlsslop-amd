// SPDX-License-Identifier: MIT
#include "offline.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace dlsslop {
void run_offline(const Options& o, Backend& engine)
{
    const unsigned passes = std::min(o.passes.value_or(kNativeDefaultPasses), engine.max_passes);
    const size_t bytes = size_t(o.width) * o.height * 4;
    std::ifstream in(o.input, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(bytes))
        throw std::runtime_error("offline input size must equal width * height * 4");
    std::vector<uint8_t> pixels(bytes), result(bytes);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("read offline input");
    engine.infer({pixels.data(), result.data()}, o.width, o.height, passes);
    std::ofstream out(o.output, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("write offline output");
    std::fprintf(stderr, "offline: passes=%u upload=%.2f ms, network=%.2f ms, readback=%.2f ms\n",
                 passes, engine.upload_ms, engine.inference_ms, engine.readback_ms);
}
} // namespace dlsslop
