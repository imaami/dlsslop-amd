// dlsslopd's command line and config file.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.hpp"
#include "shm_protocol.hpp"

#include <cstdio>
#include <optional>
#include <string>

namespace dlsslop {
struct Options {
    std::string assets, modules, shm = ShmNativeChannelPath();
    std::string input, output, trace_dir;
    std::string backend = "auto", vulkan_model;
    unsigned width = 0, height = 0, self_test_runs = 10, self_test_drops = 0;
    unsigned idle_exit = 0;
    // Unset, a serving worker keeps the channel's live values across restarts.
    std::optional<unsigned> tier, passes;
    int device = -1;
    bool diagnose = false, test_identity = false, once = false, self_test = false;
    bool cpu_codec = false, performance = false;
};

// The first option set that only the HIP network serves, or null: auto takes HIP
// for it and the Vulkan backend refuses it.
const char* hip_only(const Options& o);
Options default_options();
// The Vulkan model dlsslopd loads without options: vulkan-model from its
// default config file, else the default path.
Result<std::string> configured_vulkan_model();
void usage(FILE* out);
Result<Options> parse(int argc, char** argv);
} // namespace dlsslop
