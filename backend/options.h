// dlsslopd's command line and config file.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"
#include "shm_protocol.h"

#include <cstdio>
#include <optional>
#include <string>

namespace dlsslop {
struct Options {
    std::string assets, modules, shm = ShmNativeChannelPath();
    std::string input, output, trace_dir;
    std::string backend = "auto", vulkan_model;
    unsigned width = 0, height = 0, self_test_runs = 10;
    unsigned idle_exit = 0;
    // Unset, a serving worker keeps the channel's live values across restarts.
    std::optional<unsigned> tier, passes;
    int device = -1;
    bool diagnose = false, test_identity = false, once = false, self_test = false;
    bool cpu_compose = false, cpu_codec = false, performance = false;
};

// The first option set that only the HIP network serves, or null: auto takes HIP
// for it and the Vulkan backend refuses it.
const char* hip_only(const Options& o);
Options default_options();
void usage(FILE* out);
Result<Options> parse(int argc, char** argv);
} // namespace dlsslop
