// The boundary to the vendored lmxxf network, which throws: hip_vendor.cpp,
// the only file that includes the network, catches everything it throws. It is
// built with exceptions and as C++20, which gives the vendored code a smaller
// binary than C++23, so this header needs neither. Each function returns why
// it failed, or nothing.
// SPDX-License-Identifier: MIT
#pragma once
#include <memory>
#include <string>

namespace dlsslop::hip {

// hip_reference::Network, opaque outside hip_vendor.cpp.
class Network;
struct NetworkDeleter {
    void operator()(Network* network) const noexcept;
};
using NetworkHandle = std::unique_ptr<Network, NetworkDeleter>;

struct NetworkOptions {
    unsigned width, height, device;
    std::string modules, assets;
    bool performance; // Skip upstream's performance-preset blocks 42, 43 and 46.
};
void* stream(Network& network);

namespace vendor {
std::string network(const NetworkOptions& options, NetworkHandle& network);
std::string enqueue(Network& network, void* rgba, void* history, void* rgb);
std::string synchronize(Network& network);
std::string print_memory(Network& network);
} // namespace vendor

} // namespace dlsslop::hip
