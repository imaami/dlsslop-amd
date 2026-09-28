// SPDX-License-Identifier: MIT
// The in-layer network's module through its C functions, as the layer loads
// it, without a GPU: a frame whose settings are out of range is rejected and
// says which, and a missing model fails the network for good, naming the
// model's path and how to get it. Takes the module's path.
#include "network_module.h"
#include "shm_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

#include <sys/mman.h>

namespace {
void require(bool value, const char* message)
{
    if (value) return;
    std::fprintf(stderr, "network-module: %s\n", message);
    std::exit(1);
}
}  // namespace

int main(int argc, char** argv)
{
    require(argc == 2, "usage: network-module-test MODULE");
    // The model and the pipeline cache under a directory of the test's own.
    char directory[] = "/tmp/dlsslop-network-XXXXXX";
    require(mkdtemp(directory), "cannot make a temporary directory");
    setenv("XDG_DATA_HOME", directory, 1);
    setenv("XDG_CACHE_HOME", directory, 1);
    dlssnr::NetworkModule module;
    require(module.Load(argv[1]), module.failure.c_str());
    // No device: nothing here reaches Vulkan.
    DlsslopNetworkDevice device{};
    DlsslopNetwork* network = module.open(&device);
    require(network, "the module did not open");
    void* memory = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(memory != MAP_FAILED, "cannot map a channel header");
    auto* header = static_cast<ShmHeader*>(memory);
    ShmInitNativeDefaults(header);

    header->style.store(3);
    require(module.prepare(network, header, 1280, 720, 0) == kDlsslopNetworkRejected &&
                std::strstr(module.error(network), "style 0..2"),
            "a setting out of range was not rejected, naming it");
    header->style.store(0);
    const std::string model = std::string(directory) + "/dlsslop-amd/dlssnr.bin";
    for (int frame = 0; frame < 2; ++frame)
        require(module.prepare(network, header, 1280, 720, 0) == kDlsslopNetworkFailed &&
                    std::strstr(module.error(network), model.c_str()) &&
                    std::strstr(module.error(network), "dlsslop-setup --dll"),
                "a missing model did not fail the network for good, naming it");
    module.close(network);
    munmap(memory, kHeaderBytes);
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::puts("network module: out-of-range settings rejected, a missing model fails for good");
}
