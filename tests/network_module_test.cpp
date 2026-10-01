// SPDX-License-Identifier: MIT
// The in-layer network's module through its C functions, as the layer loads
// it, without a GPU: a frame whose settings are out of range is rejected and
// says which, a missing model fails the network for good, naming the model's
// path and how to get it, and a frame of an extent the network does not take,
// by its working extent or by the device's storage buffers, is rejected,
// naming it, without a build, and so is another shape of that extent, which
// no network was built for to reshape. The model is the one dlsslopd's config
// file names, and a config file dlsslopd refuses fails the network. Takes the
// module's path.
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
// The model and the pipeline cache resolve under a directory of the test's own.
char directory[] = "/tmp/dlsslop-network-XXXXXX";

void remove_directory()
{
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

PFN_vkVoidFunction VKAPI_PTR no_functions(VkInstance, const char*) { return nullptr; }

// A device whose storage buffers hold 1 MiB, less than any extent's arena.
void VKAPI_PTR small_properties(VkPhysicalDevice, VkPhysicalDeviceProperties2* properties)
{
    properties->properties.limits.maxStorageBufferRange = 1 << 20;
    for (auto* s = static_cast<VkBaseOutStructure*>(properties->pNext); s; s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
            reinterpret_cast<VkPhysicalDeviceMaintenance3Properties*>(s)->maxMemoryAllocationSize = 1 << 20;
}
PFN_vkVoidFunction VKAPI_PTR small_storage(VkInstance, const char* name)
{
    return std::strcmp(name, "vkGetPhysicalDeviceProperties2") ? nullptr
                                                                 : reinterpret_cast<PFN_vkVoidFunction>(small_properties);
}

void require(bool value, const char* message)
{
    if (value) return;
    std::fprintf(stderr, "network-module: %s\n", message);
    std::exit(1);
}

// The first frame's failure, of a network opened while dlsslopd's config file
// CONFIG holds TEXT.
std::string failure_with(const dlssnr::NetworkModule& module, const ShmHeader* header, const std::string& config,
                         const std::string& text)
{
    std::FILE* file = std::fopen(config.c_str(), "w");
    require(file && std::fputs(text.c_str(), file) >= 0 && !std::fclose(file), "cannot write the config file");
    DlsslopNetworkDevice device{};
    device.physicalDispatch = no_functions;
    DlsslopNetwork* network = module.open(&device);
    require(network, "the module did not open");
    const bool failed = module.prepare(network, header, 1280, 720, 0) == kDlsslopNetworkFailed;
    std::string error = failed ? module.error(network) : "";
    module.close(network);
    return error;
}
}  // namespace

int main(int argc, char** argv)
{
    require(argc == 2, "usage: network-module-test MODULE");
    dlssnr::NetworkModule module;
    const bool loaded = module.Load(argv[1]);
    require(loaded, module.failure.c_str());
    require(mkdtemp(directory), "cannot make a temporary directory");
    std::atexit(remove_directory);
    setenv("XDG_DATA_HOME", directory, 1);
    setenv("XDG_CACHE_HOME", directory, 1);
    setenv("XDG_CONFIG_HOME", directory, 1);
    // No device: nothing here reaches Vulkan.
    DlsslopNetworkDevice device{};
    device.physicalDispatch = no_functions;
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

    const std::string config = std::string(directory) + "/dlsslop-amd/dlsslopd.conf";
    const std::string custom = std::string(directory) + "/custom.bin";
    std::error_code ignored;
    std::filesystem::create_directories(std::string(directory) + "/dlsslop-amd", ignored);
    std::FILE* file = std::fopen(model.c_str(), "w");
    require(file && !std::fclose(file), "cannot write a model file");
    network = module.open(&device);
    require(network, "the module did not open");
    for (uint32_t passes = 1; passes <= 2; ++passes) {
        header->passes.store(passes);
        require(module.prepare(network, header, 16, 16, 0) == kDlsslopNetworkRejected &&
                    std::strstr(module.error(network), "does not take 16x16 frames"),
                "a frame of an extent the network does not take, or another shape of it, was not rejected, naming it");
    }
    header->passes.store(1);
    module.close(network);
    device.physicalDispatch = small_storage;
    network = module.open(&device);
    require(network, "the module did not open");
    for (int frame = 0; frame < 2; ++frame)
        require(module.prepare(network, header, 1280, 720, 0) == kDlsslopNetworkRejected &&
                    std::strstr(module.error(network), "does not take 1280x720 frames") &&
                    std::strstr(module.error(network), "exceed the device's storage buffers of 1048576 bytes"),
                "a frame whose arena exceeds the device's storage buffers was not rejected, naming it");
    module.close(network);
    std::filesystem::remove(model, ignored);
    require(failure_with(module, header, config, "vulkan-model = " + custom + "\n").find("no model at " + custom) !=
                std::string::npos,
            "the network looked for another model than dlsslopd's config file names");
    require(failure_with(module, header, config, "vulkan-model = relative.bin\n").find(config) != std::string::npos,
            "a config file dlsslopd refuses did not fail the network, naming it");
    munmap(memory, kHeaderBytes);
    std::puts("network module: out-of-range settings rejected, a missing model fails for good, extents the "
              "network or the device's storage buffers do not take rejected in any shape, and the model is the one "
              "dlsslopd's config file names");
}
