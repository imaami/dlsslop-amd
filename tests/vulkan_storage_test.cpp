#include "storage_shader.h"
#include "device_features.h"
#include "dlssnr/DlssNr_Shader_Vk.h"
#include "scaling/bcus_Shader_Vk.h"
#include "scaling/bcds_bicubic_Shader_Vk.h"
#include "scaling/bcds_catmull_Shader_Vk.h"
#include "scaling/bcds_lanczos2_Shader_Vk.h"
#include "scaling/bcds_lanczos3_Shader_Vk.h"
#include "scaling/bcds_kaiser2_Shader_Vk.h"
#include "scaling/bcds_kaiser3_Shader_Vk.h"
#include "scaling/bcds_magc_Shader_Vk.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static void Check(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static std::vector<char> Bytes(const std::vector<uint32_t>& words) {
    std::vector<char> bytes(words.size() * sizeof(uint32_t));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

template<size_t N> static void Shader(const unsigned char (&blob)[N], const char* name,
                                     const char* outputDirectory) {
    std::vector<char> source(blob, blob + N);
    std::vector<uint32_t> fixed;
    std::string reason;
    Check(dlssnr::FormatlessStorageShader(source, fixed, reason), reason.c_str());
    unsigned capabilities = 0, storage = 0;
    for (size_t i = 5; i < fixed.size(); i += fixed[i] >> 16) {
        const uint32_t op = fixed[i] & 0xffffu;
        if (op == 17 && fixed[i + 1] == 56) ++capabilities;
        if (op == 25 && fixed[i + 7] == 2) {
            Check(fixed[i + 8] == 0, "typed storage image remains");
            ++storage;
        }
    }
    Check(capabilities == 1 && storage != 0, "missing formatless storage capability/type");
    std::vector<uint32_t> again;
    Check(dlssnr::FormatlessStorageShader(Bytes(fixed), again, reason), "idempotent transform rejected");
    Check(again == fixed, "formatless transform is not idempotent");
    for (uint32_t forbidden : {98u, 60u, 320u}) {
        auto unsupported = fixed;
        unsupported.push_back((1u << 16) | forbidden);
        Check(!dlssnr::FormatlessStorageShader(Bytes(unsupported), again, reason),
              "storage read/atomic was accepted");
        Check(again.empty(), "rejected module was left usable");
    }
    auto malformed = fixed;
    malformed.push_back(0);
    Check(!dlssnr::FormatlessStorageShader(Bytes(malformed), again, reason), "zero word count accepted");
    malformed.back() = 3u << 16;
    Check(!dlssnr::FormatlessStorageShader(Bytes(malformed), again, reason), "truncated instruction accepted");
    source.pop_back();
    Check(!dlssnr::FormatlessStorageShader(source, again, reason), "unaligned module accepted");
    if (outputDirectory) {
        const std::string path = std::string(outputDirectory) + '/' + name + ".spv";
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(fixed.data()), fixed.size() * sizeof(uint32_t));
        Check(bool(output), "cannot write transformed shader for external validation");
    }
    std::printf("%s: write-only formatless storage verified\n", name);
}

static void Features() {
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    VkPhysicalDeviceFeatures original{};
    original.robustBufferAccess = VK_TRUE;
    info.pEnabledFeatures = &original;
    dlssnr::FormatlessStorageFeatures legacy;
    Check(legacy.Enable(info), "cannot enable legacy features");
    Check(info.pEnabledFeatures != &original && info.pEnabledFeatures->robustBufferAccess &&
          dlssnr::HasFormatlessStorageWrites(info) && !original.shaderStorageImageWriteWithoutFormat,
          "legacy features were not preserved/copied");

    VkBaseOutStructure unknown{};
    unknown.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    VkPhysicalDeviceFeatures2 core{};
    core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    core.features.robustBufferAccess = VK_TRUE;
    core.pNext = &unknown;
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.pNext = &core;
    vulkan12.timelineSemaphore = VK_TRUE;
    VkLayerDeviceCreateInfo loader{};
    loader.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO;
    loader.function = VK_LOADER_DATA_CALLBACK;
    loader.pNext = &vulkan12;
    info.pNext = &loader;
    info.pEnabledFeatures = nullptr;
    dlssnr::FormatlessStorageFeatures chained;
    Check(chained.Enable(info), "cannot enable known Features2 chain");
    auto* first = static_cast<const VkLayerDeviceCreateInfo*>(info.pNext);
    auto* second = static_cast<const VkPhysicalDeviceVulkan12Features*>(first->pNext);
    auto* third = static_cast<const VkPhysicalDeviceFeatures2*>(second->pNext);
    Check(first != &loader && second != &vulkan12 && third != &core && third->pNext == &unknown,
          "incorrect Features2 prefix/suffix ownership");
    Check(first->function == loader.function && second->timelineSemaphore &&
          third->features.robustBufferAccess && dlssnr::HasFormatlessStorageWrites(info),
          "caller settings lost in prefix copy");
    Check(loader.pNext == &vulkan12 && vulkan12.pNext == &core && core.pNext == &unknown &&
          !core.features.shaderStorageImageWriteWithoutFormat, "caller chain was mutated");

    unknown.pNext = reinterpret_cast<VkBaseOutStructure*>(&core);
    core.pNext = nullptr;
    info.pNext = &unknown;
    dlssnr::FormatlessStorageFeatures unsupported;
    Check(!unsupported.Enable(info) && info.pNext == &unknown &&
          !core.features.shaderStorageImageWriteWithoutFormat, "unknown prefix was not rejected intact");
    core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    Check(unsupported.Enable(info) && info.pNext == &unknown, "already enabled unknown prefix was rejected");
    std::puts("device features: private legacy/Features2 copies and unknown-prefix fallback verified");
}

int main(int argc, char** argv) {
    const char* output = argc > 1 ? argv[1] : nullptr;
    Features();
#define SHADER(name) Shader(name##_spv, #name, output)
    SHADER(dlssnr);
    SHADER(bcus);
    SHADER(bcds_bicubic);
    SHADER(bcds_catmull);
    SHADER(bcds_lanczos2);
    SHADER(bcds_lanczos3);
    SHADER(bcds_kaiser2);
    SHADER(bcds_kaiser3);
    SHADER(bcds_magc);
#undef SHADER
}
