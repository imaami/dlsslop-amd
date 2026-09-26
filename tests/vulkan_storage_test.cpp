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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void Check(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

// The layer loads these modules as they are built and enables only
// shaderStorageImageWriteWithoutFormat. Every storage image must be a
// formatless 2D image that is only written: a storage read or atomic would
// need shaderStorageImageReadWithoutFormat or a typed format too.
template<size_t N> static void Shader(const unsigned char (&blob)[N], const char* name) {
    constexpr uint32_t kOpCapability = 17, kOpTypeImage = 25, kOpImageTexelPointer = 60;
    constexpr uint32_t kOpImageRead = 98, kOpImageWrite = 99, kOpImageSparseRead = 320;
    constexpr uint32_t kReadWithoutFormat = 55, kWriteWithoutFormat = 56;
    Check(N >= 5 * sizeof(uint32_t) && N % sizeof(uint32_t) == 0, "invalid SPIR-V byte count");
    std::vector<uint32_t> words(N / sizeof(uint32_t));
    std::memcpy(words.data(), blob, N);
    Check(words[0] == 0x07230203 && words[4] == 0, "invalid SPIR-V header");
    unsigned writeCapabilities = 0, storage = 0, writes = 0;
    for (size_t i = 5; i < words.size(); i += words[i] >> 16) {
        const uint32_t count = words[i] >> 16, op = words[i] & 0xffffu;
        Check(count && count <= words.size() - i, "truncated SPIR-V instruction");
        if (op == kOpCapability) {
            Check(words[i + 1] != kReadWithoutFormat, "storage reads without format are not enabled");
            writeCapabilities += words[i + 1] == kWriteWithoutFormat;
        } else if (op == kOpTypeImage) {
            Check(count >= 9, "invalid OpTypeImage");
            if (words[i + 7] != 2) continue;
            // Dim2D, depth absent or unspecified, not arrayed or multisampled.
            Check(words[i + 3] == 1 && words[i + 4] != 1 && words[i + 4] <= 2 && !words[i + 5] &&
                  !words[i + 6], "unexpected storage image type");
            Check(words[i + 8] == 0, "typed storage image");
            ++storage;
        } else {
            Check(op != kOpImageRead && op != kOpImageSparseRead && op != kOpImageTexelPointer,
                  "storage read or atomic");
            writes += op == kOpImageWrite;
        }
    }
    Check(writeCapabilities == 1, "missing StorageImageWriteWithoutFormat capability");
    Check(storage && writes, "expected a write-only storage image shader");
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

int main() {
    Features();
#define SHADER(name) Shader(name##_spv, #name)
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
