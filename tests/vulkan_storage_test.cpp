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
#include <tuple>
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
    dlssnr::DeviceFeatureRequest legacy;
    Check(legacy.Enable(info, false), "cannot enable legacy features");
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
    dlssnr::DeviceFeatureRequest chained;
    Check(chained.Enable(info, false), "cannot enable known Features2 chain");
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
    dlssnr::DeviceFeatureRequest unsupported;
    Check(!unsupported.Enable(info, false) && info.pNext == &unknown &&
          !core.features.shaderStorageImageWriteWithoutFormat, "unknown prefix was not rejected intact");
    core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    Check(unsupported.Enable(info, false) && info.pNext == &unknown, "already enabled unknown prefix was rejected");
    std::puts("device features: private legacy/Features2 copies and unknown-prefix fallback verified");
}

// A chain vkCreateDevice accepts: no structure twice, and no core
// VkPhysicalDeviceVulkan1xFeatures beside a structure it replaces.
static bool Valid(const VkDeviceCreateInfo& info) {
    std::vector<VkStructureType> seen;
    for (auto* node = static_cast<const VkBaseInStructure*>(info.pNext); node; node = node->pNext) {
        for (VkStructureType type : seen)
            if (type == node->sType) return false;
        seen.push_back(node->sType);
    }
    for (const auto& f : dlsslop::kNetworkFeatures)
        if (f.core != dlsslop::kNoCore && dlssnr::FindStructure(info.pNext, f.core) &&
            dlssnr::FindStructure(info.pNext, f.type))
            return false;
    return true;
}

// Every network feature set where the chain enables it.
static bool AllNetworkBits(const VkDeviceCreateInfo& info) {
    for (const auto& f : dlsslop::kNetworkFeatures) {
        const auto place = dlssnr::NetworkFeatureIn(info.pNext, f);
        if (!place.first || !dlsslop::FeatureBit(const_cast<VkBaseInStructure*>(place.first), place.second))
            return false;
    }
    return true;
}

// The network's extensions, in table order: explicit workgroup layout last.
static std::vector<const char*> NetworkExtensions() {
    std::vector<const char*> names;
    for (const auto& f : dlsslop::kNetworkFeatures)
        if (f.extension && (names.empty() || std::strcmp(names.back(), f.extension)))
            names.push_back(f.extension);
    return names;
}

static void NetworkFeatures() {
    // A DXVK-like chain: the core structures, one bit already on, an unknown suffix.
    VkBaseOutStructure unknown{};
    unknown.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &unknown};
    v13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    v12.vulkanMemoryModel = VK_TRUE;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    core.features.robustBufferAccess = VK_TRUE;
    VkLayerDeviceCreateInfo loader{};
    loader.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO;
    loader.function = VK_LAYER_LINK_INFO;
    loader.pNext = &core;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &loader};
    const auto chain = [&] { return std::vector<unsigned char>(reinterpret_cast<unsigned char*>(&loader),
                                                                reinterpret_cast<unsigned char*>(&loader + 1)); };
    const auto before = std::make_tuple(core.features.shaderStorageImageWriteWithoutFormat, v11.storageBuffer16BitAccess,
                                        v12.shaderInt8, v13.subgroupSizeControl);
    const auto loaderBytes = chain();
    dlssnr::DeviceFeatureRequest request;
    Check(request.Enable(info, true), "cannot add the network to a core-structure chain");
    Check(Valid(info) && AllNetworkBits(info) && dlssnr::HasFormatlessStorageWrites(info),
          "network features missing from a core-structure chain");
    Check(static_cast<const void*>(dlssnr::FindStructure(info.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO)) == &unknown,
          "the unknown suffix was not left shared");
    Check(!dlssnr::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES),
          "a core feature was given its own structure beside Vulkan12Features");
    Check(dlssnr::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR) &&
          dlssnr::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR),
          "extension structures were not added");
    const auto* copied13 = reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(
        dlssnr::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES));
    const auto* copied2 = dlssnr::CoreFeatures2(info);
    Check(copied13 != &v13 && copied13->dynamicRendering && copied2 != &core && copied2->features.robustBufferAccess,
          "the game's own settings were lost in the copies");
    Check(before == std::make_tuple(core.features.shaderStorageImageWriteWithoutFormat, v11.storageBuffer16BitAccess,
                                    v12.shaderInt8, v13.subgroupSizeControl) &&
          loaderBytes == chain() && v12.vulkanMemoryModel && core.pNext == &v11 && v13.pNext == &unknown,
          "the game's chain was written");

    // The ledger reads what vkCreateDevice accepted: the extensions too.
    Check(!dlssnr::NetworkEnabled(info), "ledger ignored the missing extensions");
    const auto extensions = NetworkExtensions();
    info.enabledExtensionCount = uint32_t(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    Check(dlssnr::NetworkEnabled(info), "ledger missed an enabled network");
    // vulkan-nr's adopt() requires explicit workgroup layout too.
    --info.enabledExtensionCount;
    Check(!dlssnr::NetworkEnabled(info), "ledger ignored the missing explicit workgroup layout");

    // Structures of their own: one bit set in the game's, the rest added, none twice.
    VkPhysicalDeviceShaderFloat16Int8Features float16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    float16.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceFeatures legacy{};
    VkDeviceCreateInfo own{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &float16};
    own.pEnabledFeatures = &legacy;
    dlssnr::DeviceFeatureRequest separate;
    Check(separate.Enable(own, true) && Valid(own) && AllNetworkBits(own) &&
          dlssnr::HasFormatlessStorageWrites(own) && own.pEnabledFeatures != &legacy,
          "cannot add the network beside the game's own structures");
    Check(!float16.shaderInt8 && !legacy.shaderStorageImageWriteWithoutFormat &&
          static_cast<const void*>(dlssnr::FindStructure(own.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)) !=
              &float16,
          "the game's structures were written");

    // A structure to change behind one that cannot be copied declines, untouched.
    VkPhysicalDeviceVulkan12Features late{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkBaseOutStructure opaque{VK_STRUCTURE_TYPE_APPLICATION_INFO, reinterpret_cast<VkBaseOutStructure*>(&late)};
    VkDeviceCreateInfo hidden{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &opaque};
    dlssnr::DeviceFeatureRequest declined;
    Check(!declined.Enable(hidden, true) && hidden.pNext == &opaque && !late.shaderInt8 &&
          !hidden.pEnabledFeatures, "an uncopyable prefix was not declined intact");
    // With nothing to change behind it, the network's own structures go in front.
    late.storageBuffer8BitAccess = late.shaderFloat16 = late.shaderInt8 = late.vulkanMemoryModel = VK_TRUE;
    dlssnr::DeviceFeatureRequest ahead;
    Check(ahead.Enable(hidden, true) && Valid(hidden) && AllNetworkBits(hidden) &&
          static_cast<const void*>(dlssnr::FindStructure(hidden.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO)) == &opaque,
          "the network's structures were not put ahead of an uncopyable chain");
    std::puts("device features: network features set in the game's structures or added, never twice, "
              "game chains unwritten, uncopyable prefixes declined, and the ledger");
}

int main() {
    Features();
    NetworkFeatures();
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
