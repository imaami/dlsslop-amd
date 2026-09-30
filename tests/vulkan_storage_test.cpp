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
#include <string>
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
    for (const void* node = info.pNext; node; node = dlsslop::NextStructure(node)) {
        for (VkStructureType type : seen)
            if (type == dlsslop::StructureType(node)) return false;
        seen.push_back(dlsslop::StructureType(node));
    }
    for (const auto& f : dlsslop::kNetworkFeatures)
        if (f.core != dlsslop::kNoCore && dlsslop::FindStructure(info.pNext, f.core) &&
            dlsslop::FindStructure(info.pNext, f.type))
            return false;
    return true;
}

// Every network feature set where the chain enables it.
static bool AllNetworkBits(const VkDeviceCreateInfo& info) {
    for (const auto& f : dlsslop::kNetworkFeatures) {
        const auto place = dlssnr::NetworkFeatureIn(info.pNext, f);
        if (!place.first || !dlsslop::FeatureBit(place.first, place.second))
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
    Check(dlsslop::FindStructure(info.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO) == &unknown,
          "the unknown suffix was not left shared");
    Check(!dlsslop::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES),
          "a core feature was given its own structure beside Vulkan12Features");
    Check(dlsslop::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR) &&
          dlsslop::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR),
          "extension structures were not added");
    const auto* copied13 = static_cast<const VkPhysicalDeviceVulkan13Features*>(
        dlsslop::FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES));
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
    // The network requires explicit workgroup layout too (network_requirements.h).
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
          dlsslop::FindStructure(own.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) !=
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
    for (const auto& f : dlsslop::kNetworkFeatures)
        if (f.core == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) dlsslop::FeatureBit(&late, f.core_offset) = VK_TRUE;
    dlssnr::DeviceFeatureRequest ahead;
    Check(ahead.Enable(hidden, true) && Valid(hidden) && AllNetworkBits(hidden) &&
          dlsslop::FindStructure(hidden.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO) == &opaque,
          "the network's structures were not put ahead of an uncopyable chain");
    std::puts("device features: network features set in the game's structures or added, never twice, "
              "game chains unwritten, uncopyable prefixes declined, and the ledger");
}

// A device as NetworkUnsupported sees it through the next layer's functions.
namespace stub {
uint32_t api = VK_API_VERSION_1_3;
uint32_t minimumSubgroup = 32, maximumSubgroup = 64;
std::vector<std::string> extensions;
const char* unsupported = nullptr;  // a feature the device lacks

void VKAPI_CALL Properties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* properties) {
    properties->properties.apiVersion = api;
    auto* subgroup = static_cast<VkPhysicalDeviceSubgroupSizeControlProperties*>(
        dlsslop::FindStructure(properties->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES));
    if (!subgroup) return;
    subgroup->minSubgroupSize = minimumSubgroup;
    subgroup->maxSubgroupSize = maximumSubgroup;
    subgroup->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
}

void VKAPI_CALL Features2(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    for (const auto& f : dlsslop::kNetworkFeatures)
        if (void* node = dlsslop::FindStructure(features->pNext, f.type))
            dlsslop::FeatureBit(node, f.offset) = !unsupported || std::strcmp(unsupported, f.name);
}

VkResult VKAPI_CALL Extensions(VkPhysicalDevice, const char*, uint32_t* count, VkExtensionProperties* properties) {
    if (properties)
        for (uint32_t i = 0; i < *count && i < extensions.size(); ++i)
            std::snprintf(properties[i].extensionName, sizeof properties[i].extensionName, "%s", extensions[i].c_str());
    *count = uint32_t(extensions.size());
    return VK_SUCCESS;
}

const char* Unavailable(uint32_t instance) {
    return dlssnr::NetworkUnavailable(VK_NULL_HANDLE, instance, Properties2, Features2, Extensions);
}
}  // namespace stub

static bool Same(const char* a, const char* b) { return a && b && !std::strcmp(a, b); }

static void NetworkAvailability() {
    const auto all = NetworkExtensions();
    stub::extensions.assign(all.begin(), all.end());
    Check(!stub::Unavailable(VK_API_VERSION_1_3), "a capable device was refused the network");
    Check(Same(stub::Unavailable(VK_API_VERSION_1_2), "a Vulkan 1.3 instance") &&
          Same(dlssnr::NetworkUnavailable(VK_NULL_HANDLE, VK_API_VERSION_1_3, nullptr, stub::Features2,
                                          stub::Extensions), "a Vulkan 1.3 instance"),
          "a Vulkan 1.2 instance, or one without the 1.1 queries, was given the network");
    stub::api = VK_API_VERSION_1_2;
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "Vulkan 1.3"), "a Vulkan 1.2 device was given the network");
    stub::api = VK_API_VERSION_1_3;
    for (const auto& f : dlsslop::kNetworkFeatures) {
        stub::unsupported = f.name;
        Check(Same(stub::Unavailable(VK_API_VERSION_1_3), f.name), "a missing feature was not named");
    }
    stub::unsupported = nullptr;
    for (size_t i = 0; i < all.size(); ++i) {
        stub::extensions.assign(all.begin(), all.end());
        stub::extensions.erase(stub::extensions.begin() + i);
        Check(Same(stub::Unavailable(VK_API_VERSION_1_3), all[i]), "a missing extension was not named");
    }
    stub::extensions.assign(all.begin(), all.end());
    Check(Same(dlsslop::NetworkUnsupported(VK_NULL_HANDLE, stub::Properties2, stub::Features2, stub::Extensions,
                                           VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
               VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
          "the extension dlsslopd needs besides was not named");
    stub::minimumSubgroup = 64;
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "32-lane compute subgroups"),
          "a device without 32-lane subgroups was given the network");
    stub::minimumSubgroup = 32;

    // The network's extensions join the list once each, behind the game's.
    const char* game[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.enabledExtensionCount = 2;
    info.ppEnabledExtensionNames = game;
    std::vector<const char*> list;
    Check(dlssnr::AddNetworkExtensions(info, list) && info.ppEnabledExtensionNames == list.data() &&
          info.enabledExtensionCount == list.size() && list.size() == 1 + all.size() && list[0] == game[0] &&
          list[1] == game[1], "the network's extensions were not added once each behind the game's");
    Check(!dlssnr::AddNetworkExtensions(info, list) && list.size() == 1 + all.size(),
          "extensions already listed were added again");

    // A request that declined the network still enables formatless storage writes.
    VkPhysicalDeviceVulkan12Features late{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkBaseOutStructure opaque{VK_STRUCTURE_TYPE_APPLICATION_INFO, reinterpret_cast<VkBaseOutStructure*>(&late)};
    VkPhysicalDeviceFeatures legacy{};
    VkDeviceCreateInfo hidden{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &opaque};
    hidden.pEnabledFeatures = &legacy;
    dlssnr::DeviceFeatureRequest request;
    Check(!request.Enable(hidden, true) && request.Enable(hidden, false) &&
          dlssnr::HasFormatlessStorageWrites(hidden) && hidden.pNext == &opaque && !late.shaderInt8,
          "a request that declined the network did not fall back to formatless storage alone");
    std::puts("device features: the network is refused for each missing requirement, its extensions are added "
              "once, and a declined request falls back");
}

int main() {
    Features();
    NetworkFeatures();
    NetworkAvailability();
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
