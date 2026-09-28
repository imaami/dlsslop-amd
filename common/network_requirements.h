// What DLSSNR-AMD's Vulkan network needs of a device: one table for dlsslopd's
// own device and for a game's device the layer adds it to. C++17, like the
// layer that includes it.
// SPDX-License-Identifier: MIT
#pragma once
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dlsslop {

// A feature bit the network's shaders use (vulkan-nr/src/core/nrvk.hpp,
// Context::create): the structure that carries it alone, and the core
// VkPhysicalDeviceVulkan1xFeatures structure that carries it too, if any.
struct NetworkFeature {
    const char* name;
    VkStructureType type;
    uint32_t offset;
    VkStructureType core;
    uint32_t core_offset;
    // The extension that provides it; null for core Vulkan 1.3.
    const char* extension;
    // The network runs without it, more slowly: explicit workgroup layout.
    bool optional;
};
constexpr VkStructureType kNoCore = VK_STRUCTURE_TYPE_MAX_ENUM;

#define DLSSLOP_ALONE(Struct, type, field) VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##type, offsetof(Struct, field)
#define DLSSLOP_CORE(digits, version, field)                                                                           \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_##version##_FEATURES, offsetof(VkPhysicalDeviceVulkan##digits##Features, field)
constexpr NetworkFeature kNetworkFeatures[] = {
    {"cooperativeMatrix", DLSSLOP_ALONE(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, COOPERATIVE_MATRIX_FEATURES_KHR,
                                        cooperativeMatrix), kNoCore, 0, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, false},
    {"shaderFloat8", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT, shaderFloat8),
     kNoCore, 0, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, false},
    {"shaderFloat8CooperativeMatrix", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT,
                                                    shaderFloat8CooperativeMatrix),
     kNoCore, 0, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, false},
    {"storageBuffer16BitAccess", DLSSLOP_ALONE(VkPhysicalDevice16BitStorageFeatures, 16BIT_STORAGE_FEATURES,
                                               storageBuffer16BitAccess),
     DLSSLOP_CORE(11, 1_1, storageBuffer16BitAccess), nullptr, false},
    {"storageBuffer8BitAccess", DLSSLOP_ALONE(VkPhysicalDevice8BitStorageFeatures, 8BIT_STORAGE_FEATURES,
                                              storageBuffer8BitAccess),
     DLSSLOP_CORE(12, 1_2, storageBuffer8BitAccess), nullptr, false},
    {"shaderFloat16", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES, shaderFloat16),
     DLSSLOP_CORE(12, 1_2, shaderFloat16), nullptr, false},
    {"shaderInt8", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES, shaderInt8),
     DLSSLOP_CORE(12, 1_2, shaderInt8), nullptr, false},
    {"vulkanMemoryModel", DLSSLOP_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures, VULKAN_MEMORY_MODEL_FEATURES,
                                        vulkanMemoryModel),
     DLSSLOP_CORE(12, 1_2, vulkanMemoryModel), nullptr, false},
    {"subgroupSizeControl", DLSSLOP_ALONE(VkPhysicalDeviceSubgroupSizeControlFeatures, SUBGROUP_SIZE_CONTROL_FEATURES,
                                          subgroupSizeControl),
     DLSSLOP_CORE(13, 1_3, subgroupSizeControl), nullptr, false},
    {"synchronization2", DLSSLOP_ALONE(VkPhysicalDeviceSynchronization2Features, SYNCHRONIZATION_2_FEATURES,
                                       synchronization2),
     DLSSLOP_CORE(13, 1_3, synchronization2), nullptr, false},
    {"workgroupMemoryExplicitLayout",
     DLSSLOP_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
                   workgroupMemoryExplicitLayout),
     kNoCore, 0, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, true},
    {"workgroupMemoryExplicitLayout8BitAccess",
     DLSSLOP_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
                   workgroupMemoryExplicitLayout8BitAccess),
     kNoCore, 0, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, true},
    {"workgroupMemoryExplicitLayout16BitAccess",
     DLSSLOP_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
                   workgroupMemoryExplicitLayout16BitAccess),
     kNoCore, 0, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, true},
};
#undef DLSSLOP_ALONE
#undef DLSSLOP_CORE

// The bit at OFFSET in STRUCTURE.
inline VkBool32& FeatureBit(void* structure, uint32_t offset)
{
    return *reinterpret_cast<VkBool32*>(static_cast<unsigned char*>(structure) + offset);
}

// One structure of each type the table names, zeroed and chained behind a
// VkPhysicalDeviceFeatures2: for a support query or a vkCreateDevice. With
// optional false, the optional features' structure stays out of the chain.
class NetworkFeatureChain {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDevice16BitStorageFeatures storage16_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDevice8BitStorageFeatures storage8_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features float16_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceSynchronization2Features sync2_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR layout_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 head_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void* const structures_[9] = {&coop_, &fp8_, &storage16_, &storage8_, &float16_, &memory_, &subgroup_, &sync2_,
                                  &layout_};

public:
    explicit NetworkFeatureChain(bool optional)
    {
        void* next = optional ? &layout_ : nullptr;
        for (unsigned i = 8; i-- > 0;) {
            static_cast<VkBaseOutStructure*>(structures_[i])->pNext = static_cast<VkBaseOutStructure*>(next);
            next = structures_[i];
        }
        head_.pNext = next;
    }
    NetworkFeatureChain(const NetworkFeatureChain&) = delete;
    VkPhysicalDeviceFeatures2& features2() { return head_; }
    // The structure of TYPE in this chain.
    void* structure(VkStructureType type) const
    {
        for (void* s : structures_)
            if (static_cast<const VkBaseOutStructure*>(s)->sType == type) return s;
        return nullptr;
    }
    VkBool32& bit(const NetworkFeature& f) const { return FeatureBit(structure(f.type), f.offset); }
};

// What stops the network running on PHYSICAL, or null: a feature's or an
// extension's name, "Vulkan 1.3" or "32-lane compute subgroups". LAYOUT says
// whether the optional features are there too. The functions are the caller's
// route to the device: the loader's, or the next layer's.
inline const char* NetworkUnsupported(VkPhysicalDevice physical, PFN_vkGetPhysicalDeviceProperties2 properties2,
                                      PFN_vkGetPhysicalDeviceFeatures2 features2,
                                      PFN_vkEnumerateDeviceExtensionProperties extensions, bool& layout)
{
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
    properties2(physical, &properties);
    if (properties.properties.apiVersion < VK_API_VERSION_1_3) return "Vulkan 1.3";
    uint32_t count = 0;
    extensions(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> offered(count);
    extensions(physical, nullptr, &count, offered.data());
    const auto has = [&offered](const char* name) {
        for (const auto& e : offered)
            if (!std::strcmp(e.extensionName, name)) return true;
        return false;
    };
    layout = has(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    for (const auto& f : kNetworkFeatures)
        if (!f.optional && f.extension && !has(f.extension)) return f.extension;
    NetworkFeatureChain supported(layout);
    features2(physical, &supported.features2());
    for (const auto& f : kNetworkFeatures) {
        if (supported.bit(f)) continue;
        if (!f.optional) return f.name;
        layout = false;
    }
    // The cooperative-matrix fragments are laid out for 32-lane subgroups.
    if (subgroup.minSubgroupSize > 32 || subgroup.maxSubgroupSize < 32 ||
        !(subgroup.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
        return "32-lane compute subgroups";
    return nullptr;
}

} // namespace dlsslop
