// What DLSSNR-AMD's Vulkan network needs of a device: one table for dlsslopd's
// own device and for a game's device the layer adds it to.
// SPDX-License-Identifier: MIT
#pragma once
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dlsslop {

// A feature bit the network's SPIR-V uses (upstream listed them in DLSSNR-AMD's
// linux/src/core/nrvk.hpp, Context::create and adopt): the structure that
// carries it alone, and the core VkPhysicalDeviceVulkan1xFeatures structure
// that carries it too, if any.
struct NetworkFeature {
    const char* name;
    VkStructureType type;
    uint32_t offset;
    VkStructureType core;
    uint32_t core_offset;
    // The extension that provides it; null for core Vulkan 1.3.
    const char* extension;
};
constexpr VkStructureType kNoCore = VK_STRUCTURE_TYPE_MAX_ENUM;

#define DLSSLOP_ALONE(Struct, type, field) VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##type, offsetof(Struct, field)
#define DLSSLOP_CORE(digits, version, field)                                                                           \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_##version##_FEATURES, offsetof(VkPhysicalDeviceVulkan##digits##Features, field)
constexpr NetworkFeature kNetworkFeatures[] = {
    {"cooperativeMatrix", DLSSLOP_ALONE(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, COOPERATIVE_MATRIX_FEATURES_KHR,
                                        cooperativeMatrix), kNoCore, 0, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME},
    {"shaderFloat8", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT, shaderFloat8),
     kNoCore, 0, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME},
    {"shaderFloat8CooperativeMatrix", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT,
                                                    shaderFloat8CooperativeMatrix),
     kNoCore, 0, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME},
    {"storageBuffer16BitAccess", DLSSLOP_ALONE(VkPhysicalDevice16BitStorageFeatures, 16BIT_STORAGE_FEATURES,
                                               storageBuffer16BitAccess),
     DLSSLOP_CORE(11, 1_1, storageBuffer16BitAccess), nullptr},
    {"storageBuffer8BitAccess", DLSSLOP_ALONE(VkPhysicalDevice8BitStorageFeatures, 8BIT_STORAGE_FEATURES,
                                              storageBuffer8BitAccess),
     DLSSLOP_CORE(12, 1_2, storageBuffer8BitAccess), nullptr},
    {"shaderFloat16", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES, shaderFloat16),
     DLSSLOP_CORE(12, 1_2, shaderFloat16), nullptr},
    {"shaderInt8", DLSSLOP_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES, shaderInt8),
     DLSSLOP_CORE(12, 1_2, shaderInt8), nullptr},
    {"vulkanMemoryModel", DLSSLOP_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures, VULKAN_MEMORY_MODEL_FEATURES,
                                        vulkanMemoryModel),
     DLSSLOP_CORE(12, 1_2, vulkanMemoryModel), nullptr},
    // Not the network's own need: with the memory model on and this off, no shader on the device,
    // the game's or the composition's, may use Device scope (VUID-RuntimeSpirv-vulkanMemoryModel-06265).
    {"vulkanMemoryModelDeviceScope", DLSSLOP_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures,
                                                   VULKAN_MEMORY_MODEL_FEATURES, vulkanMemoryModelDeviceScope),
     DLSSLOP_CORE(12, 1_2, vulkanMemoryModelDeviceScope), nullptr},
    {"subgroupSizeControl", DLSSLOP_ALONE(VkPhysicalDeviceSubgroupSizeControlFeatures, SUBGROUP_SIZE_CONTROL_FEATURES,
                                          subgroupSizeControl),
     DLSSLOP_CORE(13, 1_3, subgroupSizeControl), nullptr},
    {"workgroupMemoryExplicitLayout",
     DLSSLOP_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
                   workgroupMemoryExplicitLayout),
     kNoCore, 0, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME},
    {"workgroupMemoryExplicitLayout8BitAccess",
     DLSSLOP_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
                   workgroupMemoryExplicitLayout8BitAccess),
     kNoCore, 0, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME},
    {"workgroupMemoryExplicitLayout16BitAccess",
     DLSSLOP_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
                   workgroupMemoryExplicitLayout16BitAccess),
     kNoCore, 0, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME},
};
#undef DLSSLOP_ALONE
#undef DLSSLOP_CORE

// A structure chain's links, read and written as bytes. Through
// VkBaseInStructure or VkBaseOutStructure they would be accesses of another
// type than the structures', which the compiler may assume never alias.
inline VkStructureType StructureType(const void* structure)
{
    VkStructureType type;
    std::memcpy(&type, structure, sizeof type);
    return type;
}
inline void* NextStructure(const void* structure)
{
    void* next;
    std::memcpy(&next, static_cast<const unsigned char*>(structure) + offsetof(VkBaseInStructure, pNext), sizeof next);
    return next;
}
inline void LinkStructure(void* structure, const void* next)
{
    std::memcpy(static_cast<unsigned char*>(structure) + offsetof(VkBaseOutStructure, pNext), &next, sizeof next);
}
// The structure of TYPE in the chain from FIRST, or null.
template<class Void> Void* FindStructure(Void* first, VkStructureType type)
{
    while (first && StructureType(first) != type) first = NextStructure(first);
    return first;
}

// The bit at OFFSET in STRUCTURE.
inline VkBool32& FeatureBit(void* structure, uint32_t offset)
{
    return *reinterpret_cast<VkBool32*>(static_cast<unsigned char*>(structure) + offset);
}
inline VkBool32 FeatureBit(const void* structure, uint32_t offset)
{
    return FeatureBit(const_cast<void*>(structure), offset);
}

// One structure of each type the table names, zeroed and chained behind a
// VkPhysicalDeviceFeatures2: for a support query or a vkCreateDevice.
class NetworkFeatureChain {
    VkPhysicalDeviceFeatures2 head_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &coop_};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                                                       &fp8_};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, &storage16_};
    VkPhysicalDevice16BitStorageFeatures storage16_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &storage8_};
    VkPhysicalDevice8BitStorageFeatures storage8_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &float16_};
    VkPhysicalDeviceShaderFloat16Int8Features float16_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                                                       &memory_};
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
                                                      &subgroup_};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
                                                          &layout_};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR layout_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};

public:
    NetworkFeatureChain() = default;
    NetworkFeatureChain(const NetworkFeatureChain&) = delete;
    VkPhysicalDeviceFeatures2& features2() { return head_; }
    // The structure of TYPE, whatever a caller has since linked it to.
    void* structure(VkStructureType type)
    {
        void* const all[] = {&coop_, &fp8_, &storage16_, &storage8_, &float16_, &memory_, &subgroup_, &layout_};
        for (void* s : all)
            if (StructureType(s) == type) return s;
        return nullptr;
    }
    VkBool32& bit(const NetworkFeature& f) { return FeatureBit(structure(f.type), f.offset); }
};

// Appends the network's extensions that LIST lacks: true if any was added.
inline bool AppendNetworkExtensions(std::vector<const char*>& list)
{
    const size_t before = list.size();
    for (const auto& f : kNetworkFeatures)
        if (f.extension &&
            std::none_of(list.begin(), list.end(), [&f](const char* name) { return !std::strcmp(name, f.extension); }))
            list.push_back(f.extension);
    return list.size() != before;
}

// Whether MATRICES lists for PHYSICAL the FP8 cooperative matrices that every
// network kernel multiplies with: e4m3 times e4m3 into FP32, 16x16x16, in a
// subgroup (upstream: nrvk's require_matrix_config). The shaders also declare
// an e4m3 accumulator that RADV lists in no configuration and accepts, so it is
// not required.
inline bool ListsNetworkMatrices(VkPhysicalDevice physical,
                                 PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrices)
{
    uint32_t count = 0;
    if (matrices(physical, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkCooperativeMatrixPropertiesKHR> listed(count, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    // VK_INCOMPLETE leaves out configurations beyond those counted.
    const VkResult result = matrices(physical, &count, listed.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) return false;
    listed.resize(count);
    return std::any_of(listed.begin(), listed.end(), [](const VkCooperativeMatrixPropertiesKHR& m) {
        return m.MSize == 16 && m.NSize == 16 && m.KSize == 16 && m.scope == VK_SCOPE_SUBGROUP_KHR &&
               m.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT && m.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
               m.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && m.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
               !m.saturatingAccumulation;
    });
}

// What stops the network running on PHYSICAL, or null: a feature's or an
// extension's name, "Vulkan 1.3", "32-lane compute subgroups", "e4m3 16x16x16
// cooperative matrices", or the name of MATRICES when that is null. ALSO names
// an extension the caller needs besides, or is null. The functions are the
// caller's route to the device: the loader's, or the next layer's.
inline const char* NetworkUnsupported(VkPhysicalDevice physical, PFN_vkGetPhysicalDeviceProperties2 properties2,
                                      PFN_vkGetPhysicalDeviceFeatures2 features2,
                                      PFN_vkEnumerateDeviceExtensionProperties extensions,
                                      PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrices,
                                      const char* also)
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
    if (also && !has(also)) return also;
    for (const auto& f : kNetworkFeatures)
        if (f.extension && !has(f.extension)) return f.extension;
    NetworkFeatureChain supported;
    features2(physical, &supported.features2());
    for (const auto& f : kNetworkFeatures)
        if (!supported.bit(f)) return f.name;
    // The cooperative-matrix fragments are laid out for 32-lane subgroups.
    if (subgroup.minSubgroupSize > 32 || subgroup.maxSubgroupSize < 32 ||
        !(subgroup.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
        return "32-lane compute subgroups";
    if (!matrices) return "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR";
    if (!ListsNetworkMatrices(physical, matrices)) return "e4m3 16x16x16 cooperative matrices";
    return nullptr;
}

} // namespace dlsslop
