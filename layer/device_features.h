#pragma once

#include "../common/network_requirements.h"

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace dlssnr {

inline const VkPhysicalDeviceFeatures2* CoreFeatures2(const VkDeviceCreateInfo& info) {
    for (auto* node = static_cast<const VkBaseInStructure*>(info.pNext); node; node = node->pNext)
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            return reinterpret_cast<const VkPhysicalDeviceFeatures2*>(node);
    return nullptr;
}

inline bool HasFormatlessStorageWrites(const VkDeviceCreateInfo& info) {
    const auto* features2 = CoreFeatures2(info);
    const auto* features = features2 ? &features2->features : info.pEnabledFeatures;
    return features && features->shaderStorageImageWriteWithoutFormat;
}

// The structure of TYPE in a pNext chain, or null.
inline const VkBaseInStructure* FindStructure(const void* chain, VkStructureType type) {
    for (auto* node = static_cast<const VkBaseInStructure*>(chain); node; node = node->pNext)
        if (node->sType == type) return node;
    return nullptr;
}

// Where a network feature is enabled in a chain: the structure carrying it
// alone, else the core structure carrying it, and the offset in it.
inline std::pair<const VkBaseInStructure*, uint32_t> NetworkFeatureIn(const void* chain,
                                                                      const dlsslop::NetworkFeature& f) {
    if (const auto* node = FindStructure(chain, f.type)) return {node, f.offset};
    if (f.core == dlsslop::kNoCore) return {nullptr, 0};
    return {FindStructure(chain, f.core), f.core_offset};
}

// The ledger: whether a device created from INFO, the request vkCreateDevice
// accepted, has every feature and extension the in-layer network needs. A
// device's supported features are no proof: only what was enabled may be used.
inline bool NetworkEnabled(const VkDeviceCreateInfo& info) {
    const auto enabled = [&info](const dlsslop::NetworkFeature& f) {
        bool extension = !f.extension;
        for (uint32_t i = 0; !extension && i < info.enabledExtensionCount; ++i)
            extension = !std::strcmp(info.ppEnabledExtensionNames[i], f.extension);
        const auto [node, offset] = NetworkFeatureIn(info.pNext, f);
        return extension && node && dlsslop::FeatureBit(const_cast<VkBaseInStructure*>(node), offset);
    };
    return std::all_of(std::begin(dlsslop::kNetworkFeatures), std::end(dlsslop::kNetworkFeatures), enabled);
}

// Whether the in-layer network is asked for: DLSSLOP_LAYER_NETWORK=1, while it
// is being developed.
inline bool NetworkRequested() {
    const char* value = std::getenv("DLSSLOP_LAYER_NETWORK");
    return value && !std::strcmp(value, "1");
}

// What keeps the in-layer network off a game's device, or null: a Vulkan 1.3
// instance, or what the device lacks (common/network_requirements.h). The
// functions are the next layer's.
inline const char* NetworkUnavailable(VkPhysicalDevice physical, uint32_t instanceVersion,
                                      PFN_vkGetPhysicalDeviceProperties2 properties2,
                                      PFN_vkGetPhysicalDeviceFeatures2 features2,
                                      PFN_vkEnumerateDeviceExtensionProperties extensions) {
    if (instanceVersion < VK_API_VERSION_1_3 || !properties2 || !features2) return "a Vulkan 1.3 instance";
    return dlsslop::NetworkUnsupported(physical, properties2, features2, extensions);
}

// Adds the network's extensions the request lacks: EXTENSIONS holds the
// request's list, as the layer extends it, and replaces it. True if any was added.
inline bool AddNetworkExtensions(VkDeviceCreateInfo& info, std::vector<const char*>& extensions) {
    if (extensions.empty())
        extensions.assign(info.ppEnabledExtensionNames, info.ppEnabledExtensionNames + info.enabledExtensionCount);
    const size_t before = extensions.size();
    for (const auto& f : dlsslop::kNetworkFeatures)
        if (f.extension && std::none_of(extensions.begin(), extensions.end(),
                                        [&f](const char* name) { return !std::strcmp(name, f.extension); }))
            extensions.push_back(f.extension);
    if (extensions.size() == before) return false;
    info.enabledExtensionCount = uint32_t(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    return true;
}

// What the layer adds to a game's vkCreateDevice: formatless storage writes
// for the composition, and on request the in-layer network's features. A bit
// is set in the structure the game chains that carries it, else in a
// structure of its own put at the head of the chain. The structures up to the
// last one changed are private copies: the game's const chain is never written.
// Loader nodes and core feature nodes cover the usual Proton chains; a
// structure to change behind one this cannot copy fails safely, and the game
// then creates its device as it asked.
class DeviceFeatureRequest {
    VkPhysicalDeviceFeatures legacy_{};
    std::vector<std::shared_ptr<void>> copies_;
    dlsslop::NetworkFeatureChain added_;

    template<class T> VkBaseOutStructure* Copy(const VkBaseInStructure* node) {
        auto copy = std::make_shared<T>(*reinterpret_cast<const T*>(node));
        auto* result = reinterpret_cast<VkBaseOutStructure*>(copy.get());
        copies_.push_back(std::move(copy));
        return result;
    }

    VkBaseOutStructure* CopyNode(const VkBaseInStructure* node) {
        switch (node->sType) {
#define COPY(type, tag) case tag: return Copy<type>(node)
            COPY(VkLayerDeviceCreateInfo, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO);
            COPY(VkPhysicalDeviceFeatures2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
            COPY(VkPhysicalDeviceVulkan11Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
            COPY(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
            COPY(VkPhysicalDeviceVulkan13Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
#ifdef VK_VERSION_1_4
            COPY(VkPhysicalDeviceVulkan14Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES);
#endif
            COPY(VkDeviceGroupDeviceCreateInfo, VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO);
            COPY(VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT);
            // The network's own structures, which a game may chain itself.
            COPY(VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR);
            COPY(VkPhysicalDeviceShaderFloat8FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT);
            COPY(VkPhysicalDevice16BitStorageFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES);
            COPY(VkPhysicalDevice8BitStorageFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES);
            COPY(VkPhysicalDeviceShaderFloat16Int8Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES);
            COPY(VkPhysicalDeviceVulkanMemoryModelFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES);
            COPY(VkPhysicalDeviceSubgroupSizeControlFeatures,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES);
            COPY(VkPhysicalDeviceSynchronization2Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES);
            COPY(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR);
#undef COPY
            default: return nullptr;
        }
    }

public:
    // Adds the features to INFO, with NETWORK the in-layer network's. False,
    // leaving INFO as it was, when a structure to change follows one this
    // cannot copy.
    bool Enable(VkDeviceCreateInfo& info, bool network) {
        // The bits to set in the game's structures, and the network's that none carries.
        std::vector<std::pair<const VkBaseInStructure*, uint32_t>> changes;
        std::vector<const dlsslop::NetworkFeature*> missing;
        const auto* features2 = FindStructure(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        constexpr uint32_t kFormatless = offsetof(VkPhysicalDeviceFeatures2, features.shaderStorageImageWriteWithoutFormat);
        if (features2 && !HasFormatlessStorageWrites(info)) changes.emplace_back(features2, kFormatless);
        if (network)
            for (const auto& f : dlsslop::kNetworkFeatures) {
                const auto place = NetworkFeatureIn(info.pNext, f);
                if (!place.first) missing.push_back(&f);
                else if (!dlsslop::FeatureBit(const_cast<VkBaseInStructure*>(place.first), place.second))
                    changes.push_back(place);
            }
        // Private copies of the chain up to the last structure changed. The
        // last copy's pNext still leads to the rest of the game's chain.
        auto* head = const_cast<VkBaseOutStructure*>(static_cast<const VkBaseOutStructure*>(info.pNext));
        const VkBaseInStructure* node = static_cast<const VkBaseInStructure*>(info.pNext);
        VkBaseOutStructure* previous = nullptr;
        for (size_t left = changes.size(); left; node = node->pNext) {
            auto* copy = CopyNode(node);
            if (!copy) {
                copies_.clear();
                return false;
            }
            if (previous)
                previous->pNext = copy;
            else
                head = copy;
            previous = copy;
            for (const auto& change : changes)
                if (change.first == node) {
                    dlsslop::FeatureBit(copy, change.second) = VK_TRUE;
                    --left;
                }
        }
        // Structures of its own, at the head, for the network features no game structure carries.
        std::vector<VkBaseOutStructure*> own;
        for (const auto* f : missing) {
            auto* structure = static_cast<VkBaseOutStructure*>(added_.structure(f->type));
            dlsslop::FeatureBit(structure, f->offset) = VK_TRUE;
            if (std::find(own.begin(), own.end(), structure) == own.end()) own.push_back(structure);
        }
        for (auto it = own.rbegin(); it != own.rend(); ++it) {
            (*it)->pNext = head;
            head = *it;
        }
        info.pNext = head;
        if (!features2 && !HasFormatlessStorageWrites(info)) {
            legacy_ = info.pEnabledFeatures ? *info.pEnabledFeatures : VkPhysicalDeviceFeatures{};
            legacy_.shaderStorageImageWriteWithoutFormat = VK_TRUE;
            info.pEnabledFeatures = &legacy_;
        }
        return true;
    }
};

}  // namespace dlssnr
