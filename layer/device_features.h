#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
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

// Copy only the prefix ending at Features2, keeping all unknown trailing nodes
// untouched. Never write through the application's const pNext chain. Loader
// nodes and core feature nodes cover the usual Proton chains; an unknown prefix
// fails safely so the application can create its device and present untouched.
class FormatlessStorageFeatures {
    VkPhysicalDeviceFeatures legacy_{};
    std::vector<std::shared_ptr<void>> copies_;

    template<class T> VkBaseOutStructure* Copy(const VkBaseInStructure* node) {
        auto copy = std::make_shared<T>(*reinterpret_cast<const T*>(node));
        auto* result = reinterpret_cast<VkBaseOutStructure*>(copy.get());
        copies_.push_back(std::move(copy));
        return result;
    }

    VkBaseOutStructure* CopyPrefix(const VkBaseInStructure* node) {
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
#undef COPY
            default: return nullptr;
        }
    }

public:
    bool Enable(VkDeviceCreateInfo& info) {
        if (HasFormatlessStorageWrites(info)) return true;
        if (!CoreFeatures2(info)) {
            legacy_ = info.pEnabledFeatures ? *info.pEnabledFeatures : VkPhysicalDeviceFeatures{};
            legacy_.shaderStorageImageWriteWithoutFormat = VK_TRUE;
            info.pEnabledFeatures = &legacy_;
            return true;
        }
        VkBaseOutStructure* first = nullptr;
        VkBaseOutStructure* previous = nullptr;
        for (auto* node = static_cast<const VkBaseInStructure*>(info.pNext); node; node = node->pNext) {
            auto* copy = CopyPrefix(node);
            if (!copy) {
                copies_.clear();
                return false;
            }
            if (!first) first = copy;
            if (previous) previous->pNext = copy;
            previous = copy;
            if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                reinterpret_cast<VkPhysicalDeviceFeatures2*>(copy)->features
                    .shaderStorageImageWriteWithoutFormat = VK_TRUE;
                info.pNext = first;
                return true;
            }
        }
        return false;
    }
};

}  // namespace dlssnr
