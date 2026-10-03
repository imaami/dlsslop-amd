/** @file
 *
 * The next layer's entry points, resolved once per instance and per device.
 *
 * A layer must never call a Vulkan function through the loader's own exported symbol: that re-enters
 * the top of the chain and, for anything this layer hooks, recurses. Everything below therefore goes
 * through pfnNextGetInstanceProcAddr / pfnNextGetDeviceProcAddr, which is what these two tables hold.
 *
 * The lists are long because the composition runs a real compute pipeline on the game's device --
 * images, views, descriptors, a pipeline and the dispatch -- rather than the transfer copies the
 * first version of this layer needed.
 */
#ifndef DLSSLOP_AMD_LAYER_VK_TABLE_H_
#define DLSSLOP_AMD_LAYER_VK_TABLE_H_

#ifndef VK_NO_PROTOTYPES
# define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#define DLSSNR_INSTANCE_FN_LIST(X) \
	X(vkDestroyInstance) \
	X(vkEnumeratePhysicalDevices) \
	X(vkGetPhysicalDeviceProperties) \
	X(vkGetPhysicalDeviceFeatures) \
	X(vkGetPhysicalDeviceMemoryProperties) \
	X(vkGetPhysicalDeviceFormatProperties) \
	X(vkGetPhysicalDeviceQueueFamilyProperties) \
	X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
	X(vkEnumerateDeviceExtensionProperties) \
	X(vkGetPhysicalDeviceProperties2) \
	X(vkGetPhysicalDeviceFeatures2) \
	X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)

#define DLSSNR_DEVICE_FN_LIST(X) \
	X(vkDestroyDevice) \
	X(vkGetDeviceQueue) \
	X(vkGetDeviceQueue2) \
	X(vkCreateSwapchainKHR) \
	X(vkDestroySwapchainKHR) \
	X(vkGetSwapchainImagesKHR) \
	X(vkQueuePresentKHR) \
	X(vkQueueSubmit) \
	X(vkQueueSubmit2) \
	X(vkQueueWaitIdle) \
	X(vkCreateCommandPool) \
	X(vkDestroyCommandPool) \
	X(vkAllocateCommandBuffers) \
	X(vkBeginCommandBuffer) \
	X(vkEndCommandBuffer) \
	X(vkCreateFence) \
	X(vkDestroyFence) \
	X(vkWaitForFences) \
	X(vkResetFences) \
	X(vkCreateSemaphore) \
	X(vkDestroySemaphore) \
	X(vkCreateImage) \
	X(vkDestroyImage) \
	X(vkGetImageMemoryRequirements) \
	X(vkAllocateMemory) \
	X(vkFreeMemory) \
	X(vkBindImageMemory) \
	X(vkCreateImageView) \
	X(vkDestroyImageView) \
	X(vkMapMemory) \
	X(vkUnmapMemory) \
	X(vkCreateBuffer) \
	X(vkDestroyBuffer) \
	X(vkGetBufferMemoryRequirements) \
	X(vkBindBufferMemory) \
	X(vkCmdCopyBufferToImage) \
	X(vkCmdCopyImageToBuffer) \
	X(vkCmdCopyImage) \
	X(vkCmdCopyBuffer) \
	X(vkCmdBlitImage) \
	X(vkCmdPipelineBarrier) \
	X(vkDeviceWaitIdle) \
	X(vkCreateSampler) \
	X(vkDestroySampler) \
	X(vkCreateShaderModule) \
	X(vkDestroyShaderModule) \
	X(vkCreateDescriptorSetLayout) \
	X(vkDestroyDescriptorSetLayout) \
	X(vkCreatePipelineLayout) \
	X(vkDestroyPipelineLayout) \
	X(vkCreateDescriptorPool) \
	X(vkDestroyDescriptorPool) \
	X(vkAllocateDescriptorSets) \
	X(vkUpdateDescriptorSets) \
	X(vkCreateComputePipelines) \
	X(vkDestroyPipeline) \
	X(vkCmdBindPipeline) \
	X(vkCmdBindDescriptorSets) \
	X(vkCmdDispatch) \
	X(vkCmdPushConstants) \
	X(vkCmdFillBuffer) \
	X(vkGetMemoryFdKHR) \
	X(vkQueueSubmit2KHR) \
	X(vkQueueBindSparse)

/** @brief The next layer's instance-level entry points.
 *
 * Zero-initialize it: a table holds no defaults of its own.
 */
struct instance_table {
	PFN_vkGetInstanceProcAddr next_gipa; //!< Resolves the others.
#define X(name) PFN_##name name;
	DLSSNR_INSTANCE_FN_LIST(X)
#undef X
};

/** @brief The next layer's device-level entry points.
 *
 * Zero-initialize it: a table holds no defaults of its own.
 */
struct device_table {
	PFN_vkGetDeviceProcAddr next_dpa; //!< Resolves the others.
#define X(name) PFN_##name name;
	DLSSNR_DEVICE_FN_LIST(X)
#undef X
};

/** @brief Resolves an instance table's entry points through its next_gipa.
 *
 * @param table    The table; next_gipa must be set.
 * @param instance The instance whose entry points to resolve.
 */
static inline void
instance_table_load (struct instance_table *table,
                     VkInstance             instance)
{
#define X(name) table->name = (PFN_##name)table->next_gipa(instance, #name);
	DLSSNR_INSTANCE_FN_LIST(X)
#undef X
}

/** @brief Resolves a device table's entry points through its next_dpa.
 *
 * @param table  The table; next_dpa must be set.
 * @param device The device whose entry points to resolve.
 */
static inline void
device_table_load (struct device_table *table,
                   VkDevice             device)
{
#define X(name) table->name = (PFN_##name)table->next_dpa(device, #name);
	DLSSNR_DEVICE_FN_LIST(X)
#undef X
}

/** @brief Whether a device table has what the composition needs to exist at all.
 *
 * Anything absent from this list is either optional (queue2) or already checked by the caller.
 *
 * @param table The table.
 * @return      true if every entry point the composition needs is there.
 */
static inline bool
device_table_complete (struct device_table const *table)
{
	return table->vkCreateImage && table->vkCreateImageView && table->vkAllocateMemory
	       && table->vkBindImageMemory && table->vkCreateBuffer && table->vkBindBufferMemory
	       && table->vkMapMemory && table->vkCreateSampler && table->vkCreateShaderModule
	       && table->vkCreateDescriptorSetLayout && table->vkCreatePipelineLayout
	       && table->vkCreateDescriptorPool && table->vkAllocateDescriptorSets
	       && table->vkUpdateDescriptorSets && table->vkCreateComputePipelines
	       && table->vkCmdBindPipeline && table->vkCmdBindDescriptorSets && table->vkCmdDispatch
	       && table->vkCmdPipelineBarrier && table->vkCmdCopyImage && table->vkCmdCopyImageToBuffer
	       && table->vkCmdCopyBufferToImage;
}

#undef DLSSNR_DEVICE_FN_LIST
#undef DLSSNR_INSTANCE_FN_LIST

#endif /* DLSSLOP_AMD_LAYER_VK_TABLE_H_ */
