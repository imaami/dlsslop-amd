/** @file
 *
 * What a pass builds itself with: Shader_Vk's protected members, which shader_vk.c, dlssnr_pass.c
 * and scaler_vk.c share. C only.
 *
 * Each function that creates an object logs why it could not, and returns the VkResult that says so.
 */
#ifndef DLSSLOP_AMD_LAYER_SHADER_VK_PRIV_H_
#define DLSSLOP_AMD_LAYER_SHADER_VK_PRIV_H_

#include <stddef.h>
#include <stdint.h>

#include "shader_vk.h"

/** @brief A binding of one descriptor that the compute stage reads, as a constant initializer of a
 *         VkDescriptorSetLayoutBinding.
 *
 * @param binding The binding's number.
 * @param type    The descriptor's type.
 */
#define SHADER_VK_BINDING(binding, type) { (binding), (type), 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }

/** @brief Returns a pass's base, which knows its device and owns nothing yet.
 *
 * @param name            What the pass's log lines call it: a string that outlives the pass.
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device that the pass runs on.
 * @param physical_device The device's physical device.
 * @return                The base.
 */
extern struct shader_vk
shader_vk (char const                  *name,
           struct device_table const   *vk,
           struct instance_table const *instance,
           VkDevice                     device,
           VkPhysicalDevice             physical_device);

/** @brief Destroys the objects whose handles a pass's base holds, then leaves it empty.
 *
 * A zeroed base owns nothing.
 *
 * @param dest The base, or nullptr.
 */
extern void
shader_vk_fini (struct shader_vk *dest);

/** @brief Finds a memory type of the physical device.
 *
 * @param s           The base.
 * @param type_filter The memory types to choose from, one bit each.
 * @param properties  The properties that the type must have.
 * @return            The first such type's index, or UINT32_MAX if there is none.
 */
extern uint32_t
shader_vk_find_memory_type (struct shader_vk const *s,
                            uint32_t                type_filter,
                            VkMemoryPropertyFlags   properties);

/** @brief Creates the pass's compute pipeline from SPIR-V, in the pass's pipeline layout.
 *
 * The SPIR-V is copied first, because Vulkan reads it as 32-bit words and a byte array is not
 * aligned for them.
 *
 * @param s           A base whose pipeline layout exists and that has no pipeline.
 * @param code        The SPIR-V.
 * @param size        Its size in bytes, a multiple of 4.
 * @param entry_point The shader's entry point.
 * @return            VK_SUCCESS, VK_ERROR_OUT_OF_HOST_MEMORY if the copy cannot be allocated, or
 *                    what vkCreateShaderModule() or vkCreateComputePipelines() returned.
 */
extern VkResult
shader_vk_create_compute_pipeline (struct shader_vk    *s,
                                   unsigned char const *code,
                                   size_t               size,
                                   char const          *entry_point);

/** @brief Creates the pass's constant ring: a buffer with memory of its own, bound to it.
 *
 * Each handle is stored in the base once the call that made it succeeded. If a later step fails,
 * what was made stays in the base, and shader_vk_fini() destroys it.
 *
 * @param s          A base that has no constant buffer.
 * @param size       The buffer's size.
 * @param usage      The buffer's usage.
 * @param properties The properties that the memory must have.
 * @return           VK_SUCCESS, VK_ERROR_FEATURE_NOT_PRESENT if no memory type has the properties,
 *                   or what the failed Vulkan call returned.
 */
extern VkResult
shader_vk_create_buffer_resource (struct shader_vk      *s,
                                  VkDeviceSize           size,
                                  VkBufferUsageFlags     usage,
                                  VkMemoryPropertyFlags  properties);

/** @brief Creates the descriptor set layout of a pass's shader, and the pipeline layout around it.
 *
 * @param s        A base that has neither layout.
 * @param bindings The shader's bindings.
 * @param count    Their number.
 * @return         VK_SUCCESS, or what the failed Vulkan call returned.
 */
extern VkResult
shader_vk_create_layouts (struct shader_vk                   *s,
                          VkDescriptorSetLayoutBinding const *bindings,
                          uint32_t                            count);

/** @brief Creates the pool that a pass's descriptor sets come from.
 *
 * @param s          A base that has no pool.
 * @param pool_sizes The descriptors of each type that the pool holds.
 * @param count      The number of pool sizes.
 * @param max_sets   The most sets that the pool holds.
 * @return           VK_SUCCESS, or what vkCreateDescriptorPool() returned.
 */
extern VkResult
shader_vk_create_descriptor_pool (struct shader_vk           *s,
                                  VkDescriptorPoolSize const *pool_sizes,
                                  uint32_t                    count,
                                  uint32_t                    max_sets);

/** @brief The most descriptor sets that shader_vk_create_descriptor_sets() allocates at once. */
static constexpr uint32_t SHADER_VK_SETS_MAX = 18;

/** @brief Allocates descriptor sets of the pass's layout from its pool.
 *
 * @param s     A base whose descriptor set layout and pool exist.
 * @param count The number of sets, from 1 to SHADER_VK_SETS_MAX.
 * @param sets  Receives the sets.
 * @return      VK_SUCCESS, VK_ERROR_INITIALIZATION_FAILED for a count out of range, or what
 *              vkAllocateDescriptorSets() returned.
 */
extern VkResult
shader_vk_create_descriptor_sets (struct shader_vk const *s,
                                  uint32_t                count,
                                  VkDescriptorSet        *sets);

/** @brief Creates the sampler that a pass's shader reads with.
 *
 * @param s            A base that has no sampler.
 * @param filter       The magnification and minification filter.
 * @param address_mode The address mode of all three coordinates.
 * @return             VK_SUCCESS, or what vkCreateSampler() returned.
 */
extern VkResult
shader_vk_create_sampler (struct shader_vk     *s,
                          VkFilter              filter,
                          VkSamplerAddressMode  address_mode);

#endif /* DLSSLOP_AMD_LAYER_SHADER_VK_PRIV_H_ */
