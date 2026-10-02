/** @file
 *
 * OptiScaler's Shader_Vk, on the layer's dispatch table.
 *
 * The original resolves Vulkan through the loader's exported symbols, which a layer must not do: the
 * hooked entry points would recurse and the unhooked ones would re-enter the top of the chain. Every
 * call here goes through the device_table the layer built from pfnNextGetDeviceProcAddr instead. The
 * C port creates the same objects as upstream's class, in the same order, so a fix on either side
 * carries over to the other. Where the class returned a bool or nothing, the port returns the VkResult
 * of the step that failed. A pass owns the objects whose handles are not null.
 *
 * A pass embeds struct shader_vk as its first member: struct dlss_nr_pass and struct scaler_vk. What
 * a pass builds itself with is in shader_vk_priv.h.
 *
 * Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_SHADER_VK_H_
#define DLSSLOP_AMD_LAYER_SHADER_VK_H_

#include "vk_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief The Vulkan objects that every pass has.
 *
 * A zeroed object is an empty one: every handle is VK_NULL_HANDLE. shader_vk_fini() destroys
 * the objects whose handles are not null. A failed Vulkan call leaves its output undefined, so a
 * handle is stored only once the call that made it has succeeded. The pipeline is created last, so it
 * exists exactly when the pass was built.
 */
struct shader_vk {
	VkPipeline                   pipeline;               //!< The pass's compute pipeline, created last.
	VkPipelineLayout             pipeline_layout;        //!< The pipeline's layout.
	VkDescriptorSetLayout        descriptor_set_layout;  //!< The bindings of the pass's shader.
	VkDescriptorPool             descriptor_pool;        //!< The pool of the pass's descriptor sets.
	VkBuffer                     constant_buffer;        //!< The ring of constant slots.
	VkDeviceMemory               constant_buffer_memory; //!< The ring's host-visible memory.
	VkSampler                    texture_sampler;        //!< The sampler that the shader reads with.
	char const                  *name;                   //!< What the pass's log lines call it.
	struct device_table const   *vk;                     //!< The device's next-layer entry points.
	struct instance_table const *instance;               //!< The instance's next-layer entry points.
	VkDevice                     device;                 //!< The device that the pass runs on.
	VkPhysicalDevice             physical_device;        //!< The device's physical device.
	void                        *mapped_constant_buffer; //!< The ring, mapped.
};

/** @brief Records a barrier that moves an image between layouts, with the conservative masks of the
 *         upstream helper.
 *
 * Callers that know better state their own barrier; this is for the ones that do not. An image
 * leaving UNDEFINED waits for nothing, and one leaving GENERAL waits for compute shader writes. An
 * image entering SHADER_READ_ONLY_OPTIMAL is made visible to compute shader reads, and one entering
 * GENERAL to compute shader writes. Any other layout waits for, or is made visible to, all memory
 * access by all commands.
 *
 * @param s                 A pass's base, or nullptr. A base without a device table, such as an
 *                          empty one, records nothing.
 * @param cmd_buffer        The command buffer to record into.
 * @param image             The image.
 * @param old_layout        Its layout before the barrier.
 * @param new_layout        Its layout after the barrier.
 * @param subresource_range The part of the image that moves.
 */
extern void
shader_vk_set_image_layout (struct shader_vk const  *s,
                            VkCommandBuffer          cmd_buffer,
                            VkImage                  image,
                            VkImageLayout            old_layout,
                            VkImageLayout            new_layout,
                            VkImageSubresourceRange  subresource_range);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_LAYER_SHADER_VK_H_ */
