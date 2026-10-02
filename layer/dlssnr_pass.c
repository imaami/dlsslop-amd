/** @file
 *
 * The composition pass: OptiScaler's DlssNr_Vk on the layer's dispatch table.
 */
#include <stdint.h>
#include <string.h>

#include "dlssnr/DlssNr_Layout.h"
#include "dlssnr/DlssNr_Shader_Vk.h"
#include "dlssnr_pass.h"
#include "log.h"
#include "shader_vk_priv.h"

/** @brief The composition shader's bindings.
 *
 * The layout mirrors the [[vk::binding]] numbers in dlssnr.hlsl, entry for entry.
 */
static VkDescriptorSetLayoutBinding const DLSS_NR_PASS_BINDINGS[] = {
	SHADER_VK_BINDING(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),         // Params
	SHADER_VK_BINDING(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gSource
	SHADER_VK_BINDING(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gModel
	SHADER_VK_BINDING(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gOriginal
	SHADER_VK_BINDING(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gMotion
	SHADER_VK_BINDING(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),          // gTarget
	SHADER_VK_BINDING(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),          // gKeep
	SHADER_VK_BINDING(7, VK_DESCRIPTOR_TYPE_SAMPLER)                 // gLinear
};

static_assert(DLSS_NR_PASS_SLOTS <= SHADER_VK_SETS_MAX, "the pass allocates a set per slot at once");

/** @brief The number of bindings. */
static constexpr uint32_t DLSS_NR_PASS_BINDING_COUNT =
	sizeof DLSS_NR_PASS_BINDINGS / sizeof *DLSS_NR_PASS_BINDINGS;

/** @brief The descriptors of every slot's set. */
static VkDescriptorPoolSize const DLSS_NR_PASS_POOL_SIZES[] = {
	{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         DLSS_NR_PASS_SLOTS     },
	{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * DLSS_NR_PASS_SLOTS },
	{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          2 * DLSS_NR_PASS_SLOTS },
	{ VK_DESCRIPTOR_TYPE_SAMPLER,                DLSS_NR_PASS_SLOTS     }
};

/** @brief The number of pool sizes. */
static constexpr uint32_t DLSS_NR_PASS_POOL_SIZE_COUNT =
	sizeof DLSS_NR_PASS_POOL_SIZES / sizeof *DLSS_NR_PASS_POOL_SIZES;

/** @brief The whole of a 2D image with one level and one layer. */
static VkImageSubresourceRange const DLSS_NR_PASS_COLOR_RANGE = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

/** @brief Builds the pass.
 *
 * @param p A pass whose base knows its device and owns nothing.
 * @return  VK_SUCCESS, or the failure that the log names.
 */
static VkResult
build (struct dlss_nr_pass *p)
{
	struct shader_vk *const s = &p->shader;
	if (!s->device || !s->physical_device) {
		log_printf("[pass] no device");
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	// Linear, because the resolve reads the model's answer at a different size than it writes --
	// the model may have run at a reduced resolution and the edit has to be stretched back over the
	// frame.
	VkResult r = shader_vk_create_sampler(s, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	if (r != VK_SUCCESS)
		return r;

	// The constant ring. A uniform buffer binding can be offset into, but only to a multiple of the
	// device's own alignment, so the stride is the struct rounded up rather than the struct itself.
	VkPhysicalDeviceProperties props;
	s->instance->vkGetPhysicalDeviceProperties(s->physical_device, &props);

	VkDeviceSize const align = props.limits.minUniformBufferOffsetAlignment;
	VkDeviceSize const alignment = align ? align : 1;
	p->slot_stride = (sizeof (struct dlss_nr_constants) + alignment - 1) / alignment * alignment;

	r = shader_vk_create_buffer_resource(s, p->slot_stride * DLSS_NR_PASS_SLOTS,
	                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
	                                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
	                                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	if (r != VK_SUCCESS) {
		log_printf("[pass] could not allocate the constant ring");
		return r;
	}

	void *mapped;
	r = s->vk->vkMapMemory(s->device, s->constant_buffer_memory, 0,
	                       p->slot_stride * DLSS_NR_PASS_SLOTS, 0, &mapped);
	if (r != VK_SUCCESS) {
		log_printf("[pass] could not map the constant ring");
		return r;
	}
	s->mapped_constant_buffer = mapped;

	r = shader_vk_create_layouts(s, DLSS_NR_PASS_BINDINGS, DLSS_NR_PASS_BINDING_COUNT);
	if (r != VK_SUCCESS)
		return r;

	// One set per slot rather than per frame: two dispatches in the same frame need two sets, or the
	// second overwrites bindings the first has not consumed yet.
	r = shader_vk_create_descriptor_pool(s, DLSS_NR_PASS_POOL_SIZES, DLSS_NR_PASS_POOL_SIZE_COUNT,
	                                     DLSS_NR_PASS_SLOTS);
	if (r != VK_SUCCESS
	    || (r = shader_vk_create_descriptor_sets(s, DLSS_NR_PASS_SLOTS, p->descriptor_sets))
	       != VK_SUCCESS) {
		log_printf("[pass] expected %u descriptor sets, got 0", DLSS_NR_PASS_SLOTS);
		return r;
	}

	r = shader_vk_create_compute_pipeline(s, dlssnr_spv, sizeof dlssnr_spv, "CSMain");
	if (r != VK_SUCCESS) {
		log_printf("[pass] could not create the compute pipeline");
		return r;
	}

	log_printf("[pass] composition up: %u constant slots, stride %llu", DLSS_NR_PASS_SLOTS,
	           (unsigned long long)p->slot_stride);
	return VK_SUCCESS;
}

struct dlss_nr_pass
dlss_nr_pass (struct device_table const   *vk,
              struct instance_table const *instance,
              VkDevice                     device,
              VkPhysicalDevice             physical_device)
{
	struct dlss_nr_pass ret = {
		.shader = shader_vk("dlssnr-composition", vk, instance, device, physical_device)
	};
	ret.error = build(&ret);
	return ret;
}

VkResult
dlss_nr_pass_init (struct dlss_nr_pass         *dest,
                   struct device_table const   *vk,
                   struct instance_table const *instance,
                   VkDevice                     device,
                   VkPhysicalDevice             physical_device)
{
	if (!dest)
		return VK_ERROR_INITIALIZATION_FAILED;
	*dest = dlss_nr_pass(vk, instance, device, physical_device);
	return dest->error;
}

void
dlss_nr_pass_fini (struct dlss_nr_pass *dest)
{
	if (!dest)
		return;

	struct device_table const *const vk = dest->shader.vk;
	VkDevice const device = dest->shader.device;
	if (dest->dummy_view)
		vk->vkDestroyImageView(device, dest->dummy_view, nullptr);
	if (dest->dummy_image)
		vk->vkDestroyImage(device, dest->dummy_image, nullptr);
	if (dest->dummy_memory)
		vk->vkFreeMemory(device, dest->dummy_memory, nullptr);

	shader_vk_fini(&dest->shader);
	*dest = (struct dlss_nr_pass){0};
}

/** @brief Creates the placeholder: one pixel, R16G16B16A16_SFLOAT so it is legal for both a sampled
 *         read and a storage write.
 *
 * The handles are made in locals and stored in the pass only once all three exist. If a step fails,
 * the function destroys what it created, and the pass still has no placeholder.
 *
 * @param p The pass, which has no placeholder.
 * @return  true if the image, its memory and its view exist.
 */
static bool
make_dummy (struct dlss_nr_pass *p)
{
	struct shader_vk const *const s = &p->shader;
	struct device_table const *const vk = s->vk;
	VkImageCreateInfo const image_info = {
		.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType     = VK_IMAGE_TYPE_2D,
		.format        = VK_FORMAT_R16G16B16A16_SFLOAT,
		.extent        = { 1, 1, 1 },
		.mipLevels     = 1,
		.arrayLayers   = 1,
		.samples       = VK_SAMPLE_COUNT_1_BIT,
		.tiling        = VK_IMAGE_TILING_OPTIMAL,
		.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
		.sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};
	VkImage image;
	if (vk->vkCreateImage(s->device, &image_info, nullptr, &image) != VK_SUCCESS) {
		log_printf("[pass] could not create the placeholder image");
		return false;
	}

	VkMemoryRequirements requirements;
	vk->vkGetImageMemoryRequirements(s->device, image, &requirements);

	uint32_t const type = shader_vk_find_memory_type(s, requirements.memoryTypeBits,
	                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (type == UINT32_MAX)
		goto destroy_image;

	VkMemoryAllocateInfo const allocate_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = requirements.size,
		.memoryTypeIndex = type
	};
	VkDeviceMemory memory;
	if (vk->vkAllocateMemory(s->device, &allocate_info, nullptr, &memory) != VK_SUCCESS) {
		log_printf("[pass] could not back the placeholder image");
		goto destroy_image;
	}
	if (vk->vkBindImageMemory(s->device, image, memory, 0) != VK_SUCCESS) {
		log_printf("[pass] could not back the placeholder image");
		goto free_memory;
	}

	VkImageViewCreateInfo const view_info = {
		.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image            = image,
		.viewType         = VK_IMAGE_VIEW_TYPE_2D,
		.format           = VK_FORMAT_R16G16B16A16_SFLOAT,
		.subresourceRange = DLSS_NR_PASS_COLOR_RANGE
	};
	VkImageView view;
	if (vk->vkCreateImageView(s->device, &view_info, nullptr, &view) == VK_SUCCESS) {
		p->dummy_image = image;
		p->dummy_memory = memory;
		p->dummy_view = view;
		return true;
	}

	log_printf("[pass] could not view the placeholder image");
free_memory:
	vk->vkFreeMemory(s->device, memory, nullptr);
destroy_image:
	vk->vkDestroyImage(s->device, image, nullptr);
	return false;
}

/** @brief Makes the placeholder ready, once: creates it, and records its move into GENERAL.
 *
 * The placeholder is ready exactly while its view exists: the move is recorded in the call that
 * makes it.
 *
 * Its content is never read: it exists because Vulkan rejects a descriptor set with an unwritten
 * binding, and the shader declares all seven resources at file scope whichever mode is running.
 * GENERAL satisfies both a sampled read and a storage write, so the placeholder can stand in for
 * either kind of slot without ever changing layout again.
 *
 * @param p        The pass.
 * @param cmd_list The command buffer to record the move into.
 * @return         true if the placeholder is ready.
 */
static bool
create_dummy (struct dlss_nr_pass *p,
              VkCommandBuffer      cmd_list)
{
	if (p->dummy_view)
		return true;
	if (!make_dummy(p))
		return false;

	shader_vk_set_image_layout(&p->shader, cmd_list, p->dummy_image, VK_IMAGE_LAYOUT_UNDEFINED,
	                           VK_IMAGE_LAYOUT_GENERAL, DLSS_NR_PASS_COLOR_RANGE);
	return true;
}

/** @brief What a sampled binding reads: the view in its layout, or the placeholder in GENERAL.
 *
 * @param p      The pass.
 * @param view   The view, or VK_NULL_HANDLE.
 * @param layout The view's layout.
 * @return       The binding's image info.
 */
static VkDescriptorImageInfo
read_info (struct dlss_nr_pass const *p,
           VkImageView                view,
           VkImageLayout              layout)
{
	if (!view)
		return (VkDescriptorImageInfo){
			p->shader.texture_sampler, p->dummy_view, VK_IMAGE_LAYOUT_GENERAL
		};
	return (VkDescriptorImageInfo){ p->shader.texture_sampler, view, layout };
}

/** @brief What a storage binding writes: the view, or the placeholder, in GENERAL.
 *
 * @param p    The pass.
 * @param view The view, or VK_NULL_HANDLE.
 * @return     The binding's image info.
 */
static VkDescriptorImageInfo
write_info (struct dlss_nr_pass const *p,
            VkImageView                view)
{
	return (VkDescriptorImageInfo){
		VK_NULL_HANDLE, view ? view : p->dummy_view, VK_IMAGE_LAYOUT_GENERAL
	};
}

/** @brief Writes every binding of a slot's set.
 *
 * @param p               The pass.
 * @param set             The slot's set.
 * @param constant_offset The slot's offset in the constant ring.
 * @param source          gSource, or VK_NULL_HANDLE.
 * @param model           gModel, or VK_NULL_HANDLE.
 * @param original        gOriginal, or VK_NULL_HANDLE.
 * @param motion          gMotion, or VK_NULL_HANDLE.
 * @param target          gTarget, or VK_NULL_HANDLE.
 * @param keep            gKeep, or VK_NULL_HANDLE.
 * @param source_layout   The layout of source.
 * @param motion_layout   The layout of motion.
 */
static void
write_descriptors (struct dlss_nr_pass const *p,
                   VkDescriptorSet            set,
                   VkDeviceSize               constant_offset,
                   VkImageView                source,
                   VkImageView                model,
                   VkImageView                original,
                   VkImageView                motion,
                   VkImageView                target,
                   VkImageView                keep,
                   VkImageLayout              source_layout,
                   VkImageLayout              motion_layout)
{
	struct shader_vk const *const s = &p->shader;
	VkDescriptorBufferInfo const buffer_info = {
		s->constant_buffer, constant_offset, sizeof (struct dlss_nr_constants)
	};
	// Bindings 1 to 7, in order.
	VkDescriptorImageInfo const image_info[DLSS_NR_PASS_BINDING_COUNT - 1] = {
		read_info(p, source, source_layout),
		read_info(p, model, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		read_info(p, original, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		read_info(p, motion, motion_layout),
		write_info(p, target),
		write_info(p, keep),
		{ s->texture_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED }
	};

	VkWriteDescriptorSet writes[DLSS_NR_PASS_BINDING_COUNT] = {
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = set,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo     = &buffer_info
		}
	};
	for (uint32_t i = 1; i < DLSS_NR_PASS_BINDING_COUNT; ++i) {
		writes[i] = (VkWriteDescriptorSet){
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = set,
			.dstBinding      = i,
			.descriptorCount = 1,
			.descriptorType  = DLSS_NR_PASS_BINDINGS[i].descriptorType,
			.pImageInfo      = &image_info[i - 1]
		};
	}

	s->vk->vkUpdateDescriptorSets(s->device, DLSS_NR_PASS_BINDING_COUNT, writes, 0, nullptr);
}

bool
dlss_nr_pass_dispatch (struct dlss_nr_pass            *p,
                       VkCommandBuffer                 cmd_list,
                       struct dlss_nr_constants const *constants,
                       uint32_t                        threads_x,
                       uint32_t                        threads_y,
                       VkImageView                     source,
                       VkImageView                     model,
                       VkImageView                     original,
                       VkImageView                     motion,
                       VkImageView                     target,
                       VkImageView                     keep,
                       VkImageLayout                   source_layout,
                       VkImageLayout                   motion_layout)
{
	if (!p || !p->shader.pipeline || !cmd_list)
		return false;

	if (!target) {
		log_printf("[pass] a dispatch with nothing to write");
		return false;
	}

	if (!create_dummy(p, cmd_list))
		return false;

	struct shader_vk const *const s = &p->shader;
	uint32_t const slot = p->slot;
	p->slot = (slot + 1) % DLSS_NR_PASS_SLOTS;

	VkDeviceSize const offset = p->slot_stride * slot;
	memcpy((unsigned char *)s->mapped_constant_buffer + offset, constants, sizeof *constants);

	write_descriptors(p, p->descriptor_sets[slot], offset, source, model, original, motion, target,
	                  keep, source_layout, motion_layout);

	s->vk->vkCmdBindPipeline(cmd_list, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
	s->vk->vkCmdBindDescriptorSets(cmd_list, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_layout, 0,
	                               1, &p->descriptor_sets[slot], 0, nullptr);

	// The shader's thread group is 8x8, the same as the D3D12 path.
	s->vk->vkCmdDispatch(cmd_list, (threads_x + 7) / 8, (threads_y + 7) / 8, 1);

	// What this dispatch wrote, the next one reads.
	VkMemoryBarrier const barrier = {
		.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
	};
	s->vk->vkCmdPipelineBarrier(cmd_list, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr,
	                            0, nullptr);
	return true;
}
