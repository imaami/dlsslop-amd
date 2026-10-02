/** @file
 *
 * OptiScaler's Shader_Vk: the objects that every pass builds, and its layout transition.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "shader_vk_priv.h"

struct shader_vk
shader_vk (char const                  *name,
           struct device_table const   *vk,
           struct instance_table const *instance,
           VkDevice                     device,
           VkPhysicalDevice             physical_device)
{
	return (struct shader_vk){
		.name            = name,
		.vk              = vk,
		.instance        = instance,
		.device          = device,
		.physical_device = physical_device
	};
}

void
shader_vk_fini (struct shader_vk *dest)
{
	if (!dest)
		return;

	struct device_table const *const vk = dest->vk;
	VkDevice const device = dest->device;
	if (dest->pipeline)
		vk->vkDestroyPipeline(device, dest->pipeline, nullptr);
	if (dest->descriptor_pool)
		vk->vkDestroyDescriptorPool(device, dest->descriptor_pool, nullptr);
	if (dest->descriptor_set_layout)
		vk->vkDestroyDescriptorSetLayout(device, dest->descriptor_set_layout, nullptr);
	if (dest->pipeline_layout)
		vk->vkDestroyPipelineLayout(device, dest->pipeline_layout, nullptr);
	if (dest->mapped_constant_buffer && vk->vkUnmapMemory)
		vk->vkUnmapMemory(device, dest->constant_buffer_memory);
	if (dest->constant_buffer)
		vk->vkDestroyBuffer(device, dest->constant_buffer, nullptr);
	if (dest->constant_buffer_memory)
		vk->vkFreeMemory(device, dest->constant_buffer_memory, nullptr);
	if (dest->texture_sampler)
		vk->vkDestroySampler(device, dest->texture_sampler, nullptr);

	*dest = (struct shader_vk){0};
}

uint32_t
shader_vk_find_memory_type (struct shader_vk const *s,
                            uint32_t                type_filter,
                            VkMemoryPropertyFlags   properties)
{
	VkPhysicalDeviceMemoryProperties memory;
	s->instance->vkGetPhysicalDeviceMemoryProperties(s->physical_device, &memory);

	for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
		if ((type_filter & (1u << i))
		    && (memory.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}

	log_printf("[%s] no memory type with %#x", s->name, (unsigned)properties);
	return UINT32_MAX;
}

VkResult
shader_vk_create_compute_pipeline (struct shader_vk    *s,
                                   unsigned char const *code,
                                   size_t               size,
                                   char const          *entry_point)
{
	uint32_t *words = malloc(size);
	if (!words) {
		log_printf("[%s] could not copy %zu bytes of SPIR-V", s->name, size);
		return VK_ERROR_OUT_OF_HOST_MEMORY;
	}

	memcpy(words, code, size);
	VkShaderModuleCreateInfo const module_info = {
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode    = words
	};
	VkShaderModule module = VK_NULL_HANDLE;
	VkResult r = s->vk->vkCreateShaderModule(s->device, &module_info, nullptr, &module);
	free(words);
	words = nullptr;
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreateShaderModule failed", s->name);
		return r;
	}

	VkComputePipelineCreateInfo const pipeline_info = {
		.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage  = {
			.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage  = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = module,
			.pName  = entry_point
		},
		.layout = s->pipeline_layout
	};
	VkPipeline pipeline;
	r = s->vk->vkCreateComputePipelines(s->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
	                                    &pipeline);
	s->vk->vkDestroyShaderModule(s->device, module, nullptr);

	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreateComputePipelines -> %d", s->name, (int)r);
		return r;
	}
	s->pipeline = pipeline;
	return VK_SUCCESS;
}

VkResult
shader_vk_create_buffer_resource (struct shader_vk      *s,
                                  VkDeviceSize           size,
                                  VkBufferUsageFlags     usage,
                                  VkMemoryPropertyFlags  properties)
{
	struct device_table const *const vk = s->vk;
	VkBufferCreateInfo const buffer_info = {
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = size,
		.usage       = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};
	VkBuffer buffer;
	VkResult r = vk->vkCreateBuffer(s->device, &buffer_info, nullptr, &buffer);
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreateBuffer failed", s->name);
		return r;
	}
	s->constant_buffer = buffer;

	VkMemoryRequirements requirements;
	vk->vkGetBufferMemoryRequirements(s->device, s->constant_buffer, &requirements);

	uint32_t const type = shader_vk_find_memory_type(s, requirements.memoryTypeBits, properties);
	if (type == UINT32_MAX)
		return VK_ERROR_FEATURE_NOT_PRESENT;

	VkMemoryAllocateInfo const allocate_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = requirements.size,
		.memoryTypeIndex = type
	};
	VkDeviceMemory memory;
	r = vk->vkAllocateMemory(s->device, &allocate_info, nullptr, &memory);
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkAllocateMemory failed (%llu bytes)", s->name,
		           (unsigned long long)requirements.size);
		return r;
	}
	s->constant_buffer_memory = memory;

	r = vk->vkBindBufferMemory(s->device, s->constant_buffer, s->constant_buffer_memory, 0);
	if (r != VK_SUCCESS)
		log_printf("[%s] vkBindBufferMemory failed", s->name);
	return r;
}

VkResult
shader_vk_create_layouts (struct shader_vk                   *s,
                          VkDescriptorSetLayoutBinding const *bindings,
                          uint32_t                            count)
{
	VkDescriptorSetLayoutCreateInfo const layout_info = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = count,
		.pBindings    = bindings
	};
	VkDescriptorSetLayout set_layout;
	VkResult r = s->vk->vkCreateDescriptorSetLayout(s->device, &layout_info, nullptr, &set_layout);
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreateDescriptorSetLayout failed", s->name);
		return r;
	}
	s->descriptor_set_layout = set_layout;

	VkPipelineLayoutCreateInfo const pipeline_layout_info = {
		.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts    = &s->descriptor_set_layout
	};
	VkPipelineLayout pipeline_layout;
	r = s->vk->vkCreatePipelineLayout(s->device, &pipeline_layout_info, nullptr, &pipeline_layout);
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreatePipelineLayout failed", s->name);
		return r;
	}
	s->pipeline_layout = pipeline_layout;
	return VK_SUCCESS;
}

VkResult
shader_vk_create_descriptor_pool (struct shader_vk           *s,
                                  VkDescriptorPoolSize const *pool_sizes,
                                  uint32_t                    count,
                                  uint32_t                    max_sets)
{
	VkDescriptorPoolCreateInfo const pool_info = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets       = max_sets,
		.poolSizeCount = count,
		.pPoolSizes    = pool_sizes
	};
	VkDescriptorPool pool;
	VkResult const r = s->vk->vkCreateDescriptorPool(s->device, &pool_info, nullptr, &pool);
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreateDescriptorPool failed", s->name);
		return r;
	}
	s->descriptor_pool = pool;
	return VK_SUCCESS;
}

VkResult
shader_vk_create_descriptor_sets (struct shader_vk const *s,
                                  uint32_t                count,
                                  VkDescriptorSet        *sets)
{
	// A fixed array: a variable one is undefined for no sets and unbounded on the stack.
	if (!count || count > SHADER_VK_SETS_MAX) {
		log_printf("[%s] %u descriptor sets asked for, not 1 to %u", s->name, count, SHADER_VK_SETS_MAX);
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	VkDescriptorSetLayout layouts[SHADER_VK_SETS_MAX];
	for (uint32_t i = 0; i < count; ++i)
		layouts[i] = s->descriptor_set_layout;

	VkDescriptorSetAllocateInfo const allocate_info = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool     = s->descriptor_pool,
		.descriptorSetCount = count,
		.pSetLayouts        = layouts
	};
	VkResult const r = s->vk->vkAllocateDescriptorSets(s->device, &allocate_info, sets);
	if (r != VK_SUCCESS)
		log_printf("[%s] vkAllocateDescriptorSets failed (%u sets)", s->name, count);
	return r;
}

VkResult
shader_vk_create_sampler (struct shader_vk     *s,
                          VkFilter              filter,
                          VkSamplerAddressMode  address_mode)
{
	VkSamplerCreateInfo const sampler_info = {
		.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter               = filter,
		.minFilter               = filter,
		.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU            = address_mode,
		.addressModeV            = address_mode,
		.addressModeW            = address_mode,
		.anisotropyEnable        = VK_FALSE,
		.maxAnisotropy           = 1.0f,
		.compareEnable           = VK_FALSE,
		.compareOp               = VK_COMPARE_OP_ALWAYS,
		.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE
	};
	VkSampler sampler;
	VkResult const r = s->vk->vkCreateSampler(s->device, &sampler_info, nullptr, &sampler);
	if (r != VK_SUCCESS) {
		log_printf("[%s] vkCreateSampler failed", s->name);
		return r;
	}
	s->texture_sampler = sampler;
	return VK_SUCCESS;
}

void
shader_vk_set_image_layout (struct shader_vk const  *s,
                            VkCommandBuffer          cmd_buffer,
                            VkImage                  image,
                            VkImageLayout            old_layout,
                            VkImageLayout            new_layout,
                            VkImageSubresourceRange  subresource_range)
{
	if (!s || !s->vk)
		return;

	VkImageMemoryBarrier barrier = {
		.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
		.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
		.oldLayout           = old_layout,
		.newLayout           = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = image,
		.subresourceRange    = subresource_range
	};

	VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkPipelineStageFlags destination_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
		barrier.srcAccessMask = 0;
		source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	} else if (old_layout == VK_IMAGE_LAYOUT_GENERAL) {
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		source_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}

	if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	} else if (new_layout == VK_IMAGE_LAYOUT_GENERAL) {
		barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}

	s->vk->vkCmdPipelineBarrier(cmd_buffer, source_stage, destination_stage, 0, 0, nullptr, 0,
	                            nullptr, 1, &barrier);
}
