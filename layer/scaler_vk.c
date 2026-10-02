/** @file
 *
 * OptiScaler's output-scaling pass: the bicubic enlarge and the seven averages.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "log.h"
#include "scaler_vk.h"
#include "scaling/bcds_bicubic_Shader_Vk.h"
#include "scaling/bcds_catmull_Shader_Vk.h"
#include "scaling/bcds_kaiser2_Shader_Vk.h"
#include "scaling/bcds_kaiser3_Shader_Vk.h"
#include "scaling/bcds_lanczos2_Shader_Vk.h"
#include "scaling/bcds_lanczos3_Shader_Vk.h"
#include "scaling/bcds_magc_Shader_Vk.h"
#include "scaling/bcus_Shader_Vk.h"
#include "shader_vk_priv.h"

/** @brief What the scaling shaders read.
 *
 * Upstream's Constants, spelled here so the layer does not have to carry the 33 KB of HLSL source
 * strings that sit beside it in OS_Common.h for the D3D11 path. The shaders read the sizes as int,
 * which holds the same bits for every Vulkan extent. The rest of the 256-byte block is a member, not
 * padding, so that an initializer zeroes it.
 */
struct scaler_constants {
	uint32_t src_width;   //!< The source's width.
	uint32_t src_height;  //!< The source's height.
	uint32_t dest_width;  //!< The destination's width.
	uint32_t dest_height; //!< The destination's height.
	uint32_t unused[60];  //!< The rest of the block, which the shaders do not read.
};

static_assert(sizeof (struct scaler_constants) == 256, "a constant slot holds one 256-byte block");

/** @brief A scaling shader, and what a pass that runs it is called and records. */
struct scaler_shader {
	unsigned char const *spv;   //!< Its SPIR-V.
	size_t               size;  //!< The SPIR-V's size in bytes.
	char const          *name;  //!< What the log calls its filter.
	char const          *pass;  //!< What the pass's log lines call the pass.
	uint64_t             flags; //!< The pass's enum scaler_vk_flags.
};

/** @brief An average's entry: its SPIR-V and the name of its filter. */
#define AVERAGE(spv, name) { spv, sizeof spv, name, "dlssnr-average", 0 }

/** @brief The averages, by enum scaler_vk_filter value. FSR1 falls back to Lanczos3. */
static struct scaler_shader const SCALER_VK_AVERAGES[SCALER_VK_COUNT] = {
	[SCALER_VK_FSR1]        = AVERAGE(bcds_lanczos3_spv, "lanczos3"),
	[SCALER_VK_BICUBIC]     = AVERAGE(bcds_bicubic_spv,  "bicubic"),
	[SCALER_VK_CATMULL_ROM] = AVERAGE(bcds_catmull_spv,  "catmull-rom"),
	[SCALER_VK_LANCZOS2]    = AVERAGE(bcds_lanczos2_spv, "lanczos2"),
	[SCALER_VK_LANCZOS3]    = AVERAGE(bcds_lanczos3_spv, "lanczos3"),
	[SCALER_VK_KAISER2]     = AVERAGE(bcds_kaiser2_spv,  "kaiser2"),
	[SCALER_VK_KAISER3]     = AVERAGE(bcds_kaiser3_spv,  "kaiser3"),
	[SCALER_VK_MAGIC]       = AVERAGE(bcds_magc_spv,     "magic")
};

#undef AVERAGE

/** @brief The enlarge. */
static struct scaler_shader const SCALER_VK_ENLARGE = {
	bcus_spv, sizeof bcus_spv, "bicubic enlarge", "dlssnr-enlarge", SCALER_VK_UPSAMPLE
};

/** @brief The scaling shaders' bindings. */
static VkDescriptorSetLayoutBinding const SCALER_VK_BINDINGS[] = {
	SHADER_VK_BINDING(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
	SHADER_VK_BINDING(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
	SHADER_VK_BINDING(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
	SHADER_VK_BINDING(3, VK_DESCRIPTOR_TYPE_SAMPLER)
};

/** @brief The number of bindings. */
static constexpr uint32_t SCALER_VK_BINDING_COUNT =
	sizeof SCALER_VK_BINDINGS / sizeof *SCALER_VK_BINDINGS;

/** @brief The descriptors of every slot's set. */
static VkDescriptorPoolSize const SCALER_VK_POOL_SIZES[] = {
	{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SCALER_VK_SLOTS },
	{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  SCALER_VK_SLOTS },
	{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  SCALER_VK_SLOTS },
	{ VK_DESCRIPTOR_TYPE_SAMPLER,        SCALER_VK_SLOTS }
};

/** @brief The number of pool sizes. */
static constexpr uint32_t SCALER_VK_POOL_SIZE_COUNT =
	sizeof SCALER_VK_POOL_SIZES / sizeof *SCALER_VK_POOL_SIZES;

/** @brief The average that a filter value selects.
 *
 * @param filter An enum scaler_vk_filter value.
 * @return       Its average; Lanczos3 for any value from SCALER_VK_COUNT on.
 */
static struct scaler_shader const *
average (uint32_t filter)
{
	return &SCALER_VK_AVERAGES[filter < SCALER_VK_COUNT ? filter : SCALER_VK_LANCZOS3];
}

char const *
scaler_vk_filter_name (uint32_t filter)
{
	return average(filter)->name;
}

/** @brief Builds a scaling pass.
 *
 * @param p      A pass whose base knows its device and owns nothing.
 * @param shader The pass's shader.
 * @return       VK_SUCCESS, or the first failure.
 */
static VkResult
build (struct scaler_vk           *p,
       struct scaler_shader const *shader)
{
	struct shader_vk *const s = &p->shader;
	if (!s->device)
		return VK_ERROR_INITIALIZATION_FAILED;

	VkResult r = shader_vk_create_sampler(s, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	if (r != VK_SUCCESS)
		return r;

	VkPhysicalDeviceProperties props;
	s->instance->vkGetPhysicalDeviceProperties(s->physical_device, &props);
	VkDeviceSize const align = props.limits.minUniformBufferOffsetAlignment;
	VkDeviceSize const alignment = align ? align : 1;
	p->slot_stride = (sizeof (struct scaler_constants) + alignment - 1) / alignment * alignment;

	r = shader_vk_create_buffer_resource(s, p->slot_stride * SCALER_VK_SLOTS,
	                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
	                                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	if (r != VK_SUCCESS)
		return r;

	void *mapped;
	r = s->vk->vkMapMemory(s->device, s->constant_buffer_memory, 0, p->slot_stride * SCALER_VK_SLOTS,
	                       0, &mapped);
	if (r != VK_SUCCESS)
		return r;
	s->mapped_constant_buffer = mapped;

	r = shader_vk_create_layouts(s, SCALER_VK_BINDINGS, SCALER_VK_BINDING_COUNT);
	if (r != VK_SUCCESS)
		return r;

	r = shader_vk_create_descriptor_pool(s, SCALER_VK_POOL_SIZES, SCALER_VK_POOL_SIZE_COUNT,
	                                     SCALER_VK_SLOTS);
	if (r != VK_SUCCESS)
		return r;

	r = shader_vk_create_descriptor_sets(s, SCALER_VK_SLOTS, p->descriptor_sets);
	if (r != VK_SUCCESS)
		return r;

	r = shader_vk_create_compute_pipeline(s, shader->spv, shader->size, "CSMain");
	if (r != VK_SUCCESS)
		return r;

	log_printf("[scaler] %s ready, %s", s->name, shader->name);
	return VK_SUCCESS;
}

struct scaler_vk
scaler_vk (struct device_table const   *vk,
           struct instance_table const *instance,
           VkDevice                     device,
           VkPhysicalDevice             physical_device,
           bool                         upsample,
           uint32_t                     filter)
{
	struct scaler_shader const *const shader = upsample ? &SCALER_VK_ENLARGE : average(filter);
	struct scaler_vk ret = {
		.shader = shader_vk(shader->pass, vk, instance, device, physical_device),
		.flags  = shader->flags
	};
	ret.error = build(&ret, shader);
	return ret;
}

VkResult
scaler_vk_init (struct scaler_vk            *dest,
                struct device_table const   *vk,
                struct instance_table const *instance,
                VkDevice                     device,
                VkPhysicalDevice             physical_device,
                bool                         upsample,
                uint32_t                     filter)
{
	if (!dest)
		return VK_ERROR_INITIALIZATION_FAILED;
	*dest = scaler_vk(vk, instance, device, physical_device, upsample, filter);
	return dest->error;
}

void
scaler_vk_fini (struct scaler_vk *dest)
{
	if (!dest)
		return;

	shader_vk_fini(&dest->shader);
	*dest = (struct scaler_vk){0};
}

bool
scaler_vk_dispatch (struct scaler_vk *p,
                    VkCommandBuffer   cb,
                    VkImageView       source,
                    VkImageView       dest,
                    uint32_t          src_width,
                    uint32_t          src_height,
                    uint32_t          dest_width,
                    uint32_t          dest_height)
{
	if (!p || !p->shader.pipeline || !cb || !source || !dest)
		return false;

	struct shader_vk const *const s = &p->shader;
	uint32_t const slot = p->slot;
	p->slot = (slot + 1) % SCALER_VK_SLOTS;
	VkDeviceSize const offset = p->slot_stride * slot;

	// From the images this call was handed, not from a global that happens to agree most of the
	// time. The initializer zeroes the rest of the block, all of which is copied.
	struct scaler_constants const c = {
		.src_width   = src_width,
		.src_height  = src_height,
		.dest_width  = dest_width,
		.dest_height = dest_height
	};
	memcpy((unsigned char *)s->mapped_constant_buffer + offset, &c, sizeof c);

	VkDescriptorSet const set = p->descriptor_sets[slot];
	VkDescriptorBufferInfo const buffer_info = { s->constant_buffer, offset, sizeof c };
	// Bindings 1 to 3, in order.
	VkDescriptorImageInfo const image_info[SCALER_VK_BINDING_COUNT - 1] = {
		{ VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
		{ VK_NULL_HANDLE, dest, VK_IMAGE_LAYOUT_GENERAL },
		{ s->texture_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED }
	};

	VkWriteDescriptorSet writes[SCALER_VK_BINDING_COUNT] = {
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = set,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo     = &buffer_info
		}
	};
	for (uint32_t i = 1; i < SCALER_VK_BINDING_COUNT; ++i) {
		writes[i] = (VkWriteDescriptorSet){
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = set,
			.dstBinding      = i,
			.descriptorCount = 1,
			.descriptorType  = SCALER_VK_BINDINGS[i].descriptorType,
			.pImageInfo      = &image_info[i - 1]
		};
	}
	s->vk->vkUpdateDescriptorSets(s->device, SCALER_VK_BINDING_COUNT, writes, 0, nullptr);

	s->vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
	s->vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_layout, 0, 1, &set,
	                               0, nullptr);

	// The enlarge shader's group is 16x16 and every averaging shader's is 8x8, which is what the
	// vendored modules declare and what upstream dispatches them at.
	uint32_t const tile = (p->flags & SCALER_VK_UPSAMPLE) ? 16 : 8;
	s->vk->vkCmdDispatch(cb, (dest_width + tile - 1) / tile, (dest_height + tile - 1) / tile, 1);

	VkMemoryBarrier const barrier = {
		.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
	};
	s->vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
	                            nullptr);
	return true;
}
