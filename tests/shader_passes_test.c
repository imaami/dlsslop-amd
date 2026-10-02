/** @file
 *
 * The composition and scaling passes on the host, on a fake device: what a build creates and what
 * the fini functions destroy, after a failure at each step of a build as well as after a whole one;
 * that zeroed, finished and unbuilt passes own nothing; the stride of the constant ring; what a
 * dispatch writes into its slot and its descriptor set, the groups it dispatches and the dispatches
 * it refuses; and the downscalers' names. What the shaders compute is hdr-shader-test's and
 * composition-rebuild-test's, on a real device.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dlssnr_pass.h"
#include "scaler_vk.h"
#include "shader_vk_priv.h"

// The SPIR-V that the passes must run, under names of the test's own: the passes' objects define the
// headers' names.
#define bcds_bicubic_spv  expected_bicubic
#define bcds_catmull_spv  expected_catmull
#define bcds_kaiser2_spv  expected_kaiser2
#define bcds_kaiser3_spv  expected_kaiser3
#define bcds_lanczos2_spv expected_lanczos2
#define bcds_lanczos3_spv expected_lanczos3
#define bcds_magc_spv     expected_magc
#define bcus_spv          expected_bcus
#define dlssnr_spv        expected_dlssnr
#include "dlssnr/DlssNr_Shader_Vk.h"
#include "scaling/bcds_bicubic_Shader_Vk.h"
#include "scaling/bcds_catmull_Shader_Vk.h"
#include "scaling/bcds_kaiser2_Shader_Vk.h"
#include "scaling/bcds_kaiser3_Shader_Vk.h"
#include "scaling/bcds_lanczos2_Shader_Vk.h"
#include "scaling/bcds_lanczos3_Shader_Vk.h"
#include "scaling/bcds_magc_Shader_Vk.h"
#include "scaling/bcus_Shader_Vk.h"
#undef dlssnr_spv
#undef bcus_spv
#undef bcds_magc_spv
#undef bcds_lanczos3_spv
#undef bcds_lanczos2_spv
#undef bcds_kaiser3_spv
#undef bcds_kaiser2_spv
#undef bcds_catmull_spv
#undef bcds_bicubic_spv

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("shader-passes-test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief The kinds of object that the fake device hands out. */
enum kind {
	KIND_SAMPLER,
	KIND_BUFFER,
	KIND_MEMORY,
	KIND_SET_LAYOUT,
	KIND_PIPELINE_LAYOUT,
	KIND_POOL,
	KIND_SET,
	KIND_MODULE,
	KIND_PIPELINE,
	KIND_IMAGE,
	KIND_VIEW,
	KIND_COUNT
};

/** @brief One object of the fake device. */
struct object {
	uint64_t  handle; //!< Its handle.
	enum kind kind;   //!< What it is.
	uint32_t  live;   //!< Whether it is alive: 32 bits wide, which fills the padding.
};

/** @brief The fake device's state. */
static struct {
	struct object          objects[4096];      //!< Every object handed out, in order.
	VkWriteDescriptorSet   writes[8];          //!< The last vkUpdateDescriptorSets()'s writes.
	VkDescriptorImageInfo  images[8];          //!< Their image infos, by write.
	VkDescriptorBufferInfo buffer;             //!< Their buffer info.
	VkImageMemoryBarrier   last_image_barrier; //!< The last image barrier recorded.
	VkDeviceSize           alignment;          //!< minUniformBufferOffsetAlignment.
	VkPipeline             bound_pipeline;     //!< The pipeline bound last.
	VkDescriptorSet        bound_set;          //!< The descriptor set bound last.
	uint64_t               module_hash;        //!< The FNV-1a hash of the last shader module's code.
	size_t                 module_size;        //!< The last shader module's size in bytes.
	uint32_t               count;              //!< Objects handed out.
	uint32_t               creates;            //!< Fallible calls so far.
	uint32_t               fail_at;            //!< The fallible call that fails, from 1; 0: none.
	uint32_t               calls;              //!< Every call of the device table.
	uint32_t               mapped;             //!< Memory mapped and not unmapped.
	uint32_t               write_count;        //!< The last vkUpdateDescriptorSets()'s writes.
	uint32_t               dispatches;         //!< Dispatches recorded.
	uint32_t               groups[3];          //!< The last dispatch's groups.
	uint32_t               barriers;           //!< Pipeline barriers recorded.
	uint32_t               image_barriers;     //!< Those of them with an image barrier.
	uint32_t               host_visible;       //!< Whether a host-visible memory type exists.
	uint32_t               device_local;       //!< Whether a device-local memory type exists.
	unsigned char          ring[1 << 16];      //!< What vkMapMemory() maps.
} fake;

/** @brief The FNV-1a hash of some bytes. */
static uint64_t
fnv1a (void const *p,
       size_t      size)
{
	unsigned char const *const bytes = p;
	uint64_t hash = 0xcbf29ce484222325;
	for (size_t i = 0; i < size; ++i)
		hash = (hash ^ bytes[i]) * 0x100000001b3;
	return hash;
}

/** @brief Hands out an object of a kind. */
static uint64_t
make (enum kind kind)
{
	require(fake.count < sizeof fake.objects / sizeof *fake.objects, "too many objects");
	uint64_t const handle = 0x1000 + fake.count;
	fake.objects[fake.count++] = (struct object){ handle, kind, true };
	return handle;
}

/** @brief Ends an object's life; it must be alive and of the kind. */
static void
end (uint64_t  handle,
     enum kind kind)
{
	require(handle, "a destroy of VK_NULL_HANDLE (kind %d)", kind);
	for (uint32_t i = 0; i < fake.count; ++i) {
		struct object *const o = &fake.objects[i];
		if (o->handle != handle)
			continue;
		require(o->kind == kind, "object %#llx destroyed as kind %d, made as %d",
		        (unsigned long long)handle, kind, o->kind);
		require(o->live, "object %#llx destroyed twice", (unsigned long long)handle);
		o->live = false;
		return;
	}
	require(false, "object %#llx was never made", (unsigned long long)handle);
}

/** @brief Whether an object is alive. */
static bool
alive (uint64_t handle)
{
	for (uint32_t i = 0; i < fake.count; ++i)
		if (fake.objects[i].handle == handle)
			return fake.objects[i].live;
	return false;
}

/** @brief The objects that are alive, sets aside: the pool frees those. */
static uint32_t
live_objects (void)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < fake.count; ++i)
		n += fake.objects[i].live && fake.objects[i].kind != KIND_SET;
	return n;
}

/** @brief Counts a fallible call and says whether it fails. */
static bool
fails (void)
{
	return ++fake.creates == fake.fail_at;
}

/** @brief Starts a scenario: no objects, no failure. */
static void
reset (void)
{
	memset(&fake, 0, sizeof fake);
	fake.host_visible = true;
	fake.device_local = true;
	fake.alignment = 64;
}

/** @brief What a fallible call that fails returns. */
static constexpr VkResult FAILURE = VK_ERROR_OUT_OF_DEVICE_MEMORY;

/** @brief What a failed call leaves in its output handle: a value that it never made, which the
 *         passes must not destroy. Vulkan leaves the output undefined after most failures. */
static constexpr uint64_t POISON = 0xdead0000;

/** @brief A handle of a type with a value. */
#define HANDLE(type, value) ((type)(uintptr_t)(value))

/** @brief The body of a fake create call: it fails, leaving POISON, or hands out an object. */
#define CREATE(type, kind, out) \
	do { \
		++fake.calls; \
		if (fails()) { \
			*(out) = HANDLE(type, POISON); \
			return FAILURE; \
		} \
		*(out) = HANDLE(type, make(kind)); \
		return VK_SUCCESS; \
	} while (0)
/** @brief The body of a fake destroy call: it ends the object's life. */
#define DESTROY(handle, kind) \
	do { \
		++fake.calls; \
		end((uint64_t)(uintptr_t)(handle), (kind)); \
	} while (0)

/** @brief The fake vkCreateSampler(): requires upstream's sampler, and hands one out. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_sampler (VkDevice                     d,
                VkSamplerCreateInfo const   *i,
                VkAllocationCallbacks const *a,
                VkSampler                   *out)
{
	require(i->magFilter == VK_FILTER_LINEAR && i->minFilter == VK_FILTER_LINEAR
	        && i->addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
	        && i->addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
	        && i->addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
	        && i->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR && i->maxAnisotropy == 1.0f
	        && i->compareOp == VK_COMPARE_OP_ALWAYS
	        && i->borderColor == VK_BORDER_COLOR_INT_OPAQUE_BLACK, "a sampler unlike upstream's");
	CREATE(VkSampler, KIND_SAMPLER, out);
}

/** @brief The fake vkDestroySampler(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_sampler (VkDevice                     d,
                 VkSampler                    s,
                 VkAllocationCallbacks const *a)
{
	DESTROY(s, KIND_SAMPLER);
}

/** @brief The fake vkCreateBuffer(). */
static VKAPI_ATTR VkResult VKAPI_CALL
create_buffer (VkDevice                     d,
               VkBufferCreateInfo const    *i,
               VkAllocationCallbacks const *a,
               VkBuffer                    *out)
{
	CREATE(VkBuffer, KIND_BUFFER, out);
}

/** @brief The fake vkDestroyBuffer(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_buffer (VkDevice                     d,
                VkBuffer                     b,
                VkAllocationCallbacks const *a)
{
	DESTROY(b, KIND_BUFFER);
}

/** @brief The fake vkGetBufferMemoryRequirements(): 4096 bytes, of either memory type. */
static VKAPI_ATTR void VKAPI_CALL
buffer_requirements (VkDevice              d,
                     VkBuffer              b,
                     VkMemoryRequirements *r)
{
	++fake.calls;
	*r = (VkMemoryRequirements){ .size = 4096, .alignment = 256, .memoryTypeBits = 3 };
}

/** @brief The fake vkGetImageMemoryRequirements(): 256 bytes, of either memory type. */
static VKAPI_ATTR void VKAPI_CALL
image_requirements (VkDevice              d,
                    VkImage               i,
                    VkMemoryRequirements *r)
{
	++fake.calls;
	*r = (VkMemoryRequirements){ .size = 256, .alignment = 256, .memoryTypeBits = 3 };
}

/** @brief The fake vkAllocateMemory(). */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_memory (VkDevice                     d,
                 VkMemoryAllocateInfo const  *i,
                 VkAllocationCallbacks const *a,
                 VkDeviceMemory              *out)
{
	CREATE(VkDeviceMemory, KIND_MEMORY, out);
}

/** @brief The fake vkFreeMemory(). */
static VKAPI_ATTR void VKAPI_CALL
free_memory (VkDevice                     d,
             VkDeviceMemory               m,
             VkAllocationCallbacks const *a)
{
	DESTROY(m, KIND_MEMORY);
}

/** @brief The fake vkBindBufferMemory(): both objects must be alive. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_buffer_memory (VkDevice       d,
                    VkBuffer       b,
                    VkDeviceMemory m,
                    VkDeviceSize   o)
{
	++fake.calls;
	require(alive((uint64_t)(uintptr_t)b) && alive((uint64_t)(uintptr_t)m), "a bind of a dead object");
	return fails() ? FAILURE : VK_SUCCESS;
}

/** @brief The fake vkBindImageMemory(): both objects must be alive. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_image_memory (VkDevice       d,
                   VkImage        i,
                   VkDeviceMemory m,
                   VkDeviceSize   o)
{
	++fake.calls;
	require(alive((uint64_t)(uintptr_t)i) && alive((uint64_t)(uintptr_t)m), "a bind of a dead object");
	return fails() ? FAILURE : VK_SUCCESS;
}

/** @brief The fake vkMapMemory(): maps the ring. */
static VKAPI_ATTR VkResult VKAPI_CALL
map_memory (VkDevice         d,
            VkDeviceMemory   m,
            VkDeviceSize     o,
            VkDeviceSize     size,
            VkMemoryMapFlags f,
            void           **out)
{
	++fake.calls;
	require(size <= sizeof fake.ring, "a map of %llu bytes", (unsigned long long)size);
	if (fails()) {
		*out = (void *)(uintptr_t)POISON;
		return FAILURE;
	}
	++fake.mapped;
	*out = fake.ring;
	return VK_SUCCESS;
}

/** @brief The fake vkUnmapMemory(): the memory must be mapped. */
static VKAPI_ATTR void VKAPI_CALL
unmap_memory (VkDevice       d,
              VkDeviceMemory m)
{
	++fake.calls;
	require(fake.mapped, "an unmap of memory that is not mapped");
	--fake.mapped;
}

/** @brief The fake vkCreateDescriptorSetLayout(): requires bindings as SHADER_VK_BINDING makes them. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_set_layout (VkDevice                               d,
                   VkDescriptorSetLayoutCreateInfo const *i,
                   VkAllocationCallbacks const           *a,
                   VkDescriptorSetLayout                 *out)
{
	for (uint32_t b = 0; b < i->bindingCount; ++b)
		require(i->pBindings[b].binding == b && i->pBindings[b].descriptorCount == 1
		        && i->pBindings[b].stageFlags == VK_SHADER_STAGE_COMPUTE_BIT
		        && !i->pBindings[b].pImmutableSamplers, "binding %u is not CreateBinding's", b);
	CREATE(VkDescriptorSetLayout, KIND_SET_LAYOUT, out);
}

/** @brief The fake vkDestroyDescriptorSetLayout(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_set_layout (VkDevice                     d,
                    VkDescriptorSetLayout        l,
                    VkAllocationCallbacks const *a)
{
	DESTROY(l, KIND_SET_LAYOUT);
}

/** @brief The fake vkCreatePipelineLayout(): requires one live set layout. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_pipeline_layout (VkDevice                          d,
                        VkPipelineLayoutCreateInfo const *i,
                        VkAllocationCallbacks const      *a,
                        VkPipelineLayout                 *out)
{
	require(i->setLayoutCount == 1 && alive((uint64_t)(uintptr_t)i->pSetLayouts[0]), "a pipeline layout"
	        " without the set layout");
	CREATE(VkPipelineLayout, KIND_PIPELINE_LAYOUT, out);
}

/** @brief The fake vkDestroyPipelineLayout(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_pipeline_layout (VkDevice                     d,
                         VkPipelineLayout             l,
                         VkAllocationCallbacks const *a)
{
	DESTROY(l, KIND_PIPELINE_LAYOUT);
}

/** @brief The fake vkCreateDescriptorPool(). */
static VKAPI_ATTR VkResult VKAPI_CALL
create_pool (VkDevice                          d,
             VkDescriptorPoolCreateInfo const *i,
             VkAllocationCallbacks const      *a,
             VkDescriptorPool                 *out)
{
	CREATE(VkDescriptorPool, KIND_POOL, out);
}

/** @brief The fake vkDestroyDescriptorPool(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_pool (VkDevice                     d,
              VkDescriptorPool             p,
              VkAllocationCallbacks const *a)
{
	DESTROY(p, KIND_POOL);
}

/** @brief The fake vkAllocateDescriptorSets(): from a live pool, of live layouts. The pool frees them. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_sets (VkDevice                           d,
               VkDescriptorSetAllocateInfo const *i,
               VkDescriptorSet                   *out)
{
	++fake.calls;
	require(alive((uint64_t)(uintptr_t)i->descriptorPool), "sets from a dead pool");
	for (uint32_t s = 0; s < i->descriptorSetCount; ++s)
		require(alive((uint64_t)(uintptr_t)i->pSetLayouts[s]), "a set of a dead layout");
	if (fails()) {
		for (uint32_t s = 0; s < i->descriptorSetCount; ++s)
			out[s] = HANDLE(VkDescriptorSet, POISON);
		return FAILURE;
	}
	for (uint32_t s = 0; s < i->descriptorSetCount; ++s)
		out[s] = HANDLE(VkDescriptorSet, make(KIND_SET));
	return VK_SUCCESS;
}

/** @brief The fake vkCreateShaderModule(): requires aligned SPIR-V words, and records their size and
 *         hash. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_module (VkDevice                        d,
               VkShaderModuleCreateInfo const *i,
               VkAllocationCallbacks const    *a,
               VkShaderModule                 *out)
{
	require(!((uintptr_t)i->pCode & 3) && i->codeSize >= 20 && !(i->codeSize & 3)
	        && i->pCode[0] == 0x07230203, "SPIR-V that is not aligned words");
	fake.module_hash = fnv1a(i->pCode, i->codeSize);
	fake.module_size = i->codeSize;
	CREATE(VkShaderModule, KIND_MODULE, out);
}

/** @brief The fake vkDestroyShaderModule(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_module (VkDevice                     d,
                VkShaderModule               m,
                VkAllocationCallbacks const *a)
{
	DESTROY(m, KIND_MODULE);
}

/** @brief The fake vkCreateComputePipelines(): requires one pipeline of CSMain, of a live module and
 *         layout. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_pipelines (VkDevice                           d,
                  VkPipelineCache                    c,
                  uint32_t                           n,
                  VkComputePipelineCreateInfo const *i,
                  VkAllocationCallbacks const       *a,
                  VkPipeline                        *out)
{
	require(n == 1 && !strcmp(i->stage.pName, "CSMain") && alive((uint64_t)(uintptr_t)i->stage.module)
	        && alive((uint64_t)(uintptr_t)i->layout), "a pipeline unlike upstream's");
	CREATE(VkPipeline, KIND_PIPELINE, out);
}

/** @brief The fake vkDestroyPipeline(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_pipeline (VkDevice                     d,
                  VkPipeline                   p,
                  VkAllocationCallbacks const *a)
{
	DESTROY(p, KIND_PIPELINE);
}

/** @brief The fake vkCreateImage(): requires the placeholder's image. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_image (VkDevice                     d,
              VkImageCreateInfo const     *i,
              VkAllocationCallbacks const *a,
              VkImage                     *out)
{
	require(i->format == VK_FORMAT_R16G16B16A16_SFLOAT && i->extent.width == 1 && i->extent.height == 1
	        && i->usage == (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT), "a placeholder unlike"
	        " upstream's");
	CREATE(VkImage, KIND_IMAGE, out);
}

/** @brief The fake vkDestroyImage(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_image (VkDevice                     d,
               VkImage                      i,
               VkAllocationCallbacks const *a)
{
	DESTROY(i, KIND_IMAGE);
}

/** @brief The fake vkCreateImageView(). */
static VKAPI_ATTR VkResult VKAPI_CALL
create_view (VkDevice                     d,
             VkImageViewCreateInfo const *i,
             VkAllocationCallbacks const *a,
             VkImageView                 *out)
{
	CREATE(VkImageView, KIND_VIEW, out);
}

/** @brief The fake vkDestroyImageView(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_view (VkDevice                     d,
              VkImageView                  v,
              VkAllocationCallbacks const *a)
{
	DESTROY(v, KIND_VIEW);
}

/** @brief The fake vkUpdateDescriptorSets(): records the writes. */
static VKAPI_ATTR void VKAPI_CALL
update_sets (VkDevice                    d,
             uint32_t                    n,
             VkWriteDescriptorSet const *w,
             uint32_t                    copies,
             VkCopyDescriptorSet const  *c)
{
	++fake.calls;
	require(n <= 8 && !copies, "%u writes and %u copies", n, copies);
	fake.write_count = n;
	for (uint32_t i = 0; i < n; ++i) {
		fake.writes[i] = w[i];
		require(w[i].descriptorCount == 1 && !w[i].dstArrayElement, "write %u is not of one descriptor", i);
		if (w[i].pImageInfo)
			fake.images[i] = *w[i].pImageInfo;
		if (w[i].pBufferInfo)
			fake.buffer = *w[i].pBufferInfo;
	}
}

/** @brief The fake vkCmdBindPipeline(): records the pipeline. */
static VKAPI_ATTR void VKAPI_CALL
bind_pipeline (VkCommandBuffer     cb,
               VkPipelineBindPoint p,
               VkPipeline          pipeline)
{
	++fake.calls;
	require(p == VK_PIPELINE_BIND_POINT_COMPUTE, "a graphics bind");
	fake.bound_pipeline = pipeline;
}

/** @brief The fake vkCmdBindDescriptorSets(): records the set. */
static VKAPI_ATTR void VKAPI_CALL
bind_sets (VkCommandBuffer        cb,
           VkPipelineBindPoint    p,
           VkPipelineLayout       l,
           uint32_t               first,
           uint32_t               n,
           VkDescriptorSet const *sets,
           uint32_t               dynamic,
           uint32_t const        *offsets)
{
	++fake.calls;
	require(p == VK_PIPELINE_BIND_POINT_COMPUTE && !first && n == 1 && !dynamic, "a bind unlike upstream's");
	fake.bound_set = sets[0];
}

/** @brief The fake vkCmdDispatch(): records the groups. */
static VKAPI_ATTR void VKAPI_CALL
dispatch (VkCommandBuffer cb,
          uint32_t        x,
          uint32_t        y,
          uint32_t        z)
{
	++fake.calls;
	++fake.dispatches;
	fake.groups[0] = x;
	fake.groups[1] = y;
	fake.groups[2] = z;
}

/** @brief The fake vkCmdPipelineBarrier(): records an image barrier; a memory barrier must be
 *         upstream's after a dispatch. */
static VKAPI_ATTR void VKAPI_CALL
pipeline_barrier (VkCommandBuffer              cb,
                  VkPipelineStageFlags         src,
                  VkPipelineStageFlags         dst,
                  VkDependencyFlags            f,
                  uint32_t                     memory,
                  VkMemoryBarrier const       *m,
                  uint32_t                     buffers,
                  VkBufferMemoryBarrier const *b,
                  uint32_t                     images,
                  VkImageMemoryBarrier const  *i)
{
	++fake.calls;
	++fake.barriers;
	if (images) {
		++fake.image_barriers;
		fake.last_image_barrier = i[0];
	} else {
		require(memory == 1 && src == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		        && dst == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
		        && m->srcAccessMask == VK_ACCESS_SHADER_WRITE_BIT
		        && m->dstAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
		        "a barrier after the dispatch unlike upstream's");
	}
}

/** @brief The fake vkGetPhysicalDeviceProperties(): the scenario's offset alignment. */
static VKAPI_ATTR void VKAPI_CALL
physical_properties (VkPhysicalDevice            p,
                     VkPhysicalDeviceProperties *out)
{
	++fake.calls;
	memset(out, 0, sizeof *out);
	out->limits.minUniformBufferOffsetAlignment = fake.alignment;
}

/** @brief The fake vkGetPhysicalDeviceMemoryProperties(): a device-local type and a host-visible one,
 *         unless the scenario removes either. */
static VKAPI_ATTR void VKAPI_CALL
memory_properties (VkPhysicalDevice                  p,
                   VkPhysicalDeviceMemoryProperties *out)
{
	++fake.calls;
	memset(out, 0, sizeof *out);
	out->memoryTypeCount = 2;
	out->memoryTypes[0].propertyFlags = fake.device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0;
	out->memoryTypes[1].propertyFlags = fake.host_visible
	                                    ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	                                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

#undef DESTROY
#undef CREATE

/** @brief The fake device's table. */
static struct device_table device_table;

/** @brief The fake instance's table. */
static struct instance_table instance_table;

/** @brief The fake device. */
static VkDevice const DEVICE = HANDLE(VkDevice, 0x10);

/** @brief The fake device's physical device. */
static VkPhysicalDevice const PHYSICAL = HANDLE(VkPhysicalDevice, 0x20);

/** @brief The command buffer that the passes record into. */
static VkCommandBuffer const CMD = HANDLE(VkCommandBuffer, 0x30);

/** @brief Both directions of the supersampling: the average, then the enlarge. */
static bool const UPSAMPLE[] = { false, true };

/** @brief Fills the fake device's tables. */
static void
tables (void)
{
	device_table = (struct device_table){
		.vkCreateSampler               = create_sampler,
		.vkDestroySampler              = destroy_sampler,
		.vkCreateBuffer                = create_buffer,
		.vkDestroyBuffer               = destroy_buffer,
		.vkGetBufferMemoryRequirements = buffer_requirements,
		.vkGetImageMemoryRequirements  = image_requirements,
		.vkAllocateMemory              = allocate_memory,
		.vkFreeMemory                  = free_memory,
		.vkBindBufferMemory            = bind_buffer_memory,
		.vkBindImageMemory             = bind_image_memory,
		.vkMapMemory                   = map_memory,
		.vkUnmapMemory                 = unmap_memory,
		.vkCreateDescriptorSetLayout   = create_set_layout,
		.vkDestroyDescriptorSetLayout  = destroy_set_layout,
		.vkCreatePipelineLayout        = create_pipeline_layout,
		.vkDestroyPipelineLayout       = destroy_pipeline_layout,
		.vkCreateDescriptorPool        = create_pool,
		.vkDestroyDescriptorPool       = destroy_pool,
		.vkAllocateDescriptorSets      = allocate_sets,
		.vkUpdateDescriptorSets        = update_sets,
		.vkCreateShaderModule          = create_module,
		.vkDestroyShaderModule         = destroy_module,
		.vkCreateComputePipelines      = create_pipelines,
		.vkDestroyPipeline             = destroy_pipeline,
		.vkCreateImage                 = create_image,
		.vkDestroyImage                = destroy_image,
		.vkCreateImageView             = create_view,
		.vkDestroyImageView            = destroy_view,
		.vkCmdBindPipeline             = bind_pipeline,
		.vkCmdBindDescriptorSets       = bind_sets,
		.vkCmdDispatch                 = dispatch,
		.vkCmdPipelineBarrier          = pipeline_barrier
	};
	instance_table = (struct instance_table){
		.vkGetPhysicalDeviceProperties       = physical_properties,
		.vkGetPhysicalDeviceMemoryProperties = memory_properties
	};
}

/** @brief Whether every byte of an object is zero. */
static bool
zeroed (void const *p,
        size_t      size)
{
	unsigned char const *const bytes = p;
	for (size_t i = 0; i < size; ++i)
		if (bytes[i])
			return false;
	return true;
}

/** @brief Builds and finishes a composition pass with each fallible call failing in turn. */
static void
check_pass_failures (void)
{
	reset();
	struct dlss_nr_pass pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, PHYSICAL);
	require(pass.error == VK_SUCCESS && pass.shader.pipeline,
	        "the composition pass was not built");
	uint32_t const steps = fake.creates;
	require(steps == 11, "the composition pass's build made %u fallible calls, not 11", steps);
	// The sampler, the ring's buffer and memory, two layouts, the pool and the pipeline.
	require(live_objects() == 7 && fake.mapped == 1, "a built composition pass holds %u objects and %u maps,"
	        " not 7 and 1", live_objects(), fake.mapped);
	dlss_nr_pass_fini(&pass);
	require(!live_objects() && !fake.mapped && zeroed(&pass, sizeof pass), "the composition pass's fini left"
	        " %u objects and %u maps", live_objects(), fake.mapped);

	for (uint32_t k = 1; k <= steps; ++k) {
		reset();
		fake.fail_at = k;
		VkResult const r = dlss_nr_pass_init(&pass, &device_table, &instance_table, DEVICE, PHYSICAL);
		require(r == FAILURE && pass.error == FAILURE && !pass.shader.pipeline,
		        "the composition pass's build with call %u failing returned %d", k, r);
		require(!dlss_nr_pass_dispatch(&pass, CMD, &(struct dlss_nr_constants){}, 8, 8, VK_NULL_HANDLE,
		                               VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, HANDLE(VkImageView, 1),
		                               VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) && !fake.dispatches,
		        "an unbuilt composition pass dispatched (call %u failing)", k);
		dlss_nr_pass_fini(&pass);
		require(!live_objects() && !fake.mapped && zeroed(&pass, sizeof pass), "with call %u failing, the"
		        " composition pass's fini left %u objects and %u maps", k, live_objects(), fake.mapped);
	}
}

/** @brief Builds and finishes the scaling passes with each fallible call failing in turn. */
static void
check_scaler_failures (void)
{
	for (size_t i = 0; i < sizeof UPSAMPLE / sizeof *UPSAMPLE; ++i) {
		bool const up = UPSAMPLE[i];
		reset();
		struct scaler_vk scaler = scaler_vk(&device_table, &instance_table, DEVICE, PHYSICAL, up,
		                                    SCALER_VK_MAGIC);
		require(scaler.error == VK_SUCCESS && scaler.shader.pipeline,
		        "the scaling pass was not built");
		uint32_t const steps = fake.creates;
		require(steps == 11, "the scaling pass's build made %u fallible calls, not 11", steps);
		require(live_objects() == 7 && fake.mapped == 1, "a built scaling pass holds %u objects", live_objects());
		// The base owns all of them: its own fini frees them and empties it.
		shader_vk_fini(&scaler.shader);
		require(!live_objects() && !fake.mapped && zeroed(&scaler.shader, sizeof scaler.shader), "the scaling"
		        " pass's base's fini left %u objects", live_objects());
		scaler_vk_fini(&scaler);
		require(!live_objects() && !fake.mapped && zeroed(&scaler, sizeof scaler), "the scaling pass's fini"
		        " left %u objects", live_objects());

		for (uint32_t k = 1; k <= steps; ++k) {
			reset();
			fake.fail_at = k;
			VkResult const r = scaler_vk_init(&scaler, &device_table, &instance_table, DEVICE, PHYSICAL, up,
			                                  SCALER_VK_LANCZOS3);
			require(r == FAILURE && scaler.error == FAILURE && !scaler.shader.pipeline,
			        "the scaling pass's build with call %u failing returned %d", k, r);
			require(!scaler_vk_dispatch(&scaler, CMD, HANDLE(VkImageView, 1), HANDLE(VkImageView, 2), 8, 8, 4, 4)
			        && !fake.dispatches, "an unbuilt scaling pass dispatched");
			scaler_vk_fini(&scaler);
			require(!live_objects() && !fake.mapped && zeroed(&scaler, sizeof scaler), "with call %u failing,"
			        " the scaling pass's fini left %u objects", k, live_objects());
		}
	}
}

/** @brief Empty, zeroed and deviceless passes own nothing and call nothing. */
static void
check_empty (void)
{
	reset();
	struct dlss_nr_pass pass = {};
	struct scaler_vk scaler = {};
	dlss_nr_pass_fini(&pass);
	dlss_nr_pass_fini(nullptr);
	scaler_vk_fini(&scaler);
	scaler_vk_fini(nullptr);
	shader_vk_fini(nullptr);
	VkImageSubresourceRange const range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	shader_vk_set_image_layout(nullptr, CMD, HANDLE(VkImage, 1), VK_IMAGE_LAYOUT_UNDEFINED,
	                           VK_IMAGE_LAYOUT_GENERAL, range);
	shader_vk_set_image_layout(&pass.shader, CMD, HANDLE(VkImage, 1), VK_IMAGE_LAYOUT_UNDEFINED,
	                           VK_IMAGE_LAYOUT_GENERAL, range);
	require(!fake.calls, "finishing empty passes, or moving an image with one, called the device %u times",
	        fake.calls);
	require(dlss_nr_pass_init(nullptr, &device_table, &instance_table, DEVICE, PHYSICAL)
	        == VK_ERROR_INITIALIZATION_FAILED
	        && scaler_vk_init(nullptr, &device_table, &instance_table, DEVICE, PHYSICAL, false, 0)
	        == VK_ERROR_INITIALIZATION_FAILED, "an init without a destination did not fail");

	pass = dlss_nr_pass(&device_table, &instance_table, VK_NULL_HANDLE, PHYSICAL);
	require(pass.error == VK_ERROR_INITIALIZATION_FAILED && !fake.calls, "a composition pass without a device");
	pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, VK_NULL_HANDLE);
	require(pass.error == VK_ERROR_INITIALIZATION_FAILED && !fake.calls, "a composition pass without a physical"
	        " device");
	dlss_nr_pass_fini(&pass);
	scaler = scaler_vk(&device_table, &instance_table, VK_NULL_HANDLE, PHYSICAL, true, 0);
	require(scaler.error == VK_ERROR_INITIALIZATION_FAILED && !fake.calls, "a scaling pass without a device");
	scaler_vk_fini(&scaler);
	require(!fake.calls, "finishing deviceless passes called the device");
	require(!dlss_nr_pass_dispatch(nullptr, CMD, &(struct dlss_nr_constants){}, 1, 1, VK_NULL_HANDLE,
	                               VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, HANDLE(VkImageView, 1),
	                               VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL)
	        && !scaler_vk_dispatch(nullptr, CMD, HANDLE(VkImageView, 1), HANDLE(VkImageView, 2), 1, 1, 1, 1),
	        "a dispatch without a pass");

	// No host-visible memory: the ring's buffer goes again, and nothing is left.
	reset();
	fake.host_visible = false;
	pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, PHYSICAL);
	require(pass.error == VK_ERROR_FEATURE_NOT_PRESENT, "a ring without host-visible memory returned %d",
	        pass.error);
	dlss_nr_pass_fini(&pass);
	require(!live_objects(), "a ring without host-visible memory left %u objects", live_objects());
}

/** @brief The constant ring's stride over device alignments. */
static void
check_stride (void)
{
	struct {
		VkDeviceSize alignment;
		VkDeviceSize stride;
	} const cases[] = { { 0, 256 }, { 1, 256 }, { 16, 256 }, { 64, 256 }, { 256, 256 }, { 384, 384 }, { 512, 512 } };
	for (size_t i = 0; i < sizeof cases / sizeof *cases; ++i) {
		reset();
		fake.alignment = cases[i].alignment;
		struct dlss_nr_pass pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, PHYSICAL);
		struct scaler_vk scaler = scaler_vk(&device_table, &instance_table, DEVICE, PHYSICAL, false, 1);
		require(pass.slot_stride == cases[i].stride && scaler.slot_stride == cases[i].stride, "alignment %llu"
		        " gave strides %llu and %llu, not %llu", (unsigned long long)cases[i].alignment,
		        (unsigned long long)pass.slot_stride, (unsigned long long)scaler.slot_stride,
		        (unsigned long long)cases[i].stride);
		dlss_nr_pass_fini(&pass);
		scaler_vk_fini(&scaler);
	}
}

/** @brief What the composition pass's dispatches record. */
static void
check_pass_dispatch (void)
{
	reset();
	struct dlss_nr_pass pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, PHYSICAL);
	require(pass.error == VK_SUCCESS && !pass.slot && !pass.dummy_image, "a fresh pass");
	VkImageLayout const read_only = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkImageView const source = HANDLE(VkImageView, 0x901), model = HANDLE(VkImageView, 0x902);
	VkImageView const original = HANDLE(VkImageView, 0x903), motion = HANDLE(VkImageView, 0x904);
	VkImageView const target = HANDLE(VkImageView, 0x905), keep = HANDLE(VkImageView, 0x906);

	// Refused: no command buffer, no target. Neither takes a slot.
	struct dlss_nr_constants k = { .mode = DLSS_NR_MODE_RESOLVE, .white_point = 1.5f, .width = 67, .height = 41 };
	require(!dlss_nr_pass_dispatch(&pass, VK_NULL_HANDLE, &k, 67, 41, source, model, original, motion, target,
	                               keep, read_only, read_only)
	        && !dlss_nr_pass_dispatch(&pass, CMD, &k, 67, 41, source, model, original, motion, VK_NULL_HANDLE,
	                                  keep, read_only, read_only)
	        && !pass.slot && !fake.dispatches && !pass.dummy_image, "a refused dispatch recorded something");

	// The first dispatch makes the placeholder and moves it into GENERAL once.
	for (uint32_t n = 0; n < 2 * DLSS_NR_PASS_SLOTS + 3; ++n) {
		uint32_t const slot = pass.slot;
		k.width = 67 + n;
		k.white_point = 1.5f + (float)n;
		bool const absent = n & 1;
		uint32_t const image_barriers = fake.image_barriers;
		require(dlss_nr_pass_dispatch(&pass, CMD, &k, 67 + n, 41 - n, absent ? VK_NULL_HANDLE : source,
		                              absent ? VK_NULL_HANDLE : model, original, absent ? VK_NULL_HANDLE : motion,
		                              target, absent ? VK_NULL_HANDLE : keep, VK_IMAGE_LAYOUT_GENERAL, read_only),
		        "dispatch %u was refused", n);
		require(pass.slot == (slot + 1) % DLSS_NR_PASS_SLOTS, "dispatch %u took slot %u to %u", n, slot, pass.slot);
		require(pass.dummy_image && pass.dummy_view && alive((uint64_t)(uintptr_t)pass.dummy_view),
		        "no placeholder");
		require(fake.image_barriers - image_barriers == (n ? 0u : 1u), "dispatch %u moved the placeholder %u"
		        " times", n, fake.image_barriers - image_barriers);
		if (!n)
			require(fake.last_image_barrier.image == pass.dummy_image
			        && fake.last_image_barrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
			        && fake.last_image_barrier.newLayout == VK_IMAGE_LAYOUT_GENERAL, "the placeholder's move");
		require(!memcmp(fake.ring + pass.slot_stride * slot, &k, sizeof k), "dispatch %u's constants are not in"
		        " slot %u", n, slot);
		require(fake.groups[0] == (67 + n + 7) / 8 && fake.groups[1] == (41 - n + 7) / 8 && fake.groups[2] == 1,
		        "dispatch %u: %u x %u x %u groups", n, fake.groups[0], fake.groups[1], fake.groups[2]);
		require(fake.bound_pipeline == pass.shader.pipeline && fake.bound_set == pass.descriptor_sets[slot],
		        "dispatch %u bound another pipeline or set", n);
		require(fake.write_count == 8 && fake.buffer.buffer == pass.shader.constant_buffer
		        && fake.buffer.offset == pass.slot_stride * slot && fake.buffer.range == 256,
		        "dispatch %u: the constant binding", n);
		VkDescriptorType const types[8] = {
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER
		};
		VkSampler const sampler = pass.shader.texture_sampler;
		VkImageView const dummy = pass.dummy_view;
		VkDescriptorImageInfo const images[8] = {
			{},
			{ sampler, absent ? dummy : source, absent ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_GENERAL },
			{ sampler, absent ? dummy : model, absent ? VK_IMAGE_LAYOUT_GENERAL : read_only },
			{ sampler, original, read_only },
			{ sampler, absent ? dummy : motion, absent ? VK_IMAGE_LAYOUT_GENERAL : read_only },
			{ VK_NULL_HANDLE, target, VK_IMAGE_LAYOUT_GENERAL },
			{ VK_NULL_HANDLE, absent ? dummy : keep, VK_IMAGE_LAYOUT_GENERAL },
			{ sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED }
		};
		for (uint32_t b = 0; b < 8; ++b) {
			require(fake.writes[b].dstSet == pass.descriptor_sets[slot] && fake.writes[b].dstBinding == b
			        && fake.writes[b].descriptorType == types[b], "dispatch %u: write %u", n, b);
			require(!b || (fake.images[b].sampler == images[b].sampler
			               && fake.images[b].imageView == images[b].imageView
			               && fake.images[b].imageLayout == images[b].imageLayout),
			        "dispatch %u: binding %u's image", n, b);
		}
	}
	require(fake.dispatches == 2 * DLSS_NR_PASS_SLOTS + 3, "%u dispatches", fake.dispatches);
	dlss_nr_pass_fini(&pass);
	require(!live_objects() && !fake.mapped, "the pass's fini left %u objects after dispatches", live_objects());

	// A placeholder that fails at each of its steps (the image, its memory, the bind and the view), or
	// that finds no device-local memory: the dispatch is refused and leaves nothing of the placeholder.
	// Then either the pass is finished at once, or the next dispatch makes the whole placeholder and
	// binds its view.
	for (uint32_t step = 0; step <= 4; ++step) {
		for (uint32_t retry = 0; retry <= 1; ++retry) {
			reset();
			pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, PHYSICAL);
			uint32_t const built = live_objects();
			fake.fail_at = step ? fake.creates + step : 0;
			fake.device_local = step != 0;
			bool const refused = !dlss_nr_pass_dispatch(&pass, CMD, &k, 8, 8, source, model, original, motion,
			                                            target, keep, read_only, read_only);
			require(refused && !fake.dispatches && !pass.slot && !fake.image_barriers
			        && live_objects() == built && !pass.dummy_image && !pass.dummy_memory
			        && !pass.dummy_view, "a dispatch whose placeholder failed at step %u", step);
			fake.device_local = true;
			require(!retry
			        || (dlss_nr_pass_dispatch(&pass, CMD, &k, 8, 8, VK_NULL_HANDLE, model, original, motion,
			                                  target, keep, read_only, read_only)
			            && fake.dispatches == 1 && fake.image_barriers == 1 && live_objects() == built + 3
			            && alive((uint64_t)(uintptr_t)pass.dummy_view)
			            && fake.images[1].imageView == pass.dummy_view
			            && fake.last_image_barrier.image == pass.dummy_image),
			        "the dispatch after a placeholder that failed at step %u", step);
			dlss_nr_pass_fini(&pass);
			require(!live_objects() && !fake.mapped, "a placeholder that failed at step %u left %u objects"
			        " (retried: %u)", step, live_objects(), retry);
		}
	}
}

/** @brief What the scaling passes' dispatches record. */
static void
check_scaler_dispatch (void)
{
	for (size_t i = 0; i < sizeof UPSAMPLE / sizeof *UPSAMPLE; ++i) {
		bool const up = UPSAMPLE[i];
		reset();
		struct scaler_vk scaler = scaler_vk(&device_table, &instance_table, DEVICE, PHYSICAL, up,
		                                    SCALER_VK_KAISER3);
		VkImageView const source = HANDLE(VkImageView, 0x801), dest = HANDLE(VkImageView, 0x802);
		require(!scaler_vk_dispatch(&scaler, VK_NULL_HANDLE, source, dest, 1, 1, 1, 1)
		        && !scaler_vk_dispatch(&scaler, CMD, VK_NULL_HANDLE, dest, 1, 1, 1, 1)
		        && !scaler_vk_dispatch(&scaler, CMD, source, VK_NULL_HANDLE, 1, 1, 1, 1)
		        && !scaler.slot && !fake.dispatches, "a refused scaling dispatch recorded something");
		uint32_t const tile = up ? 16 : 8;
		for (uint32_t n = 0; n < 2 * SCALER_VK_SLOTS + 1; ++n) {
			uint32_t const slot = scaler.slot;
			uint32_t const sw = 67 + n, sh = 41, dw = up ? 100 + n : 33 + n, dh = up ? 61 : 21;
			memset(fake.ring + scaler.slot_stride * slot, 0xa5, 256);
			require(scaler_vk_dispatch(&scaler, CMD, source, dest, sw, sh, dw, dh), "scaling dispatch %u", n);
			require(scaler.slot == (slot + 1) % SCALER_VK_SLOTS, "a scaling dispatch took slot %u", slot);
			int32_t block[64];
			memcpy(block, fake.ring + scaler.slot_stride * slot, sizeof block);
			require(block[0] == (int32_t)sw && block[1] == (int32_t)sh && block[2] == (int32_t)dw
			        && block[3] == (int32_t)dh && zeroed(block + 4, sizeof block - 16), "scaling dispatch %u's"
			        " block", n);
			require(fake.groups[0] == (dw + tile - 1) / tile && fake.groups[1] == (dh + tile - 1) / tile
			        && fake.groups[2] == 1, "scaling dispatch %u: %u x %u groups", n, fake.groups[0],
			        fake.groups[1]);
			require(fake.write_count == 4 && fake.buffer.buffer == scaler.shader.constant_buffer
			        && fake.buffer.offset == scaler.slot_stride * slot && fake.buffer.range == 256
			        && fake.bound_set == scaler.descriptor_sets[slot]
			        && fake.bound_pipeline == scaler.shader.pipeline, "scaling dispatch %u's bindings", n);
			VkDescriptorType const types[4] = {
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER
			};
			VkDescriptorImageInfo const images[4] = {
				{},
				{ VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
				{ VK_NULL_HANDLE, dest, VK_IMAGE_LAYOUT_GENERAL },
				{ scaler.shader.texture_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED }
			};
			for (uint32_t b = 0; b < 4; ++b) {
				require(fake.writes[b].dstSet == scaler.descriptor_sets[slot] && fake.writes[b].dstBinding == b
				        && fake.writes[b].descriptorType == types[b], "scaling dispatch %u: write %u", n, b);
				require(!b || (fake.images[b].sampler == images[b].sampler
				               && fake.images[b].imageView == images[b].imageView
				               && fake.images[b].imageLayout == images[b].imageLayout),
				        "scaling dispatch %u: binding %u's image", n, b);
			}
		}
		scaler_vk_fini(&scaler);
		require(!live_objects(), "the scaling pass's fini left %u objects after dispatches", live_objects());
	}
}

#undef HANDLE

/** @brief What a scaling pass must be. */
struct expected_scaler {
	unsigned char const *spv;   //!< Its SPIR-V.
	size_t               size;  //!< The SPIR-V's size.
	char const          *name;  //!< The pass's name.
	uint64_t             flags; //!< The pass's flags.
};

/** @brief The SPIR-V that each pass runs, and the scaling passes' names and flags.
 *
 * The composition pass runs its shader, and the enlarge runs bcus whatever the filter. An average
 * runs its filter's shader, and Lanczos3 for FSR1 and for any value from SCALER_VK_COUNT on.
 */
static void
check_shaders (void)
{
	reset();
	struct dlss_nr_pass pass = dlss_nr_pass(&device_table, &instance_table, DEVICE, PHYSICAL);
	require(pass.error == VK_SUCCESS && fake.module_size == sizeof expected_dlssnr
	        && fake.module_hash == fnv1a(expected_dlssnr, sizeof expected_dlssnr),
	        "the composition pass runs %zu bytes of other SPIR-V", fake.module_size);
	dlss_nr_pass_fini(&pass);

#define AVERAGE(spv) { spv, sizeof spv, "dlssnr-average", 0 }
	struct {
		struct expected_scaler average;
		uint32_t               filter;
	} const cases[] = {
		{ AVERAGE(expected_lanczos3), SCALER_VK_FSR1        },
		{ AVERAGE(expected_bicubic),  SCALER_VK_BICUBIC     },
		{ AVERAGE(expected_catmull),  SCALER_VK_CATMULL_ROM },
		{ AVERAGE(expected_lanczos2), SCALER_VK_LANCZOS2    },
		{ AVERAGE(expected_lanczos3), SCALER_VK_LANCZOS3    },
		{ AVERAGE(expected_kaiser2),  SCALER_VK_KAISER2     },
		{ AVERAGE(expected_kaiser3),  SCALER_VK_KAISER3     },
		{ AVERAGE(expected_magc),     SCALER_VK_MAGIC       },
		{ AVERAGE(expected_lanczos3), SCALER_VK_COUNT       },
		{ AVERAGE(expected_lanczos3), 9                     },
		{ AVERAGE(expected_lanczos3), 100                   },
		{ AVERAGE(expected_lanczos3), UINT32_MAX            }
	};
#undef AVERAGE
	struct expected_scaler const enlarge = {
		expected_bcus, sizeof expected_bcus, "dlssnr-enlarge", SCALER_VK_UPSAMPLE
	};
	for (size_t c = 0; c < sizeof cases / sizeof *cases; ++c) {
		struct expected_scaler const *const directions[] = { &cases[c].average, &enlarge };
		for (size_t i = 0; i < sizeof UPSAMPLE / sizeof *UPSAMPLE; ++i) {
			struct expected_scaler const *const e = directions[i];
			reset();
			struct scaler_vk scaler = scaler_vk(&device_table, &instance_table, DEVICE, PHYSICAL, UPSAMPLE[i],
			                                    cases[c].filter);
			require(scaler.error == VK_SUCCESS && fake.module_size == e->size
			        && fake.module_hash == fnv1a(e->spv, e->size), "filter %u, upsample %d: %zu bytes of other"
			        " SPIR-V", cases[c].filter, UPSAMPLE[i], fake.module_size);
			require(!strcmp(scaler.shader.name, e->name) && scaler.flags == e->flags, "filter %u, upsample %d:"
			        " the pass %s with flags %#llx", cases[c].filter, UPSAMPLE[i], scaler.shader.name,
			        (unsigned long long)scaler.flags);
			scaler_vk_fini(&scaler);
		}
	}
}

/** @brief The downscalers' names, and the fallbacks to Lanczos3. */
static void
check_names (void)
{
	char const *const names[SCALER_VK_COUNT] = {
		"lanczos3", "bicubic", "catmull-rom", "lanczos2", "lanczos3", "kaiser2", "kaiser3", "magic"
	};
	for (uint32_t f = 0; f < SCALER_VK_COUNT; ++f)
		require(!strcmp(scaler_vk_filter_name(f), names[f]), "filter %u is %s", f, scaler_vk_filter_name(f));
	uint32_t const beyond[] = { SCALER_VK_COUNT, 9, 100, UINT32_MAX };
	for (size_t i = 0; i < sizeof beyond / sizeof *beyond; ++i)
		require(!strcmp(scaler_vk_filter_name(beyond[i]), "lanczos3"), "filter %u is %s", beyond[i],
		        scaler_vk_filter_name(beyond[i]));
}

int
main (void)
{
	tables();
	check_empty();
	check_pass_failures();
	check_scaler_failures();
	check_stride();
	check_pass_dispatch();
	check_scaler_dispatch();
	check_shaders();
	check_names();
	puts("PASS: the passes build, dispatch and free what they create, after a failure at any step too");
	return 0;
}
