/** @file
 *
 * The composition on the host, on a fake device: what composition() and composition_prepare() create
 * and what composition_fini() destroys, in each arrangement of surfaces and after a failure at each
 * fallible call; that every handle the composition holds is alive and every live object is held;
 * that no command or descriptor names a destroyed object; that a rebuild keeps a captured frame of
 * the same shape; what each leg dispatches and copies; the descriptors of the transport offer; and the
 * swapchain formats that the composition cannot write. What the shaders compute is
 * composition-rebuild-test's and hdr-shader-test's, on a real device.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "composition.h"

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
	fputs("composition-test: ", stderr);
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
	void     *map;    //!< A memory's mapping, while it is mapped.
	uint64_t  handle; //!< Its handle.
	uint64_t  size;   //!< A buffer's or a memory's size.
	uint64_t  pool;   //!< A set's pool.
	enum kind kind;   //!< What it is.
	uint32_t  live;   //!< Whether it is alive: 32 bits wide, which fills the padding.
};

/** @brief The fake device's state. ownership is a size_t, which fills the padding. */
static struct {
	struct object        objects[16384]; //!< Every object handed out, in order.
	VkFormatFeatureFlags features;       //!< What every format's optimal tiling supports.
	VkFormat             no_storage[2];  //!< Formats that cannot be storage images.
	int                  fds[4];         //!< The descriptors vkGetMemoryFdKHR() made, open or -1.
	uint32_t             fd_count;       //!< The descriptors it made.
	uint32_t             count;          //!< Objects handed out.
	uint32_t             creates;        //!< Fallible calls so far.
	uint32_t             fail_at;        //!< The fallible call that fails, from 1; 0: none.
	uint32_t             calls;          //!< Every call of the device's and the instance's tables.
	uint32_t             format_queries; //!< vkGetPhysicalDeviceFormatProperties() calls.
	uint32_t             mapped;         //!< Memory mapped and not unmapped.
	uint32_t             dispatches;     //!< Dispatches recorded.
	uint32_t             image_copies;   //!< Image-to-image copies and blits recorded.
	uint32_t             blits;          //!< Blits recorded.
	uint32_t             readbacks;      //!< Image-to-buffer copies recorded.
	uint32_t             uploads;        //!< Buffer-to-image copies recorded.
	uint32_t             fills;          //!< Buffer fills recorded.
	size_t               ownership;      //!< Barriers that move a buffer to or from another family.
} fake = { .fds = { -1, -1, -1, -1 } };

/** @brief Hands out an object of a kind. */
static uint64_t
make (enum kind kind,
      uint64_t  size)
{
	require(fake.count < sizeof fake.objects / sizeof *fake.objects, "too many objects");
	uint64_t const handle = 0x1000 + fake.count;
	fake.objects[fake.count++] = (struct object){ .handle = handle, .size = size, .kind = kind, .live = true };
	return handle;
}

/** @brief An object by its handle, of a kind; it must have been made. */
static struct object *
find (uint64_t  handle,
      enum kind kind)
{
	require(handle >= 0x1000 && handle < 0x1000 + fake.count, "object %#llx was never made",
	        (unsigned long long)handle);
	struct object *const o = &fake.objects[handle - 0x1000];
	require(o->kind == kind, "object %#llx used as kind %d, made as %d", (unsigned long long)handle, kind,
	        o->kind);
	return o;
}

/** @brief Requires a live object of a kind. */
static struct object *
live (uint64_t  handle,
      enum kind kind)
{
	struct object *const o = find(handle, kind);
	require(o->live, "object %#llx of kind %d is used after its destruction", (unsigned long long)handle,
	        kind);
	return o;
}

/** @brief Ends an object's life; it must be alive and of the kind. */
static void
end (uint64_t  handle,
     enum kind kind)
{
	require(handle, "a destroy of VK_NULL_HANDLE (kind %d)", kind);
	live(handle, kind)->live = false;
}

/** @brief The objects that are alive, sets aside: their pools free them. */
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

/** @brief Starts a scenario: no objects, no failure, every format feature. */
static void
reset (void)
{
	for (uint32_t i = 0; i < fake.count; ++i) {
		free(fake.objects[i].map);
		fake.objects[i].map = nullptr;
	}
	for (size_t i = 0; i < sizeof fake.fds / sizeof *fake.fds; ++i) {
		if (fake.fds[i] >= 0) {
			require(!close(fake.fds[i]), "cannot close descriptor %d", fake.fds[i]);
			fake.fds[i] = -1;
		}
	}
	memset(&fake, 0, sizeof fake);
	for (size_t i = 0; i < sizeof fake.fds / sizeof *fake.fds; ++i)
		fake.fds[i] = -1;
	fake.features = ~(VkFormatFeatureFlags)0;
}

/** @brief What a failed call leaves in its output handle: a value that it never made, which the
 *         composition must not destroy. Vulkan leaves the output undefined after most failures. */
static constexpr uint64_t POISON = 0xdead0000;

/** @brief What a fallible call that fails returns. */
static constexpr VkResult FAILURE = VK_ERROR_OUT_OF_DEVICE_MEMORY;

/** @brief A handle of a type with a value. */
#define HANDLE(type, value) ((type)(uintptr_t)(value))

/** @brief A handle's value. */
#define VALUE(handle) ((uint64_t)(uintptr_t)(handle))

/** @brief The body of a fake create call: it fails, leaving POISON, or hands out an object. */
#define CREATE(type, kind, size, out) \
	do { \
		++fake.calls; \
		if (fails()) { \
			*(out) = HANDLE(type, POISON); \
			return FAILURE; \
		} \
		*(out) = HANDLE(type, make((kind), (size))); \
		return VK_SUCCESS; \
	} while (0)

/** @brief The body of a fake destroy call: it ends the object's life. */
#define DESTROY(handle, kind) \
	do { \
		++fake.calls; \
		end(VALUE(handle), (kind)); \
	} while (0)

/** @brief The fake vkCreateSampler(). */
static VKAPI_ATTR VkResult VKAPI_CALL
create_sampler (VkDevice                     d,
                VkSamplerCreateInfo const   *i,
                VkAllocationCallbacks const *a,
                VkSampler                   *out)
{
	CREATE(VkSampler, KIND_SAMPLER, 0, out);
}

/** @brief The fake vkDestroySampler(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_sampler (VkDevice                     d,
                 VkSampler                    s,
                 VkAllocationCallbacks const *a)
{
	DESTROY(s, KIND_SAMPLER);
}

/** @brief The fake vkCreateBuffer(): a buffer of some bytes. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_buffer (VkDevice                     d,
               VkBufferCreateInfo const    *i,
               VkAllocationCallbacks const *a,
               VkBuffer                    *out)
{
	require(i->size, "a buffer of no bytes");
	CREATE(VkBuffer, KIND_BUFFER, i->size, out);
}

/** @brief The fake vkDestroyBuffer(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_buffer (VkDevice                     d,
                VkBuffer                     b,
                VkAllocationCallbacks const *a)
{
	DESTROY(b, KIND_BUFFER);
}

/** @brief The fake vkGetBufferMemoryRequirements(): the buffer's size, of either memory type. */
static VKAPI_ATTR void VKAPI_CALL
buffer_requirements (VkDevice              d,
                     VkBuffer              b,
                     VkMemoryRequirements *r)
{
	++fake.calls;
	uint64_t const size = live(VALUE(b), KIND_BUFFER)->size;
	*r = (VkMemoryRequirements){ .size = size, .alignment = 256, .memoryTypeBits = 3 };
}

/** @brief The fake vkGetImageMemoryRequirements(): 256 bytes, of either memory type. */
static VKAPI_ATTR void VKAPI_CALL
image_requirements (VkDevice              d,
                    VkImage               i,
                    VkMemoryRequirements *r)
{
	++fake.calls;
	live(VALUE(i), KIND_IMAGE);
	*r = (VkMemoryRequirements){ .size = 256, .alignment = 256, .memoryTypeBits = 3 };
}

/** @brief The fake vkAllocateMemory(): some bytes of one of the two memory types. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_memory (VkDevice                     d,
                 VkMemoryAllocateInfo const  *i,
                 VkAllocationCallbacks const *a,
                 VkDeviceMemory              *out)
{
	require(i->memoryTypeIndex < 2 && i->allocationSize, "an allocation of type %u", i->memoryTypeIndex);
	CREATE(VkDeviceMemory, KIND_MEMORY, i->allocationSize, out);
}

/** @brief The fake vkFreeMemory(): memory that is still mapped is unmapped with it. */
static VKAPI_ATTR void VKAPI_CALL
free_memory (VkDevice                     d,
             VkDeviceMemory               m,
             VkAllocationCallbacks const *a)
{
	struct object *const o = live(VALUE(m), KIND_MEMORY);
	if (o->map) {
		free(o->map);
		o->map = nullptr;
		--fake.mapped;
	}
	DESTROY(m, KIND_MEMORY);
}

/** @brief The fake vkBindBufferMemory(): a live buffer to live memory. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_buffer_memory (VkDevice       d,
                    VkBuffer       b,
                    VkDeviceMemory m,
                    VkDeviceSize   o)
{
	++fake.calls;
	live(VALUE(b), KIND_BUFFER);
	live(VALUE(m), KIND_MEMORY);
	return fails() ? FAILURE : VK_SUCCESS;
}

/** @brief The fake vkBindImageMemory(): a live image to live memory. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_image_memory (VkDevice       d,
                   VkImage        i,
                   VkDeviceMemory m,
                   VkDeviceSize   o)
{
	++fake.calls;
	live(VALUE(i), KIND_IMAGE);
	live(VALUE(m), KIND_MEMORY);
	return fails() ? FAILURE : VK_SUCCESS;
}

/** @brief The fake vkMapMemory(): maps the whole memory, once. */
static VKAPI_ATTR VkResult VKAPI_CALL
map_memory (VkDevice         d,
            VkDeviceMemory   m,
            VkDeviceSize     offset,
            VkDeviceSize     size,
            VkMemoryMapFlags f,
            void           **out)
{
	++fake.calls;
	struct object *const o = live(VALUE(m), KIND_MEMORY);
	require(!offset && (size == VK_WHOLE_SIZE || size == o->size) && !o->map, "a map of part of a memory,"
	        " or of a mapped one");
	if (fails()) {
		*out = (void *)(uintptr_t)POISON;
		return FAILURE;
	}
	o->map = calloc(1, o->size);
	require(o->map, "out of memory");
	++fake.mapped;
	*out = o->map;
	return VK_SUCCESS;
}

/** @brief The fake vkUnmapMemory(): the memory must be mapped. */
static VKAPI_ATTR void VKAPI_CALL
unmap_memory (VkDevice       d,
              VkDeviceMemory m)
{
	++fake.calls;
	struct object *const o = live(VALUE(m), KIND_MEMORY);
	require(o->map, "an unmap of memory that is not mapped");
	free(o->map);
	o->map = nullptr;
	--fake.mapped;
}

/** @brief The fake vkCreateDescriptorSetLayout(). */
static VKAPI_ATTR VkResult VKAPI_CALL
create_set_layout (VkDevice                               d,
                   VkDescriptorSetLayoutCreateInfo const *i,
                   VkAllocationCallbacks const           *a,
                   VkDescriptorSetLayout                 *out)
{
	CREATE(VkDescriptorSetLayout, KIND_SET_LAYOUT, 0, out);
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
	require(i->setLayoutCount == 1, "a pipeline layout of %u set layouts", i->setLayoutCount);
	live(VALUE(i->pSetLayouts[0]), KIND_SET_LAYOUT);
	CREATE(VkPipelineLayout, KIND_PIPELINE_LAYOUT, 0, out);
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
	CREATE(VkDescriptorPool, KIND_POOL, 0, out);
}

/** @brief The fake vkDestroyDescriptorPool(): the pool's sets end with it. */
static VKAPI_ATTR void VKAPI_CALL
destroy_pool (VkDevice                     d,
              VkDescriptorPool             p,
              VkAllocationCallbacks const *a)
{
	DESTROY(p, KIND_POOL);
	for (uint32_t i = 0; i < fake.count; ++i) {
		struct object *const o = &fake.objects[i];
		if (o->kind == KIND_SET && o->pool == VALUE(p))
			o->live = false;
	}
}

/** @brief The fake vkAllocateDescriptorSets(): from a live pool, of live layouts. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_sets (VkDevice                           d,
               VkDescriptorSetAllocateInfo const *i,
               VkDescriptorSet                   *out)
{
	++fake.calls;
	live(VALUE(i->descriptorPool), KIND_POOL);
	for (uint32_t s = 0; s < i->descriptorSetCount; ++s)
		live(VALUE(i->pSetLayouts[s]), KIND_SET_LAYOUT);
	if (fails()) {
		for (uint32_t s = 0; s < i->descriptorSetCount; ++s)
			out[s] = HANDLE(VkDescriptorSet, POISON);
		return FAILURE;
	}
	for (uint32_t s = 0; s < i->descriptorSetCount; ++s) {
		uint64_t const set = make(KIND_SET, 0);
		fake.objects[set - 0x1000].pool = VALUE(i->descriptorPool);
		out[s] = HANDLE(VkDescriptorSet, set);
	}
	return VK_SUCCESS;
}

/** @brief The fake vkCreateShaderModule(): requires aligned SPIR-V words. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_module (VkDevice                        d,
               VkShaderModuleCreateInfo const *i,
               VkAllocationCallbacks const    *a,
               VkShaderModule                 *out)
{
	require(!((uintptr_t)i->pCode & 3) && i->codeSize >= 20 && !(i->codeSize & 3)
	        && i->pCode[0] == 0x07230203, "SPIR-V that is not aligned words");
	CREATE(VkShaderModule, KIND_MODULE, 0, out);
}

/** @brief The fake vkDestroyShaderModule(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_module (VkDevice                     d,
                VkShaderModule               m,
                VkAllocationCallbacks const *a)
{
	DESTROY(m, KIND_MODULE);
}

/** @brief The fake vkCreateComputePipelines(): one pipeline, of a live module and layout. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_pipelines (VkDevice                           d,
                  VkPipelineCache                    c,
                  uint32_t                           n,
                  VkComputePipelineCreateInfo const *i,
                  VkAllocationCallbacks const       *a,
                  VkPipeline                        *out)
{
	require(n == 1, "%u pipelines at once", n);
	live(VALUE(i->stage.module), KIND_MODULE);
	live(VALUE(i->layout), KIND_PIPELINE_LAYOUT);
	CREATE(VkPipeline, KIND_PIPELINE, 0, out);
}

/** @brief The fake vkDestroyPipeline(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_pipeline (VkDevice                     d,
                  VkPipeline                   p,
                  VkAllocationCallbacks const *a)
{
	DESTROY(p, KIND_PIPELINE);
}

/** @brief The fake vkCreateImage(): a 2D image of some pixels. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_image (VkDevice                     d,
              VkImageCreateInfo const     *i,
              VkAllocationCallbacks const *a,
              VkImage                     *out)
{
	require(i->extent.width && i->extent.height && i->extent.depth == 1, "an image of %ux%ux%u",
	        i->extent.width, i->extent.height, i->extent.depth);
	CREATE(VkImage, KIND_IMAGE, 0, out);
}

/** @brief The fake vkDestroyImage(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_image (VkDevice                     d,
               VkImage                      i,
               VkAllocationCallbacks const *a)
{
	DESTROY(i, KIND_IMAGE);
}

/** @brief The fake vkCreateImageView(): of a live image. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_view (VkDevice                     d,
             VkImageViewCreateInfo const *i,
             VkAllocationCallbacks const *a,
             VkImageView                 *out)
{
	live(VALUE(i->image), KIND_IMAGE);
	CREATE(VkImageView, KIND_VIEW, 0, out);
}

/** @brief The fake vkDestroyImageView(). */
static VKAPI_ATTR void VKAPI_CALL
destroy_view (VkDevice                     d,
              VkImageView                  v,
              VkAllocationCallbacks const *a)
{
	DESTROY(v, KIND_VIEW);
}

/** @brief The fake vkUpdateDescriptorSets(): every set, view, sampler and buffer is alive. */
static VKAPI_ATTR void VKAPI_CALL
update_sets (VkDevice                    d,
             uint32_t                    n,
             VkWriteDescriptorSet const *w,
             uint32_t                    copies,
             VkCopyDescriptorSet const  *c)
{
	++fake.calls;
	require(!copies, "%u descriptor copies", copies);
	for (uint32_t i = 0; i < n; ++i) {
		live(VALUE(w[i].dstSet), KIND_SET);
		for (uint32_t j = 0; j < w[i].descriptorCount; ++j) {
			VkDescriptorImageInfo const *const image = w[i].pImageInfo ? &w[i].pImageInfo[j] : nullptr;
			if (image && image->imageView)
				live(VALUE(image->imageView), KIND_VIEW);
			if (image && image->sampler)
				live(VALUE(image->sampler), KIND_SAMPLER);
			if (w[i].pBufferInfo)
				live(VALUE(w[i].pBufferInfo[j].buffer), KIND_BUFFER);
		}
	}
}

/** @brief An image that a command names: one of the device's, alive, or the swapchain's. */
static void
live_image (VkImage image);

/** @brief The fake vkCmdBindPipeline(): a live pipeline. */
static VKAPI_ATTR void VKAPI_CALL
bind_pipeline (VkCommandBuffer     cb,
               VkPipelineBindPoint p,
               VkPipeline          pipeline)
{
	++fake.calls;
	live(VALUE(pipeline), KIND_PIPELINE);
}

/** @brief The fake vkCmdBindDescriptorSets(): live sets in a live layout. */
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
	live(VALUE(l), KIND_PIPELINE_LAYOUT);
	for (uint32_t i = 0; i < n; ++i)
		live(VALUE(sets[i]), KIND_SET);
}

/** @brief The fake vkCmdDispatch(): counts a dispatch of some groups. */
static VKAPI_ATTR void VKAPI_CALL
dispatch (VkCommandBuffer cb,
          uint32_t        x,
          uint32_t        y,
          uint32_t        z)
{
	++fake.calls;
	++fake.dispatches;
	require(x && y && z == 1, "a dispatch of %u x %u x %u groups", x, y, z);
}

/** @brief The fake vkCmdPushConstants(): the meter's 24 bytes, in a live layout. */
static VKAPI_ATTR void VKAPI_CALL
push_constants (VkCommandBuffer    cb,
                VkPipelineLayout   l,
                VkShaderStageFlags s,
                uint32_t           offset,
                uint32_t           size,
                void const        *values)
{
	++fake.calls;
	live(VALUE(l), KIND_PIPELINE_LAYOUT);
	require(!offset && size == 24, "%u bytes of push constants at %u", size, offset);
}

/** @brief The fake vkCmdFillBuffer(): counts a fill of a live buffer. */
static VKAPI_ATTR void VKAPI_CALL
fill_buffer (VkCommandBuffer cb,
             VkBuffer        b,
             VkDeviceSize    offset,
             VkDeviceSize    size,
             uint32_t        data)
{
	++fake.calls;
	++fake.fills;
	live(VALUE(b), KIND_BUFFER);
}

/** @brief The fake vkCmdCopyBuffer(): between live buffers, within their sizes. */
static VKAPI_ATTR void VKAPI_CALL
copy_buffer (VkCommandBuffer     cb,
             VkBuffer            src,
             VkBuffer            dst,
             uint32_t            n,
             VkBufferCopy const *regions)
{
	++fake.calls;
	struct object const *const s = live(VALUE(src), KIND_BUFFER);
	struct object const *const t = live(VALUE(dst), KIND_BUFFER);
	for (uint32_t i = 0; i < n; ++i)
		require(regions[i].srcOffset + regions[i].size <= s->size
		        && regions[i].dstOffset + regions[i].size <= t->size, "a copy beyond a buffer");
}

/** @brief The fake vkCmdCopyImage(): counts a copy between live images in transfer layouts. */
static VKAPI_ATTR void VKAPI_CALL
copy_image (VkCommandBuffer    cb,
            VkImage            src,
            VkImageLayout      sl,
            VkImage            dst,
            VkImageLayout      dl,
            uint32_t           n,
            VkImageCopy const *regions)
{
	++fake.calls;
	++fake.image_copies;
	live_image(src);
	live_image(dst);
	require(sl == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && dl == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	        "a copy between layouts %d and %d", sl, dl);
}

/** @brief The fake vkCmdBlitImage(): counts a blit, and a copy, between live images. */
static VKAPI_ATTR void VKAPI_CALL
blit_image (VkCommandBuffer    cb,
            VkImage            src,
            VkImageLayout      sl,
            VkImage            dst,
            VkImageLayout      dl,
            uint32_t           n,
            VkImageBlit const *regions,
            VkFilter           filter)
{
	++fake.calls;
	++fake.image_copies;
	++fake.blits;
	live_image(src);
	live_image(dst);
	require(filter == VK_FILTER_NEAREST, "a blit that filters");
}

/** @brief The fake vkCmdCopyImageToBuffer(): counts a readback from a live image. */
static VKAPI_ATTR void VKAPI_CALL
copy_image_to_buffer (VkCommandBuffer          cb,
                      VkImage                  src,
                      VkImageLayout            sl,
                      VkBuffer                 dst,
                      uint32_t                 n,
                      VkBufferImageCopy const *regions)
{
	++fake.calls;
	++fake.readbacks;
	live_image(src);
	live(VALUE(dst), KIND_BUFFER);
	require(sl == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, "a readback from layout %d", sl);
}

/** @brief The fake vkCmdCopyBufferToImage(): counts an upload into a live image. */
static VKAPI_ATTR void VKAPI_CALL
copy_buffer_to_image (VkCommandBuffer          cb,
                      VkBuffer                 src,
                      VkImage                  dst,
                      VkImageLayout            dl,
                      uint32_t                 n,
                      VkBufferImageCopy const *regions)
{
	++fake.calls;
	++fake.uploads;
	live(VALUE(src), KIND_BUFFER);
	live_image(dst);
	require(dl == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, "an upload into layout %d", dl);
}

/** @brief The fake vkCmdPipelineBarrier(): every buffer and image is alive. */
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
	for (uint32_t k = 0; k < buffers; ++k) {
		live(VALUE(b[k].buffer), KIND_BUFFER);
		fake.ownership += b[k].srcQueueFamilyIndex != b[k].dstQueueFamilyIndex;
	}
	for (uint32_t k = 0; k < images; ++k)
		live_image(i[k].image);
}

/** @brief The fake vkGetMemoryFdKHR(): a new descriptor for a live memory. */
static VKAPI_ATTR VkResult VKAPI_CALL
memory_fd (VkDevice                    d,
           VkMemoryGetFdInfoKHR const *i,
           int                        *out)
{
	++fake.calls;
	live(VALUE(i->memory), KIND_MEMORY);
	require(i->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, "a descriptor of type %#x",
	        (unsigned)i->handleType);
	if (fails()) {
		*out = -1;
		return FAILURE;
	}
	require(fake.fd_count < sizeof fake.fds / sizeof *fake.fds, "too many descriptors");
	int const fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
	require(fd >= 0, "cannot open /dev/null");
	fake.fds[fake.fd_count++] = fd;
	// The caller owns it now, and closes it; reset() closes it if the caller did not.
	*out = fd;
	return VK_SUCCESS;
}

/** @brief The fake vkGetPhysicalDeviceProperties(): a 64-byte offset alignment. */
static VKAPI_ATTR void VKAPI_CALL
physical_properties (VkPhysicalDevice            p,
                     VkPhysicalDeviceProperties *out)
{
	++fake.calls;
	memset(out, 0, sizeof *out);
	out->limits.minUniformBufferOffsetAlignment = 64;
}

/** @brief The UUIDs that the fake device and its driver have. */
static unsigned char const DEVICE_UUID[VK_UUID_SIZE] = "fake device uui";
static unsigned char const DRIVER_UUID[VK_UUID_SIZE] = "fake driver uui";

/** @brief The fake vkGetPhysicalDeviceProperties2(): the device's and driver's UUIDs. */
static VKAPI_ATTR void VKAPI_CALL
physical_properties2 (VkPhysicalDevice             p,
                      VkPhysicalDeviceProperties2 *out)
{
	++fake.calls;
	VkPhysicalDeviceIDProperties *const ids = out->pNext;
	require(ids && ids->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, "no ID properties asked for");
	memcpy(ids->deviceUUID, DEVICE_UUID, VK_UUID_SIZE);
	memcpy(ids->driverUUID, DRIVER_UUID, VK_UUID_SIZE);
}

/** @brief The fake vkGetPhysicalDeviceMemoryProperties(): a device-local type and a host-visible,
 *         coherent and cached one. */
static VKAPI_ATTR void VKAPI_CALL
memory_properties (VkPhysicalDevice                  p,
                   VkPhysicalDeviceMemoryProperties *out)
{
	++fake.calls;
	memset(out, 0, sizeof *out);
	out->memoryTypeCount = 2;
	out->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	out->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	                                    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
}

/** @brief The fake vkGetPhysicalDeviceFormatProperties(): the scenario's features, without storage
 *         for the formats it names. */
static VKAPI_ATTR void VKAPI_CALL
format_properties (VkPhysicalDevice    p,
                   VkFormat            format,
                   VkFormatProperties *out)
{
	++fake.calls;
	++fake.format_queries;
	VkFormatFeatureFlags features = fake.features;
	for (size_t i = 0; i < sizeof fake.no_storage / sizeof *fake.no_storage; ++i) {
		if (fake.no_storage[i] == format)
			features &= ~(VkFormatFeatureFlags)VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
	}
	*out = (VkFormatProperties){ .optimalTilingFeatures = features };
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

/** @brief The command buffer that the composition records into. */
static VkCommandBuffer const CMD = HANDLE(VkCommandBuffer, 0x30);

/** @brief The swapchain image that the composition reads and writes, which the device did not make. */
static VkImage const SWAPCHAIN = HANDLE(VkImage, 0x40);

#undef HANDLE

static void
live_image (VkImage image)
{
	if (image != SWAPCHAIN)
		live(VALUE(image), KIND_IMAGE);
}

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
		.vkCmdPushConstants            = push_constants,
		.vkCmdFillBuffer               = fill_buffer,
		.vkCmdCopyBuffer               = copy_buffer,
		.vkCmdCopyImage                = copy_image,
		.vkCmdBlitImage                = blit_image,
		.vkCmdCopyImageToBuffer        = copy_image_to_buffer,
		.vkCmdCopyBufferToImage        = copy_buffer_to_image,
		.vkCmdPipelineBarrier          = pipeline_barrier,
		.vkGetMemoryFdKHR              = memory_fd
	};
	instance_table = (struct instance_table){
		.vkGetPhysicalDeviceProperties       = physical_properties,
		.vkGetPhysicalDeviceProperties2      = physical_properties2,
		.vkGetPhysicalDeviceMemoryProperties = memory_properties,
		.vkGetPhysicalDeviceFormatProperties = format_properties
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

/** @brief Whether a composition is empty: its offer is -1, and every other byte is zero. */
static bool
empty (struct composition const *c)
{
	struct composition copy = *c;
	if (copy.offer != -1)
		return false;
	copy.offer = 0;
	return zeroed(&copy, sizeof copy);
}

/** @brief Counts a handle that the composition holds, which must be alive. */
static uint32_t
held (uint64_t  handle,
      enum kind kind)
{
	if (!handle)
		return 0;
	live(handle, kind);
	return 1;
}

/** @brief The objects that a pass's base holds. */
static uint32_t
held_shader (struct shader_vk const *s)
{
	return held(VALUE(s->pipeline), KIND_PIPELINE) + held(VALUE(s->pipeline_layout), KIND_PIPELINE_LAYOUT)
	       + held(VALUE(s->descriptor_set_layout), KIND_SET_LAYOUT)
	       + held(VALUE(s->descriptor_pool), KIND_POOL) + held(VALUE(s->constant_buffer), KIND_BUFFER)
	       + held(VALUE(s->constant_buffer_memory), KIND_MEMORY)
	       + held(VALUE(s->texture_sampler), KIND_SAMPLER);
}

/** @brief The objects that an image of the composition holds. */
static uint32_t
held_image (struct composition_image const *img)
{
	return held(VALUE(img->image), KIND_IMAGE) + held(VALUE(img->memory), KIND_MEMORY)
	       + held(VALUE(img->view), KIND_VIEW);
}

/** @brief The objects that a buffer of the composition holds; it is mapped exactly when the memory
 *         is host-visible. */
static uint32_t
held_buffer (struct composition_host_buffer const *buf)
{
	if (buf->mapped)
		require(live(VALUE(buf->memory), KIND_MEMORY)->map == buf->mapped, "a buffer's stale mapping");
	return held(VALUE(buf->buffer), KIND_BUFFER) + held(VALUE(buf->memory), KIND_MEMORY);
}

/** @brief Requires that the composition holds exactly the live objects: every handle it holds is
 *         alive, and every live object other than the swapchain's is one of them. */
static void
require_held (struct composition const *c,
              char const               *when)
{
	uint32_t n = held_shader(&c->pass.shader) + held_shader(&c->super_up.shader)
	             + held_shader(&c->super_down.shader);
	n += held(VALUE(c->pass.dummy_image), KIND_IMAGE) + held(VALUE(c->pass.dummy_memory), KIND_MEMORY)
	     + held(VALUE(c->pass.dummy_view), KIND_VIEW);
	struct composition_image const *const images[] = {
		&c->frame, &c->proxy, &c->work, &c->model, &c->composed, &c->model_native, &c->meter
	};
	for (size_t i = 0; i < sizeof images / sizeof *images; ++i)
		n += held_image(images[i]);
	struct composition_host_buffer const *const buffers[] = {
		&c->meter_mirror, &c->download, &c->upload, &c->capture_buf
	};
	for (size_t i = 0; i < sizeof buffers / sizeof *buffers; ++i)
		n += held_buffer(buffers[i]);
	n += held(VALUE(c->meter_state), KIND_BUFFER) + held(VALUE(c->meter_state_memory), KIND_MEMORY)
	     + held(VALUE(c->meter_pipeline), KIND_PIPELINE)
	     + held(VALUE(c->meter_pipeline_layout), KIND_PIPELINE_LAYOUT)
	     + held(VALUE(c->meter_descriptor_layout), KIND_SET_LAYOUT)
	     + held(VALUE(c->meter_descriptor_pool), KIND_POOL) + held(VALUE(c->meter_sampler), KIND_SAMPLER);
	if (c->meter_descriptor_set)
		live(VALUE(c->meter_descriptor_set), KIND_SET);
	require(n == live_objects(), "%s: the composition holds %u objects of %u live ones", when, n,
	        live_objects());
}

#undef VALUE

/** @brief An arrangement of the composition's surfaces. objects is a size_t, which fills the padding. */
struct arrangement {
	char const *name;          //!< What it is called.
	VkFormat    format;        //!< The swapchain's format.
	float       working_scale; //!< The model's raster over the frame's.
	bool        linear_hdr;    //!< The frame holds linear light.
	bool        hdr_proxy;     //!< The proxy is float16.
	bool        export_pair;   //!< The transport pair is exported.
	bool        measured;      //!< The meter's reading is the white point.
	uint32_t    bypass;        //!< composition_bypass.
	uint32_t    leg1;          //!< The dispatches of leg 1.
	uint32_t    leg2;          //!< The dispatches of leg 2.
	size_t      objects;       //!< The live objects after a build and both legs.
};

/** @brief The arrangements, each built on 320x192 frames.
 *
 * The pass holds 7 objects and, after its first dispatch, the placeholder's 3; an image holds 3, a
 * buffer 2, and a scaling pass 7. The meter holds its image, its state and mirror buffers, and its
 * reduce pass's pipeline, two layouts, pool and sampler: 12. An 8-bit frame is its own proxy, so leg 1
 * downsamples it into the work image even at full scale, or enlarges it when the model works above
 * it: the frame, model, composed and work images and the transport pair make 26 objects with the
 * pass. Supersampling adds the averaged answer and both scaling passes. A linear frame is encoded
 * into a proxy and measured by the meter, whose reduce is a dispatch of its own; it needs a work image
 * only below full scale. The raw bypass copies the model's answer and runs no resolve.
 */
static struct arrangement const ARRANGEMENTS[] = {
	{ "8-bit",          VK_FORMAT_B8G8R8A8_UNORM,      1.0f,  false, false, false, false, 0, 1, 1, 26 },
	{ "8-bit sRGB 0.5", VK_FORMAT_R8G8B8A8_SRGB,       0.5f,  false, false, false, false, 0, 1, 1, 26 },
	{ "supersampled",   VK_FORMAT_B8G8R8A8_UNORM,      1.5f,  false, false, false, false, 0, 1, 2, 43 },
	{ "linear float16", VK_FORMAT_R16G16B16A16_SFLOAT, 1.0f,  true,  true,  false, false, 0, 3, 1, 38 },
	{ "linear 0.75",    VK_FORMAT_R16G16B16A16_SFLOAT, 0.75f, true,  false, false, true,  0, 4, 1, 41 },
	{ "exported",       VK_FORMAT_B8G8R8A8_UNORM,      1.0f,  false, false, true,  false, 0, 1, 1, 26 },
	{ "raw bypass",     VK_FORMAT_R8G8B8A8_UNORM,      1.0f,  false, false, false, false, 1, 1, 0, 26 }
};

/** @brief The frames' size. */
static constexpr uint32_t WIDTH = 320;
static constexpr uint32_t HEIGHT = 192;

/** @brief The settings of an arrangement. */
static struct composition_frame_settings
settings (struct arrangement const *a)
{
	struct composition_frame_settings s = composition_frame_settings();
	s.working_scale = a->working_scale;
	s.white_point_source = a->measured ? kWhitePointMeasured : kWhitePointManual;
	s.composition_bypass = a->bypass;
	return s;
}

/** @brief Builds a composition on the fake device, which must succeed. */
static struct composition
built (void)
{
	struct composition c = composition(&device_table, &instance_table, DEVICE, PHYSICAL);
	require(c.error == VK_SUCCESS && composition_usable(&c) && !*composition_reason(&c),
	        "the composition was not built: %s", composition_reason(&c));
	return c;
}

/** @brief Prepares a composition for an arrangement. */
static bool
prepare (struct composition       *c,
         struct arrangement const *a)
{
	struct composition_frame_settings const s = settings(a);
	return composition_prepare(c, WIDTH, HEIGHT, a->format, &s, a->linear_hdr, a->hdr_proxy, 0);
}

/** @brief Records both legs of a frame, which must succeed, and checks their dispatches and copies. */
static void
record (struct composition       *c,
        struct arrangement const *a)
{
	struct composition_frame_settings const s = settings(a);
	uint32_t const dispatches = fake.dispatches;
	uint32_t const fills = fake.fills;
	require(composition_record_capture(c, CMD, SWAPCHAIN, &s), "%s: leg 1 was not recorded", a->name);
	require(fake.dispatches - dispatches == a->leg1, "%s: leg 1 dispatched %u times, not %u", a->name,
	        fake.dispatches - dispatches, a->leg1);
	require(fake.fills - fills == (a->linear_hdr ? 1u : 0u), "%s: leg 1 cleared the meter's state %u times",
	        a->name, fake.fills - fills);
	require(fake.readbacks == 1, "%s: leg 1 read back %u images", a->name, fake.readbacks);
	uint32_t const copies = fake.image_copies;
	require(composition_record_compose(c, CMD, SWAPCHAIN, &s), "%s: leg 2 was not recorded", a->name);
	require(fake.dispatches - dispatches - a->leg1 == a->leg2, "%s: leg 2 dispatched %u times, not %u",
	        a->name, fake.dispatches - dispatches - a->leg1, a->leg2);
	require(fake.image_copies - copies == 1 && fake.uploads == 1, "%s: leg 2 made %u image copies and %u"
	        " uploads", a->name, fake.image_copies - copies, fake.uploads);
	require(!composition_capture_recorded(c), "%s: a pair was recorded without a capture", a->name);
}

/** @brief Builds, records and finishes each arrangement. */
static void
check_arrangements (void)
{
	for (size_t i = 0; i < sizeof ARRANGEMENTS / sizeof *ARRANGEMENTS; ++i) {
		struct arrangement const *const a = &ARRANGEMENTS[i];
		reset();
		struct composition c = built();
		if (a->export_pair)
			composition_enable_export(&c, 3);
		require(prepare(&c, a), "%s: the build failed: %s", a->name, composition_reason(&c));
		require_held(&c, a->name);
		require(composition_model_width(&c) == (uint32_t)(WIDTH * a->working_scale + 0.5f)
		        && composition_model_height(&c) == (uint32_t)(HEIGHT * a->working_scale + 0.5f)
		        && composition_hdr_proxy_active(&c) == a->hdr_proxy
		        && composition_model_bytes(&c) == (size_t)composition_model_width(&c)
		                                          * composition_model_height(&c) * (a->hdr_proxy ? 8 : 4),
		        "%s: a model of %ux%u", a->name, composition_model_width(&c), composition_model_height(&c));
		require(composition_transport_exported(&c) == a->export_pair
		        && composition_transport_pending(&c) == a->export_pair
		        && !composition_proxy_pixels(&c) == a->export_pair
		        && !composition_model_pixels(&c) == a->export_pair, "%s: the transport pair", a->name);

		// Nothing changed: no call but the format queries.
		uint32_t const calls = fake.calls - fake.format_queries;
		require(prepare(&c, a) && fake.calls - fake.format_queries == calls, "%s: a second build of the same"
		        " frame called the device", a->name);

		record(&c, a);
		require(fake.ownership == (a->export_pair ? 4u : 0u), "%s: %zu ownership barriers", a->name,
		        fake.ownership);
		require_held(&c, a->name);
		require(live_objects() == a->objects, "%s: %u live objects, not %zu", a->name, live_objects(),
		        a->objects);

		composition_fini(&c);
		require(!live_objects() && !fake.mapped && empty(&c), "%s: the fini left %u objects and %u"
		        " maps", a->name, live_objects(), fake.mapped);
	}
}

/** @brief Builds a composition with each fallible call of its pass failing in turn. */
static void
check_build_failures (void)
{
	reset();
	struct composition c = built();
	uint32_t const steps = fake.creates;
	composition_fini(&c);
	require(!live_objects() && !fake.mapped && empty(&c), "a built composition's fini left %u"
	        " objects", live_objects());

	for (uint32_t k = 1; k <= steps; ++k) {
		reset();
		fake.fail_at = k;
		VkResult const r = composition_init(&c, &device_table, &instance_table, DEVICE, PHYSICAL);
		require(r == FAILURE && c.error == FAILURE && !composition_usable(&c)
		        && !strcmp(composition_reason(&c), "the composition pipeline could not be built"),
		        "with call %u failing, the build returned %d: %s", k, r, composition_reason(&c));
		require(!live_objects() && !fake.mapped, "with call %u failing, the build left %u objects", k,
		        live_objects());
		struct composition_frame_settings const s = composition_frame_settings();
		require(!composition_prepare(&c, WIDTH, HEIGHT, VK_FORMAT_B8G8R8A8_UNORM, &s, false, false, 0)
		        && !composition_record_capture(&c, CMD, SWAPCHAIN, &s)
		        && !composition_record_compose(&c, CMD, SWAPCHAIN, &s), "an unbuilt composition ran");
		composition_fini(&c);
		require(!live_objects() && empty(&c), "with call %u failing, the fini left %u objects", k,
		        live_objects());
	}

	// A device table without a call that the composition needs: no pass, no call.
	reset();
	struct device_table partial = device_table;
	partial.vkCmdCopyBufferToImage = nullptr;
	c = composition(&partial, &instance_table, DEVICE, PHYSICAL);
	require(c.error == VK_ERROR_INITIALIZATION_FAILED && !composition_usable(&c) && !fake.calls
	        && !strcmp(composition_reason(&c), "the device does not expose everything a compute pass needs"),
	        "a composition on an incomplete table: %d, %s", c.error, composition_reason(&c));
	composition_fini(&c);
	require(!fake.calls && empty(&c), "the fini of an unbuilt composition called the device");
}

/** @brief Builds an arrangement with one fallible call failing, then again without one, and records
 *         and finishes it.
 *
 * @param a      The arrangement.
 * @param k      The fallible call that fails, from 1.
 * @param failed Counts the failed builds.
 * @return       The fallible calls that the failing build made: fewer than @a k when none failed.
 */
static uint32_t
prepare_failing (struct arrangement const *a,
                 uint32_t                  k,
                 uint32_t                 *failed)
{
	reset();
	struct composition c = built();
	if (a->export_pair)
		composition_enable_export(&c, 3);
	fake.creates = 0;
	fake.fail_at = k;
	bool const ok = prepare(&c, a);
	uint32_t const steps = fake.creates;
	fake.fail_at = 0;
	char when[96];
	int const length = snprintf(when, sizeof when, "%s with call %u failing", a->name, k);
	require(length >= 0 && length < (int)sizeof when, "the case %s is too long to name", a->name);
	require_held(&c, when);
	*failed += !ok;
	require(ok || (!c.frame.image && !c.download.buffer && !c.width && !c.model_w && composition_usable(&c)
	               && !strcmp(composition_reason(&c), "could not allocate the composition surfaces")),
	        "%s: the failed build left surfaces, or the reason %s", when, composition_reason(&c));

	// The next frame builds everything and composes.
	require(prepare(&c, a), "%s: the next build failed: %s", when, composition_reason(&c));
	require_held(&c, when);
	record(&c, a);
	composition_fini(&c);
	require(!live_objects() && !fake.mapped && empty(&c), "%s: the fini left %u objects", when,
	        live_objects());
	return steps;
}

/** @brief Builds each arrangement with each fallible call failing in turn. */
static void
check_prepare_failures (void)
{
	for (size_t i = 0; i < sizeof ARRANGEMENTS / sizeof *ARRANGEMENTS; ++i) {
		uint32_t failed = 0;
		for (uint32_t k = 1; prepare_failing(&ARRANGEMENTS[i], k, &failed) >= k; ++k)
			continue;
		require(failed, "%s: no failure failed the build", ARRANGEMENTS[i].name);
	}
}

/** @brief A rebuild keeps a captured frame of the same shape, and a held frame is not copied again. */
static void
check_kept_frame (void)
{
	reset();
	struct composition c = built();
	struct arrangement a = ARRANGEMENTS[0];
	struct composition_frame_settings s = settings(&a);
	require(prepare(&c, &a), "the first build failed");
	VkImage const first = c.frame.image;

	uint32_t copies = fake.image_copies;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s) && fake.image_copies - copies == 1
	        && (c.flags & COMPOSITION_FRAME_CAPTURED), "the first frame was not copied");

	// Held: the frame stays as it is.
	s.hold_frame = 1;
	copies = fake.image_copies;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s) && fake.image_copies == copies
	        && (c.flags & COMPOSITION_HOLDING), "a held frame was copied again");

	// Another model raster: the same frame, still captured and held.
	a.working_scale = 0.5f;
	s.working_scale = 0.5f;
	require(prepare(&c, &a) && c.frame.image == first && (c.flags & COMPOSITION_FRAME_CAPTURED)
	        && c.model_w == WIDTH / 2, "a rebuild for another model raster did not keep the frame");
	require_held(&c, "the kept frame");
	copies = fake.image_copies;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s) && fake.image_copies == copies,
	        "the kept, held frame was copied again");

	// Another format: a new frame, copied although the hold is still on.
	a.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	require(prepare(&c, &a) && !(c.flags & COMPOSITION_FRAME_CAPTURED), "a rebuild for another format kept"
	        " the frame");
	require_held(&c, "a new frame");
	copies = fake.image_copies;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s) && fake.image_copies - copies == 1,
	        "the new frame was not copied");

	// Released.
	s.hold_frame = 0;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s) && !(c.flags & COMPOSITION_HOLDING),
	        "the hold was not released");
	composition_fini(&c);
	require(!live_objects(), "the fini left %u objects", live_objects());
}

/** @brief Each new state of the meter is cleared once, by the first leg 1 after the build that made it. */
static void
check_meter_clear (void)
{
	reset();
	struct composition c = built();
	struct arrangement a = ARRANGEMENTS[4];
	static float const scales[] = { 0.75f, 0.5f };
	for (size_t i = 0; i < sizeof scales / sizeof *scales; ++i) {
		a.working_scale = scales[i];
		struct composition_frame_settings const s = settings(&a);
		require(prepare(&c, &a), "build %zu failed: %s", i, composition_reason(&c));
		uint32_t const fills = fake.fills;
		require(composition_record_capture(&c, CMD, SWAPCHAIN, &s) && fake.fills - fills == 1
		        && composition_record_capture(&c, CMD, SWAPCHAIN, &s) && fake.fills - fills == 1,
		        "after build %zu, two frames cleared the meter's state %u times", i, fake.fills - fills);
	}
	composition_fini(&c);
	require(!live_objects(), "the fini left %u objects", live_objects());
}

/** @brief A frame being captured is resolved and read back as a pair, even under the raw bypass, and
 *         the next frame without a capture records no pair. */
static void
check_capture_pair (void)
{
	reset();
	struct composition c = built();
	struct arrangement const *const a = &ARRANGEMENTS[6];
	struct composition_frame_settings const s = settings(a);
	require(prepare(&c, a), "the build failed: %s", composition_reason(&c));

	// A batch underway; nothing here writes its files.
	c.capture.remaining = 1;
	uint32_t dispatches = fake.dispatches;
	uint32_t readbacks = fake.readbacks;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s)
	        && composition_record_compose(&c, CMD, SWAPCHAIN, &s) && composition_capture_recorded(&c),
	        "a captured frame recorded no pair");
	require(fake.dispatches - dispatches == a->leg1 + 1 && fake.readbacks - readbacks == 3, "a captured frame:"
	        " %u dispatches and %u readbacks", fake.dispatches - dispatches, fake.readbacks - readbacks);
	require_held(&c, "a capture pair");

	c.capture.remaining = 0;
	dispatches = fake.dispatches;
	readbacks = fake.readbacks;
	require(composition_record_capture(&c, CMD, SWAPCHAIN, &s)
	        && composition_record_compose(&c, CMD, SWAPCHAIN, &s) && !composition_capture_recorded(&c),
	        "the frame after a capture recorded a pair");
	require(fake.dispatches - dispatches == a->leg1 + a->leg2 && fake.readbacks - readbacks == 1, "the frame"
	        " after a capture: %u dispatches and %u readbacks", fake.dispatches - dispatches,
	        fake.readbacks - readbacks);
	composition_fini(&c);
	require(!live_objects() && !fake.mapped, "the fini left %u objects and %u maps", live_objects(),
	        fake.mapped);
}

/** @brief Whether a descriptor is open. */
static bool
open_fd (int fd)
{
	return fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}

/** @brief The offer's connection is closed by whatever ends the offer, and descriptor 0 never is. */
static void
check_offer (void)
{
	reset();
	bool const stdin_open = open_fd(0);
	struct composition none = composition_empty();
	require(composition_offer_connection(&none) == -1 && composition_offer_connection(nullptr) == -1,
	        "an empty composition has an offer");
	composition_fini(&none);
	composition_withdraw_offer(&none);
	require(open_fd(0) == stdin_open && empty(&none), "an empty composition closed descriptor 0");

	struct composition c = built();
	struct arrangement const *const a = &ARRANGEMENTS[0];
	require(prepare(&c, a), "the build failed");
	int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
	require(fd >= 0, "cannot open /dev/null");
	composition_await_answer(&c, fd);
	require(composition_offer_connection(&c) == fd, "the offer's connection is %d, not %d",
	        composition_offer_connection(&c), fd);
	composition_withdraw_offer(&c);
	require(!open_fd(fd) && composition_offer_connection(&c) == -1, "a withdrawn offer's connection");
	fd = -1;
	composition_withdraw_offer(&c);

	// Without a composition, the connection is closed at once.
	fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
	require(fd >= 0, "cannot open /dev/null");
	composition_await_answer(nullptr, fd);
	require(!open_fd(fd), "a connection without a composition stayed open");
	fd = -1;

	// Each end of an offer closes its connection once: an answer, a replacement, a new pair, the
	// fini.
	for (uint32_t end = 0; end < 4; ++end) {
		fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
		require(fd >= 0, "cannot open /dev/null");
		composition_await_answer(&c, fd);
		switch (end) {
		case 0:
			composition_set_transport_ready(&c, false);
			break;
		case 1:
			composition_await_answer(&c, -1);
			break;
		case 2:
			composition_disable_export(&c);
			break;
		default:
			composition_fini(&c);
		}
		require(!open_fd(fd) && composition_offer_connection(&c) == -1, "end %u left the connection", end);
		fd = -1;
	}
	require(open_fd(0) == stdin_open && !live_objects(), "descriptor 0 closed, or %u objects left",
	        live_objects());
}

/** @brief The exported pair's offer: its descriptors, sizes, UUIDs and generation. */
static void
check_export (void)
{
	reset();
	struct composition c = built();
	struct arrangement const *const a = &ARRANGEMENTS[5];
	composition_enable_export(&c, 3);
	require(prepare(&c, a) && composition_transport_pending(&c) && !composition_transport_generation(&c),
	        "an exported pair was not pending");

	int fds[2] = { -1, -1 };
	struct ShmTransportOffer offer = {};
	require(composition_export_transport(&c, fds, &offer), "the export failed");
	require(fds[0] == fake.fds[0] && fds[1] == fake.fds[1] && open_fd(fds[0]) && open_fd(fds[1]),
	        "the export's descriptors");
	require(offer.allocation[0] == c.download.allocation && offer.allocation[1] == c.upload.allocation
	        && offer.size[0] == composition_model_bytes(&c) && offer.size[1] == composition_model_bytes(&c)
	        && !memcmp(offer.deviceUuid, DEVICE_UUID, VK_UUID_SIZE)
	        && !memcmp(offer.driverUuid, DRIVER_UUID, VK_UUID_SIZE) && (offer.generation & 1),
	        "the offer's sizes, UUIDs or generation");
	require(!close(fds[0]) && !close(fds[1]), "cannot close the export's descriptors");
	fds[0] = -1;
	fake.fds[0] = -1;
	fds[1] = -1;
	fake.fds[1] = -1;

	composition_set_transport_ready(&c, true);
	require(!composition_transport_pending(&c) && composition_transport_generation(&c) == offer.generation,
	        "an acknowledged pair");

	// The second descriptor fails: the first is closed.
	fake.creates = 0;
	fake.fail_at = 2;
	require(!composition_export_transport(&c, fds, &offer) && fake.fd_count == 3 && !open_fd(fake.fds[2])
	        && fds[0] == -1 && fds[1] == -1, "a failed export left a descriptor open, or in its output");
	fake.fds[2] = -1;
	fake.fail_at = 0;

	composition_disable_export(&c);
	require(!composition_transport_exported(&c) && !composition_transport_pending(&c)
	        && !composition_transport_generation(&c) && composition_proxy_pixels(&c), "host staging after the"
	        " export");
	require_held(&c, "host staging after the export");

	// Host staging has no descriptor to give.
	uint32_t const made = fake.fd_count;
	require(!composition_export_transport(&c, fds, &offer) && fake.fd_count == made && fds[0] == -1
	        && fds[1] == -1, "host staging was exported");
	composition_fini(&c);
	require(!live_objects(), "the fini left %u objects", live_objects());

	// A pair declined while no staging pair can be made drops the surfaces; the next frame builds.
	reset();
	c = built();
	composition_enable_export(&c, 3);
	require(prepare(&c, a) && composition_transport_pending(&c), "an exported pair was not pending");
	fake.creates = 0;
	fake.fail_at = 1;
	composition_disable_export(&c);
	fake.fail_at = 0;
	require(!c.frame.image && !c.download.buffer && !c.upload.buffer, "a pair declined without staging left"
	        " surfaces to record against");
	require_held(&c, "a pair declined without staging");
	require(prepare(&c, a) && composition_proxy_pixels(&c) && composition_model_pixels(&c), "the frame after a"
	        " pair declined without staging did not build");
	composition_fini(&c);
	require(!live_objects(), "the fini left %u objects", live_objects());
}

/** @brief Swapchain formats that the device cannot write. */
static void
check_formats (void)
{
	// Blitted through half float.
	reset();
	struct composition c = built();
	struct arrangement const *const a = &ARRANGEMENTS[0];
	fake.no_storage[0] = VK_FORMAT_B8G8R8A8_UNORM;
	require(prepare(&c, a) && (c.flags & COMPOSITION_BLIT_SWAPCHAIN)
	        && c.work_format == VK_FORMAT_R16G16B16A16_SFLOAT, "a format without storage was not blitted");
	record(&c, a);
	require(fake.blits == 2, "%u blits", fake.blits);
	composition_fini(&c);

	// Neither: the composition gives up for good.
	reset();
	c = built();
	fake.no_storage[0] = VK_FORMAT_B8G8R8A8_UNORM;
	fake.no_storage[1] = VK_FORMAT_R16G16B16A16_SFLOAT;
	require(!prepare(&c, a) && c.error == VK_ERROR_FORMAT_NOT_SUPPORTED && !composition_usable(&c)
	        && !strcmp(composition_reason(&c), "this device cannot write the swapchain's format as a storage"
	                                           " image"), "an unwritable format: %s", composition_reason(&c));
	uint32_t const calls = fake.calls;
	fake.no_storage[0] = fake.no_storage[1] = VK_FORMAT_UNDEFINED;
	require(!prepare(&c, a) && fake.calls == calls, "an unusable composition asked the device again");
	composition_fini(&c);
	require(!live_objects(), "the fini left %u objects", live_objects());

	// Not a format the pass works in, now or later.
	reset();
	c = built();
	struct arrangement unsupported = *a;
	unsupported.format = VK_FORMAT_R5G6B5_UNORM_PACK16;
	require(!prepare(&c, &unsupported) && composition_usable(&c)
	        && !strcmp(composition_reason(&c), "unsupported swapchain format") && prepare(&c, a)
	        && !*composition_reason(&c), "an unsupported swapchain format: %s", composition_reason(&c));
	composition_fini(&c);

	// No float16 that the shader can sample: the proxy stays 8-bit.
	reset();
	c = built();
	fake.features &= ~(VkFormatFeatureFlags)VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
	require(prepare(&c, &ARRANGEMENTS[3]) && !composition_hdr_proxy_active(&c), "a float16 proxy without"
	        " sampling");
	composition_fini(&c);
	require(!live_objects(), "the fini left %u objects", live_objects());
}

/** @brief Empty and absent compositions own nothing, call nothing and answer nothing. */
static void
check_empty (void)
{
	reset();
	struct composition c = composition_empty();
	struct composition_frame_settings const s = composition_frame_settings();
	composition_fini(&c);
	composition_fini(nullptr);
	require(!composition_usable(&c) && !composition_usable(nullptr) && !*composition_reason(&c)
	        && !composition_reason(nullptr), "an empty composition is usable or has a reason");
	require(!composition_prepare(&c, WIDTH, HEIGHT, VK_FORMAT_B8G8R8A8_UNORM, &s, false, false, 0)
	        && !composition_prepare(nullptr, WIDTH, HEIGHT, VK_FORMAT_B8G8R8A8_UNORM, &s, false, false, 0)
	        && !composition_record_capture(&c, CMD, SWAPCHAIN, &s)
	        && !composition_record_capture(nullptr, CMD, SWAPCHAIN, &s)
	        && !composition_record_compose(&c, CMD, SWAPCHAIN, &s)
	        && !composition_record_compose(nullptr, CMD, SWAPCHAIN, &s), "an empty composition ran");
	require(!composition_model_width(nullptr) && !composition_model_height(nullptr)
	        && !composition_model_bytes(nullptr) && !composition_hdr_proxy_active(nullptr)
	        && !composition_transport_pending(nullptr) && !composition_transport_generation(nullptr)
	        && !composition_proxy_pixels(nullptr) && !composition_proxy_buffer(nullptr)
	        && !composition_answer_buffer(nullptr) && !composition_transport_exported(nullptr)
	        && !composition_model_pixels(nullptr) && !composition_capture_recorded(nullptr)
	        && composition_measured_white_point(nullptr) == 0.0f, "an absent composition answered");
	composition_enable_export(nullptr, 0);
	composition_await_answer(nullptr, -1);
	composition_withdraw_offer(nullptr);
	composition_set_transport_ready(nullptr, true);
	composition_disable_export(nullptr);
	composition_disable_export(&c);
	composition_request_capture(nullptr, 1, 1);
	composition_set_capture_inference(nullptr, 1);
	composition_write_captured_frame(nullptr);
	composition_write_captured_frame(&c);
	composition_consume_meter(nullptr);
	composition_consume_meter(&c);
	int fds[2] = { -1, -1 };
	require(!composition_export_transport(nullptr, fds, &(struct ShmTransportOffer){})
	        && !composition_export_transport(&c, fds, &(struct ShmTransportOffer){}) && fds[0] == -1
	        && fds[1] == -1, "an empty or absent composition exported");
	require(!fake.calls && empty(&c), "an empty composition called the device %u times",
	        fake.calls);
	require(composition_init(nullptr, &device_table, &instance_table, DEVICE, PHYSICAL)
	        == VK_ERROR_INITIALIZATION_FAILED, "an init without a destination");
}

int
main (void)
{
	tables();
	check_empty();
	check_build_failures();
	check_arrangements();
	check_prepare_failures();
	check_kept_frame();
	check_meter_clear();
	check_capture_pair();
	check_offer();
	check_export();
	check_formats();
	reset();
	puts("PASS: the composition builds, records and frees what it creates, after a failure at any step too");
	return 0;
}
