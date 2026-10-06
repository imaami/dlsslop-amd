/** @file
 *
 * Host test of the network's motion history across frames and of its reshapes, without a GPU.
 *
 * The recorder builds the network for 64x64 frames, from a synthetic model and the build's SPIR-V, on
 * a fake device whose functions log the commands recorded, each handle named by its first use in the
 * frame and described by what it was made as, and each descriptor set by what it holds. A frame that
 * is recorded and not submitted must leave the history as it was: the next frame's commands, with
 * its motion parameters and push constants, must equal those of the frame recorded before it. The
 * first frame after a build moves the network's images into their layouts, and so does the next one
 * when that frame was not submitted. A network reshaped for another shape of its extent must record
 * no command and make no pipeline, and then record the frames that a network built for that shape
 * records. A frame of one pass without the pass stages must copy the proxy straight into the
 * network's input and its answer straight out, and copy or blit no image. From a model whose weights
 * free every Swin layer of the exponent's upper clamp, a frame must run the kernels and the temporal
 * pre block without that clamp. Every frame must judge its waits after the network's last dispatch,
 * with the plan's words that waits set when they run out, and answer with its input only over the
 * grid that verdict writes: the first pass's input into the image that the answer is then copied out
 * of. A verdict that says a wait ran out, poked into the fake device's memory, must start the next
 * frame over: its arena zeroed from the end of the values on, between barriers, and its motion
 * history dropped, as must the frame after it when that frame was not submitted. A network built for
 * frames in a caller's images binds them only when their generation changes, which keeps the motion
 * history; its frames copy no buffer, and blit at most once, into the caller's answer. Takes the
 * directory of the network's SPIR-V.
 *
 * With --build, it only builds the network for one extent, which makes every pipeline of the
 * runtime's own but the temporal pre block that the plan's pre block is not, so that
 * tests/vulkan-files.py can see the files that a build opens; with --unclamped too, from a synthetic
 * model whose weights free every Swin layer of the exponent's upper clamp.
 */
// SPDX-License-Identifier: MIT
#include <getopt.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "error.h"
#include "network/network_fallback.h"
#include "network/network_verdict.h"
#include "network_recorder.h"
#include "support.h"
#include "vulkan_frame.h"
#include "vulkan_pack.h"
#include "vulkan_plan.h"
#include "vulkan_runtime.h"

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param ok  The condition.
 * @param fmt A printf format for the message.
 * @param ... The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        ok,
         char const *fmt,
         ...)
{
	if (ok)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("network-recorder test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief A string's bytes, which need not end in a null. */
struct piece {
	char const *at;     //!< The bytes.
	size_t      length; //!< Their number.
};

/** @brief A string literal as a piece, measured by its size. */
#define PIECE(literal) ((struct piece){(literal), sizeof (literal) - 1})

/** @brief Whether a piece starts with another.
 *
 * @param s    The piece.
 * @param head The start.
 * @return     true if it does.
 */
static bool
starts (struct piece s,
        struct piece head)
{
	return s.length >= head.length && !memcmp(s.at, head.at, head.length);
}

/** @brief Whether a piece ends with another.
 *
 * @param s    The piece.
 * @param tail The end.
 * @return     true if it does.
 */
static bool
ends (struct piece s,
      struct piece tail)
{
	return s.length >= tail.length && !memcmp(s.at + s.length - tail.length, tail.at, tail.length);
}

/** @brief Whether two pieces hold the same bytes.
 *
 * @param a A piece.
 * @param b Another.
 * @return  true if they do.
 */
static bool
equals (struct piece a,
        struct piece b)
{
	return a.length == b.length && !memcmp(a.at, b.at, a.length);
}

/** @brief A text's words as a piece.
 *
 * @param text The text.
 * @return     Its words.
 */
static struct piece
whole (struct support_text const *text)
{
	return (struct piece){support_text_string(text), text->length};
}

/** @brief The FNV-1a 64 of bytes.
 *
 * @param data  The bytes.
 * @param bytes Their number.
 * @return      The hash.
 */
static uint64_t
fnv1a (void const *data,
       size_t      bytes)
{
	return vulkan_pack_fnv1a(VULKAN_PACK_FNV1A_BASIS, data, bytes);
}

/* The fake device's handles are numbers from 1, each an object of the table below: the sizes of its
 * buffers and allocations, and host memory for the allocations mapped; what each image, buffer,
 * sampler, shader module and pipeline was made as, each view's image, and each descriptor set's
 * descriptors by binding. The runtime makes pipelines on several threads at once: the functions
 * that make shader modules and pipelines use the table under a lock. */

/** @brief A descriptor that a set holds. */
struct descriptor {
	uint64_t      resource; //!< The buffer, or the image that the view is of.
	uint64_t      sampler;  //!< The sampler, or 0.
	VkImageLayout layout;   //!< The image's layout.
};

/** @brief The most bindings a set of the fake device holds. */
static constexpr uint32_t SET_BINDINGS = 16;

/** @brief What the fake device knows of a handle. */
struct object {
	VkDeviceSize       size;     //!< A buffer's or an allocation's bytes.
	uint64_t           image;    //!< A view's image.
	uint64_t           named_in; //!< The frame that named it last.
	uint8_t           *host;     //!< Its memory, mapped, or nullptr.
	struct descriptor *bindings; //!< A set's descriptors by binding, or nullptr.
	size_t             name;     //!< Its name in that frame.
	uint32_t           present;  //!< A set's bindings written, bit b for binding b.
	char               made[48]; //!< What it was made as.
};

/** @brief The last handle made. */
static atomic_uint_fast64_t handles;

/** @brief Guards the objects while pipelines are made. */
static pthread_mutex_t making = PTHREAD_MUTEX_INITIALIZER;

/** @brief The objects by handle, from 0, which no handle is. */
static struct object *objects;

/** @brief The objects' number. */
static size_t object_count;

/** @brief The pipelines made so far. */
static size_t pipelines_made;

/** @brief The descriptors written so far. */
static size_t descriptors_written;

/** @brief Those of the descriptors written that named no image view. */
static size_t null_views;

/** @brief A handle as a number.
 *
 * @param handle The handle.
 * @return       Its number.
 */
#define ID(handle) ((uint64_t)(uintptr_t)(handle))

/** @brief Makes a handle: the next number. */
#define MAKE(handle) (*(handle) = (typeof(*(handle)))(uintptr_t)(atomic_fetch_add(&handles, 1) + 1))

/** @brief What the fake device knows of a handle, which it learns of now if it is new.
 *
 * @param handle The handle.
 * @return       Its object.
 */
static struct object *
object (uint64_t handle)
{
	if (handle >= object_count) {
		size_t count = object_count ? 2 * object_count : 4096;
		while (count <= handle)
			count *= 2;
		struct object *const grown = realloc(objects, count * sizeof *grown);
		require(grown, "out of memory");
		memset(grown + object_count, 0, (count - object_count) * sizeof *grown);
		objects = grown;
		object_count = count;
	}
	return &objects[handle];
}

/** @brief Says what a handle was made as.
 *
 * @param handle The handle.
 * @param fmt    A printf format for the words.
 * @param ...    The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
made_as (uint64_t    handle,
         char const *fmt,
         ...)
{
	struct object *const o = object(handle);
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(o->made, sizeof o->made, fmt, args);
	va_end(args);
	require(n >= 0 && n < (int)sizeof o->made, "what a handle was made as does not fit");
}

/** @brief Frees what the fake device knows. */
static void
objects_fini (void)
{
	for (size_t i = 0; i < object_count; ++i) {
		free(objects[i].host);
		objects[i].host = nullptr;
		free(objects[i].bindings);
		objects[i].bindings = nullptr;
	}
	free(objects);
	objects = nullptr;
	object_count = 0;
}

/** @brief What a frame recorded: its commands, a line each, the descriptor sets it bound, and the
 *         gate of its motion parameters and the seed in the push constants of its pre block, the
 *         first dispatch after the parameters. pre is as wide as a pointer, which fills the padding. */
struct frame {
	uint64_t            id;        //!< Which frame it is: a handle's name holds in this one.
	struct support_text commands;  //!< The commands.
	uint64_t           *sets;      //!< The descriptor sets bound, in order.
	size_t              set_count; //!< Their number.
	size_t              set_room;  //!< The sets allocated.
	size_t              names;     //!< The handles it named.
	uintptr_t           pre;       //!< Whether the next push constants are the pre block's: 1 if they are.
	uint32_t            seed;      //!< The pre block's seed.
	float               gate;      //!< The motion parameters' gate.
};

/** @brief The frame being recorded. */
static struct frame frame;

/** @brief The frames recorded so far. */
static uint64_t frames;

/** @brief Frees a frame.
 *
 * @param f The frame.
 */
static void
frame_fini (struct frame *f)
{
	support_text_fini(&f->commands);
	free(f->sets);
	f->sets = nullptr;
	*f = (struct frame){};
}

/** @brief Starts a frame: nothing recorded, nothing named. What the device records between frames,
 *         such as a build's commands, goes into a frame that the next start drops. */
static void
frame_start (void)
{
	frame_fini(&frame);
	frame = (struct frame){.id = ++frames, .seed = UINT32_MAX, .gate = -1};
}

/** @brief The frame recorded, which the caller then owns; another starts.
 *
 * @return The frame.
 */
static struct frame
frame_take (void)
{
	struct frame const taken = frame;
	frame = (struct frame){};
	frame_start();
	return taken;
}

/** @brief Appends words to the frame's commands.
 *
 * @param fmt A printf format.
 * @param ... The format's arguments.
 */
[[gnu::format(printf, 1, 2)]]
static void
say (char const *fmt,
     ...)
{
	va_list args;
	va_start(args, fmt);
	bool const said = support_text_vprintf(&frame.commands, fmt, args);
	va_end(args);
	require(said, "out of memory");
}

/** @brief Appends a handle to the frame's commands as the frame names it: by its first use, which
 *         says what it was made as.
 *
 * @param prefix What goes before the name.
 * @param handle The handle.
 */
static void
say_name (char const *prefix,
          uint64_t    handle)
{
	struct object *const o = object(handle);
	if (o->named_in == frame.id) {
		require(support_text_printf(&frame.commands, "%s#%zu", prefix, o->name), "out of memory");
		return;
	}
	o->named_in = frame.id;
	o->name = frame.names++;
	require(support_text_printf(&frame.commands, "%s#%zu(%s)", prefix, o->name, o->made), "out of memory");
}

/** @brief Appends 32-bit words to the frame's commands, each after a space.
 *
 * @param words The words.
 * @param count Their number.
 */
static void
say_words (uint32_t const *words,
           size_t          count)
{
	for (size_t i = 0; i < count; ++i)
		require(support_text_printf(&frame.commands, " %" PRIu32, words[i]), "out of memory");
}

/* The fake device: the loader's functions that the runtime calls. */

/** @brief Defines a fake vkCreate function that makes a handle and nothing else. */
#define MAKER(name, info_type, handle_type) \
	VKAPI_ATTR VkResult VKAPI_CALL \
	name (VkDevice, \
	      info_type const *, \
	      VkAllocationCallbacks const *, \
	      handle_type *handle) \
	{ \
		MAKE(handle); \
		return VK_SUCCESS; \
	}
MAKER(vkCreateDescriptorSetLayout, VkDescriptorSetLayoutCreateInfo, VkDescriptorSetLayout)
MAKER(vkCreatePipelineLayout, VkPipelineLayoutCreateInfo, VkPipelineLayout)
MAKER(vkCreatePipelineCache, VkPipelineCacheCreateInfo, VkPipelineCache)
MAKER(vkCreateDescriptorPool, VkDescriptorPoolCreateInfo, VkDescriptorPool)
MAKER(vkCreateCommandPool, VkCommandPoolCreateInfo, VkCommandPool)
MAKER(vkCreateFence, VkFenceCreateInfo, VkFence)
#undef MAKER

/** @brief Defines a fake vkDestroy function that does nothing: the test never reuses a handle. */
#define DESTROYER(name, handle_type) \
	VKAPI_ATTR void VKAPI_CALL \
	name (VkDevice, \
	      handle_type, \
	      VkAllocationCallbacks const *) \
	{ \
	}
DESTROYER(vkDestroyBuffer, VkBuffer)
DESTROYER(vkDestroyImage, VkImage)
DESTROYER(vkDestroyImageView, VkImageView)
DESTROYER(vkDestroySampler, VkSampler)
DESTROYER(vkDestroyDescriptorSetLayout, VkDescriptorSetLayout)
DESTROYER(vkDestroyPipelineLayout, VkPipelineLayout)
DESTROYER(vkDestroyPipeline, VkPipeline)
DESTROYER(vkDestroyShaderModule, VkShaderModule)
DESTROYER(vkDestroyPipelineCache, VkPipelineCache)
DESTROYER(vkDestroyDescriptorPool, VkDescriptorPool)
DESTROYER(vkDestroyCommandPool, VkCommandPool)
DESTROYER(vkDestroyFence, VkFence)
#undef DESTROYER

/** @brief The fake vkFreeMemory(): drops the memory's host copy. */
VKAPI_ATTR void VKAPI_CALL
vkFreeMemory (VkDevice,
              VkDeviceMemory                memory,
              VkAllocationCallbacks const *)
{
	struct object *const o = object(ID(memory));
	free(o->host);
	o->host = nullptr;
}

/** @brief The fake vkCreateBuffer(): makes a buffer and says its size. */
VKAPI_ATTR VkResult VKAPI_CALL
vkCreateBuffer (VkDevice,
                VkBufferCreateInfo const    *info,
                VkAllocationCallbacks const *,
                VkBuffer                    *buffer)
{
	MAKE(buffer);
	object(ID(*buffer))->size = info->size;
	made_as(ID(*buffer), "buffer %" PRIu64, (uint64_t)info->size);
	return VK_SUCCESS;
}

/** @brief The fake vkCreateImage(): makes an image and says what it is. */
VKAPI_ATTR VkResult VKAPI_CALL
vkCreateImage (VkDevice,
               VkImageCreateInfo const     *info,
               VkAllocationCallbacks const *,
               VkImage                     *image)
{
	MAKE(image);
	made_as(ID(*image), "image %" PRIu32 "x%" PRIu32 " format %d usage %" PRIu32, info->extent.width,
	        info->extent.height, (int)info->format, (uint32_t)info->usage);
	return VK_SUCCESS;
}

/** @brief The fake vkCreateImageView(): makes a view of an image. */
VKAPI_ATTR VkResult VKAPI_CALL
vkCreateImageView (VkDevice,
                   VkImageViewCreateInfo const *info,
                   VkAllocationCallbacks const *,
                   VkImageView                 *view)
{
	MAKE(view);
	object(ID(*view))->image = ID(info->image);
	return VK_SUCCESS;
}

/** @brief The fake vkCreateSampler(): makes a sampler and says its filter and addressing. */
VKAPI_ATTR VkResult VKAPI_CALL
vkCreateSampler (VkDevice,
                 VkSamplerCreateInfo const   *info,
                 VkAllocationCallbacks const *,
                 VkSampler                   *sampler)
{
	MAKE(sampler);
	made_as(ID(*sampler), "sampler %d %d", (int)info->magFilter, (int)info->addressModeU);
	return VK_SUCCESS;
}

/** @brief The fake vkCreateShaderModule(): makes a module named by its SPIR-V's hash. */
VKAPI_ATTR VkResult VKAPI_CALL
vkCreateShaderModule (VkDevice,
                      VkShaderModuleCreateInfo const *info,
                      VkAllocationCallbacks const    *,
                      VkShaderModule                 *module)
{
	MAKE(module);
	require(!pthread_mutex_lock(&making), "cannot lock the fake device");
	made_as(ID(*module), "pipeline %" PRIu64, fnv1a(info->pCode, info->codeSize));
	require(!pthread_mutex_unlock(&making), "cannot unlock the fake device");
	return VK_SUCCESS;
}

/** @brief The fake vkAllocateMemory(): makes an allocation of a size. */
VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateMemory (VkDevice,
                  VkMemoryAllocateInfo const  *info,
                  VkAllocationCallbacks const *,
                  VkDeviceMemory              *memory)
{
	MAKE(memory);
	object(ID(*memory))->size = info->allocationSize;
	return VK_SUCCESS;
}

/** @brief The fake vkMapMemory(): gives the allocation zeroed host memory, the same each time. */
VKAPI_ATTR VkResult VKAPI_CALL
vkMapMemory (VkDevice,
             VkDeviceMemory   memory,
             VkDeviceSize,
             VkDeviceSize,
             VkMemoryMapFlags,
             void           **data)
{
	struct object *const o = object(ID(memory));
	if (!o->host) {
		o->host = calloc(o->size ? o->size : 1, 1);
		require(o->host, "out of memory");
	}
	*data = o->host;
	return VK_SUCCESS;
}

/** @brief The fake vkGetBufferMemoryRequirements(): a buffer's size. */
VKAPI_ATTR void VKAPI_CALL
vkGetBufferMemoryRequirements (VkDevice,
                               VkBuffer              buffer,
                               VkMemoryRequirements *req)
{
	*req = (VkMemoryRequirements){object(ID(buffer))->size, 256, 3};
}

/** @brief The fake vkGetImageMemoryRequirements(): 256 bytes. */
VKAPI_ATTR void VKAPI_CALL
vkGetImageMemoryRequirements (VkDevice,
                              VkImage,
                              VkMemoryRequirements *req)
{
	*req = (VkMemoryRequirements){256, 256, 3};
}

/** @brief The fake vkBindBufferMemory(): succeeds. */
VKAPI_ATTR VkResult VKAPI_CALL
vkBindBufferMemory (VkDevice,
                    VkBuffer,
                    VkDeviceMemory,
                    VkDeviceSize)
{
	return VK_SUCCESS;
}

/** @brief The fake vkBindImageMemory(): succeeds. */
VKAPI_ATTR VkResult VKAPI_CALL
vkBindImageMemory (VkDevice,
                   VkImage,
                   VkDeviceMemory,
                   VkDeviceSize)
{
	return VK_SUCCESS;
}

/** @brief The fake vkCreateComputePipelines(): makes pipelines named as their modules. */
VKAPI_ATTR VkResult VKAPI_CALL
vkCreateComputePipelines (VkDevice,
                          VkPipelineCache,
                          uint32_t                           count,
                          VkComputePipelineCreateInfo const *infos,
                          VkAllocationCallbacks const       *,
                          VkPipeline                        *pipelines)
{
	require(!pthread_mutex_lock(&making), "cannot lock the fake device");
	for (uint32_t i = 0; i < count; ++i) {
		MAKE(&pipelines[i]);
		// The pipeline first: a new handle can grow the objects.
		struct object *const pipeline = object(ID(pipelines[i]));
		struct object const *const module = object(ID(infos[i].stage.module));
		memcpy(pipeline->made, module->made, sizeof pipeline->made);
		++pipelines_made;
	}
	require(!pthread_mutex_unlock(&making), "cannot unlock the fake device");
	return VK_SUCCESS;
}

/** @brief The fake vkGetPipelineCacheData(): an empty cache. */
VKAPI_ATTR VkResult VKAPI_CALL
vkGetPipelineCacheData (VkDevice,
                        VkPipelineCache,
                        size_t *bytes,
                        void   *)
{
	*bytes = 0;
	return VK_SUCCESS;
}

/** @brief The fake vkAllocateDescriptorSets(): makes sets. */
VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateDescriptorSets (VkDevice,
                          VkDescriptorSetAllocateInfo const *info,
                          VkDescriptorSet                   *sets)
{
	for (uint32_t i = 0; i < info->descriptorSetCount; ++i)
		MAKE(&sets[i]);
	return VK_SUCCESS;
}

/** @brief The fake vkUpdateDescriptorSets(): keeps each set's descriptors by binding. */
VKAPI_ATTR void VKAPI_CALL
vkUpdateDescriptorSets (VkDevice,
                        uint32_t                    count,
                        VkWriteDescriptorSet const *writes,
                        uint32_t,
                        VkCopyDescriptorSet const  *)
{
	for (uint32_t i = 0; i < count; ++i) {
		VkWriteDescriptorSet const *const w = &writes[i];
		++descriptors_written;
		null_views += w->pImageInfo && !w->pImageInfo->imageView;
		require(w->dstBinding < SET_BINDINGS, "a set has too many bindings for the fake device");
		struct descriptor const d = w->pBufferInfo
		                          ? (struct descriptor){ID(w->pBufferInfo->buffer), 0,
		                                                VK_IMAGE_LAYOUT_UNDEFINED}
		                          : (struct descriptor){object(ID(w->pImageInfo->imageView))->image,
		                                                ID(w->pImageInfo->sampler), w->pImageInfo->imageLayout};
		struct object *const set = object(ID(w->dstSet));
		if (!set->bindings) {
			set->bindings = calloc(SET_BINDINGS, sizeof *set->bindings);
			require(set->bindings, "out of memory");
		}
		set->bindings[w->dstBinding] = d;
		set->present |= UINT32_C(1) << w->dstBinding;
	}
}

/** @brief The fake vkAllocateCommandBuffers(): makes a command buffer. */
VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateCommandBuffers (VkDevice,
                          VkCommandBufferAllocateInfo const *,
                          VkCommandBuffer                   *cmd)
{
	MAKE(cmd);
	return VK_SUCCESS;
}

/** @brief The fake vkBeginCommandBuffer(): succeeds. */
VKAPI_ATTR VkResult VKAPI_CALL
vkBeginCommandBuffer (VkCommandBuffer,
                      VkCommandBufferBeginInfo const *)
{
	return VK_SUCCESS;
}

/** @brief The fake vkEndCommandBuffer(): succeeds. */
VKAPI_ATTR VkResult VKAPI_CALL
vkEndCommandBuffer (VkCommandBuffer)
{
	return VK_SUCCESS;
}

/** @brief The fake vkQueueSubmit(): succeeds. */
VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit (VkQueue,
               uint32_t,
               VkSubmitInfo const *,
               VkFence)
{
	return VK_SUCCESS;
}

/** @brief The fake vkWaitForFences(): succeeds at once. */
VKAPI_ATTR VkResult VKAPI_CALL
vkWaitForFences (VkDevice,
                 uint32_t,
                 VkFence const *,
                 VkBool32,
                 uint64_t)
{
	return VK_SUCCESS;
}

/* The commands a frame records. */

/** @brief The fake vkCmdPipelineBarrier(): logs the stages and each barrier. */
VKAPI_ATTR void VKAPI_CALL
vkCmdPipelineBarrier (VkCommandBuffer,
                      VkPipelineStageFlags         src,
                      VkPipelineStageFlags         dst,
                      VkDependencyFlags,
                      uint32_t                     memories,
                      VkMemoryBarrier const       *memory,
                      uint32_t                     buffers,
                      VkBufferMemoryBarrier const *buffer,
                      uint32_t                     images,
                      VkImageMemoryBarrier const  *image)
{
	say("barrier %" PRIu32 " %" PRIu32 " %" PRIu32 "\n", (uint32_t)src, (uint32_t)dst, memories);
	for (uint32_t i = 0; i < memories; ++i)
		say(" memory %" PRIu32 " %" PRIu32 "\n", (uint32_t)memory[i].srcAccessMask,
		    (uint32_t)memory[i].dstAccessMask);
	for (uint32_t i = 0; i < buffers; ++i) {
		say_name(" buffer ", ID(buffer[i].buffer));
		say(" %" PRIu32 "\n", (uint32_t)buffer[i].srcAccessMask);
	}
	for (uint32_t i = 0; i < images; ++i) {
		say_name(" image ", ID(image[i].image));
		say(" %d %d\n", (int)image[i].oldLayout, (int)image[i].newLayout);
	}
}

/** @brief The fake vkCmdCopyBuffer(): logs the buffers. */
VKAPI_ATTR void VKAPI_CALL
vkCmdCopyBuffer (VkCommandBuffer,
                 VkBuffer            from,
                 VkBuffer            to,
                 uint32_t,
                 VkBufferCopy const *)
{
	say_name("copy buffer ", ID(from));
	say_name(" ", ID(to));
	say("\n");
}

/** @brief The fake vkCmdFillBuffer(): logs the buffer and the range. */
VKAPI_ATTR void VKAPI_CALL
vkCmdFillBuffer (VkCommandBuffer,
                 VkBuffer     buffer,
                 VkDeviceSize offset,
                 VkDeviceSize bytes,
                 uint32_t)
{
	say_name("fill ", ID(buffer));
	say(" %" PRIu64 " %" PRIu64 "\n", (uint64_t)offset, (uint64_t)bytes);
}

/** @brief The fake vkCmdCopyImage(): logs the images. */
VKAPI_ATTR void VKAPI_CALL
vkCmdCopyImage (VkCommandBuffer,
                VkImage            from,
                VkImageLayout,
                VkImage            to,
                VkImageLayout,
                uint32_t,
                VkImageCopy const *)
{
	say_name("copy image ", ID(from));
	say_name(" ", ID(to));
	say("\n");
}

/** @brief The fake vkCmdBlitImage(): logs the images. */
VKAPI_ATTR void VKAPI_CALL
vkCmdBlitImage (VkCommandBuffer,
                VkImage            from,
                VkImageLayout,
                VkImage            to,
                VkImageLayout,
                uint32_t,
                VkImageBlit const *,
                VkFilter)
{
	say_name("blit ", ID(from));
	say_name(" ", ID(to));
	say("\n");
}

/** @brief The fake vkCmdCopyBufferToImage(): logs the buffer and the image. */
VKAPI_ATTR void VKAPI_CALL
vkCmdCopyBufferToImage (VkCommandBuffer,
                        VkBuffer                 from,
                        VkImage                  to,
                        VkImageLayout,
                        uint32_t,
                        VkBufferImageCopy const *)
{
	say_name("copy buffer to image ", ID(from));
	say_name(" ", ID(to));
	say("\n");
}

/** @brief The fake vkCmdCopyImageToBuffer(): logs the image and the buffer. */
VKAPI_ATTR void VKAPI_CALL
vkCmdCopyImageToBuffer (VkCommandBuffer,
                        VkImage                  from,
                        VkImageLayout,
                        VkBuffer                 to,
                        uint32_t,
                        VkBufferImageCopy const *)
{
	say_name("copy image to buffer ", ID(from));
	say_name(" ", ID(to));
	say("\n");
}

/** @brief The fake vkCmdBindPipeline(): logs the pipeline. */
VKAPI_ATTR void VKAPI_CALL
vkCmdBindPipeline (VkCommandBuffer,
                   VkPipelineBindPoint,
                   VkPipeline pipeline)
{
	say_name("pipeline ", ID(pipeline));
	say("\n");
}

/** @brief The fake vkCmdBindDescriptorSets(): logs each set with what it holds. */
VKAPI_ATTR void VKAPI_CALL
vkCmdBindDescriptorSets (VkCommandBuffer,
                         VkPipelineBindPoint,
                         VkPipelineLayout,
                         uint32_t,
                         uint32_t               count,
                         VkDescriptorSet const *sets,
                         uint32_t,
                         uint32_t const        *)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (frame.set_count == frame.set_room) {
			size_t const room = frame.set_room ? 2 * frame.set_room : 256;
			uint64_t *const grown = realloc(frame.sets, room * sizeof *grown);
			require(grown, "out of memory");
			frame.sets = grown;
			frame.set_room = room;
		}
		frame.sets[frame.set_count++] = ID(sets[i]);
		say("set");
		struct object const *const set = object(ID(sets[i]));
		uint32_t const present = set->present;
		struct descriptor const *const bindings = set->bindings;
		for (uint32_t b = 0; b < SET_BINDINGS; ++b) {
			if (!(present >> b & 1))
				continue;
			struct descriptor const d = bindings[b];
			say(" %" PRIu32 "=", b);
			say_name("", d.resource);
			if (d.sampler)
				say_name("/", d.sampler);
			say("/%d", (int)d.layout);
		}
		say("\n");
	}
}

/** @brief The fake vkCmdPushConstants(): logs the words, and takes the pre block's seed. */
VKAPI_ATTR void VKAPI_CALL
vkCmdPushConstants (VkCommandBuffer,
                    VkPipelineLayout,
                    VkShaderStageFlags,
                    uint32_t,
                    uint32_t    bytes,
                    void const *data)
{
	uint32_t words[64];
	size_t const count = bytes / 4;
	require(count <= sizeof words / sizeof *words, "a push range is too long for the fake device");
	memcpy(words, data, count * sizeof *words);
	say("push");
	say_words(words, count);
	say("\n");
	constexpr size_t seed = (sizeof (struct push_f_swin) + offsetof(struct push_pre_image, seed)) / 4;
	if (frame.pre && seed < count)
		frame.seed = words[seed];
	frame.pre = 0;
}

/** @brief The fake vkCmdDispatch(): logs the groups. */
VKAPI_ATTR void VKAPI_CALL
vkCmdDispatch (VkCommandBuffer,
               uint32_t x,
               uint32_t y,
               uint32_t z)
{
	say("dispatch %" PRIu32 " %" PRIu32 " %" PRIu32 "\n", x, y, z);
}

/** @brief The fake vkCmdDispatchIndirect(): logs the buffer and the offset. */
VKAPI_ATTR void VKAPI_CALL
vkCmdDispatchIndirect (VkCommandBuffer,
                       VkBuffer     buffer,
                       VkDeviceSize offset)
{
	say_name("dispatch indirect ", ID(buffer));
	say(" %" PRIu64 "\n", (uint64_t)offset);
}

/** @brief The fake vkCmdUpdateBuffer(): logs the words, and takes the motion parameters' gate. */
VKAPI_ATTR void VKAPI_CALL
vkCmdUpdateBuffer (VkCommandBuffer,
                   VkBuffer     buffer,
                   VkDeviceSize,
                   VkDeviceSize bytes,
                   void const  *data)
{
	uint32_t words[64];
	size_t const count = bytes / 4;
	require(count <= sizeof words / sizeof *words, "an update is too long for the fake device");
	memcpy(words, data, count * sizeof *words);
	say_name("update ", ID(buffer));
	say_words(words, count);
	say("\n");
	memcpy(&frame.gate, data, sizeof frame.gate);
	frame.pre = 1;
}

/** @brief The fake vkCmdWriteTimestamp(): logs it. */
VKAPI_ATTR void VKAPI_CALL
vkCmdWriteTimestamp (VkCommandBuffer,
                     VkPipelineStageFlagBits,
                     VkQueryPool,
                     uint32_t)
{
	say("timestamp\n");
}

/* The physical device: one queue family for everything, every format feature and storage buffers
 * of 4 GiB. The runtime chains at most one structure to a query, which the fakes access through its
 * own type: the link-time optimizer inlines them into the runtime, and access through
 * VkBaseOutStructure breaks the aliasing rules it relies on. */

/** @brief The fake device's queue families: one for everything. */
static void VKAPI_CALL
queue_families (VkPhysicalDevice,
                uint32_t                *count,
                VkQueueFamilyProperties *families)
{
	*count = 1;
	if (families)
		families[0] = (VkQueueFamilyProperties){
			VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1, 64, {1, 1, 1}};
}

/** @brief The fake device's properties: storage buffers of 4 GiB. */
static void VKAPI_CALL
properties (VkPhysicalDevice,
            VkPhysicalDeviceProperties2 *properties)
{
	properties->properties.limits.maxStorageBufferRange = UINT32_MAX;
	VkPhysicalDeviceMaintenance3Properties *const allocation = properties->pNext;
	if (allocation && allocation->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
		allocation->maxMemoryAllocationSize = UINT64_C(1) << 32;
}

/** @brief The fake device's format properties: every feature. */
static void VKAPI_CALL
format_properties (VkPhysicalDevice,
                   VkFormat,
                   VkFormatProperties2 *properties)
{
	properties->formatProperties.optimalTilingFeatures = ~(VkFormatFeatureFlags)0;
	VkFormatProperties3 *const features = properties->pNext;
	if (features && features->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3)
		features->optimalTilingFeatures = ~(VkFormatFeatureFlags2)0;
}

/** @brief The fake device.
 *
 * @return The device.
 */
static struct vulkan_device
fake_device (void)
{
	struct vulkan_device d = {};
	MAKE(&d.instance);
	MAKE(&d.physical);
	MAKE(&d.device);
	MAKE(&d.queue);
	d.memory.memoryTypeCount = 2;
	d.memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	d.memory.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
	                                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	d.memory.memoryHeapCount = 1;
	d.functions = (struct vulkan_physical_functions){queue_families, properties, format_properties};
	return d;
}

/** @brief Makes a recorder on a device, or ends the test.
 *
 * @param dest     Receives the recorder.
 * @param device   The device.
 * @param paths    Where the network's files are.
 * @param external Frames go through a caller's images.
 */
static void
recorder_init (struct network_recorder    *dest,
               struct vulkan_device const *device,
               struct vulkan_paths const  *paths,
               bool                        external)
{
	struct error e;
	require(network_recorder_init(dest, device, paths, external, &e) == ERROR_NONE, "%s", e.what);
}

/** @brief Shapes a recorder for a frame.
 *
 * @param recorder The recorder.
 * @param f        The frame.
 * @param images   The caller's images, or nullptr.
 * @param e        Receives the words for what stopped it.
 * @return         Whether it built, reshaped or bound; false on a failure.
 */
static bool
shape (struct network_recorder          *recorder,
       struct vulkan_frame const        *f,
       struct vulkan_frame_images const *images,
       struct error                     *e)
{
	bool changed = false;
	return network_recorder_shape(recorder, f, images, &changed, e) == ERROR_NONE && changed;
}

/** @brief A frame recorded by a recorder, from and into the same buffers each time.
 *
 * @param recorder The recorder.
 * @param f        The frame.
 * @return         What it recorded, which the caller frees.
 */
static struct frame
record (struct network_recorder   *recorder,
        struct vulkan_frame const *f)
{
	static VkCommandBuffer cmd;
	static VkBuffer proxy, answer;
	if (!cmd) {
		MAKE(&cmd);
		MAKE(&proxy);
		MAKE(&answer);
	}
	frame_start();
	network_recorder_record_buffers(recorder, cmd, proxy, answer, f, 0, false, VK_NULL_HANDLE, 0);
	return frame_take();
}

/** @brief How many lines of commands start with a head and end with a tail.
 *
 * @param commands The commands.
 * @param head     The start.
 * @param tail     The end.
 * @return         The lines.
 */
static size_t
lines (struct piece commands,
       struct piece head,
       struct piece tail)
{
	size_t n = 0;
	while (commands.length) {
		char const *const newline = memchr(commands.at, '\n', commands.length);
		size_t const length = newline ? (size_t)(newline - commands.at) : commands.length;
		struct piece const line = {commands.at, length};
		n += starts(line, head) && ends(line, tail);
		size_t const skipped = length < commands.length ? length + 1 : length;
		commands.at += skipped;
		commands.length -= skipped;
	}
	return n;
}

/** @brief The barrier that moves the network's images into their layouts from UNDEFINED: from the
 *         top of the pipe to all commands. */
static char const SETTLE[] = "barrier 1 65536 0";

/** @brief A frame's gate and seed, as a message says them. */
struct history {
	char text[64]; //!< The words.
};

/** @brief A frame's gate and seed as a message.
 *
 * @param f The frame.
 * @return  "gate G, seed S".
 */
static struct history
history (struct frame const *f)
{
	struct history h;
	int const n = snprintf(h.text, sizeof h.text, "gate %f, seed %" PRIu32, (double)f->gate, f->seed);
	require(n >= 0 && n < (int)sizeof h.text, "a history does not fit");
	return h;
}

/** @brief A pipeline as a frame names it at its first use. */
struct named {
	size_t length;   //!< The text's length.
	char   text[48]; //!< "(pipeline HASH)".
};

/** @brief A pipeline made from SPIR-V as a frame names it at its first use.
 *
 * @param code  The SPIR-V.
 * @param bytes Its bytes.
 * @return      The name.
 */
static struct named
named (void const *code,
       size_t      bytes)
{
	struct named n;
	int const length = snprintf(n.text, sizeof n.text, "(pipeline %" PRIu64 ")", fnv1a(code, bytes));
	require(length >= 0 && length < (int)sizeof n.text, "a pipeline's name does not fit");
	n.length = (size_t)length;
	return n;
}

/** @brief A name as a piece.
 *
 * @param n The name.
 * @return  Its words.
 */
static struct piece
named_piece (struct named const *n)
{
	return (struct piece){n->text, n->length};
}

/** @brief The handle that a token names, as a frame's line names it, without what its first use
 *         says it was made as.
 *
 * @param token The token.
 * @return      The handle's name.
 */
static struct piece
handle (struct piece token)
{
	size_t length = 0;
	while (length < token.length && !strchr("(/ ", token.at[length]))
		++length;
	return (struct piece){token.at, length};
}

/** @brief The rest of a line from its last " #", the space skipped.
 *
 * @param line The line.
 * @return     The rest, or the whole line if it holds no " #".
 */
static struct piece
after_last_name (struct piece line)
{
	for (size_t at = line.length; at-- > 1;)
		if (line.at[at - 1] == ' ' && line.at[at] == '#')
			return (struct piece){line.at + at, line.length - at};
	return line;
}

/** @brief The handle bound at a binding of a set's line.
 *
 * @param set     The line.
 * @param binding The binding's words, such as " 1=".
 * @return        The handle's name, or nothing when the line has no such binding.
 */
static struct piece
bound (struct piece set,
       struct piece binding)
{
	for (size_t at = 0; at + binding.length <= set.length; ++at)
		if (!memcmp(set.at + at, binding.at, binding.length))
			return handle((struct piece){set.at + at + binding.length, set.length - at - binding.length});
	return (struct piece){"", 0};
}

/** @brief A frame's commands, split into lines. */
struct split {
	struct piece *lines; //!< The lines, without their newlines.
	size_t        count; //!< Their number.
};

/** @brief Splits commands into lines.
 *
 * @param commands The commands, each line ending in a newline.
 * @return         The lines, which the caller frees.
 */
static struct split
split (struct piece commands)
{
	size_t count = 0;
	for (size_t i = 0; i < commands.length; ++i)
		count += commands.at[i] == '\n';
	struct split s = {calloc(count ? count : 1, sizeof *s.lines), 0};
	require(s.lines, "out of memory");
	for (size_t at = 0; at < commands.length;) {
		char const *const newline = memchr(commands.at + at, '\n', commands.length - at);
		size_t const end = newline ? (size_t)(newline - commands.at) : commands.length;
		s.lines[s.count++] = (struct piece){commands.at + at, end - at};
		at = end + 1;
	}
	return s;
}

/** @brief Whether a frame judges its waits after the network's last dispatch and answers with its
 *         input only over the verdict's grid.
 *
 * The verdict runs over one workgroup with the fallback's grid and the plan's timeouts, a barrier
 * takes its writes into the indirect read, the fallback and the host, then the fallback runs over
 * the grid that the verdict's set binds, from the first pass's input into the image that the answer
 * is then copied out of, and no dispatch after it. The first pass's input is what the last transfer
 * before the network's first dispatch writes: the network's input, or with later passes the image
 * that keeps it. In image mode it is the caller's frame, and the caller's answer is the image the
 * fallback stores into, with nothing after it, or what that image is blitted into last.
 *
 * @param commands      The frame's commands.
 * @param width         The frame's width.
 * @param height        Its height.
 * @param timeouts      The plan's timeouts.
 * @param timeout_count Their number.
 * @param frame_name    In image mode, the caller's frame as the frame names it; otherwise empty.
 * @param answer        In image mode, the caller's answer as the frame names it; otherwise empty.
 * @return              true if it does.
 */
static bool
judged (struct piece    commands,
        uint32_t        width,
        uint32_t        height,
        uint32_t const *timeouts,
        size_t          timeout_count,
        struct piece    frame_name,
        struct piece    answer)
{
	struct split s = split(commands);
	struct piece const *const l = s.lines;
	size_t const count = s.count;
	struct piece first = frame_name;
	if (!first.length) {
		for (size_t i = 0; i < count && !starts(l[i], PIECE("dispatch")); ++i)
			if (starts(l[i], PIECE("copy buffer to image ")) || starts(l[i], PIECE("copy image #")) ||
			    starts(l[i], PIECE("blit ")))
				first = handle(after_last_name(l[i]));
	}
	struct named const verdict_name = named(kNetworkVerdictSpv, sizeof kNetworkVerdictSpv);
	struct named const fallback_name = named(kNetworkFallbackSpv, sizeof kNetworkFallbackSpv);
	size_t v = 0;
	while (v < count && !(starts(l[v], PIECE("pipeline #")) && ends(l[v], named_piece(&verdict_name))))
		++v;
	bool dispatched = false;
	for (size_t i = 0; i < v; ++i)
		dispatched |= starts(l[i], PIECE("dispatch "));
	bool ok = first.length && v < count && count - v >= 10 && dispatched;
	if (ok) {
		// The result, which the fallback stores into.
		struct piece const result = bound(l[v + 7], PIECE("set 0="));
		char push[256];
		int at = snprintf(push, sizeof push, "push %" PRIu32 " %" PRIu32 " %zu", (width + 7) / 8,
		                  (height + 7) / 8, timeout_count);
		for (size_t i = 0; i < 13 && at >= 0 && at < (int)sizeof push; ++i)
			at += snprintf(push + at, sizeof push - (size_t)at, " %" PRIu32,
			               i < timeout_count ? timeouts[i] : 0);
		require(at >= 0 && at < (int)sizeof push, "the verdict's push constants do not fit");
		VkPipelineStageFlags const after = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
		                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
		VkAccessFlags const reads = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
		                            VK_ACCESS_HOST_READ_BIT;
		char barrier[64], memory[64], fallback[64], indirect[64];
		int const barrier_length = snprintf(barrier, sizeof barrier, "barrier 2048 %" PRIu32 " 1",
		                                    (uint32_t)after);
		int const memory_length = snprintf(memory, sizeof memory, " memory 64 %" PRIu32, (uint32_t)reads);
		int const fallback_length = snprintf(fallback, sizeof fallback, "push %" PRIu32 " %" PRIu32, width,
		                                     height);
		struct piece const grid = bound(l[v + 1], PIECE(" 1="));
		int const indirect_length = snprintf(indirect, sizeof indirect, "dispatch indirect %.*s 0",
		                                     (int)grid.length, grid.at);
		require(barrier_length > 0 && barrier_length < (int)sizeof barrier && memory_length > 0 &&
		        memory_length < (int)sizeof memory && fallback_length > 0 &&
		        fallback_length < (int)sizeof fallback && indirect_length > 0 &&
		        indirect_length < (int)sizeof indirect, "a verdict's line does not fit");
		ok = equals(l[v + 2], (struct piece){push, (size_t)at}) && equals(l[v + 3], PIECE("dispatch 1 1 1")) &&
		     equals(l[v + 4], (struct piece){barrier, (size_t)barrier_length}) &&
		     equals(l[v + 5], (struct piece){memory, (size_t)memory_length}) &&
		     starts(l[v + 6], PIECE("pipeline #")) && ends(l[v + 6], named_piece(&fallback_name)) &&
		     result.length && equals(bound(l[v + 7], PIECE(" 1=")), first) &&
		     equals(l[v + 8], (struct piece){fallback, (size_t)fallback_length}) &&
		     equals(l[v + 9], (struct piece){indirect, (size_t)indirect_length});
		if (ok && answer.length && equals(result, answer)) {
			ok = v + 10 == count;
		} else if (ok) {
			// The result leaves GENERAL for the copy out, or in image mode the blit into the answer, once
			// the fallback is done.
			char image[64];
			int const image_length = snprintf(image, sizeof image, " image %.*s 1 6", (int)result.length,
			                                  result.at);
			require(image_length > 0 && image_length < (int)sizeof image, "a verdict's line does not fit");
			ok = count - v >= 12 && equals(l[v + 10], PIECE("barrier 2048 4096 0")) &&
			     equals(l[v + 11], (struct piece){image, (size_t)image_length});
			for (size_t i = v + 10; ok && i < count; ++i)
				ok = !starts(l[i], PIECE("dispatch"));
			size_t out = v + 12;
			while (ok && out < count && !starts(l[out], PIECE("copy image to buffer ")) &&
			       !starts(l[out], PIECE("blit ")))
				++out;
			if (ok && out < count) {
				char const *const name = memchr(l[out].at, '#', l[out].length);
				struct piece const from =
					name ? handle((struct piece){name, l[out].length - (size_t)(name - l[out].at)})
					     : (struct piece){"", 0};
				ok = equals(from, result) &&
				     (!answer.length ||
				      (out == v + 12 && starts(l[out], PIECE("blit ")) && count - v == 15 &&
				       equals(handle(after_last_name(l[out])), answer)));
			} else {
				ok = false;
			}
		}
	}
	free(s.lines);
	s.lines = nullptr;
	return ok;
}

/** @brief A frame of the test's extent, 64x64, with the frame's defaults otherwise.
 *
 * @return The frame.
 */
static struct vulkan_frame
frame_64 (void)
{
	struct vulkan_frame f = VULKAN_FRAME_DEFAULTS;
	f.width = f.height = 64;
	return f;
}

/** @brief Checks the motion history of frames that are and are not submitted.
 *
 * @param paths  Where the network's files are.
 * @param passes The frames' passes.
 */
static void
check (struct vulkan_paths const *paths,
       uint32_t                   passes)
{
	struct vulkan_frame f = frame_64();
	f.motion = true;
	f.passes = passes;
	struct vulkan_device const device = fake_device();
	struct network_recorder recorder;
	recorder_init(&recorder, &device, paths, false);
	struct error e = {"the network was not built"};
	require(shape(&recorder, &f, nullptr, &e), "%s", e.what);
	// The first frame, and the same frame again: the first was not submitted, so the second starts
	// the history too.
	struct frame first = record(&recorder, &f);
	require(first.gate == 0 && first.seed == 0, "the first frame has %s", history(&first).text);
	require(lines(whole(&first.commands), PIECE(SETTLE), PIECE("")) == 1,
	        "the first frame does not move the images into their layouts");
	struct frame again = record(&recorder, &f);
	require(support_text_equal(&again.commands, &first.commands),
	        "a frame after one that was not submitted differs from that one: %s", history(&again).text);
	// Submitted, the first frame starts the history, which the next frame reads.
	network_recorder_submitted(&recorder);
	struct frame second = record(&recorder, &f);
	require(second.gate == 1 && second.seed == 1, "the second frame has %s", history(&second).text);
	require(lines(whole(&second.commands), PIECE(SETTLE), PIECE("")) == 0,
	        "the second frame moves the images into their layouts again");
	// A frame with another intensity starts the history over, but is not submitted: the next frame is
	// the second frame again.
	struct vulkan_frame other = f;
	other.intensity = 0.5f;
	struct frame reset = record(&recorder, &other);
	require(reset.gate == 0 && reset.seed == 0, "a frame with another intensity has %s", history(&reset).text);
	struct frame after = record(&recorder, &f);
	require(support_text_equal(&after.commands, &second.commands),
	        "a frame after a reset that was not submitted differs from the frame before: %s", history(&after).text);
	network_recorder_submitted(&recorder);
	struct frame third = record(&recorder, &f);
	require(third.gate == 1 && third.seed == 2, "the third frame has %s", history(&third).text);
	frame_fini(&third);
	frame_fini(&after);
	frame_fini(&reset);
	frame_fini(&second);
	frame_fini(&again);
	frame_fini(&first);
	network_recorder_fini(&recorder);
}

/** @brief The first two frames of a frame, the second following the first in the motion history.
 *
 * @param recorder The recorder.
 * @param f        The frame.
 * @return         Their commands, which the caller frees.
 */
static struct support_text
frames_of (struct network_recorder   *recorder,
           struct vulkan_frame const *f)
{
	struct frame one = record(recorder, f);
	network_recorder_submitted(recorder);
	struct frame two = record(recorder, f);
	require(support_text_printf(&one.commands, "%s", support_text_string(&two.commands)), "out of memory");
	struct support_text const commands = one.commands;
	one.commands = (struct support_text){};
	frame_fini(&two);
	frame_fini(&one);
	return commands;
}

/** @brief A step of a walk through shapes of 64x64 frames. */
struct walk_step {
	uint32_t passes;    //!< The passes.
	float    sharpness; //!< The sharpness, which builds the pass stages in.
	bool     fp16;      //!< RGBA16F frames.
	bool     motion;    //!< With motion history.
};

/** @brief A walk through shapes of 64x64 frames, each a change that the recorder rebuilds for: its
 *         passes, more or fewer, format, motion history and stages. */
static struct walk_step const WALK[] = {
	{1, 0, false, false},    {1, 0.5f, false, false}, {2, 0.5f, false, false}, {2, 0.5f, false, true},
	{2, 0.5f, true, true},   {1, 0, true, false},     {1, 0, false, true},     {3, 0, false, true},
	{3, 0.5f, true, true},   {1, 0.5f, true, true},   {1, 0, true, false},     {1, 0, false, false},
	{2, 0, false, false},    {1, 0, false, false},
};

/** @brief A frame's shape as the messages say it. */
struct shape_words {
	char text[64]; //!< The words.
};

/** @brief Says a frame's shape: its passes, format, motion and pass stages.
 *
 * @param f The frame.
 * @return  The words.
 */
static struct shape_words
shape_words (struct vulkan_frame const *f)
{
	struct shape_words w;
	int const n = snprintf(w.text, sizeof w.text, "%" PRIu32 " passes, %s%s%s", f->passes,
	                       f->fp16 ? "FP16" : "RGBA8", f->motion ? ", motion" : "",
	                       f->sharpness != 0 ? ", pass stages" : "");
	require(n >= 0 && n < (int)sizeof w.text, "a shape's words do not fit");
	return w;
}

/** @brief Checks the walk's reshapes: each shape's frames after the reshape must be those of a
 *         network built for the shape.
 *
 * @param paths Where the network's files are.
 */
static void
check_reshapes (struct vulkan_paths const *paths)
{
	struct vulkan_device const device = fake_device();
	struct network_recorder recorder;
	recorder_init(&recorder, &device, paths, false);
	struct vulkan_plan plan;
	require(vulkan_plan_init(&plan, 64, 64, UINT64_MAX, nullptr) == ERROR_NONE, "cannot plan 64x64 frames");
	for (size_t i = 0; i < sizeof WALK / sizeof *WALK; ++i) {
		struct vulkan_frame f = frame_64();
		f.passes = WALK[i].passes;
		f.fp16 = WALK[i].fp16;
		f.motion = WALK[i].motion;
		f.sharpness = WALK[i].sharpness;
		struct shape_words const words = shape_words(&f);
		frame_start();
		size_t const pipelines = pipelines_made;
		bool const first = !i;
		struct error e;
		require(shape(&recorder, &f, nullptr, &e), "the network was not rebuilt for %s", words.text);
		// The first shape is a build; every other keeps the weights, the arena and the pipelines,
		// records nothing and makes no pipeline.
		require(first || (!frame.commands.length && pipelines_made == pipelines),
		        "the reshape for %s recorded a command or made a pipeline", words.text);
		// A frame that is not submitted leaves the images' layouts to the next.
		struct frame unsubmitted = record(&recorder, &f);
		require(lines(whole(&unsubmitted.commands), PIECE(SETTLE), PIECE("")) == 1,
		        "the first frame after the reshape for %s does not move the images into their layouts",
		        words.text);
		require(judged(whole(&unsubmitted.commands), f.width, f.height, plan.timeouts, plan.timeout_count,
		               PIECE(""), PIECE("")),
		        "a frame of %s does not judge its waits after the network, or answers with its input "
		        "otherwise than over the verdict's grid", words.text);
		frame_fini(&unsubmitted);
		struct support_text after = frames_of(&recorder, &f);
		require(lines(whole(&after), PIECE(SETTLE), PIECE("")) == 1,
		        "the frames after the reshape for %s do not move the images into their layouts once",
		        words.text);
		// With one pass and no pass stages, each of the two frames copies the proxy, #0, into the
		// network's input, #2, the first image it names, and an image into the answer buffer, #1. It
		// copies or blits no image.
		require(f.passes > 1 || f.sharpness != 0 ||
		        (lines(whole(&after), PIECE("copy buffer to image #0 "), PIECE(" #2")) == 2 &&
		         lines(whole(&after), PIECE("copy image to buffer #"), PIECE(" #1")) == 2 &&
		         lines(whole(&after), PIECE("copy image #"), PIECE("")) == 0 &&
		         lines(whole(&after), PIECE("blit"), PIECE("")) == 0),
		        "the frames of %s do not copy the proxy straight in and the answer straight out", words.text);
		struct network_recorder built;
		recorder_init(&built, &device, paths, false);
		require(shape(&built, &f, nullptr, &e), "cannot build the network for %s", words.text);
		struct support_text built_frames = frames_of(&built, &f);
		require(support_text_equal(&after, &built_frames),
		        "the frames after the reshape for %s differ from those of a build", words.text);
		support_text_fini(&built_frames);
		network_recorder_fini(&built);
		support_text_fini(&after);
	}
	vulkan_plan_fini(&plan);
	network_recorder_fini(&recorder);
}

/** @brief A file's pipeline as a frame names it at its first use.
 *
 * @param spirv The network's SPIR-V directory.
 * @param file  The pipeline's SPIR-V below it.
 * @return      The name.
 */
static struct named
named_file (char const *spirv,
            char const *file)
{
	char *path = support_format(nullptr, "%s/%s", spirv, file);
	require(path, "out of memory");
	size_t length = 0;
	char *code = support_read_file(path, &length);
	require(code, "cannot read %s", file);
	struct named const n = named(code, length);
	free(code);
	code = nullptr;
	free(path);
	path = nullptr;
	return n;
}

/** @brief Checks that from a model whose weights free every Swin layer of the exponent's upper clamp,
 *         a frame runs the C=32 kernels without that clamp, and with motion the temporal pre block
 *         without it, never their twins with the clamp.
 *
 * @param spirv  The network's SPIR-V directory.
 * @param length The length of its path.
 */
static void
check_unclamped (char const *spirv,
                 size_t      length)
{
	struct vulkan_plan plan;
	require(vulkan_plan_init(&plan, 64, 64, UINT64_MAX, nullptr) == ERROR_NONE, "cannot plan 64x64 frames");
	struct vulkan_pack model;
	vulkan_pack_init(&model);
	require(vulkan_pack_synthetic_model(&plan, &model, true), "cannot make the unclamped model");
	struct vulkan_paths const paths = {
		.model          = model.path,
		.shaders        = spirv,
		.model_length   = model.path_length,
		.shaders_length = length,
	};
	for (size_t i = 0; i < 2; ++i) {
		bool const motion = i;
		struct vulkan_frame f = frame_64();
		f.motion = motion;
		struct vulkan_device const device = fake_device();
		struct network_recorder recorder;
		recorder_init(&recorder, &device, &paths, false);
		struct error e;
		require(shape(&recorder, &f, nullptr, &e), "cannot build the network from the unclamped model");
		struct frame recorded = record(&recorder, &f);
		// The twins, clamped and unclamped.
		char twins[VULKAN_PLAN_UNCLAMPED_COUNT][2][64];
		size_t count = 0;
		if (motion) {
			static char const pre[] = "temporal/temporal_pre_fp32.spv";
			static char const pre_nh[] = "temporal/temporal_pre_fp32nh.spv";
			static_assert(sizeof pre <= sizeof twins[0][0] && sizeof pre_nh <= sizeof twins[0][1]);
			memcpy(twins[0][0], pre, sizeof pre);
			memcpy(twins[0][1], pre_nh, sizeof pre_nh);
			count = 1;
		} else {
			for (; count < VULKAN_PLAN_UNCLAMPED_COUNT; ++count) {
				struct vulkan_unclamped const *const u = &VULKAN_PLAN_UNCLAMPED[count];
				int const clamped = snprintf(twins[count][0], sizeof twins[count][0], "g_%s.spv",
				                             VULKAN_PLAN_KERNELS[u->kernel].stem);
				int const unclamped = snprintf(twins[count][1], sizeof twins[count][1], "g_%s.spv",
				                               VULKAN_PLAN_KERNELS[u->twin].stem);
				require(clamped > 0 && clamped < (int)sizeof twins[count][0] && unclamped > 0 &&
				        unclamped < (int)sizeof twins[count][1], "a kernel's file name does not fit");
			}
		}
		for (size_t i = 0; i < count; ++i) {
			struct named const clamped = named_file(spirv, twins[i][0]);
			struct named const unclamped = named_file(spirv, twins[i][1]);
			require(lines(whole(&recorded.commands), PIECE("pipeline #"), named_piece(&unclamped)) == 1 &&
			        !lines(whole(&recorded.commands), PIECE("pipeline #"), named_piece(&clamped)),
			        "a frame from the unclamped model does not run %s instead of %s", twins[i][1],
			        twins[i][0]);
		}
		frame_fini(&recorded);
		network_recorder_fini(&recorder);
	}
	vulkan_pack_fini(&model);
	vulkan_plan_fini(&plan);
}

/** @brief Makes the caller's images of a generation on the fake device, which names them by what
 *         they are and their generation.
 *
 * @param generation The generation.
 * @return           The images.
 */
static struct vulkan_frame_images
caller_images (uint64_t generation)
{
	struct vulkan_frame_images images = {.generation = generation};
	VkImage frame_image;
	MAKE(&frame_image);
	MAKE(&images.answer);
	MAKE(&images.frame);
	MAKE(&images.answer_view);
	made_as(ID(frame_image), "caller frame %" PRIu64, generation);
	made_as(ID(images.answer), "caller answer %" PRIu64, generation);
	object(ID(images.frame))->image = ID(frame_image);
	object(ID(images.answer_view))->image = ID(images.answer);
	return images;
}

/** @brief A frame recorded in image mode by a recorder.
 *
 * @param recorder The recorder.
 * @param f        The frame.
 * @return         What it recorded, which the caller frees.
 */
static struct frame
record_in_place (struct network_recorder   *recorder,
                 struct vulkan_frame const *f)
{
	static VkCommandBuffer cmd;
	if (!cmd)
		MAKE(&cmd);
	frame_start();
	network_recorder_record_images(recorder, cmd, f);
	return frame_take();
}

#undef MAKE

/** @brief The first two frames of a frame in image mode, the second following the first in the
 *         motion history.
 *
 * @param recorder The recorder.
 * @param f        The frame.
 * @return         Their commands, which the caller frees.
 */
static struct support_text
frames_in_place (struct network_recorder   *recorder,
                 struct vulkan_frame const *f)
{
	struct frame one = record_in_place(recorder, f);
	network_recorder_submitted(recorder);
	struct frame two = record_in_place(recorder, f);
	require(support_text_printf(&one.commands, "%s", support_text_string(&two.commands)), "out of memory");
	struct support_text const commands = one.commands;
	one.commands = (struct support_text){};
	frame_fini(&two);
	frame_fini(&one);
	return commands;
}

/** @brief How many of the sets that a frame bound name an image.
 *
 * @param f     The frame.
 * @param image The image.
 * @return      The sets.
 */
static size_t
naming (struct frame const *f,
        uint64_t            image)
{
	size_t n = 0;
	for (size_t i = 0; i < f->set_count; ++i) {
		struct object const *const set = object(f->sets[i]);
		bool names = false;
		for (uint32_t b = 0; b < SET_BINDINGS && !names; ++b)
			names = (set->present >> b & 1) && set->bindings[b].resource == image;
		n += names;
	}
	return n;
}

/** @brief An image as the frame recorded last names it. */
struct name {
	size_t length;   //!< The text's length.
	char   text[24]; //!< "#N", or nothing.
};

/** @brief An image as the frame recorded last names it, or nothing when that frame does not.
 *
 * @param f     The frame, the one recorded last.
 * @param image The image.
 * @return      The name.
 */
static struct name
as_named (struct frame const *f,
          uint64_t            image)
{
	struct name n = {};
	struct object const *const o = object(image);
	if (o->named_in == f->id) {
		int const length = snprintf(n.text, sizeof n.text, "#%zu", o->name);
		require(length > 0 && length < (int)sizeof n.text, "a name does not fit");
		n.length = (size_t)length;
	}
	return n;
}

/** @brief Checks image mode.
 *
 * A build binds no images and leaves out the sets that would, writing no descriptor without one. A
 * bind of the caller's images records nothing, makes no pipeline and writes the sets; a bind of a new
 * generation keeps the motion history and the images' layouts, and the frames after it sample the
 * new images only; images of the generation bound write nothing. Through the walk of
 * check_reshapes(), each shape with images of its own: no frame copies a buffer, makes an image in
 * the frame's format or transfers before the network's first dispatch. One pass without the pass
 * stages stores its answer in the caller's, which its fallback binds with the caller's frame, and
 * copies or blits no image; any other shape blits its answer into the caller's last, its only blit.
 * Only the first pass's blocks, the stages, the alpha pass, the fallback and the motion estimate
 * sample the caller's frame. Each shape's frames after the reshape must be those of a network built
 * for the shape and bound to the same images.
 *
 * @param paths Where the network's files are.
 */
static void
check_images (struct vulkan_paths const *paths)
{
	struct vulkan_plan plan;
	require(vulkan_plan_init(&plan, 64, 64, UINT64_MAX, nullptr) == ERROR_NONE, "cannot plan 64x64 frames");
	struct vulkan_frame f = frame_64();
	f.motion = true;
	struct vulkan_device const device = fake_device();
	struct network_recorder recorder;
	recorder_init(&recorder, &device, paths, true);
	size_t const nulls = null_views;
	struct error e;
	require(shape(&recorder, &f, nullptr, &e) && null_views == nulls,
	        "a build in image mode failed, or wrote a descriptor without an image");
	uint64_t generation = 0;
	struct vulkan_frame_images const images = caller_images(++generation);
	frame_start();
	size_t pipelines = pipelines_made, written = descriptors_written;
	require(shape(&recorder, &f, &images, &e) && !frame.commands.length && pipelines_made == pipelines &&
	        descriptors_written > written,
	        "a bind of the caller's images recorded a command, made a pipeline or wrote no descriptor");
	struct frame first = record_in_place(&recorder, &f);
	require(first.gate == 0 && first.seed == 0 && lines(whole(&first.commands), PIECE(SETTLE), PIECE("")) == 1,
	        "the first frame in image mode has %s, or does not settle the images once", history(&first).text);
	frame_fini(&first);
	network_recorder_submitted(&recorder);
	struct frame second = record_in_place(&recorder, &f);
	require(second.gate == 1, "the second frame in image mode does not follow the first");
	frame_fini(&second);
	network_recorder_submitted(&recorder);
	struct vulkan_frame_images const next = caller_images(++generation);
	frame_start();
	written = descriptors_written;
	require(shape(&recorder, &f, &next, &e) && !frame.commands.length && pipelines_made == pipelines &&
	        descriptors_written > written,
	        "a bind of new images recorded a command, made a pipeline or wrote no descriptor");
	struct frame third = record_in_place(&recorder, &f);
	require(third.gate == 1 && third.seed == 2 && lines(whole(&third.commands), PIECE(SETTLE), PIECE("")) == 0,
	        "the frame after a bind of new images has %s, or settles the images again", history(&third).text);
	require(naming(&third, object(ID(next.frame))->image) && naming(&third, ID(next.answer)) &&
	        !naming(&third, object(ID(images.frame))->image) && !naming(&third, ID(images.answer)),
	        "the frame after a bind of new images does not bind them, or binds the old ones");
	frame_fini(&third);
	written = descriptors_written;
	bool changed = true;
	require(network_recorder_shape(&recorder, &f, &next, &changed, &e) == ERROR_NONE && !changed &&
	        descriptors_written == written,
	        "images of the generation bound were bound again");

	for (size_t i = 0; i < sizeof WALK / sizeof *WALK; ++i) {
		f.passes = WALK[i].passes;
		f.fp16 = WALK[i].fp16;
		f.motion = WALK[i].motion;
		f.sharpness = WALK[i].sharpness;
		struct shape_words const words = shape_words(&f);
		struct vulkan_frame_images const own = caller_images(++generation);
		frame_start();
		pipelines = pipelines_made;
		require(shape(&recorder, &f, &own, &e) && !frame.commands.length && pipelines_made == pipelines,
		        "the reshape in image mode for %s recorded a command or made a pipeline", words.text);
		struct frame one = record_in_place(&recorder, &f);
		uint64_t const frame_image = object(ID(own.frame))->image;
		struct name const frame_name = as_named(&one, frame_image);
		struct name const answer_name = as_named(&one, ID(own.answer));
		bool const direct = f.passes == 1 && f.sharpness == 0;
		char format[32];
		int const format_length = snprintf(format, sizeof format, "format %d ",
		                                   f.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM);
		require(format_length > 0 && format_length < (int)sizeof format, "a format does not fit");
		// The commands before the network's first dispatch.
		char const *const commands = support_text_string(&one.commands);
		char const *const dispatch = strstr(commands, "\ndispatch ");
		struct piece const head = {commands, dispatch ? (size_t)(dispatch - commands) : one.commands.length};
		bool transfers = false;
		for (size_t at = 0; at < head.length && !transfers; ++at) {
			struct piece const rest = {head.at + at, head.length - at};
			transfers = head.at[at] == '\n' &&
			            (starts(rest, PIECE("\ncopy")) || starts(rest, PIECE("\nblit")));
		}
		require(!lines(whole(&one.commands), PIECE("copy buffer"), PIECE("")) &&
		        !lines(whole(&one.commands), PIECE("copy image to buffer"), PIECE("")) &&
		        !strstr(commands, format) && !transfers,
		        "a frame of %s in image mode copies a buffer, makes an image in the frame's format or "
		        "transfers before the network", words.text);
		require(direct ? !lines(whole(&one.commands), PIECE("copy image"), PIECE("")) &&
		                 !lines(whole(&one.commands), PIECE("blit"), PIECE("")) && naming(&one, ID(own.answer))
		               : lines(whole(&one.commands), PIECE("blit"), PIECE("")) == 1,
		        "a frame of %s in image mode does not store into the caller's answer without a copy, or "
		        "blits other than once", words.text);
		// The sets that sample the caller's frame: the first pass's pre and post blocks, each pass's
		// stage, the alpha pass after later passes, the fallback, and with motion the finest luma level
		// of each of the four; later passes' blocks sample the input instead.
		size_t const samplers = 3 + (f.sharpness != 0 ? f.passes : 0) + (f.passes > 1) + (f.motion ? 4 : 0);
		size_t const sampling = naming(&one, frame_image);
		require(sampling == samplers,
		        "a frame of %s in image mode samples the caller's frame in %zu sets, not %zu", words.text,
		        sampling, samplers);
		require(judged(whole(&one.commands), f.width, f.height, plan.timeouts, plan.timeout_count,
		               (struct piece){frame_name.text, frame_name.length},
		               (struct piece){answer_name.text, answer_name.length}),
		        "a frame of %s in image mode does not judge its waits after the network, or answers with the "
		        "caller's frame otherwise than into the caller's answer", words.text);
		frame_fini(&one);
		struct support_text after = frames_in_place(&recorder, &f);
		struct network_recorder built;
		recorder_init(&built, &device, paths, true);
		require(shape(&built, &f, &own, &e), "cannot build the network in image mode for %s", words.text);
		struct frame discarded = record_in_place(&built, &f);
		frame_fini(&discarded);
		struct support_text built_frames = frames_in_place(&built, &f);
		require(support_text_equal(&after, &built_frames),
		        "the frames after the reshape in image mode for %s differ from those of a build", words.text);
		support_text_fini(&built_frames);
		network_recorder_fini(&built);
		support_text_fini(&after);
	}
	require(null_views == nulls, "image mode wrote a descriptor without an image");
	network_recorder_fini(&recorder);
	vulkan_plan_fini(&plan);
}

#undef ID

/** @brief The verdict of the network built last, the fallback's grid, in the fake device's memory
 *         that its runtime keeps mapped, the only memory a built network keeps mapped.
 *
 * @return The memory.
 */
static uint8_t *
verdict (void)
{
	for (size_t h = object_count; h-- > 0;)
		if (objects[h].host && objects[h].size == sizeof (VkDispatchIndirectCommand))
			return objects[h].host;
	require(false, "the network keeps no verdict mapped");
	return nullptr;
}

/** @brief The lines the network logged that say a wait ran out. */
static size_t timeout_lines;

/** @brief Counts the lines that say a wait ran out.
 *
 * @param line A line the network logs.
 */
static void
count_timeouts (char const *line)
{
	timeout_lines += strstr(line, "ran out") != nullptr;
}

/** @brief Whether a frame starts over: it zeroes the arena from the end of the values on before any
 *         dispatch, between barriers from the compute work before it and into the compute work after
 *         it.
 *
 * @param f      The frame.
 * @param zeroed The fill's line after its buffer's name.
 * @return       true if it does.
 */
static bool
zeroes (struct frame const *f,
        struct piece        zeroed)
{
	// Compute (2048) to transfer (4096) stages, shader writes (64) to transfer writes (4096), and back
	// to shader reads and writes (96).
	static char const before[] = "barrier 2048 4096 1\n memory 64 4096\n";
	static char const after[] = "\nbarrier 4096 2048 1\n memory 4096 96\n";
	char const *const c = support_text_string(&f->commands);
	char const *const fill_line = strstr(c, "\nfill #");
	char const *const dispatch = strstr(c, "\ndispatch ");
	size_t const fill = fill_line ? (size_t)(fill_line - c) + 1 : 0;
	size_t const first_dispatch = dispatch ? (size_t)(dispatch - c) : SIZE_MAX;
	char const *const end = strchr(c + fill, '\n');
	return lines(whole(&f->commands), PIECE("fill #"), zeroed) == 1 && fill < first_dispatch &&
	       fill >= sizeof before - 1 && !memcmp(c + fill - (sizeof before - 1), before, sizeof before - 1) && end &&
	       !strncmp(end, after, sizeof after - 1);
}

/** @brief Checks the waits that run out.
 *
 * After a frame whose wait ran out, as a verdict poked into the fake device's memory says, the next
 * frame starts the network over: it zeroes the arena from the end of the values on before any
 * dispatch, between barriers, and drops the motion history. So does the frame after it, when that
 * frame was not submitted. A line says so, but not again for the frame after it. Only a frame that
 * starts over zeroes the arena.
 *
 * @param paths Where the network's files are.
 */
static void
check_timeouts (struct vulkan_paths const *paths)
{
	struct vulkan_plan plan;
	require(vulkan_plan_init(&plan, 64, 64, UINT64_MAX, nullptr) == ERROR_NONE, "cannot plan 64x64 frames");
	struct vulkan_frame f = frame_64();
	f.motion = true;
	struct vulkan_device device = fake_device();
	device.log = count_timeouts;
	struct network_recorder recorder;
	recorder_init(&recorder, &device, paths, false);
	struct error e;
	require(shape(&recorder, &f, nullptr, &e), "cannot build the network for 64x64 frames");
	uint8_t *const grid = verdict();
	char zeroed[96];
	int const zeroed_length = snprintf(zeroed, sizeof zeroed, "(buffer %" PRIu64 ") %" PRIu64 " %" PRIu64,
	                                   plan.arena_bytes, plan.values_end, (uint64_t)VK_WHOLE_SIZE);
	require(zeroed_length > 0 && zeroed_length < (int)sizeof zeroed, "a fill's line does not fit");
	struct piece const fill = {zeroed, (size_t)zeroed_length};
	// The first frame starts over, and zeroes what the build zeroed.
	struct frame first = record(&recorder, &f);
	require(zeroes(&first, fill),
	        "the first frame does not zero the arena past its values first, between barriers");
	frame_fini(&first);
	network_recorder_submitted(&recorder);
	struct frame second = record(&recorder, &f);
	require(!network_recorder_timed_out(&recorder) && second.gate == 1 &&
	        !lines(whole(&second.commands), PIECE("fill"), PIECE("")),
	        "the second frame has %s or zeroes the arena", history(&second).text);
	frame_fini(&second);
	// The second frame's wait ran out: its verdict gives the fallback workgroups.
	network_recorder_submitted(&recorder);
	uint32_t const groups = 8;
	memcpy(grid, &groups, sizeof groups);
	require(network_recorder_timed_out(&recorder),
	        "a verdict that gives the fallback workgroups does not say a wait ran out");
	struct frame over = record(&recorder, &f);
	require(over.gate == 0 && over.seed == 0 && zeroes(&over, fill),
	        "the frame after one whose wait ran out has %s, or does not zero the arena first, between barriers",
	        history(&over).text);
	struct frame again = record(&recorder, &f);
	require(support_text_equal(&again.commands, &over.commands),
	        "the frame after one that started over and was not submitted does not start over too");
	require(timeout_lines == 1, "a wait that ran out is not logged once");
	frame_fini(&again);
	frame_fini(&over);
	// Submitted, and none of its waits ran out.
	network_recorder_submitted(&recorder);
	memset(grid, 0, sizeof groups);
	struct frame next = record(&recorder, &f);
	require(!network_recorder_timed_out(&recorder) && next.gate == 1 && next.seed == 1 &&
	        !lines(whole(&next.commands), PIECE("fill"), PIECE("")),
	        "the frame after one that started over has %s or zeroes the arena", history(&next).text);
	frame_fini(&next);
	network_recorder_fini(&recorder);
	vulkan_plan_fini(&plan);
}

#undef PIECE

/** @brief What --help prints. */
static char const USAGE[] =
	"Usage: network-recorder-test [OPTION]... SPIRV-DIRECTORY\n"
	"Checks the Vulkan network's motion history and reshapes on a fake device, from a synthetic\n"
	"model and the network's SPIR-V in SPIRV-DIRECTORY (required). No GPU or model needed.\n"
	" -b, --build WxH   Instead, build the network for WxH frames, which makes every pipeline of\n"
	"                   the runtime's own but one of the temporal pre blocks (default: unset)\n"
	" -u, --unclamped   Build from a synthetic model whose position biases and head scales are\n"
	"                   zero, which frees every Swin layer of the exponent's upper clamp\n"
	"                   (default: off)\n"
	" -h, --help        Show help (default: off)\n";

int
main (int    argc,
      char **argv)
{
	char const *extent = nullptr;
	bool unclamped = false;
	struct option const options[] = {
		{"build", required_argument, nullptr, 'b'},
		{"unclamped", no_argument, nullptr, 'u'},
		{"help", no_argument, nullptr, 'h'},
		{},
	};
	for (int code; (code = getopt_long(argc, argv, "+b:uh", options, nullptr)) != -1;) {
		if (code == 'b') {
			extent = optarg;
		} else if (code == 'u') {
			unclamped = true;
		} else if (code == 'h') {
			fputs(USAGE, stdout);
			return 0;
		} else {
			return 2;
		}
	}
	uint32_t width = 64, height = 64;
	if (optind + 1 != argc || (extent && !support_extent(extent, &width, &height)))
		return 2;
	frame_start();
	struct vulkan_plan plan;
	require(vulkan_plan_init(&plan, width, height, UINT64_MAX, nullptr) == ERROR_NONE, "cannot plan the frames");
	struct vulkan_pack model;
	vulkan_pack_init(&model);
	require(vulkan_pack_synthetic_model(&plan, &model, unclamped), "cannot make the synthetic model");
	char const *const spirv = argv[optind];
	size_t const spirv_length = strlen(spirv);
	struct vulkan_paths const paths = {
		.model          = model.path,
		.shaders        = spirv,
		.model_length   = model.path_length,
		.shaders_length = spirv_length,
	};
	if (extent) {
		struct vulkan_frame f = VULKAN_FRAME_DEFAULTS;
		f.width = width;
		f.height = height;
		struct vulkan_device const device = fake_device();
		struct network_recorder recorder;
		recorder_init(&recorder, &device, &paths, false);
		struct error e = {"the network was not built"};
		require(shape(&recorder, &f, nullptr, &e), "%s", e.what);
		network_recorder_fini(&recorder);
	} else {
		check(&paths, 1);
		check(&paths, 2);
		check_reshapes(&paths);
		check_unclamped(spirv, spirv_length);
		check_timeouts(&paths);
		check_images(&paths);
		puts("network-recorder test: frames not submitted leave the motion history as it was, a reshaped "
		     "network records the frames of a built one, weights free of the upper clamp run the kernels "
		     "without it, a frame whose wait ran out answers with its input and starts the next over, and "
		     "frames in a caller's images copy no buffer and blit at most once");
	}
	vulkan_pack_fini(&model);
	vulkan_plan_fini(&plan);
	frame_fini(&frame);
	objects_fini();
	return 0;
}
