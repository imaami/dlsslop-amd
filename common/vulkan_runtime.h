/** @file
 *
 * The Vulkan network's host runtime: every Vulkan object the network needs for one frame shape,
 * built with its GPU work in one submission, and the commands of a frame. A port of the production
 * path of DLSSNR-AMD's linux/src/core (MIT): nr_runtime.cpp, the device part of NrSession::build in
 * nr_graph.cpp and nrvk.hpp. vulkan_runtime.c defines the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_VULKAN_RUNTIME_H_
#define DLSSLOP_AMD_COMMON_VULKAN_RUNTIME_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
#else
# include <stddef.h>
# include <stdint.h>
#endif

#include <vulkan/vulkan.h>

#include "error.h"
#include "vulkan_plan.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief Where the Vulkan network finds what it loads, with each path's length.
 *
 * The strings are borrowed: whoever takes these paths copies what it keeps.
 */
struct vulkan_paths {
	char const *model;          //!< dlssnr.bin, extracted from the user's nvngx_dlssnr.dll.
	char const *shaders;        //!< The network's SPIR-V and markers, with runtime/ and temporal/ below.
	char const *cache;          //!< A writable pipeline cache; none when cache_length is 0.
	STD(size_t) model_length;   //!< The model's path's length.
	STD(size_t) shaders_length; //!< The shaders' path's length.
	STD(size_t) cache_length;   //!< The cache's path's length: 0 for none.
};

/** @brief A function that a build calls with the device's context around its submission.
 *
 * @param context The device's context.
 */
typedef void
vulkan_device_queue_fn (void *context);

/** @brief A function that takes the runtime's log lines.
 *
 * @param line A line, without its newline.
 */
typedef void
vulkan_device_log_fn (char const *line);

/** @brief The physical-device queries a build makes: the loader's in dlsslopd; in the layer's
 *         module the next layer's, looked up when the module opens, so that the build's thread never
 *         looks a function up through the loader. */
struct vulkan_physical_functions {
	PFN_vkGetPhysicalDeviceQueueFamilyProperties queue_families;    //!< Queue families.
	PFN_vkGetPhysicalDeviceProperties2           properties;        //!< Properties and limits.
	PFN_vkGetPhysicalDeviceFormatProperties2     format_properties; //!< Format features.
};

/** @brief The device the network runs on, which it does not own.
 *
 * dlsslopd's own device, or a game's in the layer, which network_requirements_unsupported()
 * accepted, with every feature of network_requirements.h enabled. Its device-level functions are the
 * loader's.
 */
struct vulkan_device {
	VkInstance                       instance;  //!< The device's instance.
	VkPhysicalDevice                 physical;  //!< Its physical device.
	VkDevice                         device;    //!< The device.
	VkQueue                          queue;     //!< The queue of family that takes a build's submission.
	vulkan_device_queue_fn          *lock;      //!< Called with context before that submission, or nullptr.
	vulkan_device_queue_fn          *unlock;    //!< Called with context after it, or nullptr.
	void                            *context;   //!< What lock and unlock take.
	vulkan_device_log_fn            *log;       //!< Where the runtime's log lines go, or nullptr: nowhere.
	struct vulkan_physical_functions functions; //!< The physical-device queries.
	VkPhysicalDeviceMemoryProperties memory;    //!< The physical device's memory.
	STD(uint32_t)                    family;    //!< The queue's family.
};

/** @brief What a runtime is built for.
 *
 * Frames of width x height, RGBA8 or with fp16 RGBA16F, passes chained evaluations, with or without
 * motion history and the pass stages; with external, in a caller's images (struct
 * vulkan_frame_images) instead of buffers.
 */
struct vulkan_shape {
	STD(uint32_t) width;    //!< The frames' width.
	STD(uint32_t) height;   //!< Their height.
	STD(uint32_t) passes;   //!< The chained evaluations.
	bool          fp16;     //!< RGBA16F frames, not RGBA8.
	bool          motion;   //!< With motion history.
	bool          stages;   //!< With the pass stages.
	bool          external; //!< In a caller's images.
};

/** @brief The caller's images that frames go through, for a runtime built for an external shape.
 *
 * The first pass, the motion estimate, the pass stages, the alpha pass and the fallback sample frame
 * in SHADER_READ_ONLY_OPTIMAL. One pass's post block and the fallback store into answer through
 * answer_view, in GENERAL, or a blit fills it from the network's RGBA32F answer. Both are of the
 * shape's extent and frame format.
 */
struct vulkan_frame_images {
	STD(uint64_t) generation;  //!< Nonzero, and new whenever a handle may have changed.
	VkImageView   frame;       //!< The frame, sampled.
	VkImage       answer;      //!< The answer.
	VkImageView   answer_view; //!< The answer's view.
};

/** @brief A frame's controls (upstream: nr::Controls, whose pass count is the shape's here), in the
 *         ranges that dlsslopd's settings allow. */
struct vulkan_controls {
	STD(uint32_t) style;          //!< DLSSNR.Style.
	float         intensity;      //!< DLSSNR.Intensity.
	float         tone;           //!< LocalToneStrength.
	float         structure;      //!< LocalStructureStrength.
	/** @brief The skin's local structure under the automatic mask; below 0 it follows structure. */
	float         skin;
	float         sharpness;      //!< dlsslop-amd's sharpening stage after every pass, 0..1; 0 skips it.
	float         color_preserve; //!< Its color preservation stage, 0..1; 0 skips it.
	bool          auto_mask;      //!< The automatic mask.
};

/** @brief A runtime's image, null until made, and what it was made as. */
struct vulkan_runtime_image {
	VkImage           image;  //!< The image.
	VkDeviceMemory    memory; //!< Its memory.
	VkImageView       view;   //!< Its view, when it is sampled or stored into.
	STD(uint32_t)     width;  //!< Its width.
	STD(uint32_t)     height; //!< Its height.
	VkFormat          format; //!< Its format; VK_FORMAT_UNDEFINED for none.
	VkImageUsageFlags usage;  //!< Its usage.
};

/** @brief A runtime's buffer, null until made. */
struct vulkan_runtime_buffer {
	VkBuffer       buffer; //!< The buffer.
	VkDeviceMemory memory; //!< Its memory.
};

/** @brief A runtime's pipeline, null until made. */
struct vulkan_runtime_pipeline {
	VkDescriptorSetLayout set_layout; //!< Its one set's layout.
	VkPipelineLayout      layout;     //!< Its layout.
	VkPipeline            pipeline;   //!< The pipeline.
};

/** @brief The runtime's pipelines after the network's kernels' (upstream: the adapters of
 *         nr_runtime.cpp and the temporal variants of the pre and post blocks). */
enum vulkan_runtime_adapter : STD(uint8_t) {
	/** @brief The alpha pass, which restores the frame's alpha when later passes overwrote the
	 *         input. */
	VULKAN_RUNTIME_ALPHA = VULKAN_KERNEL_COUNT,
	VULKAN_RUNTIME_STAGES,    //!< dlsslop-amd's pass stages.
	VULKAN_RUNTIME_LUMA,      //!< The motion estimate's luma pyramid.
	VULKAN_RUNTIME_FLOW,      //!< The motion estimate's flow.
	VULKAN_RUNTIME_PRE,       //!< The pre block with motion history, with the upper clamp.
	VULKAN_RUNTIME_PRE_NH,    //!< The pre block with motion history, without it.
	VULKAN_RUNTIME_POST,      //!< The post block with motion history.
	VULKAN_RUNTIME_VERDICT,   //!< dlsslop-amd's verdict on the frame's waits.
	VULKAN_RUNTIME_FALLBACK,  //!< The fallback that answers with the input when a wait ran out.
	VULKAN_RUNTIME_PIPELINES, //!< The number of pipelines, the kernels' included.
};

/** @brief The motion estimate's pyramid's levels. */
#define VULKAN_RUNTIME_LEVELS 4

/** @brief The most passes a runtime chains. */
#define VULKAN_RUNTIME_MAX_PASSES 16

/** @brief Every object a runtime owns. */
struct vulkan_runtime_objects {
	struct vulkan_runtime_buffer   arena;                                    //!< The activation arena.
	struct vulkan_runtime_buffer   weights;                                  //!< The weights.
	struct vulkan_runtime_buffer   params;                                   //!< The motion parameters.
	/** @brief The verdict: the fallback's grid, which the host reads mapped at grid. */
	struct vulkan_runtime_buffer   verdict;
	STD(uint32_t)                 *grid;                                     //!< The verdict, mapped.
	/** @brief The network's input, sampled. In image mode the caller's frame stands in for the first
	 *         pass's input. */
	struct vulkan_runtime_image    input;
	/** @brief The network's answer. In image mode the caller's answer stands in for an answer in the
	 *         frame's format. */
	struct vulkan_runtime_image    answer;
	struct vulkan_runtime_image    second;                                   //!< The network's second output.
	/** @brief The first pass's input for later passes; in image mode the caller's frame. */
	struct vulkan_runtime_image    shown;
	struct vulkan_runtime_image    scratch;                                  //!< The pass stages' scratch.
	/** @brief The frame's image that blits convert through; in image mode the caller's answer. */
	struct vulkan_runtime_image    frame;
	/** @brief The motion history's luma pyramids, this frame's and the last frame's by parity. */
	struct vulkan_runtime_image    luma[2][VULKAN_RUNTIME_LEVELS];
	struct vulkan_runtime_image    flow[VULKAN_RUNTIME_LEVELS];              //!< The flow between them.
	struct vulkan_runtime_image    history[2];                               //!< The histories that passes read.
	struct vulkan_runtime_image    depth;                                    //!< A depth nothing writes.
	struct vulkan_runtime_image    history_store[VULKAN_RUNTIME_MAX_PASSES]; //!< One history a pass.
	VkSampler                      nearest;                                  //!< The input's sampler.
	VkSampler                      linear;                                   //!< The motion history's sampler.
	/** @brief The network's kernels' pipelines, then the runtime's own. */
	struct vulkan_runtime_pipeline pipelines[VULKAN_RUNTIME_PIPELINES];
	VkDescriptorPool               pool;                                     //!< Every set's pool.
	VkDescriptorSet                kernel_sets[VULKAN_KERNEL_COUNT];         //!< Each kernel's set.
	/* The runtime's own pipelines' sets, by what they differ in: the answer or the scratch, the
	 * parity and level, and the history. */
	VkDescriptorSet                alpha_sets[2];                            //!< The alpha pass's.
	VkDescriptorSet                stage_sets[2];                            //!< The pass stages'.
	VkDescriptorSet                luma_sets[2][VULKAN_RUNTIME_LEVELS];      //!< The luma pyramid's.
	VkDescriptorSet                flow_sets[2][VULKAN_RUNTIME_LEVELS];      //!< The flow's.
	VkDescriptorSet                pre_sets[2];                              //!< The temporal pre block's.
	VkDescriptorSet                post_sets[2];                             //!< The temporal post block's.
	VkDescriptorSet                verdict_set;                              //!< The verdict's.
	VkDescriptorSet                fallback_sets[2];                         //!< The fallback's.
	/** @brief The pre and post blocks' sets of the passes after the first, which sample the input
	 *         where the first samples the frame: in image mode sets of their own, otherwise the first
	 *         pass's. */
	VkDescriptorSet                later_sets[2];
};

/** @brief How a runtime records frames. */
struct vulkan_runtime_state {
	STD(uint32_t) level_width[VULKAN_RUNTIME_LEVELS];  //!< The pyramid's levels' widths.
	STD(uint32_t) level_height[VULKAN_RUNTIME_LEVELS]; //!< Their heights.
	STD(uint32_t) width;                               //!< The frames' width.
	STD(uint32_t) height;                              //!< Their height.
	STD(uint32_t) passes;                              //!< The passes, 1..VULKAN_RUNTIME_MAX_PASSES.
	bool          motion;                              //!< With motion history.
	bool          stages;                              //!< With the pass stages.
	bool          rgba8;                               //!< RGBA8 frames.
	bool          external;                            //!< In the caller's images.
	bool          input_direct;                        //!< The proxy is copied into the input, not blitted.
	/** @brief The post block stores the answer in the frame's format, which is then copied out, or
	 *         in image mode is the caller's answer. */
	bool          answer_direct;
	bool          post_alpha;                          //!< The post block restores the frame's alpha itself.
	/** @brief One pass with motion: the post block writes the history the next frame reads, in turn
	 *         into history[0] and history[1]. */
	bool          pingpong;
	/** @brief More passes with motion: each pass's post block writes its history into the second
	 *         output, which that pass's history_store keeps. */
	bool          stored;
};

/** @brief The motion history that a frame reads. */
struct vulkan_runtime_history {
	STD(uint32_t) parity;  //!< This frame's luma pyramid.
	STD(uint32_t) current; //!< The history it reads.
	/** @brief The pre block's noise seed for the frame, unless it starts the history over. */
	STD(uint32_t) seed;
	bool          latch;   //!< There is a last frame to follow.
};

/** @brief The network built for one frame extent on a device, for one shape of that extent at a time.
 *
 * vulkan_runtime_build() makes one and vulkan_runtime_fini() frees it; a zeroed one is none. The
 * device must have finished the runtime's work before it is freed or reshaped.
 *
 * Invariants:
 * - The activation arena is zeroed at build and never laid out again while the runtime lives: its
 *   values' zero tails depend on it. The persistent runs' sync regions and the tile counters past
 *   the values count frames from the build on, or from the last frame that started over.
 * - Descriptors are written only by a build and vulkan_runtime_reshape(). Every dispatch runs with
 *   the network's input image, or the caller's frame image in image mode, in
 *   SHADER_READ_ONLY_OPTIMAL and every other image that a set binds in GENERAL. The first frame
 *   recorded after a build or reshape moves every image of the runtime's own there from UNDEFINED,
 *   and so does each frame after it until vulkan_runtime_submitted() says one was submitted; the
 *   caller moves its own images into those layouts.
 * - The invalidate-only barriers between steps rely on gfx1201's caches below L2 being
 *   write-through (upstream: nr_graph.cpp:4660-4688); the steps that tile counters order rely on the
 *   queue starting consecutive dispatches' workgroups in order.
 * - The motion history's latch, parity and noise seed change only when vulkan_runtime_submitted()
 *   says that the last frame recorded was submitted, and start over at a reshape.
 */
struct vulkan_runtime {
	/** @brief The device, borrowed: it must outlive the runtime. nullptr: none is built. */
	struct vulkan_device const   *device;
	struct vulkan_step           *steps;         //!< The plan's steps, which the runtime frees.
	STD(uint32_t)                *push;          //!< Their push words, which the runtime frees.
	STD(uint32_t)                *timeouts;      //!< The plan's timeouts, which the runtime frees.
	STD(size_t)                   step_count;    //!< The number of steps.
	STD(size_t)                   timeout_count; //!< The number of timeouts.
	STD(uint64_t)                 values_end;    //!< The plan's values_end.
	/** @brief In image mode, the caller's images bound; none: generation 0. */
	struct vulkan_frame_images    images;
	struct vulkan_runtime_objects objects;       //!< Every object the runtime owns.
	struct vulkan_shape           shape;         //!< The shape it is built or reshaped for.
	struct vulkan_runtime_state   state;         //!< How frames of that shape are recorded.
	struct vulkan_runtime_history history;       //!< The history after the last frame submitted.
	struct vulkan_runtime_history recorded;      //!< The history after the last frame recorded.
	/** @brief A frame submitted since the last build or reshape moved the images into their
	 *         layouts. */
	bool                          settled;
};

/** @brief Checks a Vulkan result.
 *
 * @param result The result.
 * @param what   What returned it, for the words "WHAT failed (VkResult N)".
 * @param e      Receives the words, or nullptr.
 * @return       ERROR_NONE for VK_SUCCESS, otherwise ERROR_FAILED.
 */
extern enum error_code
vulkan_check (VkResult      result,
              char const   *what,
              struct error *e);

/** @brief Finds the first of a device's memory types among bits with every property wanted.
 *
 * @param memory The device's memory.
 * @param bits   The types allowed, bit i for type i.
 * @param want   The properties wanted.
 * @param type   Receives the type.
 * @param e      Receives the words for none, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED when no type is suitable.
 */
extern enum error_code
vulkan_memory_type (VkPhysicalDeviceMemoryProperties const *memory,
                    STD(uint32_t)                           bits,
                    VkMemoryPropertyFlags                   want,
                    STD(uint32_t)                          *type,
                    struct error                           *e);

/** @brief The most bytes one of a device's storage buffers holds bound whole, as vulkan_plan_init()
 *         takes it.
 *
 * @param device The device.
 * @return       The smaller of its maxStorageBufferRange and maxMemoryAllocationSize; UINT64_MAX
 *               when the device cannot say, which fails its build.
 */
extern STD(uint64_t)
vulkan_storage_limit (struct vulkan_device const *device);

/** @brief Builds the network for a shape on a device, from a plan of the shape's extent.
 *
 * The plan is of the shape's extent on the device's vulkan_storage_limit(), with the kernels without
 * the upper clamp that the model's weights allow (vulkan_plan_unclamp()). The SPIR-V comes from
 * paths->shaders, the weights from paths->model and the pipeline cache from paths->cache, with the
 * pipelines of every shape of that extent. The queue must be free of the frames of a runtime being
 * replaced.
 *
 * @param dest   Receives the runtime; none on a failure.
 * @param device The device, which must outlive the runtime.
 * @param paths  Where the network's files are.
 * @param shape  The shape.
 * @param plan   The plan. The runtime takes its steps, push words and timeouts, and
 *               vulkan_plan_fini() frees the rest, whether the build succeeds or not.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_runtime_build (struct vulkan_runtime      *dest,
                      struct vulkan_device const *device,
                      struct vulkan_paths const  *paths,
                      struct vulkan_shape const  *shape,
                      struct vulkan_plan         *plan,
                      struct error               *e);

/** @brief Frees a runtime and empties it; the device must have finished its work.
 *
 * @param runtime The runtime, or nullptr.
 */
extern void
vulkan_runtime_fini (struct vulkan_runtime *runtime);

/** @brief Makes a runtime what a build makes for a shape of the extent it was built for.
 *
 * Its images and every descriptor set become what the shape wants, with images bound in image mode.
 * It makes no pipeline and submits nothing; the next frame moves the images into their layouts. The
 * weights, the arena and the pipelines stay, and so does each image that the shape wants as it is,
 * its contents discarded as a new image's are. For the runtime's own shape it only writes the sets,
 * which binds the images, and keeps its images, their contents and the motion history. A build in
 * image mode binds no images: its sets that would bind them wait for a reshape. A runtime that fails
 * to reshape must be freed.
 *
 * @param runtime The runtime.
 * @param shape   The shape.
 * @param images  The caller's images to bind in image mode, or nullptr for those bound before.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_runtime_reshape (struct vulkan_runtime            *runtime,
                        struct vulkan_shape const        *shape,
                        struct vulkan_frame_images const *images,
                        struct error                     *e);

/** @brief Records a frame of the runtime's shape, from a proxy buffer into an answer buffer.
 *
 * The proxy holds the frame's pixels packed in its format, and the answer receives the network's in
 * the same form. Transfers read the proxy and write the answer, which the caller's barriers order
 * against the transfers before and after. When a wait of the frame runs out, the answer is the proxy
 * unchanged.
 *
 * @param runtime  The runtime.
 * @param cmd      The command buffer.
 * @param proxy    The proxy.
 * @param answer   The answer.
 * @param controls The frame's controls.
 * @param reset    Start the frame over: zero the sync regions and tile counters as the build left
 *                 them, and drop the motion history for this frame.
 * @param queries  Timestamp queries, or VK_NULL_HANDLE for none.
 * @param query    The query written once the frame is in the network's input; query + 1 is written
 *                 once the network is done.
 */
extern void
vulkan_runtime_record_buffers (struct vulkan_runtime        *runtime,
                               VkCommandBuffer               cmd,
                               VkBuffer                      proxy,
                               VkBuffer                      answer,
                               struct vulkan_controls const *controls,
                               bool                          reset,
                               VkQueryPool                   queries,
                               STD(uint32_t)                 query);

/** @brief Records a frame of the runtime's shape in image mode, from the frame image that
 *         vulkan_runtime_reshape() bound into its answer image.
 *
 * Compute shaders sample the frame, and compute shaders or a blit write the answer; the caller's
 * barriers order them against the work before and after, and the runtime records none on either
 * image.
 *
 * @param runtime  The runtime.
 * @param cmd      The command buffer.
 * @param controls The frame's controls.
 * @param reset    Start the frame over, as vulkan_runtime_record_buffers() does.
 */
extern void
vulkan_runtime_record_images (struct vulkan_runtime        *runtime,
                              VkCommandBuffer               cmd,
                              struct vulkan_controls const *controls,
                              bool                          reset);

/** @brief Says that the frame recorded last was submitted, so that the next frame reads the motion
 *         history it writes. A frame that is recorded and not submitted leaves the history as it
 *         was.
 *
 * @param runtime The runtime.
 */
extern void
vulkan_runtime_submitted (struct vulkan_runtime *runtime);

/** @brief Whether a wait of the frame submitted last ran out; that frame must have finished.
 *
 * Its answer is then its proxy, and the sync regions and counters may stay short of their counts
 * until a frame starts over.
 *
 * @param runtime The runtime.
 * @return        true if one ran out.
 */
extern bool
vulkan_runtime_timed_out (struct vulkan_runtime const *runtime);

/** @brief The generation of the caller's images that vulkan_runtime_reshape() bound.
 *
 * @param runtime The runtime.
 * @return        The generation; 0 for none.
 */
extern STD(uint64_t)
vulkan_runtime_bound (struct vulkan_runtime const *runtime);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_VULKAN_RUNTIME_H_ */
