/** @file
 *
 * The composition pass, on the layer's dispatch table.
 *
 * This is OptiScaler's DlssNr_Vk with the loader-exported Vulkan calls replaced by the layer's next-
 * chain table. The build declares its storage images with an Unknown format for the actual
 * RGBA8/BGRA8/FP16 targets; all shader arithmetic remains unchanged.
 *
 * Three things are worth knowing before reading the implementation.
 *
 * Every binding is written every dispatch. The shader declares all seven resources at file scope and
 * branches on gMode, so all of them are statically reachable from the entry point and Vulkan requires
 * a valid descriptor for each one whether a given mode reads it or not. Slots a mode has no use for
 * get a 1x1 dummy rather than a null handle.
 *
 * Constants are slotted rather than overwritten. A single mapped uniform buffer would be wrong here:
 * encode and resolve run in the same frame with different constants, and the second write would land
 * before the first dispatch had read it. The buffer holds a ring of slots and each dispatch takes the
 * next, at an offset the device's own alignment rule allows.
 *
 * Layouts are the caller's to declare and this pass's to respect. It never guesses what state an
 * image arrived in.
 *
 * Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_DLSSNR_PASS_H_
#define DLSSLOP_AMD_LAYER_DLSSNR_PASS_H_

#ifdef __cplusplus
# include <cstdint>
# define DLSS_NR_PASS_STD(x) std::x
#else
# include <stdint.h>
# define DLSS_NR_PASS_STD(x) x
#endif

#include "dlssnr/DlssNr_Common.h"
#include "shader_vk.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Constant slots for each frame in flight.
 *
 * Enough for several dispatches per frame across the frames that can be in flight. Encode and
 * resolve are two; the debug views, the calibration grid and the downsample are the others.
 */
static constexpr DLSS_NR_PASS_STD(uint32_t) DLSS_NR_PASS_SLOTS_PER_FRAME = 6;

/** @brief The frames that can be in flight. */
static constexpr DLSS_NR_PASS_STD(uint32_t) DLSS_NR_PASS_FRAMES_IN_FLIGHT = 3;

/** @brief The constant slots, and the descriptor sets: one of each per dispatch. */
static constexpr DLSS_NR_PASS_STD(uint32_t) DLSS_NR_PASS_SLOTS =
	DLSS_NR_PASS_SLOTS_PER_FRAME * DLSS_NR_PASS_FRAMES_IN_FLIGHT;

/** @brief The composition shader's pass.
 *
 * A zeroed object is an empty one. dlss_nr_pass() builds one, and dlss_nr_pass_fini() destroys what
 * it owns: the objects whose handles the pass and its base hold. The placeholder's handles are
 * stored only once all three exist, and its move to GENERAL is recorded with them. A constant slot
 * is struct dlss_nr_constants rounded up to the device's uniform buffer offset alignment, because a
 * uniform buffer binding can be offset only to a multiple of it.
 */
struct dlss_nr_pass {
	struct shader_vk           shader;                              //!< The pipeline, ring and sampler.
	VkDescriptorSet            descriptor_sets[DLSS_NR_PASS_SLOTS]; //!< One per constant slot.
	VkDeviceSize               slot_stride;                         //!< A constant slot's size.
	VkImage                    dummy_image;                         //!< The placeholder's image.
	VkDeviceMemory             dummy_memory;                        //!< The placeholder's memory.
	VkImageView                dummy_view;                          //!< The placeholder's view.
	VkResult                   error;                               //!< What the build returned.
	DLSS_NR_PASS_STD(uint32_t) slot;                                //!< The next dispatch's slot.
};

/** @brief Builds the composition pass on a device.
 *
 * Creates the sampler, a ring of DLSS_NR_PASS_SLOTS constant slots, the layouts, a descriptor set
 * per slot and the pipeline, in that order, and logs the first that fails. The placeholder is made
 * by the first dispatch.
 *
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device.
 * @param physical_device The device's physical device.
 * @return                The pass. Its error is VK_SUCCESS if it was built; otherwise the pass
 *                        holds what was built before the failure, which dlss_nr_pass_fini()
 *                        destroys.
 */
extern struct dlss_nr_pass
dlss_nr_pass (struct device_table const   *vk,
              struct instance_table const *instance,
              VkDevice                     device,
              VkPhysicalDevice             physical_device);

/** @brief Builds the composition pass in place.
 *
 * @param dest            An empty pass, or nullptr.
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device.
 * @param physical_device The device's physical device.
 * @return                The pass's error: VK_SUCCESS if it was built, VK_ERROR_INITIALIZATION_FAILED
 *                        for a null @a dest.
 */
extern VkResult
dlss_nr_pass_init (struct dlss_nr_pass         *dest,
                   struct device_table const   *vk,
                   struct instance_table const *instance,
                   VkDevice                     device,
                   VkPhysicalDevice             physical_device);

/** @brief Destroys what the pass owns, then leaves it empty.
 *
 * @param dest The pass, or nullptr.
 */
extern void
dlss_nr_pass_fini (struct dlss_nr_pass *dest);

/** @brief Records one dispatch of the composition shader, and the barrier after it.
 *
 * Any of the four read views may be VK_NULL_HANDLE, in which case the placeholder is bound; the
 * target may not, because a mode that writes nothing has no reason to run. The transitions that got
 * the images into their layouts are the caller's to record.
 *
 * The constants go into slot @a p->slot, and the call then advances it. A copy into that slot that
 * is recorded before this call therefore lands where the dispatch reads.
 *
 * @param p             The pass.
 * @param cmd_list      The command buffer to record into.
 * @param constants     What the shader reads.
 * @param threads_x     The threads across: the shader's 8x8 groups cover them.
 * @param threads_y     The threads down.
 * @param source        gSource, read in @a source_layout, or VK_NULL_HANDLE.
 * @param model         gModel, read in SHADER_READ_ONLY_OPTIMAL, or VK_NULL_HANDLE.
 * @param original      gOriginal, read in SHADER_READ_ONLY_OPTIMAL, or VK_NULL_HANDLE.
 * @param motion        gMotion, read in @a motion_layout, or VK_NULL_HANDLE.
 * @param target        gTarget, written in GENERAL.
 * @param keep          gKeep, written in GENERAL, or VK_NULL_HANDLE.
 * @param source_layout The layout of source.
 * @param motion_layout The layout of motion.
 * @return              true if the dispatch was recorded: the pass is built, the command buffer and
 *                      the target are set, and the placeholder exists.
 */
extern bool
dlss_nr_pass_dispatch (struct dlss_nr_pass            *p,
                       VkCommandBuffer                 cmd_list,
                       struct dlss_nr_constants const *constants,
                       DLSS_NR_PASS_STD(uint32_t)      threads_x,
                       DLSS_NR_PASS_STD(uint32_t)      threads_y,
                       VkImageView                     source,
                       VkImageView                     model,
                       VkImageView                     original,
                       VkImageView                     motion,
                       VkImageView                     target,
                       VkImageView                     keep,
                       VkImageLayout                   source_layout,
                       VkImageLayout                   motion_layout);

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef DLSS_NR_PASS_STD

#endif /* DLSSLOP_AMD_LAYER_DLSSNR_PASS_H_ */
