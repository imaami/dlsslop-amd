/** @file
 *
 * OptiScaler's output-scaling pass, on the layer's dispatch table.
 *
 * Used here for one thing: supersampling. When the model works above the frame's resolution the proxy
 * has to be enlarged on the way in and the model's answer averaged back on the way out, and doing
 * either with a plain bilinear sampler aliases -- which is the whole reason upstream gave the pass its
 * own downscaler rather than reusing the resolve's sampler.
 *
 * Two differences from upstream in what the pass dispatches, both deliberate.
 *
 * The filter is chosen per instance at construction rather than read from a global on every dispatch,
 * because the enlarge and the average want different ones and a global cannot hold two answers.
 *
 * The dispatch is sized from the images it is given. Upstream's Vulkan copy reads the sizes off
 * State::currentFeature instead, which is the same fault its own notes record fixing in the Direct3D 12
 * copy: "sized its dispatch from the global current feature rather than from the resources passed in.
 * Those coincide for the conventional Output Scaling chain, so the bug stayed invisible until something
 * else called it." This is something else calling it.
 *
 * Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_SCALER_VK_H_
#define DLSSLOP_AMD_LAYER_SCALER_VK_H_

#ifdef __cplusplus
# include <cstdint>
# define SCALER_VK_STD(x) std::x
#else
# include <stdint.h>
# define SCALER_VK_STD(x) x
#endif

#include "shader_vk.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Upstream's Scaler numbering, kept identical so a value copied from an OptiScaler profile
 *         means the same thing here.
 *
 * FSR1 is absent: it needs a different constant block and its own shader, and it is an upscaler
 * rather than the averaging filter this pass wants.
 */
enum scaler_vk_filter : SCALER_VK_STD(uint32_t) {
	SCALER_VK_FSR1        = 0, //!< Not supported here; falls back to Lanczos3.
	SCALER_VK_BICUBIC     = 1,
	SCALER_VK_CATMULL_ROM = 2,
	SCALER_VK_LANCZOS2    = 3,
	SCALER_VK_LANCZOS3    = 4,
	SCALER_VK_KAISER2     = 5,
	SCALER_VK_KAISER3     = 6,
	SCALER_VK_MAGIC       = 7,
	SCALER_VK_COUNT       = 8
};

// The protocol carries this number, so the two enumerations have to agree. They are separate types
// because the layer should not have to include the shared header to name a filter -- but a mismatch
// between them was a real bug: the default came out as Catmull-Rom because the protocol's older,
// narrower numbering put Lanczos3 at 2.
static_assert(SCALER_VK_LANCZOS3 == 4, "the protocol's Downscaler numbering must match this one");

/** @brief The constant slots, and the descriptor sets: one of each per dispatch. */
static constexpr SCALER_VK_STD(uint32_t) SCALER_VK_SLOTS = 6;

/** @brief What struct scaler_vk records in its flags. */
enum scaler_vk_flags {
	SCALER_VK_UPSAMPLE = 1 //!< The pass enlarges, with bcus; otherwise it averages.
};

/** @brief One direction of the supersampling: the enlarge or the average.
 *
 * A zeroed object is an empty one. scaler_vk() builds one, and scaler_vk_fini() destroys what it
 * owns. A constant slot is the scaling shaders' 256-byte block rounded up to the device's uniform
 * buffer offset alignment. The flags are 64 bits wide, which fills the padding that a narrower
 * member would leave.
 */
struct scaler_vk {
	struct shader_vk        shader;                           //!< The pipeline, ring and sampler.
	VkDescriptorSet         descriptor_sets[SCALER_VK_SLOTS]; //!< One per constant slot.
	VkDeviceSize            slot_stride;                      //!< A constant slot's size.
	SCALER_VK_STD(uint64_t) flags;                            //!< enum scaler_vk_flags.
	VkResult                error;                            //!< What the build returned.
	SCALER_VK_STD(uint32_t) slot;                             //!< The next dispatch's slot.
};

/** @brief The name of a downscaler.
 *
 * @param filter An enum scaler_vk_filter value.
 * @return       Its name; "lanczos3" for SCALER_VK_FSR1 and for any value from SCALER_VK_COUNT on,
 *               which fall back to Lanczos3.
 */
extern char const *
scaler_vk_filter_name (SCALER_VK_STD(uint32_t) filter);

/** @brief Builds a scaling pass on a device.
 *
 * Creates the sampler, a ring of SCALER_VK_SLOTS constant slots, the layouts, a descriptor set per
 * slot and the pipeline, in that order, and logs the pass's name and filter when it is built.
 *
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device.
 * @param physical_device The device's physical device.
 * @param upsample        true for the bicubic enlarge, false for an average.
 * @param filter          The average's filter: an enum scaler_vk_filter value. SCALER_VK_FSR1 and
 *                        values from SCALER_VK_COUNT on give Lanczos3; the enlarge ignores it.
 * @return                The pass. Its error is VK_SUCCESS if it was built; otherwise the pass
 *                        holds what was built before the failure, which scaler_vk_fini() destroys.
 */
extern struct scaler_vk
scaler_vk (struct device_table const   *vk,
           struct instance_table const *instance,
           VkDevice                     device,
           VkPhysicalDevice             physical_device,
           bool                         upsample,
           SCALER_VK_STD(uint32_t)      filter);

/** @brief Builds a scaling pass in place.
 *
 * @param dest            An empty pass, or nullptr.
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device.
 * @param physical_device The device's physical device.
 * @param upsample        true for the bicubic enlarge, false for an average.
 * @param filter          The average's filter, as for scaler_vk().
 * @return                The pass's error: VK_SUCCESS if it was built, VK_ERROR_INITIALIZATION_FAILED
 *                        for a null @a dest.
 */
extern VkResult
scaler_vk_init (struct scaler_vk            *dest,
                struct device_table const   *vk,
                struct instance_table const *instance,
                VkDevice                     device,
                VkPhysicalDevice             physical_device,
                bool                         upsample,
                SCALER_VK_STD(uint32_t)      filter);

/** @brief Destroys what the pass owns, then leaves it empty.
 *
 * @param dest The pass, or nullptr.
 */
extern void
scaler_vk_fini (struct scaler_vk *dest);

/** @brief Records one scaling dispatch, and the barrier after it.
 *
 * The constants come from the images this call is handed, not from a global that happens to agree
 * most of the time.
 *
 * @param p           The pass.
 * @param cb          The command buffer to record into.
 * @param source      The source view, in SHADER_READ_ONLY_OPTIMAL.
 * @param dest        The destination view, in GENERAL.
 * @param src_width   The source's width.
 * @param src_height  The source's height.
 * @param dest_width  The destination's width.
 * @param dest_height The destination's height.
 * @return            true if the dispatch was recorded: the pass is built and the command buffer
 *                    and both views are set.
 */
extern bool
scaler_vk_dispatch (struct scaler_vk        *p,
                    VkCommandBuffer          cb,
                    VkImageView              source,
                    VkImageView              dest,
                    SCALER_VK_STD(uint32_t)  src_width,
                    SCALER_VK_STD(uint32_t)  src_height,
                    SCALER_VK_STD(uint32_t)  dest_width,
                    SCALER_VK_STD(uint32_t)  dest_height);

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef SCALER_VK_STD

#endif /* DLSSLOP_AMD_LAYER_SCALER_VK_H_ */
