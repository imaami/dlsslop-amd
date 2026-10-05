/** @file
 *
 * One frame of the Vulkan network's work, for dlsslopd and the in-layer network.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_VULKAN_FRAME_H_
#define DLSSLOP_AMD_COMMON_VULKAN_FRAME_H_

#include <stdint.h>

/** @brief One frame's work.
 *
 * The frame is width x height tightly packed RGBA8, or RGBA16F with fp16, holding display-encoded
 * values, and the answer comes back in the same form. VULKAN_FRAME_DEFAULTS initializes one; a frame
 * of width 0 is no frame.
 */
struct vulkan_frame {
	uint32_t width;           //!< The frame's width.
	uint32_t height;          //!< Its height.
	uint32_t passes;          //!< The network's chained evaluations.
	uint32_t style;           //!< DLSSNR.Style, 0..2.
	/** @brief The network's own controls (NVIDIA's DLSSNR.Intensity, LocalToneStrength,
	 *         LocalStructureStrength), not the HIP backend's residual filters. The model takes each
	 *         up to NETWORK_RECORDER_MAX_CONTROL; more counts as that. */
	float    intensity;
	float    local_tone;      //!< See intensity.
	float    local_structure; //!< See intensity.
	/** @brief The local structure of skin under the automatic mask; a negative one follows
	 *         local_structure. */
	float    skin_structure;
	/** @brief dlsslop-amd's own stages after every pass, 0..1 each, as on HIP: sharpening, and color
	 *         preservation against the frame the first pass saw. Either builds them in, as a larger
	 *         pass count does. */
	float    sharpness;
	float    color_preserve;  //!< See sharpness.
	bool     fp16;            //!< The frame is RGBA16F, not RGBA8.
	bool     auto_mask;       //!< The automatic mask, under which skin takes its own structure.
	/** @brief The network's history, fed by the runtime's motion estimate. It starts over whenever
	 *         the frame's shape, pass count or controls change. */
	bool     motion;
};

/** @brief A frame's defaults, as an initializer: one pass, the network's controls at 1, the skin
 *         following the local structure, the automatic mask on, and the rest 0. */
#define VULKAN_FRAME_DEFAULTS {.passes = 1, .intensity = 1.0f, .local_tone = 1.0f, .local_structure = 1.0f, \
                               .skin_structure = -1.0f, .auto_mask = true}

#endif /* DLSSLOP_AMD_COMMON_VULKAN_FRAME_H_ */
