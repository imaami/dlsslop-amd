/** @file
 *
 * What a frame is processed with, as the channel states it, for dlsslopd and the in-layer network.
 * processing.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_PROCESSING_H_
#define DLSSLOP_AMD_BACKEND_PROCESSING_H_

#include <stdint.h>

#include "error.h"
#include "kernel_args.h"
#include "shm_protocol.h"
#include "vulkan_frame.h"

/** @brief A frame's settings: processing_settings() makes the defaults, processing_read() reads a
 *         frame's from the channel. */
struct processing_settings {
	struct native_tuning tuning;         //!< The HIP network's residual tuning, the Vulkan network's controls.
	/** @brief The Vulkan model's skin structure, -1..2; a negative one follows the local structure.
	 *         The HIP network has only the defaults of the model's own conditioning. */
	float                skin_structure;
	float                color_preserve; //!< Color preservation after every pass, 0..1.
	uint32_t             style;          //!< The Vulkan model's style, 0..2.
	uint32_t             motion_quality; //!< The motion estimate's quality: kMVecFast and the rest.
	uint32_t             motion_grid;    //!< The motion estimate's spacing: kMVecPixels1 and the rest.
	bool                 auto_mask;      //!< The Vulkan model's automatic mask.
	bool                 fp16;           //!< The frame is RGBA16F, not RGBA8.
	bool                 precision16;    //!< Passes after the first take float16 feedback.
	bool                 motion;         //!< The network's motion history.
};

/** @brief The settings of a frame that no channel states: the tuning's defaults, no color
 *         preservation, style 0, the skin following the local structure, the automatic mask on,
 *         RGBA8, float16 feedback, and no motion history, whose estimate would be balanced over
 *         4-pixel cells.
 *
 * @return The settings.
 */
extern struct processing_settings
processing_settings (void);

/** @brief Reads the channel's settings for the next frame.
 *
 * An NVIDIA preset neither network has, or a control outside its range, is rejected: older and
 * external clients can store either.
 *
 * @param h    The channel's header.
 * @param dest Receives the settings; what it holds after a rejection is not to be used.
 * @param e    Receives the words of a rejection, or nullptr.
 * @return     ERROR_NONE, or ERROR_REJECTED.
 */
extern enum error_code
processing_read (struct ShmHeader const     *h,
                 struct processing_settings *dest,
                 struct error               *e);

/** @brief A frame for the Vulkan network.
 *
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   The network's chained evaluations.
 * @param settings The frame's settings.
 * @return         The frame of @a width x @a height through the network @a passes times, as
 *                 @a settings say.
 */
extern struct vulkan_frame
processing_vulkan_frame (uint32_t                          width,
                         uint32_t                          height,
                         uint32_t                          passes,
                         struct processing_settings const *settings);

#endif /* DLSSLOP_AMD_BACKEND_PROCESSING_H_ */
