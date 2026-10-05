/** @file
 *
 * What a frame is processed with: processing.h.
 */
// SPDX-License-Identifier: MIT
#include <stdatomic.h>

#include "processing.h"
#include "tuning.h"

struct processing_settings
processing_settings (void)
{
	return (struct processing_settings){
		.tuning         = NATIVE_TUNING_DEFAULTS,
		.skin_structure = -1.0f,
		.motion_quality = kMVecBalanced,
		.motion_grid    = kMVecPixels4,
		.auto_mask      = true,
		.precision16    = true,
	};
}

enum error_code
processing_read (struct ShmHeader const     *h,
                 struct processing_settings *dest,
                 struct error               *e)
{
	// Each word is read once.
	dest->fp16 = atomic_load(&h->hdrEncode) != 0;
	bool const hdr = atomic_load(&h->hdrDetected) != kHdrNone && atomic_load(&h->colourMode) != kColourDisplay;
	dest->precision16 = hdr || atomic_load(&h->sdr16Multipass) != 0;
	dest->motion = atomic_load(&h->mvecEnabled) != 0;
	dest->motion_quality = ShmMVecQuality(h);
	dest->motion_grid = ShmMVecPixelSize(h);
	dest->tuning.intensity = BitsToFloat(atomic_load(&h->intensityBits));
	dest->tuning.tone = BitsToFloat(atomic_load(&h->localToneBits));
	dest->tuning.structure = BitsToFloat(atomic_load(&h->localStructureBits));
	dest->tuning.sharpness = BitsToFloat(atomic_load(&h->sharpnessBits));
	dest->color_preserve = BitsToFloat(atomic_load(&h->colorPreserveBits));
	dest->style = atomic_load(&h->style);
	dest->skin_structure = BitsToFloat(atomic_load(&h->skinStructureBits));
	uint32_t const mask = atomic_load(&h->autoMask);
	dest->auto_mask = mask != 0;
	if (atomic_load(&h->preset) || dest->style > 2 || mask > 1
	    || !(dest->skin_structure >= -1 && dest->skin_structure <= 2))
		return error_reject(e, "the preset must be 0, style 0..2, auto-mask 0 or 1 and skin structure -1..2");
	if (!(dest->color_preserve >= 0 && dest->color_preserve <= 1))
		return error_reject(e, "invalid color preservation strength");
	return tuning_validate(&dest->tuning, e);
}

struct vulkan_frame
processing_vulkan_frame (uint32_t                          width,
                         uint32_t                          height,
                         uint32_t                          passes,
                         struct processing_settings const *settings)
{
	struct vulkan_frame f = VULKAN_FRAME_DEFAULTS;
	f.width = width;
	f.height = height;
	f.fp16 = settings->fp16;
	f.passes = passes;
	f.intensity = settings->tuning.intensity;
	f.local_tone = settings->tuning.tone;
	f.local_structure = settings->tuning.structure;
	f.style = settings->style;
	f.skin_structure = settings->skin_structure;
	f.auto_mask = settings->auto_mask;
	f.sharpness = settings->tuning.sharpness;
	f.color_preserve = settings->color_preserve;
	f.motion = settings->motion;
	return f;
}
