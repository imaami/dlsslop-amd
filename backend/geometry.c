/** @file
 *
 * Where a source picture sits in the network's padded raster: geometry.h.
 */
// SPDX-License-Identifier: MIT
// The fit is a port of native_input_geometry.h from
// https://github.com/lmxxf/dlss5-on-amd-9070xt-porting
// See kernel_math.cpp for the retained upstream copyright and MIT permission notice.
#include <stdint.h>
#include <string.h>

#include "error.h"
#include "geometry.h"
#include "shm_protocol.h"

enum error_code
geometry_init (struct geometry *dest,
               unsigned         source_width,
               unsigned         source_height,
               unsigned         tier_height,
               struct error    *e)
{
	if (!source_width || !source_height || source_width > 16384 || source_height > 16384)
		return error_fail(e, "source extent must be in 1..16384");

	// The named tier, or the smallest that holds the source, else the largest.
	struct NativeTier const *tier;
	if (tier_height) {
		tier = ShmNativeTier(tier_height);
		if (!tier)
			return error_fail(e, "network height must be 0, 720, 900 or 1080");
	} else {
		tier = kNativeTiers;
		while (tier < kNativeTiers + kNativeTierCount - 1 &&
		       (source_width > tier->width || source_height > tier->height))
			++tier;
	}

	// The source fitted into the tier, aspect kept and centered (upstream:
	// NativeInputGeometry::Make).
	unsigned const width = tier->width;
	unsigned const valid_height = tier->height;
	unsigned fit_width = width;
	unsigned fit_height = valid_height;
	if ((uint64_t)source_width * valid_height >= (uint64_t)source_height * width)
		fit_height = (unsigned)(((uint64_t)source_height * width + source_width / 2) / source_width);
	else
		fit_width = (unsigned)(((uint64_t)source_width * valid_height + source_height / 2) / source_height);
	if (!fit_width)
		fit_width = 1;
	if (!fit_height)
		fit_height = 1;

	*dest = (struct geometry){
		.source_width = source_width,
		.source_height = source_height,
		.width = width,
		.height = tier->networkHeight,
		.valid_height = valid_height,
		.x = (width - fit_width) / 2,
		.y = (valid_height - fit_height) / 2,
		.fit_width = fit_width,
		.fit_height = fit_height,
	};
	return ERROR_NONE;
}

enum error_code
geometry_validate (struct geometry const *g,
                   struct error          *e)
{
	// struct geometry is nine unsigned members (kernel_args.h asserts its size): no padding.
	struct geometry expected;
	if (geometry_init(&expected, g->source_width, g->source_height, g->valid_height, nullptr) ||
	    memcmp(&expected, g, sizeof expected))
		return error_fail(e, "inconsistent codec geometry");
	return ERROR_NONE;
}

bool
geometry_fits (struct geometry const *g)
{
	return g->fit_width && g->fit_height && g->x < g->width && g->y < g->height &&
	       g->fit_width <= g->width - g->x && g->fit_height <= g->height - g->y;
}
