/** @file
 *
 * Where a source picture sits in the network's padded raster: struct geometry (kernel_args.h) for a
 * source extent and a tier, and the checks of one. geometry.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_GEOMETRY_H_
#define DLSSLOP_AMD_BACKEND_GEOMETRY_H_

#include "error.h"
#include "kernel_args.h"

/** @brief The geometry of a source picture at a tier: the picture fitted, aspect kept and centered,
 *         into the tier's viewport.
 *
 * A source larger than the largest tier is fitted into it.
 *
 * @param dest          Receives the geometry; unchanged on failure.
 * @param source_width  The source's width, 1..16384.
 * @param source_height The source's height, 1..16384.
 * @param tier_height   The tier's height, 720, 900 or 1080, or 0 for the smallest tier that holds
 *                      the source.
 * @param e             Receives the words of a failure, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED for a source extent or tier out of range.
 */
extern enum error_code
geometry_init (struct geometry *dest,
               unsigned         source_width,
               unsigned         source_height,
               unsigned         tier_height,
               struct error    *e);

/** @brief Checks that a geometry is what geometry_init() makes of its source extent and tier.
 *
 * @param g The geometry.
 * @param e Receives the words of a failure, or nullptr.
 * @return  ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
geometry_validate (struct geometry const *g,
                   struct error          *e);

/** @brief Whether a geometry's fitted picture is nonempty and inside the padded extent.
 *
 * @param g The geometry.
 * @return  true if it is.
 */
extern bool
geometry_fits (struct geometry const *g);

#endif /* DLSSLOP_AMD_BACKEND_GEOMETRY_H_ */
