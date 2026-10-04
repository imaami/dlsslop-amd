/** @file
 *
 * The CPU references of the daemon's kernels: reference.h.
 */
// SPDX-License-Identifier: MIT
#include <math.h>
#include <stdint.h>

#include "error.h"
#include "geometry.h"
#include "kernel_math.h"
#include "reference.h"
#include "tuning.h"

/** @brief Whether a geometry's fitted picture lies inside a padded extent of at most 16384 per
 *         dimension, as the tuning and the color preservation take it.
 *
 * @param g The geometry.
 * @return  true if it does.
 */
static bool
fits_bounded (struct geometry const *g)
{
	return geometry_fits(g) && g->width <= 16384 && g->height <= 16384;
}

enum error_code
reference_encode_proxy (uint8_t const         *source,
                        struct geometry const *g,
                        bool                   fp16,
                        float                 *rgba,
                        struct error          *e)
{
	enum error_code const code = geometry_validate(g, e);
	if (code)
		return code;
	if (!source)
		return error_fail(e, "null source image");
	if (!kernel_math_encode_proxy(source, g, fp16, rgba))
		return error_reject(e, "FP16 proxy contains nonfinite RGB samples");
	return ERROR_NONE;
}

enum error_code
reference_feedback_neural_rgb (float const           *neural_rgb,
                               struct geometry const *g,
                               bool                   precision16,
                               float                 *rgba,
                               struct error          *e)
{
	enum error_code const code = geometry_validate(g, e);
	if (code)
		return code;
	if (!neural_rgb)
		return error_fail(e, "null neural feedback image");
	if (!kernel_math_feedback_neural_rgb(neural_rgb, g, precision16, rgba))
		return error_fail(e, "neural feedback contains nonfinite or FP16-overflow samples");
	return ERROR_NONE;
}

enum error_code
reference_decode_neural_proxy (uint8_t const         *original,
                               struct geometry const *g,
                               bool                   fp16,
                               float const           *neural_rgb,
                               uint8_t               *output,
                               struct error          *e)
{
	enum error_code const code = geometry_validate(g, e);
	if (code)
		return code;
	if (!original || !neural_rgb)
		return error_fail(e, "null decode image");
	if (!kernel_math_decode_neural_proxy(original, g, fp16, neural_rgb, output))
		return error_reject(e, "neural output contains nonfinite or FP16-overflow samples");
	return ERROR_NONE;
}

enum error_code
reference_tune_neural_rgb (float const                *input_rgba,
                           float const                *raw_rgb,
                           struct geometry const      *g,
                           struct native_tuning const *tuning,
                           float                      *output,
                           struct error               *e)
{
	enum error_code const code = tuning_validate(tuning, e);
	if (code)
		return code;
	if (!fits_bounded(g))
		return error_fail(e, "invalid native tuning geometry");
	if (!input_rgba || !raw_rgb || input_rgba == output || raw_rgb == output)
		return error_fail(e, "native tuning requires distinct input and output buffers");
	if (!kernel_math_tune_neural_rgb(input_rgba, raw_rgb, g, tuning, output))
		return error_fail(e, "native tuning produced nonfinite RGB");
	return ERROR_NONE;
}

enum error_code
reference_preserve_color (float const           *original_rgba,
                          float const           *model_rgb,
                          struct geometry const *g,
                          float                  strength,
                          float                 *output,
                          struct error          *e)
{
	if (!isfinite(strength) || strength < 0.0f || strength > 1.0f)
		return error_fail(e, "color preservation must be finite and within 0..1");
	if (!original_rgba || !model_rgb || !fits_bounded(g))
		return error_fail(e, "invalid color preservation buffers or geometry");
	if (output == model_rgb || output == original_rgba)
		return error_fail(e, "color preservation requires distinct output");
	if (!kernel_math_preserve_color(original_rgba, model_rgb, g, strength, output))
		return error_fail(e, "nonfinite color preservation result");
	return ERROR_NONE;
}
