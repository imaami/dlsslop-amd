/** @file
 *
 * The check of a native tuning: tuning.h.
 */
// SPDX-License-Identifier: MIT
#include <math.h>

#include "error.h"
#include "tuning.h"

/** @brief Whether a control's value is finite and within 0..@a high.
 *
 * @param value The value.
 * @param high  The range's upper end.
 * @return      true if it is.
 */
static bool
in_range (float value,
          float high)
{
	return isfinite(value) && value >= 0.0f && value <= high;
}

enum error_code
tuning_validate (struct native_tuning const *tuning,
                 struct error               *e)
{
	if (!in_range(tuning->intensity, 4.0f) || !in_range(tuning->tone, 4.0f) ||
	    !in_range(tuning->structure, 4.0f))
		return error_reject(e, "native intensity/tone/structure must be finite and within 0..4");
	if (!in_range(tuning->sharpness, 1.0f))
		return error_reject(e, "native sharpness must be finite and within 0..1");
	return ERROR_NONE;
}
