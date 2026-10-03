/** @file
 *
 * The tier setting, in C: its values minimum + k * step are the heights of kNativeTiers, and its
 * labels spell them in order. control_settings.c pins the constants that give the heights, but a
 * static assertion cannot read the labels.
 */
// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "control_settings.h"
#include "shm_protocol.h"

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param message   What failed.
 */
static void
require (bool        condition,
         char const *message)
{
	if (condition)
		return;
	fprintf(stderr, "control-settings-test: %s\n", message);
	exit(1);
}

int
main (void)
{
	struct control_setting const *tier = nullptr;
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT && !tier; ++i)
		if (!strcmp(CONTROL_SETTINGS[i].name, "tier"))
			tier = &CONTROL_SETTINGS[i];
	require(tier && tier->choices, "no tier setting with labels");
	require(tier->maximum == kNativeTiers[kNativeTierCount - 1].height,
	        "the tier setting's maximum is not the largest tier's height");

	char const *label = tier->choices;
	for (uint32_t i = 0; i < kNativeTierCount; ++i) {
		uint32_t const height = kNativeTiers[i].height;
		require(height == tier->minimum + i * tier->step,
		        "a tier's height is not the tier setting's value of its index");
		uint32_t spelt = 0;
		char const *const start = label;
		for (; *label >= '0' && *label <= '9'; ++label)
			spelt = spelt * 10 + (uint32_t)(*label - '0');
		require(label != start && spelt == height, "a tier's label does not spell its height");
		require(*label == (i + 1 < kNativeTierCount ? '|' : '\0'),
		        "the tier setting's labels are not one for each tier, '|'-separated");
		++label;
	}

	if (puts("control settings: the tier labels spell kNativeTiers") == EOF)
		return 1;
	return 0;
}
