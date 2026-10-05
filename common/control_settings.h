/** @file
 *
 * The settings that dlsslopctl and the GUI edit and that the tests' fake layer stores: for each one
 * its options, range, help and the offset of its word in struct ShmHeader. Their defaults are what
 * ShmInitNativeDefaults() writes there, read through the same offsets by
 * control_settings_defaults(), so that the channel's initialization, a reset and --help share one
 * definition of each default. control_settings.c defines the table and the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_CONTROL_SETTINGS_H_
#define DLSSLOP_AMD_COMMON_CONTROL_SETTINGS_H_

#ifdef __cplusplus
# include <cstdint>
#else
# include <stdint.h>
#endif

#include "shm_protocol.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief The controller's pages, in their order. */
enum control_section : STD(uint8_t) {
	CONTROL_SECTION_NEURAL,      //!< Neural passes.
	CONTROL_SECTION_COMPOSITION, //!< Composition.
	CONTROL_SECTION_IMAGE,       //!< Image and HDR.
	CONTROL_SECTION_MOTION,      //!< Motion.
	CONTROL_SECTION_INSPECT,     //!< Compare and inspect.
	CONTROL_SECTION_MODEL,       //!< Model configuration: settings that admit only their default.
};

/** @brief A setting: one word of struct ShmHeader. */
struct control_setting {
	double               minimum;    //!< The least value.
	double               maximum;    //!< The greatest value.
	double               step;       //!< An integer setting's values are minimum + k * step.
	char const          *name;       //!< The long option.
	char const          *help;       //!< What the setting does.
	char const          *choices;    //!< The values' labels in order, '|'-separated; or nullptr.
	STD(uint32_t)        offset;     //!< The offset of the setting's word in struct ShmHeader.
	char                 short_name; //!< The short option.
	bool                 is_float;   //!< Whether the word holds a float's bits rather than an integer.
	enum control_section section;    //!< The controller's page.
	bool                 tuning;     //!< A native tuning value: changing it bumps tuningSeq.
};

/** @brief The number of CONTROL_SETTINGS, the rows of control_settings.c's table, which checks it. */
enum : STD(uint32_t) {
	CONTROL_SETTING_COUNT = 40,
};

/** @brief Every setting, in the order of their options and of the controller's cards.
 *
 * All controls from the original interface. Neither network has another preset, so the model
 * section's setting has a one-value range.
 */
extern struct control_setting const CONTROL_SETTINGS[CONTROL_SETTING_COUNT];

/** @brief What a setting's word in a header holds.
 *
 * @param h The header, or nullptr.
 * @param s The setting, or nullptr.
 * @return  The word's bits, or 0 if @a h or @a s is nullptr.
 */
extern STD(uint32_t)
control_setting_load (struct ShmHeader const       *h,
                      struct control_setting const *s);

/** @brief Stores bits in a setting's word in a header.
 *
 * @param h    The header, or nullptr to store nothing.
 * @param s    The setting, or nullptr to store nothing.
 * @param bits The bits.
 */
extern void
control_setting_store (struct ShmHeader             *h,
                       struct control_setting const *s,
                       STD(uint32_t)                 bits);

/** @brief Flips a setting's word in a header in one atomic step: 0 becomes 1, and any other value becomes 0.
 *
 * @param h The header, or nullptr to flip nothing.
 * @param s The setting, or nullptr to flip nothing.
 */
extern void
control_setting_toggle (struct ShmHeader             *h,
                        struct control_setting const *s);

/** @brief The value that a setting's bits hold.
 *
 * @param s   The setting, or nullptr for an integer.
 * @param raw The bits.
 * @return    The float of @a raw for a float setting, otherwise @a raw.
 */
extern double
control_setting_value (struct control_setting const *s,
                       STD(uint32_t)                 raw);

/** @brief The settings' defaults: the words that ShmInitNativeDefaults() gives them, which a new
 *         channel holds and a reset stores.
 *
 * C++ takes them here rather than from a header of its own: a header is initialized in C alone.
 *
 * @param dest   Receives each setting's default, in the order of CONTROL_SETTINGS.
 * @param bypass ShmInitNativeDefaults()'s bypass, as control_settings_worker_bypass() gives it for a
 *               channel.
 */
extern void
control_settings_defaults (STD(uint32_t) dest[CONTROL_SETTING_COUNT],
                           bool          bypass);

#undef STD

/** @brief Whether a setting admits a value.
 *
 * Readers see float settings as binary32, which rounds some minimums below themselves, so the bounds
 * compare as stored; integer bounds are exact there. An integer setting also takes only its steps.
 * NaN fails.
 *
 * @param s The setting, or nullptr.
 * @param v The value.
 * @return  true if @a s admits @a v; false for nullptr.
 */
extern bool
control_setting_in_range (struct control_setting const *s,
                          double                        v);

/** @brief Whether a setting admits only its default, as a model setting does.
 *
 * @param s The setting, or nullptr.
 * @return  true if it does; false for nullptr.
 */
extern bool
control_setting_fixed (struct control_setting const *s);

/** @brief The bypass default of a channel's worker.
 *
 * A started worker that publishes no neural raster (--test-identity) returns final
 * images. A never-started channel keeps the native-composition default.
 *
 * @param h The channel's header, or nullptr.
 * @return  ShmInitNativeDefaults()'s bypass for the worker; false for nullptr.
 */
extern bool
control_settings_worker_bypass (struct ShmHeader const *h);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_CONTROL_SETTINGS_H_ */
