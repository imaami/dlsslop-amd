/** @file
 *
 * The settings that dlsslopctl and the GUI edit and that the tests' fake layer stores: for each one
 * its options, range, help and the offset of its word in struct ShmHeader. Their defaults are what
 * ShmInitNativeDefaults() writes there, read through the same offsets, so that the channel's
 * initialization, a reset and --help share one definition of each default.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_CONTROL_SETTINGS_H_
#define DLSSLOP_AMD_COMMON_CONTROL_SETTINGS_H_

#ifdef __cplusplus
# include <cmath>
# include <cstddef>
# include <cstdint>
# include <stdatomic.h>
# define CONTROL_STD(x) std::x
// Inline, so that every translation unit shares one table and one definition of each function:
// inline C++ code in other headers, such as gui/channel.hpp's, uses them, and the one-definition rule
// asks for that.
# define CONTROL_CONST inline constexpr
# define CONTROL_INLINE inline
#else
# include <math.h>
# include <stdatomic.h>
# include <stddef.h>
# include <stdint.h>
# define CONTROL_STD(x) x
// Static: C has no inline objects, a constexpr object holds no pointer but a null one, and an inline
// function with external linkage would need an external definition in one translation unit.
# define CONTROL_CONST static const
# define CONTROL_INLINE static inline
#endif

#include <linux/input-event-codes.h>

#include "shm_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief The controller's pages, in their order. */
enum control_section : CONTROL_STD(uint8_t) {
	CONTROL_SECTION_NEURAL,      //!< Neural passes.
	CONTROL_SECTION_COMPOSITION, //!< Composition.
	CONTROL_SECTION_IMAGE,       //!< Image and HDR.
	CONTROL_SECTION_MOTION,      //!< Motion.
	CONTROL_SECTION_INSPECT,     //!< Compare and inspect.
	CONTROL_SECTION_MODEL,       //!< Model configuration: settings that admit only their default.
};

/** @brief A setting: one word of struct ShmHeader. */
struct control_setting {
	double                minimum;    //!< The least value.
	double                maximum;    //!< The greatest value.
	double                step;       //!< An integer setting's values are minimum + k * step.
	char const           *name;       //!< The long option.
	char const           *help;       //!< What the setting does.
	char const           *choices;    //!< The values' labels in order, '|'-separated; or nullptr.
	CONTROL_STD(uint32_t) offset;     //!< The offset of the setting's word in struct ShmHeader.
	char                  short_name; //!< The short option.
	bool                  is_float;   //!< Whether the word holds a float's bits rather than an integer.
	enum control_section  section;    //!< The controller's page.
	bool                  tuning;     //!< A native tuning value: changing it bumps tuningSeq.
};

/** @brief The tier setting's labels, one per height of kNativeTiers; C++ checks them below. */
CONTROL_CONST char CONTROL_SETTINGS_TIER_LABELS[] = "720|900|1080";

// A row of CONTROL_SETTINGS from its values in reading order; the struct orders them by width.
#define CONTROL_SETTING(name, word, minimum, maximum, help, short_name, is_float, section, tuning, choices, \
                        step) \
	{(minimum), (maximum), (step), (name), (help), (choices), offsetof(struct ShmHeader, word), \
	 (short_name), (is_float), CONTROL_SECTION_##section, (tuning)}

/** @brief Every setting, in the order of their options and of the controller's cards.
 *
 * All controls from the original interface. Neither network has another preset, so the model
 * section's setting has a one-value range.
 */
CONTROL_CONST struct control_setting CONTROL_SETTINGS[] = {
	CONTROL_SETTING("color-preserve", colorPreserveBits, 0, 1,
	                "Per-pass original-frame chroma anchoring (0 disables)",
	                'L', true, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("enabled", enabled, 0, 1,
	                "Enable neural rendering (0/1)",
	                'e', false, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("hdr-mode", hdrMode, 0, 2,
	                "Proxy precision: 0 auto (16-bit for HDR), 1 force 8-bit, 2 force 16-bit",
	                'E', false, IMAGE, false, "Automatic|8-bit proxy|16-bit proxy", 1),
	CONTROL_SETTING("sdr16-multipass", sdr16Multipass, 0, 1,
	                "HIP: SDR between-pass precision, 0 quantized 8-bit, 1 binary16; HDR stays binary16. "
	                "Vulkan passes 32-bit float",
	                'B', false, IMAGE, false, nullptr, 1),
	CONTROL_SETTING("preset", preset, 0, 0,
	                "Captured network preset; only 0 is supported",
	                'N', false, MODEL, false, nullptr, 1),
	CONTROL_SETTING("style", style, 0, 2,
	                "Vulkan: the model's style conditioning; HIP ignores it",
	                'y', false, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("auto-mask", autoMask, 0, 1,
	                "Vulkan: the model's automatic skin mask, which gives skin its own structure (0/1); "
	                "HIP ignores it",
	                'M', false, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("intensity", intensityBits, 0, 4,
	                "HIP: per-pass residual intensity; Vulkan: the model's own intensity, used up to 2",
	                'i', true, NEURAL, true, nullptr, 1),
	CONTROL_SETTING("local-tone", localToneBits, 0, 4,
	                "HIP: per-pass low-frequency residual strength; Vulkan: the model's local tone "
	                "control, used up to 2",
	                'o', true, NEURAL, true, nullptr, 1),
	CONTROL_SETTING("local-structure", localStructureBits, 0, 4,
	                "HIP: per-pass high-frequency residual strength; Vulkan: the model's local structure "
	                "control, used up to 2",
	                'j', true, NEURAL, true, nullptr, 1),
	CONTROL_SETTING("skin-structure", skinStructureBits, -1, 2,
	                "Vulkan: local structure on skin under the automatic mask; -1 follows local "
	                "structure; HIP ignores it",
	                'K', true, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("sharpness", sharpnessBits, 0, 1,
	                "Per-pass sharpening strength",
	                'n', true, NEURAL, true, nullptr, 1),
	CONTROL_SETTING("mvec", mvecEnabled, 0, 1,
	                "Estimate motion and reproject previous neural output (0/1)",
	                'V', false, MOTION, false, nullptr, 1),
	CONTROL_SETTING("mvec-quality", mvecQuality, 0, 2,
	                "Motion search quality: 0 fast, 1 balanced, 2 quality",
	                'Q', false, MOTION, false, "Fast|Balanced|Quality", 1),
	CONTROL_SETTING("mvec-pixels", mvecPixelSize, 0, 3,
	                "Motion grid spacing: 0=1px, 1=2px, 2=4px, 3=8px",
	                'F', false, MOTION, false, "1 px|2 px|4 px|8 px", 1),
	CONTROL_SETTING("white-point", whitePointBits, 0.0001, 2000,
	                "Manual paper white for linear-light input",
	                'W', true, IMAGE, false, nullptr, 1),
	CONTROL_SETTING("white-point-scale", whitePointScaleBits, 0.01, 100,
	                "Multiplier on manual or measured white point",
	                'G', true, IMAGE, false, nullptr, 1),
	CONTROL_SETTING("white-point-source", whitePointSource, 0, 1,
	                "0 manual paper white, 1 measured from frame",
	                'O', false, IMAGE, false, "Manual|Measured", 1),
	CONTROL_SETTING("white-point-trim", whitePointTrimBits, 0.01, 100,
	                "Multiplier on measured white point only",
	                'I', true, IMAGE, false, nullptr, 1),
	CONTROL_SETTING("color-mode", colourMode, 0, 2,
	                "0 auto, 1 display-referred, 2 linear HDR",
	                'Y', false, IMAGE, false, "Automatic|Display-referred|Linear HDR", 1),
	CONTROL_SETTING("reversible", reversibleMode, 0, 4,
	                "Proxy: 0 knee, 1 Neutwo, 2 Neutwo replace, 3 hybrid, 4 hybrid replace",
	                'Z', false, IMAGE, false, "Knee|Neutwo|Neutwo replace|Hybrid|Hybrid replace", 1),
	CONTROL_SETTING("passes", passes, 1, kMaxPasses,
	                "Successive neural evaluations per frame; each consumes the previous result; Vulkan "
	                "runs at most 16 and stores that count",
	                'P', false, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("tier", nativeTier, kNativeTierMin, kNativeTierMax,
	                "Neural raster height; a change rebuilds the network between frames, during which "
	                "the game presents its own frames; game resolution unchanged",
	                'T', false, NEURAL, false, CONTROL_SETTINGS_TIER_LABELS, kNativeTierStep),
	CONTROL_SETTING("detail", transferStrengthBits, 0, 4,
	                "Strength of the neural edit",
	                'd', true, COMPOSITION, false, nullptr, 1),
	CONTROL_SETTING("color", colourStrengthBits, 0, 4,
	                "Color contribution of the edit",
	                'C', true, COMPOSITION, false, nullptr, 1),
	CONTROL_SETTING("guard", maxRatioBits, 1, 30,
	                "Maximum per-pixel gain or reciprocal gain",
	                'g', true, COMPOSITION, false, nullptr, 1),
	CONTROL_SETTING("transfer", transfer, 0, 2,
	                "0 classic, 1 matched residual, 2 native frame + edit",
	                't', false, COMPOSITION, false,
	                "Classic ratio|Matched residual|Native frame + edit", 1),
	CONTROL_SETTING("bypass", compositionBypass, 0, 1,
	                "0 compose the edit, 1 present the raw model result; default 1 with dlsslopd "
	                "--cpu-compose or --test-identity",
	                'b', false, COMPOSITION, false, nullptr, 1),
	CONTROL_SETTING("ratio-smooth", ratioSmoothPercent, 0, 100,
	                "Neighbourhood contribution to relighting ratio (%)",
	                'a', false, COMPOSITION, false, nullptr, 1),
	CONTROL_SETTING("color-trust", colourTrustPercent, 0, 800,
	                "Allowed color displacement, in hundredths",
	                'u', false, COMPOSITION, false, nullptr, 1),
	CONTROL_SETTING("debug-view", debugView, 0, 5,
	                "0 off, 1 proxy, 2 model, 3 edit, 4/5 color-bound views",
	                'v', false, INSPECT, false,
	                "Off|Input proxy|Model output|Edit|Color bound 1|Color bound 2", 1),
	CONTROL_SETTING("debug-scale", debugScaleBits, 0.01, 100,
	                "Debug-view intensity multiplier",
	                'D', true, INSPECT, false, nullptr, 1),
	CONTROL_SETTING("working-scale", workingScaleBits, 0.25, 2,
	                "Proxy scale relative to game resolution; dlsslopd caps it at 1 and at its tier "
	                "raster unless run with --cpu-compose or --test-identity",
	                'w', true, IMAGE, false, nullptr, 1),
	CONTROL_SETTING("downscaler", scalingDownscaler, 1, 7,
	                "Supersampling down-leg filter, used only above working-scale 1 with dlsslopd "
	                "--cpu-compose or --test-identity: 1 bicubic, 2 catmull, 3 lanczos2, 4 lanczos3, "
	                "5 kaiser2, 6 kaiser3, 7 magic",
	                'f', false, IMAGE, false,
	                "Bicubic|Catmull|Lanczos 2|Lanczos 3|Kaiser 2|Kaiser 3|Magic", 1),
	CONTROL_SETTING("compare", compareMode, 0, 2,
	                "0 off, 1 side by side, 2 wipe",
	                'p', false, INSPECT, false, "Off|Side by side|Wipe", 1),
	CONTROL_SETTING("compare-split", compareSplitBits, 0, 1,
	                "Comparison split position",
	                'x', true, INSPECT, false, nullptr, 1),
	CONTROL_SETTING("compare-zoom", compareZoomBits, 1, 2,
	                "Side-by-side comparison magnification",
	                'z', true, INSPECT, false, nullptr, 1),
	CONTROL_SETTING("compare-swap", compareSwap, 0, 1,
	                "Swap comparison sides (0/1)",
	                'X', false, INSPECT, false, nullptr, 1),
	CONTROL_SETTING("apply-model", applyModel, 0, 1,
	                "Apply the edit; 0 shows the clean frame but retains inference cost",
	                'm', false, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("hold", holdFrame, 0, 1,
	                "Freeze the input frame (0/1)",
	                'H', false, NEURAL, false, nullptr, 1),
	CONTROL_SETTING("toggle-key", toggleKey, 0, KEY_MAX,
	                "Linux KEY_ code for the layer hotkey; 0 disables it",
	                'k', false, INSPECT, false, nullptr, 1),
};

#undef CONTROL_SETTING

/** @brief The number of CONTROL_SETTINGS. */
static constexpr CONTROL_STD(uint32_t) CONTROL_SETTING_COUNT =
	sizeof CONTROL_SETTINGS / sizeof *CONTROL_SETTINGS;

/** @brief A setting's word in a header.
 *
 * @param h The header, or nullptr.
 * @param s The setting, or nullptr.
 * @return  The word, or nullptr if @a h or @a s is nullptr.
 */
CONTROL_INLINE _Atomic(CONTROL_STD(uint32_t)) *
control_setting_word (struct ShmHeader             *h,
                      struct control_setting const *s)
{
	if (!h || !s)
		return nullptr;
	return (_Atomic(CONTROL_STD(uint32_t)) *)(void *)((unsigned char *)h + s->offset);
}

/** @brief What a setting's word in a header holds.
 *
 * @param h The header, or nullptr.
 * @param s The setting, or nullptr.
 * @return  The word's bits, or 0 if @a h or @a s is nullptr.
 */
CONTROL_INLINE CONTROL_STD(uint32_t)
control_setting_load (struct ShmHeader const       *h,
                      struct control_setting const *s)
{
	if (!h || !s)
		return 0;
	unsigned char const *const word = (unsigned char const *)h + s->offset;
	return atomic_load((_Atomic(CONTROL_STD(uint32_t)) const *)(void const *)word);
}

/** @brief The value that a setting's bits hold.
 *
 * @param s   The setting, or nullptr for an integer.
 * @param raw The bits.
 * @return    The float of @a raw for a float setting, otherwise @a raw.
 */
CONTROL_INLINE double
control_setting_value (struct control_setting const *s,
                       CONTROL_STD(uint32_t)         raw)
{
	return s && s->is_float ? (double)BitsToFloat(raw) : (double)raw;
}

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
CONTROL_INLINE bool
control_setting_in_range (struct control_setting const *s,
                          double                        v)
{
	if (!s || !(v >= (double)(float)s->minimum && v <= (double)(float)s->maximum))
		return false;
	if (s->is_float)
		return true;
	double const steps = (v - s->minimum) / s->step;
	return steps == CONTROL_STD(trunc)(steps);
}

/** @brief Whether a setting admits only its default, as a model setting does.
 *
 * @param s The setting, or nullptr.
 * @return  true if it does; false for nullptr.
 */
CONTROL_INLINE bool
control_setting_fixed (struct control_setting const *s)
{
	return s && s->section == CONTROL_SECTION_MODEL;
}

/** @brief The bypass default of a channel's worker.
 *
 * A started worker that publishes no neural raster (--cpu-compose, --test-identity) returns final
 * images. A never-started channel keeps the native-composition default.
 *
 * @param h The channel's header, or nullptr.
 * @return  ShmInitNativeDefaults()'s bypass for the worker; false for nullptr.
 */
CONTROL_INLINE bool
control_settings_worker_bypass (struct ShmHeader const *h)
{
	return h && atomic_load(&h->heartbeat) && !atomic_load(&h->nativeModelMaxWidth) &&
	       !atomic_load(&h->nativeModelMaxHeight);
}

#ifdef __cplusplus
} /* extern "C" */

/** @brief Whether the tier setting's labels spell the heights of kNativeTiers in order: the
 *         kNativeTierCount heights from kNativeTierMin by kNativeTierStep, which shm_protocol.c gives
 *         the tiers.
 *
 * @param labels The tier setting's labels.
 * @return       true if they do.
 */
constexpr bool
control_settings_tiers_match (char const *labels)
{
	for (std::uint32_t i = 0; i < kNativeTierCount; ++i) {
		std::uint32_t const height = kNativeTierMin + i * kNativeTierStep;
		std::uint32_t label = 0;
		while (*labels >= '0' && *labels <= '9')
			label = label * 10 + static_cast<std::uint32_t>(*labels++ - '0');
		if (label != height || *labels++ != (i + 1 < kNativeTierCount ? '|' : '\0'))
			return false;
	}
	return true;
}

static_assert(control_settings_tiers_match(CONTROL_SETTINGS_TIER_LABELS),
              "the tier setting's step and labels must follow kNativeTiers");
#endif

#undef CONTROL_INLINE
#undef CONTROL_CONST
#undef CONTROL_STD

#endif /* DLSSLOP_AMD_COMMON_CONTROL_SETTINGS_H_ */
