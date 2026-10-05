/** @file
 *
 * The settings' table and functions.
 */
// SPDX-License-Identifier: MIT
#include <linux/input-event-codes.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#include "control_settings.h"
#include "shm_protocol.h"

// The tier setting's values are minimum + k * step, from kNativeTierMin by kNativeTierStep, and its
// labels "720|900|1080" are the heights that these constants give kNativeTiers. A static assertion
// cannot read a string, so this pins the constants, and control-settings-test compares the labels.
static_assert(kNativeTierMin == 720 && kNativeTierStep == 180 && kNativeTierCount == 3,
              "the tier setting's labels \"720|900|1080\" must spell the heights of kNativeTiers");

// The rows of CONTROL_SETTINGS, each X(name, word, minimum, maximum, help, short_name, is_float,
// section, tuning, choices, step).
#define CONTROL_SETTINGS_ROWS(X) \
	X("color-preserve", colorPreserveBits, 0, 1, \
	  "Per-pass original-frame chroma anchoring (0 disables)", \
	  'L', true, NEURAL, false, nullptr, 1) \
	X("enabled", enabled, 0, 1, \
	  "Enable neural rendering (0/1)", \
	  'e', false, NEURAL, false, nullptr, 1) \
	X("hdr-mode", hdrMode, 0, 2, \
	  "Proxy precision: 0 auto (16-bit for HDR), 1 force 8-bit, 2 force 16-bit", \
	  'E', false, IMAGE, false, "Automatic|8-bit proxy|16-bit proxy", 1) \
	X("sdr16-multipass", sdr16Multipass, 0, 1, \
	  "HIP: SDR between-pass precision, 0 quantized 8-bit, 1 binary16; HDR stays binary16. " \
	  "Vulkan passes 32-bit float", \
	  'B', false, IMAGE, false, nullptr, 1) \
	X("preset", preset, 0, 0, \
	  "Captured network preset; only 0 is supported", \
	  'N', false, MODEL, false, nullptr, 1) \
	X("style", style, 0, 2, \
	  "Vulkan: the model's style conditioning; HIP ignores it", \
	  'y', false, NEURAL, false, nullptr, 1) \
	X("auto-mask", autoMask, 0, 1, \
	  "Vulkan: the model's automatic skin mask, which gives skin its own structure (0/1); " \
	  "HIP ignores it", \
	  'M', false, NEURAL, false, nullptr, 1) \
	X("intensity", intensityBits, 0, 4, \
	  "HIP: per-pass residual intensity; Vulkan: the model's own intensity, used up to 2", \
	  'i', true, NEURAL, true, nullptr, 1) \
	X("local-tone", localToneBits, 0, 4, \
	  "HIP: per-pass low-frequency residual strength; Vulkan: the model's local tone " \
	  "control, used up to 2", \
	  'o', true, NEURAL, true, nullptr, 1) \
	X("local-structure", localStructureBits, 0, 4, \
	  "HIP: per-pass high-frequency residual strength; Vulkan: the model's local structure " \
	  "control, used up to 2", \
	  'j', true, NEURAL, true, nullptr, 1) \
	X("skin-structure", skinStructureBits, -1, 2, \
	  "Vulkan: local structure on skin under the automatic mask; -1 follows local " \
	  "structure; HIP ignores it", \
	  'K', true, NEURAL, false, nullptr, 1) \
	X("sharpness", sharpnessBits, 0, 1, \
	  "Per-pass sharpening strength", \
	  'n', true, NEURAL, true, nullptr, 1) \
	X("mvec", mvecEnabled, 0, 1, \
	  "Estimate motion and reproject previous neural output (0/1)", \
	  'V', false, MOTION, false, nullptr, 1) \
	X("mvec-quality", mvecQuality, 0, 2, \
	  "Motion search quality: 0 fast, 1 balanced, 2 quality", \
	  'Q', false, MOTION, false, "Fast|Balanced|Quality", 1) \
	X("mvec-pixels", mvecPixelSize, 0, 3, \
	  "Motion grid spacing: 0=1px, 1=2px, 2=4px, 3=8px", \
	  'F', false, MOTION, false, "1 px|2 px|4 px|8 px", 1) \
	X("white-point", whitePointBits, 0.0001, 2000, \
	  "Manual paper white for linear-light input", \
	  'W', true, IMAGE, false, nullptr, 1) \
	X("white-point-scale", whitePointScaleBits, 0.01, 100, \
	  "Multiplier on manual or measured white point", \
	  'G', true, IMAGE, false, nullptr, 1) \
	X("white-point-source", whitePointSource, 0, 1, \
	  "0 manual paper white, 1 measured from frame", \
	  'O', false, IMAGE, false, "Manual|Measured", 1) \
	X("white-point-trim", whitePointTrimBits, 0.01, 100, \
	  "Multiplier on measured white point only", \
	  'I', true, IMAGE, false, nullptr, 1) \
	X("color-mode", colourMode, 0, 2, \
	  "0 auto, 1 display-referred, 2 linear HDR", \
	  'Y', false, IMAGE, false, "Automatic|Display-referred|Linear HDR", 1) \
	X("reversible", reversibleMode, 0, 4, \
	  "Proxy: 0 knee, 1 Neutwo, 2 Neutwo replace, 3 hybrid, 4 hybrid replace", \
	  'Z', false, IMAGE, false, "Knee|Neutwo|Neutwo replace|Hybrid|Hybrid replace", 1) \
	X("passes", passes, 1, kMaxPasses, \
	  "Successive neural evaluations per frame; each consumes the previous result; Vulkan " \
	  "runs at most 16 and stores that count", \
	  'P', false, NEURAL, false, nullptr, 1) \
	X("tier", nativeTier, kNativeTierMin, kNativeTierMax, \
	  "Neural raster height; a change rebuilds the network between frames, during which " \
	  "the game presents its own frames; game resolution unchanged", \
	  'T', false, NEURAL, false, "720|900|1080", kNativeTierStep) \
	X("detail", transferStrengthBits, 0, 4, \
	  "Strength of the neural edit", \
	  'd', true, COMPOSITION, false, nullptr, 1) \
	X("color", colourStrengthBits, 0, 4, \
	  "Color contribution of the edit", \
	  'C', true, COMPOSITION, false, nullptr, 1) \
	X("guard", maxRatioBits, 1, 30, \
	  "Maximum per-pixel gain or reciprocal gain", \
	  'g', true, COMPOSITION, false, nullptr, 1) \
	X("transfer", transfer, 0, 2, \
	  "0 classic, 1 matched residual, 2 native frame + edit", \
	  't', false, COMPOSITION, false, \
	  "Classic ratio|Matched residual|Native frame + edit", 1) \
	X("bypass", compositionBypass, 0, 1, \
	  "0 compose the edit, 1 present the raw model result; " \
	  "default 1 with dlsslopd --test-identity", \
	  'b', false, COMPOSITION, false, nullptr, 1) \
	X("ratio-smooth", ratioSmoothPercent, 0, 100, \
	  "Neighbourhood contribution to relighting ratio (%)", \
	  'a', false, COMPOSITION, false, nullptr, 1) \
	X("color-trust", colourTrustPercent, 0, 800, \
	  "Allowed color displacement, in hundredths", \
	  'u', false, COMPOSITION, false, nullptr, 1) \
	X("debug-view", debugView, 0, 5, \
	  "0 off, 1 proxy, 2 model, 3 edit, 4/5 color-bound views", \
	  'v', false, INSPECT, false, \
	  "Off|Input proxy|Model output|Edit|Color bound 1|Color bound 2", 1) \
	X("debug-scale", debugScaleBits, 0.01, 100, \
	  "Debug-view intensity multiplier", \
	  'D', true, INSPECT, false, nullptr, 1) \
	X("working-scale", workingScaleBits, 0.25, 1, \
	  "Proxy scale relative to game resolution; a neural dlsslopd also caps " \
	  "it at its tier raster", \
	  'w', true, IMAGE, false, nullptr, 1) \
	X("compare", compareMode, 0, 2, \
	  "0 off, 1 side by side, 2 wipe", \
	  'p', false, INSPECT, false, "Off|Side by side|Wipe", 1) \
	X("compare-split", compareSplitBits, 0, 1, \
	  "Comparison split position", \
	  'x', true, INSPECT, false, nullptr, 1) \
	X("compare-zoom", compareZoomBits, 1, 2, \
	  "Side-by-side comparison magnification", \
	  'z', true, INSPECT, false, nullptr, 1) \
	X("compare-swap", compareSwap, 0, 1, \
	  "Swap comparison sides (0/1)", \
	  'X', false, INSPECT, false, nullptr, 1) \
	X("apply-model", applyModel, 0, 1, \
	  "Apply the edit; 0 shows the clean frame but retains inference cost", \
	  'm', false, NEURAL, false, nullptr, 1) \
	X("hold", holdFrame, 0, 1, \
	  "Freeze the input frame (0/1)", \
	  'H', false, NEURAL, false, nullptr, 1) \
	X("toggle-key", toggleKey, 0, KEY_MAX, \
	  "Linux KEY_ code for the layer hotkey; 0 disables it", \
	  'k', false, INSPECT, false, nullptr, 1)

// A row of CONTROL_SETTINGS from its values in reading order; the struct orders them by width.
#define CONTROL_SETTING(name, word, minimum, maximum, help, short_name, is_float, section, tuning, choices, \
                        step) \
	{(minimum), (maximum), (step), (name), (help), (choices), offsetof(struct ShmHeader, word), \
	 (short_name), (is_float), CONTROL_SECTION_##section, (tuning)},

struct control_setting const CONTROL_SETTINGS[] = {
	CONTROL_SETTINGS_ROWS(CONTROL_SETTING)
};

#undef CONTROL_SETTING

// The header declares CONTROL_SETTINGS with CONTROL_SETTING_COUNT rows, and C pads a shorter table
// with zeroed rows without a diagnostic.
#define CONTROL_SETTING_ROW(...) + 1
static_assert(0 CONTROL_SETTINGS_ROWS(CONTROL_SETTING_ROW) == CONTROL_SETTING_COUNT,
              "CONTROL_SETTING_COUNT is not the number of rows of CONTROL_SETTINGS");
#undef CONTROL_SETTING_ROW
#undef CONTROL_SETTINGS_ROWS

/** @brief A setting's word in a header.
 *
 * @param h The header, or nullptr.
 * @param s The setting, or nullptr.
 * @return  The word, or nullptr if @a h or @a s is nullptr.
 */
static _Atomic(uint32_t) *
control_setting_word (struct ShmHeader             *h,
                      struct control_setting const *s)
{
	if (!h || !s)
		return nullptr;
	return (_Atomic(uint32_t) *)(void *)((unsigned char *)h + s->offset);
}

uint32_t
control_setting_load (struct ShmHeader const       *h,
                      struct control_setting const *s)
{
	if (!h || !s)
		return 0;
	unsigned char const *const word = (unsigned char const *)h + s->offset;
	return atomic_load((_Atomic(uint32_t) const *)(void const *)word);
}

void
control_setting_store (struct ShmHeader             *h,
                       struct control_setting const *s,
                       uint32_t                      bits)
{
	_Atomic(uint32_t) *const word = control_setting_word(h, s);
	if (word)
		atomic_store(word, bits);
}

void
control_setting_toggle (struct ShmHeader             *h,
                        struct control_setting const *s)
{
	_Atomic(uint32_t) *const word = control_setting_word(h, s);
	if (!word)
		return;
	uint32_t value = atomic_load(word);
	while (!atomic_compare_exchange_weak(word, &value, !value)) {}
}

double
control_setting_value (struct control_setting const *s,
                       uint32_t                      raw)
{
	return s && s->is_float ? (double)BitsToFloat(raw) : (double)raw;
}

void
control_settings_defaults (uint32_t dest[CONTROL_SETTING_COUNT],
                           bool     bypass)
{
	// A header of this call's own, initialized once.
	struct ShmHeader defaults;
	ShmInitNativeDefaults(&defaults, bypass);
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i)
		dest[i] = control_setting_load(&defaults, &CONTROL_SETTINGS[i]);
}

bool
control_setting_in_range (struct control_setting const *s,
                          double                        v)
{
	if (!s || !(v >= (double)(float)s->minimum && v <= (double)(float)s->maximum))
		return false;
	if (s->is_float)
		return true;
	double const steps = (v - s->minimum) / s->step;
	return steps == trunc(steps);
}

bool
control_setting_fixed (struct control_setting const *s)
{
	return s && s->section == CONTROL_SECTION_MODEL;
}

bool
control_settings_worker_bypass (struct ShmHeader const *h)
{
	return h && atomic_load(&h->heartbeat) && !atomic_load(&h->nativeModelMaxWidth) &&
	       !atomic_load(&h->nativeModelMaxHeight);
}

