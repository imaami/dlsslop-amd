/** @file
 *
 * The layer's pure functions on the host: the settings snapshot a frame composes with, the model
 * raster, the composition's format table, the toggle key's names, and the log's switches, lines and
 * clock. The values are the C++ layer's, written out, so that a port has to reproduce them.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/input-event-codes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "composition.h"
#include "hotkey.h"
#include "log.h"
#include "shm_protocol.h"
#include "support.h"

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("layer-units-test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief A float of the settings, with its initial value. */
struct real_field {
	char const *name;    //!< Its name.
	uint32_t    offset;  //!< Its offset in struct composition_frame_settings.
	float       initial; //!< Its initial value.
};

/** @brief A word of the settings, with its initial value. */
struct word_field {
	char const *name;    //!< Its name.
	uint32_t    offset;  //!< Its offset in struct composition_frame_settings.
	uint32_t    initial; //!< Its initial value.
};

#define FIELD(name, initial) {#name, offsetof(struct composition_frame_settings, name), initial}

/** @brief Every float of the settings. */
static struct real_field const REALS[] = {
	FIELD(transfer_strength, 1.0f),
	FIELD(colour_strength, 1.0f),
	FIELD(max_ratio, 2.0f),
	FIELD(debug_scale, 1.0f),
	FIELD(white_point_manual, 1.0f),
	FIELD(white_point_scale, 1.0f),
	FIELD(white_point_trim, 1.0f),
	FIELD(compare_split, 0.5f),
	FIELD(compare_zoom, 1.0f),
	FIELD(working_scale, 1.0f),
	FIELD(ghost_slack, 0.5f),
	FIELD(edit_blur, 0.04f),
	FIELD(motion_smooth, 1.0f),
	FIELD(colour_trust, 1.0f),
	FIELD(ratio_smooth, 0.0f),
};

/** @brief Every word of the settings. */
static struct word_field const WORDS[] = {
	FIELD(control_seq, 0),
	FIELD(tuning_seq, 0),
	FIELD(passes, 0),
	FIELD(white_point_source, kWhitePointManual),
	FIELD(native_model_max_width, 0),
	FIELD(native_model_max_height, 0),
	FIELD(transfer, 1),
	FIELD(debug_view, 0),
	FIELD(compare_mode, 0),
	FIELD(compare_swap, 0),
	FIELD(reversible_mode, kReversibleKnee),
	FIELD(apply_model, 1),
	FIELD(hold_frame, 0),
	FIELD(composition_bypass, 0),
};

#undef FIELD

static_assert(sizeof REALS / sizeof *REALS + sizeof WORDS / sizeof *WORDS
              == sizeof (struct composition_frame_settings) / 4, "every settings field needs a row");

/** @brief A float of the settings.
 *
 * @param s      The settings.
 * @param offset The float's offset in them.
 * @return       The float.
 */
static float *
real_at (struct composition_frame_settings *s,
         uint32_t                           offset)
{
	return (float *)((unsigned char *)s + offset);
}

/** @brief A word of the settings.
 *
 * @param s      The settings.
 * @param offset The word's offset in them.
 * @return       The word.
 */
static uint32_t *
word_at (struct composition_frame_settings *s,
         uint32_t                           offset)
{
	return (uint32_t *)((unsigned char *)s + offset);
}

/** @brief A word of a channel header.
 *
 * @param h      The header.
 * @param offset The word's offset in it.
 * @return       The word.
 */
static _Atomic(uint32_t) *
header_word (struct ShmHeader *h,
             uint32_t          offset)
{
	return (_Atomic(uint32_t) *)((unsigned char *)h + offset);
}

/** @brief The settings with every field at its initial value. */
static struct composition_frame_settings
initial (void)
{
	struct composition_frame_settings s = {};
	for (uint32_t i = 0; i < sizeof REALS / sizeof *REALS; ++i)
		*real_at(&s, REALS[i].offset) = REALS[i].initial;
	for (uint32_t i = 0; i < sizeof WORDS / sizeof *WORDS; ++i)
		*word_at(&s, WORDS[i].offset) = WORDS[i].initial;
	return s;
}

/** @brief Starts a failure's message: the test's name and a label.
 *
 * @param fmt  A printf format of the label.
 * @param args Its arguments.
 */
[[gnu::format(printf, 1, 0)]]
static void
print_label (char const *fmt,
             va_list     args)
{
	fputs("layer-units-test: ", stderr);
	vfprintf(stderr, fmt, args);
}

/** @brief Ends the test unless settings are those wanted, field by field, and bit for bit for the
 *         reals: -0.0 is not 0.0 here.
 *
 * @param got  The settings.
 * @param want The settings wanted.
 * @param fmt  A printf format of what the settings are.
 * @param ...  Its arguments.
 */
[[gnu::format(printf, 3, 4)]]
static void
expect (struct composition_frame_settings  got,
        struct composition_frame_settings  want,
        char const                        *fmt,
        ...)
{
	for (uint32_t i = 0; i < sizeof REALS / sizeof *REALS; ++i) {
		float const g = *real_at(&got, REALS[i].offset);
		float const w = *real_at(&want, REALS[i].offset);
		if (FloatToBits(g) == FloatToBits(w))
			continue;
		va_list args;
		va_start(args, fmt);
		print_label(fmt, args);
		va_end(args);
		fprintf(stderr, ": %s is %.9g, not %.9g\n", REALS[i].name, (double)g, (double)w);
		exit(1);
	}
	for (uint32_t i = 0; i < sizeof WORDS / sizeof *WORDS; ++i) {
		uint32_t const g = *word_at(&got, WORDS[i].offset);
		uint32_t const w = *word_at(&want, WORDS[i].offset);
		if (g == w)
			continue;
		va_list args;
		va_start(args, fmt);
		print_label(fmt, args);
		va_end(args);
		fprintf(stderr, ": %s is %u, not %u\n", WORDS[i].name, g, w);
		exit(1);
	}
}

/** @brief Makes a header as the daemon creates it.
 *
 * @param h The header, not yet initialized.
 */
static void
native_header (struct ShmHeader *h)
{
	ShmInitNativeDefaults(h, false);
}

/** @brief What the layer reads from a header as the daemon creates it. */
static struct composition_frame_settings
native_defaults (void)
{
	struct composition_frame_settings s = initial();
	s.passes = kNativeDefaultPasses;
	s.transfer = 2;
	s.ratio_smooth = 1.0f;
	s.colour_trust = 2.0f;
	return s;
}

/** @brief An override that composition_frame_settings_read() takes from the environment. */
struct override {
	char const *variable; //!< The variable.
	char const *value;    //!< Its value.
	uint32_t    field;    //!< The offset of the float it sets.
	float       want;     //!< The float's value.
};

#define ROW(variable, value, field, want) \
	{variable, value, offsetof(struct composition_frame_settings, field), want}

/** @brief The five overrides that composition_frame_settings_read() takes from the environment, read
 *         once per process: each row runs in a child of its own, forked before this process reads
 *         any settings.
 */
static struct override const OVERRIDES[] = {
	// Percent; no bound above, a negative number or none keeps the default.
	ROW("DLSSNR_GHOST_SLACK", "25", ghost_slack, 25 / 100.0f),
	ROW("DLSSNR_GHOST_SLACK", "150", ghost_slack, 150 / 100.0f),
	ROW("DLSSNR_GHOST_SLACK", "0", ghost_slack, 0.0f),
	ROW("DLSSNR_GHOST_SLACK", "", ghost_slack, 0.5f),
	ROW("DLSSNR_GHOST_SLACK", "-5", ghost_slack, 0.5f),
	// A prefix of digits counts, and no digits are 0; a number beyond 32 bits is the largest.
	ROW("DLSSNR_GHOST_SLACK", "12abc", ghost_slack, 12 / 100.0f),
	ROW("DLSSNR_GHOST_SLACK", "abc", ghost_slack, 0.0f),
	ROW("DLSSNR_GHOST_SLACK", "99999999999", ghost_slack, (float)2147483647 / 100.0f),
	ROW("DLSSNR_GHOST_SLACK", "99999999999999999999999", ghost_slack, (float)2147483647 / 100.0f),
	// Percent over the header's ratioSmoothPercent, up to 1.
	ROW("DLSSNR_RATIO_SMOOTH", "50", ratio_smooth, 50 / 100.0f),
	ROW("DLSSNR_RATIO_SMOOTH", "150", ratio_smooth, 1.0f),
	ROW("DLSSNR_RATIO_SMOOTH", "0", ratio_smooth, 0.0f),
	ROW("DLSSNR_RATIO_SMOOTH", "", ratio_smooth, 1.0f),
	ROW("DLSSNR_RATIO_SMOOTH", "-5", ratio_smooth, 1.0f),
	// Percent over the header's colourTrustPercent, up to 8.
	ROW("DLSSNR_COLOUR_TRUST", "50", colour_trust, 50 / 100.0f),
	ROW("DLSSNR_COLOUR_TRUST", "900", colour_trust, 8.0f),
	ROW("DLSSNR_COLOUR_TRUST", "", colour_trust, 2.0f),
	ROW("DLSSNR_COLOUR_TRUST", "-1", colour_trust, 2.0f),
	// Percent, up to 1; the header has no field for it.
	ROW("DLSSNR_MOTION_SMOOTH", "40", motion_smooth, 40 / 100.0f),
	ROW("DLSSNR_MOTION_SMOOTH", "150", motion_smooth, 1.0f),
	ROW("DLSSNR_MOTION_SMOOTH", "0", motion_smooth, 0.0f),
	ROW("DLSSNR_MOTION_SMOOTH", "", motion_smooth, 1.0f),
	ROW("DLSSNR_MOTION_SMOOTH", "-1", motion_smooth, 1.0f),
	// Per mille, up to 0.25.
	ROW("DLSSNR_EDIT_BLUR", "100", edit_blur, 100 / 1000.0f),
	ROW("DLSSNR_EDIT_BLUR", "300", edit_blur, 0.25f),
	ROW("DLSSNR_EDIT_BLUR", "0", edit_blur, 0.0f),
	ROW("DLSSNR_EDIT_BLUR", "", edit_blur, 0.04f),
	ROW("DLSSNR_EDIT_BLUR", "-1", edit_blur, 0.04f),
};

#undef ROW

/** @brief Waits for a child.
 *
 * @param child The child.
 * @return      Its status, as waitpid() reports it.
 */
static int
wait_for (pid_t child)
{
	int status = 0;
	while (waitpid(child, &status, 0) < 0)
		require(errno == EINTR, "waitpid failed");
	return status;
}

/** @brief Each override, in a child of its own. */
static void
check_overrides (void)
{
	for (uint32_t i = 0; i < sizeof OVERRIDES / sizeof *OVERRIDES; ++i) {
		struct override const *const o = &OVERRIDES[i];
		require(!fflush(nullptr), "fflush failed");
		pid_t const child = fork();
		require(child >= 0, "fork failed");
		if (!child) {
			if (setenv(o->variable, o->value, 1))
				_exit(2);
			struct ShmHeader header;
			native_header(&header);
			struct composition_frame_settings want = native_defaults();
			*real_at(&want, o->field) = o->want;
			expect(composition_frame_settings_read(&header), want, "%s=\"%s\"", o->variable, o->value);
			if (fflush(nullptr))
				_exit(2);
			_exit(0);
		}
		int const status = wait_for(child);
		require(WIFEXITED(status) && !WEXITSTATUS(status), "%s=\"%s\" gave other settings", o->variable,
		        o->value);
	}
}

/** @brief A float of the header, and what composition_frame_settings_read() makes of one that is not
 *         a number or infinite: the default, not a bound.
 */
struct header_real {
	uint32_t bits;     //!< The offset of the header's word.
	uint32_t field;    //!< The offset of the settings' float.
	float    fallback; //!< The float's default.
};

#define ROW(bits, field, fallback) \
	{offsetof(struct ShmHeader, bits), offsetof(struct composition_frame_settings, field), fallback}

/** @brief The header's floats. */
static struct header_real const HEADER_REALS[] = {
	ROW(transferStrengthBits, transfer_strength, 1.0f),
	ROW(colourStrengthBits, colour_strength, 1.0f),
	ROW(maxRatioBits, max_ratio, 2.0f),
	ROW(debugScaleBits, debug_scale, 1.0f),
	ROW(whitePointBits, white_point_manual, 1.0f),
	ROW(whitePointScaleBits, white_point_scale, 1.0f),
	ROW(whitePointTrimBits, white_point_trim, 1.0f),
	ROW(compareSplitBits, compare_split, 0.5f),
	ROW(compareZoomBits, compare_zoom, 1.0f),
	ROW(workingScaleBits, working_scale, 1.0f),
};

#undef ROW

/** @brief A finite header float and what composition_frame_settings_read() makes of it: clamped to
 *         the nearer bound, or unchanged within them. -0.0 stays -0.0, as std::max keeps its first
 *         argument when both compare equal.
 */
struct real_case {
	uint32_t bits;  //!< The offset of the header's word.
	uint32_t field; //!< The offset of the settings' float.
	float    value; //!< The header's float.
	float    want;  //!< The settings' float.
};

#define ROW(bits, field, value, want) \
	{offsetof(struct ShmHeader, bits), offsetof(struct composition_frame_settings, field), value, want}

/** @brief The finite header floats. */
static struct real_case const REAL_CASES[] = {
	ROW(transferStrengthBits, transfer_strength, -1.0f, 0.0f),
	ROW(transferStrengthBits, transfer_strength, -0.0f, -0.0f),
	ROW(transferStrengthBits, transfer_strength, 0.0f, 0.0f),
	ROW(transferStrengthBits, transfer_strength, 5.0f, 4.0f),
	ROW(colourStrengthBits, colour_strength, -2.0f, 0.0f),
	ROW(colourStrengthBits, colour_strength, 3.5f, 3.5f),
	ROW(colourStrengthBits, colour_strength, 4.5f, 4.0f),
	ROW(maxRatioBits, max_ratio, 0.5f, 1.0f),
	ROW(maxRatioBits, max_ratio, 7.25f, 7.25f),
	ROW(maxRatioBits, max_ratio, 100.0f, (float)kMaxPasses),
	ROW(debugScaleBits, debug_scale, 0.0f, 0.01f),
	ROW(debugScaleBits, debug_scale, 0.01f, 0.01f),
	ROW(debugScaleBits, debug_scale, 1000.0f, 100.0f),
	ROW(whitePointBits, white_point_manual, 0.0f, 1e-4f),
	ROW(whitePointBits, white_point_manual, 2000.0f, 2000.0f),
	ROW(whitePointBits, white_point_manual, 1e6f, 2000.0f),
	ROW(whitePointScaleBits, white_point_scale, -3.0f, 0.01f),
	ROW(whitePointScaleBits, white_point_scale, 0.3f, 0.3f),
	ROW(whitePointScaleBits, white_point_scale, 101.0f, 100.0f),
	ROW(whitePointTrimBits, white_point_trim, 0.001f, 0.01f),
	ROW(whitePointTrimBits, white_point_trim, 100.0f, 100.0f),
	ROW(whitePointTrimBits, white_point_trim, 1e30f, 100.0f),
	ROW(compareSplitBits, compare_split, -0.5f, 0.0f),
	ROW(compareSplitBits, compare_split, 1.0f, 1.0f),
	ROW(compareSplitBits, compare_split, 2.0f, 1.0f),
	ROW(compareZoomBits, compare_zoom, 0.5f, 1.0f),
	ROW(compareZoomBits, compare_zoom, 1.5f, 1.5f),
	ROW(compareZoomBits, compare_zoom, 3.0f, 2.0f),
	ROW(workingScaleBits, working_scale, 0.1f, 0.25f),
	ROW(workingScaleBits, working_scale, 0.25f, 0.25f),
	ROW(workingScaleBits, working_scale, 0.75f, 0.75f),
	ROW(workingScaleBits, working_scale, 1.5f, 1.0f),
	ROW(workingScaleBits, working_scale, 4.0f, 1.0f),
};

#undef ROW

/** @brief Reads a header that differs from the daemon's defaults in one float.
 *
 * @param bits  The offset of the header's word.
 * @param field The offset of the settings' float.
 * @param value The header's float.
 * @param want  The settings' float.
 */
static void
check_real (uint32_t bits,
            uint32_t field,
            float    value,
            float    want)
{
	struct ShmHeader header;
	native_header(&header);
	atomic_store(header_word(&header, bits), FloatToBits(value));
	struct composition_frame_settings expected = native_defaults();
	*real_at(&expected, field) = want;
	expect(composition_frame_settings_read(&header), expected, "a header value of %f", (double)value);
}

/** @brief A header word, and what the settings hold for it. */
struct word_case {
	uint32_t source; //!< The offset of the header's word.
	uint32_t value;  //!< The header's word.
	uint32_t field;  //!< The offset of the settings' word.
	uint32_t want;   //!< The settings' word.
};

/** @brief A header percentage, and what the settings hold for it. */
struct percent_case {
	uint32_t source; //!< The offset of the header's word.
	uint32_t value;  //!< The header's word.
	uint32_t field;  //!< The offset of the settings' float.
	float    want;   //!< The settings' float.
};

/** @brief The native model's maximum raster in the header, and in the settings. */
struct maxima_case {
	uint32_t width;       //!< The header's width.
	uint32_t height;      //!< The header's height.
	uint32_t want_width;  //!< The settings' width.
	uint32_t want_height; //!< The settings' height.
};

/** @brief The settings that headers give. */
static void
check_read (void)
{
	expect(composition_frame_settings(), initial(), "the defaults");
	expect(composition_frame_settings_read(nullptr), initial(), "no header");
	// Each header is initialized once, so each case has its own.
	struct ShmHeader daemon;
	native_header(&daemon);
	expect(composition_frame_settings_read(&daemon), native_defaults(), "the daemon's defaults");

	// What the layer writes when it creates the channel.
	struct ShmHeader layer;
	ShmInitDefaults(&layer);
	struct composition_frame_settings want = native_defaults();
	want.transfer = 1;
	want.composition_bypass = 1;
	expect(composition_frame_settings_read(&layer), want, "the layer's defaults");

	float const unusable[] = {NAN, INFINITY, -INFINITY};
	for (uint32_t v = 0; v < sizeof unusable / sizeof *unusable; ++v)
		for (uint32_t i = 0; i < sizeof HEADER_REALS / sizeof *HEADER_REALS; ++i)
			check_real(HEADER_REALS[i].bits, HEADER_REALS[i].field, unusable[v], HEADER_REALS[i].fallback);
	for (uint32_t i = 0; i < sizeof REAL_CASES / sizeof *REAL_CASES; ++i)
		check_real(REAL_CASES[i].bits, REAL_CASES[i].field, REAL_CASES[i].value, REAL_CASES[i].want);

	// The words: some clamp to a bound, some fall back to a default, the rest pass as they are.
#define ROW(source, value, field, want) \
	{offsetof(struct ShmHeader, source), value, offsetof(struct composition_frame_settings, field), want}
	static struct word_case const words[] = {
		ROW(whitePointSource, kWhitePointMeasured, white_point_source, kWhitePointMeasured),
		ROW(whitePointSource, kWhitePointMeasured + 1, white_point_source, kWhitePointManual),
		ROW(transfer, 0, transfer, 0),
		ROW(transfer, 3, transfer, 2),
		ROW(transfer, UINT32_MAX, transfer, 2),
		ROW(debugView, 5, debug_view, 5),
		ROW(debugView, 6, debug_view, 0),
		ROW(compareMode, 2, compare_mode, 2),
		ROW(compareMode, 3, compare_mode, 0),
		ROW(reversibleMode, kReversibleModeCount - 1, reversible_mode, kReversibleModeCount - 1),
		ROW(reversibleMode, kReversibleModeCount, reversible_mode, kReversibleKnee),
		ROW(controlSeq, 77, control_seq, 77),
		ROW(tuningSeq, 78, tuning_seq, 78),
		ROW(passes, 99, passes, 99),
		ROW(compareSwap, 7, compare_swap, 7),
		ROW(applyModel, 0, apply_model, 0),
		ROW(holdFrame, 3, hold_frame, 3),
		ROW(compositionBypass, 2, composition_bypass, 2),
	};
	for (uint32_t i = 0; i < sizeof words / sizeof *words; ++i) {
		struct word_case const *const w = &words[i];
		struct ShmHeader header;
		native_header(&header);
		atomic_store(header_word(&header, w->source), w->value);
		want = native_defaults();
		*word_at(&want, w->field) = w->want;
		expect(composition_frame_settings_read(&header), want, "word %u", w->value);
	}

	// Percentages from the header: ratio smoothing up to 1, colour trust up to 8.
	static struct percent_case const percents[] = {
		ROW(ratioSmoothPercent, 0, ratio_smooth, 0.0f),
		ROW(ratioSmoothPercent, 50, ratio_smooth, 50 / 100.0f),
		ROW(ratioSmoothPercent, 150, ratio_smooth, 1.0f),
		ROW(colourTrustPercent, 0, colour_trust, 0.0f),
		ROW(colourTrustPercent, 350, colour_trust, 350 / 100.0f),
		ROW(colourTrustPercent, 1000, colour_trust, 8.0f),
	};
#undef ROW
	for (uint32_t i = 0; i < sizeof percents / sizeof *percents; ++i) {
		struct percent_case const *const p = &percents[i];
		struct ShmHeader header;
		native_header(&header);
		atomic_store(header_word(&header, p->source), p->value);
		want = native_defaults();
		*real_at(&want, p->field) = p->want;
		expect(composition_frame_settings_read(&header), want, "percent %u", p->value);
	}

	// The native model's maximum raster: a pair within kMinW..kMaxW x kMinH..kMaxH, or none.
	static struct maxima_case const maxima[] = {
		{kMinW, kMinH, kMinW, kMinH},
		{kMaxW, kMaxH, kMaxW, kMaxH},
		{1280, 720, 1280, 720},
		{kMaxW + 1, kMaxH, 0, 0},
		{kMaxW, kMaxH + 1, 0, 0},
		{kMinW - 1, 720, 0, 0},
		{1280, kMinH - 1, 0, 0},
		{1280, 0, 0, 0},
		{0, 0, 0, 0},
	};
	for (uint32_t i = 0; i < sizeof maxima / sizeof *maxima; ++i) {
		struct maxima_case const *const m = &maxima[i];
		struct ShmHeader header;
		native_header(&header);
		atomic_store(&header.nativeModelMaxWidth, m->width);
		atomic_store(&header.nativeModelMaxHeight, m->height);
		want = native_defaults();
		want.native_model_max_width = m->want_width;
		want.native_model_max_height = m->want_height;
		expect(composition_frame_settings_read(&header), want, "maxima %ux%u", m->width, m->height);
	}
}

/** @brief A frame, its settings, and the model raster they give. */
struct extent_case {
	uint32_t width;         //!< The frame's width.
	uint32_t height;        //!< The frame's height.
	float    working_scale; //!< The working scale.
	uint32_t max_width;     //!< The native model's widest raster, or 0.
	uint32_t max_height;    //!< The native model's tallest raster, or 0.
	uint32_t model_width;   //!< The model's width.
	uint32_t model_height;  //!< The model's height.
};

/** @brief The model raster over working scales and native maxima. */
static void
check_model_extent (void)
{
	static struct extent_case const cases[] = {
		// Without a native maximum: the working scale, rounded half away from zero, and at least 64
		// unless the scale is exactly 1.
		{1920, 1080, 1.0f, 0, 0, 1920, 1080},
		{32, 16, 1.0f, 0, 0, 32, 16},
		{32, 16, 0.999f, 0, 0, 64, 64},
		{640, 360, 0.999f, 0, 0, 639, 360},
		{1920, 1080, 0.5f, 0, 0, 960, 540},
		{1001, 333, 0.5f, 0, 0, 501, 167},
		{1280, 720, 0.75f, 0, 0, 960, 540},
		{200, 100, 0.25f, 0, 0, 64, 64},
		// Half a maximum, or a frame side of zero, is no maximum.
		{1920, 1080, 0.5f, 960, 0, 960, 540},
		{1920, 1080, 0.5f, 0, 540, 960, 540},
		{0, 100, 1.0f, 960, 540, 0, 100},
		// With one: the least of the working scale and each side's ratio to the maximum, and at least
		// kMinW x kMinH.
		{1920, 1080, 1.0f, 960, 540, 960, 540},
		{1920, 1080, 1.0f, 1280, 1280, 1280, 720},
		{320, 192, 1.0f, 160, 96, 160, 96},
		{640, 360, 1.0f, 1920, 1080, 640, 360},
		{1920, 1080, 0.25f, 3840, 2160, 480, 270},
		{1001, 333, 0.5f, 3840, 2160, 501, 167},
		{100, 80, 1.0f, 64, 64, 64, 64},
		{32, 32, 1.0f, 64, 64, 64, 64},
		// Both paths take the product in double: 105 * 0.7f is 73.4999987 in double and rounds to
		// 73, while the product in float is 73.5 and rounds to 74.
		{150, 105, 0.7f, 0, 0, 105, 73},
		{105, 105, 0.7f, 3840, 2160, 73, 73},
	};
	for (uint32_t i = 0; i < sizeof cases / sizeof *cases; ++i) {
		struct extent_case const *const c = &cases[i];
		struct composition_frame_settings s = initial();
		s.working_scale = c->working_scale;
		s.native_model_max_width = c->max_width;
		s.native_model_max_height = c->max_height;
		VkExtent2D const model = composition_model_extent(c->width, c->height, &s);
		require(model.width == c->model_width && model.height == c->model_height,
		        "composition_model_extent(%ux%u, scale %g, maximum %ux%u) is %ux%u, not %ux%u", c->width,
		        c->height, (double)c->working_scale, c->max_width, c->max_height, model.width, model.height,
		        c->model_width, c->model_height);
	}
}

/** @brief A swapchain format and the composition's format for it. */
struct format_case {
	VkFormat swapchain;   //!< The swapchain's format.
	VkFormat composition; //!< The composition's, or VK_FORMAT_UNDEFINED for none.
};

/** @brief The composition's format table, and which frames hold light. */
static void
check_formats (void)
{
	static struct format_case const formats[] = {
		{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM},
		{VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM},
		{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
		{VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM},
		{VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_A8B8G8R8_UNORM_PACK32},
		{VK_FORMAT_A8B8G8R8_SRGB_PACK32, VK_FORMAT_A8B8G8R8_UNORM_PACK32},
		{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
		{VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32},
		{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
		{VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_UNDEFINED},
		{VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_UNDEFINED},
		{(VkFormat)1000452000, VK_FORMAT_UNDEFINED},
	};
	for (uint32_t i = 0; i < sizeof formats / sizeof *formats; ++i) {
		VkFormat const got = composition_format(formats[i].swapchain);
		require(got == formats[i].composition, "composition_format(%d) is %d", (int)formats[i].swapchain,
		        (int)got);
	}

	// Display-referred and linear HDR are forced; auto, and any other mode, takes a float
	// swapchain for light and everything else for a picture.
	for (uint32_t i = 0; i < sizeof formats / sizeof *formats; ++i) {
		VkFormat const format = formats[i].swapchain;
		bool const fp16 = format == VK_FORMAT_R16G16B16A16_SFLOAT;
		// linear is 32 bits wide, which fills the padding.
		struct {
			uint32_t mode;
			uint32_t linear;
		} const modes[] = {
			{kColourAuto, fp16},
			{kColourDisplay, false},
			{kColourLinearHdr, true},
			{3, fp16},
			{UINT32_MAX, fp16},
		};
		for (uint32_t m = 0; m < sizeof modes / sizeof *modes; ++m)
			require(composition_colour_is_linear_hdr(format, modes[m].mode) == modes[m].linear,
			        "composition_colour_is_linear_hdr(%d, %u)", (int)format, modes[m].mode);
	}
}

/** @brief A key's name and its code. code stays 32 bits wide, as hotkey_key_name_from_code() takes
 *         it.
 */
struct key_case {
	char const *name; //!< The name.
	uint32_t    code; //!< The code.
};

/** @brief The toggle key's names. */
static void
check_key_names (void)
{
	static struct key_case const keys[] = {
		{"", 0},
		{"F10", KEY_F10},
		{"f10", KEY_F10},
		{"KEY_HOME", KEY_HOME},
		{"key_home", KEY_HOME},
		{"Home", KEY_HOME},
		{"123", 123},
		{"nonsense", 0},
		// A bare number is a key code, also where the table names a digit key.
		{"1", 1},
		{"KEY_1", KEY_1},
		{"0", 0},
		{"KEY_0", KEY_0},
		// A number is a key code below KEY_CNT; any other binds nothing, also beyond unsigned long.
		{"767", KEY_MAX},
		{"768", 0},
		{"4294967297", 0},
		{"18446744073709551615", 0},
		{"018446744073709551615", 0},
		{"18446744073709551616", 0},
		{"99999999999999999999999999", 0},
		{"12a", 0},
		{" F10", 0},
		{"F10 ", 0},
		{"KEY_", 0},
		{"KEY_KEY_F10", 0},
		{"z", KEY_Z},
		{"grave", KEY_GRAVE},
		{"SysRq", KEY_SYSRQ},
		// The longest name, and longer ones, which no key has.
		{"KEY_SCROLLLOCK", KEY_SCROLLLOCK},
		{"key_ScrollLock", KEY_SCROLLLOCK},
		{"KEY_SCROLLLOCK1", 0},
		{"KEY_SCROLLLOCK12", 0},
		{"SCROLLLOCK123456", 0},
	};
	require(hotkey_key_code_from_name(nullptr) == 0, "hotkey_key_code_from_name(nullptr) is not 0");
	char long_name[4100];
	memcpy(long_name, "F10", 3);
	memset(long_name + 3, ' ', 4096);
	long_name[sizeof long_name - 1] = '\0';
	require(hotkey_key_code_from_name(long_name) == 0, "a name of 4099 bytes bound a key");
	for (uint32_t i = 0; i < sizeof keys / sizeof *keys; ++i) {
		uint32_t const code = hotkey_key_code_from_name(keys[i].name);
		require(code == keys[i].code, "hotkey_key_code_from_name(\"%s\") is %u, not %u", keys[i].name, code,
		        keys[i].code);
	}

	static struct key_case const codes[] = {
		{"F1", KEY_F1}, {"F10", KEY_F10}, {"F12", KEY_F12}, {"HOME", KEY_HOME}, {"END", KEY_END},
		{"INSERT", KEY_INSERT}, {"DELETE", KEY_DELETE}, {"PAGEUP", KEY_PAGEUP}, {"PAGEDOWN", KEY_PAGEDOWN},
		{"PAUSE", KEY_PAUSE}, {"SCROLLLOCK", KEY_SCROLLLOCK}, {"SYSRQ", KEY_SYSRQ}, {"GRAVE", KEY_GRAVE},
		{"A", KEY_A}, {"Z", KEY_Z}, {"0", KEY_0}, {"1", KEY_1}, {"9", KEY_9},
		{"?", 0}, {"?", KEY_ESC}, {"?", KEY_F13}, {"?", KEY_KPPLUSMINUS}, {"?", UINT32_MAX},
	};
	for (uint32_t i = 0; i < sizeof codes / sizeof *codes; ++i) {
		char const *const name = hotkey_key_name_from_code(codes[i].code);
		require(!strcmp(name, codes[i].name), "hotkey_key_name_from_code(%u) is %s", codes[i].code, name);
	}

	// Every name the table has reads back as its code, except the digits, which read as numbers.
	// 12 function keys, 10 others, 26 letters and 10 digits.
	uint32_t named = 0;
	for (uint32_t code = 0; code <= KEY_MAX; ++code) {
		char const *const name = hotkey_key_name_from_code(code);
		if (!strcmp(name, "?"))
			continue;
		++named;
		bool const digit = name[0] >= '0' && name[0] <= '9' && !name[1];
		uint32_t const want = digit ? (uint32_t)(name[0] - '0') : code;
		require(hotkey_key_code_from_name(name) == want, "the name %s does not read back", name);
	}
	require(named == 58, "the key table has %u names, not 58", named);
}

/** @brief The variables the log reads, once per process, at the first call of a log.h function: each
 *         case runs in a child of its own, forked before this process calls one. The child leaves
 *         through exit(), so the log's destructor runs.
 */
static char const *const LOG_VARIABLES[] = {"VKLayer_DLSS5", "VKLAYER_DLSS5", "DLSSNR_ENABLE", "DLSSNR_LOG",
                                            "DLSSNR_VERBOSE", "DLSSNR_TIME", "DLSSNR_TIME_EVERY"};

/** @brief A variable for a child to set. */
struct env {
	char const *variable; //!< The variable.
	char const *value;    //!< Its value, or nullptr to leave it unset.
};

/** @brief Forks a child for a log check, which sets its variables.
 *
 * @param env   The variables.
 * @param count Their number.
 * @return      0 in the child, which then checks and ends with log_child_exit(); the child's ID in
 *              this process, which then calls log_child_wait().
 */
static pid_t
log_child (struct env const *env,
           size_t            count)
{
	require(!fflush(nullptr), "fflush failed");
	pid_t const child = fork();
	require(child >= 0, "fork failed");
	if (!child) {
		for (size_t i = 0; i < count; ++i)
			if (env[i].value && setenv(env[i].variable, env[i].value, 1))
				_exit(2);
	}
	return child;
}

/** @brief Ends a child of log_child() through exit(), so the log's destructor runs. */
[[noreturn]]
static void
log_child_exit (void)
{
	require(!fflush(nullptr), "fflush failed");
	exit(0);
}

/** @brief Ends the test unless a child of log_child() exits with 0.
 *
 * @param child The child.
 * @param fmt   A printf format of what the child checked.
 * @param ...   Its arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
log_child_wait (pid_t       child,
                char const *fmt,
                ...)
{
	int const status = wait_for(child);
	if (WIFEXITED(status) && !WEXITSTATUS(status))
		return;

	va_list args;
	va_start(args, fmt);
	print_label(fmt, args);
	va_end(args);
	fprintf(stderr, ": the child exited with %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
	exit(1);
}

/** @brief A switch's value, and whether it is on. The switches do not depend on whether the layer
 *         was asked for. on is 64 bits wide, which fills the padding.
 */
struct switch_case {
	char const *value; //!< The value, or nullptr for none.
	uint64_t    on;    //!< Whether the switch is on.
};

/** @brief DLSSNR_TIME_EVERY's value, and the interval. A prefix of digits counts, leading blanks and
 *         a sign are skipped, and anything not positive or beyond 32 bits is the default. interval
 *         is 64 bits wide, which fills the padding.
 */
struct interval_case {
	char const *value;    //!< The value, or nullptr for none.
	uint64_t    interval; //!< The interval.
};

/** @brief The log's switches and its timing interval. */
static void
check_log_switches (void)
{
	static struct switch_case const switches[] = {
		{nullptr, false}, {"1", true}, {"10", true}, {"1x", true}, {"0", false}, {"", false}, {" 1", false},
		{"yes", false}, {"true", false},
	};
	static struct interval_case const intervals[] = {
		{nullptr, 30}, {"", 30}, {"0", 30}, {"-5", 30}, {"abc", 30}, {"1", 1}, {"45", 45}, {"7x", 7},
		{" 12", 12}, {"+9", 9}, {"2147483647", 2147483647}, {"4294967295", 4294967295u}, {"4294967296", 30},
		{"99999999999999999999", 30},
	};
	for (uint32_t i = 0; i < sizeof switches / sizeof *switches; ++i) {
		struct switch_case const *const c = &switches[i];
		// The value in quotes, or "unset".
		char const *const q = c->value ? "\"" : "";
		char const *const value = c->value ? c->value : "unset";

		pid_t child = log_child(&(struct env const){"DLSSNR_VERBOSE", c->value}, 1);
		if (!child) {
			require(log_verbose() == c->on, "DLSSNR_VERBOSE %s%s%s: log_verbose() is wrong", q, value, q);
			require(!log_time_enabled(), "DLSSNR_VERBOSE %s%s%s: log_time_enabled() is on", q, value, q);
			log_child_exit();
		}
		log_child_wait(child, "DLSSNR_VERBOSE %s%s%s", q, value, q);

		child = log_child(&(struct env const){"DLSSNR_TIME", c->value}, 1);
		if (!child) {
			require(log_time_enabled() == c->on, "DLSSNR_TIME %s%s%s: log_time_enabled() is wrong", q,
			        value, q);
			require(!log_verbose(), "DLSSNR_TIME %s%s%s: log_verbose() is on", q, value, q);
			log_child_exit();
		}
		log_child_wait(child, "DLSSNR_TIME %s%s%s", q, value, q);
	}
	for (uint32_t i = 0; i < sizeof intervals / sizeof *intervals; ++i) {
		struct interval_case const *const c = &intervals[i];
		char const *const q = c->value ? "\"" : "";
		char const *const value = c->value ? c->value : "unset";
		pid_t const child = log_child(&(struct env const){"DLSSNR_TIME_EVERY", c->value}, 1);
		if (!child) {
			uint32_t const got = log_time_interval();
			require(got == c->interval, "DLSSNR_TIME_EVERY %s%s%s: log_time_interval() is %u, not %" PRIu64,
			        q, value, q, got, c->interval);
			log_child_exit();
		}
		log_child_wait(child, "DLSSNR_TIME_EVERY %s%s%s", q, value, q);
	}
}

/** @brief Whether a file exists.
 *
 * @param path The file.
 * @return     true if lstat() finds it.
 */
static bool
exists (char const *path)
{
	struct stat st;
	return !lstat(path, &st);
}

/** @brief Writes a file, which must succeed.
 *
 * @param path The file.
 * @param text What it holds.
 */
static void
write_file (char const *path,
            char const *text)
{
	FILE *const f = fopen(path, "wbe");
	require(f, "cannot write %s", path);
	bool const written = fputs(text, f) >= 0;
	require(!fclose(f) && written, "cannot write %s", path);
}

/** @brief Ends the test unless a file holds what it should.
 *
 * A file that cannot be read holds nothing.
 *
 * @param path   The file.
 * @param want   What it should hold.
 * @param length The length of @a want.
 * @param fmt    A printf format of the check, which the message follows with the text in quotes.
 * @param ...    Its arguments.
 */
[[gnu::format(printf, 4, 5)]]
static void
expect_file (char const *path,
             char const *want,
             size_t      length,
             char const *fmt,
             ...)
{
	size_t got = 0;
	char *text = support_read_file(path, &got);
	if (got == length && (!length || !memcmp(text, want, length))) {
		free(text);
		text = nullptr;
		return;
	}

	va_list args;
	va_start(args, fmt);
	print_label(fmt, args);
	va_end(args);
	fprintf(stderr, " \"%s\"\n", text ? text : "");
	exit(1);
}

/** @brief Sends the child's standard error to a file.
 *
 * @param path The file.
 */
static void
redirect_stderr (char const *path)
{
	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
	if (fd < 0 || dup2(fd, 2) != 2 || close(fd))
		_exit(2);
	fd = -1;
}

/** @brief The descriptor through which this process holds a file open.
 *
 * @param path   The file's path, as /proc/self/fd names it.
 * @param length The length of @a path.
 * @return       The descriptor, -1 if there is none, or -2 if /proc/self/fd cannot be read.
 */
static int
descriptor_of (char const *path,
               size_t      length)
{
	DIR *const fds = opendir("/proc/self/fd");
	if (!fds)
		return -2;

	// A link longer than the path fills the buffer, so it is never taken for the path.
	int const dir = dirfd(fds);
	char *target = dir >= 0 ? malloc(length + 1) : nullptr;
	int found = target ? -1 : -2;
	for (struct dirent *entry; found == -1 && (entry = readdir(fds));) {
		ssize_t const n = readlinkat(dir, entry->d_name, target, length + 1);
		if (n != (ssize_t)length || memcmp(target, path, length))
			continue;
		char *end;
		errno = 0;
		long const fd = strtol(entry->d_name, &end, 10);
		if (!errno && end != entry->d_name && !*end && fd >= 0 && fd <= INT_MAX)
			found = (int)fd;
	}
	free(target);
	target = nullptr;
	if (closedir(fds))
		found = -2;
	return found;
}

/** @brief The log file that a child must have closed when it ends, or nullptr. */
static char const *closed_at_exit;

/** @brief The length of closed_at_exit. */
static size_t closed_length;

/** @brief Whether check_after_log_close() checks this process. */
static bool check_at_exit;

/** @brief After the log's destructor, which runs before this one: the child's log file is no longer
 *         open, and its standard descriptors still are. A failure ends the child with 3.
 */
[[gnu::destructor(101)]]
static void
check_after_log_close (void)
{
	if (!check_at_exit)
		return;
	for (int fd = 0; fd < 3; ++fd)
		if (fcntl(fd, F_GETFD) < 0)
			_exit(3);
	if (closed_at_exit && descriptor_of(closed_at_exit, closed_length) != -1)
		_exit(3);
}

/** @brief The threads that log at once, and the lines each writes. */
enum {
	LOG_THREADS = 8,
	LOG_LINES   = 500
};

/** @brief The length of each threaded line's tail. */
static constexpr size_t TAIL_LENGTH = 200;

/** @brief Logs LOG_LINES lines whose tail is a letter of the thread's own.
 *
 * @param arg The thread's number, below LOG_THREADS.
 * @return    nullptr.
 */
static void *
log_lines (void *arg)
{
	uint32_t const t = *(uint32_t const *)arg;
	char tail[TAIL_LENGTH + 1];
	memset(tail, 'a' + t, TAIL_LENGTH);
	tail[TAIL_LENGTH] = '\0';
	for (uint32_t i = 0; i < LOG_LINES; ++i)
		log_printf("thread %u line %u %s", t, i, tail);
	return nullptr;
}

/** @brief The log's lines in its file and on standard error.
 *
 * @param dir The test's directory.
 */
static void
check_log_lines (char const *dir)
{
	// Each variable that asks for the layer, appended to what the file held.
	static struct env const enables[] = {
		{"DLSSNR_ENABLE", "1"}, {"VKLayer_DLSS5", "1"}, {"VKLAYER_DLSS5", "1x"},
	};
	static char const want_appended[] = "an earlier line\n[dlssnr-layer] [test] one 2 3.00 (nil)\n"
	                                    "[dlssnr-layer] \n";
	for (uint32_t i = 0; i < sizeof enables / sizeof *enables; ++i) {
		struct env const *const enable = &enables[i];
		size_t log_length = 0;
		char *log_path = support_format(&log_length, "%s/%s.log", dir, enable->variable);
		char *err = support_format(nullptr, "%s/%s.err", dir, enable->variable);
		require(log_path && err, "out of memory");
		write_file(log_path, "an earlier line\n");
		pid_t const child = log_child((struct env const[]){*enable, {"DLSSNR_LOG", log_path}}, 2);
		if (!child) {
			redirect_stderr(err);
			log_printf("[test] %s %u %.2f %p", "one", 2u, 3.0, (void *)nullptr);
			log_printf("%s", "");
			closed_at_exit = log_path;
			closed_length = log_length;
			check_at_exit = true;
			log_child_exit();
		}
		log_child_wait(child, "%s=%s", enable->variable, enable->value);
		expect_file(log_path, want_appended, sizeof want_appended - 1, "%s=%s: the log holds",
		            enable->variable, enable->value);
		expect_file(err, "", 0, "%s=%s: the log wrote to stderr", enable->variable, enable->value);
		free(err);
		err = nullptr;
		free(log_path);
		log_path = nullptr;
	}

	// The programs that the process executes do not inherit the file.
	size_t exec_length = 0;
	char *exec_log = support_format(&exec_length, "%s/exec.log", dir);
	require(exec_log, "out of memory");
	pid_t child = log_child((struct env const[]){{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", exec_log}}, 2);
	if (!child) {
		log_printf("[test] %s", "exec");
		int const fd = descriptor_of(exec_log, exec_length);
		require(fd >= 0, "close on exec: the log's file is not open");
		int const flags = fcntl(fd, F_GETFD);
		require(flags >= 0 && flags & FD_CLOEXEC,
		        "close on exec: the log's file stays open in executed programs");
		log_child_exit();
	}
	log_child_wait(child, "close on exec");
	free(exec_log);
	exec_log = nullptr;

	// Not asked for: nothing is written, and DLSSNR_LOG is not created.
	char *off_log = support_format(nullptr, "%s/off.log", dir);
	char *off_err = support_format(nullptr, "%s/off.err", dir);
	require(off_log && off_err, "out of memory");
	struct env const off_only[] = {{"DLSSNR_LOG", off_log}};
	struct env const off_values[] = {{"DLSSNR_ENABLE", "0"}, {"VKLayer_DLSS5", ""}, {"VKLAYER_DLSS5", "yes"},
	                                 {"DLSSNR_LOG", off_log}};
	struct env const off_blank[] = {{"DLSSNR_ENABLE", " 1"}, {"DLSSNR_VERBOSE", "1"}, {"DLSSNR_TIME", "1"},
	                                {"DLSSNR_LOG", off_log}};
	// count is a size_t, as sizeof gives it, which fills the padding.
	struct {
		struct env const *env;
		size_t            count;
	} const not_enabled[] = {
		{off_only, sizeof off_only / sizeof *off_only},
		{off_values, sizeof off_values / sizeof *off_values},
		{off_blank, sizeof off_blank / sizeof *off_blank},
	};
	for (uint32_t row = 0; row < sizeof not_enabled / sizeof *not_enabled; ++row) {
		child = log_child(not_enabled[row].env, not_enabled[row].count);
		if (!child) {
			redirect_stderr(off_err);
			log_printf("[test] %s", "dropped");
			check_at_exit = true;
			log_child_exit();
		}
		log_child_wait(child, "not enabled, row %u", row);
		require(!exists(off_log), "not enabled, row %u: DLSSNR_LOG was created", row);
		expect_file(off_err, "", 0, "not enabled, row %u: the log wrote to stderr", row);
	}
	free(off_err);
	off_err = nullptr;
	free(off_log);
	off_log = nullptr;

	// No usable DLSSNR_LOG: standard error.
	char *missing_dir = support_format(nullptr, "%s/missing", dir);
	char *missing = support_format(nullptr, "%s/missing/layer.log", dir);
	char *stderr_err = support_format(nullptr, "%s/stderr.err", dir);
	require(missing_dir && missing && stderr_err, "out of memory");
	static char const want_stderr[] = "[dlssnr-layer] [test] to stderr\n";
	char const *const stderr_logs[] = {nullptr, "", missing, dir};
	for (uint32_t i = 0; i < sizeof stderr_logs / sizeof *stderr_logs; ++i) {
		char const *const path = stderr_logs[i];
		char const *const q = path ? "\"" : "";
		char const *const shown = path ? path : "unset";
		child = log_child((struct env const[]){{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", path}}, 2);
		if (!child) {
			redirect_stderr(stderr_err);
			log_printf("[test] to %s", "stderr");
			check_at_exit = true;
			log_child_exit();
		}
		log_child_wait(child, "DLSSNR_LOG %s%s%s", q, shown, q);
		expect_file(stderr_err, want_stderr, sizeof want_stderr - 1, "DLSSNR_LOG %s%s%s: stderr holds", q,
		            shown, q);
		require(!exists(missing_dir), "DLSSNR_LOG %s%s%s: a directory was created", q, shown, q);
	}
	free(stderr_err);
	stderr_err = nullptr;
	free(missing);
	missing = nullptr;
	free(missing_dir);
	missing_dir = nullptr;

	// The text is cut to 2047 bytes, the prefix and the newline are not.
	char *long_log = support_format(nullptr, "%s/long.log", dir);
	require(long_log, "out of memory");
	static size_t const written[] = {2046, 2047, 2048, 5000};
	static size_t const kept[] = {2046, 2047, 2047, 2047};
	char *xs = malloc(5000 + 1);
	require(xs, "out of memory");
	child = log_child((struct env const[]){{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", long_log}}, 2);
	if (!child) {
		for (uint32_t i = 0; i < sizeof written / sizeof *written; ++i) {
			memset(xs, 'x', written[i]);
			xs[written[i]] = '\0';
			log_printf("%s", xs);
		}
		log_child_exit();
	}
	log_child_wait(child, "long lines");
	static char const prefix[] = "[dlssnr-layer] ";
	size_t const prefix_length = sizeof prefix - 1;
	size_t want_length = 0;
	for (uint32_t i = 0; i < sizeof kept / sizeof *kept; ++i)
		want_length += prefix_length + kept[i] + 1;
	char *want = malloc(want_length);
	require(want, "out of memory");
	size_t at = 0;
	for (uint32_t i = 0; i < sizeof kept / sizeof *kept; ++i) {
		memcpy(want + at, prefix, prefix_length);
		memset(want + at + prefix_length, 'x', kept[i]);
		at += prefix_length + kept[i];
		want[at++] = '\n';
	}
	size_t length = 0;
	char *text = support_read_file(long_log, &length);
	require(text && length == want_length && !memcmp(text, want, length),
	        "long lines are not cut to 2047 bytes");
	free(text);
	text = nullptr;
	free(want);
	want = nullptr;
	free(xs);
	xs = nullptr;
	free(long_log);
	long_log = nullptr;

	// Lines from several threads at once stay whole, and each thread's stay in order.
	char *thread_log = support_format(nullptr, "%s/threads.log", dir);
	require(thread_log, "out of memory");
	child = log_child((struct env const[]){{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", thread_log}}, 2);
	if (!child) {
		pthread_t threads[LOG_THREADS];
		uint32_t numbers[LOG_THREADS];
		for (uint32_t t = 0; t < LOG_THREADS; ++t) {
			numbers[t] = t;
			require(!pthread_create(&threads[t], nullptr, log_lines, &numbers[t]),
			        "pthread_create failed");
		}
		for (uint32_t t = 0; t < LOG_THREADS; ++t)
			require(!pthread_join(threads[t], nullptr), "pthread_join failed");
		log_child_exit();
	}
	log_child_wait(child, "threads");
	text = support_read_file(thread_log, &length);
	require(text, "cannot read %s", thread_log);
	uint32_t next[LOG_THREADS] = {};
	char const *line = text;
	char const *const stop = text + length;
	size_t count = 0;
	for (char const *end; line < stop && (end = memchr(line, '\n', (size_t)(stop - line))); line = end + 1) {
		int const shown = (int)(end - line);
		unsigned t = 0;
		unsigned i = 0;
		char tail[256] = {};
		require(sscanf(line, "[dlssnr-layer] thread %u line %u %255s", &t, &i, tail) == 3
		        && t < LOG_THREADS, "a thread's line is broken: %.*s", shown, line);
		char const letter = (char)('a' + t);
		bool whole = strlen(tail) == TAIL_LENGTH;
		for (size_t c = 0; whole && c < TAIL_LENGTH; ++c)
			whole = tail[c] == letter;
		require(i == next[t]++ && whole, "a thread's line is out of order or broken: %.*s", shown, line);
		++count;
	}
	require(line == stop && count == LOG_THREADS * LOG_LINES, "the threads' log has %zu lines", count);
	free(text);
	text = nullptr;
	free(thread_log);
	thread_log = nullptr;
}

/** @brief Whether a child exits with 0 within five seconds. A child that does not is killed.
 *
 * @param child The child.
 * @return      true if it did.
 */
static bool
exits_in_time (pid_t child)
{
	struct timespec const ms = {.tv_nsec = 1000000};
	for (uint32_t i = 0; i < 5000; ++i) {
		int status = 0;
		pid_t const done = waitpid(child, &status, WNOHANG);
		if (done == child)
			return WIFEXITED(status) && !WEXITSTATUS(status);
		require(done == 0 || errno == EINTR, "waitpid failed");
		if (nanosleep(&ms, nullptr))
			require(errno == EINTR, "nanosleep failed");
	}
	require(!kill(child, SIGKILL), "kill failed");
	while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
	return false;
}

/** @brief Logs until told to stop.
 *
 * @param arg The atomic_bool that tells it.
 * @return    nullptr.
 */
static void *
log_until_stopped (void *arg)
{
	atomic_bool const *const stop = arg;
	while (!atomic_load_explicit(stop, memory_order_relaxed))
		log_printf("[test] %0200u", 0u);
	return nullptr;
}

/** @brief A child that fork() makes while a thread writes a line has the log's lock held by a thread
 *         it does not have. Its exit() runs the log's destructor, which must not wait for that lock.
 *
 * The writer logs to standard error, sent to /dev/null until the check ends, and allocates nothing.
 * The children therefore hold no descriptor that valgrind's --track-fds reports and no memory that
 * LeakSanitizer reports.
 */
static void
check_log_fork_exit (void)
{
	enum { FORKS = 50 };
	pid_t const child = log_child((struct env const[]){{"DLSSNR_ENABLE", "1"}}, 1);
	if (!child) {
		int err = dup(2);
		if (err < 0)
			_exit(2);
		redirect_stderr("/dev/null");
		atomic_bool stop = false;
		pthread_t writer;
		require(!pthread_create(&writer, nullptr, log_until_stopped, &stop), "pthread_create failed");
		uint32_t exited = 0;
		bool forked = true;
		while (exited < FORKS) {
			pid_t const grandchild = fork();
			if (!grandchild) {
				if (close(err))
					_exit(2);
				exit(0);
			}
			forked = grandchild > 0;
			if (!forked || !exits_in_time(grandchild))
				break;
			++exited;
		}
		atomic_store_explicit(&stop, true, memory_order_relaxed);
		require(!pthread_join(writer, nullptr), "pthread_join failed");
		if (dup2(err, 2) != 2 || close(err))
			_exit(2);
		err = -1;
		require(forked, "fork failed");
		require(exited == FORKS, "a child forked while a thread logged did not exit, after %u that did",
		        exited);
		log_child_exit();
	}
	log_child_wait(child, "fork while a thread logs");
}

/** @brief The time of CLOCK_MONOTONIC in milliseconds, as std::chrono::steady_clock reads it. */
static double
monotonic_ms (void)
{
	struct timespec now;
	require(!clock_gettime(CLOCK_MONOTONIC, &now), "clock_gettime failed");
	return (double)(now.tv_sec * INT64_C(1000000000) + now.tv_nsec) / 1e6;
}

/** @brief log_now_ms() reads the clock that std::chrono::steady_clock reads, in milliseconds. */
static void
check_clock (void)
{
	for (uint32_t i = 0; i < 1000; ++i) {
		double const before = monotonic_ms();
		double const now = log_now_ms();
		double const after = monotonic_ms();
		require(before <= now && now <= after, "log_now_ms() is not between two CLOCK_MONOTONIC readings");
	}
}

/** @brief The log, in a directory of the test's own, which it removes. */
static void
check_log (void)
{
	char const *const tmp = getenv("TMPDIR");
	char *made = support_temp_dir(tmp && *tmp ? tmp : "/tmp", "dlsslop-amd-log", nullptr);
	require(made, "mkdtemp failed");
	// /proc/self/fd names the files by their real paths.
	char *dir = realpath(made, nullptr);
	require(dir, "realpath failed");
	free(made);
	made = nullptr;

	check_log_switches();
	check_log_lines(dir);
	check_log_fork_exit();
	check_clock();

	// Files only: a directory in it fails the unlink.
	DIR *const entries = opendir(dir);
	require(entries, "cannot list %s", dir);
	int const entries_fd = dirfd(entries);
	require(entries_fd >= 0, "cannot list %s", dir);
	for (struct dirent *entry; (entry = readdir(entries));)
		if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
			require(!unlinkat(entries_fd, entry->d_name, 0), "unlink failed");
	require(!closedir(entries), "closedir failed");
	require(!rmdir(dir), "rmdir failed");
	free(dir);
	dir = nullptr;
}

int
main (void)
{
	// The log and the overrides first: the log reads its variables and composition_frame_settings_read()
	// its overrides in this process at their first call, and the children forked below must each make
	// that first call themselves.
	for (uint32_t i = 0; i < sizeof LOG_VARIABLES / sizeof *LOG_VARIABLES; ++i)
		require(!unsetenv(LOG_VARIABLES[i]), "unsetenv failed");
	check_log();
	static char const *const overrides[] = {"DLSSNR_GHOST_SLACK", "DLSSNR_RATIO_SMOOTH", "DLSSNR_COLOUR_TRUST",
	                                        "DLSSNR_MOTION_SMOOTH", "DLSSNR_EDIT_BLUR"};
	for (uint32_t i = 0; i < sizeof overrides / sizeof *overrides; ++i)
		require(!unsetenv(overrides[i]), "unsetenv failed");
	check_overrides();
	check_read();
	check_model_extent();
	check_formats();
	check_key_names();
	if (puts("layer-units-test: the log, settings, model extents, formats and key names hold") == EOF)
		return 1;
	return 0;
}
