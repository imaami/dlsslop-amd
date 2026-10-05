/** @file
 *
 * Native Linux controls. Shared-memory offsets and defaults come from the protocol.
 */
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "control_settings.h"
#include "error.h"
#include "shm_channel.h"
#include "shm_protocol.h"

/** @brief --capture's range: its count parses like an integer setting's value. */
static constexpr struct control_setting CAPTURE = { .minimum = 0, .maximum = 64, .step = 1 };

static_assert(CONTROL_SETTING_COUNT <= 64, "struct options keeps one mask bit per setting");

/** @brief The options that are not settings. */
static struct option const FIXED_OPTIONS[] = {
	{ "shm",      required_argument, nullptr, 's' },
	{ "status",   no_argument,       nullptr, 'S' },
	{ "settings", no_argument,       nullptr, 'l' },
	{ "reset",    no_argument,       nullptr, 'r' },
	{ "quit",     no_argument,       nullptr, 'q' },
	{ "resume",   no_argument,       nullptr, 'R' },
	{ "capture",  required_argument, nullptr, 'c' },
	{ "toggle",   required_argument, nullptr, 'A' },
	{ "help",     no_argument,       nullptr, 'h' }
};

/** @brief The number of FIXED_OPTIONS. */
static constexpr uint32_t FIXED_COUNT = sizeof FIXED_OPTIONS / sizeof *FIXED_OPTIONS;

/** @brief The short forms of FIXED_OPTIONS, after '+', which stops the options at the first
 *         operand.
 */
static constexpr char FIXED_SHORT[] = "+s:SlrqRc:A:h";

/** @brief What a command line asks for besides its settings, in struct options' flags. */
enum options_flags : uint32_t {
	OPTIONS_HELP     = 1 << 0, //!< --help.
	OPTIONS_STATUS   = 1 << 1, //!< --status, or neither an action nor a setting.
	OPTIONS_SETTINGS = 1 << 2, //!< --settings.
	OPTIONS_RESET    = 1 << 3, //!< --reset.
	OPTIONS_QUIT     = 1 << 4, //!< --quit.
	OPTIONS_RESUME   = 1 << 5, //!< --resume.
	OPTIONS_CAPTURE  = 1 << 6  //!< --capture.
};

/** @brief A parsed command line. */
struct options {
	uint64_t    assigned;                      //!< Bit i: values[i] is CONTROL_SETTINGS[i]'s new value.
	uint64_t    toggled;                       //!< Bit i: CONTROL_SETTINGS[i] flips once.
	char const *path;                          //!< --shm's channel, or nullptr for the native one.
	uint32_t    capture;                       //!< --capture's count.
	uint32_t    flags;                         //!< enum options_flags.
	uint32_t    values[CONTROL_SETTING_COUNT]; //!< The assigned settings' bits.
};

/** @brief Whether a command line changes the channel.
 *
 * @param options The command line.
 * @return        true if it assigns or toggles a setting, resets, quits, resumes or captures.
 */
static bool
options_changes_header (struct options const *options)
{
	return options->assigned || options->toggled
	       || (options->flags & (OPTIONS_RESET | OPTIONS_QUIT | OPTIONS_RESUME | OPTIONS_CAPTURE));
}

/** @brief A setting's bit in struct options' masks.
 *
 * @param setting A setting of CONTROL_SETTINGS.
 * @return        Its bit.
 */
static uint64_t
setting_bit (struct control_setting const *setting)
{
	return UINT64_C(1) << (setting - CONTROL_SETTINGS);
}

/** @brief Prints "; steps of N" for a setting whose values skip, as help and errors print it.
 *
 * @param stream Where to print.
 * @param s      The setting.
 */
static void
print_steps (FILE                         *stream,
             struct control_setting const *s)
{
	if (s->step != 1)
		fprintf(stream, "; steps of %g", s->step);
}

/** @brief Prints the help.
 *
 * @param channel The native tools' channel, which --shm overrides.
 */
static void
usage (char const *channel)
{
	uint32_t defaults[CONTROL_SETTING_COUNT];
	control_settings_defaults(defaults, false);
	printf("Usage: dlsslopctl [OPTION]...\n"
	       "       dlssnr-shmctl [OPTION]...\n"
	       "\n"
	       "Omitted setting options leave current values unchanged. The setting defaults\n"
	       "below are initial/reset values, not the current values. Use --settings to\n"
	       "inspect both. With no action or setting option, --status is the default.\n"
	       "All arguments are validated before opening or changing the channel.\n"
	       "\n"
	       "  -s, --shm PATH          Channel path (default: nonempty $DLSSNR_SHM, otherwise\n"
	       "                         /tmp/dlsslop-amd-UID/" kShmChannelName "; effective default: %s)\n"
	       "  -S, --status            Print live transport/status values "
	       "(default: on if no action/setting)\n"
	       "  -l, --settings          Print live settings and reset defaults (default: off)\n"
	       "  -r, --reset             Restore supported settings before explicit changes (default: off)\n"
	       "  -q, --quit              Request worker/layer stop (default: off; initial quit flag: 0)\n"
	       "  -R, --resume            Clear stop request; does not restart the worker (default: off)\n"
	       "  -c, --capture N         Request %g..%g matched before/after frames; "
	       "0 clears a pending request\n"
	       "                         (default: unchanged; initial: 0; active captures continue)\n"
	       "  -A, --toggle NAME       Toggle a boolean setting once, "
	       "such as enabled from a key binding;\n"
	       "                         repeat for different settings (default: none)\n"
	       "  -h, --help              Print help without accessing the channel (default: off)\n"
	       "\nSettings (all require a value):\n", channel, CAPTURE.minimum, CAPTURE.maximum);
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
		struct control_setting const *const s = &CONTROL_SETTINGS[i];
		printf("  -%c, --%-18s VALUE  %s\n"
		       "                                Range: %g..%g; default: %g",
		       s->short_name, s->name, s->help, s->minimum, s->maximum,
		       control_setting_value(s, defaults[i]));
		print_steps(stdout, s);
		putchar('\n');
	}
	fputs("\n"
	      "Actions run after parsing: reset, explicit settings, toggle, stop/resume, capture;\n"
	      "requested status/settings are printed last. Repeated settings use the last value.\n"
	      "--quit and --resume conflict. A setting cannot be both assigned and toggled.\n"
	      "Changes create a missing channel with the defaults, and its private (0700)\n"
	      "directory; reads and a lone --quit never create one. A file of another protocol\n"
	      "is refused. Settings persist in the shared-memory channel, not a configuration\n"
	      "file; worker startup may reapply its own settings.\n"
	      "\n"
	      "dlsslopd supports SDR, HDR and successive neural passes; --passes changes the next\n"
	      "frame's inference chain. Each added pass adds another network evaluation.\n"
	      "On Vulkan, intensity, local tone/structure, style, skin structure and the automatic\n"
	      "mask are the model's own controls. On HIP, intensity and local tone/structure are\n"
	      "residual operations after each network pass, and style, skin structure and the\n"
	      "automatic mask keep their defaults. Both sharpen and preserve color after each\n"
	      "pass. --preset is fixed.\n"
	      "White-point controls require linear-light input.\n"
	      "--hdr-mode changes proxy precision; forcing 16-bit does not reinterpret SDR as HDR.\n"
	      "HDR normalization follows --color-mode even with an 8-bit proxy.\n"
	      "Motion estimates come from the presented frames, not engine motion vectors.\n"
	      "--tier rebuilds the daemon's network for another raster, which takes under a second;\n"
	      "the game presents its own frames meanwhile. Setting the active tier does nothing.\n"
	      "Layer environment overrides take precedence over --ratio-smooth\n"
	      "(DLSSNR_RATIO_SMOOTH), --color-trust (DLSSNR_COLOUR_TRUST)\n"
	      "and --toggle-key (DLSSNR_TOGGLE_KEY).\n"
	      "\n"
	      "Examples:\n"
	      "  dlsslopctl --working-scale 1 --transfer 2\n"
	      "  dlsslopctl --passes 2\n"
	      "  dlsslopctl -e 0\n"
	      "  dlsslopctl --tier 1080\n"
	      "  dlsslopctl --reset --settings\n", stdout);
}

/** @brief Parses a setting's value.
 *
 * strtod() and strtoull() skip leading whitespace, strtoull() accepts signs and strtod() parses
 * nothing from an empty string. Reject all of them rather than silently coercing malformed
 * arguments. Both report a number outside their type's range, too large or, for strtod(), too
 * close to zero, only through errno, so errno is cleared before each call as their documentation
 * requires, and such a number is rejected.
 *
 * @param text    The argument.
 * @param setting The setting.
 * @param result  Receives the value's bits if @a setting admits the value.
 * @return        true if it does.
 */
static bool
parse_value (char const                   *text,
             struct control_setting const *setting,
             uint32_t                     *result)
{
	char *end;
	if (setting->is_float) {
		if (!*text || isspace((unsigned char)*text))
			return false;
		errno = 0;
		double const value = strtod(text, &end);
		if (errno || *end || !control_setting_in_range(setting, value))
			return false;
		// + 0.0f stores -0 as 0.
		*result = FloatToBits((float)value + 0.0f);
		return true;
	}
	if (*text < '0' || *text > '9')
		return false;
	errno = 0;
	unsigned long long const value = strtoull(text, &end, 10);
	if (errno || *end || !control_setting_in_range(setting, (double)value))
		return false;
	*result = (uint32_t)value;
	return true;
}

/** @brief The setting that a long option names.
 *
 * @param name The option, without its dashes.
 * @return     The setting, or nullptr if no setting has that name.
 */
static struct control_setting const *
find_setting (char const *name)
{
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i)
		if (!strcmp(name, CONTROL_SETTINGS[i].name))
			return &CONTROL_SETTINGS[i];
	return nullptr;
}

/** @brief The setting that a short option names.
 *
 * @param code The option, as getopt_long() returns it.
 * @return     The setting, or nullptr if no setting has that short option.
 */
static struct control_setting const *
find_short (int code)
{
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i)
		if (CONTROL_SETTINGS[i].short_name == code)
			return &CONTROL_SETTINGS[i];
	return nullptr;
}

/** @brief Parses a command line, and says what is wrong with one that does not parse.
 *
 * @param argc    The number of arguments.
 * @param argv    The arguments.
 * @param options Receives the command line; zeroed by the caller.
 * @return        true if the command line parsed.
 */
static bool
parse_options (int              argc,
               char           **argv,
               struct options  *options)
{
	struct option long_options[FIXED_COUNT + CONTROL_SETTING_COUNT + 1];
	char short_options[sizeof FIXED_SHORT + 2 * CONTROL_SETTING_COUNT];
	memcpy(long_options, FIXED_OPTIONS, sizeof FIXED_OPTIONS);
	char *next = mempcpy(short_options, FIXED_SHORT, sizeof FIXED_SHORT - 1);
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
		struct control_setting const *const s = &CONTROL_SETTINGS[i];
		long_options[FIXED_COUNT + i] = (struct option){
			.name = s->name, .has_arg = required_argument, .val = s->short_name
		};
		*next++ = s->short_name;
		*next++ = ':';
	}
	long_options[FIXED_COUNT + CONTROL_SETTING_COUNT] = (struct option){0};
	*next = '\0';

	for (int code; (code = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1;) {
		switch (code) {
		case 's':
			if (!*optarg) {
				fprintf(stderr, "--shm requires a nonempty path\n");
				return false;
			}
			options->path = optarg;
			break;
		case 'S':
			options->flags |= OPTIONS_STATUS;
			break;
		case 'l':
			options->flags |= OPTIONS_SETTINGS;
			break;
		case 'r':
			options->flags |= OPTIONS_RESET;
			break;
		case 'q':
			options->flags |= OPTIONS_QUIT;
			break;
		case 'R':
			options->flags |= OPTIONS_RESUME;
			break;
		case 'A': {
			// Repeating a toggle names the same action; it does not cancel itself out.
			struct control_setting const *const s = find_setting(optarg);
			if (!s || s->is_float || s->minimum != 0 || s->maximum != 1) {
				fprintf(stderr, "--toggle requires a boolean setting name, got '%s'\n",
				        optarg);
				return false;
			}
			options->toggled |= setting_bit(s);
			break;
		}
		case 'h':
			options->flags |= OPTIONS_HELP;
			break;
		case 'c':
			if (!parse_value(optarg, &CAPTURE, &options->capture)) {
				fprintf(stderr, "--capture: expected an integer from %g through %g, "
				        "got '%s'\n", CAPTURE.minimum, CAPTURE.maximum, optarg);
				return false;
			}
			options->flags |= OPTIONS_CAPTURE;
			break;
		default: {
			// A setting's short name, or '?' after getopt has named the bad argument.
			struct control_setting const *const s = find_short(code);
			if (!s)
				return false;
			if (!parse_value(optarg, s, &options->values[s - CONTROL_SETTINGS])) {
				fprintf(stderr, "--%s: expected a finite %s in [%g, %g]", s->name,
				        s->is_float ? "number" : "integer", s->minimum, s->maximum);
				print_steps(stderr, s);
				fprintf(stderr, ", got '%s'\n", optarg);
				return false;
			}
			options->assigned |= setting_bit(s);
			break;
		}
		}
	}
	if (optind < argc) {
		fprintf(stderr, "unexpected positional argument '%s'; use options such as --status or "
		        "--working-scale 1\n", argv[optind]);
		return false;
	}
	if ((options->flags & (OPTIONS_QUIT | OPTIONS_RESUME)) == (OPTIONS_QUIT | OPTIONS_RESUME)) {
		fprintf(stderr, "--quit and --resume cannot be combined\n");
		return false;
	}
	uint64_t const both = options->assigned & options->toggled;
	if (both) {
		char const *const name = CONTROL_SETTINGS[__builtin_ctzll(both)].name;
		fprintf(stderr, "--toggle %s and --%s cannot be combined\n", name, name);
		return false;
	}
	if (!options_changes_header(options) && !(options->flags & OPTIONS_SETTINGS))
		options->flags |= OPTIONS_STATUS;
	return true;
}

/** @brief The native tools' default channel, as ShmNativeDefaultPath() writes it.
 *
 * @param p_length Receives the path's length.
 * @return         The path, which the caller frees, or nullptr if it cannot be made.
 */
static char *
native_default_path (size_t *p_length)
{
	int const length = ShmNativeDefaultPath(nullptr, 0);
	if (length < 0)
		return nullptr;
	size_t const size = (size_t)length + 1;
	char *path = malloc(size);
	if (path && ShmNativeDefaultPath(path, size) != length) {
		free(path);
		path = nullptr;
	}
	*p_length = size - 1;
	return path;
}

/** @brief The native tools' channel, as ShmNativeChannelPath() names it: DLSSNR_SHM if it is set
 *         and not empty, otherwise ShmNativeDefaultPath().
 *
 * Says so on stderr if the path cannot be made.
 *
 * @param p_length Receives the path's length.
 * @return         The path, which the caller frees, or nullptr if it cannot be made.
 */
static char *
native_channel_path (size_t *p_length)
{
	char const *const env = getenv("DLSSNR_SHM");
	char *path;
	if (env && *env) {
		size_t const length = strlen(env);
		path = malloc(length + 1);
		if (path)
			memcpy(path, env, length + 1);
		*p_length = length;
	} else {
		path = native_default_path(p_length);
	}
	if (!path)
		fprintf(stderr, "cannot make the native channel's path\n");
	return path;
}

/** @brief Prints the live settings and their defaults.
 *
 * @param h The header.
 */
static void
print_settings (struct ShmHeader const *h)
{
	uint32_t defaults[CONTROL_SETTING_COUNT];
	control_settings_defaults(defaults, control_settings_worker_bypass(h));
	printf("# Live values; defaults are the initial/reset values for this worker mode.\n");
	// Nine digits round-trip every binary32 and ten print every uint32 exactly.
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
		struct control_setting const *const s = &CONTROL_SETTINGS[i];
		int const digits = s->is_float ? 9 : 10;
		printf("%s=%.*g default=%.*g\n", s->name,
		       digits, control_setting_value(s, control_setting_load(h, s)),
		       digits, control_setting_value(s, defaults[i]));
	}
}

/** @brief Prints the live transport and status values.
 *
 * @param h The header.
 */
static void
print_status (struct ShmHeader const *h)
{
	// shm_channel_open() refuses any other magic and version.
	printf("initialised=1\nmagic=%#x\nversion=%u\n", atomic_load(&h->magic), atomic_load(&h->version));
	printf("seq_req=%u\nseq_resp=%u\n", atomic_load(&h->seq_req), atomic_load(&h->seq_resp));
	printf("width=%u\nheight=%u\n", atomic_load(&h->width), atomic_load(&h->height));
	printf("neural_max_width=%u\nneural_max_height=%u\n", atomic_load(&h->nativeModelMaxWidth),
	       atomic_load(&h->nativeModelMaxHeight));
	printf("passes=%u\npass_ceiling=%u\n", atomic_load(&h->passes), ShmPassCeiling(h));
	printf("display_width=%u\ndisplay_height=%u\n", atomic_load(&h->layerWidth),
	       atomic_load(&h->layerHeight));
	printf("quit=%u\nheartbeat=%u\ncontrol_seq=%u\ntuning_seq=%u\n", atomic_load(&h->quit),
	       atomic_load(&h->heartbeat), atomic_load(&h->controlSeq), atomic_load(&h->tuningSeq));
	printf("helper_state=%u\nmodel_up=%u\nhelper_frames=%" PRIu64 "\n", atomic_load(&h->helperState),
	       atomic_load(&h->modelUp), atomic_load(&h->helperFrames));
	printf("upload_ms=%.3f\nnetwork_ms=%.3f\nreadback_ms=%.3f\n",
	       BitsToFloat(atomic_load(&h->helperUploadMsBits)),
	       BitsToFloat(atomic_load(&h->helperEvalMsBits)),
	       BitsToFloat(atomic_load(&h->helperReadbackMsBits)));
	printf("layer_pid=%u\nlayer_composition_up=%u\nlayer_frames=%" PRIu64 "\nlayer_ms=%.2f\n",
	       atomic_load(&h->layerPid), atomic_load(&h->layerCompositionUp),
	       atomic_load(&h->layerFrames), BitsToFloat(atomic_load(&h->layerMsBits)));
	printf("measured_white_point=%g\n", BitsToFloat(atomic_load(&h->layerMeasuredWhiteBits)));
	printf("hdr_mode=%u\nhdr_detected=%u\nhdr_active=%u\nproxy_format=%u\nhdr_encode=%u\n",
	       atomic_load(&h->hdrMode), atomic_load(&h->hdrDetected), atomic_load(&h->hdrActive),
	       atomic_load(&h->proxyFormat), atomic_load(&h->hdrEncode));
	char reason[kReasonBytes];
	ShmLoadString(h, SHM_TEXT_HELPER_REASON, reason, sizeof reason);
	if (*reason)
		printf("helper_reason=%s\n", reason);
	ShmLoadString(h, SHM_TEXT_LAYER_REASON, reason, sizeof reason);
	if (*reason)
		printf("layer_reason=%s\n", reason);
}

/** @brief Carries out a parsed command line on a channel.
 *
 * Changes create a missing channel (shm_channel_open()); reads and a quit alone do not.
 *
 * @param options The command line, which does not ask for help.
 * @param path    The channel.
 * @param length  The length of its path.
 * @return        The exit status: 0, or 1 if the channel cannot be used.
 */
static int
run (struct options const *options,
     char const           *path,
     size_t                length)
{
	bool const writable = options_changes_header(options);
	// A quit has no worker or layer to stop on a missing channel.
	bool const create = writable && !(options->flags & OPTIONS_QUIT);
	uint32_t const flags = (writable ? SHM_CHANNEL_WRITE : 0) | (create ? SHM_CHANNEL_CREATE : 0);
	struct shm_channel channel;
	struct error e;
	if (shm_channel_open(&channel, path, length, kHeaderBytes, flags, &e)) {
		bool const missing = channel.flags & SHM_CHANNEL_MISSING;
		// An absent channel has no worker or layer for a quit alone to stop.
		if (missing && options->flags == OPTIONS_QUIT && !options->assigned && !options->toggled)
			return 0;
		fprintf(stderr, "%s%s\n", e.what, missing ? "; dlsslopd, the layer and setting options create it"
		        : "");
		return 1;
	}
	struct ShmHeader *const h = channel.h;

	// Store each setting once: readers never wait on controlSeq, so a second store would show them a
	// value nobody asked for. Reset stores the controls alone, atomically, and never initialises the
	// live header: other processes use it.
	bool const reset = options->flags & OPTIONS_RESET;
	uint32_t defaults[CONTROL_SETTING_COUNT];
	if (reset)
		control_settings_defaults(defaults, control_settings_worker_bypass(h));
	bool tuning_changed = reset;
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
		uint64_t const bit = UINT64_C(1) << i;
		bool const flip = options->toggled & bit;
		struct control_setting const *const s = &CONTROL_SETTINGS[i];
		if (options->assigned & bit) {
			// Never also toggled.
			control_setting_store(h, s, options->values[i]);
			tuning_changed |= s->tuning;
		} else if (reset) {
			control_setting_store(h, s, defaults[i] ^ flip);
		} else if (flip) {
			control_setting_toggle(h, s);
		}
	}
	if (tuning_changed)
		atomic_fetch_add(&h->tuningSeq, 1);
	if (options->flags & OPTIONS_QUIT)
		atomic_store(&h->quit, 1);
	if (options->flags & OPTIONS_RESUME)
		atomic_store(&h->quit, 0);
	// Every change, a capture included, publishes a new control generation.
	uint32_t const published = writable ? atomic_fetch_add(&h->controlSeq, 1) + 1 : 0;
	// Publish capture only after its settings and generation are visible. The layer records this
	// generation with the matched frame pair.
	bool const capture = options->flags & OPTIONS_CAPTURE;
	if (capture)
		atomic_store_explicit(&h->captureRequest, options->capture, memory_order_release);
	if (options->flags & OPTIONS_STATUS) {
		print_status(h);
		if (capture)
			printf("capture_control_seq=%u\n", published);
	}
	if (options->flags & OPTIONS_SETTINGS)
		print_settings(h);
	shm_channel_fini(&channel);
	return 0;
}

int
main (int    argc,
      char **argv)
{
	struct options options = {0};
	if (!parse_options(argc, argv, &options)) {
		fprintf(stderr, "Try --help for options, ranges and defaults.\n");
		return 2;
	}
	// Only --help and a command without --shm need the native channel.
	int status = 1;
	char *channel = nullptr;
	size_t length;
	if (options.flags & OPTIONS_HELP) {
		channel = native_channel_path(&length);
		if (channel) {
			usage(channel);
			status = 0;
		}
	} else if (options.path) {
		status = run(&options, options.path, strlen(options.path));
	} else {
		channel = native_channel_path(&length);
		if (channel)
			status = run(&options, channel, length);
	}
	free(channel);
	channel = nullptr;
	// Output that did not reach its reader is a failure, not a success with less output.
	if (fflush(stdout) || ferror(stdout)) {
		fprintf(stderr, "cannot write the output\n");
		return 1;
	}
	return status;
}
