/** @file
 *
 * Native Linux controls. Shared-memory offsets and defaults come from the protocol.
 */
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "control_settings.h"
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
	struct ShmHeader defaults;
	ShmInitNativeDefaults(&defaults, false);
	printf("Usage: dlsslopctl [OPTION]...\n"
	       "       dlssnr-shmctl [OPTION]...\n"
	       "\n"
	       "Omitted setting options leave current values unchanged. The setting defaults\n"
	       "below are initial/reset values, not the current values. Use --settings to\n"
	       "inspect both. With no action or setting option, --status is the default.\n"
	       "All arguments are validated before opening or changing the channel.\n"
	       "\n"
	       "  -s, --shm PATH          Channel path (default: nonempty $DLSSNR_SHM, otherwise\n"
	       "                         /tmp/dlsslop-amd-UID/shm.bin; effective default: %s)\n"
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
		       control_setting_value(s, control_setting_load(&defaults, s)));
		print_steps(stdout, s);
		putchar('\n');
	}
	fputs("\n"
	      "Actions run after parsing: reset, explicit settings, toggle, stop/resume, capture;\n"
	      "requested status/settings are printed last. Repeated settings use the last value.\n"
	      "--quit and --resume conflict. A setting cannot be both assigned and toggled.\n"
	      "Changes require an existing channel parent directory; the default private\n"
	      "directory is created automatically. Settings persist in the shared-memory\n"
	      "channel, not a configuration file; worker startup may reapply its own settings.\n"
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
 * @return The path, which the caller frees, or nullptr if it cannot be made.
 */
static char *
native_channel_path (void)
{
	char const *const env = getenv("DLSSNR_SHM");
	size_t length;
	char *const path = env && *env ? strdup(env) : native_default_path(&length);
	if (!path)
		fprintf(stderr, "cannot make the native channel's path\n");
	return path;
}

/** @brief Creates the native default channel's private directory, if a path names that channel.
 *
 * Says on stderr why the directory could not be created.
 *
 * @param path The channel.
 * @return     true if @a path names another channel, or if the directory now exists, is this
 *             user's and grants nothing to anyone else.
 */
static bool
create_default_directory (char const *path)
{
	size_t length;
	char *directory = native_default_path(&length);
	if (!directory) {
		fprintf(stderr, "cannot make the native channel's path\n");
		return false;
	}
	int error = 0;
	if (!strcmp(path, directory)) {
		// The directory: the path up to its last slash.
		char *const slash = memrchr(directory, '/', length);
		if (slash)
			*slash = '\0';
		struct stat st;
		if (mkdir(directory, 0700) && errno != EEXIST)
			error = errno;
		else if (lstat(directory, &st))
			error = errno;
		else if (!S_ISDIR(st.st_mode) || st.st_uid != getuid() || (st.st_mode & (S_IRWXG | S_IRWXO)))
			error = EACCES;
	}
	free(directory);
	directory = nullptr;
	if (error)
		fprintf(stderr, "cannot create private channel directory: %s\n", strerror(error));
	return !error;
}

/** @brief Maps the header of the channel that a descriptor holds.
 *
 * @param fd       The channel's descriptor.
 * @param path     The channel, for the message about a file that is not one.
 * @param create   Whether to grow a file that is shorter than a channel.
 * @param writable Whether to map the header for writing; the file must then hold a channel or
 *                 nothing but zeros.
 * @param p_header Receives the header.
 * @return         0, or an errno value.
 */
static int
map_descriptor (int                fd,
                char const        *path,
                bool               create,
                bool               writable,
                struct ShmHeader **p_header)
{
	struct stat st;
	if (fstat(fd, &st))
		return errno;
	if (!S_ISREG(st.st_mode) || st.st_uid != getuid())
		return EACCES;
	if (writable) {
		// Never chmod, grow or initialise a file that is not a channel. A new or grown file reads as
		// zero, and initialisation writes the magic last: a header that is not zero and has no magic
		// is not a channel, or one whose initialisation has not finished.
		uint32_t head[sizeof (struct ShmHeader) / sizeof (uint32_t)] = {0};
		if (pread(fd, head, sizeof head, 0) < 0)
			return errno;
		uint32_t bits = 0;
		for (size_t i = 0; i < sizeof head / sizeof *head; ++i)
			bits |= head[i];
		if (head[0] != kShmMagic && bits) {
			fprintf(stderr, "'%s' is not a dlsslop channel, or one whose initialisation has not finished; "
			        "refusing to modify it\n", path);
			return EINVAL;
		}
		if (fchmod(fd, 0600))
			return errno;
	}
	off_t const total = (off_t)ShmTotalBytes();
	if (st.st_size < total) {
		if (!create)
			return EINVAL;
		if (ftruncate(fd, total))
			return errno;
	}
	void *const mapping = mmap(nullptr, kHeaderBytes, PROT_READ | (writable ? PROT_WRITE : 0), MAP_SHARED,
	                           fd, 0);
	if (mapping == MAP_FAILED)
		return errno;
	*p_header = mapping;
	return 0;
}

/** @brief Maps a channel's header.
 *
 * Readers never create, resize, initialise or write the channel. The mapping outlives the
 * descriptor and lasts until the process exits.
 *
 * @param path     The channel.
 * @param create   Whether to create the file if it is missing, and grow it if it is shorter than a
 *                 channel.
 * @param writable Whether to map the header for writing.
 * @param p_header Receives the header, or nullptr if it is not mapped.
 * @return         0, or an errno value.
 */
static int
map_header (char const        *path,
            bool               create,
            bool               writable,
            struct ShmHeader **p_header)
{
	*p_header = nullptr;
	int fd = open(path, (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0),
	              0600);
	if (fd < 0)
		return errno;
	int const error = map_descriptor(fd, path, create, writable, p_header);
	// The descriptor wrote no data, so its close has no write error to report.
	close(fd);
	fd = -1;
	return error;
}

/** @brief Makes sure that a header holds a channel of this protocol version.
 *
 * Writers initialise a new channel and, like the worker and the layer, re-initialise one left by
 * another protocol version. A process still on that version misreads the new layout and
 * re-initialises it back when it next attaches, so say so.
 *
 * @param h      The header.
 * @param create Whether to initialise a header that does not hold one.
 * @param path   The channel, for the messages.
 * @return       true if the header holds a channel of this version now.
 */
static bool
attach (struct ShmHeader *h,
        bool              create,
        char const       *path)
{
	// Each word read once: another process may be rewriting them.
	uint32_t const magic = atomic_load(&h->magic);
	uint32_t const version = atomic_load(&h->version);
	bool const channel = magic == kShmMagic;
	if (channel && version == kShmVersion)
		return true;
	if (!create) {
		fprintf(stderr, "channel '%s' is uninitialised or incompatible: magic %#x version %u, "
		        "expected %#x version %u\n", path, magic, version, kShmMagic, kShmVersion);
		return false;
	}
	if (channel)
		fprintf(stderr, "channel '%s' held protocol v%u and is re-initialised as v%u; restart any "
		        "dlsslopd, game or GUI still using it\n", path, version, kShmVersion);
	ShmInitNativeDefaults(h, false);
	return true;
}

/** @brief Prints the live settings and their defaults.
 *
 * @param h The header.
 */
static void
print_settings (struct ShmHeader const *h)
{
	struct ShmHeader defaults;
	ShmInitNativeDefaults(&defaults, control_settings_worker_bypass(h));
	printf("# Live values; defaults are the initial/reset values for this worker mode.\n");
	// Nine digits round-trip every binary32 and ten print every uint32 exactly.
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
		struct control_setting const *const s = &CONTROL_SETTINGS[i];
		int const digits = s->is_float ? 9 : 10;
		printf("%s=%.*g default=%.*g\n", s->name,
		       digits, control_setting_value(s, control_setting_load(h, s)),
		       digits, control_setting_value(s, control_setting_load(&defaults, s)));
	}
}

/** @brief Prints the live transport and status values.
 *
 * @param h The header.
 */
static void
print_status (struct ShmHeader const *h)
{
	// Each word read once: another process may be rewriting them.
	uint32_t const magic = atomic_load(&h->magic);
	uint32_t const version = atomic_load(&h->version);
	bool const initialised = magic == kShmMagic && version == kShmVersion;
	printf("initialised=%d\nmagic=%#x\nversion=%u\n", initialised, magic, version);
	if (!initialised)
		return;
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
 * @param options The command line, which does not ask for help.
 * @param path    The channel.
 * @return        The exit status: 0, or 1 if the channel cannot be used.
 */
static int
run (struct options const *options,
     char const           *path)
{
	bool const writable = options_changes_header(options);
	bool const create = writable && !(options->flags & OPTIONS_QUIT);
	if (create && !create_default_directory(path))
		return 1;
	struct ShmHeader *h;
	int const error = map_header(path, create, writable, &h);
	if (error) {
		// An absent channel has no worker or layer for a quit alone to stop.
		if (error == ENOENT && options->flags == OPTIONS_QUIT && !options->assigned
		    && !options->toggled)
			return 0;
		fprintf(stderr, "cannot access channel '%s': %s\n", path, strerror(error));
		return 1;
	}
	if (!attach(h, create, path))
		return 1;

	// Store each setting once: readers never wait on controlSeq, so a second store would show them a
	// value nobody asked for. Reset copies controls only and never memsets a live transport header.
	bool const reset = options->flags & OPTIONS_RESET;
	struct ShmHeader defaults;
	if (reset)
		ShmInitNativeDefaults(&defaults, control_settings_worker_bypass(h));
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
			control_setting_store(h, s, control_setting_load(&defaults, s) ^ flip);
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
	if (options.flags & OPTIONS_HELP) {
		channel = native_channel_path();
		if (channel) {
			usage(channel);
			status = 0;
		}
	} else if (options.path) {
		status = run(&options, options.path);
	} else {
		channel = native_channel_path();
		if (channel)
			status = run(&options, channel);
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
