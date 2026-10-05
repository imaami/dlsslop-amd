/** @file
 *
 * dlsslopd's command line and config file: options.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "files.h"
#include "network_recorder.h"
#include "options.h"
#include "paths.h"
#include "shm_protocol.h"

/** @brief How an option takes its value. */
enum spec_kind : uint32_t {
	SPEC_FLAG,  //!< None: "true" from the command line, true or false from the config file.
	SPEC_VALUE, //!< A value.
	SPEC_PATH,  //!< A path, which in the config file is absolute or starts with ~/.
};

/** @brief An option: its getopt spelling, how it takes its value, and whether the config file may
 *         set it. apply() applies a value by its letter.
 */
struct spec {
	char const     *name;   //!< The long option, and the setting's name in the config file.
	enum spec_kind  kind;   //!< How it takes its value.
	uint16_t        length; //!< The name's length.
	char            letter; //!< The short option.
	bool            config; //!< The config file may set it.
};

/** @brief A struct spec of a string literal NAME. */
#define SPEC(name, letter, kind, config) {name, SPEC_##kind, sizeof name - 1, letter, config}

/** @brief Every option, once. */
static struct spec const SPECS[] = {
	SPEC("config",          'f', PATH,  false), // Read before the others apply.
	SPEC("backend",         'b', VALUE, true),
	SPEC("vulkan-model",    'M', PATH,  true),
	SPEC("assets",          'a', PATH,  true),
	SPEC("modules",         'm', PATH,  true),
	// The channel pairs the daemon with its launcher and socket unit.
	SPEC("shm",             's', PATH,  false),
	SPEC("tier",            't', VALUE, true),
	SPEC("passes",          'P', VALUE, true),
	SPEC("device",          'd', VALUE, true),
	SPEC("diagnose",        'D', FLAG,  false),
	SPEC("self-test",       'S', FLAG,  false),
	SPEC("self-test-runs",  'r', VALUE, false),
	SPEC("self-test-drops", 'X', VALUE, false),
	SPEC("input",           'i', VALUE, false),
	SPEC("output",          'o', VALUE, false),
	SPEC("width",           'W', VALUE, false),
	SPEC("height",          'H', VALUE, false),
	SPEC("cpu-codec",       'C', FLAG,  false),
	SPEC("performance",     'p', FLAG,  true),
	SPEC("once",            '1', FLAG,  false),
	SPEC("idle-exit",       'x', VALUE, true),
	SPEC("trace-dir",       'R', VALUE, false),
	SPEC("test-identity",   'T', FLAG,  false),
	SPEC("help",            'h', FLAG,  false),
};

#undef SPEC

/** @brief The number of SPECS. */
static constexpr uint32_t SPEC_COUNT = sizeof SPECS / sizeof *SPECS;

static_assert(SPEC_COUNT <= 32, "a command line keeps one bit per option");

/** @brief The names of enum options_backend's values, as --backend takes them. */
static char const *const BACKENDS[] = {
	[OPTIONS_BACKEND_AUTO]   = "auto",
	[OPTIONS_BACKEND_VULKAN] = "vulkan",
	[OPTIONS_BACKEND_HIP]    = "hip",
};

/** @brief The number of BACKENDS. */
static constexpr uint8_t BACKEND_COUNT = sizeof BACKENDS / sizeof *BACKENDS;

/** @brief The option of a short option's letter.
 *
 * @param letter The letter, as getopt_long() returns it.
 * @return       The option, or nullptr for none.
 */
static struct spec const *
find_letter (int letter)
{
	for (uint32_t i = 0; i < SPEC_COUNT; ++i)
		if (SPECS[i].letter == letter)
			return &SPECS[i];
	return nullptr;
}

/** @brief The config file's setting of a name.
 *
 * @param name   The name; it need not be null-terminated.
 * @param length Its length.
 * @return       The option of that name that the config file may set, or nullptr for none.
 */
static struct spec const *
find_setting (char const *name,
              size_t      length)
{
	for (uint32_t i = 0; i < SPEC_COUNT; ++i) {
		struct spec const *const spec = &SPECS[i];
		if (spec->config && spec->length == length && !memcmp(spec->name, name, length))
			return spec;
	}
	return nullptr;
}

/** @brief An option's bit in a command line's mask of the options it gives.
 *
 * @param spec The option.
 * @return     Its bit.
 */
static uint32_t
spec_bit (struct spec const *spec)
{
	return UINT32_C(1) << (spec - SPECS);
}

/** @brief Whether a command line gave an option.
 *
 * @param given  The command line's mask of the options it gives.
 * @param letter The option's letter.
 * @return       true if it did.
 */
static bool
on_command_line (uint32_t given,
                 char     letter)
{
	return given & spec_bit(find_letter(letter));
}

/** @brief The words "on" or "off".
 *
 * @param value A flag.
 * @return      "on" if it is set, otherwise "off".
 */
static char const *
on_off (bool value)
{
	return value ? "on" : "off";
}

/** @brief Prints the help.
 *
 * @param out      Where.
 * @param defaults The options' defaults.
 * @param config   The default config file, or nullptr for none.
 * @param native   The native tools' default channel.
 */
static void
print_usage (FILE                 *out,
             struct options const *defaults,
             char const           *config,
             char const           *native)
{
	fprintf(out,
	        "Usage: dlsslopd [OPTIONS]\n"
	        "  -f, --config FILE       Settings file of NAME = VALUE lines\n"
	        "                          Default: %s\n"
	        "                          Uses nonempty XDG_CONFIG_HOME/dlsslop-amd/dlsslopd.conf,\n"
	        "                          otherwise ~/.config/dlsslop-amd/dlsslopd.conf;\n"
	        "                          a missing default file is skipped. NAME is one of\n"
	        "                          ",
	        config ? config : "unset; no home directory");
	char const *separator = "";
	for (uint32_t i = 0; i < SPEC_COUNT; ++i) {
		if (SPECS[i].config) {
			fprintf(out, "%s%s", separator, SPECS[i].name);
			separator = ", ";
		}
	}
	fprintf(out,
	        "\n"
	        "                          (performance = true or false); # starts a\n"
	        "                          comment line; paths are absolute or start with\n"
	        "                          ~/. Settings replace the defaults below and\n"
	        "                          options override them\n"
	        "  -b, --backend NAME      Where the network runs: vulkan, hip or auto\n"
	        "                          Default: %s; auto takes Vulkan when its model is\n"
	        "                          installed and a device supports it, else HIP.\n"
	        "                          Only HIP serves --cpu-codec and --trace-dir:\n"
	        "                          auto takes HIP for them, vulkan refuses them.\n"
	        "                          Vulkan evaluates NVIDIA's own intensity, local\n"
	        "                          tone, local structure, style, skin structure and\n"
	        "                          automatic mask controls and sizes itself to each\n"
	        "                          frame. Its SPIR-V is in\n"
	        "                          %s\n"
	        "  -M, --vulkan-model FILE The Vulkan network's model (dlssnr.bin)\n"
	        "                          Default: %s\n"
	        "                          Uses nonempty XDG_DATA_HOME/dlsslop-amd/dlssnr.bin,\n"
	        "                          otherwise ~/.local/share/dlsslop-amd/dlssnr.bin;\n"
	        "                          dlsslop-setup --dll extracts it from your own\n"
	        "                          nvngx_dlssnr.dll 310.8.0\n"
	        "  -a, --assets DIR        HIP model weights (.f16/.f32)\n"
	        "                          Default: %s\n"
	        "                          Uses nonempty XDG_DATA_HOME/dlsslop-amd/model,\n"
	        "                          otherwise ~/.local/share/dlsslop-amd/model\n"
	        "  -m, --modules DIR       Native gfx1201 HIP modules (.hsaco)\n"
	        "                          Default: %s\n"
	        "                          Uses nonempty DLSSLOP_MODULES; otherwise\n"
	        "                          <executable prefix>/share/dlsslop-amd/HIP/gfx1201\n"
	        "                          (source build fallback: ../assets/HIP/gfx1201)\n"
	        "  -s, --shm FILE          Layer transport in a private (0700) directory\n"
	        "                          Default: %s\n"
	        "                          From nonempty DLSSNR_SHM, otherwise %s\n"
	        "  -t, --tier HEIGHT       Neural work raster: 720, 900, or 1080\n"
	        "                          Default: %u on a new channel; game/display\n"
	        "                          resolution unchanged. Unset here and in FILE, a\n"
	        "                          starting worker keeps the channel's live tier,\n"
	        "                          which dlsslopctl --tier changes while it runs\n"
	        "  -P, --passes N          Chained neural evaluations per frame (1..%u)\n"
	        "                          Default: %u on a new channel; each pass consumes\n"
	        "                          the previous output. Unset here and in FILE, a\n"
	        "                          starting worker keeps the channel's live count.\n"
	        "                          Vulkan runs at most %u and stores that count\n"
	        "  -d, --device INDEX      HIP device, or with Vulkan physical device, index\n"
	        "                          Default: ",
	        BACKENDS[defaults->backend], defaults->shaders,
	        defaults->vulkan_model_length ? defaults->vulkan_model : "unset; no home directory",
	        defaults->assets_length ? defaults->assets : "unset; required for inference",
	        defaults->modules_length ? defaults->modules : "unset; required for inference",
	        defaults->shm, native, kNativeDefaultTier, kMaxPasses, kNativeDefaultPasses,
	        NETWORK_RECORDER_MAX_PASSES);
	// options_init()'s device of -1 is auto.
	if (defaults->device < 0)
		fputs("auto, the first device that can run\n"
		      "                          the network (HIP: gfx1201)\n", out);
	else
		fprintf(out, "%d\n", defaults->device);
	fprintf(out,
	        "  -D, --diagnose          Open the backend serving would use, report its\n"
	        "                          device and exit; HIP also lists its devices\n"
	        "                          Default: %s; exit status 1 if none is usable\n"
	        "  -S, --self-test         Real model test on a deterministic gradient\n"
	        "                          Also checks tuning, motion history and FP16 codec\n"
	        "                          Default: %s\n"
	        "  -r, --self-test-runs N  Identical-input runs, including baseline (2..1000)\n"
	        "                          Default: %u; requires --self-test\n"
	        "  -X, --self-test-drops N Most Vulkan runs to drop (0..999)\n"
	        "                          Default: %u; requires --self-test. A run is\n"
	        "                          dropped when a wait of the network ran out\n"
	        "                          while other GPU work held the device\n"
	        "  -i, --input FILE        Offline tightly packed RGBA8 input\n"
	        "                          Default: unset; serve shared-memory requests\n"
	        "  -o, --output FILE       Offline RGBA8 output; self-test PPM output\n"
	        "                          Default: unset; required with --input;\n"
	        "                          --self-test writes no image unless specified\n"
	        "  -W, --width PIXELS      Offline image width (1..%u)\n"
	        "                          Default: %u (unset); required with --input\n"
	        "  -H, --height PIXELS     Offline image height (1..%u)\n"
	        "                          Default: %u (unset); required with --input\n"
	        "  -C, --cpu-codec         Slow CPU codec for numerical comparison\n"
	        "                          Default: %s; use the GPU codec\n"
	        "  -p, --performance       HIP: skip blocks 42,43,46, matching upstream preset\n"
	        "                          Default: %s; evaluate all 71 blocks\n"
	        "  -1, --once              Answer one shared-memory request and exit,\n"
	        "                          with status 1 when that request failed\n"
	        "                          Default: %s; run until stopped\n"
	        "  -x, --idle-exit SECONDS Stop serving after SECONDS without a request\n"
	        "                          Default: %u (never)\n"
	        "  -R, --trace-dir DIR     Opt-in real-frame RGB float32 diagnostics\n"
	        "                          Default: disabled; no readbacks or file checks\n"
	        "                          DIR is created private (0700) or must be so\n"
	        "                          To trace one frame, write a unique TOKEN (1-64\n"
	        "                          of A-Z a-z 0-9 _ -, not 'request') to\n"
	        "                          DIR/TOKEN.tmp, then ln it to DIR/request;\n"
	        "                          DIR/TOKEN.done marks DIR/TOKEN/summary.json\n"
	        "                          complete. PFM files hold each pass's input and\n"
	        "                          raw output, plus -tuned/-color stages when\n"
	        "                          active; serving real inference only\n"
	        "  -T, --test-identity     DIAGNOSTIC ONLY: copy frames without inference\n"
	        "                          Default: %s; the backend's network runs\n"
	        "  -h, --help              Show this help and exit (default: off)\n",
	        on_off(defaults->diagnose), on_off(defaults->self_test), defaults->self_test_runs,
	        defaults->self_test_drops, kMaxW, defaults->width, kMaxH, defaults->height,
	        on_off(defaults->cpu_codec), on_off(defaults->performance), on_off(defaults->once),
	        defaults->idle_exit, on_off(defaults->test_identity));
}

/** @brief Prints the help, with the defaults that options_init() makes.
 *
 * @param out Where.
 * @param e   Receives the words for what stopped it, or nullptr.
 * @return    ERROR_NONE, or ERROR_FAILED with nothing printed: without memory, or when the native
 *            tools' default channel cannot be formatted.
 */
static enum error_code
usage (FILE         *out,
       struct error *e)
{
	// ShmNativeDefaultPath()'s path with the widest user ID.
	char native[sizeof "/tmp/dlsslop-amd-4294967295/shm.bin"];
	int const n = ShmNativeDefaultPath(native, sizeof native);
	if (n < 0 || n >= (int)sizeof native)
		return error_fail(e, "cannot format the native channel's path");

	struct paths_home const home = paths_home();
	struct options defaults;
	enum error_code code = options_init(&defaults, &home, e);
	if (code)
		return code;
	char *config;
	size_t config_length;
	code = paths_default_config(&config, &config_length, &home, e);
	if (!code)
		print_usage(out, &defaults, config, native);
	free(config);
	config = nullptr;
	options_fini(&defaults);
	return code;
}

/** @brief The channel's default: nonempty DLSSNR_SHM, otherwise the native tools' default.
 *
 * @param dest   Receives the path; nullptr on a failure.
 * @param length Receives its length.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
default_channel (char         **dest,
                 size_t        *length,
                 struct error  *e)
{
	*dest = nullptr;
	char const *const variable = getenv("DLSSNR_SHM");
	int const n = shm_native_channel_path(nullptr, 0, variable);
	if (n < 0)
		return error_fail(e, "cannot format the channel's path");
	size_t const size = (size_t)n + 1;
	char *path = malloc(size);
	if (!path)
		return error_fail(e, "out of memory");
	if (shm_native_channel_path(path, size, variable) != n) {
		free(path);
		path = nullptr;
		return error_fail(e, "cannot format the channel's path");
	}
	*dest = path;
	*length = size - 1;
	return ERROR_NONE;
}

/** @brief Makes the options' paths' defaults and the shaders' path.
 *
 * @param dest The options, whose paths are unset; receives the paths made, also on a failure.
 * @param home The home.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
default_paths (struct options          *dest,
               struct paths_home const *home,
               struct error            *e)
{
	enum error_code code = paths_default_assets(&dest->assets, &dest->assets_length, home, e);
	if (code)
		return code;
	code = paths_default_vulkan_model(&dest->vulkan_model, &dest->vulkan_model_length, home, e);
	if (code)
		return code;
	// The modules' default and the shaders are found from one read of the binary's path.
	char *binary;
	size_t binary_length;
	code = paths_executable(&binary, &binary_length, e);
	if (code)
		return code;
	code = paths_default_modules(&dest->modules, &dest->modules_length, binary, binary_length, e);
	if (!code)
		code = paths_vulkan_shaders(&dest->shaders, &dest->shaders_length, binary, binary_length, e);
	free(binary);
	binary = nullptr;
	if (code)
		return code;
	return default_channel(&dest->shm, &dest->shm_length, e);
}

enum error_code
options_init (struct options          *dest,
              struct paths_home const *home,
              struct error            *e)
{
	*dest = (struct options){
		.self_test_runs = 10,
		.device         = -1,
		.backend        = OPTIONS_BACKEND_AUTO,
	};
	enum error_code const code = default_paths(dest, home, e);
	if (code)
		options_fini(dest);
	return code;
}

void
options_fini (struct options *options)
{
	if (!options)
		return;
	free(options->assets);
	free(options->modules);
	free(options->shm);
	free(options->vulkan_model);
	free(options->shaders);
	free(options->input);
	free(options->output);
	free(options->trace_dir);
	*options = (struct options){};
}

char const *
options_hip_only (struct options const *options)
{
	if (options->cpu_codec)
		return "--cpu-codec";
	if (options->trace_dir_length)
		return "--trace-dir";
	return nullptr;
}

/** @brief Reads a number of an option, 0..100000.
 *
 * @param text The option's value.
 * @param name What the number is, for the words.
 * @param dest Receives the number; untouched on a failure.
 * @param e    Receives the words for what is wrong, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
number (char const   *text,
        char const   *name,
        uint32_t     *dest,
        struct error *e)
{
	char *end;
	// strtoul() reports a value out of its range only through errno.
	errno = 0;
	unsigned long const n = strtoul(text, &end, 10);
	if (errno || !*text || *end || *text == '-' || n > 100000)
		return error_fail(e, "invalid %s", name);
	*dest = (uint32_t)n;
	return ERROR_NONE;
}

/** @brief Replaces a string option's value.
 *
 * @param field  The option's string.
 * @param length The option's length.
 * @param value  The value.
 * @param home   For a config file's path, which is absolute or starts with ~/, the home that ~
 *               stands for; nullptr for any other value.
 * @param e      Receives the words for what is wrong, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
set_text (char                    **field,
          size_t                   *length,
          char const               *value,
          struct paths_home const  *home,
          struct error             *e)
{
	// A config file's path is never relative to wherever the daemon was started.
	size_t home_length = 0;
	if (home && *value != '/') {
		if (value[0] != '~' || value[1] != '/' || !home->path)
			return error_fail(e, "a path must be absolute or start with ~/");
		home_length = home->length;
		++value;
	}

	size_t const value_size = strlen(value) + 1;
	char *const text = malloc(home_length + value_size);
	if (!text)
		return error_fail(e, "out of memory");
	if (home_length)
		memcpy(text, home->path, home_length);
	memcpy(text + home_length, value, value_size);
	free(*field);
	*field = text;
	*length = home_length + value_size - 1;
	return ERROR_NONE;
}

/** @brief Applies an option's value.
 *
 * @param o      The options.
 * @param letter The option's letter.
 * @param value  Its value: "true" for a flag on the command line, "true" or "false" for one in the
 *               config file.
 * @param home   For a config file's path, which is absolute or starts with ~/, the home that ~
 *               stands for; nullptr for the command line's.
 * @param e      Receives the words for what is wrong, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
apply (struct options          *o,
       int                      letter,
       char const              *value,
       struct paths_home const *home,
       struct error            *e)
{
	switch (letter) {
	case 'b':
		for (uint8_t i = 0; i < BACKEND_COUNT; ++i) {
			if (!strcmp(value, BACKENDS[i])) {
				o->backend = i;
				return ERROR_NONE;
			}
		}
		return error_fail(e, "backend must be auto, vulkan or hip");
	case 'M':
		return set_text(&o->vulkan_model, &o->vulkan_model_length, value, home, e);
	case 'a':
		return set_text(&o->assets, &o->assets_length, value, home, e);
	case 'm':
		return set_text(&o->modules, &o->modules_length, value, home, e);
	case 's':
		return set_text(&o->shm, &o->shm_length, value, home, e);
	case 't': {
		enum error_code const code = number(value, "tier", &o->tier, e);
		if (code || ShmNativeTier(o->tier))
			return code;
		return error_fail(e, "tier must be 720, 900, or 1080");
	}
	case 'P': {
		enum error_code const code = number(value, "passes", &o->passes, e);
		if (code || (o->passes && o->passes <= kMaxPasses))
			return code;
		return error_fail(e, "--passes must be 1..%u", kMaxPasses);
	}
	case 'd': {
		uint32_t device;
		enum error_code const code = number(value, "device", &device, e);
		if (!code)
			o->device = (int)device;
		return code;
	}
	case 'D':
		o->diagnose = true;
		return ERROR_NONE;
	case 'S':
		o->self_test = true;
		return ERROR_NONE;
	case 'r':
		return number(value, "self-test runs", &o->self_test_runs, e);
	case 'X':
		return number(value, "self-test drops", &o->self_test_drops, e);
	case 'i':
		return set_text(&o->input, &o->input_length, value, nullptr, e);
	case 'o':
		return set_text(&o->output, &o->output_length, value, nullptr, e);
	case 'W':
		return number(value, "width", &o->width, e);
	case 'H':
		return number(value, "height", &o->height, e);
	case 'C':
		o->cpu_codec = true;
		return ERROR_NONE;
	case 'p':
		o->performance = !strcmp(value, "true");
		if (!o->performance && strcmp(value, "false"))
			return error_fail(e, "expected true or false");
		return ERROR_NONE;
	case '1':
		o->once = true;
		return ERROR_NONE;
	case 'x':
		return number(value, "idle-exit seconds", &o->idle_exit, e);
	case 'R':
		if (!*value)
			return error_fail(e, "--trace-dir requires a nonempty directory");
		return set_text(&o->trace_dir, &o->trace_dir_length, value, nullptr, e);
	case 'T':
		o->test_identity = true;
		return ERROR_NONE;
	default:
		// The config file and --help are no options to apply.
		return error_fail(e, "invalid arguments");
	}
}

/** @brief Whether a character is one that a config file's parts are trimmed of.
 *
 * @param c The character.
 * @return  true for a space or a tab.
 */
static bool
blank (char c)
{
	return c == ' ' || c == '\t';
}

/** @brief Trims blanks off both ends of a part of a config file.
 *
 * @param at     The part; receives where it starts without its blanks.
 * @param length Its length; receives its length without them.
 */
static void
trim (char   **at,
      size_t  *length)
{
	while (*length && blank(**at)) {
		++*at;
		--*length;
	}
	while (*length && blank((*at)[*length - 1]))
		--*length;
}

/** @brief Applies a config file's NAME = VALUE line, NAME a long option that the file may set.
 *
 * @param o      The options.
 * @param line   The line, trimmed; the byte after it is the line's end or a blank, which this may
 *               overwrite.
 * @param length Its length.
 * @param home   The home, for paths that start with ~/.
 * @param e      Receives the words for what is wrong, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
apply_setting (struct options          *o,
               char                    *line,
               size_t                   length,
               struct paths_home const *home,
               struct error            *e)
{
	char *const equals = memchr(line, '=', length);
	if (!equals)
		return error_fail(e, "expected NAME = VALUE");

	size_t name_length = (size_t)(equals - line);
	char *value = equals + 1;
	size_t value_length = length - name_length - 1;
	char *name = line;
	trim(&name, &name_length);
	trim(&value, &value_length);
	// Each ends before the line's end, a blank or the equals sign, and becomes a string there.
	struct spec const *const spec = find_setting(name, name_length);
	if (!spec) {
		// Cut at a null byte, the name would print as another, maybe a setting's.
		if (memchr(name, '\0', name_length))
			return error_fail(e, "a name with a null byte is not a config setting");
		name[name_length] = '\0';
		return error_fail(e, "'%s' is not a config setting", name);
	}
	value[value_length] = '\0';
	return apply(o, spec->letter, value, spec->kind == SPEC_PATH ? home : nullptr, e);
}

/** @brief Applies a config file's settings, one a line; # starts a comment line.
 *
 * @param o    The options.
 * @param path The file, for the words.
 * @param text Its text, which the parsing changes, with a null after it.
 * @param size Its size.
 * @param home The home, for paths that start with ~/.
 * @param e    Receives the words for what is wrong, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
apply_settings (struct options          *o,
                char const              *path,
                char                    *text,
                size_t                   size,
                struct paths_home const *home,
                struct error            *e)
{
	uint32_t line_number = 0;
	for (size_t start = 0; start < size; ++line_number) {
		char *line = text + start;
		char const *const newline = memchr(line, '\n', size - start);
		size_t length = newline ? (size_t)(newline - line) : size - start;
		start += length + 1;
		trim(&line, &length);
		if (!length || *line == '#')
			continue;
		enum error_code const code = apply_setting(o, line, length, home, e);
		if (code) {
			error_wrap(e, "%s:%" PRIu32 ": ", path, line_number + 1);
			return code;
		}
	}
	return ERROR_NONE;
}

/** @brief Applies a config file's settings.
 *
 * @param o    The options.
 * @param path The file.
 * @param home The home, for paths that start with ~/.
 * @param e    Receives the words for what is wrong, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED for a file that cannot be read or holds an invalid
 *             setting.
 */
static enum error_code
read_config (struct options          *o,
             char const              *path,
             struct paths_home const *home,
             struct error            *e)
{
	struct files_data text;
	enum error_code code = files_read(&text, path, e);
	if (code) {
		error_wrap(e, "cannot read config file %s: ", path);
		return code;
	}
	code = apply_settings(o, path, (char *)text.bytes, text.size, home, e);
	files_data_fini(&text);
	return code;
}

/** @brief Applies the default config file's settings, if there is one; a file that is missing is
 *         skipped, but one that cannot be inspected or read is an error.
 *
 * @param o    The options.
 * @param home The home, for the file and for paths that start with ~/.
 * @param e    Receives the words for what is wrong, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
read_default_config (struct options          *o,
                     struct paths_home const *home,
                     struct error            *e)
{
	char *path;
	size_t length;
	enum error_code code = paths_default_config(&path, &length, home, e);
	if (code || !path)
		return code;
	struct stat st;
	if (!stat(path, &st) || (errno != ENOENT && errno != ENOTDIR))
		code = read_config(o, path, home, e);
	free(path);
	path = nullptr;
	return code;
}

/** @brief Checks that the options go together.
 *
 * @param o     The options.
 * @param given The command line's mask of the options it gave.
 * @param e     Receives the words for what is wrong, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
check (struct options const *o,
       uint32_t              given,
       struct error         *e)
{
	if (o->self_test_runs < 2 || o->self_test_runs > 1000)
		return error_fail(e, "--self-test-runs must be 2..1000");
	if (on_command_line(given, 'r') && !o->self_test)
		return error_fail(e, "--self-test-runs requires --self-test");
	if (o->self_test_drops > 999)
		return error_fail(e, "--self-test-drops must be 0..999");
	if (on_command_line(given, 'X') && !o->self_test)
		return error_fail(e, "--self-test-drops requires --self-test");
	if (!o->diagnose && !o->test_identity && (!o->assets_length || !o->modules_length))
		return error_fail(e, "--assets and --modules are required for inference");
	if (o->self_test && (o->test_identity || o->input_length))
		return error_fail(e, "--self-test requires the real network and generates its own input");
	if (!o->self_test && !o->input_length != !o->output_length)
		return error_fail(e, "--input and --output must be supplied together");
	if (o->input_length && (!o->width || !o->height || o->width > kMaxW || o->height > kMaxH))
		return error_fail(e, "offline dimensions must be 1..7680 by 1..4320");
	if (o->trace_dir_length && (o->self_test || o->input_length || o->test_identity || o->diagnose))
		return error_fail(e, "--trace-dir requires serving real shared-memory inference");
	// A configured idle exit is for serving; only an explicit one is an error.
	if (on_command_line(given, 'x') && o->idle_exit && (o->self_test || o->input_length || o->diagnose))
		return error_fail(e, "--idle-exit requires serving shared-memory requests");
	char const *const hip = options_hip_only(o);
	if (hip && o->backend == OPTIONS_BACKEND_VULKAN)
		return error_fail(e, "%s requires --backend hip", hip);
	return ERROR_NONE;
}

/** @brief Applies the options of a command line, in its order.
 *
 * @param o            The options.
 * @param argc         The number of arguments.
 * @param argv         The arguments, which getopt_long() has scanned once.
 * @param letters      getopt_long()'s short options.
 * @param long_options Its long options.
 * @param e            Receives the words for what is wrong, or nullptr.
 * @return             ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
apply_command_line (struct options      *o,
                    int                  argc,
                    char               **argv,
                    char const          *letters,
                    struct option const *long_options,
                    struct error        *e)
{
	// Scanned again from the start.
	optind = 0;
	for (int c; (c = getopt_long(argc, argv, letters, long_options, nullptr)) != -1;) {
		if (c == 'f')
			continue;
		enum error_code const code = apply(o, c, optarg ? optarg : "true", nullptr, e);
		if (code)
			return code;
	}
	return ERROR_NONE;
}

enum error_code
options_parse (struct options  *dest,
               int              argc,
               char           **argv,
               struct error    *e)
{
	struct option long_options[SPEC_COUNT + 1];
	char letters[2 * SPEC_COUNT + 1];
	char *next = letters;
	for (uint32_t i = 0; i < SPEC_COUNT; ++i) {
		struct spec const *const spec = &SPECS[i];
		bool const flag = spec->kind == SPEC_FLAG;
		long_options[i] = (struct option){spec->name, flag ? no_argument : required_argument, nullptr, spec->letter};
		*next++ = spec->letter;
		if (!flag)
			*next++ = ':';
	}
	long_options[SPEC_COUNT] = (struct option){};
	*next = '\0';

	// The options override the config file, which one of them may name: they are applied in a
	// second scan, after the config file.
	*dest = (struct options){};
	char const *config = nullptr;
	uint32_t given = 0;
	for (int c; (c = getopt_long(argc, argv, letters, long_options, nullptr)) != -1;) {
		struct spec const *const spec = find_letter(c);
		if (!spec) {
			enum error_code const code = usage(stderr, e);
			return code ? code : error_fail(e, "invalid arguments");
		}
		if (c == 'h') {
			dest->help = true;
			return usage(stdout, e);
		}
		if (c == 'f')
			config = optarg;
		given |= spec_bit(spec);
	}
	if (optind != argc)
		return error_fail(e, "unexpected positional argument");

	struct paths_home const home = paths_home();
	enum error_code code = options_init(dest, &home, e);
	if (code)
		return code;
	code = config ? read_config(dest, config, &home, e) : read_default_config(dest, &home, e);
	if (!code)
		code = apply_command_line(dest, argc, argv, letters, long_options, e);
	if (!code)
		code = check(dest, given, e);
	if (code)
		options_fini(dest);
	return code;
}

enum error_code
options_configured_vulkan_model (char                    **dest,
                                 size_t                   *length,
                                 struct paths_home const  *home,
                                 struct error             *e)
{
	// Only the model's default is needed: the file's other settings are applied only to be checked.
	struct options o = {};
	enum error_code code = paths_default_vulkan_model(&o.vulkan_model, &o.vulkan_model_length, home, e);
	if (!code)
		code = read_default_config(&o, home, e);
	if (code) {
		*dest = nullptr;
		*length = 0;
	} else {
		*dest = o.vulkan_model;
		*length = o.vulkan_model_length;
		o.vulkan_model = nullptr;
	}
	options_fini(&o);
	return code;
}
