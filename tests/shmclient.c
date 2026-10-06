/** @file
 *
 * shmclient, the smallest fake layer, for tracing served frames (tests/hiptrace/trace.sh and
 * tests/vktrace/trace.sh): creates a private channel, starts dlsslopd on it, publishes RGBA8 or FP16
 * frames the way the layer's shm_map_process_frame() does, changes settings between frames as
 * dlsslopctl does, and prints the FNV-1a 64 of every input and answer. No device-local transport is
 * offered.
 */
// SPDX-License-Identifier: MIT
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <linux/futex.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "control_settings.h"
#include "error.h"
#include "shm_channel.h"
#include "shm_protocol.h"
#include "support.h"
#include "util.h"

/** @brief The usage line, which a usage error repeats. */
static char const USAGE[] = "Usage: shmclient [OPTIONS] -- DLSSLOPD [ARGUMENTS...]\n";

/** @brief --frames when not given. */
static char const DEFAULT_FRAMES[] = "AAB";

/** @brief --log when not given. */
static char const DEFAULT_LOG[] = "/dev/null";

/** @brief The letters that --frames takes. */
static char const LETTERS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZf";

/** @brief The flat frame f: (1, 0.25, 0.75, 1) as RGBA8 UNORM. */
static uint8_t const FLAT[4] = {255, 64, 191, 255};

/** @brief The getopt code of the first setting's option; each setting's is this plus its index in
 *         CONTROL_SETTINGS.
 */
static constexpr int SETTING_CODE = 256;

/** @brief The nanoseconds in a second. */
static constexpr uint64_t NS_PER_SECOND = 1000000000;

/** @brief The client's own options; the settings' follow them. */
static struct option const OWN_OPTIONS[] = {
	{"shm",    required_argument, nullptr, 's'},
	{"width",  required_argument, nullptr, 'W'},
	{"height", required_argument, nullptr, 'H'},
	{"frames", required_argument, nullptr, 'f'},
	{"fp16",   no_argument,       nullptr, 'F'},
	{"dump",   required_argument, nullptr, 'd'},
	{"log",    required_argument, nullptr, 'l'},
	{"help",   no_argument,       nullptr, 'h'},
};

/** @brief The number of OWN_OPTIONS. */
static constexpr size_t OWN_OPTION_COUNT = sizeof OWN_OPTIONS / sizeof *OWN_OPTIONS;

/** @brief A setting's new value, stored before frame FRAME is sent (0: before the daemon starts). */
struct change {
	struct control_setting const *setting; //!< The setting.
	size_t                        frame;   //!< The frame it is stored before.
	uint32_t                      value;   //!< The bits it stores.
};

/** @brief What the client holds: its command line, the channel, the frame it sends and the daemon. */
struct client {
	struct shm_channel channel;      //!< The channel, or empty.
	struct change     *changes;      //!< The settings' changes in the options' order; room for one per argument.
	uint8_t           *rgba;         //!< The frame being sent, as RGBA8, or nullptr.
	uint8_t           *wide;         //!< Its FP16 proxy with --fp16, or nullptr.
	char const        *shm;          //!< --shm, or nullptr.
	char const        *frames;       //!< --frames.
	char const        *dump;         //!< --dump, or nullptr.
	char const        *log;          //!< --log.
	size_t             change_count; //!< The changes.
	size_t             frame_count;  //!< The frames, the length of --frames.
	uint32_t           width;        //!< --width, or 0 if it is missing or out of range.
	uint32_t           height;       //!< --height, or 0 if it is missing or out of range.
	uint32_t           fp16;         //!< --fp16: 1 for FP16 proxies, the requests' hdrEncode; 0 for RGBA8.
	pid_t              daemon;       //!< The daemon until it is reaped, or -1.
};

/** @brief The FNV-1a 64 of bytes.
 *
 * @param p The bytes.
 * @param n How many.
 * @return  Their hash.
 */
static uint64_t
fnv (uint8_t const *p,
     size_t         n)
{
	uint64_t h = UINT64_C(14695981039346656037);
	for (size_t i = 0; i < n; ++i)
		h = (h ^ p[i]) * UINT64_C(1099511628211);
	return h;
}

/** @brief Wakes the daemon if it waits on a word of the channel.
 *
 * @param word The word.
 */
static void
wake (_Atomic(uint32_t) *word)
{
	syscall(SYS_futex, word, FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

/** @brief Writes the self-test's gradient (backend/self_test.c), varied per letter.
 *
 * @param rgba   Receives the frame, RGBA8.
 * @param w      Its width.
 * @param h      Its height.
 * @param letter The letter's index from A.
 */
static void
gradient (uint8_t  *rgba,
          uint32_t  w,
          uint32_t  h,
          uint32_t  letter)
{
	uint8_t const on = (uint8_t)(192 + 8 * letter);
	uint8_t const off = (uint8_t)(64 - 8 * letter);
	for (uint32_t y = 0; y < h; ++y)
		for (uint32_t x = 0; x < w; ++x) {
			uint8_t *const p = &rgba[((size_t)y * w + x) * 4];
			p[0] = (uint8_t)((x + 16 * letter) * 255 / (w - 1 + 16 * letter));
			p[1] = (uint8_t)(y * 255 / (h - 1));
			p[2] = (((x + 8 * letter) / 32 ^ y / 32) & 1) ? on : off;
			p[3] = 255;
		}
}

/** @brief Writes the FP16 proxy of an RGBA8 frame: the same display-encoded values as binary16.
 *
 * @param out    Receives the proxy, two bytes a value.
 * @param rgba   The frame.
 * @param values Its values, four a pixel.
 */
static void
widen (uint8_t       *out,
       uint8_t const *rgba,
       size_t         values)
{
	for (size_t i = 0; i < values; ++i) {
		_Float16 const v = (_Float16)((float)rgba[i] / 255.0f);
		memcpy(out + i * sizeof v, &v, sizeof v);
	}
}

/** @brief Reports a usage error.
 *
 * @param problem What is wrong.
 * @return        2, the exit status of a usage error.
 */
static int
usage (char const *problem)
{
	fprintf(stderr, "shmclient: %s\n%s(see --help)\n", problem, USAGE);
	return 2;
}

/** @brief Prints the help, with every option's default and every setting's range and default. */
static void
help (void)
{
	uint32_t defaults[CONTROL_SETTING_COUNT];
	control_settings_defaults(defaults, false);
	printf("%s", USAGE);
	printf("Create the channel FILE, start DLSSLOPD, which must serve FILE (--shm FILE),\n"
	       "send it the frames of --frames and print the FNV-1a 64 of every input and\n"
	       "answer. Exit status 0 when the daemon exits with 0 and answers every frame,\n"
	       "2 on a usage error, otherwise 1.\n"
	       "\n"
	       "  -s, --shm FILE       Channel to create, in a private (0700) directory\n"
	       "                       Default: unset; required\n"
	       "  -W, --width PIXELS   Source width (2..%u)\n"
	       "                       Default: unset; required\n"
	       "  -H, --height PIXELS  Source height (2..%u)\n"
	       "                       Default: unset; required\n"
	       "  -f, --frames SPEC    One letter per frame naming its input: A to Z the\n"
	       "                       self-test's gradient, shifted by the letter; f the\n"
	       "                       flat colour (%u, %u, %u, %u)\n"
	       "                       Default: %s\n"
	       "  -F, --fp16           Send FP16 proxies of the same values (hdrEncode 1)\n"
	       "                       Default: off; RGBA8\n"
	       "  -d, --dump DIR       Write every answer to DIR/answer-N.rgba8, or\n"
	       "                       DIR/answer-N.rgba16f with --fp16 (N from 0)\n"
	       "                       Default: unset; no files\n"
	       "  -l, --log FILE       The daemon's standard output and error\n"
	       "                       Default: %s\n"
	       "  -h, --help           Show this help and exit\n"
	       "                       Default: off\n"
	       "\n"
	       "Settings, dlsslopctl's with their ranges and channel defaults, take\n"
	       "[FRAME:]VALUE and may be repeated: VALUE is stored before frame FRAME (from\n"
	       "0) is sent, and before the daemon starts when FRAME is 0 or omitted. The\n"
	       "daemon follows a tier only between requests, so for a --tier from a later\n"
	       "frame the client waits until the daemon publishes the new neural raster\n"
	       "before it sends that frame; a daemon that publishes none (--test-identity)\n"
	       "is refused as a usage error.\n",
	       kMaxW, kMaxH, FLAT[0], FLAT[1], FLAT[2], FLAT[3], DEFAULT_FRAMES, DEFAULT_LOG);
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
		struct control_setting const *const s = &CONTROL_SETTINGS[i];
		printf("      --%s [FRAME:]VALUE\n"
		       "                       %s\n"
		       "                       Range: %g..%g; default: %g\n",
		       s->name, s->help, s->minimum, s->maximum, control_setting_value(s, defaults[i]));
	}
}

/** @brief The exit status that a shell reports for a wait status.
 *
 * @param status The wait status.
 * @return       The exit status, or 128 plus the signal that ended the process.
 */
static int
exit_code (int status)
{
	return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

/** @brief Reads a decimal number in [1, limit].
 *
 * @param text  The number.
 * @param limit The greatest value.
 * @return      The number, or 0 if the text is not one in the range.
 */
static uint32_t
number (char const *text,
        uint32_t    limit)
{
	char *end;
	// A number out of strtoul()'s range reads as ULONG_MAX, above every limit: errno is not needed.
	unsigned long const value = strtoul(text, &end, 10);
	return *text >= '0' && *text <= '9' && !*end && value <= limit ? (uint32_t)value : 0;
}

/** @brief Reads [FRAME:]VALUE of a setting, VALUE as dlsslopctl reads it.
 *
 * @param text    The option's value.
 * @param setting The setting.
 * @param dest    Receives the change.
 * @return        Whether the text is one.
 */
static bool
change (char const                   *text,
        struct control_setting const *setting,
        struct change                *dest)
{
	*dest = (struct change){ .setting = setting };
	char *end;
	if (*text >= '0' && *text <= '9') {
		// strtoull() reports a value out of its range only through errno.
		errno = 0;
		unsigned long long const frame = strtoull(text, &end, 10);
		if (errno)
			return false;
		if (*end == ':') {
			dest->frame = frame;
			text = end + 1;
		}
	}
	if (setting->is_float) {
		if (!*text || isspace((unsigned char)*text))
			return false;
		errno = 0;
		double const value = strtod(text, &end);
		if (errno || *end || !control_setting_in_range(setting, value))
			return false;
		// + 0.0f stores -0 as 0.
		dest->value = FloatToBits((float)value + 0.0f);
		return true;
	}
	if (*text < '0' || *text > '9')
		return false;
	errno = 0;
	unsigned long long const value = strtoull(text, &end, 10);
	dest->value = (uint32_t)value;
	return !errno && !*end && control_setting_in_range(setting, (double)value);
}

/** @brief Stores the changes for a frame, publishes them as dlsslopctl does and prints them.
 *
 * @param c     The client.
 * @param frame The frame.
 */
static void
apply (struct client *c,
       size_t         frame)
{
	struct ShmHeader *const h = c->channel.h;
	bool any = false;
	bool tuning = false;
	for (size_t i = 0; i < c->change_count; ++i) {
		struct change const *const edit = &c->changes[i];
		if (edit->frame != frame)
			continue;
		control_setting_store(h, edit->setting, edit->value);
		printf("%s%s=%g", any ? " " : "settings: ", edit->setting->name,
		       control_setting_value(edit->setting, edit->value));
		any = true;
		tuning |= edit->setting->tuning;
	}
	if (!any)
		return;
	printf(" before frame %zu\n", frame);
	if (tuning)
		atomic_fetch_add(&h->tuningSeq, 1);
	atomic_fetch_add(&h->controlSeq, 1);
}

/** @brief The tier stored before a frame.
 *
 * @param c     The client.
 * @param frame The frame.
 * @return      The last tier that a change stores before the frame, or 0 if none does.
 */
static uint32_t
tier_at (struct client const *c,
         size_t               frame)
{
	uint32_t tier = 0;
	for (size_t i = 0; i < c->change_count; ++i) {
		struct change const *const edit = &c->changes[i];
		if (edit->frame == frame && edit->setting->offset == offsetof(struct ShmHeader, nativeTier))
			tier = edit->value;
	}
	return tier;
}

/** @brief Reaps the daemon if it has exited.
 *
 * @param c      The client.
 * @param status Receives its wait status if it has.
 * @return       Whether it has.
 */
static bool
exited (struct client *c,
        int           *status)
{
	if (waitpid(c->daemon, status, WNOHANG) != c->daemon)
		return false;
	c->daemon = -1;
	return true;
}

/** @brief Waits until the daemon runs, at a tier if one is asked for.
 *
 * @param c    The client.
 * @param tier The tier whose raster the daemon must have published, or 0 for any.
 * @param what What the daemon must do, for the messages.
 * @return     0 once it runs; otherwise 1 after a message, the daemon reaped or left to
 *             client_fini() to stop.
 */
static int
await (struct client *c,
       uint32_t       tier,
       char const    *what)
{
	struct ShmHeader *const h = c->channel.h;
	uint64_t const deadline = now_ns() + 300 * NS_PER_SECOND;
	while ((tier && atomic_load(&h->nativeModelMaxHeight) != tier)
	       || atomic_load_explicit(&h->helperState, memory_order_acquire) != kHelperRunning) {
		int status;
		if (exited(c, &status)) {
			fprintf(stderr, "daemon exited before %s (status %d)\n", what, exit_code(status));
			return 1;
		}
		if (now_ns() > deadline) {
			fprintf(stderr, "daemon timed out before %s\n", what);
			return 1;
		}
		support_sleep_ms(10);
	}
	return 0;
}

/** @brief Writes an answer to a file of --dump.
 *
 * @param path   The file.
 * @param answer The answer.
 * @param bytes  Its size.
 * @return       Whether it was written; false after a message.
 */
static bool
dump_answer (char const    *path,
             uint8_t const *answer,
             size_t         bytes)
{
	FILE *const file = fopen(path, "wb");
	bool written = file && fwrite(answer, 1, bytes, file) == bytes;
	int err = errno;
	if (file && fclose(file) && written) {
		written = false;
		err = errno;
	}
	if (!written)
		fprintf(stderr, "cannot write %s: %s\n", path, strerror(err));
	return written;
}

/** @brief Waits for the answer to a request.
 *
 * @param c       The client.
 * @param request The request's number.
 * @return        0 once the daemon has answered; otherwise 1 after a message, the daemon reaped or
 *                left to client_fini() to stop.
 */
static int
answer_wait (struct client *c,
             uint32_t       request)
{
	struct ShmHeader *const h = c->channel.h;
	uint64_t const deadline = now_ns() + 180 * NS_PER_SECOND;
	for (uint32_t response; (response = atomic_load_explicit(&h->seq_resp, memory_order_acquire)) != request;) {
		int status;
		if (exited(c, &status)) {
			fprintf(stderr, "daemon exited during request %u (status %d)\n", request, exit_code(status));
			return 1;
		}
		if (now_ns() > deadline) {
			fprintf(stderr, "no answer to request %u\n", request);
			return 1;
		}
		struct timespec const timeout = { .tv_nsec = 10000000 };
		syscall(SYS_futex, &h->seq_resp, FUTEX_WAIT, response, &timeout, nullptr, 0);
	}
	return 0;
}

/** @brief Sends the frames one at a time, printing each and its answer, and asks the daemon to quit.
 *
 * @param c The client, its daemon running.
 * @return  The exit status, as serve() gives it.
 */
static int
send_frames (struct client *c)
{
	struct ShmHeader *const h = c->channel.h;
	uint8_t *const input = (uint8_t *)h + kHeaderBytes;
	uint8_t const *const output = input + kMaxFrame;
	size_t const pixels = (size_t)c->width * c->height;
	size_t const bytes = pixels * (c->fp16 ? 8 : 4);
	c->rgba = malloc(pixels * 4);
	c->wide = c->fp16 ? malloc(bytes) : nullptr;
	if (!c->rgba || (c->fp16 && !c->wide)) {
		fputs("shmclient: out of memory\n", stderr);
		return 1;
	}
	uint8_t const *const image = c->fp16 ? c->wide : c->rgba;
	uint32_t failures = 0;
	for (size_t index = 0; index < c->frame_count; ++index) {
		if (index) {
			apply(c, index);
			// The daemon follows a tier only between requests: wake it and wait until it has
			// published the new raster, so that this frame is served at that tier.
			uint32_t const tier = tier_at(c, index);
			if (tier) {
				wake(&h->seq_req);
				if (await(c, tier, "it followed the tier"))
					return 1;
			}
		}
		char const letter = c->frames[index];
		if (letter == 'f')
			for (size_t p = 0; p < pixels; ++p)
				memcpy(&c->rgba[p * 4], FLAT, 4);
		else
			gradient(c->rgba, c->width, c->height, (uint32_t)(letter - 'A'));
		if (c->fp16)
			widen(c->wide, c->rgba, pixels * 4);
		memcpy(input, image, bytes);
		atomic_store(&h->width, c->width);
		atomic_store(&h->height, c->height);
		atomic_store(&h->hdrEncode, c->fp16);
		uint32_t const request = atomic_load(&h->seq_req) + 1;
		uint64_t const sent = now_ns();
		atomic_thread_fence(memory_order_release);
		atomic_store(&h->seq_req, request);
		wake(&h->seq_req);
		if (answer_wait(c, request))
			return 1;
		double const ms = (double)(now_ns() - sent) / 1e6;
		bool const ok = atomic_load(&h->seq_ok) == request;
		uint32_t const aw = atomic_load(&h->answeredW);
		uint32_t const ah = atomic_load(&h->answeredH);
		size_t const answer = ok ? (size_t)aw * ah * (c->fp16 ? 8 : 4) : 0;
		failures += !ok;
		printf("frame %zu input=%c input_fnv=%016" PRIx64 " request=%u ok=%d answered=%ux%u answer_fnv=%016"
		       PRIx64 " ms=%.2f\n",
		       index, letter, fnv(image, bytes), request, ok, aw, ah, ok ? fnv(output, answer) : 0, ms);
		if (!ok) {
			char reason[kReasonBytes];
			// An empty reason if it changed during every copy.
			ShmLoadString(h, SHM_TEXT_HELPER_REASON, reason, sizeof reason);
			printf("reason: %s\n", reason);
		}
		if (ok && c->dump) {
			char *path = support_format(nullptr, "%s/answer-%zu%s", c->dump, index,
			                            c->fp16 ? ".rgba16f" : ".rgba8");
			if (!path) {
				fputs("shmclient: out of memory\n", stderr);
				++failures;
			} else if (!dump_answer(path, output, answer)) {
				++failures;
			}
			free(path);
			path = nullptr;
		}
		if (fflush(stdout)) {
			perror("shmclient: standard output");
			return 1;
		}
	}
	atomic_store(&h->quit, 1);
	wake(&h->seq_req);
	int status;
	for (uint64_t const deadline = now_ns() + 60 * NS_PER_SECOND; !exited(c, &status);) {
		if (now_ns() > deadline) {
			fputs("daemon did not quit\n", stderr);
			return 1;
		}
		support_sleep_ms(10);
	}
	int const code = exit_code(status);
	printf("daemon exit=%d failures=%u\n", code, failures);
	if (fflush(stdout)) {
		perror("shmclient: standard output");
		return 1;
	}
	if (unlink(c->shm)) {
		fprintf(stderr, "shmclient: cannot remove %s: %s\n", c->shm, strerror(errno));
		return 1;
	}
	return code || failures;
}

/** @brief Starts the daemon on the channel, sends it the frames and asks it to quit.
 *
 * @param c      The client, its channel open and its changes of frame 0 stored.
 * @param daemon The daemon's path and its arguments, ending in nullptr.
 * @return       The exit status: 0 if the daemon exited with 0 after answering every frame, 2 on a
 *               usage error, otherwise 1; the daemon reaped or left to client_fini() to stop.
 */
static int
serve (struct client     *c,
       char const *const  daemon[])
{
	struct ShmHeader *const h = c->channel.h;
	c->daemon = support_spawn(c->log, daemon);
	if (c->daemon < 0) {
		perror("fork");
		return 1;
	}
	if (await(c, 0, "it was ready"))
		return 1;
	printf("ready tier=%u state=%u fp16=%u\n", atomic_load(&h->nativeTier), atomic_load(&h->helperState), c->fp16);
	// Without a published raster, nothing shows when the daemon has followed a tier.
	if (!atomic_load(&h->nativeModelMaxHeight))
		for (size_t index = 1; index < c->frame_count; ++index)
			if (tier_at(c, index)) {
				fprintf(stderr,
				        "shmclient: --tier from frame %zu needs a daemon that publishes its neural "
				        "raster, and this one publishes none\n",
				        index);
				return 2;
			}
	return send_frames(c);
}

/** @brief Reads the command line, creates the channel and serves.
 *
 * @param c    The client, empty but for its defaults and room for its changes.
 * @param argc The arguments' count.
 * @param argv The arguments.
 * @return     The exit status, as serve() gives it.
 */
static int
client_run (struct client  *c,
            int             argc,
            char          **argv)
{
	// The client's own options, then one long option per setting.
	struct option options[OWN_OPTION_COUNT + CONTROL_SETTING_COUNT + 1];
	memcpy(options, OWN_OPTIONS, sizeof OWN_OPTIONS);
	for (uint32_t i = 0; i < CONTROL_SETTING_COUNT; ++i)
		options[OWN_OPTION_COUNT + i] = (struct option){
			.name = CONTROL_SETTINGS[i].name,
			.has_arg = required_argument,
			.val = SETTING_CODE + (int)i,
		};
	options[OWN_OPTION_COUNT + CONTROL_SETTING_COUNT] = (struct option){};
	for (int code; (code = getopt_long(argc, argv, "+s:W:H:f:Fd:l:h", options, nullptr)) != -1;) {
		switch (code) {
		case 's':
			c->shm = optarg;
			break;
		case 'W':
			c->width = number(optarg, kMaxW);
			break;
		case 'H':
			c->height = number(optarg, kMaxH);
			break;
		case 'f':
			c->frames = optarg;
			break;
		case 'F':
			c->fp16 = 1;
			break;
		case 'd':
			c->dump = optarg;
			break;
		case 'l':
			c->log = optarg;
			break;
		case 'h':
			help();
			return 0;
		default:
			if (code < SETTING_CODE || code - SETTING_CODE >= (int)CONTROL_SETTING_COUNT)
				return usage("unknown option or missing value");
			struct control_setting const *const s = &CONTROL_SETTINGS[code - SETTING_CODE];
			// Each setting's option is an argument at least: the changes fit.
			if (!change(optarg, s, &c->changes[c->change_count])) {
				fprintf(stderr,
				        "shmclient: --%s: expected [FRAME:]VALUE with VALUE in [%g, %g], got '%s'\n",
				        s->name, s->minimum, s->maximum, optarg);
				return 2;
			}
			++c->change_count;
		}
	}
	if (!c->shm || !*c->shm)
		return usage("--shm is required");
	if (c->width < 2 || c->height < 2)
		return usage("--width and --height are required and must be in range");
	c->frame_count = strlen(c->frames);
	if (!c->frame_count || strspn(c->frames, LETTERS) != c->frame_count)
		return usage("--frames must be letters from A to Z, or f");
	for (size_t i = 0; i < c->change_count; ++i)
		if (c->changes[i].frame >= c->frame_count)
			return usage("a setting's FRAME is not one of --frames");
	if (optind >= argc)
		return usage("DLSSLOPD is required");

	// A new channel, created as every program creates one, where an earlier run may have left one.
	if (unlink(c->shm) && errno != ENOENT) {
		fprintf(stderr, "shmclient: cannot remove %s: %s\n", c->shm, strerror(errno));
		return 1;
	}
	struct error e;
	if (shm_channel_open(&c->channel, c->shm, strlen(c->shm), ShmTotalBytes(),
	                     SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, &e)) {
		fprintf(stderr, "shmclient: %s\n", e.what);
		return 1;
	}
	apply(c, 0);
	// C converts char ** to char const *const * only by a cast.
	return serve(c, (char const *const *)(argv + optind));
}

/** @brief Stops the daemon if it still runs and frees what the client holds, leaving it empty.
 *
 * @param c The client.
 */
static void
client_fini (struct client *c)
{
	if (c->daemon > 0 && !kill(c->daemon, SIGKILL))
		while (waitpid(c->daemon, nullptr, 0) < 0 && errno == EINTR) {}
	shm_channel_fini(&c->channel);
	free(c->wide);
	c->wide = nullptr;
	free(c->rgba);
	c->rgba = nullptr;
	free(c->changes);
	c->changes = nullptr;
	*c = (struct client){ .channel = { .fd = -1 }, .daemon = -1 };
}

int
main (int    argc,
      char **argv)
{
	// Each change is a setting's option, which is an argument at least.
	struct change *const changes = malloc((size_t)argc * sizeof *changes);
	struct client c = {
		.channel = { .fd = -1 },
		.changes = changes,
		.frames = DEFAULT_FRAMES,
		.log = DEFAULT_LOG,
		.daemon = -1,
	};
	int status = 1;
	if (c.changes)
		status = client_run(&c, argc, argv);
	else
		fputs("shmclient: out of memory\n", stderr);
	client_fini(&c);
	return status;
}
