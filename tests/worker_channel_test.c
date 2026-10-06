/** @file
 *
 * Drives `dlsslopd --test-identity` over its shared-memory channel the way the layer's
 * shm_map_process_frame() does. Needs neither HIP nor model weights.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <linux/futex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "error.h"
#include "shm_channel.h"
#include "shm_protocol.h"
#include "support.h"
#include "util.h"

/** @brief The width of the frames that the test sends. */
static constexpr uint32_t WIDTH = 64;

/** @brief Their height. */
static constexpr uint32_t HEIGHT = 36;

/** @brief The nanoseconds in a second. */
static constexpr uint64_t NS_PER_SECOND = 1000000000;

/** @brief The worker that a failed check stops, or -1: at most one runs at a time. */
static pid_t running = -1;

/** @brief The test's directory, removed however the test ends; nullptr until it is made. */
static char *scratch;

/** @brief Stops the running worker and ends the test with status 1. */
[[noreturn]]
static void
end_failed (void)
{
	if (running > 0 && !kill(running, SIGKILL))
		while (waitpid(running, nullptr, 0) < 0 && errno == EINTR) {}
	exit(1);
}

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param fmt       A printf format for what failed.
 * @param ...       The format's arguments.
 */
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
	fputs("worker channel: ", stderr);
	vfprintf(stderr, fmt, args);
	va_end(args);
	fputc('\n', stderr);
	end_failed();
}

/** @brief Prints that a part of the test passed.
 *
 * @param what What passed.
 */
static void
pass (char const *what)
{
	require(printf("PASS: %s\n", what) >= 0, "cannot write to the standard output");
}

/** @brief Wakes the worker if it waits on a word of the channel.
 *
 * @param word The word.
 */
static void
wake (_Atomic(uint32_t) *word)
{
	syscall(SYS_futex, word, FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

/** @brief A path in the test's directory.
 *
 * @param name   The file's name.
 * @param length Receives the path's length, or nullptr.
 * @return       The path, which the caller frees.
 */
static char *
scratch_path (char const *name,
              size_t     *length)
{
	char *const path = support_format(length, "%s/%s", scratch, name);
	require(path, "out of memory");
	return path;
}

/** @brief How many times a text holds a needle, overlaps counted.
 *
 * @param text   The text.
 * @param needle The needle.
 * @return       The count.
 */
static size_t
count (char const *text,
       char const *needle)
{
	size_t found = 0;
	for (char const *at = strstr(text, needle); at; at = strstr(at + 1, needle))
		++found;
	return found;
}

/** @brief The bytes of a frame of the test's size.
 *
 * @param fp16 Whether the frame is FP16 rather than RGBA8.
 * @return     Its size.
 */
static size_t
frame_bytes (bool fp16)
{
	return (size_t)WIDTH * HEIGHT * (fp16 ? 8 : 4);
}

/** @brief The layer's side of a channel, created before any worker exists, as every program creates
 *         one.
 */
struct channel {
	struct shm_channel shm;    //!< The channel, open and mapped whole.
	char              *path;   //!< Its path.
	uint8_t           *input;  //!< The region of the requests' proxies.
	uint8_t           *output; //!< The region of the answers.
};

/** @brief Creates a channel in the test's directory.
 *
 * @param dest Receives the channel.
 * @param name The channel file's name.
 */
static void
channel_init (struct channel *dest,
              char const     *name)
{
	size_t length;
	char *const path = scratch_path(name, &length);
	*dest = (struct channel){ .shm = { .fd = -1 }, .path = path };
	struct error e;
	require(!shm_channel_open(&dest->shm, dest->path, length, ShmTotalBytes(),
	                          SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, &e),
	        "create channel: %s", e.what);
	require(dest->shm.flags & SHM_CHANNEL_CREATED, "channel %s existed", dest->path);
	dest->input = (uint8_t *)dest->shm.h + kHeaderBytes;
	dest->output = dest->input + kMaxFrame;
}

/** @brief Unmaps and closes a channel and frees its path, leaving it empty.
 *
 * @param c The channel.
 */
static void
channel_fini (struct channel *c)
{
	shm_channel_fini(&c->shm);
	free(c->path);
	c->path = nullptr;
	*c = (struct channel){ .shm = { .fd = -1 } };
}

/** @brief Publishes one request as the layer does.
 *
 * @param c     The channel.
 * @param fp16  Whether the request's proxy is FP16 rather than RGBA8.
 * @param seed  The first of the proxy's bytes, which step by 7.
 * @param width The request's width: WIDTH, or above kMaxW for a malformed request.
 * @return      The request's number.
 */
static uint32_t
channel_publish (struct channel *c,
                 bool            fp16,
                 uint8_t         seed,
                 uint32_t        width)
{
	size_t const bytes = frame_bytes(fp16);
	for (size_t i = 0; i < bytes; ++i)
		c->input[i] = (uint8_t)(seed + i * 7);
	struct ShmHeader *const h = c->shm.h;
	atomic_store(&h->width, width);
	atomic_store(&h->height, HEIGHT);
	atomic_store(&h->hdrEncode, fp16 ? 1u : 0u);
	uint32_t const request = atomic_load(&h->seq_req) + 1;
	atomic_thread_fence(memory_order_release);
	atomic_store(&h->seq_req, request);
	wake(&h->seq_req);
	return request;
}

/** @brief Waits for the answer to a request.
 *
 * @param c       The channel.
 * @param request The request's number.
 * @param fp16    Whether its proxy is FP16 rather than RGBA8.
 * @return        Whether the worker delivered an identity frame.
 */
static bool
channel_answered (struct channel const *c,
                  uint32_t              request,
                  bool                  fp16)
{
	struct ShmHeader *const h = c->shm.h;
	uint64_t const deadline = now_ns() + 5 * NS_PER_SECOND;
	for (uint32_t response; (response = atomic_load_explicit(&h->seq_resp, memory_order_acquire)) != request;) {
		require(now_ns() < deadline, "no answer to request %u", request);
		struct timespec const timeout = { .tv_nsec = 10000000 };
		syscall(SYS_futex, &h->seq_resp, FUTEX_WAIT, response, &timeout, nullptr, 0);
	}
	if (atomic_load(&h->seq_ok) != request)
		return false;
	require(atomic_load(&h->answeredW) == WIDTH && atomic_load(&h->answeredH) == HEIGHT,
	        "answer dimensions differ");
	require(!memcmp(c->input, c->output, frame_bytes(fp16)), "identity answer differs from its request");
	return true;
}

/** @brief Reads the reason that the worker published.
 *
 * @param c      The channel.
 * @param reason Receives the reason, or an empty one if it changed during every copy.
 */
static void
channel_reason (struct channel const *c,
                char                  reason[static kReasonBytes])
{
	ShmLoadString(c->shm.h, SHM_TEXT_HELPER_REASON, reason, kReasonBytes);
}

/** @brief A worker that serves a channel: `dlsslopd --test-identity`, its output in a log. */
struct worker {
	char const *log; //!< Its standard output and error, a file of the test's directory.
	pid_t       pid; //!< Its process until it is reaped, or -1.
};

/** @brief Reads a worker's log.
 *
 * @param w The worker.
 * @return  The log, which the caller frees.
 */
static char *
worker_log (struct worker const *w)
{
	char *const log = support_read_file(w->log, nullptr);
	require(log, "cannot read %s", w->log);
	return log;
}

/** @brief Ends the test with a message and the worker's log unless a condition holds.
 *
 * @param w         The worker.
 * @param condition The condition.
 * @param fmt       A printf format for what failed.
 * @param ...       The format's arguments.
 */
[[gnu::format(printf, 3, 4)]]
static void
worker_require (struct worker const *w,
                bool                 condition,
                char const          *fmt,
                ...)
{
	if (condition)
		return;
	char *log = support_read_file(w->log, nullptr);
	va_list args;
	va_start(args, fmt);
	fputs("worker channel: ", stderr);
	vfprintf(stderr, fmt, args);
	va_end(args);
	fprintf(stderr, ":\n%s\n", log ? log : "");
	free(log);
	log = nullptr;
	end_failed();
}

/** @brief Starts a worker on a channel and waits until it serves.
 *
 * @param dest       Receives the worker.
 * @param executable dlsslopd.
 * @param c          The channel.
 * @param log        The worker's log, a path that outlives it.
 * @param option     An option after --test-identity and --shm, or nullptr.
 * @param value      The option's value, or nullptr.
 */
static void
worker_start (struct worker        *dest,
              char const           *executable,
              struct channel const *c,
              char const           *log,
              char const           *option,
              char const           *value)
{
	char const *const args[] = {executable, "--test-identity", "--shm", c->path, option, value, nullptr};
	pid_t const pid = support_spawn(log, args);
	*dest = (struct worker){ .log = log, .pid = pid };
	require(dest->pid >= 0, "fork worker");
	running = dest->pid;
	uint64_t const deadline = now_ns() + 10 * NS_PER_SECOND;
	while (atomic_load_explicit(&c->shm.h->helperState, memory_order_acquire) != kHelperRunning) {
		pid_t const done = waitpid(dest->pid, nullptr, WNOHANG);
		// Reaped, or there is no such child to stop.
		if (done)
			dest->pid = running = -1;
		worker_require(dest, !done, "worker exited before it was ready");
		worker_require(dest, now_ns() < deadline, "worker did not become ready");
		support_sleep_ms(5);
	}
}

/** @brief Waits for a worker that exits by itself, and reaps it.
 *
 * @param w The worker.
 * @return  Its exit status, or 128 plus the signal that ended it.
 */
static int
worker_status (struct worker *w)
{
	uint64_t const deadline = now_ns() + 5 * NS_PER_SECOND;
	int status;
	pid_t done;
	while (!(done = waitpid(w->pid, &status, WNOHANG))) {
		worker_require(w, now_ns() < deadline, "worker did not exit");
		support_sleep_ms(5);
	}
	require(done == w->pid, "cannot wait for the worker");
	w->pid = running = -1;
	return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

/** @brief Asks a worker to quit and requires that it exits with status 0.
 *
 * @param w The worker.
 * @param c Its channel.
 */
static void
worker_quit (struct worker  *w,
             struct channel *c)
{
	atomic_store(&c->shm.h->quit, 1);
	wake(&c->shm.h->seq_req);
	worker_require(w, !worker_status(w), "worker did not quit cleanly");
}

/** @brief A worker stopped by a signal leaves the layer's request unanswered. Its successor fails
 *         that request at once (the layer presents its own frame), then serves every request made
 *         after it attached, in either precision.
 *
 * @param executable dlsslopd.
 */
static void
restart (char const *executable)
{
	struct channel channel;
	channel_init(&channel, "restart.bin");
	struct ShmHeader *const h = channel.shm.h;
	atomic_store(&h->seq_req, 6);
	atomic_store(&h->seq_resp, 6);
	atomic_store(&h->seq_ok, 6);
	uint32_t const stale = channel_publish(&channel, true, 1, WIDTH);
	char *log = scratch_path("restart.log", nullptr);
	struct worker worker;
	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	require(atomic_load(&h->seq_resp) == stale && atomic_load(&h->seq_ok) == 0,
	        "a restarted worker did not fail the request left before it started");
	bool const precisions[] = {true, false, true};
	for (size_t i = 0; i < sizeof precisions / sizeof *precisions; ++i) {
		bool const fp16 = precisions[i];
		uint32_t const request = channel_publish(&channel, fp16, (uint8_t)(fp16 + 2), WIDTH);
		require(channel_answered(&channel, request, fp16), "a restarted worker failed a new request");
	}
	worker_quit(&worker, &channel);
	char *text = worker_log(&worker);
	worker_require(&worker, !strstr(text, "failed"), "a restarted worker reported a failed frame");
	free(text);
	text = nullptr;
	free(log);
	log = nullptr;
	channel_fini(&channel);
	pass("a restarted worker failed the stale request and served new FP16 and RGBA8 ones");
}

/** @brief Requires that the channel's reason starts with a rejection's words.
 *
 * @param c         The channel.
 * @param rejection The words.
 * @param length    Their length.
 */
static void
reports (struct channel const *c,
         char const           *rejection,
         size_t                length)
{
	char reason[kReasonBytes];
	channel_reason(c, reason);
	require(!strncmp(reason, rejection, length), "the rejection is not the reason: %s", reason);
}

/** @brief Requires that a served request leaves the reason that the worker published when it was
 *         ready.
 *
 * @param c     The channel.
 * @param ready That reason.
 */
static void
reports_ready (struct channel const *c,
               char const           *ready)
{
	char reason[kReasonBytes];
	channel_reason(c, reason);
	require(!strcmp(reason, ready), "a served request left the rejection as the reason: %s", reason);
}

/** @brief A rejected request is answered as failed and the worker keeps serving. A rejection that
 *         repeats is logged and reported once; a served request makes the worker report itself
 *         ready again.
 *
 * @param executable dlsslopd.
 */
static void
rejection (char const *executable)
{
	struct channel channel;
	channel_init(&channel, "rejection.bin");
	char *log = scratch_path("rejection.log", nullptr);
	struct worker worker;
	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	char ready[kReasonBytes];
	channel_reason(&channel, ready);
	require(strstr(ready, "IDENTITY"), "an identity worker did not say so: %s", ready);
	// The rejection's words, a string literal, start the reason.
#define REPORTS(words) reports(&channel, words, sizeof (words) - 1)
	bool const precisions[] = {false, true, false};
	for (size_t i = 0; i < sizeof precisions / sizeof *precisions; ++i) {
		bool const fp16 = precisions[i];
		for (uint8_t seed = 4; seed <= 5; ++seed) {
			uint32_t const request = channel_publish(&channel, fp16, seed, kMaxW + 1);
			require(!channel_answered(&channel, request, fp16), "a malformed request succeeded");
		}
		REPORTS("unsupported request");
		uint32_t const request = channel_publish(&channel, fp16, 6, WIDTH);
		require(channel_answered(&channel, request, fp16), "the worker stopped serving after a rejection");
		reports_ready(&channel, ready);
	}
	// A preset neither network has, and conditioning outside the model's range.
	struct ShmHeader *const h = channel.shm.h;
	atomic_store(&h->preset, 1);
	uint32_t request = channel_publish(&channel, false, 7, WIDTH);
	require(!channel_answered(&channel, request, false), "an unmapped preset was accepted");
	atomic_store(&h->preset, 0);
	atomic_store(&h->style, 3);
	request = channel_publish(&channel, false, 8, WIDTH);
	require(!channel_answered(&channel, request, false), "style 3 was accepted");
	REPORTS("the preset must be 0");
	request = channel_publish(&channel, false, 9, kMaxW + 1);
	require(!channel_answered(&channel, request, false), "a malformed request succeeded");
	REPORTS("unsupported request");
#undef REPORTS
	atomic_store(&h->style, 0);
	request = channel_publish(&channel, false, 10, WIDTH);
	require(channel_answered(&channel, request, false), "the worker stopped serving after a rejection");
	reports_ready(&channel, ready);
	worker_quit(&worker, &channel);
	// Once per run of the same rejection: three malformed runs, the preset and style run and the
	// malformed request that interrupted it.
	char *text = worker_log(&worker);
	bool const once_each = count(text, " failed: unsupported request") == 4
	                       && count(text, " failed: the preset must be 0") == 1;
	worker_require(&worker, once_each, "each run of a rejection must be logged once");
	free(text);
	text = nullptr;
	free(log);
	log = nullptr;
	channel_fini(&channel);
	pass("rejected requests were answered as failed while serving continued");
}

/** @brief Native tuning applies on the next request. The worker only reads controls: it publishes
 *         no control generation of its own, however long after a change.
 *
 * @param executable dlsslopd.
 */
static void
controls (char const *executable)
{
	struct channel channel;
	channel_init(&channel, "controls.bin");
	char *log = scratch_path("controls.log", nullptr);
	struct worker worker;
	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	struct ShmHeader *const h = channel.shm.h;
	atomic_store(&h->intensityBits, FloatToBits(0.5f));
	atomic_fetch_add(&h->tuningSeq, 1);
	uint32_t const control = atomic_fetch_add(&h->controlSeq, 1) + 1;
	uint32_t request = channel_publish(&channel, false, 11, WIDTH);
	require(channel_answered(&channel, request, false), "a request after a tuning change failed");
	// Long enough after the change for a debounced rebuild to have published.
	support_sleep_ms(300);
	request = channel_publish(&channel, false, 12, WIDTH);
	require(channel_answered(&channel, request, false), "a later request failed");
	require(atomic_load(&h->controlSeq) == control, "the worker published a control generation");
	worker_quit(&worker, &channel);
	free(log);
	log = nullptr;
	channel_fini(&channel);
	pass("the worker published no control generation after a tuning change");
}

/** @brief A restarted worker keeps the channel's live pass count; only --passes, when given,
 *         replaces it.
 *
 * @param executable dlsslopd.
 */
static void
live_settings (char const *executable)
{
	struct channel channel;
	channel_init(&channel, "live.bin");
	struct ShmHeader *const h = channel.shm.h;
	char *log = scratch_path("live.log", nullptr);
	struct worker worker;
	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	atomic_store(&h->passes, 3);
	worker_quit(&worker, &channel);

	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	require(atomic_load(&h->passes) == 3, "a restarted worker reset the live pass count");
	worker_quit(&worker, &channel);

	worker_start(&worker, executable, &channel, log, "--passes", "2");
	require(atomic_load(&h->passes) == 2, "--passes did not replace the live pass count");
	worker_quit(&worker, &channel);

	// main() points XDG_CONFIG_HOME at the test's directory.
	char *directory = scratch_path("dlsslop-amd", nullptr);
	require(!mkdir(directory, 0700), "create the config directory");
	free(directory);
	directory = nullptr;
	char *config = scratch_path("dlsslop-amd/dlsslopd.conf", nullptr);
	FILE *const file = fopen(config, "w");
	bool const written = file && fputs("# live settings\npasses = 4\n", file) >= 0;
	require(file && !fclose(file) && written, "write the config file");
	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	require(atomic_load(&h->passes) == 4, "the config file did not set the pass count");
	worker_quit(&worker, &channel);

	worker_start(&worker, executable, &channel, log, "--passes", "2");
	require(!unlink(config), "remove the config file");
	require(atomic_load(&h->passes) == 2, "--passes did not override the config file");
	worker_quit(&worker, &channel);
	free(config);
	config = nullptr;
	free(log);
	log = nullptr;
	channel_fini(&channel);
	pass("a restarted worker kept the live pass count; the config file and --passes replaced it");
}

/** @brief Whether a worker has rebuilt itself a number of times and serves again.
 *
 * @param w     The worker.
 * @param h     Its channel's header.
 * @param times The rebuilds.
 * @return      true if its log says "rebuilding" that many times and it runs.
 */
static bool
rebuilt (struct worker const    *w,
         struct ShmHeader const *h,
         size_t                  times)
{
	char *text = worker_log(w);
	bool const done = count(text, "rebuilding") == times && atomic_load(&h->helperState) == kHelperRunning;
	free(text);
	text = nullptr;
	return done;
}

/** @brief Waits until a worker has rebuilt itself a number of times and serves again.
 *
 * @param w       The worker.
 * @param h       Its channel's header.
 * @param times   The rebuilds.
 * @param failure What failed if it does not.
 */
static void
await_rebuilt (struct worker const    *w,
               struct ShmHeader const *h,
               size_t                  times,
               char const             *failure)
{
	uint64_t const deadline = now_ns() + 10 * NS_PER_SECOND;
	while (!rebuilt(w, h, times)) {
		worker_require(w, now_ns() < deadline, "%s", failure);
		support_sleep_ms(5);
	}
}

/** @brief A different tier rebuilds the worker once, between frames; the active tier again rebuilds
 *         nothing, and an unusable one is overwritten with the active one. A restarted worker keeps
 *         the live tier; only --tier replaces it.
 *
 * @param executable dlsslopd.
 */
static void
tiers (char const *executable)
{
	struct channel channel;
	channel_init(&channel, "tiers.bin");
	struct ShmHeader *const h = channel.shm.h;
	char *log = scratch_path("tiers.log", nullptr);
	struct worker worker;
	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	require(atomic_load(&h->nativeTier) == kNativeDefaultTier, "a worker changed a new channel's tier");
	atomic_store(&h->nativeTier, 900);
	atomic_fetch_add(&h->controlSeq, 1);
	await_rebuilt(&worker, h, 1, "the worker did not switch to tier 900");
	uint32_t const stores[] = {900, 900, 900, 0, 800};
	for (size_t i = 0; i < sizeof stores / sizeof *stores; ++i) {
		atomic_store(&h->nativeTier, stores[i]);
		atomic_fetch_add(&h->controlSeq, 1);
		uint64_t const deadline = now_ns() + 10 * NS_PER_SECOND;
		while (atomic_load(&h->nativeTier) != 900) {
			worker_require(&worker, now_ns() < deadline, "an unusable tier was not replaced");
			support_sleep_ms(5);
		}
	}
	worker_require(&worker, rebuilt(&worker, h, 1), "storing the active tier rebuilt the worker");
	uint32_t const request = channel_publish(&channel, false, 13, WIDTH);
	require(channel_answered(&channel, request, false), "a request after a tier switch failed");
	atomic_store(&h->nativeTier, 1080);
	await_rebuilt(&worker, h, 2, "the worker did not switch to tier 1080");
	require(!atomic_load(&h->nativeModelMaxWidth) && !atomic_load(&h->nativeModelMaxHeight),
	        "an identity worker published a neural raster");
	worker_quit(&worker, &channel);

	worker_start(&worker, executable, &channel, log, nullptr, nullptr);
	require(atomic_load(&h->nativeTier) == 1080, "a restarted worker reset the live tier");
	worker_quit(&worker, &channel);

	worker_start(&worker, executable, &channel, log, "--tier", "900");
	require(atomic_load(&h->nativeTier) == 900, "--tier did not replace the live tier");
	worker_quit(&worker, &channel);
	free(log);
	log = nullptr;
	channel_fini(&channel);
	pass("each tier change rebuilt the worker once; repeats and unusable tiers did not");
}

/** @brief --once exits after its one answer, with status 1 when that answer failed.
 *
 * @param executable dlsslopd.
 */
static void
once (char const *executable)
{
	struct channel channel;
	channel_init(&channel, "once.bin");
	struct ShmHeader *const h = channel.shm.h;
	char *log = scratch_path("once.log", nullptr);
	bool const outcomes[] = {false, true};
	for (size_t i = 0; i < sizeof outcomes / sizeof *outcomes; ++i) {
		bool const good = outcomes[i];
		struct worker worker;
		worker_start(&worker, executable, &channel, log, "--once", nullptr);
		uint32_t const request = channel_publish(&channel, true, 8, good ? WIDTH : kMaxW + 1);
		require(channel_answered(&channel, request, true) == good, "--once answered its request wrongly");
		int const status = worker_status(&worker);
		worker_require(&worker, status == (good ? 0 : 1), "--once exit status after a %s request",
		               good ? "successful" : "failed");
		require(atomic_load(&h->helperState) == (good ? kHelperStopped : kHelperModelFailed),
		        "--once left the wrong helper state");
		require(atomic_load(&h->seq_req) == request, "--once consumed another request");
	}
	free(log);
	log = nullptr;
	channel_fini(&channel);
	pass("--once exited after one answer, reporting a failed one");
}

/** @brief Removes the test's directory: the handler that exit() runs. */
static void
remove_scratch (void)
{
	if (scratch && !support_remove_tree(scratch))
		fprintf(stderr, "worker channel: cannot remove %s\n", scratch);
	free(scratch);
	scratch = nullptr;
}

int
main (int    argc,
      char **argv)
{
	if (argc != 2) {
		fputs("usage: worker-channel-test DLSSLOPD\n", stderr);
		return 2;
	}
	// mkdtemp() makes the private (0700) directory that the worker requires.
	char const *const tmp = getenv("TMPDIR");
	scratch = support_temp_dir(tmp && *tmp ? tmp : "/tmp", "dlsslop-worker-channel", nullptr);
	if (!scratch) {
		perror("mkdtemp");
		return 1;
	}
	if (atexit(remove_scratch)) {
		remove_scratch();
		fputs("worker channel: atexit failed\n", stderr);
		return 1;
	}
	// Workers read no config file but the test's own.
	require(!setenv("XDG_CONFIG_HOME", scratch, 1), "setenv failed");
	restart(argv[1]);
	rejection(argv[1]);
	controls(argv[1]);
	once(argv[1]);
	live_settings(argv[1]);
	tiers(argv[1]);
	return 0;
}
