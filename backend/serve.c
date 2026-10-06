/** @file
 *
 * Serving shared-memory requests, whatever the engine: serve.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <inttypes.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "channel.h"
#include "engine.h"
#include "error.h"
#include "open.h"
#include "options.h"
#include "processing.h"
#include "serve.h"
#include "shm_protocol.h"
#include "trace.h"
#include "transport.h"
#include "util.h"

/** @brief Set by SIGINT and SIGTERM once serving catches them. */
static volatile sig_atomic_t stopping;

/** @brief Asks serving to stop.
 *
 * @param number The signal's number.
 */
static void
stop_handler (int number)
{
	stopping = 1;
}

void
serve_catch_signals (void)
{
	signal(SIGINT, stop_handler);
	signal(SIGTERM, stop_handler);
}

bool
serve_stopping (void)
{
	return stopping;
}

/** @brief Wakes a thread that waits on a word of the channel.
 *
 * @param word The word.
 */
static void
wake (_Atomic(uint32_t) *word)
{
	syscall(SYS_futex, word, FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

void
serve_publish_raster (struct options const *o,
                      struct ShmHeader     *h,
                      uint32_t              tier)
{
	bool const neural = !o->test_identity;
	atomic_store(&h->nativeModelMaxWidth, neural ? ShmNativeTier(tier)->width : 0);
	atomic_store(&h->nativeModelMaxHeight, neural ? tier : 0);
}

/** @brief The channel's heartbeat, counted every 100 ms from heartbeat_init() until
 *         heartbeat_fini(), so that the layer sees a live daemon even while it builds a network.
 *         stopping is as wide as a pointer, which fills the padding. */
struct heartbeat {
	struct ShmHeader *h;        //!< The channel's header.
	pthread_t         thread;   //!< The thread that counts it.
	atomic_uintptr_t  stopping; //!< Whether heartbeat_fini() asked the thread to stop: 1 if it did.
};

/** @brief Counts the heartbeat until asked to stop.
 *
 * @param self The heartbeat.
 * @return     nullptr.
 */
static void *
beat (void *self)
{
	struct heartbeat *const heartbeat = self;
	while (!atomic_load_explicit(&heartbeat->stopping, memory_order_relaxed)) {
		atomic_fetch_add_explicit(&heartbeat->h->heartbeat, 1, memory_order_relaxed);
		struct timespec wait = {0, 100000000};
		while (nanosleep(&wait, &wait) && errno == EINTR) {}
	}
	return nullptr;
}

/** @brief Starts the heartbeat, in place: its thread holds its address.
 *
 * @param dest The heartbeat; nothing to finish on a failure.
 * @param h    The channel's header.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
heartbeat_init (struct heartbeat *dest,
                struct ShmHeader *h,
                struct error     *e)
{
	dest->h = h;
	atomic_init(&dest->stopping, 0);
	int const err = pthread_create(&dest->thread, nullptr, beat, dest);
	if (err) {
		char buf[64];
		return error_fail(e, "start the heartbeat: %s", strerror_r(err, buf, sizeof buf));
	}
	return ERROR_NONE;
}

/** @brief Stops the heartbeat and waits for its thread.
 *
 * @param heartbeat The heartbeat.
 */
static void
heartbeat_fini (struct heartbeat *heartbeat)
{
	atomic_store_explicit(&heartbeat->stopping, 1, memory_order_relaxed);
	(void)pthread_join(heartbeat->thread, nullptr);
}

/** @brief Idle exit: stops taking requests before leaving. A layer that reads Stopped from now on
 *         sends none, and one whose request got in first is served.
 *
 * @param h       The channel's header.
 * @param request The latest request's sequence number.
 * @return        true when none got in.
 */
static bool
retire (struct ShmHeader *h,
        uint32_t          request)
{
	atomic_store(&h->helperState, kHelperStopped);
	if (atomic_load(&h->seq_req) == request)
		return true;
	atomic_store_explicit(&h->helperState, kHelperRunning, memory_order_release);
	return false;
}

bool
serve_wait (struct options const *o,
            struct ShmHeader     *h,
            uint32_t              number,
            uint64_t              active)
{
	struct timespec const timeout = {0, 100000000};
	syscall(SYS_futex, &h->seq_req, FUTEX_WAIT, number, &timeout, nullptr, 0);
	if (!o->idle_exit || atomic_load_explicit(&h->seq_req, memory_order_acquire) != number
	    || now_ns() - active < o->idle_exit * UINT64_C(1000000000) || !retire(h, number))
		return false;
	fprintf(stderr, "no request for %" PRIu32 " s; stopping\n", o->idle_exit);
	return true;
}

enum error_code
serve_read_request (struct options const *o,
                    struct ShmHeader     *h,
                    struct serve_request *r,
                    uint32_t              max_passes,
                    uint32_t             *previous_passes,
                    struct error         *e)
{
	if (!r->width || !r->height || r->width > kMaxW || r->height > kMaxH)
		return error_reject(e, "unsupported request dimensions");
	enum error_code const code = processing_read(h, &r->settings, e);
	if (code)
		return code;
	r->bytes = (size_t)r->width * r->height * (r->settings.fp16 ? 8 : 4);
	// A live control change takes effect on the next request;
	// never shorten or extend a chain partway through a frame.
	r->passes = ShmPasses(h);
	if (r->passes > max_passes) {
		r->passes = max_passes;
		atomic_store(&h->passes, max_passes);
	}
	if (!o->test_identity && r->passes != *previous_passes) {
		fprintf(stderr, "neural passes=%" PRIu32 "; one final composition per frame\n", r->passes);
		*previous_passes = r->passes;
	}
	return ERROR_NONE;
}

void
serve_answered (struct ShmHeader           *h,
                struct serve_request const *r,
                struct engine_times const  *times,
                uint64_t                    frames)
{
	atomic_store(&h->answeredW, r->width);
	atomic_store(&h->answeredH, r->height);
	atomic_store(&h->helperEvalMsBits, FloatToBits(times->inference_ms));
	atomic_store(&h->helperUploadMsBits, FloatToBits(times->upload_ms));
	atomic_store(&h->helperReadbackMsBits, FloatToBits(times->readback_ms));
	atomic_store(&h->helperFrames, frames);
	atomic_store(&h->seq_ok, r->number);
	atomic_store_explicit(&h->seq_resp, r->number, memory_order_release);
	wake(&h->seq_resp);
}

void
serve_failed (struct ShmHeader *h,
              uint32_t          number)
{
	atomic_store(&h->answeredW, 0);
	atomic_store(&h->answeredH, 0);
	atomic_store_explicit(&h->seq_resp, number, memory_order_release);
	wake(&h->seq_resp);
}

/** @brief serve_run() with the channel mapped, the trace directory taken, the heartbeat running
 *         and the tier published: opens the socket for offers, then serves with the engine that
 *         --backend selects.
 *
 * @param o       The options.
 * @param tier    The tier.
 * @param mapping The channel.
 * @param traces  The trace directory, or nullptr.
 * @param e       Receives the words for what ended serving, or nullptr.
 * @return        ERROR_NONE, or the code of what ended serving.
 */
static enum error_code
serve_engine (struct options              *o,
              uint32_t                     tier,
              struct mapping              *mapping,
              struct trace_requests const *traces,
              struct error                *e)
{
	struct transport_listener transport;
	enum error_code code = transport_listener_init(&transport, o->shm, o->shm_length,
	                                               !o->test_identity && !o->cpu_codec, e);
	if (code)
		return code;
	struct serving const serving = {mapping, &transport, traces};
	code = open_run_engine(o, tier, &serving, e);
	transport_listener_fini(&transport);
	return code;
}

/** @brief serve_run() with the channel mapped and the trace directory, if any, taken.
 *
 * @param o       The options.
 * @param mapping The channel.
 * @param traces  The trace directory, or nullptr.
 * @param e       Receives the words for what ended serving, or nullptr.
 * @return        ERROR_NONE, or the code of what ended serving.
 */
static enum error_code
run_traced (struct options              *o,
            struct mapping              *mapping,
            struct trace_requests const *traces,
            struct error                *e)
{
	struct ShmHeader *const h = mapping->channel.h;
	// An explicit tier replaces the channel's; otherwise a usable live one stays.
	uint32_t const live = atomic_load(&h->nativeTier);
	uint32_t const tier = o->tier ? o->tier : (ShmNativeTier(live) ? live : kNativeDefaultTier);
	atomic_store(&h->nativeTier, tier);
	if (o->passes)
		atomic_store(&h->passes, o->passes);
	atomic_store(&h->compositionBypass, o->test_identity ? 1 : 0);
	serve_publish_raster(o, h, tier);
	struct heartbeat heartbeat;
	enum error_code code = heartbeat_init(&heartbeat, h, e);
	if (code)
		return code;
	mapping_reason(mapping, o->test_identity ? "IDENTITY TEST: inference disabled" : "initializing the network");
	// The channel reports the words too.
	struct error why;
	code = serve_engine(o, tier, mapping, traces, &why);
	if (code) {
		mapping_reason(mapping, why.what);
		atomic_store(&h->helperState, kHelperModelFailed);
		if (e)
			*e = why;
	}
	heartbeat_fini(&heartbeat);
	return code;
}

/** @brief serve_run() with the channel mapped.
 *
 * @param o       The options.
 * @param mapping The channel.
 * @param e       Receives the words for what ended serving, or nullptr.
 * @return        ERROR_NONE, or the code of what ended serving.
 */
static enum error_code
run_mapped (struct options *o,
            struct mapping *mapping,
            struct error   *e)
{
	if (!o->trace_dir_length)
		return run_traced(o, mapping, nullptr, e);
	struct trace_requests traces;
	enum error_code code = trace_requests_init(&traces, o->trace_dir, o->trace_dir_length, o->shm,
	                                           o->shm_length, e);
	if (code)
		return code;
	code = run_traced(o, mapping, &traces, e);
	trace_requests_fini(&traces);
	return code;
}

enum error_code
serve_run (struct options *o,
           struct error   *e)
{
	struct mapping mapping;
	enum error_code code = mapping_init(&mapping, o->shm, o->shm_length, e);
	if (code)
		return code;
	code = run_mapped(o, &mapping, e);
	mapping_fini(&mapping);
	return code;
}
