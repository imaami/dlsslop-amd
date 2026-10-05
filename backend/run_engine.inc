/** @file
 *
 * What an engine runs, written once for every engine: serving, the offline mode, --diagnose and
 * the self-test. A file that includes it compiles it for one engine: it includes the engine's
 * header and defines
 *
 * - ENGINE, the engine's prefix, such as vulkan_engine;
 * - ENGINE_NAME, ENGINE_MAX_PASSES and ENGINE_REBUILDS_FOR_TIER, the engine's traits (engine.h).
 *
 * It defines run_ENGINE() (open.h) and undefines those macros at its end. It has no include guard:
 * each file includes it once, and no header does.
 */
// SPDX-License-Identifier: MIT
#include <inttypes.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "channel.h"
#include "engine.h"
#include "error.h"
#include "files.h"
#include "geometry.h"
#include "open.h"
#include "options.h"
#include "processing.h"
#include "serve.h"
#include "shm_protocol.h"
#include "trace.h"
#include "transport.h"
#include "util.h"

#if !defined ENGINE || !defined ENGINE_NAME || !defined ENGINE_MAX_PASSES || !defined ENGINE_REBUILDS_FOR_TIER
# error "run_engine.inc needs ENGINE, ENGINE_NAME, ENGINE_MAX_PASSES and ENGINE_REBUILDS_FOR_TIER"
#endif

/** @brief A and B pasted, once both are expanded. */
#define RUN_ENGINE_PASTE(a, b) RUN_ENGINE_PASTE_(a, b)
#define RUN_ENGINE_PASTE_(a, b) a##b

/** @brief The engine's operation OP: ENGINE_OP. */
#define E(op) RUN_ENGINE_PASTE(ENGINE, _##op)

/** @brief Imports an offered proxy/answer pair into the slot of its generation, or else the
 *         oldest one.
 *
 * @param engine The engine.
 * @param offer  The offer.
 * @param fds    The offered memory; the engine owns each descriptor it imported and sets it to -1,
 *               and the caller closes the rest.
 * @return       true if the pair is in a slot.
 */
static bool
import_offer (struct ENGINE                  *engine,
              struct ShmTransportOffer const *offer,
              int                             fds[2])
{
	if (!offer->generation)
		return false;
	ptrdiff_t const slot = engine_slots_choose(&engine->slots, offer->generation);
	if (!E(import_into)(engine, slot, offer, fds))
		return false;
	engine_slots_hold(&engine->slots, slot, offer);
	return true;
}

/** @brief Imports every pending offer and answers each on its own connection.
 *
 * @param listener The socket on which offers arrive.
 * @param engine   The engine.
 */
static void
accept_offers (struct transport_listener const *listener,
               struct ENGINE                   *engine)
{
	if (listener->socket < 0)
		return;
	for (int peer = -1; (peer = accept4(listener->socket, nullptr, nullptr, SOCK_CLOEXEC)) >= 0;) {
		struct ShmTransportOffer offer = {0};
		int fds[2] = {-1, -1};
		// A start or liveness probe sends nothing.
		if (transport_receive_offer(peer, &offer, fds)) {
			transport_answer(peer, import_offer(engine, &offer, fds));
			// What the engine did not import.
			for (unsigned i = 0; i < 2; ++i) {
				if (fds[i] >= 0) {
					// Never used: close() has nothing to report.
					close(fds[i]);
					fds[i] = -1;
				}
			}
		}
		// Only read and answered: close() has nothing to report.
		close(peer);
		peer = -1;
	}
}

/** @brief Between frames: rebuilds the network once for each tier a controller stores that differs
 *         from the active one.
 *
 * The layer presents its own frames while helperState reads Starting, then offers its device-local
 * frames to the rebuilt engine. An engine that does not rebuild for a tier only follows it. An
 * unusable tier is overwritten with the active one. A failed rebuild ends serving, leaving the
 * active tier for the next daemon.
 *
 * @param engine  The engine.
 * @param o       The options.
 * @param mapping The channel.
 * @param ready   The status text of a ready daemon.
 * @param rebuilt Receives whether the engine was rebuilt.
 * @param e       Receives the words for what failed, or nullptr.
 * @return        ERROR_NONE, or the code of the rebuild's failure.
 */
static enum error_code
follow_tier (struct ENGINE        *engine,
             struct options const *o,
             struct mapping       *mapping,
             char const           *ready,
             bool                 *rebuilt,
             struct error         *e)
{
	struct ShmHeader *const h = mapping->channel.h;
	*rebuilt = false;
	uint32_t const wanted = atomic_load(&h->nativeTier);
	uint32_t const active = engine->tier;
	if (wanted == active)
		return ERROR_NONE;
	if (!ShmNativeTier(wanted)) {
		atomic_store(&h->nativeTier, active);
		return ERROR_NONE;
	}
	enum error_code code;
	if (!ENGINE_REBUILDS_FOR_TIER) {
		code = E(retier)(engine, wanted, e);
		if (code)
			return code;
		fprintf(stderr, "neural tier %" PRIu32 " -> %" PRIu32 "\n", active, wanted);
		serve_publish_raster(o, h, wanted);
		return ERROR_NONE;
	}
	atomic_store(&h->helperState, kHelperStarting);
	atomic_store(&h->modelUp, 0);
	fprintf(stderr, "neural tier %" PRIu32 " -> %" PRIu32 ": rebuilding\n", active, wanted);
	// Sized for the longest tier.
	char reason[sizeof "rebuilding for neural tier 4294967295"];
	snprintf(reason, sizeof reason, "rebuilding for neural tier %" PRIu32, wanted);
	mapping_reason(mapping, reason);
	code = E(retier)(engine, wanted, e);
	if (code) {
		atomic_store(&h->nativeTier, active);
		return code;
	}
	serve_publish_raster(o, h, wanted);
	mapping_reason(mapping, ready);
	atomic_store(&h->modelUp, o->test_identity ? 0 : 1);
	atomic_store_explicit(&h->helperState, kHelperRunning, memory_order_release);
	*rebuilt = true;
	return ERROR_NONE;
}

/** @brief A traced request's metadata, as JSON.
 *
 * @param r        The request.
 * @param engine   The engine.
 * @param h        The channel's header.
 * @param proxy    The request's proxy, which the engine reads back.
 * @param metadata Receives the JSON text.
 * @param e        Receives the words for what failed, or nullptr.
 * @return         ERROR_NONE, or the code of what failed.
 */
static enum error_code
describe_frame (struct serve_request const *r,
                struct ENGINE              *engine,
                struct ShmHeader const     *h,
                uint8_t const              *proxy,
                char                        metadata[TRACE_METADATA_BYTES],
                struct error               *e)
{
	struct trace_metadata m = {0};
	m.frame_seq = r->number;
	m.control_seq = r->control_sequence;
	m.tuning_seq = r->tuning_sequence;
	m.control_seq_end = atomic_load(&h->controlSeq);
	m.tuning_seq_end = atomic_load(&h->tuningSeq);
	m.held_input = r->held_input;
	m.held_input_end = atomic_load(&h->holdFrame);
	uint8_t *copy = malloc(r->bytes);
	if (!copy)
		return error_fail(e, "out of memory");
	enum error_code code = E(read_back)(engine, copy, proxy, r->bytes, e);
	if (!code) {
		uint64_t hash = UINT64_C(14695981039346656037);
		for (size_t i = 0; i < r->bytes; ++i)
			hash = (hash ^ copy[i]) * UINT64_C(1099511628211);
		m.source_proxy_hash = hash;
	}
	free(copy);
	copy = nullptr;
	if (code)
		return code;
	m.passes = r->passes;
	code = geometry_init(&m.geometry, r->width, r->height, engine->tier, e);
	if (code)
		return code;
	struct processing_settings const *const settings = &r->settings;
	m.fp16_proxy = settings->fp16;
	m.fp16_feedback = settings->precision16;
	m.motion = settings->motion;
	m.intensity = settings->tuning.intensity;
	m.local_tone = settings->tuning.tone;
	m.local_structure = settings->tuning.structure;
	m.sharpness = settings->tuning.sharpness;
	m.color_preserve = settings->color_preserve;
	return trace_metadata_json(&m, metadata, e);
}

/** @brief Writes a request's answer.
 *
 * @param r        The request.
 * @param engine   The engine.
 * @param mapping  The channel.
 * @param trace    The request's trace, or nullptr.
 * @param metadata Receives a traced request's metadata.
 * @param e        Receives the words for why not, or nullptr.
 * @return         ERROR_NONE; ERROR_REJECTED or ERROR_DROPPED for a frame that serving answers as
 *                 failed and goes on; ERROR_FAILED.
 */
static enum error_code
process (struct serve_request const *r,
         struct ENGINE              *engine,
         struct mapping             *mapping,
         struct frame_trace         *trace,
         char                        metadata[TRACE_METADATA_BYTES],
         struct error               *e)
{
	struct ShmHeader *const h = mapping->channel.h;
	struct processing_settings const *const settings = &r->settings;
	// The request's frames: an imported device-local pair, or the channel's slots.
	struct engine_frames io = {mapping->input, mapping->output, -1};
	uint32_t const generation = atomic_load(&h->transportGen);
	if (generation) {
		ptrdiff_t const slot = engine_slots_find(&engine->slots, generation, r->bytes);
		if (slot < 0) {
			atomic_store(&h->transportMiss, generation);
			return error_reject(e, "request names device-local frames this worker has not imported");
		}
		io = E(frames_of)(engine, slot);
	} else {
		E(pin)(engine, mapping->input, mapping->output, r->bytes);
	}
	enum error_code code;
	// A frame of a new shape needs a build first: seconds for a new extent,
	// in which the layer presents its own frames rather than waiting for this
	// one, or milliseconds for a reshape that keeps the weights.
	if (!E(fits)(engine, r->width, r->height, r->passes, settings)) {
		// A shape the engine does not take is rejected before a build: serving goes on as it was.
		code = E(admit)(engine, r->width, r->height, r->passes, settings, e);
		if (code)
			return code;
		atomic_store(&h->helperState, kHelperStarting);
		fprintf(stderr, "building the network for %" PRIu32 "x%" PRIu32 "%s\n", r->width, r->height,
		        settings->fp16 ? " FP16" : "");
		code = E(reshape)(engine, r->width, r->height, r->passes, settings, e);
		if (code)
			return code;
		atomic_store_explicit(&h->helperState, kHelperRunning, memory_order_release);
	}
	code = E(infer)(engine, &io, r->width, r->height, r->passes, settings, trace, e);
	if (code)
		return code;
	if (atomic_load_explicit(&h->seq_req, memory_order_acquire) != r->number)
		return error_reject(e, "request changed during inference; old answer discarded");
	if (!trace)
		return ERROR_NONE;
	return describe_frame(r, engine, h, io.proxy, metadata, e);
}

/** @brief Answers requests until stopped, told to quit, idle, or failed.
 *
 * A rejected request is answered as failed and serving goes on; any other failure ends it, as does
 * any failure with --once: a HIP or engine fault is never silently replaced by fake output. A
 * request claimed for a trace is pending.
 *
 * @param o       The options.
 * @param s       What serving serves with.
 * @param engine  The engine.
 * @param pending Receives a request's trace once claimed; none until then.
 * @param e       Receives the words for what ended serving, or nullptr.
 * @return        ERROR_NONE, or the code of what ended serving.
 */
static enum error_code
serve_frames (struct options const *o,
              struct serving const *s,
              struct ENGINE        *engine,
              struct frame_trace   *pending,
              struct error         *e)
{
	struct ShmHeader *const h = s->mapping->channel.h;
	enum error_code code = E(prepare)(engine, e);
	if (code)
		return code;
	// Only serving stops gracefully, from the ready announcement on.
	// Before it, and in every other mode, SIGINT and SIGTERM terminate.
	serve_catch_signals();
	atomic_store(&h->modelUp, o->test_identity ? 0 : 1);
	atomic_store_explicit(&h->helperState, kHelperRunning, memory_order_release);
	char const *const ready = o->test_identity ? "IDENTITY TEST: no neural rendering"
	                                           : "native " ENGINE_NAME " ready; display-encoded RGBA8/FP16 proxy";
	mapping_reason(s->mapping, ready);
	fprintf(stderr, "worker ready: %s%s\n", o->shm, o->test_identity ? " [IDENTITY TEST]" : "");
	if (!o->test_identity) {
		char processing[ENGINE_DESCRIPTION_BYTES];
		fprintf(stderr, "neural tier=%" PRIu32 "; %s; native-resolution Vulkan composition; "
		        "live controls enabled\n", engine->tier, E(describe_processing)(engine, processing));
	}
	uint64_t frames_answered = 0;
	uint32_t previous_passes = 0;
	uint32_t last = atomic_load_explicit(&h->seq_resp, memory_order_acquire);
	// When the latest work ended: an answer, a rebuild, or readiness.
	uint64_t active = now_ns();
	// The failure reported last, which the channel shows until serving recovers; empty for none.
	struct error failure = {0};
	while (!serve_stopping() && !atomic_load_explicit(&h->quit, memory_order_relaxed)) {
		accept_offers(s->transport, engine);
		if (s->traces && !pending->path) {
			struct error why;
			if (trace_requests_take(s->traces, pending, &why))
				fprintf(stderr, "diagnostic request rejected: %s\n", why.what);
			else if (pending->path)
				atomic_fetch_add(&h->controlSeq, 1);
		}
		uint32_t const number = atomic_load_explicit(&h->seq_req, memory_order_acquire);
		if (number == last) {
			bool rebuilt;
			code = follow_tier(engine, o, s->mapping, ready, &rebuilt, e);
			if (code)
				return code;
			if (rebuilt)
				active = now_ns();
			if (serve_wait(o, h, number, active))
				break;
			continue;
		}
		// Writers store a setting before bumping controlSeq, so sample the
		// generations after this loop's own bump (the trace claim) and
		// before reading any setting, the dimensions or the input; sample
		// again after inference.
		struct serve_request request;
		request.number = number;
		request.control_sequence = atomic_load(&h->controlSeq);
		request.tuning_sequence = atomic_load(&h->tuningSeq);
		request.held_input = pending->path ? atomic_load(&h->holdFrame) : 0;
		request.width = atomic_load(&h->width);
		request.height = atomic_load(&h->height);
		last = number;
		char metadata[TRACE_METADATA_BYTES];
		// The trace, the channel and the log need the words of a failed frame.
		struct error why;
		code = serve_read_request(o, h, &request, ENGINE_MAX_PASSES, &previous_passes, &why);
		if (!code)
			code = process(&request, engine, s->mapping, pending->path ? pending : nullptr, metadata, &why);
		if (!code) {
			// Serving recovered: the failure is no longer current.
			if (failure.what[0]) {
				failure.what[0] = '\0';
				mapping_reason(s->mapping, ready);
			}
			serve_answered(h, &request, &engine->times, ++frames_answered);
			if (pending->path)
				frame_trace_finish(pending, metadata, nullptr);
			if (o->once)
				break;
		} else {
			if (pending->path) {
				// Sized for the largest sequence number.
				char summary[sizeof "{\"frame_seq\":4294967295}"];
				snprintf(summary, sizeof summary, "{\"frame_seq\":%" PRIu32 "}", number);
				frame_trace_finish(pending, summary, why.what);
			}
			// Report a persistent rejection once, not every frame. A frame that the Vulkan network
			// dropped is not serving's failure: the network logs drops, at most every 10 s.
			if (code != ERROR_DROPPED && strcmp(failure.what, why.what)) {
				failure = why;
				mapping_reason(s->mapping, failure.what);
				fprintf(stderr, "frame %" PRIu32 " failed: %s\n", number, failure.what);
			}
			serve_failed(h, number);
			if (o->once || code == ERROR_FAILED) {
				if (e)
					*e = why;
				return code;
			}
		}
		active = now_ns();
	}
	// Stopped before the teardown, which can take a while and unlinks a
	// socket this daemon bound: a layer that offers or presents from now on
	// leaves its frames for the next daemon instead of waiting on this one.
	atomic_store(&h->helperState, kHelperStopped);
	return ERROR_NONE;
}

/** @brief Serves until stopped, told to quit, idle, or failed (serve_frames()), and publishes a
 *         request claimed for a trace however serving ends.
 *
 * @param o      The options.
 * @param s      What serving serves with.
 * @param engine The engine.
 * @param e      Receives the words for what ended serving, or nullptr.
 * @return       ERROR_NONE, or the code of what ended serving.
 */
static enum error_code
serve (struct options const *o,
       struct serving const *s,
       struct ENGINE        *engine,
       struct error         *e)
{
	struct frame_trace pending = {0};
	enum error_code const code = serve_frames(o, s, engine, &pending, e);
	frame_trace_fini(&pending);
	return code;
}

/** @brief The offline mode: one RGBA8 image from a file to a file.
 *
 * @param o      The options: the input and output files and the image's extent.
 * @param engine The engine, prepared.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
offline (struct options const *o,
         struct ENGINE        *engine,
         struct error         *e)
{
	uint32_t const wanted = o->passes ? o->passes : kNativeDefaultPasses;
	uint32_t const passes = ENGINE_MAX_PASSES < wanted ? ENGINE_MAX_PASSES : wanted;
	size_t const bytes = (size_t)o->width * o->height * 4;
	struct files_data input;
	if (files_read(&input, o->input, nullptr) || input.size != bytes) {
		files_data_fini(&input);
		return error_fail(e, "offline input size must equal width * height * 4");
	}
	uint8_t *result = calloc(bytes, 1);
	if (!result) {
		files_data_fini(&input);
		return error_fail(e, "out of memory");
	}
	struct engine_frames const io = {input.bytes, result, -1};
	struct processing_settings const settings = processing_settings();
	enum error_code code = E(infer)(engine, &io, o->width, o->height, passes, &settings, nullptr, e);
	if (!code) {
		if (files_write(o->output, result, bytes, nullptr))
			code = error_fail(e, "write offline output");
		else
			fprintf(stderr, "offline: passes=%" PRIu32 " upload=%.2f ms, network=%.2f ms, "
			        "readback=%.2f ms\n", passes, engine->times.upload_ms, engine->times.inference_ms,
			        engine->times.readback_ms);
	}
	free(result);
	result = nullptr;
	files_data_fini(&input);
	return code;
}

enum error_code
RUN_ENGINE_PASTE(run_, ENGINE) (struct options const *o,
                                struct ENGINE        *engine,
                                struct serving const *serving,
                                struct error         *e)
{
	if (serving)
		return serve(o, serving, engine, e);
	if (o->diagnose) {
		char device[ENGINE_DESCRIPTION_BYTES];
		fprintf(stderr, "selected %s\n", E(describe_device)(engine, device));
		return ERROR_NONE;
	}
	enum error_code const code = E(prepare)(engine, e);
	if (code)
		return code;
	if (o->self_test)
		return E(self_test)(engine, o, e);
	return offline(o, engine, e);
}

#undef E
#undef RUN_ENGINE_PASTE_
#undef RUN_ENGINE_PASTE
#undef ENGINE_REBUILDS_FOR_TIER
#undef ENGINE_MAX_PASSES
#undef ENGINE_NAME
#undef ENGINE
