/** @file
 *
 * Serving shared-memory requests: what serving does whatever the engine. serve.c defines the
 * functions; run_engine.inc serves with each engine.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_SERVE_H_
#define DLSSLOP_AMD_BACKEND_SERVE_H_

#include <stddef.h>
#include <stdint.h>

#include "engine.h"
#include "error.h"
#include "processing.h"
#include "shm_protocol.h"

struct mapping;
struct options;
struct trace_requests;
struct transport_listener;

/** @brief What serving serves with: the channel, the socket on which offers arrive, and the trace
 *         directory. */
struct serving {
	struct mapping                  *mapping;   //!< The channel, mapped.
	struct transport_listener const *transport; //!< The socket on which the layer offers its frames.
	struct trace_requests const     *traces;    //!< The trace directory, or nullptr without --trace-dir.
};

/** @brief One request as the channel states it: serve_read_request() checks it. */
struct serve_request {
	size_t                     bytes;            //!< The bytes of each of its frames.
	struct processing_settings settings;         //!< Its settings.
	uint32_t                   number;           //!< Its sequence number.
	uint32_t                   control_sequence; //!< controlSeq, sampled before any setting.
	uint32_t                   tuning_sequence;  //!< tuningSeq, sampled then.
	uint32_t                   held_input;       //!< holdFrame then, for a trace; else 0.
	uint32_t                   width;            //!< Its width.
	uint32_t                   height;           //!< Its height.
	uint32_t                   passes;           //!< Its passes.
};

/** @brief Serves the channel until stopped, told to quit or idle.
 *
 * Maps the channel, takes the trace directory, publishes the tier and the raster, starts the
 * heartbeat, opens the socket for offers, then opens the engine that --backend selects and serves
 * with it (open_run_engine()). A failure that ends serving is reported in the channel too.
 *
 * @param o The options; the HIP device selected is recorded in them.
 * @param e Receives the words for what ended serving, or nullptr.
 * @return  ERROR_NONE, or the code of what ended serving.
 */
extern enum error_code
serve_run (struct options *o,
           struct error   *e);

/** @brief From the ready announcement on, SIGINT and SIGTERM stop serving gracefully rather than
 *         terminate the daemon (serve_stopping()). */
extern void
serve_catch_signals (void);

/** @brief Whether SIGINT or SIGTERM asked serving to stop.
 *
 * @return true if one did.
 */
extern bool
serve_stopping (void);

/** @brief Publishes the neural raster of a tier, or none in the identity mode.
 *
 * The layer must know the real neural raster before building its proxy. Otherwise it mistakes a
 * worker-upscaled answer for native-resolution output and skips its detail-preserving composition
 * branch. The identity mode publishes none: the mapping may retain a preceding neural worker's.
 *
 * @param o    The options.
 * @param h    The channel's header.
 * @param tier The tier.
 */
extern void
serve_publish_raster (struct options const *o,
                      struct ShmHeader     *h,
                      uint32_t              tier);

/** @brief Waits up to 100 ms for a request after the latest, then decides on --idle-exit.
 *
 * The layer's futex wake ends the wait; its timeout keeps the loop live for old clients, signals
 * and quit, with no GPU polling. A request that arrived during the wait is served, however late.
 * With no request for --idle-exit seconds since the latest work, serving stops taking requests
 * (helperState Stopped), unless one got in first.
 *
 * @param o      The options.
 * @param h      The channel's header.
 * @param number The latest request's sequence number.
 * @param active When the latest work ended: an answer, a rebuild, or readiness, in nanoseconds of
 *               CLOCK_MONOTONIC.
 * @return       true when serving stops for idleness.
 */
extern bool
serve_wait (struct options const *o,
            struct ShmHeader     *h,
            uint32_t              number,
            uint64_t              active);

/** @brief Reads and checks a request's settings. A pass count beyond what the engine runs is
 *         replaced in the channel with the most it can, as an unusable tier is.
 *
 * @param o               The options.
 * @param h               The channel's header.
 * @param r               The request, whose sequence numbers and extent are sampled; receives its
 *                        settings, its frames' bytes and its passes.
 * @param max_passes      The most passes the engine runs.
 * @param previous_passes The passes logged last; updated when they change.
 * @param e               Receives the words of a rejection, or nullptr.
 * @return                ERROR_NONE, or ERROR_REJECTED.
 */
extern enum error_code
serve_read_request (struct options const *o,
                    struct ShmHeader     *h,
                    struct serve_request *r,
                    uint32_t              max_passes,
                    uint32_t             *previous_passes,
                    struct error         *e);

/** @brief Publishes a request's answer: its extent, the engine's times and the frames answered,
 *         then seq_ok and seq_resp, and wakes the layer.
 *
 * @param h      The channel's header.
 * @param r      The request.
 * @param times  The engine's times for it.
 * @param frames The frames answered, this one included.
 */
extern void
serve_answered (struct ShmHeader           *h,
                struct serve_request const *r,
                struct engine_times const  *times,
                uint64_t                    frames);

/** @brief Publishes a failed request: no extent, then seq_resp, and wakes the layer.
 *
 * @param h      The channel's header.
 * @param number The request's sequence number.
 */
extern void
serve_failed (struct ShmHeader *h,
              uint32_t          number);

#endif /* DLSSLOP_AMD_BACKEND_SERVE_H_ */
