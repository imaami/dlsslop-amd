/** @file
 *
 * Opt-in diagnostics (--trace-dir): one requested frame's stages as PFM images and a JSON summary.
 * trace.c defines the functions.
 *
 * A client writes a request file, DIR/request, that holds one unique token. The daemon claims it
 * once by an atomic rename, writes each stage as DIR/TOKEN/NAME.pfm, then DIR/TOKEN/summary.json,
 * and last the marker DIR/TOKEN.done. Only one daemon serves a trace directory, and its
 * DIR/owner.json says which. The files' formats are independent of the shared-memory protocol.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_TRACE_H_
#define DLSSLOP_AMD_BACKEND_TRACE_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "kernel_args.h"
#include "shm_protocol.h"

/** @brief The schema of a frame's metadata.
 *
 * Schema 2 guarantees the held-state, control-sequence and source-hash evidence needed to compare
 * traces of the same frozen input. owner.json advertises it before a diagnostic client changes
 * any live controls.
 */
#define TRACE_METADATA_SCHEMA 2

/** @brief The bytes of a frame's metadata text and its null (trace_metadata_json()). */
#define TRACE_METADATA_BYTES 1024

/** @brief The bytes of a stage's name and its null: "pass-NN-STAGE". */
#define TRACE_STAGE_BYTES 32

/** @brief The most stages a trace holds: the input, raw, tuned and color stages of each pass. */
#define TRACE_MAX_STAGES (4 * kMaxPasses)

/** @brief What a traced frame's summary says of the frame. */
struct trace_metadata {
	uint64_t        source_proxy_hash; //!< FNV-1a of the proxy as the engine read it.
	struct geometry geometry;          //!< Where the source sits in the network's raster.
	uint32_t        frame_seq;         //!< The request's sequence number.
	uint32_t        control_seq;       //!< controlSeq before any setting was read.
	uint32_t        tuning_seq;        //!< tuningSeq then.
	uint32_t        control_seq_end;   //!< controlSeq after inference.
	uint32_t        tuning_seq_end;    //!< tuningSeq then.
	uint32_t        held_input;        //!< holdFrame before the request was read.
	uint32_t        held_input_end;    //!< holdFrame after inference.
	float           intensity;         //!< The tuning's intensity.
	float           local_tone;        //!< Its tone.
	float           local_structure;   //!< Its structure.
	float           sharpness;         //!< Its sharpness.
	float           color_preserve;    //!< The color preservation.
	uint8_t         passes;            //!< The passes run, at most kMaxPasses.
	bool            fp16_proxy;        //!< Whether the proxy is RGBA16F.
	bool            fp16_feedback;     //!< Whether later passes take 16-bit feedback.
	bool            motion;            //!< Whether motion vectors were used.
};

/** @brief A claimed request's trace: trace_requests_take() makes one, frame_trace_image() adds its
 *         stages and frame_trace_finish() publishes it.
 *
 * A trace whose path is nullptr is none, or finished.
 */
struct frame_trace {
	char         *path;         //!< DIR/TOKEN, then room for a file's name; nullptr for none.
	size_t        length;       //!< The length of DIR/TOKEN.
	size_t        token_length; //!< The token's: DIR/TOKEN ends in it.
	size_t        stage_count;  //!< The stages written.
	struct error  error;        //!< Why a stage failed; empty while none did.
	char          stages[TRACE_MAX_STAGES][TRACE_STAGE_BYTES]; //!< The stages' names, in order.
};

/** @brief The trace directory of a daemon: trace_requests_init() takes it and
 *         trace_requests_fini() leaves it.
 */
struct trace_requests {
	char   *directory; //!< DIR, absolute; nullptr for none.
	char   *owner;     //!< DIR/owner.json, which this daemon wrote and removes; nullptr before.
	char   *request;   //!< DIR/request, which a client writes.
	char   *claimed;   //!< DIR/.request-PID, to which this daemon renames a request it claims.
	size_t  length;    //!< DIR's length.
	int     lock;      //!< DIR/.worker-lock, locked while this daemon serves, or -1.
};

/** @brief Formats a frame's metadata as JSON.
 *
 * @param m    The metadata.
 * @param text Receives the JSON text.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
trace_metadata_json (struct trace_metadata const *m,
                     char                         text[TRACE_METADATA_BYTES],
                     struct error                *e);

/** @brief Whether a request's token is safe to name its files: 1 to 64 letters, digits, dashes and
 *         underscores, and not "request", which would make DIR/request a directory.
 *
 * @param token  The token; it need not be null-terminated.
 * @param length Its length.
 * @return       true if it is.
 */
extern bool
trace_valid_token (char const *token,
                   size_t      length);

/** @brief Writes a stage's fitted picture as a PFM image: bottom-up RGB float32, a negative scale
 *         for little-endian data.
 *
 * Retains model-domain values, including signed and extended values and NaNs: this is evidence, not
 * a preview image, and no display transform is applied.
 *
 * @param file     The image's path.
 * @param data     The stage: g->width x g->height pixels of @a channels floats.
 * @param g        The geometry; its fitted picture is written.
 * @param channels 3 for RGB, 4 for RGBA.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
trace_write_pfm (char const            *file,
                 float const           *data,
                 struct geometry const *g,
                 unsigned               channels,
                 struct error          *e);

/** @brief Writes a stage of a trace, unless a stage failed: then the frame's later stages are not
 *         written. A stage that fails keeps its words for the summary.
 *
 * @param t        The trace.
 * @param name     The stage's name.
 * @param length   Its length, less than TRACE_STAGE_BYTES.
 * @param data     The stage: g->width x g->height pixels of @a channels floats.
 * @param g        The geometry; its fitted picture is written.
 * @param channels 3 for RGB, 4 for RGBA.
 */
extern void
frame_trace_image (struct frame_trace    *t,
                   char const            *name,
                   size_t                 length,
                   float const           *data,
                   struct geometry const *g,
                   unsigned               channels);

/** @brief Publishes a trace, its summary first and its marker last, and ends it.
 *
 * A trace that cannot be published is logged.
 *
 * @param t        The trace.
 * @param metadata The summary's metadata, a JSON object.
 * @param failure  Why the frame failed, which replaces any stage's words, or nullptr.
 */
extern void
frame_trace_finish (struct frame_trace *t,
                    char const         *metadata,
                    char const         *failure);

/** @brief Publishes a trace that was not finished as failed: a claimed request always gets its
 *         summary and marker, even when the daemon stops first, so the client need not wait for a
 *         timeout.
 *
 * @param t The trace, or nullptr.
 */
extern void
frame_trace_fini (struct frame_trace *t);

/** @brief Takes a trace directory, created private if it is missing, for this daemon alone, and
 *         writes its owner.json.
 *
 * @param dest       Receives the trace directory; none on a failure.
 * @param path       The directory's path.
 * @param length     The length of its path.
 * @param shm        The channel file's path, for owner.json.
 * @param shm_length The length of its path.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
trace_requests_init (struct trace_requests *dest,
                     char const            *path,
                     size_t                 length,
                     char const            *shm,
                     size_t                 shm_length,
                     struct error          *e);

/** @brief Removes the trace directory's owner.json and unlocks the directory.
 *
 * @param r The trace directory, or nullptr.
 */
extern void
trace_requests_fini (struct trace_requests *r);

/** @brief Claims the next request: its trace, with its directory DIR/TOKEN made private.
 *
 * @param r    The trace directory.
 * @param dest Receives the trace, which must be none; it stays none without a request.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED for a request that cannot be claimed or is invalid.
 */
extern enum error_code
trace_requests_take (struct trace_requests const *r,
                     struct frame_trace          *dest,
                     struct error                *e);

#endif /* DLSSLOP_AMD_BACKEND_TRACE_H_ */
