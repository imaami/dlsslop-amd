/** @file
 *
 * Opt-in diagnostics (--trace-dir): trace.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "files.h"
#include "geometry.h"
#include "trace.h"

/** @brief The room a trace's path keeps after DIR/TOKEN for a file's name: "/NAME.pfm" and the
 *         null, at the longest.
 */
#define TAIL_BYTES (TRACE_STAGE_BYTES + sizeof "/.pfm")

static_assert(sizeof "/summary.json" <= TAIL_BYTES && sizeof ".done" <= TAIL_BYTES);
static_assert(kMaxPasses <= UINT8_MAX);

/** @brief Puts an action and strerror()'s words for an errno value in an error.
 *
 * @param e      The error, or nullptr.
 * @param action What failed.
 * @param err    The errno value.
 * @return       ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
fail_errno (struct error *e,
            char const   *action,
            int           err)
{
	char buf[64];
	return error_fail(e, "%s: %s", action, strerror_r(err, buf, sizeof buf));
}

/** @brief Writes a text's characters as a JSON string's, without the quotes.
 *
 * @param f    The file.
 * @param text The text.
 */
static void
json_escaped (FILE       *f,
              char const *text)
{
	for (unsigned char const *c = (unsigned char const *)text; *c; ++c) {
		if (*c == '"' || *c == '\\') {
			fputc('\\', f);
			fputc(*c, f);
		} else if (*c < 32) {
			fprintf(f, "\\u%04x", *c);
		} else {
			fputc(*c, f);
		}
	}
}

/** @brief Writes a text as a JSON string.
 *
 * @param f    The file.
 * @param text The text.
 */
static void
json_string (FILE       *f,
             char const *text)
{
	fputc('"', f);
	json_escaped(f, text);
	fputc('"', f);
}

/** @brief A text file being written: FILE.tmp, which text_commit() renames to FILE. */
struct text {
	FILE *file;      //!< FILE.tmp, open for writing, or nullptr if it could not be opened.
	char *temporary; //!< Its path, or nullptr without memory for it.
};

/** @brief Opens a text that replaces a file once it is written whole.
 *
 * @param path   The file.
 * @param length The length of its path.
 * @return       The text; its file is nullptr if it cannot be written.
 */
static struct text
text_open (char const *path,
           size_t      length)
{
	char *const temporary = malloc(length + sizeof ".tmp");
	if (!temporary)
		return (struct text){0};
	memcpy(temporary, path, length);
	memcpy(temporary + length, ".tmp", sizeof ".tmp");
	FILE *const file = fopen(temporary, "we");
	return (struct text){file, temporary};
}

/** @brief Closes a text and puts it in its file's place.
 *
 * @param t    The text, which is closed and freed.
 * @param path The file.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
text_commit (struct text  *t,
             char const   *path,
             struct error *e)
{
	bool written = false;
	if (t->file) {
		bool const failed = ferror(t->file);
		written = !fclose(t->file) && !failed;
		t->file = nullptr;
	}
	written = written && !rename(t->temporary, path);
	free(t->temporary);
	t->temporary = nullptr;
	if (!written)
		return error_fail(e, "write diagnostic file: %s", path);
	return ERROR_NONE;
}

enum error_code
trace_metadata_json (struct trace_metadata const *m,
                     char                         text[TRACE_METADATA_BYTES],
                     struct error                *e)
{
	struct geometry const *const g = &m->geometry;
	int const n = snprintf(text, TRACE_METADATA_BYTES,
	                       "{\"trace_metadata_schema\":%d,\"frame_seq\":%" PRIu32 ",\"control_seq\":%" PRIu32
	                       ",\"tuning_seq\":%" PRIu32 ",\"control_seq_end\":%" PRIu32 ",\"tuning_seq_end\":%" PRIu32
	                       ",\"held_input\":%" PRIu32 ",\"held_input_end\":%" PRIu32 ",\"source_proxy_hash\":\"%016"
	                       PRIx64 "\",\"passes\":%" PRIu8 ",\"source_width\":%u,\"source_height\":%u,\"network_width\":%u,"
	                       "\"network_height\":%u,\"fit_x\":%u,\"fit_y\":%u,\"fit_width\":%u,\"fit_height\":%u,"
	                       "\"fp16_proxy\":%d,\"fp16_feedback\":%d,\"motion\":%d,\"intensity\":%.9g,"
	                       "\"local_tone\":%.9g,\"color_preserve\":%.9g,\"local_structure\":%.9g,\"sharpness\":%.9g}",
	                       TRACE_METADATA_SCHEMA, m->frame_seq, m->control_seq, m->tuning_seq, m->control_seq_end,
	                       m->tuning_seq_end, m->held_input, m->held_input_end, m->source_proxy_hash, m->passes,
	                       g->source_width, g->source_height, g->width, g->height, g->x, g->y, g->fit_width,
	                       g->fit_height, m->fp16_proxy, m->fp16_feedback, m->motion, m->intensity,
	                       m->local_tone, m->color_preserve, m->local_structure, m->sharpness);
	if (n < 0 || n >= TRACE_METADATA_BYTES)
		return error_fail(e, "diagnostic metadata does not fit");
	return ERROR_NONE;
}

bool
trace_valid_token (char const *token,
                   size_t      length)
{
	if (!length || length > 64)
		return false;
	for (size_t i = 0; i < length; ++i) {
		char const c = token[i];
		if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-'
		      || c == '_'))
			return false;
	}
	// DIR/request is the request file itself.
	return length != sizeof "request" - 1 || memcmp(token, "request", length);
}

enum error_code
trace_write_pfm (char const            *file,
                 float const           *data,
                 struct geometry const *g,
                 unsigned               channels,
                 struct error          *e)
{
	if (!data || (channels != 3 && channels != 4) || !geometry_fits(g))
		return error_fail(e, "invalid diagnostic image geometry");
	size_t const row_floats = (size_t)g->fit_width * 3;
	float *row = malloc(row_floats * sizeof *row);
	if (!row)
		return error_fail(e, "out of memory");

	bool written = false;
	FILE *f = fopen(file, "we");
	if (f) {
		fprintf(f, "PF\n%u %u\n%s", g->fit_width, g->fit_height,
		        __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ ? "-1.0\n" : "1.0\n");
		for (unsigned iy = g->fit_height; iy-- > 0;) {
			float const *const source = data + ((size_t)(g->y + iy) * g->width + g->x) * channels;
			for (size_t x = 0; x < g->fit_width; ++x)
				memcpy(row + x * 3, source + x * channels, 3 * sizeof *row);
			fwrite(row, sizeof *row, row_floats, f);
		}
		bool const failed = ferror(f);
		written = !fclose(f) && !failed;
		f = nullptr;
	}
	free(row);
	row = nullptr;
	if (!written)
		return error_fail(e, "write diagnostic image: %s", file);
	return ERROR_NONE;
}

void
frame_trace_image (struct frame_trace    *t,
                   char const            *name,
                   size_t                 length,
                   float const           *data,
                   struct geometry const *g,
                   unsigned               channels)
{
	if (t->error.what[0])
		return;
	if (length >= TRACE_STAGE_BYTES) {
		error_fail(&t->error, "diagnostic stage name too long: %s", name);
		return;
	}
	if (t->stage_count == TRACE_MAX_STAGES) {
		error_fail(&t->error, "too many diagnostic stages");
		return;
	}

	// DIR/TOKEN/NAME.pfm
	char *const tail = t->path + t->length;
	tail[0] = '/';
	memcpy(tail + 1, name, length);
	memcpy(tail + 1 + length, ".pfm", sizeof ".pfm");
	if (trace_write_pfm(t->path, data, g, channels, &t->error))
		return;
	memcpy(t->stages[t->stage_count++], name, length + 1);
}

/** @brief Writes a trace's summary: DIR/TOKEN/summary.json.
 *
 * @param t        The trace.
 * @param metadata The frame's metadata, a JSON object.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
write_summary (struct frame_trace *t,
               char const         *metadata,
               struct error       *e)
{
	memcpy(t->path + t->length, "/summary.json", sizeof "/summary.json");
	struct text summary = text_open(t->path, t->length + sizeof "/summary.json" - 1);
	if (summary.file) {
		FILE *const f = summary.file;
		fprintf(f, "{\"schema\":1,\"status\":\"%s\",\"encoding\":\"display-encoded network RGB; not linear "
		           "light\",\"metadata\":%s,\"error\":", t->error.what[0] ? "failed" : "complete", metadata);
		json_string(f, t->error.what);
		fputs(",\"stages\":[", f);
		for (size_t i = 0; i < t->stage_count; ++i) {
			fputs(i ? ",{\"name\":" : "{\"name\":", f);
			json_string(f, t->stages[i]);
			fputs(",\"file\":\"", f);
			json_escaped(f, t->stages[i]);
			fputs(".pfm\"}", f);
		}
		fputs("]}\n", f);
	}
	return text_commit(&summary, t->path, e);
}

/** @brief Writes a trace's marker beside its directory, DIR/TOKEN.done, which names its summary.
 *
 * A watcher on the opt-in root sees this atomic marker even though the images and the summary are
 * written one directory below its watch.
 *
 * @param t The trace.
 * @param e Receives the words for what stopped it, or nullptr.
 * @return  ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
write_marker (struct frame_trace *t,
              struct error       *e)
{
	memcpy(t->path + t->length, ".done", sizeof ".done");
	struct text marker = text_open(t->path, t->length + sizeof ".done" - 1);
	if (marker.file) {
		fwrite(t->path + t->length - t->token_length, 1, t->token_length, marker.file);
		fputs("/summary.json\n", marker.file);
	}
	return text_commit(&marker, t->path, e);
}

void
frame_trace_finish (struct frame_trace *t,
                    char const         *metadata,
                    char const         *failure)
{
	if (failure)
		error_fail(&t->error, "%s", failure);
	struct error e;
	enum error_code code = write_summary(t, metadata, &e);
	if (!code)
		code = write_marker(t, &e);
	if (code)
		fprintf(stderr, "diagnostic trace incomplete: %s\n", e.what);
	free(t->path);
	t->path = nullptr;
}

void
frame_trace_fini (struct frame_trace *t)
{
	if (!t || !t->path)
		return;
	// A stage's failure is why the frame failed; otherwise the daemon stopped.
	frame_trace_finish(t, "{}", t->error.what[0] ? nullptr : "worker stopped before a traced frame completed");
}

/** @brief Starts the trace of a claimed request: its directory, DIR/TOKEN, made private.
 *
 * @param dest         Receives the trace.
 * @param root         DIR.
 * @param root_length  Its length.
 * @param token        The request's token, checked by trace_valid_token().
 * @param token_length Its length.
 * @param e            Receives the words for what stopped it, or nullptr.
 * @return             ERROR_NONE, or ERROR_FAILED with @a dest untouched.
 */
static enum error_code
frame_trace_init (struct frame_trace *dest,
                  char const         *root,
                  size_t              root_length,
                  char const         *token,
                  size_t              token_length,
                  struct error       *e)
{
	size_t length;
	char *path = files_join(root, root_length, token, token_length, &length);
	char *const grown = path ? realloc(path, length + TAIL_BYTES) : nullptr;
	if (!grown) {
		free(path);
		path = nullptr;
		return error_fail(e, "out of memory");
	}
	path = grown;
	if (mkdir(path, 0700)) {
		int const err = errno;
		char buf[64];
		enum error_code const code = error_fail(e, "create diagnostic directory %s: %s", path,
		                                        strerror_r(err, buf, sizeof buf));
		free(path);
		path = nullptr;
		return code;
	}
	dest->path = path;
	dest->length = length;
	dest->token_length = token_length;
	dest->stage_count = 0;
	dest->error.what[0] = '\0';
	return ERROR_NONE;
}

/** @brief Writes a trace directory's owner.json: the daemon's process, its channel and the
 *         metadata's schema.
 *
 * @param owner      DIR/owner.json.
 * @param length     The length of its path.
 * @param shm        The channel file's path.
 * @param shm_length The length of its path.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
write_owner (char const   *owner,
             size_t        length,
             char const   *shm,
             size_t        shm_length,
             struct error *e)
{
	char *channel = files_absolute(shm, shm_length, nullptr);
	if (!channel)
		return error_fail(e, "out of memory");
	struct text text = text_open(owner, length);
	if (text.file) {
		fprintf(text.file, "{\"pid\":%d,\"shm\":", getpid());
		json_string(text.file, channel);
		fprintf(text.file, ",\"trace_metadata_schema\":%d}\n", TRACE_METADATA_SCHEMA);
	}
	free(channel);
	channel = nullptr;
	return text_commit(&text, owner, e);
}

/** @brief Takes a trace directory whose path is made: checks the directory, makes the paths in it,
 *         locks it and writes its owner.json.
 *
 * @param r          The trace directory, with only its directory and length set; trace_requests_fini()
 *                   frees what it holds on a failure.
 * @param shm        The channel file's path.
 * @param shm_length The length of its path.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
take_directory (struct trace_requests *r,
                char const            *shm,
                size_t                 shm_length,
                struct error          *e)
{
	enum error_code const code = files_private_directory(r->directory, r->length, "trace", e);
	if (code)
		return code;

	// The request names are made once, for every look at the directory.
	char claimed[sizeof ".request-" + 3 * sizeof (pid_t)];
	int const claimed_length = snprintf(claimed, sizeof claimed, ".request-%d", getpid());
	if (claimed_length < 0 || claimed_length >= (int)sizeof claimed)
		return error_fail(e, "cannot name a claimed trace request");
	size_t owner_length;
	char *owner = files_join(r->directory, r->length, "owner.json", sizeof "owner.json" - 1, &owner_length);
	char *lock = files_join(r->directory, r->length, ".worker-lock", sizeof ".worker-lock" - 1, nullptr);
	r->request = files_join(r->directory, r->length, "request", sizeof "request" - 1, nullptr);
	r->claimed = files_join(r->directory, r->length, claimed, (size_t)claimed_length, nullptr);
	if (!owner || !lock || !r->request || !r->claimed) {
		free(owner);
		owner = nullptr;
		free(lock);
		lock = nullptr;
		return error_fail(e, "out of memory");
	}

	r->lock = open(lock, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
	free(lock);
	lock = nullptr;
	if (r->lock < 0 || flock(r->lock, LOCK_EX | LOCK_NB)) {
		free(owner);
		owner = nullptr;
		return error_fail(e, "trace directory is unavailable or owned by another worker");
	}
	// Only the directory's owner writes owner.json, and removes it.
	if (write_owner(owner, owner_length, shm, shm_length, e)) {
		free(owner);
		owner = nullptr;
		return ERROR_FAILED;
	}
	r->owner = owner;
	return ERROR_NONE;
}

enum error_code
trace_requests_init (struct trace_requests *dest,
                     char const            *path,
                     size_t                 length,
                     char const            *shm,
                     size_t                 shm_length,
                     struct error          *e)
{
	*dest = (struct trace_requests){.lock = -1};
	struct trace_requests r = {.lock = -1};
	r.directory = files_absolute(path, length, &r.length);
	if (!r.directory)
		return error_fail(e, "out of memory");
	enum error_code const code = take_directory(&r, shm, shm_length, e);
	if (code) {
		trace_requests_fini(&r);
		return code;
	}
	*dest = r;
	return ERROR_NONE;
}

void
trace_requests_fini (struct trace_requests *r)
{
	if (!r)
		return;
	if (r->owner) {
		if (unlink(r->owner) && errno != ENOENT) {
			int const err = errno;
			char buf[64];
			fprintf(stderr, "cannot remove %s: %s\n", r->owner, strerror_r(err, buf, sizeof buf));
		}
		free(r->owner);
		r->owner = nullptr;
	}
	if (r->lock >= 0) {
		// Closing the lock file unlocks the directory; nothing was written to it.
		close(r->lock);
		r->lock = -1;
	}
	free(r->claimed);
	r->claimed = nullptr;
	free(r->request);
	r->request = nullptr;
	free(r->directory);
	r->directory = nullptr;
	*r = (struct trace_requests){.lock = -1};
}

enum error_code
trace_requests_take (struct trace_requests const *r,
                     struct frame_trace          *dest,
                     struct error                *e)
{
	dest->path = nullptr;
	if (rename(r->request, r->claimed)) {
		int const err = errno;
		if (err == ENOENT)
			return ERROR_NONE;
		return fail_errno(e, "claim trace request", err);
	}

	int fd = open(r->claimed, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
	// This also clears a stray empty directory at DIR/request, which would block every claim.
	if (remove(r->claimed) && errno != ENOENT) {
		int const err = errno;
		if (fd >= 0) {
			close(fd);
			fd = -1;
		}
		return fail_errno(e, "remove claimed trace request", err);
	}
	if (fd < 0)
		return error_fail(e, "open diagnostic request");

	// A token, a newline after it, and a byte more to tell a request that is too long.
	char token[66];
	struct stat st;
	ssize_t const size = fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid()
	                     ? -1 : read(fd, token, sizeof token);
	// Only read: close() has nothing to report.
	close(fd);
	fd = -1;
	if (size < 1 || size >= (ssize_t)sizeof token)
		return error_fail(e, "invalid diagnostic request length");
	size_t length = (size_t)size;
	if (token[length - 1] == '\n')
		--length;
	if (!trace_valid_token(token, length))
		return error_fail(e, "invalid diagnostic request token");
	return frame_trace_init(dest, r->directory, r->length, token, length, e);
}
