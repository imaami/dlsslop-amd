/** @file
 *
 * Host test of dlsslopd's diagnostic traces (trace.h): the fitted RGB PFM, safe request tokens, a
 * private trace root that one daemon owns, a request claimed once, a summary written before its
 * marker, failed and abandoned traces, and summaries that Python's json module parses strictly.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "error.h"
#include "files.h"
#include "support.h"
#include "trace.h"

/** @brief The test's directory, which exit() removes. */
static char *scratch;

/** @brief The length of its path. */
static size_t scratch_length;

/** @brief A path in the test's directory, at() returns it. */
static char *scratch_path;

/** @brief The longest name in the test's directory that at() takes, and its null. */
#define NAME_BYTES 64

/** @brief A string literal and its length. */
#define LITERAL(text) "" text, sizeof "" text - 1

/** @brief Removes the test's directory: atexit()'s handler. */
static void
remove_scratch (void)
{
	if (scratch && !support_remove_tree(scratch))
		fprintf(stderr, "trace test: cannot remove %s\n", scratch);
	free(scratch);
	scratch = nullptr;
	free(scratch_path);
	scratch_path = nullptr;
}

/** @brief Ends the test unless a condition holds.
 *
 * @param ok      The condition.
 * @param message What failed.
 */
static void
require (bool        ok,
         char const *message)
{
	if (ok)
		return;
	fprintf(stderr, "trace test: %s\n", message);
	exit(1);
}

/** @brief A path in the test's directory.
 *
 * @param name        The path relative to the directory.
 * @param length      Its length, less than NAME_BYTES.
 * @param path_length Receives the path's length, or nullptr.
 * @return            The path, which the next call replaces.
 */
static char const *
at (char const *name,
    size_t      length,
    size_t     *path_length)
{
	require(length < NAME_BYTES, "a name in the test's directory is too long");
	scratch_path[scratch_length] = '/';
	memcpy(scratch_path + scratch_length + 1, name, length + 1);
	if (path_length)
		*path_length = scratch_length + 1 + length;
	return scratch_path;
}

/** @brief A path in the test's directory, named by a string literal. */
#define AT(name) at(LITERAL(name), nullptr)

/** @brief Whether a file exists; an error reads as no.
 *
 * @param path The file.
 * @return     true if it does.
 */
static bool
present (char const *path)
{
	struct stat st;
	return !stat(path, &st);
}

/** @brief Whether a file's permission bits are exactly 0700.
 *
 * @param path The file.
 * @return     true if they are; false also if it cannot be found.
 */
static bool
is_private (char const *path)
{
	struct stat st;
	return !stat(path, &st) && (st.st_mode & 0777) == 0700;
}

/** @brief Whether a file holds exactly a text.
 *
 * @param path The file.
 * @param text The text.
 * @return     true if it does; false also if it cannot be read.
 */
static bool
holds (char const *path,
       char const *text)
{
	char *held = support_read_file(path, nullptr);
	bool const same = held && !strcmp(held, text);
	free(held);
	held = nullptr;
	return same;
}

/** @brief A file's text, which must be readable.
 *
 * @param path The file.
 * @return     The text, which the caller frees.
 */
static char *
text_of (char const *path)
{
	char *const text = support_read_file(path, nullptr);
	require(text, "read a written file");
	return text;
}

/** @brief Writes a request file, DIR/request.
 *
 * @param text   The request.
 * @param length Its length.
 */
static void
write_request (char const *text,
               size_t      length)
{
	require(!files_write(AT("request"), text, length, nullptr), "write a text file");
}

/** @brief The next claimed trace; failing to claim one fails the test.
 *
 * @param requests The trace directory.
 * @return         The trace, which may be none.
 */
static struct frame_trace
taken (struct trace_requests const *requests)
{
	struct frame_trace trace;
	require(!trace_requests_take(requests, &trace, nullptr), "claim a trace request");
	return trace;
}

/** @brief Whether claiming the next request fails.
 *
 * @param requests The trace directory.
 * @return         true if it does.
 */
static bool
claim_fails (struct trace_requests const *requests)
{
	struct frame_trace trace;
	if (trace_requests_take(requests, &trace, nullptr))
		return true;
	frame_trace_fini(&trace);
	return false;
}

/** @brief Whether a trace root is rejected and left without a lock or an owner.
 *
 * @param root   The root.
 * @param length Its length.
 * @return       true if it is.
 */
static bool
rejected (char const *root,
          size_t      length)
{
	struct trace_requests requests;
	if (!trace_requests_init(&requests, root, length, LITERAL("/tmp/example-shm"), nullptr)) {
		trace_requests_fini(&requests);
		return false;
	}
	char *lock = files_join(root, length, LITERAL(".worker-lock"), nullptr);
	char *owner = files_join(root, length, LITERAL("owner.json"), nullptr);
	require(lock && owner, "out of memory");
	bool const clean = !present(lock) && !present(owner);
	free(owner);
	owner = nullptr;
	free(lock);
	lock = nullptr;
	return clean;
}

/** @brief Whether Python's json module parses files strictly: nan, inf and a missing separator must
 *         fail here, not in a diagnostic run.
 *
 * @param python The Python interpreter.
 * @param files  The files, ending in nullptr.
 * @return       true if every file parses.
 */
static bool
strict_json (char const *python,
             char       *files[])
{
	static char const script[] = "import json, sys\n"
	                             "def constant(name): raise ValueError(name)\n"
	                             "for name in sys.argv[1:]: json.load(open(name), parse_constant=constant)";
	size_t count = 0;
	while (files[count])
		++count;
	char const **argv = calloc(count + 4, sizeof *argv);
	require(argv, "out of memory");
	argv[0] = python;
	argv[1] = "-c";
	argv[2] = script;
	memcpy(argv + 3, files, count * sizeof *files);

	pid_t child;
	int const err = posix_spawnp(&child, python, nullptr, nullptr, (char *const *)argv, environ);
	free(argv);
	argv = nullptr;
	require(!err, "start Python");
	int status;
	while (waitpid(child, &status, 0) < 0)
		require(errno == EINTR, "wait for Python");
	return WIFEXITED(status) && !WEXITSTATUS(status);
}

/** @brief Writes a PFM and reads it back: rows bottom-up, channels RGB, padding and alpha excluded,
 *         extended values retained.
 *
 * @param g    The geometry.
 * @param rgba Its RGBA pixels.
 */
static void
check_pfm (struct geometry const *g,
           float const           *rgba)
{
	char const *const file = AT("test.pfm");
	require(!trace_write_pfm(file, rgba, g, 4, nullptr), "write a PFM");
	FILE *const pfm = fopen(file, "rbe");
	require(pfm, "open the PFM");
	char *line = nullptr;
	size_t size = 0;
	require(getline(&line, &size, pfm) > 0 && !strcmp(line, "PF\n"), "RGB PFM magic");
	require(getline(&line, &size, pfm) > 0 && !strcmp(line, "2 2\n"), "PFM fitted viewport dimensions");
	require(getline(&line, &size, pfm) > 0
	        && !strcmp(line, __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ ? "-1.0\n" : "1.0\n"), "PFM byte order");
	free(line);
	line = nullptr;
	float actual[12];
	bool const read = fread(actual, sizeof actual, 1, pfm) == 1;
	require(!fclose(pfm) && read, "read the PFM");
	static float const expected[12] = {2, 1, -.5f, 2, 2, -.5f, 1, 1, -.5f, 1, 2, -.5f};
	for (size_t i = 0; i < sizeof expected / sizeof *expected; ++i)
		require(actual[i] == expected[i], "PFM rows bottom-up, channels RGB, padding and alpha excluded, "
		                                  "extended values retained");
}

/** @brief Checks request tokens: bounded, safe in a path, and not the request file's name. */
static void
check_tokens (void)
{
	require(trace_valid_token(LITERAL("stage_01-A")) && trace_valid_token(LITERAL("requests")),
	        "safe bounded request tokens");
	require(!trace_valid_token(LITERAL("../escape")) && !trace_valid_token(LITERAL(""))
	        && !trace_valid_token(LITERAL("request")), "safe bounded request tokens");
	char a[65];
	memset(a, 'a', sizeof a);
	require(trace_valid_token(a, 64) && !trace_valid_token(a, 65), "safe bounded request tokens");
}

/** @brief Checks that only a private trace root is accepted, so that other users cannot plant files
 *         (such as a symlinked owner.json.tmp) in it, and that a created one is private.
 */
static void
check_roots (void)
{
	static mode_t const shared[] = {0777, 0755};
	for (size_t i = 0; i < sizeof shared / sizeof *shared; ++i) {
		require(!chmod(scratch, shared[i]) && rejected(scratch, scratch_length),
		        "reject a trace root other users can read or write");
	}
	require(!chmod(scratch, 0700), "restore the trace root's permissions");
	size_t link_length;
	char const *const link = at(LITERAL("link"), &link_length);
	require(!symlink(scratch, link) && rejected(link, link_length), "reject a symlinked trace root");
	require(!unlink(link), "remove the symlinked root");

	struct trace_requests created;
	size_t nested_length;
	char const *const nested = at(LITERAL("new/nested"), &nested_length);
	require(!trace_requests_init(&created, nested, nested_length, LITERAL("/tmp/example-shm"), nullptr)
	        && is_private(nested), "a created trace root is private");
	trace_requests_fini(&created);
	require(support_remove_tree(AT("new")), "remove the created trace root");
}

/** @brief Checks a complete trace: a request claimed once, a private token directory, the summary
 *         and its metadata, and the marker written last.
 *
 * @param requests The trace directory.
 * @param g        The geometry.
 * @param rgba     Its RGBA pixels.
 */
static void
check_complete (struct trace_requests const *requests,
                struct geometry const       *g,
                float const                 *rgba)
{
	require(!taken(requests).path, "no request means no trace");
	write_request(LITERAL("frame_1\n"));
	struct frame_trace frame = taken(requests);
	require(frame.path && !taken(requests).path, "request consumed exactly once");
	require(is_private(AT("frame_1")), "a token directory is private");
	frame_trace_image(&frame, LITERAL("pass-01-input"), rgba, g, 4);
	frame_trace_image(&frame, LITERAL("pass-01-raw"), rgba, g, 4);
	require(!present(AT("frame_1/summary.json")), "summary must complete last");
	require(!present(AT("frame_1.done")), "no early root completion marker");

	// Exercise the serializer used by the worker, not a hand-written summary that could silently
	// omit fields required by the client.
	struct trace_metadata const metadata = {
		.source_proxy_hash = 0x1234abcd,
		.geometry          = *g,
		.frame_seq         = 123,
		.control_seq       = 11,
		.tuning_seq        = 7,
		.control_seq_end   = 12,
		.tuning_seq_end    = 8,
		.held_input        = 1,
		.held_input_end    = 0,
		.intensity         = 0.5f,
		.local_tone        = 1,
		.local_structure   = 1,
		.passes            = 3,
		.fp16_proxy        = true,
		.fp16_feedback     = true,
	};
	char json[TRACE_METADATA_BYTES];
	require(!trace_metadata_json(&metadata, json, nullptr), "format the metadata");
	frame_trace_finish(&frame, json, nullptr);
	require(holds(AT("frame_1.done"), "frame_1/summary.json\n"), "root marker follows summary commit");

	char *summary = text_of(AT("frame_1/summary.json"));
	require(strstr(summary, "\"status\":\"complete\"") && strstr(summary, "\"frame_seq\":123"),
	        "summary completion and metadata");
	require(strstr(summary, "\"schema\":1"), "metadata changes preserve the outer summary file format");
	static char const *const fields[] = {
		"\"trace_metadata_schema\":2", "\"control_seq\":11", "\"tuning_seq\":7",
		"\"control_seq_end\":12", "\"tuning_seq_end\":8",
		"\"held_input\":1", "\"held_input_end\":0",
		"\"source_proxy_hash\":\"000000001234abcd\"", "\"passes\":3",
		"\"source_width\":2", "\"source_height\":2",
		"\"network_width\":4", "\"network_height\":4",
		"\"fit_x\":1", "\"fit_y\":1", "\"fit_width\":2", "\"fit_height\":2",
		"\"fp16_proxy\":1", "\"fp16_feedback\":1", "\"motion\":0",
		"\"intensity\":0.5", "\"local_tone\":1", "\"local_structure\":1",
		"\"sharpness\":0",
		"\"file\":\"pass-01-input.pfm\"", "\"file\":\"pass-01-raw.pfm\"",
	};
	for (size_t i = 0; i < sizeof fields / sizeof *fields; ++i)
		require(strstr(summary, fields[i]), fields[i]);

	write_request(LITERAL("frame_1\n"));
	require(claim_fails(requests) && holds(AT("frame_1/summary.json"), summary),
	        "duplicate token must never overwrite evidence");
	write_request(LITERAL("../escape\n"));
	require(claim_fails(requests), "reject path traversal");
	// The protocol file's own name would make the evidence directory DIR/request, which the next
	// claim would take as a request.
	write_request(LITERAL("request\n"));
	require(claim_fails(requests) && !present(AT("request")), "reject the reserved token");
	require(!mkdir(AT("request"), 0700), "create a stray request directory");
	char claimed[NAME_BYTES];
	int const length = snprintf(claimed, sizeof claimed, ".request-%d", getpid());
	require(length > 0 && length < (int)sizeof claimed, "name the claimed request");
	require(claim_fails(requests) && !present(AT("request")) && !present(at(claimed, (size_t)length, nullptr)),
	        "a stray empty request directory is cleared, not left to block claims");
	free(summary);
	summary = nullptr;
}

/** @brief Checks failed traces: a failed frame publishes its summary and marker with its error
 *         escaped and no stage after a failed one, an abandoned trace publishes a failed summary,
 *         and an unfinished failure keeps its error.
 *
 * @param requests The trace directory.
 * @param g        The geometry.
 * @param rgba     Its RGBA pixels.
 */
static void
check_failed (struct trace_requests const *requests,
              struct geometry const       *g,
              float const                 *rgba)
{
	write_request(LITERAL("frame_2\n"));
	struct frame_trace failed = taken(requests);
	require(failed.path, "later requests are claimed");
	frame_trace_image(&failed, LITERAL("pass-01-input"), rgba, g, 4);
	frame_trace_image(&failed, LITERAL("pass-01-tuned"), rgba, g, 2);
	frame_trace_image(&failed, LITERAL("pass-01-raw"), rgba, g, 4);
	frame_trace_finish(&failed, "{\"frame_seq\":124}", "bad \"quote\"\n\x01\\");
	require(holds(AT("frame_2.done"), "frame_2/summary.json\n"), "failed trace marker");
	char *failure = text_of(AT("frame_2/summary.json"));
	static char const *const failure_fields[] = {
		"\"status\":\"failed\"", "\"metadata\":{\"frame_seq\":124}",
		"\"error\":\"bad \\\"quote\\\"\\u000a\\u0001\\\\\"", "\"file\":\"pass-01-input.pfm\"",
	};
	for (size_t i = 0; i < sizeof failure_fields / sizeof *failure_fields; ++i)
		require(strstr(failure, failure_fields[i]), failure_fields[i]);
	require(!strstr(failure, "pass-01-raw") && !strstr(failure, "pass-01-tuned")
	        && !present(AT("frame_2/pass-01-raw.pfm")) && !present(AT("frame_2/pass-01-tuned.pfm")),
	        "no stage from or after a failed stage");
	free(failure);
	failure = nullptr;

	// A trace dropped unfinished, as when the worker stops after the claim, still publishes a
	// failed summary and its root marker.
	write_request(LITERAL("frame_3\n"));
	struct frame_trace abandoned = taken(requests);
	frame_trace_image(&abandoned, LITERAL("pass-01-input"), rgba, g, 4);
	frame_trace_fini(&abandoned);
	require(holds(AT("frame_3.done"), "frame_3/summary.json\n"), "abandoned trace marker");
	char *dropped = text_of(AT("frame_3/summary.json"));
	static char const *const dropped_fields[] = {
		"\"status\":\"failed\"", "\"metadata\":{}",
		"\"error\":\"worker stopped before a traced frame completed\"", "\"file\":\"pass-01-input.pfm\"",
	};
	for (size_t i = 0; i < sizeof dropped_fields / sizeof *dropped_fields; ++i)
		require(strstr(dropped, dropped_fields[i]), dropped_fields[i]);
	free(dropped);
	dropped = nullptr;

	// A stage failure recorded but not finished keeps its own error.
	write_request(LITERAL("frame_4\n"));
	struct frame_trace unfinished = taken(requests);
	frame_trace_image(&unfinished, LITERAL("pass-01-input"), rgba, g, 2);
	frame_trace_fini(&unfinished);
	char *kept = text_of(AT("frame_4/summary.json"));
	require(strstr(kept, "\"status\":\"failed\"")
	        && strstr(kept, "\"error\":\"invalid diagnostic image geometry\""), "an unfinished failure keeps its error");
	free(kept);
	kept = nullptr;

	// A finished trace is published once, not again when it is dropped.
	require(!unlink(AT("frame_2.done")), "remove a completion marker");
	frame_trace_fini(&failed);
	require(!present(AT("frame_2.done")), "finish publishes once");
}

int
main (int    argc,
      char **argv)
{
	if (argc != 2) {
		fputs("usage: trace-test PYTHON\n", stderr);
		return 2;
	}
	scratch = support_temp_dir("/tmp", "dlsslop-amd-trace-test", &scratch_length);
	if (!scratch)
		return 1;
	scratch_path = malloc(scratch_length + 1 + NAME_BYTES);
	if (!scratch_path || atexit(remove_scratch)) {
		remove_scratch();
		return 1;
	}
	memcpy(scratch_path, scratch, scratch_length);

	struct geometry const g = {2, 2, 4, 4, 4, 1, 1, 2, 2};
	float rgba[4 * 4 * 4];
	for (size_t i = 0; i < sizeof rgba / sizeof *rgba; ++i)
		rgba[i] = 99.0f;
	for (unsigned y = 1; y <= 2; ++y) {
		for (unsigned x = 1; x <= 2; ++x) {
			unsigned const p = (y * 4 + x) * 4;
			rgba[p] = (float)y;
			rgba[p + 1] = (float)x;
			rgba[p + 2] = -0.5f;
		}
	}
	check_pfm(&g, rgba);
	check_tokens();
	check_roots();

	struct trace_requests requests;
	require(!trace_requests_init(&requests, scratch, scratch_length, LITERAL("/tmp/example-shm"), nullptr),
	        "open a private trace root");
	char *owner = text_of(AT("owner.json"));
	char *pid = support_format(nullptr, "\"pid\":%d,", getpid());
	require(pid, "out of memory");
	require(strstr(owner, pid) && strstr(owner, "\"shm\":\"/tmp/example-shm\""), "owner discovery metadata");
	require(strstr(owner, "\"trace_metadata_schema\":2"), "owner must advertise frozen-input evidence before capture");
	free(pid);
	pid = nullptr;
	free(owner);
	owner = nullptr;
	struct trace_requests other;
	require(trace_requests_init(&other, scratch, scratch_length, LITERAL("/tmp/other"), nullptr),
	        "one worker per trace directory");

	check_complete(&requests, &g, rgba);
	check_failed(&requests, &g, rgba);

	static struct {
		char const *text;
		size_t      length;
	} const names[] = {
		{LITERAL("owner.json")}, {LITERAL("frame_1/summary.json")}, {LITERAL("frame_2/summary.json")},
		{LITERAL("frame_3/summary.json")}, {LITERAL("frame_4/summary.json")},
	};
	size_t const count = sizeof names / sizeof *names;
	char *json[sizeof names / sizeof *names + 1];
	for (size_t i = 0; i < count; ++i) {
		json[i] = files_join(scratch, scratch_length, names[i].text, names[i].length, nullptr);
		require(json[i], "out of memory");
	}
	json[count] = nullptr;
	require(strict_json(argv[1], json), "summaries are strict JSON");
	for (size_t i = 0; i < count; ++i) {
		free(json[i]);
		json[i] = nullptr;
	}

	trace_requests_fini(&requests);
	require(!present(AT("owner.json")), "remove stale owner on shutdown");
	puts("trace: fitted RGB PFM, safe requests, ownership, failure, completion and JSON passed");
	return 0;
}

#undef NAME_BYTES

#undef AT
#undef LITERAL
