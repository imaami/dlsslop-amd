/** @file
 *
 * The channel's initialization, text fields and frame counts, in C. Initialization gives every word
 * of a header a value, whatever its bytes held before. A store to a text field bumps that field's
 * sequence number alone, and a load takes the string: cut to the field's size or to the reader's
 * buffer, or an empty one for a null string. A frame count is read whole: while a child process adds
 * to one count and stores the other in a shared mapping, this process never reads either torn.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "shm_protocol.h"

/** @brief A text field of the header and the words that hold it. */
struct text_field {
	_Atomic(uint32_t) const *seq;   //!< Its sequence number.
	char const              *text;  //!< Its bytes.
	size_t                   size;  //!< Its size.
	enum shm_text            field; //!< The field.
};

/** @brief What the writer adds to a count: one to each half, so a count written whole has equal
 *         halves, and one read in halves of different writes does not.
 */
static constexpr uint64_t COUNT_STEP = UINT64_C(0x100000001);

/** @brief The fewest steps the writer takes, and the fewest times the reader reads each count,
 *         before the reader stops the writer.
 */
static constexpr uint64_t COUNT_STEPS = UINT64_C(1) << 20;

/** @brief The byte that fills a header before it is initialized; no word of an initialized header is
 *         four of it.
 */
static constexpr unsigned char PATTERN = 0xa5;

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param message   What failed.
 */
static void
require (bool        condition,
         char const *message)
{
	if (condition)
		return;
	fprintf(stderr, "shm-protocol-test: %s\n", message);
	exit(1);
}

/** @brief Initializes a header filled with PATTERN, as storage is filled with something, and ends
 *         the test if a word of it is left as it was.
 */
static void
check_init (void)
{
	uint32_t pattern;
	memset(&pattern, PATTERN, sizeof pattern);
	struct ShmHeader header;
	// No atomic object of the header is valid after this, until the header is initialized.
	memset(&header, PATTERN, sizeof header);
	ShmInitNativeDefaults(&header, false);
	unsigned char const *const bytes = (unsigned char const *)&header;
	for (size_t offset = 0; offset < sizeof header; offset += sizeof pattern) {
		uint32_t word;
		memcpy(&word, bytes + offset, sizeof word);
		if (word == pattern) {
			fprintf(stderr, "shm-protocol-test: ShmInitNativeDefaults() left the word at offset %zu as "
			        "it was\n", offset);
			exit(1);
		}
	}
}

/** @brief Steps the frame counts until the reader stores quit: adds COUNT_STEP to helperFrames, as
 *         a counter does, and stores the same multiple of COUNT_STEP in layerFrames, as the layer
 *         and dlsslopd do. The kernel kills the writer when the reader ends, also by a failed
 *         require(), which stores no quit.
 *
 * @param h      The header, in a mapping shared with the reader.
 * @param reader The reader's process.
 */
[[noreturn]] static void
write_counts (struct ShmHeader *h,
              pid_t             reader)
{
	if (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != reader)
		_exit(1);
	for (uint64_t v = COUNT_STEP; !atomic_load(&h->quit); v += COUNT_STEP) {
		atomic_fetch_add(&h->helperFrames, COUNT_STEP);
		atomic_store(&h->layerFrames, v);
	}
	_exit(0);
}

/** @brief Reads a count that write_counts() writes, and ends the test if it is torn or went back.
 *
 * @param count The count.
 * @param last  The value read last; receives this one.
 * @return      Whether the value changed.
 */
static bool
read_count (_Atomic(uint64_t) const *count,
            uint64_t                *last)
{
	uint64_t const v = atomic_load(count);
	require((uint32_t)(v >> 32) == (uint32_t)v, "a frame count was read torn");
	require(v >= *last, "a frame count went back");
	bool const changed = v != *last;
	*last = v;
	return changed;
}

/** @brief Waits for a child, or checks on it.
 *
 * @param child   The child.
 * @param status  Receives its status, as waitpid() reports it.
 * @param options WNOHANG to only check on it, or 0.
 * @return        The child, or 0 if WNOHANG is set and the child is still running.
 */
static pid_t
reap (pid_t  child,
      int   *status,
      int    options)
{
	pid_t done;
	while ((done = waitpid(child, status, options)) < 0)
		require(errno == EINTR, "waitpid failed");
	return done;
}

/** @brief Reads the frame counts while a child process writes them, and ends the test if any value
 *         read is torn.
 */
static void
check_counts (void)
{
	struct ShmHeader *const h = mmap(nullptr, sizeof *h, PROT_READ | PROT_WRITE,
	                                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	require(h != MAP_FAILED, "mmap failed");
	ShmInitNativeDefaults(h, false);
	require(!fflush(nullptr), "fflush failed");
	pid_t const reader = getpid();
	pid_t const child = fork();
	require(child >= 0, "fork failed");
	if (!child)
		write_counts(h, reader);

	int status = 0;
	pid_t done = 0;
	uint64_t helper = 0, layer = 0, reads = 0, changes = 0;
	while (!done && (reads < COUNT_STEPS || helper < COUNT_STEPS * COUNT_STEP)) {
		changes += read_count(&h->helperFrames, &helper);
		changes += read_count(&h->layerFrames, &layer);
		if (++reads % 65536 == 0)
			done = reap(child, &status, WNOHANG);
	}
	atomic_store(&h->quit, 1);
	if (!done)
		reap(child, &status, 0);
	require(WIFEXITED(status) && !WEXITSTATUS(status), "the writer failed");
	require(!munmap(h, sizeof *h), "munmap failed");
	require(printf("shm protocol: %" PRIu64 " reads of each frame count while another process wrote "
	               "them, %" PRIu64 " changes seen, none torn\n", reads, changes) > 0,
	        "printf failed");
}

/** @brief The sum of the text fields' sequence numbers.
 *
 * @param h The header.
 * @return  The sum.
 */
static uint32_t
sequence_sum (struct ShmHeader const *h)
{
	return atomic_load(&h->helperReasonSeq) + atomic_load(&h->layerReasonSeq)
	       + atomic_load(&h->gameNameSeq);
}

int
main (void)
{
	static struct ShmHeader header;
	struct text_field const fields[] = {
		{&header.helperReasonSeq, header.helperReason, sizeof header.helperReason, SHM_TEXT_HELPER_REASON},
		{&header.layerReasonSeq,  header.layerReason,  sizeof header.layerReason,  SHM_TEXT_LAYER_REASON},
		{&header.gameNameSeq,     header.gameName,     sizeof header.gameName,     SHM_TEXT_GAME_NAME},
	};
	char too_long[kReasonBytes + 1];
	memset(too_long, 'x', kReasonBytes);
	too_long[kReasonBytes] = '\0';

	for (uint32_t i = 0; i < sizeof fields / sizeof *fields; ++i) {
		struct text_field const *const f = &fields[i];
		char text[kReasonBytes];

		ShmStoreString(&header, f->field, "first");
		require(atomic_load(f->seq) == 1 && sequence_sum(&header) == 3 * i + 1
		        && ShmLoadString(&header, f->field, text, sizeof text) && !strcmp(text, "first")
		        && !strcmp(f->text, "first"),
		        "a text field was not published in its own words");

		ShmStoreString(&header, f->field, too_long);
		require(atomic_load(f->seq) == 2 && sequence_sum(&header) == 3 * i + 2
		        && ShmLoadString(&header, f->field, text, sizeof text)
		        && strlen(text) == f->size - 1 && !memcmp(text, too_long, f->size - 1),
		        "a text field did not cut a string to its size");

		char small[8];
		require(ShmLoadString(&header, f->field, small, sizeof small)
		        && !small[sizeof small - 1] && !memcmp(small, too_long, sizeof small - 1),
		        "a load did not cut a string to the reader's buffer");

		ShmStoreString(&header, f->field, nullptr);
		require(atomic_load(f->seq) == 3 && sequence_sum(&header) == 3 * i + 3
		        && ShmLoadString(&header, f->field, text, sizeof text) && !*text && !f->text[0],
		        "a null string did not empty a text field");
	}

	check_init();
	check_counts();
	if (puts("shm protocol: initialization, text fields and frame counts in C") == EOF)
		return 1;
	return 0;
}
