/** @file
 *
 * The channel's text fields and 64-bit counts, in C. A store to a text field bumps that field's
 * sequence number alone, and a load takes the string: cut to the field's size or to the reader's
 * buffer, or an empty one for a null string. A count is stored in its own two words and read back.
 */
// SPDX-License-Identifier: MIT
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shm_protocol.h"

/** @brief A text field of the header and the words that hold it. */
struct text_field {
	_Atomic(uint32_t) const *seq;   //!< Its sequence number.
	char const              *text;  //!< Its bytes.
	size_t                   size;  //!< Its size.
	enum shm_text            field; //!< The field.
};

/** @brief A 64-bit count of the header and the words that hold it. */
struct count_words {
	_Atomic(uint32_t) const *lo;    //!< The low word.
	_Atomic(uint32_t) const *hi;    //!< The high word.
	enum shm_count           count; //!< The count.
};

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

	struct count_words const counts[] = {
		{&header.helperFramesLo, &header.helperFramesHi, SHM_COUNT_HELPER_FRAMES},
		{&header.layerFramesLo,  &header.layerFramesHi,  SHM_COUNT_LAYER_FRAMES},
	};
	for (uint32_t i = 0; i < sizeof counts / sizeof *counts; ++i) {
		struct count_words const *const c = &counts[i];
		ShmStore64(&header, c->count, UINT64_C(0x123456789) + i);
		require(atomic_load(c->hi) == 1 && atomic_load(c->lo) == 0x23456789 + i
		        && ShmLoad64(&header, c->count) == UINT64_C(0x123456789) + i,
		        "a 64-bit count did not survive its two words");
	}
	for (uint32_t i = 0; i < sizeof counts / sizeof *counts; ++i)
		require(ShmLoad64(&header, counts[i].count) == UINT64_C(0x123456789) + i,
		        "a 64-bit count was stored in another count's words");

	if (puts("shm protocol: text fields and 64-bit counts in C") == EOF)
		return 1;
	return 0;
}
