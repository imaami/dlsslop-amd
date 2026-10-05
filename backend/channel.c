/** @file
 *
 * The daemon's side of the shared-memory channel: channel.h.
 */
// SPDX-License-Identifier: MIT
#include <linux/futex.h>
#include <stdatomic.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "channel.h"
#include "shm_channel.h"

enum error_code
mapping_init (struct mapping *dest,
              char const     *name,
              size_t          length,
              struct error   *e)
{
	*dest = (struct mapping){ .channel = { .fd = -1 } };
	enum error_code const code = shm_channel_open(&dest->channel, name, length, ShmTotalBytes(),
	                                              SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, e);
	if (code)
		return code;
	// The lock lasts as long as the descriptor, and tells other daemons, and the sweeps of other
	// protocols' channels (shm_channel_open()), that this one serves the channel.
	if (flock(dest->channel.fd, LOCK_EX | LOCK_NB)) {
		shm_channel_fini(&dest->channel);
		return error_fail(e, "another worker owns this shared-memory file");
	}

	struct ShmHeader *const h = dest->channel.h;
	atomic_store(&h->quit, 0);
	atomic_store(&h->modelUp, 0);
	atomic_store(&h->seq_ok, 0);
	// Answer a request left by a previous worker as failed, so the layer presents its own frame;
	// requests made from here on are served. No layer need be waiting to be woken.
	atomic_store_explicit(&h->seq_resp, atomic_load_explicit(&h->seq_req, memory_order_acquire),
	                      memory_order_release);
	syscall(SYS_futex, &h->seq_resp, FUTEX_WAKE, 1, nullptr, nullptr, 0);
	atomic_store(&h->helperState, kHelperStarting);

	uint8_t *const input = (uint8_t *)h + kHeaderBytes;
	dest->input = input;
	dest->output = input + kMaxFrame;
	return ERROR_NONE;
}

void
mapping_fini (struct mapping *mapping)
{
	if (!mapping)
		return;
	struct ShmHeader *const h = mapping->channel.h;
	if (h) {
		atomic_store(&h->modelUp, 0);
		if (atomic_load(&h->helperState) != kHelperModelFailed)
			atomic_store(&h->helperState, kHelperStopped);
	}
	// Closing the file unlocks it for the next daemon.
	shm_channel_fini(&mapping->channel);
	*mapping = (struct mapping){ .channel = { .fd = -1 } };
}

void
mapping_reason (struct mapping const *mapping,
                char const           *text)
{
	ShmStoreString(mapping->channel.h, SHM_TEXT_HELPER_REASON, text);
}
