/** @file
 *
 * The daemon's side of the shared-memory channel: channel.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "channel.h"
#include "files.h"

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

/** @brief Takes the channel file and maps it.
 *
 * @param fd      The channel file, open for reading and writing.
 * @param mapping Receives the mapping; untouched on a failure.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
map (int            fd,
     void         **mapping,
     struct error  *e)
{
	if (flock(fd, LOCK_EX | LOCK_NB))
		return error_fail(e, "another worker owns this shared-memory file");
	struct stat st;
	if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid())
		return error_fail(e, "shared-memory file must be regular and owned by the current user");
	if (fchmod(fd, 0600))
		return fail_errno(e, "make shared-memory file private", errno);
	size_t const bytes = ShmTotalBytes();
	if (ftruncate(fd, (off_t)bytes))
		return fail_errno(e, "size shared-memory file", errno);
	void *const mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (mapped == MAP_FAILED)
		return fail_errno(e, "map shared-memory file", errno);
	*mapping = mapped;
	return ERROR_NONE;
}

enum error_code
mapping_init (struct mapping *dest,
              char const     *name,
              size_t          length,
              struct error   *e)
{
	*dest = (struct mapping){.fd = -1};
	size_t const parent = files_parent(name, length);
	if (!parent)
		return error_fail(e, "--shm requires a path inside a private directory");
	enum error_code code = files_private_directory(name, parent, "shared-memory", e);
	if (code)
		return code;

	int fd = open(name, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
	if (fd < 0)
		return fail_errno(e, "open shared-memory file", errno);
	void *mapping;
	code = map(fd, &mapping, e);
	if (code) {
		// Nothing was written through it: close() has nothing to report.
		close(fd);
		fd = -1;
		return code;
	}

	struct ShmHeader *const h = mapping;
	if (atomic_load(&h->magic) != kShmMagic || atomic_load(&h->version) != kShmVersion)
		ShmInitNativeDefaults(h, false);
	atomic_store(&h->quit, 0);
	atomic_store(&h->modelUp, 0);
	atomic_store(&h->seq_ok, 0);
	// Answer a request left by a previous worker as failed, so the layer presents its own frame;
	// requests made from here on are served. No layer need be waiting to be woken.
	atomic_store_explicit(&h->seq_resp, atomic_load_explicit(&h->seq_req, memory_order_acquire),
	                      memory_order_release);
	syscall(SYS_futex, &h->seq_resp, FUTEX_WAKE, 1, nullptr, nullptr, 0);
	atomic_store(&h->helperState, kHelperStarting);

	uint8_t *const input = (uint8_t *)mapping + kHeaderBytes;
	*dest = (struct mapping){
		.h      = h,
		.input  = input,
		.output = input + kMaxFrame,
		.fd     = fd,
	};
	return ERROR_NONE;
}

void
mapping_fini (struct mapping *mapping)
{
	if (!mapping)
		return;
	struct ShmHeader *const h = mapping->h;
	if (h) {
		atomic_store(&h->modelUp, 0);
		if (atomic_load(&h->helperState) != kHelperModelFailed)
			atomic_store(&h->helperState, kHelperStopped);
		// The daemon is leaving: an unmapping that fails changes nothing for it.
		munmap(h, ShmTotalBytes());
	}
	if (mapping->fd >= 0) {
		// Closing the file unlocks it for the next daemon; nothing was written through the descriptor.
		close(mapping->fd);
		mapping->fd = -1;
	}
	*mapping = (struct mapping){.fd = -1};
}

void
mapping_reason (struct mapping const *mapping,
                char const           *text)
{
	ShmStoreString(mapping->h, SHM_TEXT_HELPER_REASON, text);
}
