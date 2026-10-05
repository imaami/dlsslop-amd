/** @file
 *
 * The channel's file, opened and created in one place: shm_channel.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

#include "error.h"
#include "shm_channel.h"
#include "shm_protocol.h"

/** @brief Puts what failed, where, and strerror()'s words for an errno value in an error.
 *
 * @param e      The error, or nullptr.
 * @param action What failed.
 * @param path   Where.
 * @param err    The errno value.
 * @return       ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
fail_errno (struct error *e,
            char const   *action,
            char const   *path,
            int           err)
{
	char buf[64];
	return error_fail(e, "%s %s: %s", action, path, strerror_r(err, buf, sizeof buf));
}

/** @brief Refuses a file whose magic or version is not this protocol's.
 *
 * @param path    The file.
 * @param magic   Its magic.
 * @param version Its version.
 * @param e       Receives the words, or nullptr.
 * @return        ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
refuse (char const   *path,
        uint32_t      magic,
        uint32_t      version,
        struct error *e)
{
	if (magic == kShmMagic)
		return error_fail(e, "the channel %s holds protocol v%u, and this build's is v%u: stop the "
		                  "programs that use it and remove it, or use another channel", path, version,
		                  kShmVersion);
	return error_fail(e, "%s holds no channel (magic %#x, version %u; a v%u channel's magic is %#x): "
	                  "remove it, or use another channel", path, magic, version, kShmVersion, kShmMagic);
}

/** @brief Refuses a file of another size than a channel's.
 *
 * @param fd   The file.
 * @param path Its path.
 * @param size Its size.
 * @param e    Receives the words, or nullptr.
 * @return     ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
refuse_size (int           fd,
             char const   *path,
             off_t         size,
             struct error *e)
{
	// The magic and the version, as the file's bytes hold them, to name another protocol's channel.
	uint32_t head[2];
	if (pread(fd, head, sizeof head, 0) == (ssize_t)sizeof head && head[0] == kShmMagic
	    && head[1] != kShmVersion)
		return refuse(path, head[0], head[1], e);
	return error_fail(e, "%s holds no v%u channel: it has %jd bytes, not %zu; remove it, or use "
	                  "another channel", path, kShmVersion, (intmax_t)size, ShmTotalBytes());
}

/** @brief Maps an open channel file, if it holds a channel of this protocol.
 *
 * @param fd    The file.
 * @param path  Its path.
 * @param bytes How much of it to map.
 * @param flags What shm_channel_open() was asked to do.
 * @param dest  Receives the channel, which then holds @a fd; untouched on a failure.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
map_existing (int                 fd,
              char const         *path,
              size_t              bytes,
              uint32_t            flags,
              struct shm_channel *dest,
              struct error       *e)
{
	struct stat st;
	if (fstat(fd, &st))
		return fail_errno(e, "inspect shared-memory file", path, errno);
	if (!S_ISREG(st.st_mode) || st.st_uid != getuid())
		return error_fail(e, "shared-memory file must be regular and owned by the current user: %s",
		                  path);
	if (st.st_size != (off_t)ShmTotalBytes())
		return refuse_size(fd, path, st.st_size, e);

	int const protection = PROT_READ | (flags & SHM_CHANNEL_WRITE ? PROT_WRITE : 0);
	void *const mapping = mmap(nullptr, bytes, protection, MAP_SHARED, fd, 0);
	if (mapping == MAP_FAILED)
		return fail_errno(e, "map shared-memory file", path, errno);
	// Whoever created the channel initialized these before the file had a name.
	struct ShmHeader *const h = mapping;
	uint32_t const magic = atomic_load(&h->magic);
	uint32_t const version = atomic_load(&h->version);
	if (magic != kShmMagic || version != kShmVersion) {
		// Unmapping a mapping that this process made cannot fail.
		munmap(mapping, bytes);
		return refuse(path, magic, version, e);
	}
	*dest = (struct shm_channel){ .h = h, .bytes = bytes, .fd = fd, .flags = flags };
	return ERROR_NONE;
}

/** @brief Makes a new file that no other process can open yet a channel: mode 0600, the channel's
 *         size, mapped for writing, and the native defaults.
 *
 * @param fd    The file.
 * @param path  The channel's path, for the words.
 * @param bytes How much of it to map.
 * @param h     Receives the mapping; untouched on a failure.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
prepare (int                fd,
         char const        *path,
         size_t             bytes,
         struct ShmHeader **h,
         struct error      *e)
{
	// The mode that the umask may have narrowed.
	if (fchmod(fd, 0600))
		return fail_errno(e, "make shared-memory file private", path, errno);
	if (ftruncate(fd, (off_t)ShmTotalBytes()))
		return fail_errno(e, "size shared-memory file", path, errno);
	void *const mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (mapping == MAP_FAILED)
		return fail_errno(e, "map shared-memory file", path, errno);
	// Each atomic object's one initialization, before any other process can see the file.
	ShmInitNativeDefaults(mapping, false);
	*h = mapping;
	return ERROR_NONE;
}

/** @brief Creates a channel through a uniquely named file in its directory, which linkat(2) then
 *         gives the channel's name, where no unnamed file can be made or named.
 *
 * The file is made and named through the directory's descriptor, so it lands in the directory that
 * open_directory() checked, or nowhere if that directory has been removed. Linking fails if the name
 * exists, as in create(). The temporary name, hidden and this module's alone, goes whatever happens.
 *
 * @param dir    The channel's directory.
 * @param name   The channel's name in it.
 * @param path   The channel's path, for the words.
 * @param bytes  How much of it to map.
 * @param flags  What shm_channel_open() was asked to do.
 * @param dest   Receives the channel if this process named it.
 * @param linked Receives whether it did: false if another process named its channel first.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
create_named (int                 dir,
              char const         *name,
              char const         *path,
              size_t              bytes,
              uint32_t            flags,
              struct shm_channel *dest,
              bool               *linked,
              struct error       *e)
{
	static char const prefix[] = ".shm-channel-";
	uint64_t bits;
	// Reads of up to 256 bytes are whole, or fail.
	if (getrandom(&bits, sizeof bits, 0) < 0)
		return fail_errno(e, "name a temporary file for", path, errno);
	char temporary[sizeof prefix + 16];
	snprintf(temporary, sizeof temporary, "%s%016" PRIx64, prefix, bits);
	int fd = openat(dir, temporary, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
	if (fd < 0)
		return fail_errno(e, "create shared-memory file", path, errno);

	struct ShmHeader *h;
	enum error_code const code = prepare(fd, path, bytes, &h, e);
	int const err = code ? 0 : linkat(dir, temporary, dir, name, 0) ? errno : 0;
	// The channel has its own name now, or never will through this file.
	unlinkat(dir, temporary, 0);
	if (!code && !err) {
		*dest = (struct shm_channel){
			.h = h, .bytes = bytes, .fd = fd, .flags = flags | SHM_CHANNEL_CREATED
		};
		*linked = true;
		return ERROR_NONE;
	}
	if (!code)
		munmap(h, bytes);
	// Nothing was written through the descriptor: close() has nothing to report.
	close(fd);
	fd = -1;
	if (code)
		return code;
	if (err != EEXIST)
		return fail_errno(e, "name shared-memory file", path, err);
	*linked = false;
	return ERROR_NONE;
}

/** @brief Gives an unnamed file a name.
 *
 * linkat(2) of the descriptor itself needs CAP_DAC_READ_SEARCH on older kernels; the descriptor's
 * link in /proc names the file without it.
 *
 * @param fd   The file.
 * @param dir  The directory.
 * @param name The name.
 * @return     0, or an errno value: EEXIST if the name exists.
 */
static int
link_unnamed (int         fd,
              int         dir,
              char const *name)
{
	if (!linkat(fd, "", dir, name, AT_EMPTY_PATH))
		return 0;
	if (errno == EEXIST)
		return EEXIST;
	char proc[sizeof "/proc/self/fd/-2147483648"];
	if (snprintf(proc, sizeof proc, "/proc/self/fd/%d", fd) < 0)
		return EINVAL;
	return linkat(AT_FDCWD, proc, dir, name, AT_SYMLINK_FOLLOW) ? errno : 0;
}

/** @brief Creates a channel whole and unnamed, then names it, unless another process named its
 *         channel first.
 *
 * Falls back to create_named() where the directory holds no unnamed files or no link names one.
 *
 * @param dir    The channel's directory.
 * @param name   The channel's name in it.
 * @param path   The channel's path, for the words.
 * @param bytes  How much of it to map.
 * @param flags  What shm_channel_open() was asked to do.
 * @param dest   Receives the channel if this process named it.
 * @param linked Receives whether it did: false if another process named its channel first.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
create (int                 dir,
        char const         *name,
        char const         *path,
        size_t              bytes,
        uint32_t            flags,
        struct shm_channel *dest,
        bool               *linked,
        struct error       *e)
{
	int fd = openat(dir, ".", O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
	if (fd < 0) {
		int const err = errno;
		// A file system without unnamed files, or a kernel before 3.11, which opens the directory.
		if (err == EOPNOTSUPP || err == EISDIR)
			return create_named(dir, name, path, bytes, flags, dest, linked, e);
		return fail_errno(e, "create shared-memory file", path, err);
	}

	struct ShmHeader *h;
	enum error_code code = prepare(fd, path, bytes, &h, e);
	int const err = code ? 0 : link_unnamed(fd, dir, name);
	if (!code && !err) {
		*dest = (struct shm_channel){
			.h = h, .bytes = bytes, .fd = fd, .flags = flags | SHM_CHANNEL_CREATED
		};
		*linked = true;
		return ERROR_NONE;
	}
	if (!code)
		munmap(h, bytes);
	// The file goes with its descriptor: nobody else ever saw it.
	close(fd);
	fd = -1;
	if (code)
		return code;
	if (err == EEXIST) {
		*linked = false;
		return ERROR_NONE;
	}
	// Neither link names an unnamed file on an older kernel without CAP_DAC_READ_SEARCH and /proc. A
	// directory removed meanwhile fails both the same way, and then takes no named file either.
	if (err == ENOENT)
		return create_named(dir, name, path, bytes, flags, dest, linked, e);
	return fail_errno(e, "name shared-memory file", path, err);
}

/** @brief Opens a channel in its directory, creating it if it is missing and asked to.
 *
 * @param dir    The channel's directory.
 * @param name   The channel's name in it.
 * @param path   The channel's path.
 * @param bytes  How much of it to map.
 * @param flags  What shm_channel_open() was asked to do.
 * @param dest   Receives the channel; its flags say SHM_CHANNEL_MISSING if there was none to open.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
open_in (int                 dir,
         char const         *name,
         char const         *path,
         size_t              bytes,
         uint32_t            flags,
         struct shm_channel *dest,
         struct error       *e)
{
	// O_NONBLOCK: a FIFO in the channel's place would block a reader's open.
	int const oflag = (flags & SHM_CHANNEL_WRITE ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
	// A channel that another process names between this one's open and its link is opened on the
	// next turn, and one removed meanwhile is created again.
	for (uint32_t turn = 0; turn < 4; ++turn) {
		int fd = openat(dir, name, oflag);
		if (fd >= 0) {
			enum error_code const code = map_existing(fd, path, bytes, flags, dest, e);
			if (code) {
				// Nothing was written through the descriptor: close() has nothing to report.
				close(fd);
				fd = -1;
			}
			return code;
		}
		int const err = errno;
		if (err == ELOOP)
			return error_fail(e, "shared-memory file must not be a symlink: %s", path);
		if (err != ENOENT)
			return fail_errno(e, "open shared-memory file", path, err);
		if (!(flags & SHM_CHANNEL_CREATE)) {
			dest->flags = SHM_CHANNEL_MISSING;
			return error_fail(e, "no channel at %s", path);
		}
		bool linked;
		enum error_code const code = create(dir, name, path, bytes, flags, dest, &linked, e);
		if (code || linked)
			return code;
	}
	return error_fail(e, "the channel %s was removed while it was opened", path);
}

/** @brief Creates a directory and its missing parents, as mkdir -p does: the directory with mode
 *         0700, the parents with the umask's.
 *
 * @param dir    The directory; each parent is null-terminated in place while it is created.
 * @param length The length of its path.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_directory (char         *dir,
                size_t        length,
                struct error *e)
{
	// Each parent, from the first slash after the first byte.
	for (size_t end = 1; end < length; ++end) {
		if (dir[end] != '/')
			continue;
		dir[end] = '\0';
		int const err = mkdir(dir, 0777) ? errno : 0;
		if (err && err != EEXIST)
			return fail_errno(e, "create shared-memory directory", dir, err);
		dir[end] = '/';
	}
	if (mkdir(dir, 0700) && errno != EEXIST)
		return fail_errno(e, "create shared-memory directory", dir, errno);
	return ERROR_NONE;
}

/** @brief Opens a channel's directory, creating it if it is missing and asked to, and accepts only a
 *         real directory that this user owns and that grants nothing to others, so that no other
 *         user can plant or swap files in it.
 *
 * A creator gives the directory mode 0700, which a umask may have narrowed, whichever process made
 * it: one that finds a directory that another has just made cannot tell it from its own.
 *
 * @param dir    The directory; make_directory() cuts it in place while it creates it.
 * @param length The length of its path.
 * @param create Whether to create the directory if it is missing.
 * @param fd     Receives the directory's descriptor, or -1 if it is missing and not created.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, also for a missing directory, or ERROR_FAILED.
 */
static enum error_code
open_directory (char         *dir,
                size_t        length,
                bool          create,
                int          *fd,
                struct error *e)
{
	static int const oflag = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
	int dirfd = open(dir, oflag);
	if (dirfd < 0 && errno == ENOENT && create) {
		enum error_code const code = make_directory(dir, length, e);
		if (code)
			return code;
		dirfd = open(dir, oflag);
	}
	if (dirfd < 0) {
		int const err = errno;
		if (err == ENOENT)
			return ERROR_NONE;
		// O_NOFOLLOW with O_DIRECTORY refuses a symbolic link with ENOTDIR.
		if (err == ENOTDIR || err == ELOOP)
			return error_fail(e, "shared-memory directory must be owned by the current user and not a "
			                  "symlink: %s", dir);
		return fail_errno(e, "open shared-memory directory", dir, err);
	}

	struct stat st;
	enum error_code code = ERROR_NONE;
	if (fstat(dirfd, &st))
		code = fail_errno(e, "inspect shared-memory directory", dir, errno);
	else if (!S_ISDIR(st.st_mode) || st.st_uid != getuid())
		code = error_fail(e, "shared-memory directory must be owned by the current user and not a "
		                  "symlink: %s", dir);
	else if (st.st_mode & 077)
		code = error_fail(e, "shared-memory directory must be private (mode 0700): %s", dir);
	else if (create && (st.st_mode & 0700) != 0700 && fchmod(dirfd, 0700))
		code = fail_errno(e, "give mode 0700 to shared-memory directory", dir, errno);
	if (code) {
		// Nothing was written through the descriptor: close() has nothing to report.
		close(dirfd);
		dirfd = -1;
		return code;
	}
	*fd = dirfd;
	return ERROR_NONE;
}

enum error_code
shm_channel_open (struct shm_channel *dest,
                  char const         *path,
                  size_t              length,
                  size_t              bytes,
                  uint32_t            flags,
                  struct error       *e)
{
	*dest = (struct shm_channel){ .fd = -1 };
	// The channel's name, after the last slash.
	char const *const slash = memrchr(path, '/', length);
	char const *const name = slash ? slash + 1 : path;
	if (!*name)
		return error_fail(e, "the channel's path names a directory: %s", path);
	// Its directory, null-terminated: "." without a slash, without trailing slashes but at the root.
	size_t dir_length = slash ? (size_t)(slash - path) : 1;
	while (dir_length > 1 && path[dir_length - 1] == '/')
		--dir_length;
	dir_length += !dir_length;
	char *dir = malloc(dir_length + 1);
	if (!dir)
		return error_fail(e, "out of memory");
	memcpy(dir, slash ? path : ".", dir_length);
	dir[dir_length] = '\0';

	int dirfd = -1;
	enum error_code code = open_directory(dir, dir_length, flags & SHM_CHANNEL_CREATE, &dirfd, e);
	if (!code && dirfd < 0) {
		dest->flags = SHM_CHANNEL_MISSING;
		code = error_fail(e, "no channel at %s: its directory is missing", path);
	}
	if (!code) {
		code = open_in(dirfd, name, path, bytes, flags, dest, e);
		// Nothing was written through the descriptor: close() has nothing to report.
		close(dirfd);
		dirfd = -1;
	}
	free(dir);
	dir = nullptr;
	return code;
}

void
shm_channel_fini (struct shm_channel *channel)
{
	if (!channel)
		return;
	if (channel->h)
		// Unmapping a mapping that this process made cannot fail.
		munmap(channel->h, channel->bytes);
	if (channel->fd >= 0) {
		// Nothing was written through the descriptor: close() has nothing to report.
		close(channel->fd);
		channel->fd = -1;
	}
	*channel = (struct shm_channel){ .fd = -1 };
}
