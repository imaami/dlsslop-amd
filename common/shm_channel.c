/** @file
 *
 * The channel's file, opened and created in one place: shm_channel.h.
 */
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>
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

/** @brief Marks a channel as used for as long as its file is open: a read lock on its first byte
 *         that belongs to the open file description (F_OFD_SETLK), apart from dlsslopd's flock(2).
 *
 * A sweep of another protocol's version (sweep_take()) removes a channel only under a write lock
 * there, which no process that holds this lock lets it take. A file system without such locks
 * leaves the channel unmarked.
 *
 * @param fd The channel's file.
 * @return   false if a sweep holds the write lock, which it does only to remove the channel.
 */
static bool
hold (int fd)
{
	struct flock lock = { .l_type = F_RDLCK, .l_whence = SEEK_SET, .l_len = 1 };
	return !fcntl(fd, F_OFD_SETLK, &lock) || (errno != EAGAIN && errno != EACCES);
}

/** @brief Marks an open channel file as used (hold()) and maps it, if it holds a channel of this
 *         protocol and a sweep has not removed it.
 *
 * @param fd    The file.
 * @param path  Its path.
 * @param bytes How much of it to map.
 * @param flags What shm_channel_open() was asked to do.
 * @param dest  Receives the channel, which then holds @a fd; untouched on a failure.
 * @param gone  Receives whether a sweep has removed the file, or is removing it: then nothing is
 *              mapped, and ERROR_NONE returned.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
map_existing (int                 fd,
              char const         *path,
              size_t              bytes,
              uint32_t            flags,
              struct shm_channel *dest,
              bool               *gone,
              struct error       *e)
{
	// A sweep unlinks the file before it lets go of its write lock.
	bool const held = hold(fd);
	struct stat st;
	if (fstat(fd, &st))
		return fail_errno(e, "inspect shared-memory file", path, errno);
	*gone = !held || !st.st_nlink;
	if (*gone)
		return ERROR_NONE;
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
 *         size, mapped for writing, the native defaults, and marked as used (hold()).
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
	// Nobody else has the file to lock it.
	hold(fd);
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
	// next turn, and one removed meanwhile, as a sweep removes another protocol's, is created again.
	for (uint32_t turn = 0; turn < 4; ++turn) {
		int fd = openat(dir, name, oflag);
		if (fd >= 0) {
			bool gone;
			enum error_code const code = map_existing(fd, path, bytes, flags, dest, &gone, e);
			if (!code && !gone)
				return ERROR_NONE;
			// Nothing was written through the descriptor: close() has nothing to report.
			close(fd);
			fd = -1;
			if (code)
				return code;
			continue;
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
 * @param dest   Receives the directory's descriptor, or -1 if it is missing and not created.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, also for a missing directory, or ERROR_FAILED.
 */
static enum error_code
open_directory (char         *dir,
                size_t        length,
                bool          create,
                int          *dest,
                struct error *e)
{
	static int const oflag = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
	int fd = open(dir, oflag);
	if (fd < 0 && errno == ENOENT && create) {
		enum error_code const code = make_directory(dir, length, e);
		if (code)
			return code;
		fd = open(dir, oflag);
	}
	if (fd < 0) {
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
	if (fstat(fd, &st))
		code = fail_errno(e, "inspect shared-memory directory", dir, errno);
	else if (!S_ISDIR(st.st_mode) || st.st_uid != getuid())
		code = error_fail(e, "shared-memory directory must be owned by the current user and not a "
		                  "symlink: %s", dir);
	else if (st.st_mode & 077)
		code = error_fail(e, "shared-memory directory must be private (mode 0700): %s", dir);
	else if (create && (st.st_mode & 0700) != 0700 && fchmod(fd, 0700))
		code = fail_errno(e, "give mode 0700 to shared-memory directory", dir, errno);
	if (code) {
		// Nothing was written through the descriptor: close() has nothing to report.
		close(fd);
		fd = -1;
		return code;
	}
	*dest = fd;
	return ERROR_NONE;
}

/** @brief Which default channel a path names. */
enum shm_default {
	SHM_DEFAULT_NONE,   //!< None: the path was chosen.
	SHM_DEFAULT_SHARED, //!< ShmDefaultPath().
	SHM_DEFAULT_NATIVE, //!< ShmNativeDefaultPath().
};

/** @brief Whether a default's formatter wrote a path.
 *
 * @param path   The path.
 * @param length The length of its path.
 * @param buf    What the formatter wrote, in room for @a length bytes and a null.
 * @param n      What it returned.
 * @return       true if it wrote the path.
 */
static bool
formatted (char const *path,
           size_t      length,
           char const *buf,
           int         n)
{
	return n >= 0 && (size_t)n == length && !memcmp(buf, path, length);
}

/** @brief Which default channel a path names.
 *
 * @param path   The path.
 * @param length The length of its path.
 * @return       The default, or SHM_DEFAULT_NONE if it names none, or without memory to tell.
 */
static enum shm_default
default_path (char const *path,
              size_t      length)
{
	char *buf = malloc(length + 1);
	if (!buf)
		return SHM_DEFAULT_NONE;
	enum shm_default which = SHM_DEFAULT_NONE;
	int n = ShmDefaultPath(buf, length + 1);
	if (formatted(path, length, buf, n)) {
		which = SHM_DEFAULT_SHARED;
	} else {
		n = ShmNativeDefaultPath(buf, length + 1);
		if (formatted(path, length, buf, n))
			which = SHM_DEFAULT_NATIVE;
	}
	free(buf);
	buf = nullptr;
	return which;
}

/** @brief How long another protocol's files lie unmodified before the sweep removes them: a day. */
static constexpr time_t SWEEP_AGE = 24 * 60 * 60;

/** @brief The defaults whose directories this process has swept, as bits 1 << enum shm_default: one
 *         sweep a process is enough for files a day old.
 */
static _Atomic(uint32_t) swept;

/** @brief A sweep of a default channel's directory. */
struct sweep {
	time_t before; //!< The files last modified at this time or before go, unless they are in use.
	int    dir;    //!< The directory.
	uid_t  uid;    //!< The user, who owns the files that go.
};

/** @brief What a file that the sweep may remove is. */
enum sweep_kind {
	SWEEP_NONE,    //!< None: it stays.
	SWEEP_CHANNEL, //!< A channel.
	SWEEP_SOCKET,  //!< A channel's transport socket (kShmTransportSuffix).
	SWEEP_LOCK,    //!< A channel's producer lock (kShmProducerLockSuffix).
};

/** @brief The file type of each enum sweep_kind but SWEEP_NONE. */
static mode_t const SWEEP_TYPES[] = {
	[SWEEP_CHANNEL] = S_IFREG,
	[SWEEP_SOCKET]  = S_IFSOCK,
	[SWEEP_LOCK]    = S_IFREG,
};

/** @brief Whether a file is the sweep's to remove: this user's, of a type, and old.
 *
 * @param s    The sweep.
 * @param st   The file's status, not following a symbolic link.
 * @param type Its type, as S_IFMT masks it.
 * @return     true if it is.
 */
static bool
sweep_stale (struct sweep const *s,
             struct stat const  *st,
             mode_t              type)
{
	return (st->st_mode & S_IFMT) == type && st->st_uid == s->uid && st->st_mtim.tv_sec <= s->before;
}

/** @brief Takes an old channel that nobody uses: opens it and takes the lock that dlsslopd holds
 *         for as long as it serves a channel, and the write lock that no process that has the
 *         channel open lets anyone take (hold()).
 *
 * The file is the one that has the name once both locks are held: whoever removes the channel
 * holds them, and nobody else names a file there while the name exists.
 *
 * @param s    The sweep.
 * @param name The channel's name in the directory.
 * @return     The channel's descriptor, which holds both locks, or -1 if it is missing, not the
 *             sweep's or in use.
 */
static int
sweep_take (struct sweep const *s,
            char const         *name)
{
	// For writing, which a write lock needs.
	int fd = openat(s->dir, name, O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
	if (fd < 0)
		return -1;
	struct flock lock = { .l_type = F_WRLCK, .l_whence = SEEK_SET, .l_len = 1 };
	struct stat st, named;
	if (fstat(fd, &st) || !sweep_stale(s, &st, S_IFREG) || flock(fd, LOCK_EX | LOCK_NB)
	    || fcntl(fd, F_OFD_SETLK, &lock) || fstatat(s->dir, name, &named, AT_SYMLINK_NOFOLLOW)
	    || named.st_dev != st.st_dev || named.st_ino != st.st_ino) {
		// Nothing was written through the descriptor: close() has nothing to report.
		close(fd);
		fd = -1;
	}
	return fd;
}

/** @brief The length of the channel's name that a file's name starts with: kShmChannelName's
 *         scheme of any version, or the legacy shm.bin.
 *
 * @param name   The file's name.
 * @param legacy Whether shm.bin is a channel's name.
 * @return       The length, or 0 if the name starts with no channel's.
 */
static size_t
sweep_channel_length (char const *name,
                      bool        legacy)
{
	static char const untagged[] = "shm.bin";
	if (legacy && !strncmp(name, untagged, sizeof untagged - 1))
		return sizeof untagged - 1;
	static char const prefix[] = kShmChannelPrefix;
	static char const suffix[] = kShmChannelSuffix;
	if (strncmp(name, prefix, sizeof prefix - 1))
		return 0;
	size_t const digits = strspn(name + sizeof prefix - 1, "0123456789");
	size_t const end = sizeof prefix - 1 + digits;
	return digits && !strncmp(name + end, suffix, sizeof suffix - 1) ? end + sizeof suffix - 1 : 0;
}

/** @brief What a file is to the sweep, by what follows the channel's name in its name.
 *
 * @param rest What follows.
 * @return     The kind.
 */
static enum sweep_kind
sweep_kind (char const *rest)
{
	if (!*rest)
		return SWEEP_CHANNEL;
	if (!strcmp(rest, kShmTransportSuffix))
		return SWEEP_SOCKET;
	return strcmp(rest, kShmProducerLockSuffix) ? SWEEP_NONE : SWEEP_LOCK;
}

/** @brief Removes a file of a default channel's directory if it is another protocol's and the
 *         sweep's: sweep().
 *
 * @param s      The sweep.
 * @param name   The file's name.
 * @param legacy Whether shm.bin is a channel's name.
 */
static void
sweep_file (struct sweep const *s,
            char const         *name,
            bool                legacy)
{
	size_t const length = sweep_channel_length(name, legacy);
	enum sweep_kind const kind = length ? sweep_kind(name + length) : SWEEP_NONE;
	// This protocol's files are in use.
	bool const ours = length == sizeof kShmChannelName - 1 && !memcmp(name, kShmChannelName, length);
	if (kind == SWEEP_NONE || ours)
		return;
	// A failed removal leaves the file for the next sweep.
	if (kind == SWEEP_CHANNEL) {
		int fd = sweep_take(s, name);
		if (fd < 0)
			return;
		unlinkat(s->dir, name, 0);
		// Nothing was written through the descriptor: close() has nothing to report.
		close(fd);
		fd = -1;
		return;
	}
	// A channel's socket and producer lock go with it: under its locks, or once it is missing.
	char channel[sizeof ((struct dirent *)nullptr)->d_name];
	memcpy(channel, name, length);
	channel[length] = '\0';
	struct stat st;
	int fd = -1;
	if (!fstatat(s->dir, channel, &st, AT_SYMLINK_NOFOLLOW)) {
		fd = sweep_take(s, channel);
		if (fd < 0)
			return;
	} else if (errno != ENOENT) {
		return;
	}
	if (!fstatat(s->dir, name, &st, AT_SYMLINK_NOFOLLOW) && sweep_stale(s, &st, SWEEP_TYPES[kind]))
		unlinkat(s->dir, name, 0);
	if (fd >= 0) {
		// Nothing was written through the descriptor: close() has nothing to report.
		close(fd);
		fd = -1;
	}
}

/** @brief Removes other protocols' channels from a default channel's directory, with their
 *         sockets and producer locks: each that this user owns and that has not been modified for
 *         SWEEP_AGE, unless the channel it belongs to is younger or in use.
 *
 * A write through a mapping updates the modification time only on the first write after the time
 * last changed, so an old time alone does not tell that nobody uses a channel: every process that
 * has a channel open holds a read lock on it (hold()), and dlsslopd, also one of a protocol before
 * these locks, holds its flock(2). Symbolic links are neither followed nor removed. The legacy name
 * shm.bin is another protocol's only in the native tools' directory: in the shared one it can be
 * upstream's.
 *
 * @param dir    The directory.
 * @param legacy Whether shm.bin is a channel's name.
 */
static void
sweep (int  dir,
       bool legacy)
{
	time_t const now = time(nullptr);
	if (now == (time_t)-1)
		return;
	struct sweep const s = { .before = now - SWEEP_AGE, .dir = dir, .uid = getuid() };
	// readdir() reads a descriptor of its own, which closedir() closes.
	int fd = openat(dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (fd < 0)
		return;
	DIR *const entries = fdopendir(fd);
	if (!entries) {
		close(fd);
		fd = -1;
		return;
	}
	for (struct dirent const *entry; (entry = readdir(entries));)
		sweep_file(&s, entry->d_name, legacy);
	closedir(entries);
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
	char *directory = malloc(dir_length + 1);
	if (!directory)
		return error_fail(e, "out of memory");
	memcpy(directory, slash ? path : ".", dir_length);
	directory[dir_length] = '\0';

	int dir = -1;
	enum error_code code = open_directory(directory, dir_length, flags & SHM_CHANNEL_CREATE, &dir, e);
	if (!code && dir < 0) {
		dest->flags = SHM_CHANNEL_MISSING;
		code = error_fail(e, "no channel at %s: its directory is missing", path);
	}
	if (!code) {
		enum shm_default const which = default_path(path, length);
		uint32_t const bit = 1u << which;
		if (which && !(atomic_fetch_or_explicit(&swept, bit, memory_order_relaxed) & bit))
			sweep(dir, which == SHM_DEFAULT_NATIVE);
		code = open_in(dir, name, path, bytes, flags, dest, e);
		// Nothing was written through the descriptor: close() has nothing to report.
		close(dir);
		dir = -1;
	}
	free(directory);
	directory = nullptr;
	return code;
}

void
shm_channel_fini (struct shm_channel *channel)
{
	if (!channel)
		return;
	// Unmapping a mapping that this process made cannot fail.
	if (channel->h)
		munmap(channel->h, channel->bytes);
	if (channel->fd >= 0) {
		// Nothing was written through the descriptor: close() has nothing to report.
		close(channel->fd);
		channel->fd = -1;
	}
	*channel = (struct shm_channel){ .fd = -1 };
}
