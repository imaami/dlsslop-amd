/** @file
 *
 * shm_channel_open() in C, in private directories. Processes that open a missing channel at once,
 * released together, all map the same file, which exactly one of them created and which holds the
 * native defaults; nothing else is left in the directory. That holds for each way of creating it:
 * an unnamed file that linkat(2) names by its descriptor or by its link in /proc, and a named
 * temporary file where the directory holds no unnamed files or no link names one; and it holds
 * when the racers also make the channel's directory under a umask that narrows its mode. The test
 * forces the other ways by defining openat() and linkat() itself, which the module's calls then
 * reach. A file of another protocol, size or type, a symbolic link and a directory that is not
 * private are refused and left as they were, and a directory replaced while a channel is created
 * receives nothing; an open that may not create finds nothing and creates nothing. A channel that
 * a sweep of another version removes while a process opens it is not used. Opening a default
 * channel removes, once a process, another protocol's files that have lain unused for a day, and
 * nothing that is in use or not such a file: the native tools' default in a private /tmp of a user
 * and mount namespace, and the shared default there too, or in /tmp under a DLSSNR_UID of the
 * test's own where namespaces are not available.
 */
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "error.h"
#include "shm_channel.h"
#include "shm_protocol.h"
#include "support.h"

/** @brief How the test's openat() and linkat() fail the module's calls. */
enum force {
	FORCE_NONE,        //!< They do not: unnamed files, named by their descriptors.
	FORCE_PROC,        //!< linkat() of a descriptor fails, as without CAP_DAC_READ_SEARCH on older kernels.
	FORCE_NO_LINK,     //!< No linkat() names an unnamed file, as without /proc there either.
	FORCE_NO_TMPFILE,  //!< An unnamed file fails as on a file system without them.
	FORCE_OLD_KERNEL,  //!< An unnamed file fails as on a kernel before 3.11.
	FORCE_COUNT,       //!< The number of ways.
};

/** @brief The ways, by enum force, for the messages. */
static char const *const FORCE_NAMES[] = {
	[FORCE_NONE]       = "linkat() of the descriptor",
	[FORCE_PROC]       = "linkat() of the descriptor's link in /proc",
	[FORCE_NO_LINK]    = "a named file after no link named an unnamed one",
	[FORCE_NO_TMPFILE] = "a named file on a file system without unnamed ones",
	[FORCE_OLD_KERNEL] = "a named file on a kernel before 3.11",
};

/** @brief How the test's openat() and linkat() fail the module's calls now; children inherit it. */
static enum force force;

/** @brief The unnamed files that openat() was asked for, and the links that linkat() made. */
static uint32_t unnamed_files, descriptor_links, proc_links, named_links;

/** @brief A directory that linkat() replaces with one that others may enter before it links, as
 *         another process could; nullptr if none.
 */
static char const *replace;

/** @brief The name of a channel that openat() removes as a sweep of another protocol's version
 *         would, right after the module opens it; nullptr if none.
 */
static char const *sweep_name;

/** @brief Whether that sweep still holds its write lock when the module locks the channel, and
 *         removes it only when the module opens the name again.
 */
static bool sweep_holds;

/** @brief That sweep's descriptor while it holds its write lock, or -1. */
static int sweeper = -1;

/** @brief The processes that open a missing channel at once. */
static constexpr uint32_t RACERS = 32;

/** @brief The races for each way of creating a channel. */
static constexpr uint32_t ROUNDS = 8;

/** @brief The test's own process, which alone removes the test's directory. */
static pid_t owner;

/** @brief The private directory that the test works in. */
static char *root;

/** @brief Its path's length. */
static size_t root_length;

/** @brief What a channel holds when it has been created: a header that main() maps and initializes
 *         once.
 */
static struct ShmHeader *defaults;

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param fmt       A printf format for what failed.
 * @param ...       The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;
	va_list args;
	va_start(args, fmt);
	fputs("shm-channel-test: ", stderr);
	vfprintf(stderr, fmt, args);
	va_end(args);
	fputc('\n', stderr);
	exit(1);
}

/** @brief The test's openat(2), which the module's calls reach: counts the unnamed files asked for,
 *         fails them as enum force says, and runs the sweep that sweep_name asks for.
 *
 * @param dir   The directory.
 * @param path  The path, relative to it.
 * @param flags open(2)'s flags.
 * @param ...   The mode, with O_CREAT or O_TMPFILE.
 * @return      The descriptor, or -1 with errno set.
 */
int
openat (int         dir,
        char const *path,
        int         flags,
        ...)
{
	mode_t mode = 0;
	if (flags & O_CREAT || (flags & O_TMPFILE) == O_TMPFILE) {
		va_list args;
		va_start(args, flags);
		mode = va_arg(args, mode_t);
		va_end(args);
	}
	if ((flags & O_TMPFILE) == O_TMPFILE) {
		++unnamed_files;
		if (force == FORCE_NO_TMPFILE || force == FORCE_OLD_KERNEL) {
			errno = force == FORCE_NO_TMPFILE ? EOPNOTSUPP : EISDIR;
			return -1;
		}
	}
	if (sweeper >= 0) {
		// The sweep ends as one does: the name goes, then the lock.
		require(!unlinkat(dir, sweep_name, 0) && !close(sweeper), "the sweep failed");
		sweeper = -1;
		sweep_name = nullptr;
	}
	int const fd = (int)syscall(SYS_openat, dir, path, flags, mode);
	if (fd < 0 || !sweep_name || strcmp(path, sweep_name))
		return fd;
	// The sweep takes the channel after the module opened it, before the module locks it.
	sweeper = (int)syscall(SYS_openat, dir, path, O_RDWR | O_CLOEXEC, 0);
	struct flock lock = { .l_type = F_WRLCK, .l_whence = SEEK_SET, .l_len = 1 };
	require(sweeper >= 0 && !fcntl(sweeper, F_OFD_SETLK, &lock), "the sweep cannot lock %s", path);
	if (!sweep_holds) {
		require(!unlinkat(dir, path, 0) && !close(sweeper), "the sweep failed");
		sweeper = -1;
		sweep_name = nullptr;
	}
	return fd;
}

/** @brief glibc's openat(2) for a call without a mode whose flags are not constant: with
 *         _FORTIFY_SOURCE, as Ubuntu's GCC enables by default, the module's three-argument calls
 *         reach this and not openat(), so it is the test's openat() too.
 *
 * @param dir   The directory.
 * @param path  The path, relative to it.
 * @param flags open(2)'s flags, which need no mode.
 * @return      The descriptor, or -1 with errno set.
 */
int
__openat_2 (int         dir,
            char const *path,
            int         flags)
{
	require(!(flags & O_CREAT) && (flags & O_TMPFILE) != O_TMPFILE, "an openat() without a mode asked for %#x",
	        (unsigned)flags);
	return openat(dir, path, flags);
}

/** @brief The test's linkat(2), which the module's calls reach: replaces the directory named by
 *         replace first, fails the links of unnamed files as enum force says, and counts the links
 *         made of each kind.
 *
 * @param from_dir The directory of the file to link.
 * @param from     Its path, relative to that directory.
 * @param to_dir   The directory of the new name.
 * @param to       The new name, relative to that directory.
 * @param flags    linkat(2)'s flags.
 * @return         0, or -1 with errno set.
 */
int
linkat (int         from_dir,
        char const *from,
        int         to_dir,
        char const *to,
        int         flags)
{
	if (replace) {
		// Empty: the file to link has no name in it yet.
		require(!rmdir(replace) && !mkdir(replace, 0700) && !chmod(replace, 0755), "cannot replace %s",
		        replace);
		replace = nullptr;
	}
	// An unnamed file is linked by its descriptor, or by the descriptor's link in /proc.
	bool const descriptor = flags & AT_EMPTY_PATH;
	bool const unnamed = descriptor || !strncmp(from, "/proc/", sizeof "/proc/" - 1);
	if (unnamed && (force == FORCE_NO_LINK || (descriptor && force == FORCE_PROC))) {
		errno = ENOENT;
		return -1;
	}
	int const linked = (int)syscall(SYS_linkat, from_dir, from, to_dir, to, flags);
	if (!linked)
		++*(descriptor ? &descriptor_links : unnamed ? &proc_links : &named_links);
	return linked;
}

/** @brief A path in the test's directory, on the heap.
 *
 * @param length Receives its length, or nullptr.
 * @param name   What follows the directory.
 * @return       The path, which the caller frees.
 */
static char *
path_in (size_t     *length,
         char const *name)
{
	char *const path = support_format(length, "%s/%s", root, name);
	require(path, "out of memory");
	return path;
}

/** @brief The names in a directory, but . and .., sorted and joined by spaces, on the heap. */
static char *
listing (char const *dir)
{
	DIR *const entries = opendir(dir);
	require(entries, "cannot read %s", dir);
	char *names[64];
	uint32_t count = 0;
	for (;;) {
		struct dirent const *entry;
		require(support_next_entry(entries, &entry), "cannot read %s", dir);
		if (!entry)
			break;
		if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
			continue;
		require(count < sizeof names / sizeof *names, "%s holds too many files", dir);
		names[count] = strdup(entry->d_name);
		require(names[count++], "out of memory");
	}
	require(!closedir(entries), "closedir failed");
	// Insertion sort: a handful of names.
	for (uint32_t i = 1; i < count; ++i)
		for (uint32_t j = i; j && strcmp(names[j - 1], names[j]) > 0; --j) {
			char *const swap = names[j];
			names[j] = names[j - 1];
			names[j - 1] = swap;
		}
	struct support_text text = {};
	for (uint32_t i = 0; i < count; ++i) {
		require(support_text_printf(&text, "%s%s", i ? " " : "", names[i]), "out of memory");
		free(names[i]);
		names[i] = nullptr;
	}
	char *const joined = strdup(support_text_string(&text));
	require(joined, "out of memory");
	support_text_fini(&text);
	return joined;
}

/** @brief Opens a channel and ends the test if that fails.
 *
 * @param channel Receives the channel.
 * @param path    Its path.
 * @param bytes   How much of it to map.
 * @param flags   shm_channel_open()'s flags.
 */
static void
open_channel (struct shm_channel *channel,
              char const         *path,
              size_t              bytes,
              uint32_t            flags)
{
	struct error e;
	require(!shm_channel_open(channel, path, strlen(path), bytes, flags, &e), "%s", e.what);
	require(channel->h && channel->fd >= 0 && channel->bytes == bytes, "an open channel is not mapped");
}

/** @brief Opens a channel that must be refused, and ends the test unless it is, saying why in words
 *         that hold each of the needles.
 *
 * @param path    The channel's path.
 * @param flags   shm_channel_open()'s flags.
 * @param needles The words, ending with nullptr.
 */
static void
refused (char const         *path,
         uint32_t            flags,
         char const *const  *needles)
{
	struct shm_channel channel;
	struct error e;
	require(shm_channel_open(&channel, path, strlen(path), kHeaderBytes, flags, &e) == ERROR_FAILED,
	        "%s was not refused", path);
	require(!channel.h && channel.fd == -1, "a refused channel is open");
	for (; *needles; ++needles)
		require(strstr(e.what, *needles), "refusing %s did not say '%s': %s", path, *needles, e.what);
}

/** @brief What a racer saw, in a mapping that the racers share with the test. */
struct racer {
	uint64_t inode;  //!< The channel's inode.
	uint32_t flags;  //!< The channel's flags.
	uint32_t failed; //!< 0, or the line that found something wrong.
};

/** @brief Opens a channel once the gate opens, and records what it saw.
 *
 * @param gate  The gate's end to read: its end of file opens it.
 * @param path  The channel.
 * @param bytes How much of it to map.
 * @param seen  Receives what the racer saw.
 */
[[noreturn]] static void
race (int            gate,
      char const    *path,
      size_t         bytes,
      struct racer  *seen)
{
	char byte;
	if (read(gate, &byte, 1))
		_exit(1);
	struct shm_channel channel;
	struct error e;
	if (shm_channel_open(&channel, path, strlen(path), bytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, &e)) {
		seen->failed = __LINE__;
		_exit(1);
	}
	struct stat st;
	if (fstat(channel.fd, &st))
		seen->failed = __LINE__;
	else if (st.st_size != (off_t)ShmTotalBytes() || (st.st_mode & 0777) != 0600)
		seen->failed = __LINE__;
	else if (memcmp(channel.h, defaults, sizeof *defaults))
		seen->failed = __LINE__;
	seen->inode = st.st_ino;
	seen->flags = channel.flags;
	shm_channel_fini(&channel);
	_exit(0);
}

/** @brief RACERS processes open a missing channel at once: each maps the same file, which holds the
 *         defaults, exactly one of them created it, and the directory holds nothing else.
 *
 * In odd rounds the racers make the channel's directory too, under a umask that leaves it without
 * its owner's write bit until a creator gives it mode 0700: each finds it private.
 *
 * @param round Which race this is, which names the channel.
 */
static void
check_race (uint32_t round)
{
	bool const missing = round & 1;
	char *race_dir = support_format(nullptr, "%s/race-%u-%u", root, (unsigned)force, round);
	require(race_dir && !mkdir(race_dir, 0700), "cannot make a race's directory");
	char *dir = missing ? support_format(nullptr, "%s/sub", race_dir) : race_dir;
	require(dir, "out of memory");
	char *path = support_format(nullptr, "%s/channel", dir);
	require(path, "out of memory");
	struct racer *const seen = mmap(nullptr, RACERS * sizeof *seen, PROT_READ | PROT_WRITE,
	                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	require(seen != MAP_FAILED, "mmap failed");
	int gate[2];
	require(!pipe2(gate, O_CLOEXEC), "pipe2 failed");
	require(!fflush(nullptr), "fflush failed");
	mode_t const mask = missing ? umask(0277) : 0;
	pid_t racers[RACERS];
	// Half map the header, half the whole channel.
	for (uint32_t i = 0; i < RACERS; ++i) {
		racers[i] = fork();
		require(racers[i] >= 0, "fork failed");
		if (!racers[i]) {
			close(gate[1]);
			race(gate[0], path, i & 1 ? ShmTotalBytes() : kHeaderBytes, &seen[i]);
		}
	}
	if (missing)
		umask(mask);
	// Every racer reads end of file at once.
	require(!close(gate[1]) && !close(gate[0]), "close failed");
	uint32_t created = 0;
	for (uint32_t i = 0; i < RACERS; ++i) {
		int status;
		require(waitpid(racers[i], &status, 0) == racers[i], "waitpid failed");
		require(WIFEXITED(status) && !WEXITSTATUS(status) && !seen[i].failed,
		        "%s: racer %u failed (line %u)", FORCE_NAMES[force], i, seen[i].failed);
		require(seen[i].inode == seen[0].inode, "%s: racers mapped different files", FORCE_NAMES[force]);
		created += !!(seen[i].flags & SHM_CHANNEL_CREATED);
	}
	require(created == 1, "%s: %u racers created the channel", FORCE_NAMES[force], created);
	char *names = listing(dir);
	require(!strcmp(names, "channel"), "%s: a race left %s", FORCE_NAMES[force], names);
	free(names);
	names = nullptr;
	struct stat st;
	require(!stat(dir, &st) && (st.st_mode & 0777) == 0700, "%s: a race's directory is not private",
	        FORCE_NAMES[force]);
	require(!munmap(seen, RACERS * sizeof *seen), "munmap failed");
	free(path);
	path = nullptr;
	if (missing) {
		free(dir);
		dir = nullptr;
	}
	free(race_dir);
	race_dir = nullptr;
}

/** @brief Creating a channel in each way: a missing directory is made private, the channel holds the
 *         defaults, mode 0600 and the channel's size, the way was the one forced, and racers agree.
 *         Missing parents are made as mkdir -p makes them.
 */
static void
check_create (void)
{
	// A umask that would leave the directory and the file without what they need.
	mode_t const mask = umask(0277);
	for (force = FORCE_NONE; force < FORCE_COUNT; ++force) {
		char name[32];
		snprintf(name, sizeof name, "made-%u/channel", (unsigned)force);
		char *path = path_in(nullptr, name);
		unnamed_files = descriptor_links = proc_links = named_links = 0;
		struct shm_channel channel;
		open_channel(&channel, path, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
		require(channel.flags == (SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE | SHM_CHANNEL_CREATED),
		        "%s: a new channel's flags are %#x", FORCE_NAMES[force], channel.flags);
		require(!memcmp(channel.h, defaults, sizeof *defaults), "%s: a new channel lacks the defaults",
		        FORCE_NAMES[force]);
		struct stat st;
		require(!fstat(channel.fd, &st) && st.st_size == (off_t)ShmTotalBytes()
		        && (st.st_mode & 0777) == 0600 && st.st_nlink == 1,
		        "%s: a new channel's size, mode or links are wrong", FORCE_NAMES[force]);
		static uint32_t const files[] = { 1, 1, 1, 1, 1 };
		static uint32_t const by_descriptor[] = { 1, 0, 0, 0, 0 };
		static uint32_t const by_proc[] = { 0, 1, 0, 0, 0 };
		static uint32_t const by_name[] = { 0, 0, 1, 1, 1 };
		require(unnamed_files == files[force] && descriptor_links == by_descriptor[force]
		        && proc_links == by_proc[force] && named_links == by_name[force],
		        "%s: the channel was made another way (%u %u %u %u)", FORCE_NAMES[force], unnamed_files,
		        descriptor_links, proc_links, named_links);
		shm_channel_fini(&channel);
		require(channel.fd == -1 && !channel.h && !channel.flags, "fini left a channel open");

		char *dir = path_in(nullptr, name);
		*strrchr(dir, '/') = '\0';
		require(!stat(dir, &st) && (st.st_mode & 0777) == 0700, "a created directory is not private");
		char *names = listing(dir);
		require(!strcmp(names, "channel"), "%s: creating left %s", FORCE_NAMES[force], names);
		free(names);
		names = nullptr;
		free(dir);
		dir = nullptr;

		// Opened again, nothing is created, also without asking to write.
		open_channel(&channel, path, ShmTotalBytes(), SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
		require(channel.flags == (SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE), "an existing channel was created");
		atomic_store(&channel.h->passes, 3);
		shm_channel_fini(&channel);
		open_channel(&channel, path, kHeaderBytes, 0);
		require(!channel.flags && atomic_load(&channel.h->passes) == 3, "a channel was not opened as it was");
		shm_channel_fini(&channel);
		free(path);
		path = nullptr;
	}
	umask(mask);

	char *nested = path_in(nullptr, "nested/parent/dir/channel");
	struct shm_channel channel;
	open_channel(&channel, nested, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
	require(channel.flags & SHM_CHANNEL_CREATED, "a channel in missing parents was not created");
	shm_channel_fini(&channel);
	struct stat st;
	*strrchr(nested, '/') = '\0';
	require(!stat(nested, &st) && (st.st_mode & 0777) == 0700, "a created directory is not private");
	free(nested);
	nested = nullptr;

	// A directory that grants others nothing but lacks its owner's write bit, as a umask leaves one:
	// a reader opens it as it is, and a creator gives it mode 0700.
	char *narrow = path_in(nullptr, "narrow");
	require(!mkdir(narrow, 0700) && !chmod(narrow, 0500), "cannot make a narrow directory");
	char *in_narrow = path_in(nullptr, "narrow/channel");
	struct error e;
	require(shm_channel_open(&channel, in_narrow, strlen(in_narrow), kHeaderBytes, 0, &e)
	        && channel.flags == SHM_CHANNEL_MISSING, "a reader refused a narrow directory: %s", e.what);
	require(!stat(narrow, &st) && (st.st_mode & 0777) == 0500, "a reader changed a directory's mode");
	open_channel(&channel, in_narrow, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
	require(channel.flags & SHM_CHANNEL_CREATED, "a channel in a narrow directory was not created");
	shm_channel_fini(&channel);
	require(!stat(narrow, &st) && (st.st_mode & 0777) == 0700, "a creator left a directory narrow");
	free(in_narrow);
	in_narrow = nullptr;
	free(narrow);
	narrow = nullptr;

	for (force = FORCE_NONE; force < FORCE_COUNT; ++force)
		for (uint32_t round = 0; round < ROUNDS; ++round)
			check_race(round);
	force = FORCE_NONE;
}

/** @brief Writes a file.
 *
 * @param path  The file.
 * @param bytes What it holds.
 * @param size  Their number.
 * @param total The file's size: more than @a size leaves zeros after them.
 */
static void
write_file (char const *path,
            void const *bytes,
            size_t      size,
            off_t       total)
{
	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
	require(fd >= 0 && write(fd, bytes, size) == (ssize_t)size && !ftruncate(fd, total) && !close(fd),
	        "cannot write %s", path);
	fd = -1;
}

/** @brief Files that hold no channel of this protocol are refused, by readers and creators alike,
 *         and left as they were; a symbolic link, a FIFO and a directory that is not private too.
 */
static void
check_refusals (void)
{
	char *path = path_in(nullptr, "other-version");
	struct shm_channel channel;
	open_channel(&channel, path, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
	atomic_store(&channel.h->version, kShmVersion + 1);
	shm_channel_fini(&channel);
	char other[16], ours[16];
	snprintf(other, sizeof other, "v%u,", (unsigned)kShmVersion + 1);
	snprintf(ours, sizeof ours, "v%u:", (unsigned)kShmVersion);
	struct stat before, after;
	require(!stat(path, &before), "stat failed");
	for (uint32_t flags = 0; flags <= SHM_CHANNEL_CREATE; flags += SHM_CHANNEL_CREATE)
		refused(path, flags | SHM_CHANNEL_WRITE,
		        (char const *const[]){ path, other, ours, "remove it, or use another channel", nullptr });
	require(!stat(path, &after) && after.st_ino == before.st_ino
	        && after.st_mtim.tv_sec == before.st_mtim.tv_sec && after.st_mtim.tv_nsec == before.st_mtim.tv_nsec,
	        "a refused channel was changed");

	// No channel at all: zeros, or a file of another size.
	char *zeros = path_in(nullptr, "zeros");
	write_file(zeros, "", 0, (off_t)ShmTotalBytes());
	refused(zeros, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	        (char const *const[]){ zeros, "holds no channel (magic 0, version 0;", nullptr });
	char *notes = path_in(nullptr, "notes");
	write_file(notes, "notes\n", 6, 6);
	char size[64];
	snprintf(size, sizeof size, "holds no v%u channel: it has 6 bytes, not %zu;", (unsigned)kShmVersion,
	         ShmTotalBytes());
	refused(notes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, (char const *const[]){ notes, size, nullptr });
	uint32_t const head[2] = { kShmMagic, kShmVersion - 1 };
	char *short_channel = path_in(nullptr, "short");
	write_file(short_channel, head, sizeof head, 4096);
	snprintf(other, sizeof other, "v%u,", (unsigned)kShmVersion - 1);
	refused(short_channel, 0, (char const *const[]){ short_channel, other, ours, nullptr });
	char *text = support_read_file(notes, nullptr);
	require(text && !strcmp(text, "notes\n"), "a refused file was changed");
	free(text);
	text = nullptr;

	// A symbolic link to a channel, a FIFO, and a directory.
	char *good = path_in(nullptr, "good");
	open_channel(&channel, good, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
	shm_channel_fini(&channel);
	char *link = path_in(nullptr, "link");
	require(!symlink(good, link), "symlink failed");
	refused(link, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	        (char const *const[]){ "must not be a symlink", link, nullptr });
	char *fifo = path_in(nullptr, "fifo");
	require(!mkfifo(fifo, 0600), "mkfifo failed");
	refused(fifo, 0, (char const *const[]){ "must be regular and owned by the current user", nullptr });
	char *dir = path_in(nullptr, "a directory");
	require(!mkdir(dir, 0700), "mkdir failed");
	refused(dir, 0, (char const *const[]){ "must be regular", nullptr });
	refused(dir, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	        (char const *const[]){ "open shared-memory file ", dir, nullptr });
	char *trailing = path_in(nullptr, "a directory/");
	refused(trailing, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	        (char const *const[]){ "names a directory", nullptr });

	// A directory that others may enter, and one that is a symbolic link, receive no channel.
	char *shared = path_in(nullptr, "shared");
	require(!mkdir(shared, 0700) && !chmod(shared, 0755), "cannot make a shared directory");
	char *in_shared = path_in(nullptr, "shared/channel");
	refused(in_shared, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	        (char const *const[]){ "shared-memory directory must be private (mode 0700): ", shared, nullptr });
	char *linked = path_in(nullptr, "linked");
	require(!symlink(dir, linked), "symlink failed");
	char *in_linked = path_in(nullptr, "linked/channel");
	refused(in_linked, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	        (char const *const[]){ "must be owned by the current user and not a symlink", nullptr });
	char *names = listing(shared);
	require(!*names, "a shared directory received %s", names);
	free(names);
	names = listing(dir);
	require(!*names, "a linked directory received %s", names);
	free(names);
	names = nullptr;

	free(linked);
	linked = nullptr;
	free(in_linked);
	in_linked = nullptr;
	free(shared);
	shared = nullptr;
	free(in_shared);
	in_shared = nullptr;
	free(trailing);
	trailing = nullptr;
	free(dir);
	dir = nullptr;
	free(fifo);
	fifo = nullptr;
	free(link);
	link = nullptr;
	free(good);
	good = nullptr;
	free(short_channel);
	short_channel = nullptr;
	free(notes);
	notes = nullptr;
	free(zeros);
	zeros = nullptr;
	free(path);
	path = nullptr;
}

/** @brief A channel's directory that another process replaces, while this one creates the channel,
 *         with one that others may enter receives nothing: whatever way the module then tries works
 *         in the directory that it checked, which has gone.
 */
static void
check_replaced (void)
{
	char *dir = path_in(nullptr, "replaced");
	require(!mkdir(dir, 0700), "mkdir failed");
	char *path = path_in(nullptr, "replaced/channel");
	struct shm_channel channel;
	struct error e;
	replace = dir;
	require(shm_channel_open(&channel, path, strlen(path), kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
	                         &e) == ERROR_FAILED && strstr(e.what, path),
	        "a channel was created in a replaced directory");
	require(!replace, "the directory was not replaced");
	require(!channel.h && channel.fd == -1, "a refused channel is open");
	char *names = listing(dir);
	require(!*names, "a replaced directory received %s", names);
	free(names);
	names = nullptr;
	struct stat st;
	require(!stat(dir, &st) && (st.st_mode & 0777) == 0755, "a replaced directory was changed");
	free(path);
	path = nullptr;
	free(dir);
	dir = nullptr;
}

/** @brief A channel that a sweep of another protocol's version removes while this process opens it
 *         is not used, whether the sweep removed it before the module locked it or still held its
 *         write lock then: a creator makes a new channel, and a reader finds none.
 */
static void
check_swept (void)
{
	static struct {
		uint32_t flags; //!< shm_channel_open()'s flags.
		bool     holds; //!< sweep_holds.
	} const opens[] = {
		{ SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, false },
		{ SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, true  },
		{ 0,                                      false },
	};
	for (uint32_t i = 0; i < sizeof opens / sizeof *opens; ++i) {
		char name[16];
		snprintf(name, sizeof name, "swept-%u", i);
		char *path = path_in(nullptr, name);
		struct shm_channel channel;
		open_channel(&channel, path, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
		// A mark of the channel that the sweep removes.
		uint32_t const passes = atomic_load(&defaults->passes);
		atomic_store(&channel.h->passes, passes + 1);
		shm_channel_fini(&channel);
		sweep_name = name;
		sweep_holds = opens[i].holds;
		struct error e;
		bool const opened = !shm_channel_open(&channel, path, strlen(path), kHeaderBytes, opens[i].flags, &e);
		require(!sweep_name && sweeper < 0, "the sweep did not end");
		struct stat st;
		if (opens[i].flags)
			require(opened && channel.flags & SHM_CHANNEL_CREATED && atomic_load(&channel.h->passes) == passes
			        && !fstat(channel.fd, &st) && st.st_nlink == 1,
			        "a channel that a sweep removed was used (open %u): %s", i, opened ? "opened" : e.what);
		else
			require(!opened && channel.flags == SHM_CHANNEL_MISSING, "a reader used a channel that a sweep removed");
		shm_channel_fini(&channel);
		free(path);
		path = nullptr;
	}
}

/** @brief An open that may not create finds no channel, says so, and creates nothing: neither the
 *         channel nor its directory. A bare name is a channel in the working directory.
 */
static void
check_missing (void)
{
	char *missing_dir = path_in(nullptr, "missing/channel");
	char *missing = path_in(nullptr, "missing-channel");
	for (uint32_t flags = 0; flags <= SHM_CHANNEL_WRITE; ++flags)
		for (char const *const *path = (char const *const[]){ missing_dir, missing, nullptr }; *path; ++path) {
			struct shm_channel channel;
			struct error e;
			require(shm_channel_open(&channel, *path, strlen(*path), kHeaderBytes, flags, &e)
			        && channel.flags == SHM_CHANNEL_MISSING && channel.fd == -1 && !channel.h
			        && strstr(e.what, "no channel at ") && strstr(e.what, *path),
			        "a missing channel was not reported as missing: %s", e.what);
		}
	struct stat st;
	require(stat(missing, &st) && errno == ENOENT, "a reader created a channel");
	*strrchr(missing_dir, '/') = '\0';
	require(stat(missing_dir, &st) && errno == ENOENT, "a reader created a directory");

	char *cwd = getcwd(nullptr, 0);
	require(cwd && !chdir(root), "chdir failed");
	struct shm_channel channel;
	open_channel(&channel, "bare", kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
	require(channel.flags & SHM_CHANNEL_CREATED, "a bare name was not created");
	shm_channel_fini(&channel);
	require(!chdir(cwd), "chdir failed");
	free(cwd);
	cwd = nullptr;
	char *bare = path_in(nullptr, "bare");
	require(!stat(bare, &st) && S_ISREG(st.st_mode), "a bare name is not in the working directory");
	free(bare);
	bare = nullptr;
	free(missing);
	missing = nullptr;
	free(missing_dir);
	missing_dir = nullptr;
}

/** @brief What a file that a sweep finds is. */
enum sweep_type {
	SWEEP_FILE,      //!< A regular file.
	SWEEP_LOCKED,    //!< A regular file that the test locks, as dlsslopd locks its channel.
	SWEEP_HELD,      //!< A regular file that the test holds open, as every program its channel.
	SWEEP_SOCKET,    //!< A socket, as a socket unit leaves one.
	SWEEP_SYMLINK,   //!< A symbolic link to "target".
	SWEEP_DIRECTORY, //!< A directory.
	SWEEP_FIFO,      //!< A FIFO.
};

/** @brief A file that a sweep finds. */
struct sweep_case {
	char const *name;   //!< Its name.
	uint32_t    hours;  //!< How long ago it was modified.
	uint8_t     type;   //!< enum sweep_type.
	bool        shared; //!< Whether it stays in the shared default's directory.
	bool        native; //!< Whether it stays in the native tools' default's directory.
};

/** @brief What a sweep finds: another protocol's channel, its socket and producer lock, old or
 *         young, missing, locked, held open or not a regular file; the legacy shm.bin; this
 *         protocol's files; and other names, a creator's temporary name among them.
 */
static struct sweep_case const SWEEP_CASES[] = {
	{"shm-v30.bin",                 25, SWEEP_FILE,      false, false},
	{"shm-v30.bin.sock",            25, SWEEP_SOCKET,    false, false},
	{"shm-v30.bin.producer.lock",   25, SWEEP_FILE,      false, false},
	{"shm-v30.bin.Ab9xYz",          25, SWEEP_FILE,      true,  true },
	{"shm-v30.bin.Ab9xY",           25, SWEEP_FILE,      true,  true },
	{"shm-v30.bin.lock",            25, SWEEP_FILE,      true,  true },
	{"shm-v30.binx",                25, SWEEP_FILE,      true,  true },
	{"shm-v29.bin",                 23, SWEEP_FILE,      true,  true },
	{"shm-v29.bin.sock",            25, SWEEP_SOCKET,    true,  true },
	{"shm-v29.bin.producer.lock",   25, SWEEP_FILE,      true,  true },
	{"shm-v28.bin",                 25, SWEEP_LOCKED,    true,  true },
	{"shm-v28.bin.sock",            25, SWEEP_SOCKET,    true,  true },
	{"shm-v23.bin",                 25, SWEEP_HELD,      true,  true },
	{"shm-v23.bin.producer.lock",   25, SWEEP_FILE,      true,  true },
	{"shm-v27.bin.producer.lock",   25, SWEEP_FILE,      false, false},
	{"shm-v27.bin.sock",            23, SWEEP_SOCKET,    true,  true },
	{"shm-v26.bin",                 25, SWEEP_SYMLINK,   true,  true },
	{"target",                      25, SWEEP_FILE,      true,  true },
	{"shm-v25.bin",                 25, SWEEP_DIRECTORY, true,  true },
	{"shm-v24.bin",                 25, SWEEP_FIFO,      true,  true },
	{"shm-v1.bin",                  99, SWEEP_FILE,      false, false},
	{"shm-v.bin",                   25, SWEEP_FILE,      true,  true },
	{"shm.bin",                     25, SWEEP_FILE,      true,  false},
	{"shm.bin.sock",                25, SWEEP_SOCKET,    true,  false},
	{"shm.bin.producer.lock",       25, SWEEP_FILE,      true,  false},
	{kShmChannelName ".sock",       25, SWEEP_SOCKET,    true,  true },
	{kShmChannelName ".backup",     25, SWEEP_FILE,      true,  true },
	{".shm-channel-0123456789abcdef", 25, SWEEP_FILE,    true,  true },
	{"layer.log",                   25, SWEEP_FILE,      true,  true },
};

/** @brief Makes a file of a sweep's directory, modified some hours ago.
 *
 * @param dir  The directory.
 * @param c    The file.
 * @param now  The time now.
 */
static void
sweep_make (char const              *dir,
            struct sweep_case const *c,
            time_t                   now)
{
	size_t length;
	char *path = support_format(&length, "%s/%s", dir, c->name);
	require(path, "out of memory");
	switch (c->type) {
	case SWEEP_FILE:
	case SWEEP_LOCKED:
	case SWEEP_HELD:
		write_file(path, "", 0, 0);
		break;
	case SWEEP_SOCKET: {
		struct sockaddr_un address = { .sun_family = AF_UNIX };
		require(length < sizeof address.sun_path, "%s is too long for a socket", path);
		memcpy(address.sun_path, path, length + 1);
		int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
		require(fd >= 0 && !bind(fd, (struct sockaddr const *)&address, sizeof address) && !close(fd),
		        "cannot make the socket %s", path);
		fd = -1;
		break;
	}
	case SWEEP_SYMLINK:
		require(!symlink("target", path), "symlink failed");
		break;
	case SWEEP_DIRECTORY:
		require(!mkdir(path, 0700), "mkdir failed");
		break;
	case SWEEP_FIFO:
		require(!mkfifo(path, 0600), "mkfifo failed");
		break;
	}
	time_t const then = now - (time_t)c->hours * 3600;
	struct timespec const times[2] = { { .tv_sec = then }, { .tv_sec = then } };
	require(!utimensat(AT_FDCWD, path, times, AT_SYMLINK_NOFOLLOW), "cannot age %s", path);
	free(path);
	path = nullptr;
}

/** @brief Where sweep_open() opens a channel, which says what stays. */
enum sweep_at {
	SWEEP_AT_CHOSEN, //!< A chosen path: everything stays.
	SWEEP_AT_SHARED, //!< The shared default: what struct sweep_case's shared says.
	SWEEP_AT_NATIVE, //!< The native tools' default: what struct sweep_case's native says.
	SWEEP_AT_AGAIN,  //!< A default whose directory this process has swept: everything stays.
};

/** @brief Opens a channel in a directory of SWEEP_CASES' files and ends the test unless exactly the
 *         files that should go have gone.
 *
 * @param path The channel's path.
 * @param at   Where that is.
 */
static void
sweep_open (char const    *path,
            enum sweep_at  at)
{
	char *dir = strdup(path);
	require(dir, "out of memory");
	*strrchr(dir, '/') = '\0';
	time_t const now = time(nullptr);
	require(now != (time_t)-1, "time failed");
	char const *kept[sizeof SWEEP_CASES / sizeof *SWEEP_CASES + 1];
	uint32_t count = 0;
	// The files that the test locks and holds, by enum sweep_type from SWEEP_LOCKED.
	int fds[] = { -1, -1 };
	for (uint32_t i = 0; i < sizeof SWEEP_CASES / sizeof *SWEEP_CASES; ++i) {
		struct sweep_case const *const c = &SWEEP_CASES[i];
		sweep_make(dir, c, now);
		if (at == SWEEP_AT_SHARED ? c->shared : at != SWEEP_AT_NATIVE || c->native)
			kept[count++] = c->name;
		if (c->type != SWEEP_LOCKED && c->type != SWEEP_HELD)
			continue;
		char *name = support_format(nullptr, "%s/%s", dir, c->name);
		require(name, "out of memory");
		int *const fd = &fds[c->type - SWEEP_LOCKED];
		*fd = open(name, O_RDONLY | O_CLOEXEC);
		struct flock lock = { .l_type = F_RDLCK, .l_whence = SEEK_SET, .l_len = 1 };
		require(*fd >= 0 && (c->type == SWEEP_LOCKED ? !flock(*fd, LOCK_EX | LOCK_NB) : !fcntl(*fd, F_OFD_SETLK, &lock)),
		        "cannot lock %s", name);
		free(name);
		name = nullptr;
	}
	kept[count++] = kShmChannelName;

	struct shm_channel channel;
	open_channel(&channel, path, kHeaderBytes, SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE);
	require(channel.flags & SHM_CHANNEL_CREATED, "the sweep's channel was not created");
	shm_channel_fini(&channel);
	for (uint32_t i = 0; i < sizeof fds / sizeof *fds; ++i) {
		require(!close(fds[i]), "close failed");
		fds[i] = -1;
	}

	for (uint32_t i = 1; i < count; ++i)
		for (uint32_t j = i; j && strcmp(kept[j - 1], kept[j]) > 0; --j) {
			char const *const swap = kept[j];
			kept[j] = kept[j - 1];
			kept[j - 1] = swap;
		}
	struct support_text want = {};
	for (uint32_t i = 0; i < count; ++i)
		require(support_text_printf(&want, "%s%s", i ? " " : "", kept[i]), "out of memory");
	char *names = listing(dir);
	static char const *const AT[] = {
		[SWEEP_AT_CHOSEN] = "a chosen path",
		[SWEEP_AT_SHARED] = "the shared default",
		[SWEEP_AT_NATIVE] = "the native default",
		[SWEEP_AT_AGAIN]  = "a default swept before",
	};
	require(!strcmp(names, support_text_string(&want)), "opening %s at %s left\n  %s\nnot\n  %s", path,
	        AT[at], names, support_text_string(&want));
	free(names);
	names = nullptr;
	support_text_fini(&want);
	require(support_remove_tree(dir), "cannot remove %s", dir);
	free(dir);
	dir = nullptr;
}

/** @brief Writes a file of /proc.
 *
 * @param path The file.
 * @param text What to write.
 * @return     true if it was written whole.
 */
static bool
proc_write (char const *path,
            char const *text)
{
	int fd = open(path, O_WRONLY | O_CLOEXEC);
	if (fd < 0)
		return false;
	size_t const length = strlen(text);
	bool const written = write(fd, text, length) == (ssize_t)length;
	return !close(fd) && written;
}

/** @brief Gives this process a /tmp of its own: a tmpfs in a mount namespace of a user namespace
 *         in which the user keeps its IDs.
 *
 * @return true if it did; false if the namespaces are not available.
 */
static bool
private_tmp (void)
{
	unsigned const uid = (unsigned)getuid();
	unsigned const gid = (unsigned)getgid();
	if (unshare(CLONE_NEWUSER | CLONE_NEWNS))
		return false;
	char map[32];
	snprintf(map, sizeof map, "%u %u 1\n", uid, uid);
	if (!proc_write("/proc/self/setgroups", "deny\n") || !proc_write("/proc/self/uid_map", map))
		return false;
	snprintf(map, sizeof map, "%u %u 1\n", gid, gid);
	return proc_write("/proc/self/gid_map", map) && !mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr)
	       && !mount("tmpfs", "/tmp", "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777");
}

/** @brief A default's path, made with its function, in a directory made private for the sweep.
 *
 * @param format ShmDefaultPath() or ShmNativeDefaultPath().
 * @return       The path, which the caller frees.
 */
static char *
default_in_private (int (*format)(char *, size_t))
{
	int const length = format(nullptr, 0);
	require(length > 0, "cannot format a default path");
	char *path = malloc((size_t)length + 1);
	require(path && format(path, (size_t)length + 1) == length, "cannot format a default path");
	char *const slash = strrchr(path, '/');
	*slash = '\0';
	require(!mkdir(path, 0700), "cannot make %s", path);
	*slash = '/';
	return path;
}

/** @brief Opens channels in directories of SWEEP_CASES' files: the default channels in a private
 *         /tmp of the test's namespaces, the native one twice, a chosen path, and without namespaces
 *         the shared default under a DLSSNR_UID of the test's own in /tmp.
 *
 * @return What was swept, for the summary.
 */
static char const *
check_sweeps (void)
{
	char *chosen = path_in(nullptr, "chosen");
	require(!mkdir(chosen, 0700), "mkdir failed");
	char *chosen_path = support_format(nullptr, "%s/%s", chosen, kShmChannelName);
	require(chosen_path, "out of memory");
	sweep_open(chosen_path, SWEEP_AT_CHOSEN);
	free(chosen_path);
	chosen_path = nullptr;
	free(chosen);
	chosen = nullptr;

	// The defaults in a /tmp of the child's own, which the user's real channels are not in.
	require(!fflush(nullptr), "fflush failed");
	pid_t const child = fork();
	require(child >= 0, "fork failed");
	if (!child) {
		if (!private_tmp())
			_exit(77);
		require(!unsetenv("DLSSNR_UID"), "unsetenv failed");
		static struct {
			int         (*format)(char *, size_t);
			enum sweep_at at;
		} const opens[] = {
			{ ShmDefaultPath,       SWEEP_AT_SHARED },
			{ ShmNativeDefaultPath, SWEEP_AT_NATIVE },
			// A process sweeps a default's directory once.
			{ ShmNativeDefaultPath, SWEEP_AT_AGAIN  },
		};
		for (uint32_t i = 0; i < sizeof opens / sizeof *opens; ++i) {
			char *path = default_in_private(opens[i].format);
			sweep_open(path, opens[i].at);
			free(path);
			path = nullptr;
		}
		_exit(0);
	}
	int status;
	require(waitpid(child, &status, 0) == child, "waitpid failed");
	require(WIFEXITED(status) && (!WEXITSTATUS(status) || WEXITSTATUS(status) == 77),
	        "a sweep of a default channel failed");
	if (!WEXITSTATUS(status))
		return "both defaults in a private /tmp, once a process, and a chosen path";

	// No namespaces: the shared default in /tmp, in a directory that DLSSNR_UID names for the test.
	char *dir = support_temp_dir("/tmp", "dlssnr-shm-channel-test", nullptr);
	require(dir, "cannot make a temporary directory");
	require(!setenv("DLSSNR_UID", dir + sizeof "/tmp/dlssnr-" - 1, 1), "setenv failed");
	size_t length;
	char *shared = support_format(&length, "%s/%s", dir, kShmChannelName);
	require(shared, "out of memory");
	char buf[256];
	int const n = ShmDefaultPath(buf, sizeof buf);
	require(n >= 0 && (size_t)n == length && !memcmp(buf, shared, length),
	        "DLSSNR_UID did not name the test's directory");
	sweep_open(shared, SWEEP_AT_SHARED);
	free(shared);
	shared = nullptr;
	free(dir);
	dir = nullptr;
	require(!unsetenv("DLSSNR_UID"), "unsetenv failed");
	return "the shared default under DLSSNR_UID and a chosen path; no namespaces for the native one";
}

/** @brief Removes the test's directory. */
static void
remove_root (void)
{
	if (getpid() != owner)
		return;
	if (root && !support_remove_tree(root))
		fprintf(stderr, "shm-channel-test: cannot remove %s\n", root);
	free(root);
	root = nullptr;
}

int
main (void)
{
	owner = getpid();
	// Zeros, as a new file holds, until each atomic object's one initialization.
	defaults = mmap(nullptr, sizeof *defaults, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	require(defaults != MAP_FAILED, "mmap failed");
	ShmInitNativeDefaults(defaults, false);
	char const *const tmp = getenv("TMPDIR");
	root = support_temp_dir(tmp && *tmp ? tmp : "/tmp", "dlsslop-shm-channel", &root_length);
	require(root, "cannot make a temporary directory");
	require(!atexit(remove_root), "atexit failed");

	check_create();
	check_refusals();
	check_replaced();
	check_swept();
	check_missing();
	char const *const sweeps = check_sweeps();
	if (printf("shm channel: %u races of %u processes for each of %u ways of creating a channel, half of "
	           "them in a missing directory, refusals, a replaced directory, channels that a sweep removes "
	           "while they open, missing channels, and sweeps of %s\n", ROUNDS, RACERS, (unsigned)FORCE_COUNT,
	           sweeps) < 0)
		return 1;
	return 0;
}
