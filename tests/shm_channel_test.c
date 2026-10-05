/** @file
 *
 * shm_channel_open() in C, in a private directory. Processes that open a missing channel at once,
 * released together, all map the same file, which exactly one of them created and which holds the
 * native defaults; nothing else is left in the directory. That holds for each way of creating it:
 * an unnamed file that linkat(2) names by its descriptor or by its link in /proc, and a named
 * temporary file where the directory holds no unnamed files or no link names one; and it holds
 * when the racers also make the channel's directory under a umask that narrows its mode. The test
 * forces the other ways by defining openat() and linkat() itself, which the module's calls then
 * reach. A file of another protocol, size or type, a symbolic link and a directory that is not
 * private are refused and left as they were, and a directory replaced while a channel is created
 * receives nothing; an open that may not create finds nothing and creates nothing.
 */
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
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

/** @brief The processes that open a missing channel at once. */
static constexpr uint32_t RACERS = 32;

/** @brief The races for each way of creating a channel. */
static constexpr uint32_t ROUNDS = 8;

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
 *         and fails them as enum force says.
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
	return (int)syscall(SYS_openat, dir, path, flags, mode);
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
	for (struct dirent const *entry; (entry = readdir(entries));) {
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

/** @brief Removes the test's directory. */
static void
remove_root (void)
{
	if (root && !support_remove_tree(root))
		fprintf(stderr, "shm-channel-test: cannot remove %s\n", root);
	free(root);
	root = nullptr;
}

int
main (void)
{
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
	check_missing();
	if (printf("shm channel: %u races of %u processes for each of %u ways of creating a channel, half of "
	           "them in a missing directory, refusals, a replaced directory and missing channels\n", ROUNDS,
	           RACERS, (unsigned)FORCE_COUNT) < 0)
		return 1;
	return 0;
}
