/** @file
 *
 * Paths and whole files through POSIX calls: files.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "error.h"
#include "files.h"

/** @brief Puts strerror()'s words for an errno value in an error.
 *
 * @param e   The error, or nullptr.
 * @param err The errno value.
 * @return    ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
fail_errno (struct error *e,
            int           err)
{
	char buf[64];
	return error_fail(e, "%s", strerror_r(err, buf, sizeof buf));
}

/** @brief A copy of a string on the heap.
 *
 * @param s      The string; it need not be null-terminated.
 * @param length Its length.
 * @return       The copy, null-terminated, or nullptr without memory for it.
 */
static char *
copy (char const *s,
      size_t      length)
{
	char *const text = malloc(length + 1);
	if (text) {
		memcpy(text, s, length);
		text[length] = '\0';
	}
	return text;
}

/** @brief The type bits of a file's mode.
 *
 * @param path The file.
 * @return     Its S_IFMT bits, following symbolic links, or 0 if it cannot be found.
 */
static mode_t
file_type (char const *path)
{
	struct stat st;
	return stat(path, &st) ? 0 : st.st_mode & S_IFMT;
}

bool
files_is_directory (char const *path)
{
	return file_type(path) == S_IFDIR;
}

bool
files_is_regular_file (char const *path)
{
	return file_type(path) == S_IFREG;
}

size_t
files_parent (char const *path,
              size_t      length)
{
	// The length up to and with the last slash.
	while (length && path[length - 1] != '/')
		--length;
	return length > 1 ? length - 1 : length;
}

char *
files_join (char const *a,
            size_t      a_length,
            char const *b,
            size_t      b_length,
            size_t     *length)
{
	size_t const slash = a_length && a[a_length - 1] != '/';
	size_t const joined_length = a_length + slash + b_length;
	char *const joined = malloc(joined_length + 1);
	if (joined) {
		memcpy(joined, a, a_length);
		if (slash)
			joined[a_length] = '/';
		memcpy(joined + a_length + slash, b, b_length);
		joined[joined_length] = '\0';
		if (length)
			*length = joined_length;
	}
	return joined;
}

char *
files_absolute (char const *path,
                size_t      length,
                size_t     *absolute_length)
{
	char *cwd = length && *path == '/' ? nullptr : getcwd(nullptr, 0);
	if (!cwd) {
		// Absolute, or relative to a working directory that has no path.
		char *const same = copy(path, length);
		if (same && absolute_length)
			*absolute_length = length;
		return same;
	}

	char *const joined = files_join(cwd, strlen(cwd), path, length, absolute_length);
	free(cwd);
	cwd = nullptr;
	return joined;
}

/** @brief Reads a descriptor to its end.
 *
 * @param fd       The descriptor.
 * @param capacity The bytes to allocate first, the null included; more than 1.
 * @param dest     Receives the bytes; untouched on a failure.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
read_to_end (int                fd,
             size_t             capacity,
             struct files_data *dest,
             struct error      *e)
{
	uint8_t *bytes = malloc(capacity);
	if (!bytes)
		return error_fail(e, "out of memory");

	size_t size = 0;
	for (;;) {
		if (size == capacity - 1) {
			uint8_t *const grown = realloc(bytes, 2 * capacity);
			if (!grown) {
				free(bytes);
				bytes = nullptr;
				return error_fail(e, "out of memory");
			}
			bytes = grown;
			capacity *= 2;
		}
		ssize_t const got = read(fd, bytes + size, capacity - 1 - size);
		if (!got)
			break;
		if (got > 0) {
			size += (size_t)got;
		} else if (errno != EINTR) {
			int const err = errno;
			free(bytes);
			bytes = nullptr;
			return fail_errno(e, err);
		}
	}
	bytes[size] = '\0';
	*dest = (struct files_data){bytes, size};
	return ERROR_NONE;
}

enum error_code
files_read (struct files_data *dest,
            char const        *path,
            struct error      *e)
{
	*dest = (struct files_data){};
	int fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return fail_errno(e, errno);

	// Room for the size that fstat() gives, a byte more for the read that finds the end, and the
	// null; a file of no size, such as one in /proc, may still hold bytes.
	struct stat st;
	size_t const capacity = fstat(fd, &st) || st.st_size <= 0 ? 65536 : (size_t)st.st_size + 2;
	enum error_code const code = read_to_end(fd, capacity, dest, e);
	// Nothing was written to it: close() has nothing to report.
	close(fd);
	fd = -1;
	return code;
}

void
files_data_fini (struct files_data *data)
{
	if (data) {
		free(data->bytes);
		*data = (struct files_data){};
	}
}

enum error_code
files_read_all (int           fd,
                void         *data,
                size_t        size,
                struct error *e)
{
	for (uint8_t *at = data; size;) {
		ssize_t const got = read(fd, at, size);
		if (got > 0) {
			at += got;
			size -= (size_t)got;
		} else if (!got) {
			return error_fail(e, "unexpected end of file");
		} else if (errno != EINTR) {
			return fail_errno(e, errno);
		}
	}
	return ERROR_NONE;
}

enum error_code
files_read_at (int           fd,
               uint64_t      offset,
               void         *data,
               size_t        size,
               struct error *e)
{
	off_t position = (off_t)offset;
	for (uint8_t *at = data; size;) {
		ssize_t const got = pread(fd, at, size, position);
		if (got > 0) {
			at += got;
			position += got;
			size -= (size_t)got;
		} else if (!got) {
			return error_fail(e, "unexpected end of file");
		} else if (errno != EINTR) {
			return fail_errno(e, errno);
		}
	}
	return ERROR_NONE;
}

enum error_code
files_write_all (int           fd,
                 void const   *data,
                 size_t        size,
                 struct error *e)
{
	for (uint8_t const *at = data; size;) {
		ssize_t const wrote = write(fd, at, size);
		if (wrote > 0) {
			at += wrote;
			size -= (size_t)wrote;
		} else if (!wrote) {
			return fail_errno(e, EIO);
		} else if (errno != EINTR) {
			return fail_errno(e, errno);
		}
	}
	return ERROR_NONE;
}

enum error_code
files_write (char const   *path,
             void const   *data,
             size_t        size,
             struct error *e)
{
	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
	if (fd < 0)
		return fail_errno(e, errno);

	enum error_code code = files_write_all(fd, data, size, e);
	// A write can fail late, where only close() reports it.
	if (close(fd) && !code)
		code = fail_errno(e, errno);
	fd = -1;
	return code;
}

/** @brief Writes bytes as a file through a temporary file that rename(2) puts in its place.
 *
 * @param temporary The temporary file's mkostemp(3) template, which receives its name.
 * @param path      The file.
 * @param data      The bytes.
 * @param size      Their number.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED with the temporary file removed.
 */
static enum error_code
replace_through (char         *temporary,
                 char const   *path,
                 void const   *data,
                 size_t        size,
                 struct error *e)
{
	int fd = mkostemp(temporary, O_CLOEXEC);
	if (fd < 0)
		return fail_errno(e, errno);

	enum error_code code = files_write_all(fd, data, size, e);
	if (close(fd) && !code)
		code = fail_errno(e, errno);
	fd = -1;
	if (!code && rename(temporary, path))
		code = fail_errno(e, errno);
	// The failure is reported whether or not the temporary file goes.
	if (code)
		unlink(temporary);
	return code;
}

enum error_code
files_replace (char const   *path,
               size_t        length,
               void const   *data,
               size_t        size,
               struct error *e)
{
	static char const suffix[] = ".XXXXXX";
	char *temporary = malloc(length + sizeof suffix);
	if (!temporary)
		return error_fail(e, "out of memory");
	memcpy(temporary, path, length);
	memcpy(temporary + length, suffix, sizeof suffix);

	enum error_code const code = replace_through(temporary, path, data, size, e);
	free(temporary);
	temporary = nullptr;
	return code;
}

/** @brief Creates a directory and any missing parents.
 *
 * @param directory The directory, without a trailing slash; each parent is null-terminated in
 *                  place while it is created.
 * @param length    The length of its path.
 * @param created   Receives whether the directory itself was created; untouched on a failure.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_directories (char         *directory,
                  size_t        length,
                  bool         *created,
                  struct error *e)
{
	// Each parent, from the first slash after the first byte.
	for (size_t end = 1; end < length; ++end) {
		if (directory[end] != '/')
			continue;
		directory[end] = '\0';
		int const err = mkdir(directory, 0777) ? errno : 0;
		directory[end] = '/';
		if (err && err != EEXIST)
			return fail_errno(e, err);
	}

	int const err = mkdir(directory, 0777) ? errno : 0;
	if (err && err != EEXIST)
		return fail_errno(e, err);
	*created = !err;
	return ERROR_NONE;
}

/** @brief Creates a private directory, or accepts an existing one: files_private_directory().
 *
 * @param directory The directory, without a trailing slash.
 * @param length    The length of its path.
 * @param what      What the directory is, for the words.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_private (char         *directory,
              size_t        length,
              char const   *what,
              struct error *e)
{
	bool created;
	enum error_code const code = make_directories(directory, length, &created, e);
	if (code) {
		error_wrap(e, "create %s directory: ", what);
		return code;
	}

	struct stat st;
	if (lstat(directory, &st) || !S_ISDIR(st.st_mode) || st.st_uid != getuid())
		return error_fail(e, "%s directory must be owned by the current user and not a symlink", what);
	if (created ? chmod(directory, 0700) != 0 : (st.st_mode & 0777) != 0700)
		return error_fail(e, "%s directory must be private (mode 0700): %s", what, directory);
	return ERROR_NONE;
}

/** @brief A path without its trailing slashes, on the heap.
 *
 * @param path   The path.
 * @param length Its length; receives the copy's.
 * @return       The copy, or nullptr without memory for it.
 */
static char *
without_trailing_slashes (char const *path,
                          size_t     *length)
{
	size_t kept = *length;
	while (kept && path[kept - 1] == '/')
		--kept;
	char *const directory = copy(path, kept);
	if (directory)
		*length = kept;
	return directory;
}

enum error_code
files_make_directories (char const   *path,
                        size_t        length,
                        bool         *created,
                        struct error *e)
{
	char *directory = without_trailing_slashes(path, &length);
	if (!directory)
		return error_fail(e, "out of memory");

	enum error_code const code = make_directories(directory, length, created, e);
	free(directory);
	directory = nullptr;
	return code;
}

enum error_code
files_private_directory (char const   *path,
                         size_t        length,
                         char const   *what,
                         struct error *e)
{
	char *directory = without_trailing_slashes(path, &length);
	if (!directory)
		return error_fail(e, "out of memory");

	enum error_code const code = make_private(directory, length, what, e);
	free(directory);
	directory = nullptr;
	return code;
}
