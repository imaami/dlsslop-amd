/** @file
 *
 * What several of the C tests need: support.h.
 */
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "support.h"

char *
support_vformat (size_t     *length,
                 char const *fmt,
                 va_list     args)
{
	va_list measured;
	va_copy(measured, args);
	int const n = vsnprintf(nullptr, 0, fmt, measured);
	va_end(measured);
	if (n < 0)
		return nullptr;

	size_t const size = (size_t)n + 1;
	char *text = malloc(size);
	if (!text)
		return nullptr;

	if (vsnprintf(text, size, fmt, args) != n) {
		free(text);
		text = nullptr;
	} else if (length) {
		*length = size - 1;
	}
	return text;
}

char *
support_format (size_t     *length,
                char const *fmt,
                ...)
{
	va_list args;
	va_start(args, fmt);
	char *const text = support_vformat(length, fmt, args);
	va_end(args);
	return text;
}

char *
support_temp_dir (char const *parent,
                  char const *prefix,
                  size_t     *length)
{
	// mkdtemp() replaces the Xs in place: the length stays the formatted one.
	char *path = support_format(length, "%s/%s-XXXXXX", parent, prefix);
	if (path && !mkdtemp(path)) {
		free(path);
		path = nullptr;
	}
	return path;
}

/** @brief Removes what nftw() reports, a directory after what it holds.
 *
 * @param path The entry.
 * @param st   Its status; unused.
 * @param type What nftw() found; unused.
 * @param ftw  Where it is; unused.
 * @return     0 to go on, -1 to stop the walk.
 */
static int
remove_entry (char const        *path,
              struct stat const *st,
              int                type,
              struct FTW        *ftw)
{
	return remove(path) ? -1 : 0;
}

bool
support_remove_tree (char const *path)
{
	return !nftw(path, remove_entry, 16, FTW_DEPTH | FTW_PHYS);
}

bool
support_next_entry (DIR                  *dir,
                    struct dirent const **entry)
{
	// readdir() tells a failure from the end only through errno.
	errno = 0;
	struct dirent const *const next = readdir(dir);
	*entry = next;
	return next || !errno;
}

char *
support_read_file (char const *path,
                   size_t     *length)
{
	FILE *const file = fopen(path, "rbe");
	if (!file)
		return nullptr;

	size_t size = 4096;
	size_t used = 0;
	char *text = malloc(size);
	while (text) {
		// Room for the terminating null stays free.
		used += fread(text + used, 1, size - 1 - used, file);
		if (used < size - 1)
			break;
		char *const grown = realloc(text, 2 * size);
		if (!grown) {
			free(text);
			text = nullptr;
			break;
		}
		text = grown;
		size *= 2;
	}

	bool const failed = ferror(file);
	if (fclose(file) || failed) {
		free(text);
		text = nullptr;
	}
	if (text) {
		text[used] = '\0';
		if (length)
			*length = used;
	}
	return text;
}

bool
support_text_vprintf (struct support_text *text,
                      char const          *fmt,
                      va_list              args)
{
	va_list measured;
	va_copy(measured, args);
	size_t const room = text->capacity - text->length;
	int const n = vsnprintf(text->bytes ? text->bytes + text->length : nullptr, room, fmt, measured);
	va_end(measured);
	if (n < 0)
		return false;
	if ((size_t)n < room) {
		text->length += (size_t)n;
		return true;
	}

	// Too long for the room left: grown, and formatted again.
	size_t const need = text->length + (size_t)n + 1;
	size_t capacity = text->capacity ? 2 * text->capacity : 256;
	while (capacity < need)
		capacity *= 2;
	char *const grown = realloc(text->bytes, capacity);
	if (!grown) {
		if (text->bytes)
			text->bytes[text->length] = '\0';
		return false;
	}
	text->bytes = grown;
	text->capacity = capacity;
	if (vsnprintf(grown + text->length, capacity - text->length, fmt, args) != n) {
		grown[text->length] = '\0';
		return false;
	}
	text->length += (size_t)n;
	return true;
}

bool
support_text_printf (struct support_text *text,
                     char const          *fmt,
                     ...)
{
	va_list args;
	va_start(args, fmt);
	bool const ok = support_text_vprintf(text, fmt, args);
	va_end(args);
	return ok;
}

char const *
support_text_string (struct support_text const *text)
{
	return text->bytes ? text->bytes : "";
}

bool
support_text_equal (struct support_text const *a,
                    struct support_text const *b)
{
	return a->length == b->length && (!a->length || !memcmp(a->bytes, b->bytes, a->length));
}

void
support_text_fini (struct support_text *text)
{
	if (text) {
		free(text->bytes);
		text->bytes = nullptr;
		*text = (struct support_text){};
	}
}

pid_t
support_spawn (char const        *log,
               char const *const  argv[])
{
	pid_t const child = fork();
	if (child)
		return child;
	// Only async-signal-safe calls from here: the parent may have threads. The file becomes
	// descriptors 1 and 2, and a descriptor of its own, if it got one, goes.
	int const fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (fd < 0 || dup2(fd, 1) < 0 || dup2(fd, 2) < 0 || (fd > 2 && close(fd)))
		_exit(127);
	// execv() declares char *const[] for older code; it changes neither the array nor the strings.
	execv(argv[0], (char *const *)argv);
	_exit(127);
}

void
support_sleep_ms (uint32_t ms)
{
	struct timespec wait = { .tv_sec = ms / 1000, .tv_nsec = ms % 1000 * 1000000 };
	// nanosleep() fails otherwise only for a time that this cannot be.
	while (nanosleep(&wait, &wait) && errno == EINTR) {}
}

/** @brief Reads a decimal number of 32 bits at the start of a text: digits only.
 *
 * @param text The text.
 * @param end  Receives where the number ends.
 * @param dest Receives the number; untouched on a failure.
 * @return     true if the text starts with such a number.
 */
static bool
number32 (char const  *text,
          char       **end,
          uint32_t    *dest)
{
	if (*text < '0' || *text > '9')
		return false;
	// strtoul() reports a value out of its range only through errno.
	errno = 0;
	unsigned long const n = strtoul(text, end, 10);
	if (errno || n > UINT32_MAX)
		return false;
	*dest = (uint32_t)n;
	return true;
}

bool
support_extent (char const *text,
                uint32_t   *width,
                uint32_t   *height)
{
	char *end;
	uint32_t w;
	uint32_t h;
	if (!number32(text, &end, &w) || *end != 'x' || !number32(end + 1, &end, &h) || *end)
		return false;
	*width = w;
	*height = h;
	return true;
}

int
support_written (char const *program,
                 int         status)
{
	if (!fflush(stdout) && !ferror(stdout))
		return status;
	fprintf(stderr, "%s: cannot write the output\n", program);
	return 1;
}
