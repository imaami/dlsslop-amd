/** @file
 *
 * What several of the C tests need: support.h.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "support.h"

char *
support_format (size_t     *length,
                char const *fmt,
                ...)
{
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(nullptr, 0, fmt, args);
	va_end(args);
	if (n < 0)
		return nullptr;

	size_t const size = (size_t)n + 1;
	char *text = malloc(size);
	if (!text)
		return nullptr;

	va_start(args, fmt);
	int const written = vsnprintf(text, size, fmt, args);
	va_end(args);
	if (written != n) {
		free(text);
		text = nullptr;
	} else if (length) {
		*length = size - 1;
	}
	return text;
}

char *
support_temp_dir (char const *parent,
                  char const *prefix)
{
	char *path = support_format(nullptr, "%s/%s-XXXXXX", parent, prefix);
	if (path && !mkdtemp(path)) {
		free(path);
		path = nullptr;
	}
	return path;
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
