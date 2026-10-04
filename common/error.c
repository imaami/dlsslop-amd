/** @file
 *
 * The words of errors: error.h.
 */
// SPDX-License-Identifier: MIT
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "error.h"

/** @brief What an error says whose words vsnprintf() cannot format. */
static char const unformatted[] = "an error whose words cannot be formatted";

/** @brief Ends an error's words in "..." to show that they were cut.
 *
 * @param e The error, whose words fill it.
 */
static void
cut (struct error *e)
{
	memcpy(e->what + ERROR_WHAT_BYTES - sizeof "...", "...", sizeof "...");
}

/** @brief Formats an error's words, cut to fit.
 *
 * @param e    The error.
 * @param fmt  A printf format.
 * @param args The format's arguments.
 */
[[gnu::format(printf, 2, 0)]]
static void
vformat (struct error *e,
         char const   *fmt,
         va_list       args)
{
	int const n = vsnprintf(e->what, sizeof e->what, fmt, args);
	if (n < 0)
		memcpy(e->what, unformatted, sizeof unformatted);
	else if (n >= ERROR_WHAT_BYTES)
		cut(e);
}

enum error_code
error_fail (struct error *e,
            char const   *fmt,
            ...)
{
	if (e) {
		va_list args;
		va_start(args, fmt);
		vformat(e, fmt, args);
		va_end(args);
	}
	return ERROR_FAILED;
}

enum error_code
error_reject (struct error *e,
              char const   *fmt,
              ...)
{
	if (e) {
		va_list args;
		va_start(args, fmt);
		vformat(e, fmt, args);
		va_end(args);
	}
	return ERROR_REJECTED;
}

enum error_code
error_drop (struct error *e,
            char const   *fmt,
            ...)
{
	if (e) {
		va_list args;
		va_start(args, fmt);
		vformat(e, fmt, args);
		va_end(args);
	}
	return ERROR_DROPPED;
}

void
error_wrap (struct error *e,
            char const   *fmt,
            ...)
{
	if (!e)
		return;

	// The prefix is formatted apart: formatting the words into themselves would overlap.
	char prefix[ERROR_WHAT_BYTES];
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(prefix, sizeof prefix, fmt, args);
	va_end(args);
	if (n < 0)
		return;

	// The prefix as far as it fits, then as much of the words as fits behind it.
	size_t const length = n < ERROR_WHAT_BYTES ? (size_t)n : ERROR_WHAT_BYTES - 1;
	size_t const words = strlen(e->what);
	size_t const room = ERROR_WHAT_BYTES - 1 - length;
	size_t const kept = words < room ? words : room;
	memmove(e->what + length, e->what, kept);
	memcpy(e->what, prefix, length);
	e->what[length + kept] = '\0';
	if (kept < words || n >= ERROR_WHAT_BYTES)
		cut(e);
}
