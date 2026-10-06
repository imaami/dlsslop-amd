/** @file
 *
 * What several of the C tests need: a private temporary directory, removed with everything in it,
 * a file's whole contents, a formatted string on the heap, text that grows as it is written, a
 * program run with its output in a file, a pause, an extent read from a command line, and the
 * search of a sorted key array.
 */
#ifndef DLSSLOP_AMD_TESTS_SUPPORT_H_
#define DLSSLOP_AMD_TESTS_SUPPORT_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

/** @brief Formats a string on the heap, measured once.
 *
 * @param length Receives the string's length, or nullptr.
 * @param fmt    A printf format.
 * @param args   The format's arguments.
 * @return       The string, which the caller frees, or nullptr if it cannot be formatted.
 */
[[gnu::format(printf, 2, 0)]]
extern char *
support_vformat (size_t     *length,
                 char const *fmt,
                 va_list     args);

/** @brief Formats a string on the heap, measured once.
 *
 * @param length Receives the string's length, or nullptr.
 * @param fmt    A printf format.
 * @param ...    The format's arguments.
 * @return       The string, which the caller frees, or nullptr if it cannot be formatted.
 */
[[gnu::format(printf, 2, 3)]]
extern char *
support_format (size_t     *length,
                char const *fmt,
                ...);

/** @brief Makes a private directory, PARENT/PREFIX-XXXXXX with the Xs replaced, as mkdtemp() does.
 *
 * @param parent The directory to make it in.
 * @param prefix The start of its name.
 * @param length Receives its path's length, or nullptr.
 * @return       Its path, which the caller frees, or nullptr if it cannot be made.
 */
extern char *
support_temp_dir (char const *parent,
                  char const *prefix,
                  size_t     *length);

/** @brief Removes a directory and everything in it, without following symbolic links.
 *
 * @param path The directory.
 * @return     true if everything was removed.
 */
extern bool
support_remove_tree (char const *path);

/** @brief Reads a whole file.
 *
 * @param path   The file.
 * @param length Receives the file's length, or nullptr.
 * @return       The file's bytes and a terminating null, which the caller frees, or nullptr if the
 *               file cannot be read.
 */
extern char *
support_read_file (char const *path,
                   size_t     *length);

/** @brief Text that grows as it is written: a log to compare whole. A zeroed one is empty. */
struct support_text {
	char  *bytes;    //!< The text and a null, or nullptr while nothing was written.
	size_t length;   //!< The text's length.
	size_t capacity; //!< The bytes allocated.
};

/** @brief Appends formatted words to a text.
 *
 * @param text The text.
 * @param fmt  A printf format.
 * @param args The format's arguments.
 * @return     false if they cannot be formatted or without memory for them; the text is then as
 *             it was.
 */
[[gnu::format(printf, 2, 0)]]
extern bool
support_text_vprintf (struct support_text *text,
                      char const          *fmt,
                      va_list              args);

/** @brief Appends formatted words to a text.
 *
 * @param text The text.
 * @param fmt  A printf format.
 * @param ...  The format's arguments.
 * @return     false if they cannot be formatted or without memory for them; the text is then as
 *             it was.
 */
[[gnu::format(printf, 2, 3)]]
extern bool
support_text_printf (struct support_text *text,
                     char const          *fmt,
                     ...);

/** @brief A text's words, as a string.
 *
 * @param text The text.
 * @return     Its bytes, or "" while it is empty.
 */
extern char const *
support_text_string (struct support_text const *text);

/** @brief Whether two texts hold the same words.
 *
 * @param a A text.
 * @param b Another.
 * @return  true if they do.
 */
extern bool
support_text_equal (struct support_text const *a,
                    struct support_text const *b);

/** @brief Frees a text and empties it.
 *
 * @param text The text, or nullptr.
 */
extern void
support_text_fini (struct support_text *text);

/** @brief Starts a program in a child process, its standard output and error in a file.
 *
 * The child opens the file, created or emptied with mode 0600, as both, then runs the program with
 * execv(). Only async-signal-safe calls run in the child.
 *
 * @param log  The file.
 * @param argv The program's path and its arguments, ending in nullptr.
 * @return     The child's process ID, or -1 if fork() failed. A child that cannot open the file or
 *             run the program exits with status 127.
 */
extern pid_t
support_spawn (char const        *log,
               char const *const  argv[]);

/** @brief Sleeps, also through signals that interrupt it.
 *
 * @param ms The milliseconds to sleep.
 */
extern void
support_sleep_ms (uint32_t ms);

/** @brief Reads an extent, WxH: two decimal numbers of 32 bits, digits only.
 *
 * @param text   The extent.
 * @param width  Receives its width; untouched on a failure.
 * @param height Receives its height; untouched on a failure.
 * @return       true if the text is such an extent; false for anything else, a number out of
 *               range included.
 */
extern bool
support_extent (char const *text,
                uint32_t   *width,
                uint32_t   *height);

/** @brief The index of the first key above a key in a sorted array: the key before it, if any, is
 *         the greatest that is not above the key.
 *
 * @param keys  The keys, ascending.
 * @param count How many.
 * @param key   The key.
 * @return      The index of the first greater key, or @a count.
 */
static inline size_t
support_keys_upper_bound (uint64_t const *keys,
                          size_t          count,
                          uint64_t        key)
{
	size_t low = 0;
	while (count) {
		size_t const half = count / 2;
		if (key < keys[low + half]) {
			count = half;
		} else {
			low += half + 1;
			count -= half + 1;
		}
	}
	return low;
}

/** @brief Where a key is in a sorted array, or where it goes.
 *
 * @param keys  The keys, ascending.
 * @param count How many.
 * @param key   The key.
 * @param found Receives whether the key is there.
 * @return      Its index, or the index it goes to.
 */
static inline size_t
support_keys_find (uint64_t const *keys,
                   size_t          count,
                   uint64_t        key,
                   bool           *found)
{
	size_t const at = support_keys_upper_bound(keys, count, key);
	bool const there = at && keys[at - 1] == key;
	*found = there;
	return there ? at - 1 : at;
}

/** @brief The keys and values that a SUPPORT_SORTED_MAP's first growth makes room for. */
static constexpr size_t SUPPORT_MAP_MINIMUM = 16;

/** @brief Defines struct NAME, a map from 64-bit keys to values of TYPE that is empty when zeroed: its
 *         keys ascending, each value at its key's index. The functions that come with it:
 *         - NAME_find(), a key's value, or nullptr;
 *         - NAME_slot(), a key's value, added uninitialized if the key is new;
 *         - NAME_insert(), which makes room for a key at the index that support_keys_find() gave and
 *           moves the later ones;
 *         - NAME_erase(), which removes the key and value at an index and moves the later ones;
 *         - NAME_remove(), which removes a key and its value, if the map has them.
 *         A value's address holds until the next insert or removal. GROW(pointer, bytes) is the
 *         including file's realloc(), which ends the process rather than return nullptr.
 */
#define SUPPORT_SORTED_MAP(name, type, grow) \
	struct name { \
		uint64_t *keys;     /* Ascending. */ \
		type     *values;   /* Each key's value, at its index. */ \
		size_t    count;    /* The keys. */ \
		size_t    capacity; /* The keys and values allocated. */ \
	}; \
	\
	[[maybe_unused]] static type * \
	name##_find (struct name const *map, uint64_t key) \
	{ \
		bool found; \
		size_t const at = support_keys_find(map->keys, map->count, key, &found); \
		return found ? &map->values[at] : nullptr; \
	} \
	\
	[[maybe_unused]] static type * \
	name##_insert (struct name *map, size_t at, uint64_t key) \
	{ \
		if (map->count == map->capacity) { \
			size_t const capacity = map->capacity ? map->capacity * 2 : SUPPORT_MAP_MINIMUM; \
			map->keys = grow(map->keys, capacity * sizeof *map->keys); \
			map->values = grow(map->values, capacity * sizeof *map->values); \
			map->capacity = capacity; \
		} \
		memmove(&map->keys[at + 1], &map->keys[at], (map->count - at) * sizeof *map->keys); \
		memmove(&map->values[at + 1], &map->values[at], (map->count - at) * sizeof *map->values); \
		++map->count; \
		map->keys[at] = key; \
		return &map->values[at]; \
	} \
	\
	[[maybe_unused]] static type * \
	name##_slot (struct name *map, uint64_t key, bool *added) \
	{ \
		bool found; \
		size_t const at = support_keys_find(map->keys, map->count, key, &found); \
		*added = !found; \
		return found ? &map->values[at] : name##_insert(map, at, key); \
	} \
	\
	[[maybe_unused]] static void \
	name##_erase (struct name *map, size_t at) \
	{ \
		--map->count; \
		memmove(&map->keys[at], &map->keys[at + 1], (map->count - at) * sizeof *map->keys); \
		memmove(&map->values[at], &map->values[at + 1], (map->count - at) * sizeof *map->values); \
	} \
	\
	[[maybe_unused]] static bool \
	name##_remove (struct name *map, uint64_t key) \
	{ \
		bool found; \
		size_t const at = support_keys_find(map->keys, map->count, key, &found); \
		if (found) \
			name##_erase(map, at); \
		return found; \
	}

#endif /* DLSSLOP_AMD_TESTS_SUPPORT_H_ */
