/** @file
 *
 * What several of the C tests need: a private temporary directory, removed with everything in it,
 * a file's whole contents, a formatted string on the heap, text that grows as it is written, a
 * program run with its output in a file, and a pause.
 */
#ifndef DLSSLOP_AMD_TESTS_SUPPORT_H_
#define DLSSLOP_AMD_TESTS_SUPPORT_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
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

#endif /* DLSSLOP_AMD_TESTS_SUPPORT_H_ */
