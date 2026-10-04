/** @file
 *
 * What several of the C tests need: a private temporary directory, removed with everything in it,
 * a file's whole contents, and a formatted string on the heap.
 */
#ifndef DLSSLOP_AMD_TESTS_SUPPORT_H_
#define DLSSLOP_AMD_TESTS_SUPPORT_H_

#include <stdarg.h>
#include <stddef.h>

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

#endif /* DLSSLOP_AMD_TESTS_SUPPORT_H_ */
