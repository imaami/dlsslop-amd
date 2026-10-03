/** @file
 *
 * What several of the C tests need: a private temporary directory, a file's whole contents, and a
 * formatted string on the heap.
 */
#ifndef DLSSLOP_AMD_TESTS_SUPPORT_H_
#define DLSSLOP_AMD_TESTS_SUPPORT_H_

#include <stddef.h>

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
 * @return       Its path, which the caller frees, or nullptr if it cannot be made.
 */
extern char *
support_temp_dir (char const *parent,
                  char const *prefix);

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
