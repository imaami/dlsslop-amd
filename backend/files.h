/** @file
 *
 * Paths and whole files through POSIX calls. Paths are null-terminated; a function that builds on a
 * path or a part of one also takes its length, and a path it makes is a heap string, which the
 * caller frees. files.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_FILES_H_
#define DLSSLOP_AMD_BACKEND_FILES_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"

/** @brief A whole file's bytes, which files_data_fini() frees. */
struct files_data {
	uint8_t *bytes; //!< The bytes and a null byte after them, or nullptr.
	size_t   size;  //!< The number of bytes, without the null.
};

/** @brief Whether a path names a directory, following symbolic links.
 *
 * @param path The path.
 * @return     true if it does.
 */
extern bool
files_is_directory (char const *path);

/** @brief Whether a path names a regular file, following symbolic links.
 *
 * @param path The path.
 * @return     true if it does.
 */
extern bool
files_is_regular_file (char const *path);

/** @brief Where a path's last component starts, less its slash.
 *
 * @param path   The path; it need not be null-terminated.
 * @param length Its length.
 * @return       The length of the path without its last component: 0 for a bare name, 1 for a
 *               name at the root.
 */
extern size_t
files_parent (char const *path,
              size_t      length);

/** @brief Joins two paths by one slash; the second alone when the first is empty.
 *
 * @param a        The first path; it need not be null-terminated.
 * @param a_length Its length.
 * @param b        The second path; it need not be null-terminated.
 * @param b_length Its length.
 * @param length   Receives the joined path's length, or nullptr.
 * @return         The joined path, or nullptr without memory for it.
 */
extern char *
files_join (char const *a,
            size_t      a_length,
            char const *b,
            size_t      b_length,
            size_t     *length);

/** @brief A path, relative to the working directory unless it is absolute.
 *
 * @param path            The path.
 * @param length          Its length.
 * @param absolute_length Receives the result's length, or nullptr.
 * @return                The working directory and @a path joined, or a copy of @a path if it is
 *                        absolute or the working directory has no path; nullptr without memory
 *                        for it.
 */
extern char *
files_absolute (char const *path,
                size_t      length,
                size_t     *absolute_length);

/** @brief Reads a whole file.
 *
 * @param dest Receives the file's bytes, or nothing ({0}) on a failure.
 * @param path The file.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_read (struct files_data *dest,
            char const        *path,
            struct error      *e);

/** @brief Frees a file's bytes and empties them.
 *
 * @param data The bytes, or nullptr.
 */
extern void
files_data_fini (struct files_data *data);

/** @brief Reads bytes from a descriptor: a file that ends first is an error.
 *
 * @param fd   The descriptor.
 * @param data Receives the bytes.
 * @param size Their number.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_read_all (int           fd,
                void         *data,
                size_t        size,
                struct error *e);

/** @brief Reads bytes from a descriptor at an offset, leaving its position as it was: a file that
 *         ends first is an error.
 *
 * @param fd     The descriptor.
 * @param offset Where in the file the bytes start.
 * @param data   Receives the bytes.
 * @param size   Their number.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_read_at (int           fd,
               uint64_t      offset,
               void         *data,
               size_t        size,
               struct error *e);

/** @brief Writes bytes to a descriptor, all of them.
 *
 * @param fd   The descriptor.
 * @param data The bytes.
 * @param size Their number.
 * @param e    Receives strerror()'s words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_write_all (int           fd,
                 void const   *data,
                 size_t        size,
                 struct error *e);

/** @brief Writes bytes as a file, created or truncated.
 *
 * @param path The file.
 * @param data The bytes.
 * @param size Their number.
 * @param e    Receives strerror()'s words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_write (char const   *path,
             void const   *data,
             size_t        size,
             struct error *e);

/** @brief Writes bytes as a file through a new file beside it, which rename(2) then puts in the
 *         file's place.
 *
 * A reader finds the old file or the new one, whole, and concurrent writers each write a file of
 * their own. The new file has mode 0600, as mkstemp(3) creates it. A failure leaves the file as it
 * was.
 *
 * @param path   The file.
 * @param length The length of its path.
 * @param data   The bytes.
 * @param size   Their number.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_replace (char const   *path,
               size_t        length,
               void const   *data,
               size_t        size,
               struct error *e);

/** @brief Creates a directory and any missing parents.
 *
 * @param path    The directory.
 * @param length  The length of its path.
 * @param created Receives whether the directory itself was created; untouched on a failure.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_make_directories (char const   *path,
                        size_t        length,
                        bool         *created,
                        struct error *e);

/** @brief Creates a directory, with any missing parents, as 0700, or accepts an existing real
 *         directory that the current user owns with exactly that mode, so that no other user can
 *         plant or swap files in it.
 *
 * @param path   The directory.
 * @param length The length of its path.
 * @param what   What the directory is, for the words.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
files_private_directory (char const   *path,
                         size_t        length,
                         char const   *what,
                         struct error *e);

#endif /* DLSSLOP_AMD_BACKEND_FILES_H_ */
