/** @file
 *
 * The channel's file, opened and created in this one place by every program that uses it.
 *
 * A process opens the file. If there is none, it creates the channel whole before it has a name:
 * an unnamed file (O_TMPFILE) in the channel's directory, sized, mapped and given the native
 * defaults, which linkat(2) then names. Linking fails if the name exists, so of processes that
 * create a channel at once exactly one links its file, and the others drop theirs and open that
 * one. A file that has a name is complete: nothing initializes it again, and a file that holds
 * another protocol, or no channel, is refused. Which process creates the channel makes no
 * difference.
 *
 * The directory must be a real directory that this user owns and that grants nothing to group or
 * others; a process that creates the channel gives it mode 0700. Opening a channel at a default
 * path (ShmNativeDefaultPath(), ShmDefaultPath()) also removes other protocols' channels that have
 * lain unused in that directory for a day, once a process. A process marks each channel that it
 * has open as used, with a read lock on the file's first byte that belongs to its open file
 * description (F_OFD_SETLK); a sweep removes a channel only under a write lock there. That lock,
 * and dlsslopd's flock(2) of the channel that it serves, tell every version that a channel is in
 * use. shm_channel.c defines the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_SHM_CHANNEL_H_
#define DLSSLOP_AMD_COMMON_SHM_CHANNEL_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
#else
# include <stddef.h>
# include <stdint.h>
#endif

#include "error.h"
#include "shm_protocol.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief What shm_channel_open() is asked to do, and what it did, in struct shm_channel's flags. */
enum shm_channel_flags : STD(uint32_t) {
	SHM_CHANNEL_WRITE   = 1 << 0, //!< Map the channel for writing.
	SHM_CHANNEL_CREATE  = 1 << 1, //!< Create the channel, and its directory, if it is missing.
	SHM_CHANNEL_CREATED = 1 << 2, //!< This call created the channel.
	SHM_CHANNEL_MISSING = 1 << 3, //!< There was no channel, and this call created none.
};

/** @brief A channel file, open and mapped: shm_channel_open() opens it and shm_channel_fini()
 *         closes it.
 */
struct shm_channel {
	struct ShmHeader *h;     //!< The mapping, which starts with the header; nullptr if none.
	STD(size_t)       bytes; //!< The mapping's size.
	int               fd;    //!< The channel file, which holds its read lock, or -1.
	STD(uint32_t)     flags; //!< enum shm_channel_flags: what was asked, and what happened.
};

/** @brief Opens a channel and maps its start, creating the channel if it is missing and asked to.
 *
 * Refuses a file that is not a regular file of this user's, a symbolic link, a file of another size
 * than ShmTotalBytes() and a file whose magic or version is not this protocol's, saying which
 * protocol it holds and what to do. Refuses a directory that is not a real one of this user's, or
 * that grants group or others anything. Creating makes the directory, mode 0700, and its missing
 * parents, as mkdir -p does.
 *
 * @param dest   Receives the channel. On a failure nothing is open or mapped, and its flags say
 *               SHM_CHANNEL_MISSING if there was no channel and @a flags did not ask to create one.
 * @param path   The channel's path.
 * @param length The length of its path.
 * @param bytes  How much of the channel to map, from its start: at least kHeaderBytes, at most
 *               ShmTotalBytes().
 * @param flags  SHM_CHANNEL_WRITE and SHM_CHANNEL_CREATE, which needs SHM_CHANNEL_WRITE.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
shm_channel_open (struct shm_channel *dest,
                  char const         *path,
                  STD(size_t)         length,
                  STD(size_t)         bytes,
                  STD(uint32_t)       flags,
                  struct error       *e);

/** @brief Unmaps a channel and closes it, leaving it empty.
 *
 * @param channel The channel, or nullptr.
 */
extern void
shm_channel_fini (struct shm_channel *channel);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_SHM_CHANNEL_H_ */
