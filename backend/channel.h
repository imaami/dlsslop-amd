/** @file
 *
 * The daemon's side of the shared-memory channel: the header and the two frame slots, mapped from a
 * private file that one daemon locks at a time. channel.c defines the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_CHANNEL_H_
#define DLSSLOP_AMD_BACKEND_CHANNEL_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
#else
# include <stddef.h>
# include <stdint.h>
#endif

#include "error.h"
#include "shm_channel.h"
#include "shm_protocol.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief The channel as this daemon maps it: mapping_init() maps it and mapping_fini() unmaps it. */
struct mapping {
	struct shm_channel channel; //!< The channel file, locked, mapped whole.
	STD(uint8_t)      *input;   //!< The frame slot of the layer's proxy.
	STD(uint8_t)      *output;  //!< The frame slot of the answer.
};

/** @brief Maps the channel file and takes the channel over.
 *
 * shm_channel_open() opens the channel, or creates it if it is missing; it refuses a channel of
 * another protocol. A request left by a previous daemon is answered as failed, so that the layer
 * presents its own frame, and helperState reads Starting.
 *
 * @param dest   Receives the mapping; nothing is mapped on a failure.
 * @param name   The channel file's path.
 * @param length The length of its path.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
mapping_init (struct mapping *dest,
              char const     *name,
              STD(size_t)     length,
              struct error   *e);

/** @brief Leaves the channel: modelUp reads 0 and helperState Stopped, unless it reads ModelFailed;
 *         then unmaps the file and unlocks it.
 *
 * @param mapping The mapping, or nullptr.
 */
extern void
mapping_fini (struct mapping *mapping);

/** @brief Publishes the daemon's status text, which the layer and the GUI show.
 *
 * @param mapping The mapping.
 * @param text    The text; ShmStoreString() cuts it to fit.
 */
extern void
mapping_reason (struct mapping const *mapping,
                char const           *text);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_BACKEND_CHANNEL_H_ */
