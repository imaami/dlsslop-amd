/** @file
 *
 * The network on Vulkan: DLSSNR-AMD's network (external/vulkan/linux/) on a device of the daemon's
 * own. The network is opaque: this header names no recorder, runtime or plan type. vulkan_network.c
 * defines the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_VULKAN_NETWORK_H_
#define DLSSLOP_AMD_BACKEND_VULKAN_NETWORK_H_

#ifdef __cplusplus
# include <cstdint>
#else
# include <stdint.h>
#endif

#include "error.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

struct ShmTransportOffer;
struct vulkan_frame;
struct vulkan_paths;

/** @brief The network on a Vulkan device of its own: vulkan_network_create() makes one and
 *         vulkan_network_destroy() frees it. */
struct vulkan_network;

/** @brief The most passes a frame chains on the network (NETWORK_RECORDER_MAX_PASSES); more count
 *         as this many. */
#define VULKAN_NETWORK_MAX_PASSES UINT32_C(16)

/** @brief The import slots, each of which holds a proxy/answer pair that the layer exported. */
#define VULKAN_NETWORK_IMPORT_SLOTS 4

/** @brief Why vulkan_network_infer() drops a frame (ERROR_DROPPED): a wait of the network ran out
 *         while other GPU work held the device. The network answered the frame with its input and
 *         logs that itself, at most every 10 s. Serving answers it as failed, so that the layer shows
 *         the game's own frame. */
#define VULKAN_NETWORK_DROPPED "a wait of the network ran out while other GPU work held the device"

/** @brief The GPU time of a frame's parts, in milliseconds. */
struct vulkan_network_times {
	float upload_ms;    //!< From the frame's start until it is in the network's input.
	float inference_ms; //!< The network.
	float readback_ms;  //!< From the network's end until the answer is out.
};

/** @brief Makes the network on a Vulkan device of its own, with no network built.
 *
 * Its instance keeps implicit layers out unless the environment says otherwise.
 *
 * @param dest   Receives the network; nullptr on a failure.
 * @param paths  Where the network's files are; the network copies them.
 * @param device The index of the physical device to run on, or -1 for the first that can run the
 *               network.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_network_create (struct vulkan_network     **dest,
                       struct vulkan_paths const  *paths,
                       int                         device,
                       struct error               *e);

/** @brief Waits for the network's device, frees the network and sets the caller's pointer to
 *         nullptr.
 *
 * @param p_n The caller's pointer to the network, or nullptr; the network may be nullptr.
 */
extern void
vulkan_network_destroy (struct vulkan_network **p_n);

/** @brief The name of the network's physical device.
 *
 * @param n The network, or nullptr.
 * @return  The name, or nullptr for no network.
 */
extern char const *
vulkan_network_device_name (struct vulkan_network const *n);

/** @brief The index of the network's physical device, as --device names it.
 *
 * @param n The network, or nullptr.
 * @return  The index, or UINT32_MAX for no network.
 */
extern STD(uint32_t)
vulkan_network_device_index (struct vulkan_network const *n);

/** @brief Whether vulkan_network_shape() would build for a frame.
 *
 * @param n     The network.
 * @param frame The frame.
 * @return      true if it would.
 */
extern bool
vulkan_network_shape_differs (struct vulkan_network const *n,
                              struct vulkan_frame const   *frame);

/** @brief Builds the network for a frame's shape now, rather than on its first request, unless it
 *         has that shape: seconds of work for a new extent, all of it while the caller still reports
 *         itself starting.
 *
 * @param n     The network.
 * @param frame The frame.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE; ERROR_REJECTED for an extent the network does not take; ERROR_FAILED.
 */
extern enum error_code
vulkan_network_shape (struct vulkan_network     *n,
                      struct vulkan_frame const *frame,
                      struct error              *e);

/** @brief Rejects a frame of an extent the network does not take, before vulkan_network_shape():
 *         milliseconds of work for a new extent, none after.
 *
 * @param n     The network.
 * @param frame The frame.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE; ERROR_REJECTED for an extent the network does not take; ERROR_FAILED.
 */
extern enum error_code
vulkan_network_plan (struct vulkan_network     *n,
                     struct vulkan_frame const *frame,
                     struct error              *e);

/** @brief Imports an offered proxy/answer pair into a slot, releasing the pair it held once the new
 *         one is in, and declines one of another device or driver.
 *
 * @param n     The network.
 * @param slot  The slot, below VULKAN_NETWORK_IMPORT_SLOTS.
 * @param offer The offer.
 * @param fds   The offered proxy and answer memory. The network takes ownership of each descriptor
 *              it imports and sets it to -1; the caller closes the rest.
 * @return      true if the pair is in the slot.
 */
extern bool
vulkan_network_import (struct vulkan_network          *n,
                       STD(uint32_t)                   slot,
                       struct ShmTransportOffer const *offer,
                       int                             fds[2]);

/** @brief Runs the network on a frame, built for the frame's shape first if it is not.
 *
 * @param n      The network.
 * @param frame  The frame.
 * @param slot   -1: the frame and the answer are host memory at @a input and @a output. Otherwise
 *               they are the pair imported into that slot, which the caller knows holds them.
 * @param input  The frame in host memory, for slot -1.
 * @param output Receives the answer in host memory, for slot -1.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE; ERROR_REJECTED for an extent the network does not take; ERROR_DROPPED
 *               with VULKAN_NETWORK_DROPPED for a frame whose network wait ran out; ERROR_FAILED.
 */
extern enum error_code
vulkan_network_infer (struct vulkan_network     *n,
                      struct vulkan_frame const *frame,
                      int                        slot,
                      STD(uint8_t) const        *input,
                      STD(uint8_t)              *output,
                      struct error              *e);

#undef STD

/** @brief The GPU times of the last frame whose timestamps the device gave; all 0 before the first.
 *
 * @param n The network, or nullptr.
 * @return  The times; all 0 for no network.
 */
extern struct vulkan_network_times
vulkan_network_frame_times (struct vulkan_network const *n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_BACKEND_VULKAN_NETWORK_H_ */
