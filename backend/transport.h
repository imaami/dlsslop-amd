/** @file
 *
 * The device-local transport: the socket beside the channel file on which the layer offers its
 * exported frames (ShmTransportOffer). systemd's socket unit hands it over already bound
 * (LISTEN_FDS), and connecting to it is then how a client starts the daemon; otherwise the daemon
 * binds it itself, only when its engine can import frames. transport.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_TRANSPORT_H_
#define DLSSLOP_AMD_BACKEND_TRANSPORT_H_

#include <stddef.h>

#include "error.h"
#include "shm_protocol.h"

/** @brief The socket on which offers arrive: transport_listener_init() opens it and
 *         transport_listener_fini() closes it.
 */
struct transport_listener {
	char *bound_path; //!< The socket's path, which this daemon bound and removes again, or nullptr.
	int   socket;     //!< The listening socket, nonblocking, or -1 for none: nothing to accept on.
};

/** @brief Opens the socket on which the layer offers its frames.
 *
 * Takes the socket that systemd passed (LISTEN_PID, LISTEN_FDS) whether or not one is wanted.
 * Otherwise binds the channel's ShmTransportPath() if one is wanted, but never replaces a socket
 * that is there: systemd's, left listening by an idle dlsslop.socket, would be gone for good once
 * this daemon exits. A socket that cannot be had otherwise is logged, and the daemon serves the
 * channel's frame slots without it.
 *
 * @param dest           Receives the listener.
 * @param channel        The channel file's path.
 * @param channel_length The length of its path.
 * @param wanted         Whether the engine can import offered frames.
 * @param e              Receives the words for what stopped it, or nullptr.
 * @return               ERROR_NONE, or ERROR_FAILED when another socket is bound there or without
 *                       memory; @a dest has no socket then.
 */
extern enum error_code
transport_listener_init (struct transport_listener *dest,
                         char const                *channel,
                         size_t                     channel_length,
                         bool                       wanted,
                         struct error              *e);

/** @brief Removes the socket's path if this daemon bound it and closes the socket.
 *
 * @param listener The listener, or nullptr.
 */
extern void
transport_listener_fini (struct transport_listener *listener);

/** @brief Receives the one offer that the layer sends right after it connects.
 *
 * A start or liveness probe sends nothing: a connection that sends nothing within 100 ms has no
 * offer. Each offered buffer must lie within its memory, as a backend maps the memory and fits
 * frames to the buffer.
 *
 * @param peer  The accepted connection.
 * @param offer Receives the offer.
 * @param fds   Receives the offered proxy and answer memory, which the caller then owns; both -1
 *              without a valid offer, the descriptors of an invalid one closed.
 * @return      true for a valid offer.
 */
extern bool
transport_receive_offer (int                       peer,
                         struct ShmTransportOffer *offer,
                         int                       fds[2]);

/** @brief Answers an offer on its connection: one byte, nonzero when it was imported.
 *
 * Logs an offer that was not imported, and an answer that cannot be sent.
 *
 * @param peer     The offer's connection.
 * @param imported Whether the offer was imported.
 */
extern void
transport_answer (int  peer,
                  bool imported);

#endif /* DLSSLOP_AMD_BACKEND_TRANSPORT_H_ */
