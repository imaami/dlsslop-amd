/** @file
 *
 * The device-local transport: transport.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "transport.h"

/** @brief Logs why the daemon serves without the transport.
 *
 * @param channel The channel file's path, which the socket's path extends.
 * @param err     The errno value of what stopped it.
 * @return        ERROR_NONE: serving goes on with the channel's frame slots.
 */
[[gnu::cold]]
static enum error_code
unavailable (char const *channel,
             int         err)
{
	char buf[64];
	fprintf(stderr, "device-local transport unavailable (%s%s): %s\n", channel, kShmTransportSuffix,
	        strerror_r(err, buf, sizeof buf));
	return ERROR_NONE;
}

/** @brief The first descriptor that systemd passes (SD_LISTEN_FDS_START). */
#define LISTEN_FDS_START 3

/** @brief Takes the socket that systemd passed, bound and listening, and makes it nonblocking.
 *
 * @param dest    Receives the socket.
 * @param channel The channel file's path, for the log.
 * @return        ERROR_NONE, with no socket if it cannot be made nonblocking.
 */
static enum error_code
adopt (struct transport_listener *dest,
       char const                *channel)
{
	int fd = LISTEN_FDS_START;
	int const flags = fcntl(fd, F_GETFL);
	if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) || fcntl(fd, F_SETFD, FD_CLOEXEC)) {
		int const err = errno;
		// A socket that would block the serving loop is not served: systemd keeps its own.
		close(fd);
		fd = -1;
		return unavailable(channel, err);
	}
	dest->socket = fd;
	return ERROR_NONE;
}

#undef LISTEN_FDS_START

/** @brief Binds the socket beside the channel file and listens on it.
 *
 * @param dest           Receives the socket and its path.
 * @param channel        The channel file's path.
 * @param channel_length The length of its path.
 * @param e              Receives the words for what stopped it, or nullptr.
 * @return               ERROR_NONE, also without a socket, or ERROR_FAILED.
 */
static enum error_code
bind_socket (struct transport_listener *dest,
             char const                *channel,
             size_t                     channel_length,
             struct error              *e)
{
	struct sockaddr_un address = {.sun_family = AF_UNIX};
	// A path cut to fit would name another socket.
	size_t const length = channel_length + sizeof kShmTransportSuffix - 1;
	if (length >= sizeof address.sun_path)
		return unavailable(channel, ENAMETOOLONG);
	memcpy(address.sun_path, channel, channel_length);
	memcpy(address.sun_path + channel_length, kShmTransportSuffix, sizeof kShmTransportSuffix);
	char *path = malloc(length + 1);
	if (!path)
		return error_fail(e, "out of memory");
	memcpy(path, address.sun_path, length + 1);

	int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	if (fd < 0) {
		int const err = errno;
		free(path);
		path = nullptr;
		return unavailable(channel, err);
	}
	if (bind(fd, (struct sockaddr const *)&address, sizeof address)) {
		int const err = errno;
		close(fd);
		fd = -1;
		free(path);
		path = nullptr;
		// Never replace a socket someone else owns: systemd's, left listening by an idle
		// dlsslop.socket, would be gone for good once this worker exits.
		if (err == EADDRINUSE)
			return error_fail(e, "%s exists: stop dlsslop.socket before starting dlsslopd by hand, "
			                     "or remove the file if nothing uses it", address.sun_path);
		return unavailable(channel, err);
	}

	// The path is this daemon's now, and removed again when it stops.
	dest->bound_path = path;
	if (listen(fd, 4)) {
		int const err = errno;
		close(fd);
		fd = -1;
		return unavailable(channel, err);
	}
	dest->socket = fd;
	return ERROR_NONE;
}

enum error_code
transport_listener_init (struct transport_listener *dest,
                         char const                *channel,
                         size_t                     channel_length,
                         bool                       wanted,
                         struct error              *e)
{
	*dest = (struct transport_listener){.socket = -1};
	char const *const pid = getenv("LISTEN_PID");
	char const *const count = getenv("LISTEN_FDS");
	// A LISTEN_PID out of range reads as LONG_MAX or LONG_MIN, which no process has.
	if (pid && count && strtol(pid, nullptr, 10) == getpid() && !strcmp(count, "1"))
		return adopt(dest, channel);
	if (!wanted)
		return ERROR_NONE;
	return bind_socket(dest, channel, channel_length, e);
}

void
transport_listener_fini (struct transport_listener *listener)
{
	if (!listener)
		return;
	if (listener->bound_path) {
		// A path left behind keeps the next daemon from binding its socket.
		if (unlink(listener->bound_path) && errno != ENOENT) {
			int const err = errno;
			char buf[64];
			fprintf(stderr, "cannot remove the device-local transport socket %s: %s\n",
			        listener->bound_path, strerror_r(err, buf, sizeof buf));
		}
		free(listener->bound_path);
		listener->bound_path = nullptr;
	}
	if (listener->socket >= 0) {
		// A listening socket: close() has nothing to report.
		close(listener->socket);
		listener->socket = -1;
	}
	*listener = (struct transport_listener){.socket = -1};
}

/** @brief Closes a pair of offered descriptors.
 *
 * @param fds The descriptors; each is -1 afterwards.
 */
static void
close_pair (int fds[2])
{
	for (size_t i = 0; i < 2; ++i) {
		if (fds[i] >= 0) {
			// Never used: close() has nothing to report.
			close(fds[i]);
			fds[i] = -1;
		}
	}
}

/** @brief Whether an offered buffer lies within its memory and holds a frame.
 *
 * @param offer The offer.
 * @param i     0 for the proxy, 1 for the answer.
 * @return      true if it does.
 */
static bool
fits (struct ShmTransportOffer const *offer,
      size_t                          i)
{
	return offer->size[i] && offer->size[i] <= offer->allocation[i];
}

bool
transport_receive_offer (int                       peer,
                         struct ShmTransportOffer *offer,
                         int                       fds[2])
{
	fds[0] = -1;
	fds[1] = -1;
	struct pollfd ready = {.fd = peer, .events = POLLIN};
	if (poll(&ready, 1, 100) != 1)
		return false;

	alignas(struct cmsghdr) char control[CMSG_SPACE(2 * sizeof (int))] = {0};
	struct iovec data = {.iov_base = offer, .iov_len = sizeof *offer};
	struct msghdr message = {
		.msg_iov        = &data,
		.msg_iovlen     = 1,
		.msg_control    = control,
		.msg_controllen = sizeof control,
	};
	ssize_t const got = recvmsg(peer, &message, MSG_CMSG_CLOEXEC);
	if (got < 0)
		return false;

	// Descriptors also come with a message of the wrong size: they are taken to be closed.
	struct cmsghdr const *const rights = CMSG_FIRSTHDR(&message);
	size_t count = 0;
	if (rights && rights->cmsg_level == SOL_SOCKET && rights->cmsg_type == SCM_RIGHTS) {
		count = (rights->cmsg_len - CMSG_LEN(0)) / sizeof (int);
		if (count > 2)
			count = 2;
		memcpy(fds, CMSG_DATA(rights), count * sizeof (int));
	}
	if (got == (ssize_t)sizeof *offer && count == 2 && offer->magic == kShmMagic
	    && !(message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) && fits(offer, 0) && fits(offer, 1))
		return true;
	close_pair(fds);
	return false;
}

void
transport_answer (int  peer,
                  bool imported)
{
	if (!imported)
		fputs("device-local transport offer rejected\n", stderr);
	uint8_t const answer = imported;
	if (send(peer, &answer, 1, MSG_NOSIGNAL) < 0) {
		// The layer stopped waiting for the answer.
		int const err = errno;
		char buf[64];
		fprintf(stderr, "device-local transport offer not answered: %s\n", strerror_r(err, buf, sizeof buf));
	}
}
