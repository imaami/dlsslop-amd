// SPDX-License-Identifier: MIT
#include "transport.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace dlsslop {
Result<TransportListener> TransportListener::open(const std::string& channel, bool wanted)
{
    TransportListener listener(ShmTransportPath(channel));
    const char* pid = std::getenv("LISTEN_PID");
    const char* count = std::getenv("LISTEN_FDS");
    if (pid && count && std::strtol(pid, nullptr, 10) == getpid() && !std::strcmp(count, "1")) {
        listener.socket.fd = 3; // SD_LISTEN_FDS_START
        fcntl(listener.socket.fd, F_SETFL, fcntl(listener.socket.fd, F_GETFL) | O_NONBLOCK);
        fcntl(listener.socket.fd, F_SETFD, FD_CLOEXEC);
        return listener;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (!wanted) return listener;
    const std::string& path = listener.path;
    errno = ENAMETOOLONG;
    if (path.size() < sizeof address.sun_path) {
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        listener.socket.fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        listener.bound = listener.socket.fd >= 0 &&
                         !bind(listener.socket.fd, reinterpret_cast<const sockaddr*>(&address), sizeof address);
        // Never replace a socket someone else owns: systemd's, left listening by
        // an idle dlsslop.socket, would be gone for good once this worker exits.
        if (!listener.bound && errno == EADDRINUSE)
            return fail(path + " exists: stop dlsslop.socket before starting dlsslopd "
                               "by hand, or remove the file if nothing uses it");
        if (listener.bound && !listen(listener.socket.fd, 4)) return listener;
    }
    std::fprintf(stderr, "device-local transport unavailable (%s): %s\n", path.c_str(), std::strerror(errno));
    listener.socket = Descriptor(); // Nothing to accept on.
    return listener;
}

bool receive_offer(int peer, ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2])
{
    pollfd ready{peer, POLLIN, 0};
    if (poll(&ready, 1, 100) != 1) return false;
    alignas(cmsghdr) char control[CMSG_SPACE(2 * sizeof(int))]{};
    iovec data{&offer, sizeof offer};
    msghdr message{};
    message.msg_iov = &data;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof control;
    const ssize_t got = recvmsg(peer, &message, MSG_CMSG_CLOEXEC);
    const cmsghdr* rights = got > 0 ? CMSG_FIRSTHDR(&message) : nullptr;
    if (!rights || rights->cmsg_level != SOL_SOCKET || rights->cmsg_type != SCM_RIGHTS) return false;
    const size_t count = std::min<size_t>(2, (rights->cmsg_len - CMSG_LEN(0)) / sizeof(int));
    for (size_t i = 0; i < count; ++i) std::memcpy(&fds[i].fd, CMSG_DATA(rights) + i * sizeof(int), sizeof(int));
    // Each buffer lies within its memory: a backend maps the memory and fits frames to the buffer.
    const auto fits = [&offer](unsigned i) { return offer.size[i] && offer.size[i] <= offer.allocation[i]; };
    return size_t(got) == sizeof offer && count == 2 && offer.magic == kShmMagic &&
           !(message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) && fits(0) && fits(1);
}

} // namespace dlsslop
