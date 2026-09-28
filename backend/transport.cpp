// SPDX-License-Identifier: MIT
#include "transport.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace dlsslop {
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

// Imports every pending offer and answers each on its own connection: one
// byte, nonzero when imported.
void accept_offers(const TransportListener& listener, Backend& engine)
{
    for (int peer; (peer = accept4(listener.socket.fd, nullptr, nullptr, SOCK_CLOEXEC)) >= 0; close(peer)) {
        ShmTransportOffer offer{};
        dlsslop::Descriptor fds[2];
        if (!receive_offer(peer, offer, fds)) continue; // A start or liveness probe sends nothing.
        const uint8_t imported = engine.import(offer, fds);
        if (!imported) std::fprintf(stderr, "device-local transport offer rejected\n");
        send(peer, &imported, 1, MSG_NOSIGNAL);
    }
}
} // namespace dlsslop
