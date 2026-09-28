// Device-local transport: the socket on which the layer offers its exported frames.
// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"
#include "shm_protocol.h"
#include "trace.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace dlsslop {
// The socket beside the channel file on which the layer offers its exported
// frames. systemd's socket unit hands it over already bound (LISTEN_FDS), and
// connecting to it is then how a client starts the worker; otherwise the
// worker binds it itself, only when the GPU codec can import frames.
struct TransportListener {
    dlsslop::Descriptor socket;
    std::string path;
    bool bound = false; // This worker created the pathname and removes it again.
    TransportListener(const std::string& channel, bool wanted) : path(ShmTransportPath(channel))
    {
        const char* pid = std::getenv("LISTEN_PID");
        const char* count = std::getenv("LISTEN_FDS");
        if (pid && count && std::strtol(pid, nullptr, 10) == getpid() && !std::strcmp(count, "1")) {
            socket.fd = 3; // SD_LISTEN_FDS_START
            fcntl(socket.fd, F_SETFL, fcntl(socket.fd, F_GETFL) | O_NONBLOCK);
            fcntl(socket.fd, F_SETFD, FD_CLOEXEC);
            return;
        }
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (!wanted) return;
        errno = ENAMETOOLONG;
        if (path.size() < sizeof address.sun_path) {
            std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
            socket.fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
            bound = socket.fd >= 0 && !bind(socket.fd, reinterpret_cast<const sockaddr*>(&address), sizeof address);
            // Never replace a socket someone else owns: systemd's, left listening by
            // an idle dlsslop.socket, would be gone for good once this worker exits.
            if (!bound && errno == EADDRINUSE)
                throw std::runtime_error(path + " exists: stop dlsslop.socket before starting dlsslopd "
                                         "by hand, or remove the file if nothing uses it");
            if (bound && !listen(socket.fd, 4))
                return;
        }
        std::fprintf(stderr, "device-local transport unavailable (%s): %s\n", path.c_str(), std::strerror(errno));
    }
    ~TransportListener() { if (bound) unlink(path.c_str()); }
};

// One offer: the layer sends it right after connecting.
bool receive_offer(int peer, ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2]);
// Imports every pending offer and answers each on its own connection: one
// byte, nonzero when imported.
void accept_offers(const TransportListener& listener, Backend& engine);
} // namespace dlsslop
