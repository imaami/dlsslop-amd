// Device-local transport: the socket on which the layer offers its exported frames.
// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"
#include "shm_protocol.h"
#include "files.h"
#include "result.h"

#include <string>
#include <unistd.h>

namespace dlsslop {
// The socket beside the channel file on which the layer offers its exported
// frames. systemd's socket unit hands it over already bound (LISTEN_FDS), and
// connecting to it is then how a client starts the worker; otherwise the
// worker binds it itself, only when the GPU codec can import frames.
struct TransportListener {
    Descriptor socket;
    std::string path;
    bool bound = false; // This worker created the pathname and removes it again.

    static Result<TransportListener> open(const std::string& channel, bool wanted);
    TransportListener(TransportListener&& other) noexcept
        : socket(std::move(other.socket)), path(std::move(other.path)), bound(std::exchange(other.bound, false))
    {
    }
    ~TransportListener()
    {
        if (bound) unlink(path.c_str());
    }
private:
    explicit TransportListener(std::string path) : path(std::move(path)) {}
};

// One offer: the layer sends it right after connecting.
bool receive_offer(int peer, ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2]);
// Imports every pending offer and answers each on its own connection: one
// byte, nonzero when imported.
void accept_offers(const TransportListener& listener, Backend& engine);
} // namespace dlsslop
