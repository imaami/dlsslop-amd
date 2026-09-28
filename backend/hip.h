// HIP for the daemon's own code: the runtime's entry points as a table, and the
// vendored lmxxf network behind functions that return Results.
// SPDX-License-Identifier: MIT
#pragma once
#include "hip_vendor.h"
#include "result.h"

#include <string>
#include <utility>

namespace dlsslop::hip {

struct Api : Entries {
    // Nothing unless RESULT is an error: then "WHAT: <its name> (RESULT)".
    Result<void> check(int result, const char* what) const
    {
        if (!result) return {};
        return fail(std::string(what) + ": " + hipGetErrorName(result) + " (" + std::to_string(result) + ")");
    }
};

// Nothing, or the vendored network's failure.
inline Result<void> vendor_result(std::string failure)
{
    if (failure.empty()) return {};
    return fail(std::move(failure));
}

// The runtime, found as the vendored network finds it: DLSSLOP_HIP_LIBRARY, or
// the libamdhip64 sonames, then /opt/rocm/lib. On failure, every candidate's
// loader error.
inline Result<Api> load()
{
    Api api{};
    DLSSLOP_TRY(vendor_result(vendor::load(api)));
    return api;
}
// The pinned network with dlsslopd's production options, on the device it makes
// current for this thread.
inline Result<NetworkHandle> network(const NetworkOptions& options)
{
    NetworkHandle network;
    DLSSLOP_TRY(vendor_result(vendor::network(options, network)));
    return network;
}
// One evaluation, queued on the network's stream.
inline Result<void> enqueue(Network& network, void* rgba, void* history, void* rgb)
{
    return vendor_result(vendor::enqueue(network, rgba, history, rgb));
}
inline Result<void> synchronize(Network& network) { return vendor_result(vendor::synchronize(network)); }
// The network's memory use, to stdout.
inline Result<void> print_memory(Network& network) { return vendor_result(vendor::print_memory(network)); }

} // namespace dlsslop::hip
