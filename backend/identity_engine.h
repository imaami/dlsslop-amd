// DIAGNOSTIC ONLY (--test-identity): each frame answered with itself, without a
// network, HIP or a device.
// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"

#include <cstring>

namespace dlsslop {

class IdentityEngine : public EngineBase<IdentityEngine> {
    unsigned tier_;

public:
    static constexpr unsigned max_passes = kMaxPasses;
    // Like the HIP engine, so a GPU-less daemon follows a tier through its rebuild.
    static constexpr bool rebuilds_for_tier = true;

    explicit IdentityEngine(unsigned tier) : tier_(tier) {}
    const char* name() const { return "identity"; }
    std::string device() const { return "no device (identity test)"; }
    unsigned tier() const { return tier_; }
    std::string processing() const { return "no processing"; }
    Result<void> retier(unsigned tier)
    {
        tier_ = tier;
        return {};
    }
    Result<void> prepare() { return {}; }
    Result<void> infer(const Frames& io, unsigned w, unsigned h, unsigned, const ProcessingSettings& settings = {},
                       FrameTrace* = nullptr)
    {
        std::memcpy(io.answer, io.proxy, size_t(w) * h * (settings.fp16 ? 8 : 4));
        return {};
    }
};

} // namespace dlsslop
