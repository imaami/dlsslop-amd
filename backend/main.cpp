// Native Linux worker for DLSS5VKLayer's shared-memory transport.
// SPDX-License-Identifier: MIT
#include "offline.hpp"
#include "open.hpp"
#include "options.hpp"
#include "serve.hpp"

#include <cstdio>

namespace {
// The exit status: 0, or 1 once the error is reported.
int status(const dlsslop::Result<void>& ran)
{
    if (ran) return 0;
    std::fprintf(stderr, "dlsslopd: %s\n", ran.error().what.c_str());
    return 1;
}
} // namespace

int main(int argc, char** argv)
{
    auto parsed = dlsslop::parse(argc, argv);
    if (!parsed) return status(std::unexpected(std::move(parsed).error()));
    dlsslop::Options& o = *parsed;
    const unsigned tier = o.tier.value_or(kNativeDefaultTier);
    if (o.diagnose)
        return status(dlsslop::with_engine(o, tier, [](auto& engine) -> dlsslop::Result<void> {
            std::fprintf(stderr, "selected %s\n", engine.device().c_str());
            return {};
        }));
    if (o.test_identity) std::fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
    if (o.self_test)
        return status(dlsslop::with_engine(o, tier, [&o](auto& engine) -> dlsslop::Result<void> {
            DLSSLOP_TRY(engine.prepare());
            if constexpr (requires { engine.self_test(o); })
                return engine.self_test(o);
            else
                return dlsslop::fail("--self-test requires the real network");
        }));
    if (!o.input.empty())
        return status(dlsslop::with_engine(o, tier, [&o](auto& engine) -> dlsslop::Result<void> {
            DLSSLOP_TRY(engine.prepare());
            return dlsslop::run_offline(o, engine);
        }));
    return status(dlsslop::run_worker(o));
}
