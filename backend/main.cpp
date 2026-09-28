// Native Linux worker for DLSS5VKLayer's shared-memory transport.
// SPDX-License-Identifier: MIT
#include "offline.h"
#include "open.h"
#include "options.h"
#include "serve.h"

#include <cstdio>

int main(int argc, char** argv)
{
    const auto failed = [](const dlsslop::Error& error) {
        std::fprintf(stderr, "dlsslopd: %s\n", error.what.c_str());
        return 1;
    };
    auto parsed = dlsslop::parse(argc, argv);
    if (!parsed) return failed(parsed.error());
    dlsslop::Options& o = *parsed;
    if (o.test_identity && !o.diagnose)
        std::fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
    if (!o.diagnose && !o.self_test && o.input.empty()) {
        const auto served = dlsslop::run_worker(o);
        return served ? 0 : failed(served.error());
    }
    const auto ran = dlsslop::with_engine(o, o.tier.value_or(kNativeDefaultTier), [&o](auto& engine) -> dlsslop::Result<void> {
        if (o.diagnose) {
            std::fprintf(stderr, "selected %s\n", engine.device().c_str());
            return {};
        }
        DLSSLOP_TRY(engine.prepare());
        if (!o.self_test) return dlsslop::run_offline(o, engine);
        if constexpr (requires { engine.self_test(o); })
            return engine.self_test(o);
        else
            return dlsslop::fail("--self-test requires the real network");
    });
    return ran ? 0 : failed(ran.error());
}
