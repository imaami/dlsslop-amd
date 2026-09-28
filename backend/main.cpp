// Native Linux worker for DLSS5VKLayer's shared-memory transport.
// SPDX-License-Identifier: MIT
#include "offline.h"
#include "open.h"
#include "options.h"
#include "serve.h"

#include <cstdio>
#include <exception>


int main(int argc, char** argv)
{
    try {
        auto parsed = dlsslop::parse(argc, argv);
        if (!parsed) {
            std::fprintf(stderr, "dlsslopd: %s\n", parsed.error().what.c_str());
            return 1;
        }
        dlsslop::Options& o = *parsed;
        const unsigned tier = o.tier.value_or(kNativeDefaultTier);
        const auto failed = [](const dlsslop::Error& error) {
            std::fprintf(stderr, "dlsslopd: %s\n", error.what.c_str());
            return 1;
        };
        if (o.diagnose) {
            const auto engine = dlsslop::open_backend(o, tier);
            if (!engine) return failed(engine.error());
            std::fprintf(stderr, "selected %s\n", (*engine)->device().c_str());
            return 0;
        }
        if (o.test_identity)
            std::fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
        if (o.self_test || !o.input.empty()) {
            const auto engine = dlsslop::open_backend(o, tier);
            if (!engine) return failed(engine.error());
            const auto ran = [&]() -> dlsslop::Result<void> {
                DLSSLOP_TRY((*engine)->prepare());
                return o.self_test ? (*engine)->self_test(o) : dlsslop::run_offline(o, **engine);
            }();
            if (!ran) return failed(ran.error());
        } else {
            dlsslop::run_worker(o);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dlsslopd: %s\n", e.what());
        return 1;
    }
}
