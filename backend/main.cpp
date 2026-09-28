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
        dlsslop::Options o = dlsslop::parse(argc, argv);
        const unsigned tier = o.tier.value_or(kNativeDefaultTier);
        if (o.diagnose) {
            std::fprintf(stderr, "selected %s\n", dlsslop::open_backend(o, tier)->device().c_str());
            return 0;
        }
        if (o.test_identity)
            std::fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
        if (o.self_test || !o.input.empty()) {
            const auto engine = dlsslop::open_backend(o, tier);
            engine->prepare();
            if (o.self_test) engine->self_test(o);
            else dlsslop::run_offline(o, *engine);
        } else {
            dlsslop::run_worker(o);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dlsslopd: %s\n", e.what());
        return 1;
    }
}
