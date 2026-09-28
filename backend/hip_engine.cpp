// SPDX-License-Identifier: MIT
#include "hip_engine.h"

namespace dlsslop {
int select_device(int requested)
{
    hip_probe::Api api;
    api.Check(api.hipInit(0), "hipInit (check /dev/kfd permissions and ROCm userspace)");
    int count = 0, version = 0, selected = -1;
    api.Check(api.hipRuntimeGetVersion(&version), "HIP runtime version");
    api.Check(api.hipGetDeviceCount(&count), "HIP device count");
    std::fprintf(stderr, "HIP runtime=%d, visible devices=%d\n", version, count);
    for (int i = 0; i < count; ++i) {
        auto p = api.Properties(i);
        std::fprintf(stderr, "  device %d: %s; arch=%s; PCI=%04x:%02x:%02x; VRAM=%.0f MiB\n",
                     i, p.name, p.gcnArchName, p.pciDomainID, p.pciBusID, p.pciDeviceID,
                     p.totalGlobalMem / 1048576.0);
        const bool gfx1201 = !std::strncmp(p.gcnArchName, "gfx1201", 7) &&
                            (!p.gcnArchName[7] || p.gcnArchName[7] == ':');
        if (gfx1201 && ((requested < 0 && selected < 0) || requested == i)) selected = i;
    }
    if (selected < 0) throw std::runtime_error(requested < 0
        ? "no gfx1201 device found (RX 9070/9070 XT required); check HIP_VISIBLE_DEVICES"
        : "selected device is unavailable or is not gfx1201");
    return selected;
}
} // namespace dlsslop
