// One frame of the Vulkan network's work, for dlsslopd and the in-layer network.
// SPDX-License-Identifier: MIT
#pragma once

namespace dlsslop {

// One frame's work. The frame is w x h tightly packed RGBA8, or RGBA16F with fp16,
// holding display-encoded values, and the answer comes back in the same form.
struct VulkanFrame {
    unsigned width = 0, height = 0;
    bool fp16 = false;
    // The network's own controls (NVIDIA's DLSSNR.Intensity, LocalToneStrength,
    // LocalStructureStrength), not the HIP backend's residual filters. The model
    // takes each up to kMaxControl; more counts as kMaxControl.
    unsigned passes = 1;
    float intensity = 1, local_tone = 1, local_structure = 1;
    // DLSSNR.Style, 0..2, and the automatic mask, under which skin takes its own
    // local structure; a negative one follows local_structure.
    unsigned style = 0;
    float skin_structure = -1;
    bool auto_mask = true;
    // dlsslop-amd's own stages after every pass, 0..1 each, as on HIP: sharpening,
    // and color preservation against the frame the first pass saw. Either builds
    // them in, as a larger pass count does.
    float sharpness = 0, color_preserve = 0;
    // The network's history, fed by the runtime's motion estimate. It starts over
    // whenever the frame's shape, pass count or controls change.
    bool motion = false;
};

}  // namespace dlsslop
