// What a frame is processed with, as the channel states it.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.hpp"
#include "shm_protocol.h"
#include "tuning_math.hpp"
#include "vulkan_frame.hpp"

namespace dlsslop {

struct ProcessingSettings {
    NativeTuning tuning;
    // The Vulkan model's own conditioning; the HIP network has only the defaults.
    unsigned style = 0;
    float skin_structure = -1;
    bool auto_mask = true;
    float color_preserve = 0;
    bool fp16 = false;
    bool precision16 = true;
    bool motion = false;
    unsigned motion_quality = kMVecBalanced;
    unsigned motion_grid = kMVecPixels4;
};

// The channel's settings for the next frame, or its rejection: an NVIDIA
// preset neither network has, or a control outside its range. Older and
// external clients can store either.
Result<ProcessingSettings> read_settings(const ShmHeader* h);

// A frame of W x H through the Vulkan network, PASSES times, as SETTINGS say.
inline VulkanFrame vulkan_frame(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings)
{
    VulkanFrame f;
    f.width = w;
    f.height = h;
    f.fp16 = settings.fp16;
    f.passes = passes;
    f.intensity = settings.tuning.intensity;
    f.local_tone = settings.tuning.tone;
    f.local_structure = settings.tuning.structure;
    f.style = settings.style;
    f.skin_structure = settings.skin_structure;
    f.auto_mask = settings.auto_mask;
    f.sharpness = settings.tuning.sharpness;
    f.color_preserve = settings.color_preserve;
    f.motion = settings.motion;
    return f;
}

}  // namespace dlsslop
