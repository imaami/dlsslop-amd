// The network on Vulkan: DLSSNR-AMD's runtime (vulkan-nr/) on a device of the daemon's own.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace dlsslop {

// Where the Vulkan network finds what it loads.
struct VulkanPaths {
    std::string model;    // dlssnr.bin, extracted from the user's nvngx_dlssnr.dll
    std::string shaders;  // the network's SPIR-V and markers, with runtime/ and temporal/ below
    std::string cache;    // writable pipeline cache; empty: none
};

// One request's work. The frame is w x h tightly packed RGBA8, or RGBA16F with fp16,
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

class VulkanNetwork {
public:
    static constexpr unsigned kMaxPasses = 16;
    static constexpr float kMaxControl = 2;
    // device < 0: the first physical device the network can run on.
    VulkanNetwork(const VulkanPaths& paths, int device);
    ~VulkanNetwork();
    VulkanNetwork(const VulkanNetwork&) = delete;
    VulkanNetwork& operator=(const VulkanNetwork&) = delete;

    const std::string& device_name() const;
    unsigned device_index() const;
    // Builds the network for a frame's shape now, rather than on its first request:
    // seconds of work, all of it while the caller still reports itself starting.
    // True when that took a build.
    bool shape(const VulkanFrame& frame);
    // True when shape() would build.
    bool shape_differs(const VulkanFrame& frame) const;
    // Imports an offered proxy/answer pair (ShmTransportOffer): memory of allocation
    // bytes, each bound to a buffer of size bytes. Takes ownership of each descriptor it
    // imports and sets it to -1; the caller closes the rest.
    bool import(uint32_t generation, const uint64_t allocation[2], const uint64_t size[2], int fds[2]);
    bool holds(uint32_t generation, size_t bytes) const;
    // Generation 0: the frame and answer are host memory at input and output.
    // Otherwise they are the imported pair of that generation (holds() first).
    void infer(const VulkanFrame& frame, uint32_t generation, const uint8_t* input, uint8_t* output);

    float upload_ms = 0, inference_ms = 0, readback_ms = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dlsslop
