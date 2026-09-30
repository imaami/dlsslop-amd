// The Vulkan network's host runtime: every Vulkan object the network needs for
// one frame shape, built with its GPU work in one submission, and the commands
// of a frame. A port of the production path of DLSSNR-AMD's linux/src/core
// (MIT): nr_runtime.cpp, the device part of NrSession::build in nr_graph.cpp
// and nrvk.hpp.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"
#include "vulkan_plan.h"

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace dlsslop {

// Where the Vulkan network finds what it loads.
struct VulkanPaths {
    std::string model;   // dlssnr.bin, extracted from the user's nvngx_dlssnr.dll
    std::string shaders; // the network's SPIR-V and markers, with runtime/ and temporal/ below
    std::string cache;   // writable pipeline cache; empty: none
};

// VkResult as a Result: WHAT, and the code, when it is not VK_SUCCESS.
Result<void> vk_check(VkResult result, const char* what);
// The first of MEMORY's types among BITS with every property in WANT.
Result<uint32_t> memory_type(const VkPhysicalDeviceMemoryProperties& memory, uint32_t bits,
                             VkMemoryPropertyFlags want);

} // namespace dlsslop

namespace dlsslop::vulkan {

class Model;

// The physical-device queries a build makes: the loader's in dlsslopd; in the
// layer's module the next layer's, looked up when the module opens, so that
// the build's thread never looks a function up through the loader.
struct PhysicalFunctions {
    PFN_vkGetPhysicalDeviceQueueFamilyProperties queue_families;
    PFN_vkGetPhysicalDeviceProperties2 properties;
    PFN_vkGetPhysicalDeviceFormatProperties2 format_properties;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrix_properties;
};

// The device the network runs on, which it does not own: dlsslopd's, or a
// game's in the layer, with every feature of network_requirements.h enabled.
// Its device-level functions are the loader's.
struct Device {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    // The queue of FAMILY that takes a build's submission, with LOCK and
    // UNLOCK, when set, called with CONTEXT around that submission only.
    VkQueue queue;
    uint32_t family;
    void (*lock)(void* context);
    void (*unlock)(void* context);
    void* context;
    VkPhysicalDeviceMemoryProperties memory;
    PhysicalFunctions functions;
    // Where the runtime's log lines go; null: nowhere.
    void (*log)(const char* line);
};

// The most bytes one of DEVICE's storage buffers holds bound whole, as plan()
// takes it: the smaller of its maxStorageBufferRange and
// maxMemoryAllocationSize; no limit when DEVICE cannot say, which fails its
// build.
uint64_t storage_limit(const Device& device);

// What a runtime is built for: frames of WIDTH x HEIGHT, RGBA8 or with FP16
// RGBA16F, up to PASSES chained evaluations, with or without motion history
// and the pass stages.
struct Shape {
    uint32_t width, height;
    bool fp16, motion, stages;
    uint8_t passes;
};

// A frame's controls (upstream: nr::Controls), in the ranges dlsslopd's
// settings allow. PASSES is at most the build's.
struct Controls {
    uint32_t passes, style;
    float intensity, tone, structure;
    // The skin's local structure under the automatic mask; below 0 it follows
    // STRUCTURE.
    float skin;
    bool auto_mask;
    // dlsslop-amd's stages after every pass, 0..1 each; 0 skips one.
    float sharpness, color_preserve;
};

// Handles a runtime owns, null until made.
struct Image {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
};
struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
};
struct Pipeline {
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
};

// The network built for one Shape on a Device. It is movable and not
// copyable; the device must have finished its work before it is destroyed.
//
// Invariants:
// - The activation arena is zeroed once at build and never cleared or laid
//   out again while the runtime lives: its zero tails, the persistent runs'
//   epochs and the tile counters depend on it.
// - Descriptors are written once. Every dispatch runs with the network's
//   input image in SHADER_READ_ONLY_OPTIMAL and every other image in
//   GENERAL.
// - The invalidate-only barriers between steps rely on gfx1201's caches
//   below L2 being write-through (upstream: nr_graph.cpp:4230-4258); the
//   steps that tile counters order rely on the queue starting consecutive
//   dispatches' workgroups in order.
// - The motion history's latch and parity flip when a frame is recorded,
//   not when it is submitted.
class Runtime {
public:
    // The network for SHAPE on DEVICE, from PLAN, which is of SHAPE's extent
    // on DEVICE's storage_limit(): its SPIR-V from PATHS.shaders, its weights
    // from PATHS.model and its pipeline cache at PATHS.cache. The queue must
    // be free of the frames of a runtime being replaced.
    static Result<Runtime> build(const Device& device, const VulkanPaths& paths, const Shape& shape,
                                 const Plan& plan);
    Runtime(Runtime&& other) noexcept;
    Runtime& operator=(Runtime&&) = delete;
    ~Runtime();

    // Records a frame of the shape: from FRAME, an image of it in
    // TRANSFER_DST_OPTIMAL written by transfers, through the network with
    // CONTROLS and back into FRAME, left in TRANSFER_SRC_OPTIMAL for
    // transfers. RESET drops the motion history for this frame.
    void record(VkCommandBuffer cmd, VkImage frame, const Controls& controls, bool reset);

private:
    // The network's kernels' pipelines, then the runtime's own.
    enum Adapter : size_t { kAlpha = size_t(Kernel::kCount), kStages, kLuma, kFlow, kPre, kPost, kPipelines };
    static constexpr size_t kAdapters = kPipelines - kAlpha;
    static constexpr uint32_t kLevels = 4; // the motion estimate's pyramid
    static constexpr uint32_t kMaxPasses = 16;
    static constexpr uint32_t kTimingSlots = 4;

    // Every object the runtime owns.
    struct Objects {
        Buffer arena, weights, params;
        // The network's input and answer, its second output, the first
        // pass's input for later passes, the pass stages' scratch, and the
        // motion history's luma pyramids, flow, histories, depth and one
        // history a pass.
        Image input, answer, second, shown, scratch;
        Image luma[2][kLevels], flow[kLevels], history[2], depth, history_store[kMaxPasses];
        VkSampler nearest, linear;
        VkQueryPool timing;
        Pipeline pipelines[kPipelines];
        VkDescriptorPool pool;
        // Each kernel's set, and the runtime's own pipelines' by what they
        // differ in: the answer or the scratch, the parity and level, and
        // the history.
        VkDescriptorSet kernel_sets[size_t(Kernel::kCount)];
        VkDescriptorSet alpha_sets[2], stage_sets[2], luma_sets[2][kLevels], flow_sets[2][kLevels];
        VkDescriptorSet pre_sets[2], post_sets[2];
    };
    // How frames are recorded, and the state they carry.
    struct State {
        uint32_t width, height, passes;
        bool motion, stages, rgba8;
        // The frame is copied into the input, not blitted; the post block
        // stores the answer in the frame's format, which is then copied out;
        // the post block restores the frame's alpha itself.
        bool input_direct, answer_direct, post_alpha;
        // One pass with motion: the post block writes the history the next
        // frame reads, in turn into history[0] and history[1].
        bool pingpong;
        uint32_t level_width[kLevels], level_height[kLevels];
        bool latch;
        uint32_t parity, current;
        // The timing ring's next slot and the frames recorded into it.
        uint32_t timing_slot;
        uint64_t timed;
    };

    Device device_;
    Objects objects_{};
    State state_{};
    std::vector<Step> steps_;
    std::vector<uint32_t> push_;

    explicit Runtime(const Device& device) : device_(device) {}
    Result<void> make(const VulkanPaths& paths, const Shape& shape, const Plan& plan);
    Result<void> make_resources(const Shape& shape, const Plan& plan, bool timing);
    Result<void> make_pipelines(const VulkanPaths& paths);
    Result<void> make_sets();
    Result<void> run_setup(const Plan& plan, const Model& model, double& packing);
    void dispatch(VkCommandBuffer cmd, size_t pipeline, VkDescriptorSet set, uint32_t x, uint32_t y,
                  const void* push, uint32_t bytes, uint32_t z = 1) const;
    void run_step(VkCommandBuffer cmd, const Step& step, size_t pipeline, VkDescriptorSet set,
                  const uint32_t* push) const;
};

} // namespace dlsslop::vulkan
