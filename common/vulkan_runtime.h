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
};

// The device the network runs on, which it does not own: dlsslopd's, or a
// game's in the layer, which NetworkUnsupported() accepted, with every feature
// of network_requirements.h enabled.
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
// RGBA16F, PASSES chained evaluations, with or without motion history and
// the pass stages.
struct Shape {
    uint32_t width, height;
    bool fp16, motion, stages;
    uint8_t passes;
};

// A frame's controls (upstream: nr::Controls, whose pass count is the
// Shape's here), in the ranges dlsslopd's settings allow.
struct Controls {
    uint32_t style;
    float intensity, tone, structure;
    // The skin's local structure under the automatic mask; below 0 it follows
    // STRUCTURE.
    float skin;
    bool auto_mask;
    // dlsslop-amd's stages after every pass, 0..1 each; 0 skips one.
    float sharpness, color_preserve;
};

// Handles a runtime owns, null until made, and what an image was made as.
struct Image {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    uint32_t width, height;
    VkFormat format;
    VkImageUsageFlags usage;
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

// The network built for one frame extent on a Device, for one Shape of that
// extent at a time. It is movable and not copyable; the device must have
// finished its work before it is destroyed or reshaped.
//
// Invariants:
// - The activation arena is zeroed once at build and never cleared or laid
//   out again while the runtime lives: its zero tails, the persistent runs'
//   epochs and the tile counters depend on it.
// - Descriptors are written only by build() and reshape(). Every dispatch
//   runs with the network's input image in SHADER_READ_ONLY_OPTIMAL and
//   every other image that a set binds in GENERAL. The first frame recorded
//   after build() or reshape() moves every image there from UNDEFINED, and so
//   does each frame after it until submitted() says one was submitted.
// - The invalidate-only barriers between steps rely on gfx1201's caches
//   below L2 being write-through (upstream: nr_graph.cpp:4660-4688); the
//   steps that tile counters order rely on the queue starting consecutive
//   dispatches' workgroups in order.
// - The motion history's latch, parity and noise seed change only when
//   submitted() says that the last frame recorded was submitted, and start
//   over at reshape().
class Runtime {
public:
    // The network for SHAPE on DEVICE, from PLAN, which is of SHAPE's extent
    // on DEVICE's storage_limit(), with the kernels without the upper clamp
    // that the model's weights allow (unclamp()): its SPIR-V from
    // PATHS.shaders, its weights from PATHS.model and its pipeline cache at
    // PATHS.cache, with the pipelines of every shape of that extent. The queue
    // must be free of the frames of a runtime being replaced.
    static Result<Runtime> build(const Device& device, const VulkanPaths& paths, const Shape& shape, Plan plan);
    Runtime(Runtime&& other) noexcept;
    Runtime& operator=(Runtime&&) = delete;
    ~Runtime();

    // Makes the runtime what build() makes for SHAPE, of the extent it was
    // built for: its images and every descriptor set as SHAPE wants them. It
    // makes no pipeline and submits nothing; the next frame moves the images
    // into their layouts. The weights, the arena and the pipelines stay, and
    // so does each image that SHAPE wants as it is, its contents discarded as
    // a new image's are. A runtime that fails to reshape must be destroyed.
    Result<void> reshape(const Shape& shape);

    // Records a frame of the shape: from PROXY, the frame's pixels packed in
    // its format, through the network with CONTROLS into ANSWER in the same
    // form. Transfers read PROXY and write ANSWER, which the caller's
    // barriers order against the transfers before and after. RESET drops the
    // motion history for this frame. With QUERIES, writes timestamps QUERY,
    // once the frame is in the network's input, and QUERY + 1, once the
    // network is done.
    void record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const Controls& controls, bool reset,
                VkQueryPool queries, uint32_t query);
    // Says that the frame record() recorded last was submitted, so that the
    // next frame reads the motion history it writes. A frame that is
    // recorded and not submitted leaves the history as it was.
    void submitted();

private:
    // The network's kernels' pipelines, then the runtime's own.
    enum Adapter : size_t { kAlpha = size_t(Kernel::kCount), kStages, kLuma, kFlow, kPre, kPreNh, kPost, kPipelines };
    static constexpr size_t kAdapters = kPipelines - kAlpha;
    static constexpr uint32_t kLevels = 4; // the motion estimate's pyramid
    static constexpr uint32_t kMaxPasses = 16;

    // Every object the runtime owns.
    struct Objects {
        Buffer arena, weights, params;
        // The network's input and answer, its second output, the first
        // pass's input for later passes, the pass stages' scratch, the
        // frame's image that blits convert through, and the motion history's
        // luma pyramids, flow, histories, depth and one history a pass.
        Image input, answer, second, shown, scratch, frame;
        Image luma[2][kLevels], flow[kLevels], history[2], depth, history_store[kMaxPasses];
        VkSampler nearest, linear;
        Pipeline pipelines[kPipelines];
        VkDescriptorPool pool;
        // Each kernel's set, and the runtime's own pipelines' by what they
        // differ in: the answer or the scratch, the parity and level, and
        // the history.
        VkDescriptorSet kernel_sets[size_t(Kernel::kCount)];
        VkDescriptorSet alpha_sets[2], stage_sets[2], luma_sets[2][kLevels], flow_sets[2][kLevels];
        VkDescriptorSet pre_sets[2], post_sets[2];
    };
    // How frames are recorded.
    struct State {
        uint32_t width, height, passes;
        bool motion, stages, rgba8;
        // The proxy is copied into the input, not blitted; the post block
        // stores the answer in the frame's format, which is then copied out;
        // the post block restores the frame's alpha itself.
        bool input_direct, answer_direct, post_alpha;
        // One pass with motion: the post block writes the history the next
        // frame reads, in turn into history[0] and history[1]. More passes
        // with motion: each pass's post block writes its history into the
        // second output, which that pass's history_store keeps.
        bool pingpong, stored;
        uint32_t level_width[kLevels], level_height[kLevels];
    };
    // The motion history that the next frame reads.
    struct History {
        bool latch;
        uint32_t parity, current;
        // The pre block's noise seed for the next frame, unless that frame
        // starts the history over.
        uint32_t seed;
    };

    Device device_;
    Objects objects_{};
    State state_{};
    // The history after the last frame submitted, and after the last frame
    // recorded.
    History history_{}, recorded_{};
    // A frame submitted since the last build or reshape moved the images into
    // their layouts.
    bool settled_ = false;
    std::vector<Step> steps_;
    std::vector<uint32_t> push_;

    // The commands of a build, submitted once.
    struct Setup;

    explicit Runtime(const Device& device) : device_(device) {}
    Result<void> make(const VulkanPaths& paths, const Shape& shape, Plan& plan);
    // The pre block with motion history: kPreNh when the plan's pre block is
    // without the upper clamp, else kPre.
    size_t temporal_pre() const;
    Result<void> adopt(const Shape& shape);
    Result<void> make_images();
    Result<void> make_pipelines(const VulkanPaths& paths);
    Result<void> make_sets();
    void settle_images(VkCommandBuffer cmd) const;
    Result<void> begin_setup(Setup& setup) const;
    Result<void> upload(const Plan& plan, const Model& model, Setup& setup) const;
    Result<void> end_setup(Setup& setup);
    std::string described() const;
    void dispatch(VkCommandBuffer cmd, size_t pipeline, VkDescriptorSet set, uint32_t x, uint32_t y,
                  const void* push, uint32_t bytes, uint32_t z = 1) const;
    void run_step(VkCommandBuffer cmd, const Step& step, size_t pipeline, VkDescriptorSet set,
                  const uint32_t* push) const;
};

} // namespace dlsslop::vulkan
