// SPDX-License-Identifier: MIT
#pragma once
#include "codec.hpp"
#include "temporal_math.hpp"
#include "native_kernels.hpp"
#include <vector>

namespace dlsslop {

// GPU optical flow and the network's real temporal RGB input. Flow is measured
// on consecutive original frames; each neural pass owns its own previous output.
class GpuTemporal {
    struct Level {
        struct temporal_extent image{}, grid{};
        void *current = nullptr, *previous = nullptr, *flow = nullptr;
    };
    const NativeKernels& kernels_;
    const hip::Api& api_;
    const hip::Handle stream_;
    std::vector<Level> levels_;
    std::vector<void*> history_;
    void *warped_ = nullptr, *scene_cut_ = nullptr;
    struct geometry geometry_{};
    unsigned quality_ = 0, grid_ = 0, passes_ = 0, completed_ = 0;
    bool configured_ = false, valid_ = false, pending_ = false;

    Result<void> allocate(void*& ptr, std::size_t bytes, const char* what)
    {
        return api_.check(api_.hipMalloc(&ptr, bytes), what);
    }
    void release_buffers() noexcept
    {
        api_.hipStreamSynchronize(stream_);
        for (auto& level : levels_)
            for (void* ptr : {level.current, level.previous, level.flow})
                if (ptr) api_.hipFree(ptr);
        for (void* ptr : history_) if (ptr) api_.hipFree(ptr);
        for (void* ptr : {warped_, scene_cut_}) if (ptr) api_.hipFree(ptr);
        levels_.clear(); history_.clear();
        warped_ = scene_cut_ = nullptr;
        configured_ = valid_ = pending_ = false;
    }
    Result<void> configure(const struct geometry& g, unsigned quality, unsigned grid, unsigned passes)
    {
        if (!g.width || !g.valid_height || g.valid_height > g.height || g.height > 2 * g.valid_height - 2)
            return fail("invalid temporal geometry");
        // Buffer sizes do not depend on the source size or the picture's
        // placement; only a new placement invalidates the history.
        const bool same_size = configured_ && g.width == geometry_.width && g.height == geometry_.height &&
            g.valid_height == geometry_.valid_height && quality == quality_ && grid == grid_ && passes == passes_;
        const bool moved = g.x != geometry_.x || g.y != geometry_.y || g.fit_width != geometry_.fit_width ||
            g.fit_height != geometry_.fit_height;
        geometry_ = g;
        if (same_size) {
            valid_ = valid_ && !moved;
            return {};
        }
        release_buffers();
        quality_ = quality; grid_ = grid; passes_ = passes;
        if (auto allocated = allocate_buffers(g, quality, grid, passes); !allocated) {
            release_buffers();
            return allocated;
        }
        configured_ = true;
        return {};
    }
    Result<void> allocate_buffers(const struct geometry& g, unsigned quality, unsigned grid, unsigned passes)
    {
        const std::size_t pixels = std::size_t(g.width) * g.height;
        DLSSLOP_TRY(allocate(warped_, pixels * 16, "allocate warped temporal history"));
        DLSSLOP_TRY(allocate(scene_cut_, sizeof(unsigned), "allocate temporal scene cut flag"));
        history_.resize(passes, nullptr);
        for (auto& ptr : history_) DLSSLOP_TRY(allocate(ptr, pixels * 12, "allocate per-pass neural history"));
        unsigned width = g.width, height = g.valid_height;
        const unsigned count = 3 + quality, step = 1u << grid;
        levels_.reserve(count);
        for (unsigned i = 0; i < count; ++i) {
            levels_.push_back({{width, height}, {(width + step - 1) / step, (height + step - 1) / step}});
            auto& level = levels_.back();
            DLSSLOP_TRY(allocate(level.current, std::size_t(width) * height * 4, "allocate current luma pyramid"));
            DLSSLOP_TRY(allocate(level.previous, std::size_t(width) * height * 4, "allocate previous luma pyramid"));
            DLSSLOP_TRY(allocate(level.flow, std::size_t(level.grid.width) * level.grid.height * sizeof(struct temporal_flow),
                                 "allocate flow pyramid"));
            if (width < 8 || height < 8) break;
            width = (width + 1) / 2; height = (height + 1) / 2;
        }
        return {};
    }
    struct temporal_warp warp_geometry() const
    {
        return {{geometry_.width, geometry_.valid_height}, levels_[0].grid,
                geometry_.height, 1u << grid_, geometry_.x, geometry_.y,
                geometry_.fit_width, geometry_.fit_height};
    }
public:
    explicit GpuTemporal(const NativeKernels& kernels) : kernels_(kernels), api_(kernels.api), stream_(kernels.stream) {}
    GpuTemporal(const GpuTemporal&) = delete;
    GpuTemporal& operator=(const GpuTemporal&) = delete;
    ~GpuTemporal() { release_buffers(); }
    void reset() noexcept { valid_ = false; pending_ = false; completed_ = 0; }
    // Self-test only, as it waits for the stream: whether the latest begin()
    // found a scene cut.
    Result<bool> cut_rejected()
    {
        if (!valid_) return false;
        unsigned cut = 0;
        DLSSLOP_TRY(api_.check(api_.hipStreamSynchronize(stream_), "complete temporal scene cut"));
        DLSSLOP_TRY(api_.check(api_.hipMemcpy(&cut, scene_cut_, sizeof cut, 2), "read temporal scene cut"));
        return cut != 0;
    }

    // Inputs are float4 raster data in the same encoding as the network. Commands
    // remain on its HIP stream; begin reads rgba there before multi-pass feedback.
    // Motion settings are in the ranges the protocol's ShmMVec* readers return.
    Result<void> begin(void* rgba, const struct geometry& g, unsigned quality, unsigned grid, unsigned passes,
                       bool reset_history = false)
    {
        if (!rgba || pending_) return fail("temporal begin without previous end or input");
        DLSSLOP_TRY(configure(g, quality, grid, passes));
        if (reset_history) reset();
        pending_ = true; completed_ = 0;
        unsigned count = g.width * g.valid_height;
        void* args[] = {&rgba, &levels_[0].current, &count};
        DLSSLOP_TRY(kernels_.launch(kTemporalLuma, count, args));
        for (std::size_t i = 1; i < levels_.size(); ++i) {
            auto& source = levels_[i - 1]; auto& target = levels_[i];
            void* down_args[] = {&source.current, &target.current, &source.image, &target.image};
            DLSSLOP_TRY(kernels_.launch(kTemporalReduce, target.image.width * target.image.height, down_args));
        }
        if (!valid_) return {};
        for (std::size_t i = levels_.size(); i-- > 0;) {
            auto& level = levels_[i];
            const bool coarse = i + 1 < levels_.size();
            void* coarse_flow = coarse ? levels_[i + 1].flow : level.flow;
            struct temporal_search search{level.image, level.grid,
                coarse ? levels_[i + 1].grid : level.grid, 1u << grid_, quality_ + 1,
                quality_ == 2 ? 2u : 1u, unsigned(coarse), unsigned(i == 0)};
            void* flow_args[] = {&level.current, &level.previous, &coarse_flow, &level.flow, &search};
            DLSSLOP_TRY(kernels_.launch(kTemporalFlow, level.grid.width * level.grid.height, flow_args));
        }
        auto w = warp_geometry();
        void* cut_args[] = {&levels_[0].current, &levels_[0].previous, &levels_[0].flow, &scene_cut_, &w};
        return kernels_.launch(kTemporalCut, 32, cut_args);
    }
    // Call immediately before Enqueue for this pass. The returned image lives
    // until history() is called again; both it and inference share one stream.
    Result<void*> history(unsigned pass, void* current_pass_rgba)
    {
        if (!pending_ || pass != completed_ || pass >= passes_ || !current_pass_rgba)
            return fail("temporal history pass order");
        if (!valid_) return nullptr;
        auto w = warp_geometry();
        void* args[] = {&levels_[0].current, &levels_[0].previous, &history_[pass], &current_pass_rgba,
                        &levels_[0].flow, &warped_, &scene_cut_, &w};
        DLSSLOP_TRY(kernels_.launch(kTemporalWarp, geometry_.width * geometry_.height, args));
        return warped_;
    }
    // Call after history() for this pass: the pass's final RGB answer goes
    // here, where the next frame's warp for the same pass reads it.
    Result<void*> target(unsigned pass)
    {
        if (!pending_ || pass != completed_ || pass >= passes_) return fail("temporal target pass order");
        ++completed_;
        return history_[pass];
    }
    Result<void> end()
    {
        if (!pending_ || completed_ != passes_) return fail("incomplete temporal frame");
        for (auto& level : levels_) std::swap(level.current, level.previous);
        valid_ = true; pending_ = false;
        return {};
    }
};
} // namespace dlsslop
