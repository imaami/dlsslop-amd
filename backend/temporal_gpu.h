// SPDX-License-Identifier: MIT
#pragma once
#include "codec.h"
#include "temporal_math.h"
#include "vendor/hip_api.h"
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace dlsslop {

// GPU optical flow and the network's real temporal RGB input. Flow is measured
// on consecutive original frames; each neural pass owns its own previous output.
class GpuTemporal {
    struct Level {
        dlsslop_temporal::Extent image{}, grid{};
        void *current = nullptr, *previous = nullptr, *flow = nullptr;
    };
    hip_probe::Api& api_;
    hip_probe::Handle stream_{}, module_{}, luma_{}, reduce_{}, flow_{}, warp_{}, cut_{};
    std::vector<Level> levels_;
    std::vector<void*> history_;
    void *warped_ = nullptr, *errors_ = nullptr;
    Geometry geometry_{};
    unsigned quality_ = 0, grid_ = 0, units_ = 0, passes_ = 0, completed_ = 0;
    bool configured_ = false, valid_ = false, pending_ = false, cut_rejected_ = false;

    void allocate(void*& ptr, std::size_t bytes, const char* what)
    {
        api_.Check(api_.hipMalloc(&ptr, bytes), what);
    }
    void release_buffers() noexcept
    {
        api_.hipStreamSynchronize(stream_);
        for (auto& level : levels_)
            for (void* ptr : {level.current, level.previous, level.flow})
                if (ptr) api_.hipFree(ptr);
        for (void* ptr : history_) if (ptr) api_.hipFree(ptr);
        for (void* ptr : {warped_, errors_}) if (ptr) api_.hipFree(ptr);
        levels_.clear(); history_.clear();
        warped_ = errors_ = nullptr;
        configured_ = valid_ = pending_ = false;
    }
    void configure(const Geometry& g, unsigned quality, unsigned grid, unsigned units, unsigned passes)
    {
        if (!passes || passes > 30 ||
            !g.width || !g.valid_height || g.valid_height > g.height ||
            g.height > 2 * g.valid_height - 2 || g.valid_width != g.width)
            throw std::invalid_argument("invalid temporal settings or geometry");
        if (configured_ && !std::memcmp(&g, &geometry_, sizeof g) && quality == quality_ &&
            grid == grid_ && units == units_ && passes == passes_) return;
        release_buffers();
        geometry_ = g; quality_ = quality; grid_ = grid; units_ = units; passes_ = passes;
        try {
            const std::size_t pixels = std::size_t(g.width) * g.height;
            allocate(warped_, pixels * 16, "allocate warped temporal history");
            allocate(errors_, 64 * sizeof(float), "allocate temporal cut samples");
            history_.resize(passes, nullptr);
            for (auto& ptr : history_) allocate(ptr, pixels * 12, "allocate per-pass neural history");
            unsigned width = g.width, height = g.valid_height;
            const unsigned count = 3 + quality, step = 1u << grid;
            levels_.reserve(count);
            for (unsigned i = 0; i < count; ++i) {
                levels_.push_back({{width, height}, {(width + step - 1) / step, (height + step - 1) / step}});
                auto& level = levels_.back();
                allocate(level.current, std::size_t(width) * height * 4, "allocate current luma pyramid");
                allocate(level.previous, std::size_t(width) * height * 4, "allocate previous luma pyramid");
                allocate(level.flow, std::size_t(level.grid.width) * level.grid.height * sizeof(dlsslop_temporal::Flow), "allocate flow pyramid");
                if (width < 8 || height < 8) break;
                width = (width + 1) / 2; height = (height + 1) / 2;
            }
            configured_ = true;
        } catch (...) { release_buffers(); throw; }
    }
    dlsslop_temporal::Warp warp_geometry() const
    {
        return {{geometry_.width, geometry_.valid_height}, levels_[0].grid,
                geometry_.height, 1u << grid_, units_, geometry_.x, geometry_.y,
                geometry_.fit_width, geometry_.fit_height};
    }
    void launch(hip_probe::Handle kernel, unsigned count, void** args, const char* what)
    {
        api_.Check(api_.hipModuleLaunchKernel(kernel, (count + 255) / 256, 1, 1,
                   256, 1, 1, 0, stream_, args, nullptr), what);
    }
public:
    GpuTemporal(hip_probe::Api& api, hip_probe::Handle stream, const std::string& path)
        : api_(api), stream_(stream)
    {
        try {
            api_.Check(api_.LoadModule(&module_, path.c_str()), "load temporal module");
            api_.Check(api_.hipModuleGetFunction(&luma_, module_, "dlsslop_temporal_luma"), "find temporal luma kernel");
            api_.Check(api_.hipModuleGetFunction(&reduce_, module_, "dlsslop_temporal_reduce"), "find temporal reduce kernel");
            api_.Check(api_.hipModuleGetFunction(&flow_, module_, "dlsslop_temporal_flow"), "find temporal flow kernel");
            api_.Check(api_.hipModuleGetFunction(&warp_, module_, "dlsslop_temporal_warp"), "find temporal warp kernel");
            api_.Check(api_.hipModuleGetFunction(&cut_, module_, "dlsslop_temporal_cut"), "find temporal cut kernel");
        } catch (...) { if (module_) api_.hipModuleUnload(module_); throw; }
    }
    GpuTemporal(const GpuTemporal&) = delete;
    GpuTemporal& operator=(const GpuTemporal&) = delete;
    ~GpuTemporal() { release_buffers(); if (module_) api_.hipModuleUnload(module_); }
    void reset() noexcept { valid_ = false; pending_ = false; completed_ = 0; }
    bool cut_rejected() const noexcept { return cut_rejected_; }

    // Inputs are float4 raster data in the same encoding as the network. Commands
    // remain on its HIP stream; begin reads rgba there before multi-pass feedback.
    // Motion settings are in the ranges the protocol's ShmMVec* readers return.
    void begin(void* rgba, const Geometry& g, unsigned quality, unsigned grid,
               unsigned units, unsigned passes, bool reset_history = false)
    {
        if (!rgba || pending_) throw std::logic_error("temporal begin without previous end or input");
        configure(g, quality, grid, units, passes);
        if (reset_history) reset();
        pending_ = true; completed_ = 0; cut_rejected_ = false;
        unsigned count = g.width * g.valid_height;
        void* args[] = {&rgba, &levels_[0].current, &count};
        launch(luma_, count, args, "build temporal luma");
        for (std::size_t i = 1; i < levels_.size(); ++i) {
            auto& source = levels_[i - 1]; auto& target = levels_[i];
            void* down_args[] = {&source.current, &target.current, &source.image, &target.image};
            launch(reduce_, target.image.width * target.image.height, down_args, "reduce temporal pyramid");
        }
        if (!valid_) return;
        for (std::size_t i = levels_.size(); i-- > 0;) {
            auto& level = levels_[i];
            const bool coarse = i + 1 < levels_.size();
            void* coarse_flow = coarse ? levels_[i + 1].flow : level.flow;
            dlsslop_temporal::Search search{level.image, level.grid,
                coarse ? levels_[i + 1].grid : level.grid, 1u << grid_, quality_ + 1,
                quality_ == 2 ? 2u : 1u, units_, unsigned(coarse), unsigned(i == 0)};
            void* flow_args[] = {&level.current, &level.previous, &coarse_flow, &level.flow, &search};
            launch(flow_, level.grid.width * level.grid.height, flow_args, "estimate pyramidal optical flow");
        }
        auto w = warp_geometry();
        void* cut_args[] = {&levels_[0].current, &levels_[0].previous, &levels_[0].flow, &errors_, &w};
        launch(cut_, 64, cut_args, "measure temporal scene cut");
        api_.Check(api_.hipStreamSynchronize(stream_), "complete temporal flow");
        std::array<float, 64> errors{};
        api_.Check(api_.hipMemcpy(errors.data(), errors_, sizeof errors, 2), "read temporal cut samples");
        unsigned rejected = 0;
        for (float error : errors) rejected += !(error < .1f);
        if (rejected >= 48) { valid_ = false; cut_rejected_ = true; }
    }
    // Call immediately before Enqueue for this pass. The returned image lives
    // until history() is called again; both it and inference share one stream.
    void* history(unsigned pass, void* current_pass_rgba)
    {
        if (!pending_ || pass != completed_ || pass >= passes_ || !current_pass_rgba)
            throw std::logic_error("temporal history pass order");
        if (!valid_) return nullptr;
        auto w = warp_geometry();
        void* args[] = {&levels_[0].current, &levels_[0].previous, &history_[pass], &current_pass_rgba,
                        &levels_[0].flow, &warped_, &w};
        launch(warp_, geometry_.width * geometry_.height, args, "warp same-pass neural history");
        return warped_;
    }
    void finish_pass(unsigned pass, void* rgb)
    {
        if (!pending_ || pass != completed_ || pass >= passes_ || !rgb)
            throw std::logic_error("temporal finish pass order");
        api_.Check(api_.hipMemcpyAsync(history_[pass], rgb,
            std::size_t(geometry_.width) * geometry_.height * 12, 3, stream_), "retain per-pass neural history");
        ++completed_;
    }
    void end()
    {
        if (!pending_ || completed_ != passes_) throw std::logic_error("incomplete temporal frame");
        for (auto& level : levels_) std::swap(level.current, level.previous);
        valid_ = true; pending_ = false;
    }
};
} // namespace dlsslop
