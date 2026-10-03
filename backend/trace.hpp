// Opt-in diagnostics (--trace-dir): one requested frame's stages as PFM images and a JSON summary.
// SPDX-License-Identifier: MIT
#pragma once
#include "files.hpp"
#include "geometry.hpp"
#include "result.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace dlsslop {
// Independent of the shared-memory protocol and PFM/summary file format.
// Schema 2 guarantees the held-state, control-sequence and source-hash evidence
// needed to compare traces of the same frozen input. Advertise it before a
// diagnostic client changes any live controls.
inline constexpr unsigned kTraceMetadataSchema = 2;

struct TraceFrameMetadata {
    std::uint32_t frame_seq{}, control_seq{}, tuning_seq{};
    std::uint32_t control_seq_end{}, tuning_seq_end{};
    std::uint32_t held_input{}, held_input_end{};
    std::uint64_t source_proxy_hash{};
    unsigned passes{};
    Geometry geometry{};
    bool fp16_proxy{}, fp16_feedback{}, motion{};
    float intensity{}, local_tone{}, local_structure{}, sharpness{}, color_preserve{};

    std::string json() const;
};

std::string trace_json_string(const std::string& text);
bool trace_valid_token(const std::string& token);
// Replaces FILE atomically, through FILE.tmp.
Result<void> trace_write_text(const std::string& file, const std::string& text);
// PFM is bottom-up RGB float32. Negative scale identifies little-endian data.
// Retain model-domain values, including signed/extended values and NaNs: this
// is evidence, not a preview image, and no display transform is applied.
Result<void> trace_write_pfm(const std::string& file, const float* data, const Geometry& g, unsigned channels);

class FrameTrace {
    std::string root_, token_;
    std::string stages_; // Comma-separated JSON stage objects.
    std::string error_;
    bool finished_ = false;
    FrameTrace(std::string root, std::string token) : root_(std::move(root)), token_(std::move(token)) {}
public:
    // ROOT/TOKEN, created private.
    static Result<std::unique_ptr<FrameTrace>> create(std::string root, std::string token);
    FrameTrace(const FrameTrace&) = delete;
    // A claimed request always gets its summary and marker, even when the
    // worker stops or unwinds first; the client need not wait for a timeout.
    ~FrameTrace();
    // After a failed stage, the frame's later stages are not written.
    void image(const std::string& name, const float* data, const Geometry& g, unsigned channels);
    // A failure replaces any error a stage recorded.
    void finish(const std::string& metadata, const char* failure = nullptr);
};

// Only created in explicit --trace-dir mode. A request file contains one
// unique token. Atomic rename claims it once; summaries are committed last.
class TraceRequests {
    std::string directory_;
    Descriptor lock_;
    TraceRequests(std::string directory, Descriptor lock) : directory_(std::move(directory)), lock_(std::move(lock)) {}
public:
    static Result<TraceRequests> open(const std::string& directory, const std::string& shm);
    // The moved-from requests remove no owner file.
    TraceRequests(TraceRequests&& other) noexcept
        : directory_(std::exchange(other.directory_, {})), lock_(std::move(other.lock_))
    {
    }
    ~TraceRequests();
    // The next requested frame's trace, or none without a request.
    Result<std::unique_ptr<FrameTrace>> take();
};
} // namespace dlsslop
