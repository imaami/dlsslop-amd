// SPDX-License-Identifier: MIT
#pragma once
#include "codec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

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
    const char* color_backend = "off";
    float detail{}, color{};

    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(9) << "{\"trace_metadata_schema\":" << kTraceMetadataSchema
            << ",\"frame_seq\":" << frame_seq
            << ",\"control_seq\":" << control_seq << ",\"tuning_seq\":" << tuning_seq
            << ",\"control_seq_end\":" << control_seq_end << ",\"tuning_seq_end\":" << tuning_seq_end
            << ",\"held_input\":" << held_input << ",\"held_input_end\":" << held_input_end
            << ",\"source_proxy_hash\":\"" << std::hex << std::setw(16) << std::setfill('0')
            << source_proxy_hash << std::dec << '"'
            << ",\"passes\":" << passes << ",\"source_width\":" << geometry.source_width
            << ",\"source_height\":" << geometry.source_height << ",\"network_width\":" << geometry.width
            << ",\"network_height\":" << geometry.height
            << ",\"fit_x\":" << geometry.x << ",\"fit_y\":" << geometry.y
            << ",\"fit_width\":" << geometry.fit_width << ",\"fit_height\":" << geometry.fit_height
            << ",\"fp16_proxy\":" << fp16_proxy << ",\"fp16_feedback\":" << fp16_feedback
            << ",\"motion\":" << motion
            << ",\"intensity\":" << intensity << ",\"local_tone\":" << local_tone
            << ",\"color_backend\":\"" << color_backend << "\""
            << ",\"color_preserve\":" << color_preserve
            << ",\"local_structure\":" << local_structure << ",\"sharpness\":" << sharpness
            << ",\"detail\":" << detail << ",\"color\":" << color << '}';
        return out.str();
    }
};

inline std::string trace_json_string(const std::string& text)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
    out << '"';
    return out.str();
}

inline bool trace_valid_token(const std::string& token)
{
    if (token.empty() || token.size() > 64) return false;
    for (unsigned char c : token)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
    return token != "request"; // DIR/request is the request file itself
}

// Creates DIR as 0700, or accepts an existing real directory the current user
// owns with exactly that mode, so no other user can plant or swap files in it.
inline void private_directory(const std::filesystem::path& directory, const char* what)
{
    std::error_code error;
    const bool created = std::filesystem::create_directories(directory, error);
    if (error) throw std::runtime_error(std::string("create ") + what + " directory: " + error.message());
    struct stat st{};
    if (lstat(directory.c_str(), &st) || !S_ISDIR(st.st_mode) || st.st_uid != getuid())
        throw std::runtime_error(std::string(what) + " directory must be owned by the current user and not a symlink");
    if (created ? chmod(directory.c_str(), 0700) != 0 : (st.st_mode & 0777) != 0700)
        throw std::runtime_error(std::string(what) + " directory must be private (mode 0700): " + directory.string());
}

inline void trace_write_text(const std::filesystem::path& file, const std::string& text)
{
    const auto temporary = file.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) throw std::runtime_error("write diagnostic file: " + file.string());
    std::filesystem::rename(temporary, file);
}

struct TraceStats {
    std::array<double, 3> sum{}, minimum{}, maximum{};
    std::array<std::uint64_t, 3> finite{}, at_zero{}, at_one{}, nonfinite{};
    std::uint64_t pixels = 0;
    TraceStats() {
        minimum.fill(std::numeric_limits<double>::infinity());
        maximum.fill(-std::numeric_limits<double>::infinity());
    }
    void add(const float* rgb) {
        ++pixels;
        for (unsigned c = 0; c < 3; ++c) {
            const double value = rgb[c];
            if (!std::isfinite(value)) { ++nonfinite[c]; continue; }
            ++finite[c]; sum[c] += value;
            minimum[c] = std::min(minimum[c], value);
            maximum[c] = std::max(maximum[c], value);
            at_zero[c] += value <= 0; at_one[c] += value >= 1;
        }
    }
    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(17) << "{\"pixels\":" << pixels << ",\"channels\":[";
        for (unsigned c = 0; c < 3; ++c) {
            if (c) out << ',';
            out << "{\"finite\":" << finite[c] << ",\"nonfinite\":" << nonfinite[c]
                << ",\"at_or_below_zero\":" << at_zero[c] << ",\"at_or_above_one\":" << at_one[c]
                << ",\"mean\":";
            if (finite[c]) out << sum[c] / double(finite[c]); else out << "null";
            out << ",\"min\":";
            if (finite[c]) out << minimum[c]; else out << "null";
            out << ",\"max\":";
            if (finite[c]) out << maximum[c]; else out << "null";
            out << '}';
        }
        out << "]}";
        return out.str();
    }
};

// PFM is bottom-up RGB float32. Negative scale identifies little-endian data.
// Retain model-domain values, including signed/extended values and NaNs: this
// is evidence, not a preview image, and no display transform is applied.
inline TraceStats trace_write_pfm(const std::filesystem::path& file, const float* data,
                                  const Geometry& g, unsigned channels)
{
    if (!data || (channels != 3 && channels != 4) || !g.fit_width || !g.fit_height ||
        g.x >= g.width || g.y >= g.height || g.fit_width > g.width - g.x ||
        g.fit_height > g.height - g.y)
        throw std::invalid_argument("invalid diagnostic image geometry");
    const std::uint16_t endian = 1;
    const bool little = *reinterpret_cast<const unsigned char*>(&endian) == 1;
    std::ofstream out(file, std::ios::binary);
    out << "PF\n" << g.fit_width << ' ' << g.fit_height << '\n' << (little ? "-1.0\n" : "1.0\n");
    TraceStats stats;
    std::vector<float> row(std::size_t(g.fit_width) * 3);
    for (unsigned iy = g.fit_height; iy-- > 0;) {
        for (unsigned x = 0; x < g.fit_width; ++x) {
            const float* rgb = data + (std::size_t(g.y + iy) * g.width + g.x + x) * channels;
            std::memcpy(row.data() + std::size_t(x) * 3, rgb, 3 * sizeof(float));
            stats.add(rgb);
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    out.close();
    if (!out) throw std::runtime_error("write diagnostic image: " + file.string());
    return stats;
}

class FrameTrace {
    std::filesystem::path directory_;
    std::vector<std::string> stages_;
    std::string error_;
    bool finished_ = false;
public:
    explicit FrameTrace(const std::filesystem::path& directory) : directory_(directory) {
        if (!std::filesystem::create_directory(directory_))
            throw std::runtime_error("diagnostic token already exists: " + directory_.string());
        if (chmod(directory_.c_str(), 0700))
            throw std::runtime_error("make diagnostic directory private");
    }
    FrameTrace(const FrameTrace&) = delete;
    // A claimed request always gets its summary and marker, even when the
    // worker stops or unwinds first; the client need not wait for a timeout.
    ~FrameTrace() {
        if (finished_) return;
        if (error_.empty()) error_ = "worker stopped before a traced frame completed";
        finish("{}");
    }
    void image(const std::string& name, const float* data, const Geometry& g, unsigned channels) noexcept {
        if (!error_.empty()) return;
        try {
            const auto stats = trace_write_pfm(directory_ / (name + ".pfm"), data, g, channels);
            stages_.push_back("{\"name\":" + trace_json_string(name) + ",\"file\":" +
                trace_json_string(name + ".pfm") + ",\"stats\":" + stats.json() + "}");
        } catch (const std::exception& error) { error_ = error.what(); }
    }
    void finish(const std::string& metadata) noexcept {
        finished_ = true;
        try {
            std::ostringstream out;
            out << "{\"schema\":1,\"status\":" << trace_json_string(error_.empty() ? "complete" : "failed")
                << ",\"encoding\":\"display-encoded network RGB; not linear light\",\"metadata\":"
                << metadata << ",\"error\":" << trace_json_string(error_) << ",\"stages\":[";
            for (std::size_t i = 0; i < stages_.size(); ++i) {
                if (i) out << ',';
                out << stages_[i];
            }
            out << "]}\n";
            trace_write_text(directory_ / "summary.json", out.str());
            // A watcher on the opt-in root sees this atomic marker even though
            // image/summary creation happens one directory below its watch.
            trace_write_text(directory_.parent_path() / (directory_.filename().string() + ".done"),
                             directory_.filename().string() + "/summary.json\n");
        } catch (const std::exception& error) {
            std::fprintf(stderr, "diagnostic trace incomplete: %s\n", error.what());
        }
    }
    void fail(const std::string& error) { error_ = error; }
};

// Only constructed in explicit --trace-dir mode. A request file contains one
// unique token. Atomic rename claims it once; summaries are committed last.
class TraceRequests {
    std::filesystem::path directory_;
    int lock_ = -1;
public:
    TraceRequests(const std::string& directory, const std::string& shm, unsigned version)
        : directory_(std::filesystem::absolute(directory)) {
        private_directory(directory_, "trace");
        lock_ = open((directory_ / ".worker-lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lock_ < 0 || flock(lock_, LOCK_EX | LOCK_NB)) {
            if (lock_ >= 0) close(lock_);
            lock_ = -1;
            throw std::runtime_error("trace directory is unavailable or owned by another worker");
        }
        try {
            trace_write_text(directory_ / "owner.json", "{\"pid\":" + std::to_string(getpid()) +
                ",\"shm\":" + trace_json_string(std::filesystem::absolute(shm).string()) +
                ",\"protocol_version\":" + std::to_string(version) +
                ",\"trace_metadata_schema\":" + std::to_string(kTraceMetadataSchema) + "}\n");
        } catch (...) { close(lock_); lock_ = -1; throw; }
    }
    TraceRequests(const TraceRequests&) = delete;
    ~TraceRequests() {
        if (lock_ >= 0) {
            std::error_code ignored;
            std::filesystem::remove(directory_ / "owner.json", ignored);
            close(lock_);
        }
    }
    std::unique_ptr<FrameTrace> take() {
        const auto claimed = directory_ / (".request-" + std::to_string(getpid()));
        if (rename((directory_ / "request").c_str(), claimed.c_str())) {
            if (errno == ENOENT) return {};
            throw std::runtime_error("claim trace request: " + std::string(std::strerror(errno)));
        }
        const int fd = open(claimed.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        std::remove(claimed.c_str()); // Also clears a stray empty directory at DIR/request
        if (fd < 0) throw std::runtime_error("open diagnostic request");
        struct stat st{};
        char bytes[66]{};
        const bool valid = !fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_uid == getuid();
        const ssize_t size = valid ? read(fd, bytes, sizeof bytes) : -1;
        close(fd);
        if (size < 1 || size > 65) throw std::runtime_error("invalid diagnostic request length");
        std::string token(bytes, std::size_t(size));
        if (token.back() == '\n') token.pop_back();
        if (!trace_valid_token(token)) throw std::runtime_error("invalid diagnostic request token");
        return std::make_unique<FrameTrace>(directory_ / token);
    }
};
} // namespace dlsslop
