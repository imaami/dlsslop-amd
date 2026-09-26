// SPDX-License-Identifier: MIT
#pragma once
#include "codec.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
            << ",\"color_preserve\":" << color_preserve
            << ",\"local_structure\":" << local_structure << ",\"sharpness\":" << sharpness << '}';
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

// Owns a descriptor, and with it any flock on the file.
struct Descriptor { int fd = -1; ~Descriptor() { if (fd >= 0) close(fd); } };

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

// PFM is bottom-up RGB float32. Negative scale identifies little-endian data.
// Retain model-domain values, including signed/extended values and NaNs: this
// is evidence, not a preview image, and no display transform is applied.
inline void trace_write_pfm(const std::filesystem::path& file, const float* data,
                            const Geometry& g, unsigned channels)
{
    if (!data || (channels != 3 && channels != 4) || !fits(g))
        throw std::invalid_argument("invalid diagnostic image geometry");
    const std::uint16_t endian = 1;
    const bool little = *reinterpret_cast<const unsigned char*>(&endian) == 1;
    std::ofstream out(file, std::ios::binary);
    out << "PF\n" << g.fit_width << ' ' << g.fit_height << '\n' << (little ? "-1.0\n" : "1.0\n");
    std::vector<float> row(std::size_t(g.fit_width) * 3);
    for (unsigned iy = g.fit_height; iy-- > 0;) {
        const float* source = data + (std::size_t(g.y + iy) * g.width + g.x) * channels;
        for (unsigned x = 0; x < g.fit_width; ++x)
            std::memcpy(row.data() + std::size_t(x) * 3, source + std::size_t(x) * channels, 3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    out.close();
    if (!out) throw std::runtime_error("write diagnostic image: " + file.string());
}

class FrameTrace {
    std::filesystem::path directory_;
    std::string stages_; // Comma-separated JSON stage objects.
    std::string error_;
    bool finished_ = false;
public:
    explicit FrameTrace(const std::filesystem::path& directory) : directory_(directory) {
        if (mkdir(directory_.c_str(), 0700))
            throw std::runtime_error("create diagnostic directory " + directory_.string() + ": " + std::strerror(errno));
    }
    FrameTrace(const FrameTrace&) = delete;
    // A claimed request always gets its summary and marker, even when the
    // worker stops or unwinds first; the client need not wait for a timeout.
    ~FrameTrace() {
        if (!finished_) finish("{}", error_.empty() ? "worker stopped before a traced frame completed" : nullptr);
    }
    // After a failed stage, the frame's later stages are not written.
    void image(const std::string& name, const float* data, const Geometry& g, unsigned channels) noexcept {
        if (!error_.empty()) return;
        try {
            trace_write_pfm(directory_ / (name + ".pfm"), data, g, channels);
            stages_ += (stages_.empty() ? "{\"name\":" : ",{\"name\":") + trace_json_string(name) +
                ",\"file\":" + trace_json_string(name + ".pfm") + "}";
        } catch (const std::exception& error) { error_ = error.what(); }
    }
    // A failure replaces any error a stage recorded.
    void finish(const std::string& metadata, const char* failure = nullptr) noexcept {
        finished_ = true;
        try {
            if (failure) error_ = failure;
            trace_write_text(directory_ / "summary.json", "{\"schema\":1,\"status\":" +
                trace_json_string(error_.empty() ? "complete" : "failed") +
                ",\"encoding\":\"display-encoded network RGB; not linear light\",\"metadata\":" + metadata +
                ",\"error\":" + trace_json_string(error_) + ",\"stages\":[" + stages_ + "]}\n");
            // A watcher on the opt-in root sees this atomic marker even though
            // image/summary creation happens one directory below its watch.
            trace_write_text(directory_.parent_path() / (directory_.filename().string() + ".done"),
                             directory_.filename().string() + "/summary.json\n");
        } catch (const std::exception& error) {
            std::fprintf(stderr, "diagnostic trace incomplete: %s\n", error.what());
        }
    }
};

// Only constructed in explicit --trace-dir mode. A request file contains one
// unique token. Atomic rename claims it once; summaries are committed last.
class TraceRequests {
    std::filesystem::path directory_;
    Descriptor lock_;
public:
    TraceRequests(const std::string& directory, const std::string& shm)
        : directory_(std::filesystem::absolute(directory)) {
        private_directory(directory_, "trace");
        lock_.fd = open((directory_ / ".worker-lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lock_.fd < 0 || flock(lock_.fd, LOCK_EX | LOCK_NB))
            throw std::runtime_error("trace directory is unavailable or owned by another worker");
        trace_write_text(directory_ / "owner.json", "{\"pid\":" + std::to_string(getpid()) +
            ",\"shm\":" + trace_json_string(std::filesystem::absolute(shm).string()) +
            ",\"trace_metadata_schema\":" + std::to_string(kTraceMetadataSchema) + "}\n");
    }
    TraceRequests(const TraceRequests&) = delete;
    ~TraceRequests() {
        std::error_code ignored;
        std::filesystem::remove(directory_ / "owner.json", ignored);
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
