// SPDX-License-Identifier: MIT
#include "trace.hpp"

#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace dlsslop {

std::string TraceFrameMetadata::json() const
{
    char text[1024];
    std::snprintf(text, sizeof text,
                  "{\"trace_metadata_schema\":%u,\"frame_seq\":%u,\"control_seq\":%u,\"tuning_seq\":%u,"
                  "\"control_seq_end\":%u,\"tuning_seq_end\":%u,\"held_input\":%u,\"held_input_end\":%u,"
                  "\"source_proxy_hash\":\"%016llx\",\"passes\":%u,\"source_width\":%u,\"source_height\":%u,"
                  "\"network_width\":%u,\"network_height\":%u,\"fit_x\":%u,\"fit_y\":%u,\"fit_width\":%u,"
                  "\"fit_height\":%u,\"fp16_proxy\":%d,\"fp16_feedback\":%d,\"motion\":%d,\"intensity\":%.9g,"
                  "\"local_tone\":%.9g,\"color_preserve\":%.9g,\"local_structure\":%.9g,\"sharpness\":%.9g}",
                  kTraceMetadataSchema, frame_seq, control_seq, tuning_seq, control_seq_end, tuning_seq_end,
                  held_input, held_input_end, static_cast<unsigned long long>(source_proxy_hash), passes,
                  geometry.source_width, geometry.source_height, geometry.width, geometry.height, geometry.x,
                  geometry.y, geometry.fit_width, geometry.fit_height, fp16_proxy, fp16_feedback, motion,
                  double(intensity), double(local_tone), double(color_preserve), double(local_structure),
                  double(sharpness));
    return text;
}

std::string trace_json_string(const std::string& text)
{
    std::string json = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            json += '\\';
            json += char(c);
        } else if (c < 32) {
            char escaped[7];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", unsigned(c));
            json += escaped;
        } else {
            json += char(c);
        }
    }
    return json += '"';
}

bool trace_valid_token(const std::string& token)
{
    if (token.empty() || token.size() > 64) return false;
    for (unsigned char c : token)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
    return token != "request"; // DIR/request is the request file itself
}

Result<void> trace_write_text(const std::string& file, const std::string& text)
{
    const std::string temporary = file + ".tmp";
    if (!write_file(temporary, text) || rename(temporary.c_str(), file.c_str()))
        return fail("write diagnostic file: " + file);
    return {};
}

Result<void> trace_write_pfm(const std::string& file, const float* data, const struct geometry& g, unsigned channels)
{
    if (!data || (channels != 3 && channels != 4) || !fits(g)) return fail("invalid diagnostic image geometry");
    char header[64];
    const int header_bytes = std::snprintf(header, sizeof header, "PF\n%u %u\n%s", g.fit_width, g.fit_height,
                                           std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    std::string pfm(header, size_t(header_bytes));
    std::vector<float> row(std::size_t(g.fit_width) * 3);
    for (unsigned iy = g.fit_height; iy-- > 0;) {
        const float* source = data + (std::size_t(g.y + iy) * g.width + g.x) * channels;
        for (unsigned x = 0; x < g.fit_width; ++x)
            std::memcpy(row.data() + std::size_t(x) * 3, source + std::size_t(x) * channels, 3 * sizeof(float));
        pfm.append(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    if (!write_file(file, pfm)) return fail("write diagnostic image: " + file);
    return {};
}

Result<std::unique_ptr<FrameTrace>> FrameTrace::create(std::string root, std::string token)
{
    const std::string directory = join(root, token);
    if (mkdir(directory.c_str(), 0700))
        return fail("create diagnostic directory " + directory + ": " + std::strerror(errno));
    return std::unique_ptr<FrameTrace>(new FrameTrace(std::move(root), std::move(token)));
}

FrameTrace::~FrameTrace()
{
    if (!finished_) finish("{}", error_.empty() ? "worker stopped before a traced frame completed" : nullptr);
}

void FrameTrace::image(const std::string& name, const float* data, const struct geometry& g, unsigned channels)
{
    if (!error_.empty()) return;
    if (const auto written = trace_write_pfm(join(join(root_, token_), name + ".pfm"), data, g, channels); !written) {
        error_ = written.error().what;
        return;
    }
    stages_ += (stages_.empty() ? "{\"name\":" : ",{\"name\":") + trace_json_string(name) +
               ",\"file\":" + trace_json_string(name + ".pfm") + "}";
}

void FrameTrace::finish(const std::string& metadata, const char* failure)
{
    finished_ = true;
    if (failure) error_ = failure;
    auto written = trace_write_text(join(join(root_, token_), "summary.json"),
                                    "{\"schema\":1,\"status\":" + trace_json_string(error_.empty() ? "complete" : "failed") +
                                        ",\"encoding\":\"display-encoded network RGB; not linear light\",\"metadata\":" +
                                        metadata + ",\"error\":" + trace_json_string(error_) + ",\"stages\":[" + stages_ +
                                        "]}\n");
    // A watcher on the opt-in root sees this atomic marker even though
    // image/summary creation happens one directory below its watch.
    if (written) written = trace_write_text(join(root_, token_ + ".done"), token_ + "/summary.json\n");
    if (!written) std::fprintf(stderr, "diagnostic trace incomplete: %s\n", written.error().what.c_str());
}

Result<TraceRequests> TraceRequests::open(const std::string& path, const std::string& shm)
{
    const std::string directory = absolute(path);
    DLSSLOP_TRY(private_directory(directory, "trace"));
    Descriptor lock(::open(join(directory, ".worker-lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (lock.fd < 0 || flock(lock.fd, LOCK_EX | LOCK_NB))
        return fail("trace directory is unavailable or owned by another worker");
    DLSSLOP_TRY(trace_write_text(join(directory, "owner.json"),
                                 "{\"pid\":" + std::to_string(getpid()) + ",\"shm\":" + trace_json_string(absolute(shm)) +
                                     ",\"trace_metadata_schema\":" + std::to_string(kTraceMetadataSchema) + "}\n"));
    return TraceRequests(directory, std::move(lock));
}

TraceRequests::~TraceRequests()
{
    if (!directory_.empty()) unlink(join(directory_, "owner.json").c_str());
}

Result<std::unique_ptr<FrameTrace>> TraceRequests::take()
{
    const std::string claimed = join(directory_, ".request-" + std::to_string(getpid()));
    if (rename(join(directory_, "request").c_str(), claimed.c_str())) {
        if (errno == ENOENT) return nullptr;
        return fail_errno("claim trace request");
    }
    const Descriptor request(::open(claimed.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    std::remove(claimed.c_str()); // Also clears a stray empty directory at DIR/request
    if (request.fd < 0) return fail("open diagnostic request");
    struct stat st{};
    char bytes[66]{};
    const bool valid = !fstat(request.fd, &st) && S_ISREG(st.st_mode) && st.st_uid == getuid();
    const ssize_t size = valid ? read(request.fd, bytes, sizeof bytes) : -1;
    if (size < 1 || size > 65) return fail("invalid diagnostic request length");
    std::string token(bytes, std::size_t(size));
    if (token.back() == '\n') token.pop_back();
    if (!trace_valid_token(token)) return fail("invalid diagnostic request token");
    return FrameTrace::create(directory_, std::move(token));
}

} // namespace dlsslop
