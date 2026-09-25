// SPDX-License-Identifier: MIT
#include "../backend/trace.h"
#include <iterator>

namespace {
void require(bool yes, const char* message) {
    if (!yes) throw std::runtime_error(message);
}
std::string read_text(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}

int main()
{
    char temporary[] = "/tmp/dlsslop-amd-trace-test-XXXXXX";
    const char* made = mkdtemp(temporary);
    if (!made) return 1;
    const std::filesystem::path directory(made);
    try {
        dlsslop::Geometry g{2, 2, 4, 4, 4, 4, 1, 1, 2, 2};
        std::vector<float> rgba(4 * 4 * 4, 99.0f);
        for (unsigned y = 1; y <= 2; ++y) for (unsigned x = 1; x <= 2; ++x) {
            const unsigned p = (y * 4 + x) * 4;
            rgba[p] = float(y); rgba[p + 1] = float(x); rgba[p + 2] = -0.5f;
        }
        const auto stats = dlsslop::trace_write_pfm(directory / "test.pfm", rgba.data(), g, 4);
        require(stats.pixels == 4 && stats.sum[0] == 6 && stats.sum[1] == 6 && stats.sum[2] == -2,
                "stats must exclude padding and alpha");
        require(stats.at_one[0] == 4 && stats.at_zero[2] == 4, "extended values must be retained");
        std::ifstream pfm(directory / "test.pfm", std::ios::binary);
        std::string line;
        std::getline(pfm, line); require(line == "PF", "RGB PFM magic");
        std::getline(pfm, line); require(line == "2 2", "PFM fitted viewport dimensions");
        std::getline(pfm, line);
        const std::uint16_t endian = 1;
        require(line == (*reinterpret_cast<const unsigned char*>(&endian) ? "-1.0" : "1.0"), "PFM byte order");
        std::array<float, 12> actual{};
        pfm.read(reinterpret_cast<char*>(actual.data()), sizeof actual);
        const std::array<float, 12> expected{2,1,-.5f, 2,2,-.5f, 1,1,-.5f, 1,2,-.5f};
        require(actual == expected, "PFM rows bottom-up and channels RGB");
        float nonfinite[] = {std::numeric_limits<float>::quiet_NaN(), 0, 1};
        dlsslop::TraceStats invalid; invalid.add(nonfinite);
        require(invalid.nonfinite[0] == 1 && invalid.json().find("\"mean\":null") != std::string::npos,
                "nonfinite statistics must remain valid JSON");
        require(dlsslop::trace_valid_token("stage_01-A") && !dlsslop::trace_valid_token("../escape") &&
                !dlsslop::trace_valid_token("") && !dlsslop::trace_valid_token(std::string(65, 'a')),
                "safe bounded request tokens");
        {
            dlsslop::TraceRequests requests(directory.string(), "/tmp/example-shm", 22);
            require(read_text(directory / "owner.json").find("\"protocol_version\":22") != std::string::npos,
                    "owner discovery metadata");
            require(read_text(directory / "owner.json").find("\"trace_metadata_schema\":2") != std::string::npos,
                    "owner must advertise frozen-input evidence before capture");
            bool locked = false;
            try { dlsslop::TraceRequests duplicate(directory.string(), "/tmp/other", 22); }
            catch (const std::runtime_error&) { locked = true; }
            require(locked, "one worker per trace directory");
            require(!requests.take(), "no request means no trace");
            dlsslop::trace_write_text(directory / "request", "frame_1\n");
            auto frame = requests.take();
            require(frame && !requests.take(), "request consumed exactly once");
            frame->image("pass-01-input", rgba.data(), g, 4);
            require(!std::filesystem::exists(directory / "frame_1/summary.json"), "summary must complete last");
            require(!std::filesystem::exists(directory / "frame_1.done"), "no early root completion marker");
            // Exercise the serializer used by the worker, not a hand-written
            // summary that could silently omit fields required by the client.
            dlsslop::TraceFrameMetadata metadata;
            metadata.frame_seq = 123;
            metadata.control_seq = 11;
            metadata.tuning_seq = 7;
            metadata.control_seq_end = 12;
            metadata.tuning_seq_end = 8;
            metadata.held_input = 1;
            metadata.held_input_end = 0;
            metadata.source_proxy_hash = 0x1234abcd;
            metadata.passes = 3;
            metadata.geometry = g;
            metadata.fp16_proxy = true;
            metadata.fp16_feedback = true;
            metadata.intensity = 0.5f;
            metadata.local_tone = 1;
            metadata.local_structure = 1;
            metadata.detail = 0.75f;
            metadata.color = 0.25f;
            frame->finish(metadata.json());
            require(read_text(directory / "frame_1.done") == "frame_1/summary.json\n", "root marker follows summary commit");
            const auto summary = read_text(directory / "frame_1/summary.json");
            require(summary.find("\"status\":\"complete\"") != std::string::npos &&
                    summary.find("\"frame_seq\":123") != std::string::npos, "summary completion and metadata");
            require(summary.find("\"schema\":1") != std::string::npos,
                    "metadata changes preserve the outer summary file format");
            for (const char* field : {
                    "\"trace_metadata_schema\":2", "\"control_seq\":11", "\"tuning_seq\":7",
                    "\"control_seq_end\":12", "\"tuning_seq_end\":8",
                    "\"held_input\":1", "\"held_input_end\":0",
                    "\"source_proxy_hash\":\"000000001234abcd\"", "\"passes\":3",
                    "\"source_width\":2", "\"source_height\":2",
                    "\"network_width\":4", "\"network_height\":4",
                    "\"fit_x\":1", "\"fit_y\":1", "\"fit_width\":2", "\"fit_height\":2",
                    "\"fp16_proxy\":1", "\"fp16_feedback\":1", "\"motion\":0",
                    "\"intensity\":0.5", "\"local_tone\":1", "\"local_structure\":1",
                    "\"sharpness\":0", "\"detail\":0.75", "\"color\":0.25"})
                require(summary.find(field) != std::string::npos, field);
            dlsslop::trace_write_text(directory / "request", "frame_1\n");
            bool duplicate = false;
            try { requests.take(); } catch (const std::runtime_error&) { duplicate = true; }
            require(duplicate && read_text(directory / "frame_1/summary.json") == summary,
                    "duplicate token must never overwrite evidence");
            dlsslop::trace_write_text(directory / "request", "../escape\n");
            bool traversal = false;
            try { requests.take(); } catch (const std::runtime_error&) { traversal = true; }
            require(traversal, "reject path traversal");
        }
        require(!std::filesystem::exists(directory / "owner.json"), "remove stale owner on shutdown");
        std::filesystem::remove_all(directory);
        std::puts("trace: fitted RGB PFM, statistics, safe requests, ownership and completion passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "trace test: %s\n", error.what());
        std::filesystem::remove_all(directory);
        return 1;
    }
}
