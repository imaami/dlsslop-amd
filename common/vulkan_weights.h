// The Vulkan network's weights: dlssnr.bin read and packed into the blob that
// the network's kernels read, byte for byte as upstream packs it. A port of
// the weight lowering of DLSSNR-AMD's linux/src/core/nr_graph.cpp and
// tinlayout.hpp (MIT).
// SPDX-License-Identifier: MIT
#pragma once
#include "files.h"
#include "result.h"
#include "vulkan_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dlsslop::vulkan {

// SOURCE's name in the model pack, upstream's:
// DIRECTORY/blockB.layerL.layer.SUFFIX.bin, or DIRECTORY/blockB.layerL.layer.bin
// for a record.
std::string entry_name(const Source&);

// A model pack in linux/package/model-tools' NRMODEL1 format (upstream:
// open_model_pack): "NRMODEL1", a u32 entry count and a u32 that is not read,
// then for each entry a u32 name length of at most 4096, the name, and its
// data's u64 offset and u64 size.
class Model {
public:
    // PATH's index. It is refused when PATH lacks the magic, when its index
    // ends early or holds a name longer than 4096 bytes, when an entry reaches
    // past the end of PATH, and when a name is listed twice; the errors name
    // PATH.
    static Result<Model> open(const std::string& path);
    // SOURCE's bytes, read into the start of SCRATCH, which grows to hold them.
    Result<std::span<uint8_t>> read(const Source&, std::vector<uint8_t>& scratch) const;
    const std::string& path() const { return path_; }

private:
    struct Entry {
        uint64_t offset, size;
        size_t name;
        uint32_t length;
    };
    std::string_view name(const Entry& e) const { return {names_.data() + e.name, e.length}; }
    Descriptor fd_;
    std::string path_, names_;
    // Sorted by name.
    std::vector<Entry> entries_;
};

// SEGMENTS of BLOB, in offset order, made of MODEL's entries; the bytes between
// and after them are zeroed. TABLES hold the words that kTable segments copy.
// An entry must hold at least the bytes that its recipe reads, or exactly
// those with kExact; the errors name MODEL's path and the entry. A segment out
// of order, outside BLOB or of a size that its recipe does not write is the
// plan's error.
Result<void> pack(std::span<const Segment> segments, std::span<const uint32_t> tables, const Model& model,
                  std::span<uint8_t> blob);

// The packing's primitives, exposed for the tests.

// An E4M3 code through upstream's round trip (upstream:
// e4m3_requantise_table): 0x80 (-0) for 0x00 (+0), 0x7f for 0xff (both NaN),
// and every other code as it is.
uint8_t requantise(uint8_t code);
// Binary16 widened to f32 bits as tin::f16_to_f widens it: NaN becomes the
// quiet 0x7fc00000.
uint32_t widen_half(uint16_t half);
// Binary16 widened and narrowed again (upstream: tin::f_to_f16 of
// tin::f16_to_f): NaN becomes 0x7e00, every other value is kept.
uint16_t recode_half(uint16_t half);
// A ROWS x COLS row-major matrix as 16x16 tiles, 256 bytes each (upstream:
// tin::tile_blocked).
void tile_blocked(const uint8_t* matrix, size_t rows, size_t cols, uint8_t* out);
// The same, then each pair of 16-row tiles interleaved in 8-byte runs
// (upstream: pack_matrix in NrSession::build, nr_graph.cpp:3270-3281).
void npair_blocked(const uint8_t* matrix, size_t rows, size_t cols, uint8_t* out);
// The bytes of a C=512 FFN record that hold A[g][row][k], Q0[g][j][row] and
// Q2[g][n][j], upstream's gathers.
size_t ff_a(unsigned g, unsigned row, unsigned k);
size_t ff_q0(unsigned g, unsigned j, unsigned row);
size_t ff_q2(unsigned g, unsigned n, unsigned j);
// The byte of a ViT QKV record that holds row ROW, column K of matrix WHICH
// (0 Q, 1 K, 2 V), upstream's gather.
size_t vit_qkv_weight_byte(size_t which, size_t row, size_t k);
// Which of a head's 4096 biases, in MMA C-fragment order, is bias [I][J]
// (upstream: deswizzle_bias).
size_t bias_source(size_t i, size_t j);
// The activation table (upstream: nr::detail::activation_lut_v1).
std::array<uint8_t, 4096> activation_table();

} // namespace dlsslop::vulkan
