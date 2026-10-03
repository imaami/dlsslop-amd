// The HIP network's weights: each model file read and packed in place into the
// image its kernels read, byte for byte as upstream packs it. A port of the
// production path of lmxxf's packed_weights.h and the weight loaders of
// hip_reference_network.h (MIT).
// SPDX-License-Identifier: MIT
#pragma once
#include "result.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dlsslop::hip {

// How a weight file becomes its image: each is one of upstream's loaders.
// Their other arguments follow from the stem: the channel count, whether a C32
// weight is attention or FFN, and a ViT weight's shape.
enum class Recipe : uint8_t {
    kRaw,             // Weight: the values as read.
    kC32,             // PackedC32Weight
    kDsCast,          // PackedDsWeightCast, at 32 channels
    kDsFrag,          // PackedDsWeightFrag
    kMhFfn,           // PackedFusedMhWeight below the tiled width: PackedMhWeight(ffn)
    kMhAttention,     // PackedMhWeight(attention)
    kMhAttentionDiag, // PackedMhWeightDiag
    kFfnFrag,         // PackedFusedMhWeightFrag
    kQkvFragOnly,     // PackedMhWeightQkvFragOnly
    kSplitMixF16,     // PackedSplitFfnWeightMixHalf
    kProjFrag,        // PackedSplitProjectionFrag
    kQkvFrag,         // PackedMhWeightQkvFrag
    kVitFrag,         // PackedVitWeight, fragment tiles
    kQkvF16Frag,      // PackedVitQkvWeightFrag
    kVitProjFrag,     // PackedVitProjectionFrag
    kDecoderF16r,     // PackedDecoderHalf
};
// Upstream's key suffix for each recipe, a different one each. Logs name a
// packed weight STEM@SUFFIX, and a raw one by its stem alone.
inline constexpr const char* kRecipeSuffix[] = {"", "c32fp8", "ds-cast", "ds-frag", "fp8-g128", "fp8", "fp8-diag",
    "ffn-frag", "qkv-frag-only", "split-mix-f16", "proj-frag", "qkv-frag", "vit-frag", "qkv-f16-frag",
    "vit-proj-frag", "decoder-f16r"};
static_assert(std::size(kRecipeSuffix) == size_t(Recipe::kDecoderF16r) + 1);

// A weight: its file's stem (block23-ffwd-projection) and one of the recipes
// upstream applies to such a stem.
struct WeightSpec {
    char stem[24];
    Recipe recipe;
};

// The values in SPEC's file and the channel count of its stem, which upstream
// passes the recipe's loader (upstream: WeightElements); 0 for a stem that
// upstream does not know.
size_t file_elements(const WeightSpec&);
unsigned channels(const WeightSpec&);
// The size of the image pack() makes of a file of file_elements() values: the
// bytes upstream uploads.
size_t packed_bytes(const WeightSpec&);

// A weight file's values, binary16 ones widened, and the file's path.
struct WeightFile {
    std::string path;
    std::vector<float> values;
};
// ASSETS/STEM.f32, or ASSETS/STEM.f16 when there is no .f32 (upstream:
// ReadWeights), of a stem upstream knows and applies SPEC's recipe to, holding
// exactly file_elements() values. Upstream reads the .f16 whenever the .f32
// does not open, and its ds-cast, ds-frag and decoder loaders accept longer
// files.
Result<WeightFile> read_weights(std::string_view assets, const WeightSpec&);
// FILE's values packed in place into the image upstream uploads, resized where
// upstream resizes them; FILE is one read_weights() accepted for SPEC. Bytes
// that a recipe does not rewrite keep the file's f32 bytes, which upstream
// uploads as they are. Errors name the file and the element.
Result<void> pack(const WeightSpec&, WeightFile& file);

// The packing primitives, exposed for the tests. Those over many values fail
// with "element N: WORDS" for the first value N of the file they reject.

// Why a value has no exact encoding; kFlawWords holds upstream's words for it.
// The encodings of single values return one, so their loops build no words.
enum class Flaw : uint8_t {
    kFp8Nonfinite, kFp8Subnormal, kFp8Inexact, kHalfNonfinite, kHalfOverflow, kHalfInexact, kHalfSubnormal,
    kUngrouped
};
inline constexpr const char* kFlawWords[] = {"nonfinite FP8 matrix weight", "matrix weight not exact FP8 subnormal",
    "matrix weight not exact finite FP8", "nonfinite half weight", "half weight overflow", "weight not exact half",
    "weight not exact half subnormal", "nonzero outside grouped contraction"};
static_assert(std::size(kFlawWords) == size_t(Flaw::kUngrouped) + 1);
// A run of a weight's values: FIRST and COUNT index its floats.
struct Region {
    size_t first, count;
};

// Binary16 widened exactly; NaN payloads are kept (upstream: Half).
float widen_half(uint16_t half);
// E4M3 of an exactly representable value (upstream: ExactWeightFp8).
std::expected<uint8_t, Flaw> exact_fp8(float value);
// Binary16 of an exactly representable value (upstream: ExactWeightHalf).
std::expected<uint16_t, Flaw> exact_half(float value);
// Binary16 rounded to nearest even, subnormals included, overflow made
// infinity and NaN 0x7e00 with its sign (upstream: RoundWeightHalf).
uint16_t round_half(float value);
// VALUE rounded to nearest even E4M3 and saturated at ±448, NaN and infinity
// included (upstream: HostF).
float saturate_fp8(float value);
// The next E4M3 piece of a residual scale, as the device splits it (upstream:
// HostScalePiece).
float scale_piece(float value);
// Each region's values as E4M3, one byte each, from the region's first byte on
// (upstream: PackWeightRegions).
Result<void> pack_fp8_regions(std::vector<float>& values, std::span<const Region> regions);
// The ROWS x COLUMNS bytes from float START on, as fragment-native 512-byte
// tiles (upstream: FragmentPackedMatrix).
void fragment_tiles(std::vector<float>& values, size_t start, size_t rows, size_t columns);
// A C32 FFN weight with its residual scales' pieces appended as diagonal E4M3
// tiles (upstream: AppendC32ResidualDiagonals).
void append_c32_diagonals(std::vector<float>& values);
// An MH attention weight of C channels likewise (upstream:
// AppendMhResidualDiagonals).
void append_mh_diagonals(std::vector<float>& values, unsigned c);
// Whether the FFN weight of C channels contracts in groups only, which its
// kernels assume (upstream: ValidateGroupedMhContract).
Result<void> check_grouped_contract(const std::vector<float>& values, unsigned c);

} // namespace dlsslop::hip
