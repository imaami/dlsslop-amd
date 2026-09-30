// The Vulkan network's plan: what the host works out from the frame's extent
// before it makes any Vulkan object. So far, the segments of the weight blob.
// A port of the production path of DLSSNR-AMD's linux/src/core (MIT).
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>

namespace dlsslop::vulkan {

// A directory of the model pack (upstream: the tree under --unpacked).
enum class Directory : uint8_t { kUnpacked, kPreblock, kPostblock, kSplitSwin, kVit, kRecords };
// What follows a layer's name in the names of its entries; a record has none.
enum class Suffix : uint8_t {
    kNone, kMlpExpand, kMlpContract, kMlpMid, kQkv, kAttnOutProj, kAttnPosBias, kResidualScale,
    kAttnResidualScale, kScalarsB, kResample, kUpsampleGain, kInputLift, kOutProject, kSkipGain, kMainGain,
    kWeight, kSkipWeight, kTail
};
// A model entry: one of a layer's weights, or the layer's record.
struct Source {
    Directory directory;
    uint8_t block, layer;
    Suffix suffix;
};

// How a segment of the weight blob is made from its entry. Each recipe is one
// of the put() paths of upstream's NrSession::build (nr_graph.cpp:1650-3071,
// 3315-3390 and 3643-3824) as the network takes it.
enum class Recipe : uint8_t {
    kZeros,       // Zeros: the noise field that the build fills on the GPU,
                  // and the post block's unread gain.
    kTable,       // Words of the plan's tables, from word INDEX on.
    kActivations, // The 4096-byte activation table (activation_lut_v1).
    kMatrix,      // ROWS x COLS E4M3 codes, tile-blocked.
    kFfwd,        // Part INDEX (0 A, 1 Q0, 2 Q2) of a C=512 FFN record,
                  // gathered and tile-blocked.
    kVitQkv,      // The 3072 x 1024 weight of a ViT QKV record, gathered and
                  // tile-blocked.
    kBias,        // Position biases, 4096 binary16 values a head in MMA
                  // C-fragment order, as f32 in [i][j] order.
    kScales,      // Binary16 values widened to f32.
    kHalf,        // Binary16 values through upstream's f32 round trip.
    kDiagonal,    // 16x16 binary16 blocks with the values on their diagonals.
    kBytes,       // The entry's first bytes.
    kLift,        // The input lift, 32 x 16 binary16, in [channel][k] order.
};

// BYTES of the blob from OFFSET on, made of SOURCE's entry by RECIPE. Counts
// of values follow from BYTES: 16384 bytes a head for kBias, 4 a value for
// kScales, 2 for kHalf and 32 for kDiagonal.
struct Segment {
    // What a recipe does besides, in FLAGS.
    enum Flag : uint8_t {
        kRequantise = 1, // kMatrix: each E4M3 code through upstream's round trip.
        kNpair = 2,      // kMatrix, kFfwd, kVitQkv: pairs of 16-row tiles
                         // interleaved in 8-byte runs (weight layout 3).
        kAffine = 4,     // kBias: math profile 3's exponent affine folded in.
        kPadded = 8,     // kScales, kDiagonal: the values from the ninth on
                         // when the first eight are zero.
        kGainTail = 16,  // kHalf: an entry short of values is completed in
                         // front by the last values of its layer's
                         // residual_scale.
        kExact = 32,     // The entry holds exactly the bytes the recipe reads.
    };
    uint32_t offset, bytes;
    Recipe recipe;
    uint8_t flags;
    Source source;
    uint16_t rows, cols;
    uint32_t index;
};

} // namespace dlsslop::vulkan
