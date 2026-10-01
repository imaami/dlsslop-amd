// The Vulkan network's plan: what the host works out from the frame's extent
// before it makes any Vulkan object. Its steps with their push constants, the
// segments of the weight blob and the activation arena's size, all of which
// depend on the extent alone. A port of the production path of DLSSNR-AMD's
// linux/src/core (MIT): the layer table and walk of nr_native_plan.cpp, and
// NrSession::build in nr_graph.cpp up to the device.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace dlsslop::vulkan {

// The shader build's constants that the plan's arithmetic depends on
// (upstream: linux/build/arch/rdna4.sh, and nr_graph.cpp's defaults for those
// it leaves unset). The SPIR-V records them in shader-constants.txt.
inline constexpr uint32_t kGemmWideMt = 64, kGemmWideNt = 256; // The ViT's wide GEMM tile.
inline constexpr uint32_t kGemmProjMt = 32, kGemmProjNt = 128; // The residual projections' tile.
inline constexpr uint32_t kGemmProjwMt = 64, kGemmProjwNt = 128; // gemmprojw's tile,
inline constexpr uint32_t kProjwMinTokens = 768;                 // from this many ViT tokens on.
inline constexpr uint32_t kFfwdWgw = 4;             // ffwd3's subgroups a workgroup.
inline constexpr uint32_t kFfwdFm2MinTokens = 2560; // ffwd3w from this many tokens on.
inline constexpr uint32_t kDecupsVec = 16;          // decups' channels an invocation.
inline constexpr uint32_t kVattnQt = 32;            // vitattn's query tokens a workgroup.
inline constexpr uint32_t kStragglerPercent = 25;   // A persistent run's straggler threshold.
inline constexpr uint32_t kPersistOneMask = 6;      // Widths whose runs of 65 to kPersistOneMax
inline constexpr uint32_t kPersistOneMax = 400;     // windows a layer take a workgroup an item
                                                    // (bit 2 C=128, 4 C=256).
inline constexpr uint32_t kSpinLimit = 4194304;     // A persistent run's polls before it gives up.
inline constexpr uint32_t kArenaCold = 1;           // A value smaller than kArenaColdMax takes only
inline constexpr uint64_t kArenaColdMax = 8 << 20;  // memory freed this many steps before it.
inline constexpr uint32_t kAttentionSplit = 16;     // attn's head groups (its grid's z).

// shader-constants.txt as linux/build/build_network.py writes it from
// pipelines.json's markers: kManifest, then each constant a line of its key
// and value, in this order. The SPIR-V the plan's kernels load must carry it
// exactly; the fixed paths the plan takes (weight layout 3, math profile 3 and
// the fusions) are among them.
inline constexpr const char* kManifest = "nr_shader_manifest v3";
inline constexpr struct {
    const char* key;
    uint32_t value;
} kShaderConstants[] = {{"weight_layout", 3},
                        {"math_profile", 3},
                        {"vattn_qt", kVattnQt},
                        {"gemm_wide_mt", kGemmWideMt},
                        {"gemm_wide_nt", kGemmWideNt},
                        {"qkv_fused_norm", 1},
                        {"ups_fused_mode", 2},
                        {"wide_ups_fused_mode", 2},
                        {"gemm_proj_mt", kGemmProjMt},
                        {"gemm_proj_nt", kGemmProjNt},
                        {"gemm_projw_mt", kGemmProjwMt},
                        {"gemm_projw_nt", kGemmProjwNt},
                        {"ffwd_wgw", kFfwdWgw},
                        {"ffwd_gmajor", 1},
                        {"gemm_remap", 1},
                        {"upsview_vec", 1},
                        {"repack_vec", 1},
                        {"wide_ups_mask", 7},
                        {"persist_df", 1},
                        {"ds_fuse", 15},
                        {"decups_vec", kDecupsVec},
                        {"persist_ds", 7},
                        {"persist_up", 7},
                        {"persist_strag", kStragglerPercent},
                        {"persist_one", kPersistOneMask},
                        {"noise_field", 1},
                        {"ffwd_fm2_min", kFfwdFm2MinTokens},
                        {"post_alpha", 1},
                        {"tchain", 1}};
// The other markers beside the SPIR-V, each a file of one line.
inline constexpr struct {
    const char* file;
    const char* line;
} kMarkers[] = {{"accumulation.txt", "fp32"}, {"coherent-act.txt", "0"}, {"swin-bias-storage.txt", "fp32"}};

// The network's pipelines: those the plan's steps run, and the noise field
// that the build fills the weight blob's noise region with. Each is the
// SPIR-V g_STEM.spv.
enum class Kernel : uint8_t {
    kFswin32, kFswinImagePreds32, kFswinDsp32, kFswinPds64, kFswinPds128, kFswinPds256,
    kFswinPup64, kFswinPup128, kFswinPup256, kFswinFusedUp32, kFswinImagePost32,
    kFfwd3, kFfwd3w, kAttn,
    kGemmProjc, kGemmPool, kGemmNores, kGemmVact, kGemmProj, kGemmProjw, kGemmProjt, kGemmVqkvNorm, kGemmVqkvs,
    kVitAttn, kDecUps, kUpsView, kNoiseField,
    kCount
};
// The images a kernel binds after its buffers: none; the input, sampled; or
// the answer and the second output as storage, then the input, sampled.
enum class Images : uint8_t { kNone, kInput, kOutput };
// A kernel's SPIR-V stem, its storage buffers in binding order ('a' the
// activation arena, 'w' the weight blob), its images, and the bytes of its
// push-constant range (upstream: the Kernel::create calls of NrSession::build,
// nr_graph.cpp:4487-4546).
struct KernelInfo {
    const char* stem;
    const char* buffers;
    Images images;
    uint8_t push;
};
inline constexpr KernelInfo kKernels[] = {
    {"fswin32", "aawwwa", Images::kNone, 84},
    {"fswinimagepreds32", "aawwwa", Images::kInput, 128},
    {"fswindsp32", "aawwwa", Images::kNone, 128},
    {"fswinpds64", "aawwwa", Images::kNone, 36},
    {"fswinpds128", "aawwwa", Images::kNone, 36},
    {"fswinpds256", "aawwwa", Images::kNone, 36},
    {"fswinpup64", "aawwwa", Images::kNone, 36},
    {"fswinpup128", "aawwwa", Images::kNone, 36},
    {"fswinpup256", "aawwwa", Images::kNone, 36},
    {"fswinfusedup32", "aawww", Images::kNone, 128},
    {"fswinimagepost32", "aawww", Images::kOutput, 128},
    {"ffwd3", "aaawwa", Images::kNone, 32},
    {"ffwd3w", "aaawwa", Images::kNone, 32},
    {"attn", "aawwa", Images::kNone, 48},
    {"gemmprojc", "aaawwawa", Images::kNone, 88},
    {"gemmpool", "aaawwawa", Images::kNone, 84},
    {"gemmnores", "aaawwawa", Images::kNone, 84},
    {"gemmvact", "aaawwawa", Images::kNone, 88},
    {"gemmproj", "aaawwawa", Images::kNone, 88},
    {"gemmprojw", "aaawwawa", Images::kNone, 88},
    {"gemmprojt", "aaawwawa", Images::kNone, 88},
    {"gemmvqkvnorm", "aaawwawa", Images::kNone, 88},
    {"gemmvqkvs", "aaawwawa", Images::kNone, 84},
    {"vitattn", "awa", Images::kNone, 24},
    {"decups", "aaw", Images::kNone, 28},
    {"upsview", "a", Images::kNone, 32},
    {"noisefield", "w", Images::kNone, 20},
};
static_assert(std::size(kKernels) == size_t(Kernel::kCount));

// The push blocks, as upstream's host declares them (nr_graph.cpp:144 and
// 776-858) and the GLSL does: the SPIR-V's contract, field for field. A fused
// kernel's block is several of them in a row. A kernel that waits on or
// signals tile counters takes one more word, the u32 index in the weight blob
// of its record, or ~0 for none. Offsets name the activation arena and the
// weight blob in bytes (_off), or in binary16, f32 or u32 elements where
// upstream's comments say so.
struct PushFSwin {
    uint32_t x_off, o_off, e_off, ct_off, qkv_off, op_off, b_off, rs_off, ars_off, s_off;
    uint32_t rsd_off, ard_off, mid_off, tiles_x, tiles_y;
    int32_t shift, shift_y;
    uint32_t pool_off, pool_tiles_x, pool_tiles_y;
};
// A downsampling body's resample projection (fswindsp).
struct PushDsProj {
    uint32_t w_off, o_off, otx, raster, crow, rows, mode, writer_rows, n, clear_x, clear_y;
};
// An upsampling body's blend of the skip (fswinfusedup, fswinimagepost).
struct PushUps {
    uint32_t p_off, s_off, o_off, g_off, tiles_x, tiles_y, itiles_x, itiles_y, mode, stiles_x;
};
// The pre block's image features (fswinimagepreds32).
struct PushPreImage {
    uint32_t lift, source_W, source_H, seed;
    float style, tone, structure, skin, other, noise, constant;
    uint32_t noise_off;
};
// The post block's output projection (fswinimagepost32).
struct PushImageTail {
    uint32_t w_off;
    float intensity;
};
// A persistent run of layers (fswinpds, fswinpup): its PersistRec array,
// sync region, dependency tables and DS or UPS table, as u32 indices.
struct PushPersist {
    uint32_t layers_off, sync_off, n_layers, spin_limit, total_windows, df_off, df_n0, ds_off;
};
// A layer of a persistent run, in the weight blob.
struct PersistRec {
    PushFSwin p;
    uint32_t gx, gy, windows, flag_base;
};
struct PushGemm {
    uint32_t gd_off, x_off, r_off, o_off, d_off, w_off, g_off, M, N, K, p_off, W;
    uint32_t otx, raster, crow, rows, ds_raster, writer_rows, clear_x, clear_y, remap;
};
struct PushFfwd3 {
    uint32_t x_off, o_off, a_off, q0_off, q2_off, M, C;
};
struct PushAttn {
    uint32_t x_off, o_off, w_off, b_off, s_off, C, wins_x, tiles_x, tiles_y;
    int32_t shift, shift_y;
};
struct PushVAttn {
    uint32_t x_off, o_off, tokens, s_off, mode;
};
struct PushDecoderUps {
    uint32_t src, skip, dst, gain_off, IW, OW, OH;
};
// The post block's input when it is not its predecessor's raster (upsview).
struct PushUpsView {
    uint32_t x_off, o_off, M, C, W, H, RW, RH;
};
// The noise field's region of the weight blob (noisefield's push block).
struct NoiseJob {
    uint32_t off, width, height, seed;
    float noise;
};
static_assert(sizeof(PushFSwin) == 80 && offsetof(PushFSwin, shift) == 60 && offsetof(PushFSwin, pool_off) == 68);
static_assert(sizeof(PushDsProj) == 44 && sizeof(PushUps) == 40 && sizeof(PushImageTail) == 8);
static_assert(sizeof(PushPreImage) == 48 && offsetof(PushPreImage, style) == 16);
static_assert(sizeof(PushPersist) == 32 && sizeof(PersistRec) == 96 && offsetof(PersistRec, gx) == 80);
static_assert(sizeof(PushGemm) == 84 && offsetof(PushGemm, p_off) == 40 && offsetof(PushGemm, remap) == 80);
static_assert(sizeof(PushFfwd3) == 28 && sizeof(PushAttn) == 44 && offsetof(PushAttn, shift) == 36);
static_assert(sizeof(PushVAttn) == 20 && sizeof(PushDecoderUps) == 28 && sizeof(PushUpsView) == 32);
static_assert(sizeof(NoiseJob) == 20 && offsetof(NoiseJob, noise) == 16);
// The fused blocks fill their ranges: 128 bytes, with the counter word where
// the kernel takes one.
static_assert(sizeof(PushFSwin) + sizeof(PushPreImage) == 128);
static_assert(sizeof(PushFSwin) + sizeof(PushUps) + sizeof(PushImageTail) == 128);
static_assert(sizeof(PushFSwin) + sizeof(PushDsProj) + 4 == 128);
static_assert(sizeof(PushFSwin) + sizeof(PushUps) + 4 + 4 == 128);

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
// of the put() paths of upstream's NrSession::build (nr_graph.cpp:1700-3259,
// 3525-3600 and 3966-4221) as the network takes it.
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

// What orders a step before the next one: nothing, where tile counters do; a
// barrier that makes the arena's writes visible to the next step (upstream:
// compute to compute, source access 0, destination shader reads and writes,
// which gfx1201's write-through caches make enough); or, after the last step,
// a full one.
enum class After : uint8_t { kNothing, kInvalidate, kFull };
// A dispatch of the network: KERNEL over GROUPS with WORDS push words from
// Plan::push[PUSH] on. It runs the network's layers FIRST to LAST, by their
// place in the layer table, starting with layer LAYER of block BLOCK.
struct Step {
    Kernel kernel;
    After after;
    uint8_t words;
    uint8_t block, layer, first, last;
    uint16_t push;
    uint32_t groups[3];
};
// A value in the activation arena: its bytes and those its readers reach
// past them, which stay zero, at OFFSET (upstream: voff, vsize and overread).
// KEY is 8 * block + slot, the slot a layer's output (its layer), or the
// block's input lift or upsampled input (5), pooled output (6) or skip output
// (7).
struct Value {
    uint16_t key;
    uint64_t offset, bytes, overread;
};

// The network at one frame extent, which the working extent pads.
struct Plan {
    uint32_t width, height, work_width, work_height;
    std::vector<Step> steps;
    std::vector<uint32_t> push;
    // The weight blob, BLOB_BYTES in all: segments in offset order, the gaps
    // between them zero. kTable segments copy TABLES' words.
    std::vector<Segment> segments;
    std::vector<uint32_t> tables;
    uint32_t blob_bytes;
    // The activation arena, zeroed at build and never again while the network
    // lives: the values up to VALUES_END, shared by lifetime; the persistent
    // runs' sync regions; COUNTER_WORDS words of tile counters.
    std::vector<Value> values;
    uint64_t values_end, arena_bytes;
    uint32_t counter_words;
    // The noise field, which the build writes into the weight blob on the GPU.
    NoiseJob noise;
    // The steps that tile counters order instead of a barrier.
    uint16_t chained;
};

// The plan for frames of WIDTH x HEIGHT pixels on a device whose storage
// buffers hold at most STORAGE bytes. A frame the network cannot take is
// rejected: a side of 0 or above 16384, a working extent whose sides are not
// multiples of 8 (a side of 16 pixels or less, where upstream fails), one
// whose arena or tile counters overflow the 32-bit offsets and indices of the
// push constants (upstream wrapped them), and one whose arena or weights
// exceed STORAGE (upstream did not check). The arena's size is known only
// once the network is lowered, so a rejection costs as much as a plan: a
// caller keeps the plan it checked an extent with, and remembers an extent
// that was rejected.
Result<Plan> plan(uint32_t width, uint32_t height, uint64_t storage = UINT64_MAX);

} // namespace dlsslop::vulkan
