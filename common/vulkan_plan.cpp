// SPDX-License-Identifier: MIT
#include "vulkan_schedule.h"

#include <algorithm>
#include <iterator>
#include <string>

namespace dlsslop::vulkan {

namespace {

// What a layer computes (upstream: its type column, and the "_ds_" and
// "_upsample" of its kernel column): a Swin block, one that downsamples after
// it and one that upsamples before it; the pre and post blocks; the C=512
// split Swin's FFN, its projection, attention and output projections, the
// last of them pooled, and the head after them; the ViT's layers; and the
// decoder's input.
enum class Family : uint8_t {
    kSwin, kSwinDs, kSwinUp, kPre, kPost, kFfwd, kFfwdProj, kQkvAttn, kProj, kProjPool, kFinalHead,
    kVitExpand, kVitContract, kVitQkv, kVitAttention, kVitProjection, kDecoderUp
};
using F = Family;

// A layer of the network (upstream: the records of nr_native_plan_data.hpp,
// from the frame plan and layer descriptor of nvngx_dlssnr 310.8.0): its block
// and layer, its width C and heads, its input and output channels, the blocks
// it reads (-1 for none), for an upsampling layer the row whose extent it
// takes, as it was before that row ran when ORIGINAL, and its window shift in
// tiles.
struct Row {
    uint8_t block, layer;
    Family family;
    uint16_t c;
    uint8_t heads;
    uint16_t ci, co;
    int8_t in0, in1, skip;
    bool original;
    int8_t sx, sy;
};
constexpr Row kRows[] = {
    {0, 0, F::kPre, 3, 0, 3, 32, -1, -1, -1, false, 0, 0},
    {1, 0, F::kSwin, 32, 1, 32, 32, 0, -1, -1, false, 0, 0},
    {2, 0, F::kSwin, 32, 1, 32, 32, 1, -1, -1, false, -1, -1},
    {3, 0, F::kSwin, 32, 1, 32, 32, 2, -1, -1, false, -1, 0},
    {4, 0, F::kSwinDs, 32, 1, 32, 64, 3, -1, -1, false, 0, -1},
    {5, 0, F::kSwin, 64, 2, 64, 64, 4, -1, -1, false, 0, 0},
    {6, 0, F::kSwin, 64, 2, 64, 64, 5, -1, -1, false, -1, -1},
    {7, 0, F::kSwin, 64, 2, 64, 64, 6, -1, -1, false, -1, 0},
    {8, 0, F::kSwinDs, 64, 2, 64, 128, 7, -1, -1, false, 0, -1},
    {9, 0, F::kSwin, 128, 4, 128, 128, 8, -1, -1, false, 0, 0},
    {10, 0, F::kSwin, 128, 4, 128, 128, 9, -1, -1, false, -1, -1},
    {11, 0, F::kSwin, 128, 4, 128, 128, 10, -1, -1, false, -1, 0},
    {12, 0, F::kSwin, 128, 4, 128, 128, 11, -1, -1, false, 0, -1},
    {13, 0, F::kSwin, 128, 4, 128, 128, 12, -1, -1, false, 0, 0},
    {14, 0, F::kSwinDs, 128, 4, 128, 256, 13, -1, -1, false, -1, -1},
    {15, 0, F::kSwin, 256, 8, 256, 256, 14, -1, -1, false, 0, 0},
    {16, 0, F::kSwin, 256, 8, 256, 256, 15, -1, -1, false, -1, -1},
    {17, 0, F::kSwin, 256, 8, 256, 256, 16, -1, -1, false, -1, 0},
    {18, 0, F::kSwin, 256, 8, 256, 256, 17, -1, -1, false, 0, -1},
    {19, 0, F::kSwin, 256, 8, 256, 256, 18, -1, -1, false, 0, 0},
    {20, 0, F::kSwin, 256, 8, 256, 256, 19, -1, -1, false, -1, -1},
    {21, 0, F::kSwin, 256, 8, 256, 256, 20, -1, -1, false, -1, 0},
    {22, 0, F::kSwinDs, 256, 8, 256, 512, 21, -1, -1, false, 0, -1},
    {23, 0, F::kFfwd, 512, 4, 512, 512, 22, -1, -1, false, 0, 0},
    {23, 1, F::kFfwdProj, 512, 4, 512, 512, 23, -1, -1, false, 0, 0},
    {23, 2, F::kQkvAttn, 512, 4, 512, 512, 23, -1, -1, false, 0, 0},
    {23, 3, F::kProj, 512, 8, 512, 512, 23, -1, -1, false, 0, 0},
    {24, 0, F::kFfwd, 512, 4, 512, 512, 23, -1, -1, false, 0, 0},
    {24, 1, F::kFfwdProj, 512, 4, 512, 512, 24, -1, -1, false, 0, 0},
    {24, 2, F::kQkvAttn, 512, 4, 512, 512, 24, -1, -1, false, -1, -1},
    {24, 3, F::kProj, 512, 8, 512, 512, 24, -1, -1, false, 0, 0},
    {25, 0, F::kFfwd, 512, 4, 512, 512, 24, -1, -1, false, 0, 0},
    {25, 1, F::kFfwdProj, 512, 4, 512, 512, 25, -1, -1, false, 0, 0},
    {25, 2, F::kQkvAttn, 512, 4, 512, 512, 25, -1, -1, false, -1, 0},
    {25, 3, F::kProj, 512, 8, 512, 512, 25, -1, -1, false, 0, 0},
    {26, 0, F::kFfwd, 512, 4, 512, 512, 25, -1, -1, false, 0, 0},
    {26, 1, F::kFfwdProj, 512, 4, 512, 512, 26, -1, -1, false, 0, 0},
    {26, 2, F::kQkvAttn, 512, 4, 512, 512, 26, -1, -1, false, 0, -1},
    {26, 3, F::kProj, 512, 8, 512, 512, 26, -1, -1, false, 0, 0},
    {27, 0, F::kFfwd, 512, 4, 512, 512, 26, -1, -1, false, 0, 0},
    {27, 1, F::kFfwdProj, 512, 4, 512, 512, 27, -1, -1, false, 0, 0},
    {27, 2, F::kQkvAttn, 512, 4, 512, 512, 27, -1, -1, false, 0, 0},
    {27, 3, F::kProj, 512, 8, 512, 512, 27, -1, -1, false, 0, 0},
    {28, 0, F::kFfwd, 512, 4, 512, 512, 27, -1, -1, false, 0, 0},
    {28, 1, F::kFfwdProj, 512, 4, 512, 512, 28, -1, -1, false, 0, 0},
    {28, 2, F::kQkvAttn, 512, 4, 512, 512, 28, -1, -1, false, -1, -1},
    {28, 3, F::kProj, 512, 8, 512, 512, 28, -1, -1, false, 0, 0},
    {29, 0, F::kFfwd, 512, 4, 512, 512, 28, -1, -1, false, 0, 0},
    {29, 1, F::kFfwdProj, 512, 4, 512, 512, 29, -1, -1, false, 0, 0},
    {29, 2, F::kQkvAttn, 512, 4, 512, 512, 29, -1, -1, false, -1, 0},
    {29, 3, F::kProj, 512, 8, 512, 512, 29, -1, -1, false, 0, 0},
    {30, 0, F::kFfwd, 512, 4, 512, 512, 29, -1, -1, false, 0, 0},
    {30, 1, F::kFfwdProj, 512, 4, 512, 512, 30, -1, -1, false, 0, 0},
    {30, 2, F::kQkvAttn, 512, 4, 512, 512, 30, -1, -1, false, 0, -1},
    {30, 3, F::kProjPool, 512, 4, 512, 512, 30, -1, -1, false, 0, 0},
    {30, 4, F::kFinalHead, 512, 8, 512, 1024, 30, -1, -1, false, 0, 0},
    {31, 0, F::kVitExpand, 1024, 4, 1024, 4096, 30, -1, -1, false, 0, 0},
    {31, 1, F::kVitContract, 1024, 4, 4096, 1024, 31, -1, -1, false, 0, 0},
    {31, 2, F::kVitQkv, 1024, 4, 1024, 1024, 31, -1, -1, false, 0, 0},
    {31, 3, F::kVitAttention, 1024, 4, 1024, 1024, 31, -1, -1, false, 0, 0},
    {31, 4, F::kVitProjection, 1024, 4, 1024, 1024, 31, -1, -1, false, 0, 0},
    {32, 0, F::kVitExpand, 1024, 4, 1024, 4096, 31, -1, -1, false, 0, 0},
    {32, 1, F::kVitContract, 1024, 4, 4096, 1024, 32, -1, -1, false, 0, 0},
    {32, 2, F::kVitQkv, 1024, 4, 1024, 1024, 32, -1, -1, false, 0, 0},
    {32, 3, F::kVitAttention, 1024, 4, 1024, 1024, 32, -1, -1, false, 0, 0},
    {32, 4, F::kVitProjection, 1024, 4, 1024, 1024, 32, -1, -1, false, 0, 0},
    {33, 0, F::kVitExpand, 1024, 4, 1024, 4096, 32, -1, -1, false, 0, 0},
    {33, 1, F::kVitContract, 1024, 4, 4096, 1024, 33, -1, -1, false, 0, 0},
    {33, 2, F::kVitQkv, 1024, 4, 1024, 1024, 33, -1, -1, false, 0, 0},
    {33, 3, F::kVitAttention, 1024, 4, 1024, 1024, 33, -1, -1, false, 0, 0},
    {33, 4, F::kVitProjection, 1024, 4, 1024, 1024, 33, -1, -1, false, 0, 0},
    {34, 0, F::kVitExpand, 1024, 4, 1024, 4096, 33, -1, -1, false, 0, 0},
    {34, 1, F::kVitContract, 1024, 4, 4096, 1024, 34, -1, -1, false, 0, 0},
    {34, 2, F::kVitQkv, 1024, 4, 1024, 1024, 34, -1, -1, false, 0, 0},
    {34, 3, F::kVitAttention, 1024, 4, 1024, 1024, 34, -1, -1, false, 0, 0},
    {34, 4, F::kVitProjection, 1024, 4, 1024, 1024, 34, -1, -1, false, 0, 0},
    {35, 0, F::kVitExpand, 1024, 4, 1024, 4096, 34, -1, -1, false, 0, 0},
    {35, 1, F::kVitContract, 1024, 4, 4096, 1024, 35, -1, -1, false, 0, 0},
    {35, 2, F::kVitQkv, 1024, 4, 1024, 1024, 35, -1, -1, false, 0, 0},
    {35, 3, F::kVitAttention, 1024, 4, 1024, 1024, 35, -1, -1, false, 0, 0},
    {35, 4, F::kVitProjection, 1024, 4, 1024, 1024, 35, -1, -1, false, 0, 0},
    {36, 0, F::kVitExpand, 1024, 4, 1024, 4096, 35, -1, -1, false, 0, 0},
    {36, 1, F::kVitContract, 1024, 4, 4096, 1024, 36, -1, -1, false, 0, 0},
    {36, 2, F::kVitQkv, 1024, 4, 1024, 1024, 36, -1, -1, false, 0, 0},
    {36, 3, F::kVitAttention, 1024, 4, 1024, 1024, 36, -1, -1, false, 0, 0},
    {36, 4, F::kVitProjection, 1024, 4, 1024, 1024, 36, -1, -1, false, 0, 0},
    {37, 0, F::kVitExpand, 1024, 4, 1024, 4096, 36, -1, -1, false, 0, 0},
    {37, 1, F::kVitContract, 1024, 4, 4096, 1024, 37, -1, -1, false, 0, 0},
    {37, 2, F::kVitQkv, 1024, 4, 1024, 1024, 37, -1, -1, false, 0, 0},
    {37, 3, F::kVitAttention, 1024, 4, 1024, 1024, 37, -1, -1, false, 0, 0},
    {37, 4, F::kVitProjection, 1024, 4, 1024, 1024, 37, -1, -1, false, 0, 0},
    {38, 0, F::kVitExpand, 1024, 4, 1024, 4096, 37, -1, -1, false, 0, 0},
    {38, 1, F::kVitContract, 1024, 4, 4096, 1024, 38, -1, -1, false, 0, 0},
    {38, 2, F::kVitQkv, 1024, 4, 1024, 1024, 38, -1, -1, false, 0, 0},
    {38, 3, F::kVitAttention, 1024, 4, 1024, 1024, 38, -1, -1, false, 0, 0},
    {38, 4, F::kVitProjection, 1024, 4, 1024, 1024, 38, -1, -1, false, 0, 0},
    {39, 0, F::kDecoderUp, 512, 1, 1024, 512, 38, 30, 54, false, 0, 0},
    {40, 0, F::kFfwd, 512, 4, 512, 512, 39, -1, -1, false, 0, 0},
    {40, 1, F::kFfwdProj, 512, 4, 512, 512, 40, -1, -1, false, 0, 0},
    {40, 2, F::kQkvAttn, 512, 4, 512, 512, 40, -1, -1, false, 0, 0},
    {40, 3, F::kProj, 512, 8, 512, 512, 40, -1, -1, false, 0, 0},
    {41, 0, F::kFfwd, 512, 4, 512, 512, 40, -1, -1, false, 0, 0},
    {41, 1, F::kFfwdProj, 512, 4, 512, 512, 41, -1, -1, false, 0, 0},
    {41, 2, F::kQkvAttn, 512, 4, 512, 512, 41, -1, -1, false, -1, -1},
    {41, 3, F::kProj, 512, 8, 512, 512, 41, -1, -1, false, 0, 0},
    {42, 0, F::kFfwd, 512, 4, 512, 512, 41, -1, -1, false, 0, 0},
    {42, 1, F::kFfwdProj, 512, 4, 512, 512, 42, -1, -1, false, 0, 0},
    {42, 2, F::kQkvAttn, 512, 4, 512, 512, 42, -1, -1, false, -1, 0},
    {42, 3, F::kProj, 512, 8, 512, 512, 42, -1, -1, false, 0, 0},
    {43, 0, F::kFfwd, 512, 4, 512, 512, 42, -1, -1, false, 0, 0},
    {43, 1, F::kFfwdProj, 512, 4, 512, 512, 43, -1, -1, false, 0, 0},
    {43, 2, F::kQkvAttn, 512, 4, 512, 512, 43, -1, -1, false, 0, -1},
    {43, 3, F::kProj, 512, 8, 512, 512, 43, -1, -1, false, 0, 0},
    {44, 0, F::kFfwd, 512, 4, 512, 512, 43, -1, -1, false, 0, 0},
    {44, 1, F::kFfwdProj, 512, 4, 512, 512, 44, -1, -1, false, 0, 0},
    {44, 2, F::kQkvAttn, 512, 4, 512, 512, 44, -1, -1, false, 0, 0},
    {44, 3, F::kProj, 512, 8, 512, 512, 44, -1, -1, false, 0, 0},
    {45, 0, F::kFfwd, 512, 4, 512, 512, 44, -1, -1, false, 0, 0},
    {45, 1, F::kFfwdProj, 512, 4, 512, 512, 45, -1, -1, false, 0, 0},
    {45, 2, F::kQkvAttn, 512, 4, 512, 512, 45, -1, -1, false, -1, -1},
    {45, 3, F::kProj, 512, 8, 512, 512, 45, -1, -1, false, 0, 0},
    {46, 0, F::kFfwd, 512, 4, 512, 512, 45, -1, -1, false, 0, 0},
    {46, 1, F::kFfwdProj, 512, 4, 512, 512, 46, -1, -1, false, 0, 0},
    {46, 2, F::kQkvAttn, 512, 4, 512, 512, 46, -1, -1, false, -1, 0},
    {46, 3, F::kProj, 512, 8, 512, 512, 46, -1, -1, false, 0, 0},
    {47, 0, F::kFfwd, 512, 4, 512, 512, 46, -1, -1, false, 0, 0},
    {47, 1, F::kFfwdProj, 512, 4, 512, 512, 47, -1, -1, false, 0, 0},
    {47, 2, F::kQkvAttn, 512, 4, 512, 512, 47, -1, -1, false, 0, -1},
    {47, 3, F::kProj, 512, 4, 512, 512, 47, -1, -1, false, 0, 0},
    {48, 0, F::kSwinUp, 256, 8, 512, 256, 47, 22, 22, true, 0, 0},
    {49, 0, F::kSwin, 256, 8, 256, 256, 48, -1, -1, false, -1, -1},
    {50, 0, F::kSwin, 256, 8, 256, 256, 49, -1, -1, false, -1, 0},
    {51, 0, F::kSwin, 256, 8, 256, 256, 50, -1, -1, false, 0, -1},
    {52, 0, F::kSwin, 256, 8, 256, 256, 51, -1, -1, false, 0, 0},
    {53, 0, F::kSwin, 256, 8, 256, 256, 52, -1, -1, false, -1, -1},
    {54, 0, F::kSwin, 256, 8, 256, 256, 53, -1, -1, false, -1, 0},
    {55, 0, F::kSwin, 256, 8, 256, 256, 54, -1, -1, false, 0, -1},
    {56, 0, F::kSwinUp, 128, 4, 256, 128, 55, 14, 14, true, -1, 0},
    {57, 0, F::kSwin, 128, 4, 128, 128, 56, -1, -1, false, 0, -1},
    {58, 0, F::kSwin, 128, 4, 128, 128, 57, -1, -1, false, 0, 0},
    {59, 0, F::kSwin, 128, 4, 128, 128, 58, -1, -1, false, -1, -1},
    {60, 0, F::kSwin, 128, 4, 128, 128, 59, -1, -1, false, -1, 0},
    {61, 0, F::kSwin, 128, 4, 128, 128, 60, -1, -1, false, 0, -1},
    {62, 0, F::kSwinUp, 64, 2, 128, 64, 61, 8, 8, true, 0, 0},
    {63, 0, F::kSwin, 64, 2, 64, 64, 62, -1, -1, false, -1, -1},
    {64, 0, F::kSwin, 64, 2, 64, 64, 63, -1, -1, false, -1, 0},
    {65, 0, F::kSwin, 64, 2, 64, 64, 64, -1, -1, false, 0, -1},
    {66, 0, F::kSwinUp, 32, 1, 64, 32, 65, 4, 4, true, 0, 0},
    {67, 0, F::kSwin, 32, 1, 32, 32, 66, -1, -1, false, -1, -1},
    {68, 0, F::kSwin, 32, 1, 32, 32, 67, -1, -1, false, -1, 0},
    {69, 0, F::kSwin, 32, 1, 32, 32, 68, -1, -1, false, 0, -1},
    {70, 0, F::kPost, 3, 0, 32, 3, 69, 0, -1, false, -1, -1},
};
constexpr size_t kLayers = std::size(kRows);
static_assert(kLayers == 152);

// The network's blocks: 0 the pre block to 70 the post block.
constexpr int kBlocks = 71;
static_assert(kKeys == 8 * kBlocks);

uint32_t up(uint32_t n, uint32_t alignment) { return (n + alignment - 1) / alignment * alignment; }
uint64_t align(uint64_t n, uint64_t alignment) { return (n + alignment - 1) / alignment * alignment; }
uint64_t pad4(uint64_t n) { return align(n, 4); }
uint64_t pad8(uint64_t n) { return align(n, 8); }
// The 4x4-pixel tiles a row of PX pixels holds whole (upstream: tiles_of).
uint32_t tiles_of(uint32_t px) { return px / 4; }

// A Swin layer's body by family: at C=32 its kernel, which the pre and post
// blocks replace; at C>=64 the body of its persistent run.
constexpr struct {
    Kernel kernel;
    Body body;
} kSwinBodies[] = {{Kernel::kFswin32, Body::kSwin},
                   {Kernel::kFswinDsp32, Body::kSwinDs},
                   {Kernel::kFswinFusedUp32, Body::kSwinUp},
                   {Kernel::kFswin32, Body::kNone},
                   {Kernel::kFswin32, Body::kNone}};
static_assert(size_t(F::kSwinDs) == 1 && size_t(F::kSwinUp) == 2 && size_t(F::kPost) + 1 == std::size(kSwinBodies));

// Keys of the arena's values (upstream: key_of, lift_key, ups_key, pool_key
// and skip_key): a layer's output, the input lift of the pre block and the
// ViT or the upsampled input of an upsampling block, a pooled output and the
// skip output of a downsampling block.
int key(int block, int layer) { return 8 * block + layer; }
int lift(int block) { return 8 * block + 5; }
int ups(int block) { return 8 * block + 5; }
int pool(int block) { return 8 * block + 6; }
int skip(int block) { return 8 * block + 7; }

// A layer where the walk over the working extent puts it (upstream: the plan
// row make_native_plan writes). W and H keep upstream's order: W is the
// level's height and H its width. Its window grid is GX x GY.
struct Layer {
    Row row;
    uint32_t W, H, gx, gy;
    uint64_t tokens;
};

// The network over a working extent, and what the lowering asks of its layer
// table (upstream: plan, last_layer and reader_tokens).
struct Network {
    uint32_t width, height, work_width, work_height;
    Layer layers[kLayers];
    uint8_t last_layer[kBlocks];
    uint64_t reader_tokens[kBlocks]; // 0 when no layer of another block reads it.

    // Where layer LAYER of BLOCK is.
    const Layer& at(int block, int layer) const
    {
        const Layer* l = std::find_if(std::begin(layers), std::end(layers),
                                      [=](const Layer& l) { return l.row.block == block && l.row.layer == layer; });
        return *l;
    }
    // Its lowering's place.
    int step(int block, int layer) const { return int(&at(block, layer) - layers); }
    // The output of BLOCK's last layer (upstream: blk_out).
    int block_out(int block) const { return key(block, last_layer[block]); }
    // The extent an upsampling layer works at: its encoder skip's (upstream:
    // up_extent), as upstream's (W, H), padded to whole windows at C=32.
    std::pair<uint64_t, uint64_t> up_extent(const Layer& l) const
    {
        const Layer& enc = at(l.row.in1, 0);
        return l.row.co == 32 ? std::pair(pad8(enc.W), pad8(enc.H)) : std::pair<uint64_t, uint64_t>(enc.W, enc.H);
    }
};

// The strict halvings along each axis of the walk over a raster.
struct Halvings {
    uint32_t x, y;
};
// The walk down and up the levels from a WIDTH x HEIGHT raster (upstream:
// walk() in nr_native_plan.cpp), each layer where it puts it with LAYERS.
Halvings walk(uint32_t width, uint32_t height, Layer* layers)
{
    // Upstream's host tensor scalars: w is the height and h the width.
    struct Shape {
        uint32_t w, h;
    };
    Shape current{height, width}, previous = current, before[kLayers], after[kLayers];
    Halvings halvings{};
    int last_block = -1;
    for (size_t i = 0; i < kLayers; ++i) {
        const Row& r = kRows[i];
        if (r.block != last_block) {
            halvings.x += current.h < previous.h;
            halvings.y += current.w < previous.w;
            previous = current;
            last_block = r.block;
        }
        Shape out = current;
        if (r.family == F::kPre || r.family == F::kSwinDs || r.family == F::kFinalHead)
            out = {up((current.w + 1) / 2, 4), up((current.h + 1) / 2, 4)};
        else if (r.family == F::kSwinUp || r.family == F::kDecoderUp) {
            out = r.original ? before[r.skip] : after[r.skip];
            if (r.family == F::kSwinUp && r.co == 32) out = {up(out.w, 8), up(out.h, 8)};
        } else if (r.family == F::kPost)
            out = {height, width};
        if (layers) {
            Shape plan = current;
            if (r.family == F::kFinalHead) plan = out;
            if (r.family == F::kDecoderUp) plan = {up(out.w, 4), up(out.h, 4)};
            if (r.family == F::kPost) plan = {height / 2, width / 2};
            // The window grid pads the extent for a negative shift.
            const Shape grid = r.family == F::kPost || r.family == F::kSwinUp ? out : plan;
            layers[i] = {r, plan.w, plan.h, uint32_t((int64_t(grid.h) - 4 * r.sx + 7) / 8),
                         uint32_t((int64_t(grid.w) - 4 * r.sy + 7) / 8),
                         r.block >= 31 && r.block <= 38 ? uint64_t(plan.w) * plan.h
                                                        : uint64_t(up(plan.w, 8)) * up(plan.h, 8)};
        }
        before[i] = current;
        after[i] = out;
        current = out;
    }
    return halvings;
}

[[gnu::cold, gnu::noinline]] Failure<std::string> refuse(uint32_t width, uint32_t height, const std::string& why)
{
    return {"the network does not take " + std::to_string(width) + "x" + std::to_string(height) + " frames: " + why,
            true};
}

// The network over the working extent of WIDTH x HEIGHT frames (upstream:
// make_native_plan): the raster padded to its halvings, at least 320 pixels
// a side, and one halving's step wider when both sides would divide by four
// of them.
Result<void> place(uint32_t width, uint32_t height, Network& net)
{
    if (!width || !height || width > 16384 || height > 16384)
        return refuse(width, height, "a side is 0 or above 16384");
    const Halvings halvings = walk(width, height, nullptr);
    const uint32_t ax = 1u << halvings.x, ay = 1u << halvings.y;
    net.width = width;
    net.height = height;
    net.work_width = std::max(320u, up(width, ax));
    net.work_height = std::max(320u, up(height, ay));
    if (!(net.work_height % (4 * ay)) && !(net.work_width % (4 * ax))) net.work_width += ax;
    // Upstream's graph needs the input lift fused, which takes whole windows.
    if (net.work_width % 8 || net.work_height % 8)
        return refuse(width, height,
                      "its working extent " + std::to_string(net.work_width) + "x" + std::to_string(net.work_height) +
                          " is not a multiple of 8");
    walk(net.work_width, net.work_height, net.layers);
    std::fill(std::begin(net.last_layer), std::end(net.last_layer), 0);
    std::fill(std::begin(net.reader_tokens), std::end(net.reader_tokens), 0);
    for (const Layer& l : net.layers) {
        net.last_layer[l.row.block] = std::max(net.last_layer[l.row.block], l.row.layer);
        for (int in : {l.row.in0, l.row.in1})
            if (in >= 0 && in != l.row.block && !net.reader_tokens[in]) net.reader_tokens[in] = l.tokens;
    }
    return {};
}

// The width of a layer's output (upstream: shape_of's N).
uint64_t outputs(const Row& r)
{
    switch (r.family) {
    case F::kSwin: return r.c;
    case F::kSwinDs: return 2 * r.c;
    case F::kSwinUp:
    case F::kDecoderUp: return r.co;
    case F::kPre:
    case F::kPost: return 32;
    case F::kFfwd:
    case F::kFfwdProj:
    case F::kQkvAttn:
    case F::kProj:
    case F::kProjPool: return 512;
    case F::kFinalHead:
    case F::kVitContract:
    case F::kVitAttention:
    case F::kVitProjection: return 1024;
    case F::kVitExpand: return 4096;
    case F::kVitQkv: return 3072;
    }
    return 0;
}
bool vit(const Row& r) { return r.family >= F::kVitExpand && r.family <= F::kVitProjection; }
void grow(uint64_t& at, uint64_t bytes) { at = std::max(at, bytes); }

// Each value's size, and the bytes read past it (upstream: the vsize and
// overread tables, nr_graph.cpp:1284-1395).
void size_values(const Network& net, Values& v)
{
    for (const Layer& l : net.layers) {
        const Row& r = l.row;
        const int b = r.block, k = key(b, r.layer);
        // A downsampling layer's output is sized by its reader's tokens.
        const uint64_t out_tokens = net.reader_tokens[b] ? net.reader_tokens[b] : l.tokens / 4;
        switch (r.family) {
        case F::kSwinUp: {
            const auto [w, h] = net.up_extent(l);
            const uint64_t tokens = pad8(w) * pad8(h);
            grow(v.size[pool(b)], l.tokens * r.co * 2);
            grow(v.size[ups(b)], tokens * r.co * (r.co == 32 ? 2 : 1));
            grow(v.size[k], tokens * r.co);
            continue;
        }
        case F::kSwinDs:
            grow(v.size[skip(b)], l.tokens * r.c);
            grow(v.size[pool(b)], l.tokens / 4 * r.c);
            grow(v.size[k], out_tokens * 2 * r.c);
            // The pooled grid is padded to its reader's, which reads the
            // padding as zero.
            grow(v.overread[k], v.size[k]);
            continue;
        case F::kPre:
            grow(v.size[lift(b)], l.tokens * 32 * 2);
            grow(v.size[skip(b)], l.tokens * 32);
            grow(v.size[k], out_tokens * 32);
            continue;
        case F::kPost:
            v.size[pool(b)] = l.tokens * 32;
            grow(v.size[ups(b)], l.tokens * 4 * 32 * 2);
            grow(v.size[k], l.tokens * 4 * 32 * 2);
            break;
        case F::kDecoderUp: {
            const Layer& prev = net.at(r.in0, 4);
            v.size[pool(b)] = align(uint64_t(prev.W) * prev.H, 32) * 512 * 2;
            break;
        }
        case F::kVitQkv: v.size[pool(b)] = align(l.tokens, 32) * 3072 * 2; break;
        case F::kProjPool:
            v.size[pool(b)] = pad4((l.H + 1) / 2) * pad4((l.W + 1) / 2) * 512;
            grow(v.overread[pool(b)], v.size[pool(b)]);
            break;
        case F::kFinalHead:
            // FinalHead reads its plan's tokens off the smaller pooled grid.
            grow(v.overread[pool(b)], l.tokens * 512);
            break;
        default: break;
        }
        if (b == 31 && r.layer == 0) v.size[lift(31)] = align(l.tokens, 32) * 1024;
        if (b == 38 && r.layer == 4) v.size[ups(38)] = align(l.tokens, 32) * 1024;
        // A partial ViT token tile still spans every channel group.
        grow(v.size[k], (vit(r) ? align(l.tokens, 32) : l.tokens) * outputs(r));
    }
}

// Value K at AT, unless PLACED already; where the next value goes.
uint64_t place_value(Values& v, bool (&placed)[kKeys], int k, uint64_t at)
{
    if (placed[k]) return at;
    placed[k] = true;
    v.offset[k] = at;
    return at + align(v.size[k], 256);
}

// Every value its own bytes: the layers' outputs in order, then the others by
// key (upstream: the plain layout, nr_graph.cpp:1524-1536). A key that is not
// sized stays at 0.
void plain_layout(const Network& net, Values& v)
{
    bool placed[kKeys] = {};
    uint64_t at = 0;
    for (const Layer& l : net.layers) at = place_value(v, placed, key(l.row.block, l.row.layer), at);
    for (int k = 0; k < kKeys; ++k)
        if (v.size[k]) at = place_value(v, placed, k, at);
}

// The lowering of the network's layers into dispatches, and of their weights
// into the blob's segments (upstream: the loop of NrSession::build,
// nr_graph.cpp:1727-3254), on the path dlsslopd's configuration takes.
class Lowering {
public:
    Lowering(const Network& net, Values& values, Blob& blob) : net_(net), values_(values), blob_(blob) {}
    Result<void> run();

    std::vector<Dispatch> dispatches;
    NoiseJob noise{};

private:
    const Network& net_;
    Values& values_;
    Blob& blob_;
    int step_ = 0;

    // Value KEY's offset (upstream: voff[KEY]), which the lowering of the
    // current layer names.
    uint64_t at(int key)
    {
        int16_t &first = values_.first[key], &last = values_.last[key];
        first = first < 0 ? int16_t(step_) : std::min(first, int16_t(step_));
        last = std::max(last, int16_t(step_));
        return values_.offset[key];
    }
    // As a push constant.
    uint32_t off(int key) { return uint32_t(at(key)); }
    // The value a layer reads, and the one its residual adds (upstream: x_src
    // and r_src).
    int x_src(const Layer& l) const;
    int r_src(const Layer& l) const;
    // A dispatch of KERNEL for layer L over the groups.
    Dispatch start(const Layer& l, Kernel kernel, uint32_t gx, uint32_t gy, uint32_t gz) const
    {
        Dispatch d{};
        d.kernel = kernel;
        d.after = After::kInvalidate;
        d.block = l.row.block;
        d.layer = l.row.layer;
        d.first = d.last = uint8_t(step_);
        d.groups[0] = gx;
        d.groups[1] = gy;
        d.groups[2] = gz;
        return d;
    }
    // A segment of the blob.
    uint32_t put(uint32_t bytes, Recipe recipe, uint8_t flags, Source source, uint32_t align, uint16_t rows = 0,
                 uint16_t cols = 0, uint32_t index = 0)
    {
        return uint32_t(blob_.put({0, bytes, recipe, flags, source, rows, cols, index}, align));
    }

    Result<void> swin(const Layer&);
    Result<void> decoder(const Layer&);
    void vit_attention(const Layer&);
    void ffwd(const Layer&);
    void attention(const Layer&);
    Result<void> gemm(const Layer&);
};

int Lowering::x_src(const Layer& l) const
{
    const Row& r = l.row;
    if (r.family == F::kFinalHead) return pool(r.block);
    if (r.block == 31 && r.layer == 0) return lift(31);
    if (r.block == 39) return ups(38);
    if (r.layer > 0) return key(r.block, r.layer - 1);
    return net_.block_out(r.in0 >= 0 ? r.in0 : r.block);
}

int Lowering::r_src(const Layer& l) const
{
    const Row& r = l.row;
    if (r.family == F::kProj || r.family == F::kProjPool || r.family == F::kVitProjection) return key(r.block, 1);
    if (r.family == F::kFfwdProj || r.family == F::kVitContract) return x_src(net_.at(r.block, 0));
    return r.layer > 1 ? key(r.block, r.layer - 2) : net_.block_out(r.in0);
}

Result<void> Lowering::run()
{
    // The activation table comes first: the Swin bodies find it at offset 0.
    put(4096, Recipe::kActivations, 0, {}, 256);
    for (step_ = 0; step_ < int(kLayers); ++step_) {
        const Layer& l = net_.layers[step_];
        switch (l.row.family) {
        case F::kSwin:
        case F::kSwinDs:
        case F::kSwinUp:
        case F::kPre:
        case F::kPost: DLSSLOP_TRY(swin(l)); break;
        case F::kDecoderUp: DLSSLOP_TRY(decoder(l)); break;
        case F::kVitAttention: vit_attention(l); break;
        case F::kFfwd: ffwd(l); break;
        case F::kQkvAttn: attention(l); break;
        default: DLSSLOP_TRY(gemm(l)); break;
        }
    }
    return {};
}

// A Swin body, and the pre and post blocks, downsampling and upsampling
// around it (upstream: nr_graph.cpp:1734-2734).
Result<void> Lowering::swin(const Layer& l)
{
    using enum Suffix;
    using enum Segment::Flag;
    const Row& r = l.row;
    const int b = r.block;
    const bool pre = r.family == F::kPre, post = r.family == F::kPost, ds = r.family == F::kSwinDs,
               upsample = r.family == F::kSwinUp;
    const uint32_t c = pre || post ? 32 : r.c, heads = pre || post ? 1 : r.heads, hidden = 4 * c;
    const Directory directory = pre ? Directory::kPreblock : post ? Directory::kPostblock : Directory::kUnpacked;
    auto source = [&](Suffix suffix) { return Source{directory, r.block, r.layer, suffix}; };
    // Every Swin kernel reads its matrices N-paired (weight layout 3).
    auto matrix = [&](Suffix suffix, uint32_t rows, uint32_t cols) {
        return put(rows * cols, Recipe::kMatrix, kRequantise | kNpair, source(suffix), 256, uint16_t(rows),
                   uint16_t(cols));
    };
    PushFSwin p{};
    p.e_off = matrix(kMlpExpand, hidden, c);
    p.ct_off = matrix(kMlpContract, c, heads > 1 ? c : hidden);
    p.qkv_off = matrix(kQkv, 3 * c, c);
    p.op_off = matrix(kAttnOutProj, c, c);
    p.mid_off = heads > 1 ? matrix(kMlpMid, c, hidden / heads) : 0;
    p.b_off = put(heads * 16384, Recipe::kBias, kAffine, source(kAttnPosBias), 16) / 4;
    p.rs_off = put(4 * c, Recipe::kScales, kPadded, source(kResidualScale), 16) / 4;
    p.ars_off = put(4 * c, Recipe::kScales, 0, source(kAttnResidualScale), 16) / 4;
    p.s_off = put(4 * std::max(4u, heads), Recipe::kBytes, kExact, source(kScalarsB), 16) / 4;
    // Unread: math profile 3 points rsd_off at the activation table instead.
    put(32 * c, Recipe::kDiagonal, kPadded, source(kResidualScale), 16);
    p.rsd_off = 0;
    p.ard_off = put(32 * c, Recipe::kDiagonal, 0, source(kAttnResidualScale), 16) / 2;
    p.x_off = off(upsample || post ? ups(b) : pre ? lift(b) : x_src(l));
    p.o_off = off(ds || pre ? skip(b) : key(b, r.layer));
    // An upsampling body's tiles are its output level's: its encoder skip's,
    // or in the post block twice its own extent.
    const uint32_t scale = post ? 2 : 1;
    const auto [out_w, out_h] =
        upsample ? net_.up_extent(l) : std::pair<uint64_t, uint64_t>(scale * l.W, scale * l.H);
    p.tiles_x = tiles_of(uint32_t(out_h));
    p.tiles_y = tiles_of(uint32_t(out_w));
    p.shift = r.sx;
    p.shift_y = r.sy;
    if (pre || ds) {
        p.pool_off = off(pre ? key(b, r.layer) : pool(b));
        p.pool_tiles_x = pre ? tiles_of(l.H / 2) : uint32_t(pad4((l.H + 1) / 2) / 4);
        p.pool_tiles_y = (p.tiles_y + 1) / 2;
    }
    Dispatch d = start(l, kSwinBodies[size_t(r.family)].kernel, l.gx, l.gy, 1);
    if (c != 32) {
        d.kernel = Kernel::kCount;
        d.body = kSwinBodies[size_t(r.family)].body;
        d.c = uint16_t(c);
    }
    store(d, p);
    d.words = sizeof p / 4;

    if (pre) {
        // Upstream's standalone input lift, whose weights stay in the blob;
        // fused into the body, it needs the input's extent in whole windows.
        put(1024, Recipe::kBytes, kExact, source(kInputLift), 16);
        off(lift(b));
        if (l.H % 8 || l.W % 8 || p.tiles_x * 4 != l.H || p.tiles_y * 4 != l.W || p.shift || p.shift_y ||
            l.gx != (l.H + 7) / 8 || l.gy != (l.W + 7) / 8)
            return fail("network plan: the pre block's input lift does not fuse");
        // The noise features the build writes, a pixel of the window grid
        // four binary16s.
        const uint32_t noise_width = l.gx * 8, noise_height = l.gy * 8;
        const uint32_t noise_off = put(noise_width * noise_height * 8, Recipe::kZeros, 0, {}, 256) / 4;
        noise = {noise_off, noise_width, noise_height, 0, 1.0f};
        // Upstream's defaults, which each frame's controls replace.
        const PushPreImage image{put(1024, Recipe::kLift, kExact, source(kInputLift), 16) / 2,
                                 net_.width,
                                 net_.height,
                                 0,
                                 0.0f,
                                 1.0f,
                                 1.0f,
                                 1.0f,
                                 1.0f,
                                 1.0f,
                                 1.0f,
                                 noise_off};
        d.kernel = Kernel::kFswinImagePreds32;
        store(d, image, sizeof p / 4);
        d.words = 32;
        // The 2x2 mean's slots, pooled inside the body.
        off(skip(b));
        off(key(b, r.layer));
        dispatches.push_back(d);
        return {};
    }
    if (post) {
        // Unread: upstream's --post-skip gain, zero.
        put(64, Recipe::kZeros, 0, {}, 16);
        PushUps u{};
        u.p_off = off(x_src(l));
        u.s_off = off(skip(r.in1 >= 0 ? r.in1 : b));
        u.o_off = off(ups(b));
        u.tiles_x = p.tiles_x;
        u.tiles_y = p.tiles_y;
        u.itiles_y = tiles_of(l.W);
        u.itiles_x = tiles_of(l.H);
        u.g_off = put(64, Recipe::kHalf, kExact, source(kSkipGain), 16) / 2;
        put(64, Recipe::kHalf, kExact, source(kMainGain), 16);
        u.mode = 6;
        // The input's own raster, as its producer wrote it, unless it is the
        // body's.
        const Layer& prev = net_.at(r.in0, net_.last_layer[r.in0]);
        const PushUpsView view{off(x_src(l)), off(pool(b)), l.H * l.W, 32, prev.H, prev.W, l.H, l.W};
        if (view.W == view.RW && view.H == view.RH && view.W % 4 == 0 && view.H % 4 == 0 &&
            view.M == view.W * view.H)
            u.p_off = view.x_off;
        else {
            Dispatch v = start(l, Kernel::kUpsView, (view.M * (view.C / 16) + 63) / 64, 1, 1);
            store(v, view);
            v.words = sizeof view / 4;
            dispatches.push_back(v);
            u.p_off = view.o_off;
        }
        const PushImageTail tail{put(1024, Recipe::kBytes, kExact, source(kOutProject), 16) / 2, 1.0f};
        // Upstream's standalone output projection's input.
        off(key(b, r.layer));
        d.kernel = Kernel::kFswinImagePost32;
        store(d, u, sizeof p / 4);
        store(d, tail, (sizeof p + sizeof u) / 4);
        d.words = 32;
        dispatches.push_back(d);
        return {};
    }
    if (upsample) {
        // At C>=64 the resample reads its input in the view's raster, which
        // the producer writes straight into when the view is a copy.
        const bool identity = l.H % 4 == 0 && l.W % 4 == 0 && l.tokens == uint64_t(l.H) * l.W;
        const bool view = c >= 64 && !identity;
        if (view) {
            const uint32_t x = off(x_src(l)), o = off(ups(b)), tokens = uint32_t(l.tokens);
            Dispatch* producer = nullptr;
            if (l.H % 4 == 0 && l.W % 4 == 0 && tokens >= l.H * l.W) {
                for (auto q = dispatches.rbegin(); q != dispatches.rend() && !producer; ++q)
                    if (q->kernel == Kernel::kGemmProj || q->kernel == Kernel::kGemmProjc) {
                        if (q->words == sizeof(PushGemm) / 4 && load<PushGemm>(*q).o_off == x &&
                            load<PushGemm>(*q).M == l.H * l.W)
                            producer = &*q;
                    } else if (q->body == Body::kSwin && q->c == 2 * c && q->words == sizeof(PushFSwin) / 4 &&
                               load<PushFSwin>(*q).o_off == x)
                        producer = &*q;
            }
            if (!producer) return fail("network plan: an upsampling layer's view has no producer to fold into");
            if (producer->kernel != Kernel::kCount) {
                auto g = load<PushGemm>(*producer);
                g.o_off = o;
                store(*producer, g);
            } else {
                auto f = load<PushFSwin>(*producer);
                f.o_off = o;
                store(*producer, f);
            }
            // The view's padding past the producer's tokens is read as zero.
            grow(values_.overread[ups(b)], std::max(values_.size[ups(b)], uint64_t(tokens) * 2 * c));
        }
        const uint32_t w_off =
            put(2 * c * c, Recipe::kMatrix, 0, source(kResample), 256, uint16_t(c), uint16_t(2 * c));
        const uint32_t x = off(view ? ups(b) : x_src(l));
        // The resample's output, which the blend read: fused away.
        off(pool(b));
        PushUps u{};
        u.s_off = off(skip(r.in1 >= 0 ? r.in1 : b));
        u.o_off = off(ups(b));
        u.g_off = put(2 * c, Recipe::kHalf, kGainTail, source(kUpsampleGain), 16) / 2;
        u.tiles_x = p.tiles_x;
        u.tiles_y = p.tiles_y;
        u.itiles_x = tiles_of(l.H);
        u.itiles_y = tiles_of(l.W);
        u.stiles_x = tiles_of(net_.at(r.in1, 0).H);
        // Fused: the body reads the resample's input and projects it itself.
        u.p_off = x;
        store(d, u, sizeof p / 4);
        d.push[(sizeof p + sizeof u) / 4] = w_off;
        d.words = (sizeof p + sizeof u) / 4 + 1;
        dispatches.push_back(d);
        return {};
    }
    if (ds) {
        // The 2x2 mean's slots, the second the resample's input: pooled and
        // projected inside the body.
        off(skip(b));
        off(pool(b));
        PushDsProj q{};
        q.w_off = put(2 * c * c, Recipe::kMatrix, 0, source(kResample), 256, uint16_t(2 * c), uint16_t(c));
        q.o_off = off(key(b, r.layer));
        q.otx = uint32_t(pad4((l.H + 1) / 2) / 4);
        q.crow = q.raster = q.otx * 4;
        q.rows = uint32_t(pad4((l.W + 1) / 2));
        if (c == 32) {
            q.writer_rows = l.W / 2;
            q.raster = l.H / 2;
        } else {
            // The view's padding, zeroed each frame.
            q.clear_x = (l.H + 1) / 2;
            q.clear_y = (l.W + 1) / 2;
        }
        q.n = 2 * c;
        if (p.pool_tiles_x != q.otx) return fail("network plan: a downsampling layer's pooled grid is not its own");
        store(d, q, sizeof p / 4);
        d.words = (sizeof p + sizeof q) / 4;
    }
    dispatches.push_back(d);
    return {};
}

// The decoder's input: a projection into binary16, then upsampled with the
// encoder's skip added (upstream: nr_graph.cpp:2735-2799).
Result<void> Lowering::decoder(const Layer& l)
{
    const Row& r = l.row;
    const int b = r.block;
    const uint32_t n = r.co, k = r.ci;
    auto source = [&](Suffix suffix) { return Source{Directory::kSplitSwin, r.block, r.layer, suffix}; };
    PushGemm p{};
    p.w_off = put(n * k, Recipe::kMatrix, Segment::kNpair, source(Suffix::kWeight), 256, uint16_t(n), uint16_t(k));
    p.g_off = put(2 * n, Recipe::kHalf, 0, source(Suffix::kSkipWeight), 16) / 2;
    p.gd_off = put(n / 16 * 512, Recipe::kDiagonal, 0, source(Suffix::kSkipWeight), 16) / 2;
    p.r_off = off(skip(r.in1 >= 0 ? r.in1 : b));
    p.x_off = off(x_src(l));
    off(key(b, r.layer));
    const Layer& prev = net_.at(r.in0, 4);
    p.M = prev.W * prev.H;
    p.N = n;
    p.K = k;
    p.W = l.W;
    p.o_off = uint32_t(at(pool(b)) / 2);
    if (n % kGemmProjNt) return fail("network plan: the decoder's projection does not divide its tile");
    Dispatch d = start(l, Kernel::kGemmVqkvs, (p.M + kGemmProjMt - 1) / kGemmProjMt, n / kGemmProjNt, 1);
    store(d, p);
    d.words = sizeof p / 4;
    dispatches.push_back(d);
    const PushDecoderUps u{p.o_off, off(key(r.in1, 3)), off(key(b, r.layer)), p.g_off, prev.H, l.H, l.W};
    // Whole 4x4 token tiles take the tile form of kDecupsVec 16; others the
    // shader's per-element fallback.
    static_assert(kDecupsVec == 16);
    const uint32_t groups =
        u.OW % 4 == 0 && u.OH % 4 == 0 ? (u.OW * u.OH / 16 * 1024 + 63) / 64 : (u.OW * u.OH * 512 + 63) / 64;
    Dispatch v = start(l, Kernel::kDecUps, groups, 1, 1);
    store(v, u);
    v.words = sizeof u / 4;
    dispatches.push_back(v);
    return {};
}

// The ViT's global attention over its tokens, W x H of them, with the
// temperatures of the QKV layer before it (upstream: nr_graph.cpp:2802-2851).
void Lowering::vit_attention(const Layer& l)
{
    const Row& r = l.row;
    PushVAttn p{};
    p.x_off = off(x_src(l));
    p.o_off = off(key(r.block, r.layer));
    p.tokens = l.W * l.H;
    p.s_off = put(128, Recipe::kBytes, 0, {Directory::kRecords, r.block, uint8_t(r.layer - 1), Suffix::kNone}, 16) / 4;
    // Q and K arrive normalized and scaled; the weights rounded to E4M3.
    p.mode = 4;
    Dispatch d = start(l, Kernel::kVitAttn, (p.tokens + kVattnQt - 1) / kVattnQt, 32, 1);
    store(d, p);
    d.words = sizeof p / 4;
    dispatches.push_back(d);
}

// The C=512 FFN, over the tile grid (upstream: nr_graph.cpp:2852-2927).
void Lowering::ffwd(const Layer& l)
{
    const Row& r = l.row;
    const Source record{Directory::kRecords, r.block, r.layer, Suffix::kNone};
    PushFfwd3 p{};
    p.a_off = put(8 * 64 * 512, Recipe::kFfwd, Segment::kNpair, record, 256, 0, 0, 0);
    p.q0_off = put(8 * 256 * 64, Recipe::kFfwd, Segment::kNpair, record, 256, 0, 0, 1);
    p.q2_off = put(8 * 64 * 256, Recipe::kFfwd, Segment::kNpair, record, 256, 0, 0, 2);
    p.x_off = off(x_src(l));
    p.o_off = off(key(r.block, r.layer));
    p.M = tiles_of(l.H) * tiles_of(l.W) * 16;
    p.C = 512;
    // ffwd3w: two token tiles a subgroup. Workgroups are group-major, eight
    // weight groups times the token units kFfwdWgw a workgroup.
    const bool two = p.M >= kFfwdFm2MinTokens;
    const uint32_t units = (p.M / 16 + (two ? 1 : 0)) / (two ? 2 : 1);
    Dispatch d = start(l, two ? Kernel::kFfwd3w : Kernel::kFfwd3, 8 * ((units + kFfwdWgw - 1) / kFfwdWgw), 1, 1);
    store(d, p);
    d.words = sizeof p / 4;
    dispatches.push_back(d);
}

// The C=512 windowed attention, its 16 heads split kAttentionSplit ways
// (upstream: nr_graph.cpp:2928-2984).
void Lowering::attention(const Layer& l)
{
    const Row& r = l.row;
    auto source = [&](Suffix suffix) { return Source{Directory::kSplitSwin, r.block, r.layer, suffix}; };
    PushAttn p{};
    // Its QKV matrix N-paired, as attn.comp reads it (NR_ATTN_WPAIR).
    p.w_off = put(1536 * 512, Recipe::kMatrix, Segment::kNpair, source(Suffix::kQkv), 256, 1536, 512);
    p.b_off = put(16 * 16384, Recipe::kBias, 0, source(Suffix::kAttnPosBias), 16) / 4;
    p.s_off = put(64, Recipe::kBytes, Segment::kExact, source(Suffix::kTail), 16) / 4;
    p.x_off = off(x_src(l));
    p.o_off = off(key(r.block, r.layer));
    p.C = 512;
    p.wins_x = l.gx;
    p.tiles_x = tiles_of(l.H);
    p.tiles_y = tiles_of(l.W);
    p.shift = r.sx;
    p.shift_y = r.sy;
    Dispatch d = start(l, Kernel::kAttn, l.gx, l.gy, kAttentionSplit);
    store(d, p);
    d.words = sizeof p / 4;
    dispatches.push_back(d);
}

// The GEMMs: the C=512 projections and head, and the ViT's layers
// (upstream: nr_graph.cpp:2985-3254).
Result<void> Lowering::gemm(const Layer& l)
{
    const Row& r = l.row;
    const int b = r.block;
    const Family family = r.family;
    if (b == 31 && r.layer == 0) {
        // FinalHead stores the ViT's input in its raster itself, and the
        // value's life starts there.
        if (l.W % 4 || l.H % 4) return fail("network plan: the ViT's input does not fold into FinalHead");
        auto head = std::find_if(dispatches.rbegin(), dispatches.rend(), [](const Dispatch& d) {
            return d.block == 30 && d.layer == 4 && d.kernel == Kernel::kGemmNores;
        });
        if (head == dispatches.rend()) return fail("network plan: the ViT's input has no FinalHead to fold into");
        auto q = load<PushGemm>(*head);
        const int now = step_;
        step_ = net_.step(30, 4);
        q.p_off = off(lift(31));
        step_ = now;
        q.W = l.H;
        q.rows = l.W;
        store(*head, q);
    }
    const bool split = family <= F::kFinalHead, residual = family == F::kFfwdProj || family == F::kProj ||
                                                          family == F::kProjPool || family == F::kVitContract ||
                                                          family == F::kVitProjection;
    const Directory directory = split ? Directory::kSplitSwin : Directory::kVit;
    auto source = [&](Suffix suffix) { return Source{directory, r.block, r.layer, suffix}; };
    const uint32_t n = uint32_t(outputs(r)),
                   k = family == F::kFinalHead ? 512 : family == F::kVitContract ? 4096 : split ? 512 : 1024;
    // The C=512 layers compute the tile grid, the ViT its tokens.
    const uint32_t m = split ? tiles_of(l.H) * tiles_of(l.W) * 16 : uint32_t(l.tokens);
    PushGemm p{};
    const size_t weight = blob_.segments.size();
    p.w_off = family == F::kVitQkv
                  ? put(n * k, Recipe::kVitQkv, 0, {Directory::kRecords, r.block, r.layer, Suffix::kNone}, 256,
                        uint16_t(n), uint16_t(k))
                  : put(n * k, Recipe::kMatrix, 0, source(Suffix::kWeight), 256, uint16_t(n), uint16_t(k));
    if (residual) {
        p.g_off = put(2 * n, Recipe::kHalf, 0, source(Suffix::kSkipWeight), 16) / 2;
        p.gd_off = put(n / 16 * 512, Recipe::kDiagonal, 0, source(Suffix::kSkipWeight), 16) / 2;
        p.r_off = off(r_src(l));
    }
    p.x_off = off(x_src(l));
    p.o_off = off(key(b, r.layer));
    p.M = m;
    p.N = n;
    p.K = k;
    p.W = l.W;
    // The ViT's products without a residual take the wide tile, the residual
    // projections theirs (gemmprojw's at many ViT tokens), the rest 64x128.
    const bool wide = vit(r) && !residual,
               projection = !wide && residual && family != F::kProjPool && family != F::kVitExpand,
               projw = projection && family == F::kVitContract && m >= kProjwMinTokens;
    const uint32_t mt = wide ? kGemmWideMt : projw ? kGemmProjwMt : projection ? kGemmProjMt : 64,
                   nt = wide ? kGemmWideNt : projw ? kGemmProjwNt : projection ? kGemmProjNt : 128;
    if (n % nt) return fail("network plan: a GEMM's outputs do not divide its tile");
    Kernel kernel = family == F::kVitExpand ? Kernel::kGemmVact
                    : family == F::kVitQkv  ? Kernel::kGemmVqkvNorm
                    : !residual             ? Kernel::kGemmNores
                    : projw                 ? Kernel::kGemmProjw
                    : split                 ? Kernel::kGemmProjc
                                            : Kernel::kGemmProj;
    if (family == F::kProjPool) {
        kernel = Kernel::kGemmPool;
        p.p_off = off(pool(b));
        p.W = l.H;
        p.rows = l.W;
    }
    // Two token tiles, then every output tile, then the next two.
    if (family == F::kVitContract || (family == F::kVitProjection && m < kProjwMinTokens) ||
        (family == F::kVitExpand && m <= 1024))
        p.remap = 2;
    if (family == F::kVitQkv) {
        // Q normalized and scaled in the GEMM's epilogue, with the record's
        // first 128 bytes, its temperatures.
        p.o_off = uint32_t(at(pool(b)) / 2);
        const uint32_t scales =
            put(128, Recipe::kBytes, 0, {Directory::kRecords, r.block, r.layer, Suffix::kNone}, 16) / 4;
        if (n != 3072 || k != 1024) return fail("network plan: the ViT's QKV is not 3072x1024");
        p.o_off = off(key(b, r.layer));
        p.r_off = scales;
    }
    if (b == 38 && r.layer == 4) {
        // The last projection stores in the raster the decoder reads.
        if (kernel != Kernel::kGemmProj || l.W % 4 || l.H % 4)
            return fail("network plan: the ViT's output does not fold into its projection");
        p.o_off = off(ups(38));
        p.W = l.H;
        p.rows = l.W;
        kernel = Kernel::kGemmProjt;
    }
    // Weight layout 3 pairs every GEMM's weights but FinalHead's and ProjPool's.
    if (kernel != Kernel::kGemmNores && kernel != Kernel::kGemmPool) blob_.segments[weight].flags |= Segment::kNpair;
    Dispatch d = start(l, kernel, (m + mt - 1) / mt, n / nt, 1);
    store(d, p);
    d.words = sizeof p / 4;
    dispatches.push_back(d);
    return {};
}

} // namespace

uint64_t Blob::put(Segment segment, uint32_t alignment)
{
    bytes = align(bytes, alignment);
    segment.offset = uint32_t(bytes);
    segments.push_back(segment);
    bytes += segment.bytes;
    return segment.offset;
}

uint64_t Blob::put_words(const uint32_t* words, size_t count)
{
    const uint32_t index = uint32_t(tables.size());
    tables.insert(tables.end(), words, words + count);
    return put({0, uint32_t(count * 4), Recipe::kTable, 0, {}, 0, 0, index}, 16);
}

Result<uint32_t> Blob::word(uint64_t index) const
{
    for (auto s = segments.rbegin(); s != segments.rend(); ++s)
        if (s->recipe == Recipe::kTable && 4 * index >= s->offset && 4 * index < uint64_t(s->offset) + s->bytes)
            return tables[s->index + (4 * index - s->offset) / 4];
    return fail("network plan: a table word is outside the tables");
}

Result<Plan> plan(uint32_t width, uint32_t height, uint64_t storage)
{
    Network net;
    DLSSLOP_TRY(place(width, height, net));
    Values values{};
    std::fill(std::begin(values.first), std::end(values.first), -1);
    std::fill(std::begin(values.last), std::end(values.last), -1);
    size_values(net, values);
    plain_layout(net, values);

    // Upstream's layout probe: the network lowered over every value in its own
    // bytes, for the layers that name each one. Its weights are not read.
    std::vector<Segment> probe_segments;
    std::vector<uint32_t> probe_tables;
    Blob probe_blob{probe_segments, probe_tables, 0};
    Lowering probe(net, values, probe_blob);
    DLSSLOP_TRY(probe.run());
    uint64_t probe_arena = 0;
    bool epoch = false;
    const auto probed = DLSSLOP_TRY(merge(probe.dispatches, probe_blob, probe_arena, epoch));
    if (!epoch) return fail("network plan: no persistent run at C=256 to count frames with");

    Plan p{};
    p.width = width;
    p.height = height;
    p.work_width = net.work_width;
    p.work_height = net.work_height;
    p.values_end = DLSSLOP_TRY(share(probed, epoch, values));
    if (p.values_end > UINT32_MAX)
        return refuse(width, height,
                      "its activation arena of at least " + std::to_string(p.values_end) +
                          " bytes overflows 32-bit offsets");
    // The network lowered again over the shared arena, which must lower the
    // same way.
    Blob blob{p.segments, p.tables, 0};
    Lowering lowering(net, values, blob);
    DLSSLOP_TRY(lowering.run());
    uint64_t arena = p.values_end;
    auto dispatches = DLSSLOP_TRY(merge(lowering.dispatches, blob, arena, epoch));
    if (!std::equal(dispatches.begin(), dispatches.end(), probed.begin(), probed.end(),
                    [](const Dispatch& a, const Dispatch& b) {
                        return a.kernel == b.kernel && a.first == b.first && a.last == b.last;
                    }))
        return fail("network plan: the network lowered differently over the shared arena");
    auto counters = chain(dispatches, blob, arena);
    if (!counters && counters.error().rejected) return refuse(width, height, counters.error().what);
    p.counter_words = DLSSLOP_TRY(std::move(counters));
    if (arena > UINT32_MAX)
        return refuse(width, height,
                      "its activation arena of " + std::to_string(arena) + " bytes overflows 32-bit offsets");
    if (blob.bytes > UINT32_MAX) return fail("network plan: the weight blob overflows 32-bit offsets");
    // The arena and the weights are each one storage buffer, bound whole.
    if (arena > storage || blob.bytes > storage)
        return refuse(width, height,
                      "its activation arena of " + std::to_string(arena) + " bytes or weights of " +
                          std::to_string(blob.bytes) + " bytes exceed the device's storage buffers of " +
                          std::to_string(storage) + " bytes");
    p.arena_bytes = arena;
    p.blob_bytes = uint32_t(blob.bytes);
    p.noise = lowering.noise;

    for (const Dispatch& d : dispatches) {
        if (d.kernel == Kernel::kCount || d.words * 4 > kKernels[size_t(d.kernel)].push)
            return fail("network plan: a dispatch has no kernel, or a push block past its range");
        p.steps.push_back({d.kernel, d.after, d.words, d.block, d.layer, d.first, d.last, uint16_t(p.push.size()),
                           {d.groups[0], d.groups[1], d.groups[2]}});
        p.push.insert(p.push.end(), d.push, d.push + d.words);
        p.chained += d.after == After::kNothing;
    }
    p.steps.back().after = After::kFull;
    for (int k = 0; k < kKeys; ++k)
        if (values.size[k] || values.first[k] >= 0)
            p.values.push_back({uint16_t(k), values.offset[k], values.size[k], values.overread[k]});
    return p;
}

} // namespace dlsslop::vulkan
