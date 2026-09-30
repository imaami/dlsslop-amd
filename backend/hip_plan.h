// The HIP network's launch plan: upstream's production RunGraph turned into
// data once per tier and preset, with the buffers its pool settles on. A port
// of the production path of lmxxf's hip_reference_network.h (MIT).
// SPDX-License-Identifier: MIT
#pragma once
#include "hip_weights.h"
#include "result.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace dlsslop::hip {

// The code objects the network launches kernels from.
enum class Module : uint8_t { kMhReference, kDeepReference, kC32Fused, kMhFused, kDeepFast, kMhFast, kCount };
inline constexpr const char* kModuleFiles[] = {"multihead-reference.hsaco", "deep_reference.hsaco",
    "c32_fused_ffn_attention-packed.hsaco", "multihead_fused_attention.hsaco", "deep_fast-packed.hsaco",
    "multihead-fast-padded-wave-packed.hsaco"};
static_assert(std::size(kModuleFiles) == size_t(Module::kCount));

// How upstream's Run() makes a launch's groups of the count it is given.
enum class Grid : uint8_t {
    kGroups,    // The count is of groups: windows.
    kDefault,   // A group per 256 elements, rounded up, whatever the group's size.
    k512,       // A group per 512 elements.
    k1024,      // A group per 1024 elements.
    kFfn,       // A group per 16 tokens of c channels, for 2c threads.
    kPoolGroup, // A group per 16 output tokens of 2c channels, rounded up, for c threads.
    kScalar,    // 64 tokens by 64 channels a group, c from argument 5.
    kDecoder,   // 16 tokens by 16 channels a group, c from argument 9.
};

// Every kernel the network can launch at dlsslopd's tiers.
enum class Kernel : uint8_t {
    kC32Prefix, kC32Mapped, kC32Chain, kC32ChainFinish, kC32ChainFinishDcrop, kC32Post,
    kPool32,
    kFfnC64, kFfnC64Bytein, kFfnC64Mapped, kFfnC64MappedBytein,
    kFfnC128, kFfnC128Bytein, kFfnC128Mapped, kFfnC128MappedBytein,
    kFfnC256, kFfnC256Bytein, kFfnC256Mapped, kFfnC256MappedBytein,
    kAttentionC64, kAttentionC64Bout, kAttentionC128, kAttentionC128Bout, kAttentionC256, kAttentionC256Bout,
    kPoolGroupC64, kPoolGroupC128, kPoolGroupC256, kPoolGroupC512,
    kShiftPack, kSplitMix, kSplitFfn, kSplitProjection, kQkvC512, kAttentionC512, kProjectC512, kProjectScalar,
    kVitGather, kVitPack, kVitExpand, kVitContract, kVitQkv, kVitAttention256, kVitAttention400, kVitAttention640,
    kVitProject, kDecoder, kDecoderByteout,
    kCount
};
struct KernelInfo {
    Module module;
    Grid grid;
    uint16_t threads;
    const char* name;
};
inline constexpr KernelInfo kKernels[] = {
    {Module::kC32Fused, Grid::kGroups, 128, "c32_fast_ffn_attention_fused_half_prefix_finish_main8"},
    {Module::kC32Fused, Grid::kGroups, 128, "c32_fast_ffn_attention_fused_half_mapped"},
    {Module::kC32Fused, Grid::kGroups, 128, "c32_fast_ffn_attention_fused_half_chain"},
    {Module::kC32Fused, Grid::kGroups, 128, "c32_fast_ffn_attention_fused_half_chain_finish"},
    {Module::kC32Fused, Grid::kGroups, 128, "c32_fast_ffn_attention_fused_half_chain_finish_dcrop"},
    {Module::kC32Fused, Grid::kGroups, 128, "c32_post_merge_head_half"},
    {Module::kMhFast, Grid::kDefault, 32, "mh_pool_project_production_h16w"},
    {Module::kMhFast, Grid::kFfn, 128, "mh_ffn_fused_c64_project_g128_qkv_fb"},
    {Module::kMhFast, Grid::kFfn, 128, "mh_ffn_fused_c64_project_g128_qkv_bytein_fb"},
    {Module::kMhFast, Grid::kFfn, 128, "mh_ffn_fused_c64_project_mapped_g128_qkv_fb"},
    {Module::kMhFast, Grid::kFfn, 128, "mh_ffn_fused_c64_project_mapped_g128_qkv_bytein_fb"},
    {Module::kMhFast, Grid::kFfn, 256, "mh_ffn_fused_c128_project_g128_qkv_fb"},
    {Module::kMhFast, Grid::kFfn, 256, "mh_ffn_fused_c128_project_g128_qkv_bytein_fb"},
    {Module::kMhFast, Grid::kFfn, 256, "mh_ffn_fused_c128_project_mapped_g128_qkv_fb"},
    {Module::kMhFast, Grid::kFfn, 256, "mh_ffn_fused_c128_project_mapped_g128_qkv_bytein_fb"},
    {Module::kMhFast, Grid::kFfn, 512, "mh_ffn_fused_c256_frag_project_g128_qkv_fb"},
    {Module::kMhFast, Grid::kFfn, 512, "mh_ffn_fused_c256_frag_project_g128_qkv_bytein_fb"},
    {Module::kMhFast, Grid::kFfn, 512, "mh_ffn_fused_c256_frag_project_mapped_g128_qkv_fb"},
    {Module::kMhFast, Grid::kFfn, 512, "mh_ffn_fused_c256_frag_project_mapped_g128_qkv_bytein_fb"},
    {Module::kMhFused, Grid::kGroups, 256, "c64_attention_project_fb_diag"},
    {Module::kMhFused, Grid::kGroups, 256, "c64_attention_project_fb_bout_diag"},
    {Module::kMhFused, Grid::kGroups, 512, "c128_attention_project_fb_diag"},
    {Module::kMhFused, Grid::kGroups, 512, "c128_attention_project_fb_bout_diag"},
    {Module::kMhFused, Grid::kGroups, 512, "c256_attention_project_fb_diag"},
    {Module::kMhFused, Grid::kGroups, 512, "c256_attention_project_fb_bout_diag"},
    {Module::kMhFast, Grid::kPoolGroup, 64, "mh_pool_project_group_c64"},
    {Module::kMhFast, Grid::kPoolGroup, 128, "mh_pool_project_group_c128"},
    {Module::kMhFast, Grid::kPoolGroup, 256, "mh_pool_project_group_c256"},
    {Module::kMhFast, Grid::kPoolGroup, 512, "mh_pool_project_group_c512"},
    {Module::kMhReference, Grid::kDefault, 256, "mh_shift_pack"},
    {Module::kDeepFast, Grid::k1024, 32, "split_mix_blocked_h16w"},
    {Module::kDeepFast, Grid::k1024, 128, "split_ffn_fused_fp8_t8"},
    {Module::kDeepFast, Grid::k1024, 32, "split_projection_frag"},
    {Module::kMhFast, Grid::k1024, 32, "mh_qkv_normalize_frag_c512"},
    {Module::kMhFused, Grid::kGroups, 128, "mh_attention_fused_fp8_out"},
    {Module::kMhFast, Grid::k1024, 32, "mh_attention_project_frag_c512"},
    {Module::kMhFast, Grid::kScalar, 512, "mh_attention_project_fast_scalar_fp8"},
    {Module::kDeepReference, Grid::kDefault, 256, "vit_gather"},
    {Module::kDeepFast, Grid::kDefault, 256, "vit_pack_input"},
    {Module::kDeepFast, Grid::k1024, 32, "vit_expand_blocked_fp8_frag_bytein"},
    {Module::kDeepFast, Grid::k1024, 32, "vit_contract_blocked_fp8_frag"},
    {Module::kDeepFast, Grid::k512, 32, "vit_qkv_project_normalize_fused_f16compact_fp8_frag"},
    {Module::kDeepFast, Grid::kDefault, 32, "vit_attention_fused_256_bytein"},
    {Module::kDeepFast, Grid::kDefault, 32, "vit_attention_fused_400_bytein"},
    {Module::kDeepFast, Grid::kDefault, 32, "vit_attention_fused_640_bytein"},
    {Module::kDeepFast, Grid::kDefault, 32, "vit_project_frag"},
    {Module::kDeepFast, Grid::kDecoder, 32, "decoder_project2x_h16w"},
    {Module::kDeepFast, Grid::kDecoder, 32, "decoder_project2x_h16w_byteout"},
};
static_assert(std::size(kKernels) == size_t(Kernel::kCount));

// A kernel argument. The frame's own ones are known only as it is queued.
struct Arg {
    enum Kind : uint8_t {
        kU32,      // VALUE.
        kF32,      // The float with VALUE's bits.
        kNull,     // A null pointer.
        kTensor,   // Tensor VALUE's buffer in the pool.
        kWeight,   // Plan::weights[VALUE]'s image.
        kGather,   // The ViT gather map, the inverse one when VALUE is 1.
        kRgba,     // The frame's input.
        kHistory,  // The frame's history, or its input when it has none.
        kTemporal, // 1 when the frame has a history, else 0.
        kOutput,   // The frame's RGB output.
    } kind;
    uint32_t value;
};
// A kernel launch: GRID one-dimensional groups of the kernel's threads, and
// its COUNT arguments, each passed as a 4-byte value or an 8-byte pointer.
struct Launch {
    Kernel kernel;
    uint8_t count;
    uint32_t grid;
    Arg args[15];
};

// One evaluation of the network at a geometry, as upstream's production
// RunGraph queues it on its stream.
struct Plan {
    uint16_t width, height;
    uint16_t tokens; // The ViT's; each gather map holds 1024 of them.
    std::vector<Launch> launches;
    // The floats of each tensor, by its id (upstream's New(n)).
    std::vector<uint32_t> floats;
    // Tensors taken from the pool (id) and returned (~id), in program order.
    std::vector<int32_t> events;
    // Every weight upstream uploads, in its order. Some no kernel reads.
    std::vector<WeightSpec> weights;
};
// The plan at WIDTHxHEIGHT, one of the network's padded tiers. PERFORMANCE
// skips the blocks dlsslopd --performance skips (42, 43 and 46).
Result<Plan> plan(unsigned width, unsigned height, bool performance);
// The ViT's gather map for TOKENS tokens of 1024 channels: for each element,
// the one it takes, with tokens permuted within 16 and channels within 32
// into the order the ViT's kernels read, or back with INVERSE (upstream:
// Gather).
std::vector<uint32_t> gather_map(unsigned tokens, bool inverse);

// The pool buffers a plan's tensors live in, as upstream's allocator assigns
// them: the smallest free buffer that holds a tensor, the earliest of equal
// ones, or a new buffer of its exact size. The first frame starts with no
// buffers and sizes the pool. Each later frame starts with all of them free
// and settles on another assignment; place() fails if it needs more.
struct Placement {
    std::vector<size_t> buffers;        // Bytes, in creation order.
    std::vector<uint16_t> first, later; // Each tensor's buffer.
};
Result<Placement> place(const Plan&);

} // namespace dlsslop::hip
