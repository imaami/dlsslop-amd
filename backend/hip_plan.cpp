// SPDX-License-Identifier: MIT
#include "hip_plan.hpp"

#include <bit>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

namespace dlsslop::hip {

namespace {
// The shift of each block of an encoder level, and of the C512 encoder, by its
// place in the group (upstream: RunGraph's shifts).
constexpr unsigned kShift[8] = {0, 3, 1, 2, 0, 3, 1, 2};
// The shift of decoder blocks 40 to 69 (upstream: Shift).
constexpr unsigned kDecoderShift[30] = {0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2,
                                        1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2};
// The multihead blocks of each level: c64, c128 and c256 down, then back up.
struct Level {
    unsigned first, last, c;
};
constexpr Level kEncoderLevels[3] = {{5, 8, 64}, {9, 14, 128}, {15, 22, 256}};
constexpr Level kDecoderLevels[3] = {{48, 55, 256}, {56, 61, 128}, {62, 65, 64}};

// The fused FFN and QKV kernel of a level, for an identity block's input or a
// mapped one, read as floats or as bytes.
constexpr Kernel kFfn[3][2][2] = {
    {{Kernel::kFfnC64, Kernel::kFfnC64Bytein}, {Kernel::kFfnC64Mapped, Kernel::kFfnC64MappedBytein}},
    {{Kernel::kFfnC128, Kernel::kFfnC128Bytein}, {Kernel::kFfnC128Mapped, Kernel::kFfnC128MappedBytein}},
    {{Kernel::kFfnC256, Kernel::kFfnC256Bytein}, {Kernel::kFfnC256Mapped, Kernel::kFfnC256MappedBytein}}};
// The attention and projection kernel of a level, writing floats or bytes.
constexpr Kernel kAttention[3][2] = {{Kernel::kAttentionC64, Kernel::kAttentionC64Bout},
                                     {Kernel::kAttentionC128, Kernel::kAttentionC128Bout},
                                     {Kernel::kAttentionC256, Kernel::kAttentionC256Bout}};
constexpr Kernel kPoolGroup[4] = {Kernel::kPoolGroupC64, Kernel::kPoolGroupC128, Kernel::kPoolGroupC256,
                                  Kernel::kPoolGroupC512};
// By the most tokens each one attends over: 256, 400 and 640.
constexpr Kernel kVitAttention[3] = {Kernel::kVitAttention256, Kernel::kVitAttention400, Kernel::kVitAttention640};

// The geometries upstream accepts that dlsslopd's tiers use.
constexpr struct {
    unsigned width, height;
} kGeometries[] = {{1280, 768}, {1600, 960}, {1920, 1152}};

// Upstream's Run(): the groups of a launch of COUNT elements.
uint32_t groups(const KernelInfo& k, size_t count, const Arg* args)
{
    switch (k.grid) {
    case Grid::kGroups: return uint32_t(count);
    case Grid::kDefault: return uint32_t((count + 255) / 256);
    case Grid::k512: return uint32_t(count / 512);
    case Grid::k1024: return uint32_t(count / 1024);
    case Grid::kFfn: return uint32_t(count / (8 * k.threads));
    case Grid::kPoolGroup: return uint32_t((count / (2 * k.threads) + 15) / 16);
    case Grid::kScalar: {
        const size_t c = args[5].value;
        return uint32_t((count / c + 63) / 64 * ((c + 63) / 64));
    }
    case Grid::kDecoder: {
        const size_t c = args[9].value;
        return uint32_t((count / c + 15) / 16 * (c / 16));
    }
    }
    return 0;
}

// Records upstream's RunGraph as it queues its production path: each launch,
// each weight a launch reads when upstream first asks for it, and each tensor
// it takes from its pool and returns.
class Builder {
public:
    // Upstream's pooled Tensor: a reference to a tensor. When the last one is
    // gone, the pool may hand the tensor's buffer to the next.
    class Tensor {
        friend class Builder;
        Builder* builder_ = nullptr;
        uint32_t id_ = 0;
        Tensor(Builder* builder, uint32_t id) : builder_(builder), id_(id) {}

    public:
        Tensor() = default;
        Tensor(const Tensor& other) : builder_(other.builder_), id_(other.id_)
        {
            if (builder_) ++builder_->references_[id_];
        }
        Tensor(Tensor&& other) noexcept : builder_(std::exchange(other.builder_, nullptr)), id_(other.id_) {}
        Tensor& operator=(Tensor other) noexcept
        {
            std::swap(builder_, other.builder_);
            std::swap(id_, other.id_);
            return *this;
        }
        ~Tensor() { reset(); }
        void reset()
        {
            if (builder_ && !--builder_->references_[id_]) builder_->plan_.events.push_back(~int32_t(id_));
            builder_ = nullptr;
        }
        explicit operator bool() const { return builder_; }
    };

    Builder(unsigned width, unsigned height, bool performance) : performance_(performance)
    {
        plan_.width = uint16_t(width);
        plan_.height = uint16_t(height);
    }
    // Every tensor is returned when run() does, so the plan is complete.
    void run();
    Plan plan() && { return std::move(plan_); }

private:
    struct Weight {
        uint32_t index;
    };
    // Upstream's C32Result, less what production never reads.
    struct C32 {
        Tensor main, down, raw;
        unsigned workw, sx, sy;
    };
    // The tokens a pooling keeps: OW x OH, of which VW x VH hold the input
    // when those are nonzero, the rest being zero padding.
    struct Pooled {
        unsigned ow, oh, vw, vh;
    };

    Plan plan_;
    std::vector<uint32_t> references_;
    bool performance_;

    // Upstream's New(FLOATS).
    Tensor tensor(size_t floats)
    {
        const uint32_t id = uint32_t(plan_.floats.size());
        plan_.floats.push_back(uint32_t(floats));
        plan_.events.push_back(int32_t(id));
        references_.push_back(1);
        return Tensor(this, id);
    }
    // The weight STEM packed by RECIPE, listed the first time upstream asks
    // for it.
    Weight weight(const char* stem, Recipe recipe)
    {
        for (uint32_t i = 0; i < plan_.weights.size(); ++i)
            if (plan_.weights[i].recipe == recipe && !std::strcmp(plan_.weights[i].stem, stem)) return {i};
        WeightSpec& spec = plan_.weights.emplace_back(WeightSpec{{}, recipe});
        std::snprintf(spec.stem, sizeof spec.stem, "%s", stem);
        return {uint32_t(plan_.weights.size() - 1)};
    }
    Weight weight(unsigned block, const char* part, Recipe recipe)
    {
        char stem[sizeof WeightSpec::stem];
        std::snprintf(stem, sizeof stem, "block%u-%s", block, part);
        return weight(stem, recipe);
    }

    static Arg argument(const Tensor& tensor) { return tensor ? Arg{Arg::kTensor, tensor.id_} : Arg{Arg::kNull, 0}; }
    static Arg argument(Weight weight) { return {Arg::kWeight, weight.index}; }
    static Arg argument(unsigned value) { return {Arg::kU32, value}; }
    static Arg argument(float value) { return {Arg::kF32, std::bit_cast<uint32_t>(value)}; }
    static Arg argument(std::nullptr_t) { return {Arg::kNull, 0}; }
    static Arg argument(Arg arg) { return arg; }
    // A literal's type must say whether it is a value or a pointer.
    static Arg argument(int) = delete;

    // Upstream's Run(): KERNEL over COUNT elements.
    template <class... A>
    void launch(Kernel kernel, size_t count, const A&... args)
    {
        static_assert(sizeof...(A) <= sizeof(Launch::args) / sizeof(Arg));
        Launch l{kernel, uint8_t(sizeof...(A)), 0, {argument(args)...}};
        l.grid = groups(kKernels[size_t(kernel)], count, l.args);
        plan_.launches.push_back(l);
    }

    C32 c32_chain(Tensor input, const C32* prev, unsigned w, unsigned h, unsigned shift, unsigned block, bool finish,
                  bool down);
    Tensor mh_block(Tensor input, unsigned w, unsigned h, unsigned c, unsigned shift, unsigned block, bool raw,
                    bool byte_in, bool byte_out);
    Tensor c512_block(Tensor input, unsigned w, unsigned h, unsigned shift, unsigned block, bool raw);
    Tensor down(Tensor raw, unsigned w, unsigned c, Weight weights, Pooled p);
    Tensor gather(Tensor input, unsigned tokens, bool inverse);
    Tensor vit(Tensor input, unsigned tokens);
    Tensor vit_block(Tensor input, unsigned n, unsigned block);
    Tensor up(Tensor input, Tensor skip, unsigned iw, unsigned ih, unsigned ow, unsigned oh, unsigned ic, unsigned oc,
              Weight weights, bool byte_out);
};

// Upstream: C32Chain. A block of the C32 chain at W x H, which passes on its
// raw window tiles; a finishing block writes the main output or its pooled
// quarter instead. Upstream finishes the first block of a chain in a kernel
// of its own, but no first block finishes.
Builder::C32 Builder::c32_chain(Tensor input, const C32* prev, unsigned w, unsigned h, unsigned shift, unsigned block,
                                bool finish, bool down)
{
    const unsigned sx = shift & 1 ? 4 : 0, sy = shift & 2 ? 4 : 0, ww = w + 2 * sx, hh = h + 2 * sy,
                   windows = ww * hh / 64;
    // Upstream asks for the weights as it evaluates the launch's arguments,
    // right to left.
    const Weight attention = weight(block, "attention", Recipe::kC32),
                 ffn = weight(block, "ffn", Recipe::kC32);
    if (prev && (finish || down)) {
        C32 result{finish ? tensor(size_t(w) * h * 32) : Tensor(),
                   down ? tensor(size_t(w / 2) * (h / 2) * 32) : Tensor(), Tensor(), ww, sx, sy};
        launch(down ? Kernel::kC32ChainFinishDcrop : Kernel::kC32ChainFinish, windows, prev->raw, ffn, attention,
               result.main, result.down, windows, 3u, 1u, w, h, sx, sy, prev->workw, prev->sx, prev->sy);
        return result;
    }
    // Upstream asks for 16 floats a token; the kernels use half of them.
    C32 result{Tensor(), Tensor(), tensor(size_t(ww) * hh * 16), ww, sx, sy};
    if (prev)
        launch(Kernel::kC32Chain, windows, prev->raw, ffn, attention, result.raw, windows, 3u, 1u, w, h, sx, sy,
               prev->workw, prev->sx, prev->sy);
    else
        launch(Kernel::kC32Mapped, windows, input, ffn, attention, result.raw, windows, 0u, 1u, w, h, sx, sy);
    return result;
}

// Upstream: Body below 512 channels, with AttentionFast. The input is mapped
// onto the block's shifted lattice as the FFN reads it, and the attention
// crops back; within a level the blocks pass bytes.
Builder::Tensor Builder::mh_block(Tensor input, unsigned w, unsigned h, unsigned c, unsigned shift, unsigned block,
                                  bool raw, bool byte_in, bool byte_out)
{
    const unsigned sx = shift & 1 ? 4 : 0, sy = shift & 2 ? 4 : 0, ww = (w + sx + 7) & ~7u, hh = (h + sy + 7) & ~7u,
                   n = ww * hh, level = unsigned(std::countr_zero(c)) - 6;
    const bool identity = !sx && !sy && ww == w && hh == h;
    const Tensor ffn = tensor(size_t(n) * c / 4), norm = tensor(size_t(n) * 3 * c / 4);
    const bool frag = c == 256;
    // At c256 upstream also uploads the attention weight as kMhAttention packs
    // it, which no kernel reads.
    const Weight ffn_weights = weight(block, "ffn", frag ? Recipe::kFfnFrag : Recipe::kMhFfn),
                 qkv = weight(block, "attention", frag ? Recipe::kQkvFragOnly : Recipe::kMhAttention);
    launch(kFfn[level][!identity][byte_in], size_t(n) * c, input, ffn_weights, qkv, ffn, norm, n, w, h, ww, sx, sy);
    const Tensor out = tensor(size_t(identity ? n : w * h) * c / (byte_out ? 4 : 1));
    // The output's rounding: Hrtz at the end of an encoder level, which is a
    // skip; F of Hrtz at blocks 48, 55, 61 and 65; F elsewhere.
    const unsigned post = raw ? 3 : block == 48 || block == 55 || block == 61 || block == 65 ? 0 : 4;
    launch(kAttention[level][byte_out], n / 64, norm, weight(block, "attention", Recipe::kMhAttentionDiag), ffn,
           out, ww, hh, post, identity ? 0u : w, identity ? 0u : h, sx, sy);
    return out;
}

// Upstream: Body at 512 channels, with AttentionFast. Unless the block is an
// identity (unshifted, on a grid that is already a multiple of 8, as at
// 1280x768), its input is packed onto the padded, shifted lattice first.
Builder::Tensor Builder::c512_block(Tensor input, unsigned w, unsigned h, unsigned shift, unsigned block, bool raw)
{
    if (performance_ && (block == 42 || block == 43 || block == 46)) return input;
    const unsigned sx = shift & 1 ? 4 : 0, sy = shift & 2 ? 4 : 0, ww = (w + sx + 7) & ~7u, hh = (h + sy + 7) & ~7u,
                   n = ww * hh;
    const bool identity = !sx && !sy && ww == w && hh == h;
    Tensor packed = identity ? input : tensor(size_t(n) * 512);
    if (!identity) launch(Kernel::kShiftPack, size_t(n) * 512, input, packed, w, h, ww, hh, sx, sy, 512u, 0u);
    const Tensor ffn = tensor(size_t(n) * 512);
    Tensor norm;
    {
        Tensor mixed = tensor(size_t(n) * 512);
        const Tensor contract = tensor(size_t(n) * 512);
        // Upstream also uploads the unpacked ffwd weights, which no kernel
        // reads.
        const Weight mix = weight(block, "ffwd", Recipe::kSplitMixF16);
        launch(Kernel::kSplitMix, size_t(n) * 512, packed, mix, mixed, n);
        const Tensor contract8 = tensor(size_t(n) * 128);
        // The kernel also writes CONTRACT, which nothing reads.
        launch(Kernel::kSplitFfn, size_t(n) * 512, mixed, mix, contract, contract8, n);
        mixed.reset();
        const Tensor ffn8 = tensor(size_t(n) * 128);
        launch(Kernel::kSplitProjection, size_t(n) * 512, contract8,
               weight(block, "ffwd-projection", Recipe::kProjFrag), packed, ffn, ffn8, n);
        norm = tensor(size_t(n) * 384);
        launch(Kernel::kQkvC512, size_t(n) * 1536, ffn8, weight(block, "attention", Recipe::kQkvFrag), norm, n);
    }
    packed.reset();
    const Weight attention = weight(block, "attention", Recipe::kMhAttention);
    const Tensor av = tensor(size_t(n) * 128), out = tensor(size_t(identity ? n : w * h) * 512);
    launch(Kernel::kAttentionC512, size_t(n / 64) * 16, norm, attention, av, ww, hh, 512u);
    // F of Hrtz, or Hrtz at the end of the encoder, which is a skip.
    if (identity)
        launch(Kernel::kProjectScalar, size_t(n) * 512, av, ffn, attention, out, n, 512u, raw ? 3u : 0u);
    else
        launch(Kernel::kProjectC512, size_t(n) * 512, av, ffn, weight(block, "attention", Recipe::kQkvFrag), out,
               n, raw ? 3u : 0u, w, h, ww, sx, sy);
    return out;
}

// Upstream: Down, pooling a W-wide level's output and projecting it to twice
// its C channels.
Builder::Tensor Builder::down(Tensor raw, unsigned w, unsigned c, Weight weights, Pooled p)
{
    const Tensor out = tensor(size_t(p.ow) * p.oh * c * 2);
    launch(kPoolGroup[std::countr_zero(c) - 6], size_t(p.ow) * p.oh * c * 2, raw, weights, out, p.ow, p.oh, w, p.vw,
           p.vh);
    return out;
}

// Upstream: Gather, which permutes the tokens into or out of the ViT's layout.
Builder::Tensor Builder::gather(Tensor input, unsigned tokens, bool inverse)
{
    const Tensor out = tensor(size_t(tokens) * 1024);
    launch(Kernel::kVitGather, size_t(tokens) * 1024, input, Arg{Arg::kGather, inverse}, out, tokens * 1024);
    return out;
}

// Upstream: AdaptiveVitGroup without its approximate reuse: blocks 31 to 38.
Builder::Tensor Builder::vit(Tensor input, unsigned tokens)
{
    Tensor full = input;
    for (unsigned b = 31; b <= 38; ++b) full = vit_block(full, tokens, b);
    return full;
}

// Upstream: Vit. Each kernel takes a trailing gate for the adaptive reuse,
// null without it.
Builder::Tensor Builder::vit_block(Tensor input, unsigned n, unsigned block)
{
    Tensor packed = tensor(size_t(n) * 256);
    launch(Kernel::kVitPack, size_t(n) * 256, input, packed, n * 1024, nullptr);
    Tensor hidden = tensor(size_t(n) * 1024);
    const Tensor contract = tensor(size_t(n) * 1024);
    launch(Kernel::kVitExpand, size_t(n) * 4096, packed, weight(block, "expand", Recipe::kVitFrag), hidden, n,
           1024u, 4096u, nullptr);
    packed.reset();
    launch(Kernel::kVitContract, size_t(n) * 1024, hidden, weight(block, "contract", Recipe::kVitFrag), input,
           contract, n, 4096u, 1024u, nullptr);
    hidden.reset();
    Tensor norm = tensor(size_t(n) * 768);
    launch(Kernel::kVitQkv, size_t(n) * 3072, contract, weight(block, "qkv", Recipe::kQkvF16Frag), norm, n,
           nullptr);
    const Tensor av = tensor(size_t(n) * 1024);
    launch(kVitAttention[n <= 256 ? 0 : n <= 400 ? 1 : 2], size_t(n) * 512, norm, av, n, nullptr);
    norm.reset();
    const Tensor out = tensor(size_t(n) * 1024);
    launch(Kernel::kVitProject, size_t(n) * 1024, av, weight(block, "projection", Recipe::kVitProjFrag), contract,
           out, n, 1024u, 1024u, nullptr);
    return out;
}

// Upstream: Up, projecting IW x IH tokens of IC channels onto the skip's OW x
// OH of OC channels, twice the size.
Builder::Tensor Builder::up(Tensor input, Tensor skip, unsigned iw, unsigned ih, unsigned ow, unsigned oh, unsigned ic,
                            unsigned oc, Weight weights, bool byte_out)
{
    const Tensor out = tensor(size_t(ow) * oh * oc / (byte_out ? 4 : 1));
    launch(byte_out ? Kernel::kDecoderByteout : Kernel::kDecoder, size_t(iw) * ih * oc, input, weights, skip, out, iw,
           ih, ow, oh, ic, oc);
    return out;
}

// Upstream: RunGraph's production path, which reads the frame in place, with
// the prefix fused into block 0 and the RGB head into the post.
void Builder::run()
{
    const unsigned W = plan_.width, H = plan_.height;
    // Block 0 reads the frame's input, and its history when it has one.
    Tensor skip0, source;
    {
        const unsigned windows = W * H / 64;
        skip0 = tensor(size_t(W) * H * 8);
        source = tensor(size_t(W) * H / 4 * 32);
        const Weight attention = weight(0, "attention", Recipe::kC32),
                     ffn = weight(0, "ffn", Recipe::kC32);
        launch(Kernel::kC32Prefix, windows, Arg{Arg::kRgba}, Arg{Arg::kHistory}, ffn, attention, skip0, source, windows,
               0u, 1u, W, H, 0u, Arg{Arg::kTemporal});
    }
    Tensor skips[5];
    C32 last{};
    for (unsigned b = 1; b <= 4; ++b) {
        last = c32_chain(source, last.raw ? &last : nullptr, W / 2, H / 2, kShift[b - 1], b, b == 4, b == 4);
        source = last.main;
    }
    skips[0] = source;
    Tensor pooled = last.down;
    source = tensor(size_t(W / 4) * (H / 4) * 64);
    launch(Kernel::kPool32, size_t(W / 4) * (H / 4) * 64, pooled, weight(4, "ds", Recipe::kDsCast), source, W / 4,
           H / 4, 0u, 0u, 32u);
    pooled.reset();
    last = {};

    for (unsigned g = 0; g < 3; ++g) {
        const Level& l = kEncoderLevels[g];
        const unsigned w = W / (4u << g), h = H / (4u << g);
        for (unsigned b = l.first; b <= l.last; ++b)
            source = mh_block(source, w, h, l.c, kShift[b - l.first], b, b == l.last, b > l.first, b < l.last);
        skips[g + 1] = source;
        source = down(source, w, l.c, weight(l.last, "ds", Recipe::kDsFrag), {w / 2, h / 2, 0, 0});
    }
    for (unsigned j = 0; j < 8; ++j) source = c512_block(source, W / 32, H / 32, kShift[j], 23 + j, j == 7);
    skips[4] = source;

    // The head pools to the ViT's token grid, padded to a multiple of 16
    // tokens: at 1600x960 by a row, at 1920x1152 to 32x20. Upstream works the
    // grid out again for the ViT by another rule, which agrees at every tier.
    const unsigned w = W / 32, h = H / 32, rw = w / 2, rh = h / 2;
    Pooled head = w == 60 && h == 36 ? Pooled{32, 20, 0, 0} : Pooled{rw, (rw * rh) % 16 ? rh + 1 : rh, 0, 0};
    if (head.ow * head.oh != rw * rh) {
        head.vw = rw;
        head.vh = rh;
    }
    const unsigned tokens = head.ow * head.oh;
    plan_.tokens = uint16_t(tokens);
    source = down(source, w, 512, weight("head-matrix", Recipe::kDsFrag), head);
    source = gather(source, tokens, false);
    source = vit(source, tokens);
    source = gather(source, tokens, true);
    source = up(source, skips[4], head.ow, head.oh, w, h, 1024, 512,
                weight("decoder39-weights", Recipe::kDecoderF16r), false);
    skips[4].reset();

    for (unsigned b = 40; b <= 47; ++b) source = c512_block(source, w, h, kDecoderShift[b - 40], b, false);
    for (unsigned g = 0; g < 3; ++g) {
        const Level& l = kDecoderLevels[g];
        const unsigned ow = W / (16u >> g), oh = H / (16u >> g);
        source = up(source, skips[3 - g], ow / 2, oh / 2, ow, oh, 2 * l.c, l.c,
                    weight(l.first, "weights", Recipe::kDecoderF16r), true);
        skips[3 - g].reset();
        for (unsigned b = l.first; b <= l.last; ++b)
            source = mh_block(source, ow, oh, l.c, kDecoderShift[b - 40], b, false, true, b < l.last);
    }
    source = up(source, skips[0], W / 4, H / 4, W / 2, H / 2, 64, 32, weight(66, "weights", Recipe::kDecoderF16r),
                false);
    skips[0].reset();
    C32 chain{};
    for (unsigned b = 66; b <= 69; ++b) {
        chain = c32_chain(source, chain.raw ? &chain : nullptr, W / 2, H / 2, kDecoderShift[b - 40], b, b == 69, false);
        source = chain.main;
    }
    chain = {};

    // The post merges block 0's bytes back in, and writes the frame's RGB.
    const unsigned windows = (W + 8) * (H + 8) / 64;
    const Weight head_weights = weight("post70-head", Recipe::kRaw),
                 attention = weight("post70-attention", Recipe::kC32),
                 ffn = weight("post70-ffn", Recipe::kC32), scales = weight("post70-scales", Recipe::kRaw);
    launch(Kernel::kC32Post, windows, source, skip0, scales, ffn, attention, Arg{Arg::kRgba}, head_weights,
           Arg{Arg::kOutput}, windows, W, H, 4u, 4u, .03125f);
}
} // namespace

Result<Plan> plan(unsigned width, unsigned height, bool performance)
{
    bool supported = false;
    for (const auto& g : kGeometries) supported |= g.width == width && g.height == height;
    if (!supported)
        return fail("unsupported processing geometry " + std::to_string(width) + "x" + std::to_string(height));
    Builder builder(width, height, performance);
    builder.run();
    return std::move(builder).plan();
}

std::vector<uint32_t> gather_map(unsigned tokens, bool inverse)
{
    std::vector<uint32_t> map(size_t(tokens) * 1024);
    for (uint32_t t = 0; t < tokens; ++t)
        for (uint32_t c = 0; c < 1024; ++c) {
            const uint32_t raster = (t & ~15u) | ((t & 1u) << 3) | ((t & 14u) >> 1),
                           channel = (c & ~31u) | ((c & 1u) << 1) | ((c & 2u) >> 1) | ((c & 4u) << 2) |
                                     ((c & 24u) >> 1),
                           to = t * 1024 + c, from = raster * 1024 + channel;
            if (inverse)
                map[from] = to;
            else
                map[to] = from;
        }
    return map;
}

Result<Placement> place(const Plan& plan)
{
    Placement p;
    std::vector<bool> taken;
    // Upstream's New() over its pool, which only grows: GROW adds a buffer
    // when none is free that holds the tensor, else it is a failure.
    auto assign = [&](std::vector<uint16_t>& buffer_of, bool grow) {
        buffer_of.assign(plan.floats.size(), 0);
        taken.assign(p.buffers.size(), false);
        for (const int32_t event : plan.events) {
            if (event < 0) {
                taken[buffer_of[~event]] = false;
                continue;
            }
            const size_t bytes = size_t(plan.floats[event]) * 4, none = p.buffers.size();
            size_t best = none;
            for (size_t i = 0; i < p.buffers.size(); ++i)
                if (!taken[i] && p.buffers[i] >= bytes && (best == none || p.buffers[i] < p.buffers[best])) best = i;
            if (best == none) {
                if (!grow) return false;
                p.buffers.push_back(bytes);
                taken.push_back(false);
            }
            taken[best] = true;
            buffer_of[event] = uint16_t(best);
        }
        return true;
    };
    assign(p.first, true);
    if (!assign(p.later, false)) return fail("the HIP network's buffers do not suffice after its first frame");
    return p;
}

} // namespace dlsslop::hip
