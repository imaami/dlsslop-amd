// SPDX-License-Identifier: MIT
// SDR boundary codec adapted from lmxxf/dlss5-on-amd-9070xt-porting.
#pragma once

#include "geometry.h"

#include <cstdint>
#include <vector>

namespace dlsslop {

// tier_height: 0 selects the smallest fitting tier; otherwise 720, 900 or 1080.
// Larger source pictures are fitted into the largest tier. Throws on invalid
// geometry; sources must be nonempty and no larger than 16384 per dimension.
Geometry geometry(unsigned source_width, unsigned source_height,
                  unsigned tier_height = 0);

// Throws std::invalid_argument unless g is what geometry() returns for its
// source extent and tier.
void validate(const Geometry& g);

// Source is tightly packed display-referred SDR sRGB RGBA8, or with fp16
// native-endian RGBA16F, deliberately NOT gamma-decoded or clamped before
// neural inference; nonfinite FP16 samples are rejected. Output is padded,
// raster-interleaved RGBA32F; each input RGB sample is rounded through FP16 as
// in the upstream RGBA16_FLOAT encoding texture. Alpha is one. A source no
// larger than the fit is sampled bilinearly; a larger one is averaged over
// each pixel's footprint.
void encode_proxy(const std::uint8_t* source, const Geometry& g, bool fp16,
                  std::vector<float>& rgba);

// Feed a raw neural result into another pass at the same processing extent.
// Preserve the fitted viewport, round RGB through binary16 without UNORM8
// quantization/clamping or resampling, set alpha to one, and regenerate black
// letterbox and reflected bottom padding. Reject nonfinite/FP16-overflow values
// within the fitted viewport. With precision16=false, clamp/round finite input
// to UNORM8 before the binary16 conversion instead. Input/output must not alias.
void feedback_neural_rgb(const float* neural_rgb, const Geometry& g,
                         std::vector<float>& rgba, bool precision16 = true);

// Raw neural view: undo aspect fitting and padding, round through the upstream
// FP16 neural surface, clamp to UNORM8, or with fp16 round the RGB to binary16,
// keeping signed and >1 values. The layer may then compose this with the
// original frame. The neural buffer contains width*height*3 RGB32F values. Like
// the GPU codec, a nonfinite or FP16-overflow neural sample that the resampling
// reads throws std::range_error. The original source's alpha is preserved.
// Output holds the source's extent and format; input/output must not alias.
void decode_neural_proxy(const std::uint8_t* original, const Geometry& g, bool fp16,
                         const float* neural_rgb, std::uint8_t* output);

// Full upstream SDR composition, at full strength, for offline validation or
// a layer bypass: linear-light Upgrade + Oklab hue correction + AP1 gamut
// clamp and sRGB output. encoded_rgba is the RGBA8 encode_proxy result.
// Neural samples are rejected as above. Input/output must not alias.
void decode_rgba8(const std::uint8_t* original, const Geometry& g,
                  const float* encoded_rgba, const float* neural_rgb,
                  std::uint8_t* output);

} // namespace dlsslop
