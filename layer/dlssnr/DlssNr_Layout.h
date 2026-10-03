/** @file
 *
 * The host's constant struct and the compiled shader's cbuffer, pinned together.
 *
 * DlssNr_Common.h and DlssNr_Shader_Vk.spv are vendored as a pair from upstream, and nothing in the
 * build makes the host struct and the shader's uniform block agree -- a member added to one and not
 * the other compiles, links, runs, and quietly feeds the shader the wrong numbers from that member
 * onward. The failure looks like a settings bug, not a layout bug.
 *
 * These offsets were read out of the vendored module itself:
 *
 *     spirv-dis DlssNr_Shader_Vk.spv | grep 'OpMemberDecorate %type_Params'
 *
 * If a re-vendor changes the block, this stops compiling and whoever did it is pointed at the one
 * command that says what the new layout is. That is the whole intent: it is cheaper to re-read the
 * offsets than to debug a frame that is subtly wrong.
 */
#ifndef DLSSLOP_AMD_LAYER_DLSSNR_DLSSNR_LAYOUT_H_
#define DLSSLOP_AMD_LAYER_DLSSNR_DLSSNR_LAYOUT_H_

#include <stddef.h>

#include "DlssNr_Common.h"

static_assert(sizeof (struct dlss_nr_constants) == 256,
              "the cbuffer is declared alignas(256); see DlssNr_Common.h");

#define DLSSNR_PIN(member, offset) \
	static_assert(offsetof(struct dlss_nr_constants, member) == (offset), \
	              "dlss_nr_constants." #member " no longer matches the vendored SPIR-V")

DLSSNR_PIN(mode, 0);
DLSSNR_PIN(white_point, 4);
DLSSNR_PIN(width, 8);
DLSSNR_PIN(height, 12);
DLSSNR_PIN(transfer_strength, 16);
DLSSNR_PIN(colour_strength, 20);
DLSSNR_PIN(debug_view, 24);
DLSSNR_PIN(max_ratio, 28);
DLSSNR_PIN(passthrough, 32);
DLSSNR_PIN(mv_scale_x, 36);
DLSSNR_PIN(mv_scale_y, 40);
DLSSNR_PIN(guide_width, 44);
DLSSNR_PIN(guide_height, 48);
DLSSNR_PIN(compare_mode, 52);
DLSSNR_PIN(compare_split, 56);
DLSSNR_PIN(compare_zoom, 60);
DLSSNR_PIN(compare_swap, 64);
DLSSNR_PIN(transfer, 68);
DLSSNR_PIN(debug_scale, 72);
DLSSNR_PIN(reversible_mode, 76);
DLSSNR_PIN(apply_model, 80);
// The last two are the D3D12 zero-latency exposure path, which VK_MODE compiles out of the shader.
// They stay in the struct because the block's layout is shared with the D3D12 build; the layer
// leaves use_game_exposure at 0 and the shader never reads either.
DLSSNR_PIN(use_game_exposure, 84);
DLSSNR_PIN(exposure_pre_mul, 88);
DLSSNR_PIN(hdr_proxy, 92);
DLSSNR_PIN(hdr_transfer, 96);
DLSSNR_PIN(colour_trust, 100);
DLSSNR_PIN(ratio_smooth, 104);

#undef DLSSNR_PIN

#endif /* DLSSLOP_AMD_LAYER_DLSSNR_DLSSNR_LAYOUT_H_ */
