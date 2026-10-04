/** @file
 *
 * The DLSS-NR pass, arranged for a present-time layer.
 *
 * OptiScaler runs this immediately after the game's upscaler, on surfaces it already holds. Here the
 * only thing available is a finished swapchain image, and the model lives in another process behind a
 * shared-memory round trip, so the same pass has to be split in two around that round trip:
 *
 *   leg 1   swapchain -> frame -> ENCODE -> proxy, unless a display-referred UNORM frame is its own
 *                              -> DOWNSAMPLE -> work, when the model runs below the frame or
 *                                 reads the frame directly
 *                              -> host buffer
 *     ...   the helper runs the model on those pixels and answers
 *   leg 2   host buffer -> model image -> RESOLVE -> composed -> swapchain
 *
 * What crosses the shared memory is the proxy, not the frame: it is display-referred and 8-bit by
 * construction, so the helper never has to know what format the game presents in, and the working
 * scale reduces it quadratically.
 *
 * The in-layer network runs the model on the game's device between the legs instead, in the same
 * submission: it samples the proxy where leg 1 leaves it and answers into the model image, so nothing
 * crosses and there is no host buffer.
 *
 * The composition itself -- what the resolve does with the model's answer -- is entirely the vendored
 * shader's. Everything in this file is plumbing: which image is bound where, in what layout, and what
 * goes in the constant block.
 */
#ifndef DLSSLOP_AMD_LAYER_COMPOSITION_H_
#define DLSSLOP_AMD_LAYER_COMPOSITION_H_

#include <stddef.h>
#include <stdint.h>

#include "../common/shm_protocol.h"
#include "capture.h"
#include "dlssnr_pass.h"
#include "vk_table.h"

/** @brief One frame's worth of settings, read from the shared header once so that a control changed
 *         mid-frame cannot make the encode and the resolve disagree about what they are doing.
 *
 * The white point is three numbers, not one: where it comes from, the multiplier that says what the
 * model should consider white, and the trim that belongs to a measured reading rather than to the
 * slider. Keeping them apart is upstream's fix for a real bug -- sharing one stored value meant
 * touching the slider in one mode silently destroyed the number found in the other.
 *
 * composition_bypass 1 presents the model's raw answer as the frame -- no blend, no guard, no
 * compare. ghost_slack is how much of the way toward a newly arrived answer the running pair moves
 * each frame, 0..1, and edit_blur the radius splitting the stale edit's safe half from the half that
 * can ghost, in uv. colour_trust is how much of the chroma-agreement gate to apply (see
 * colourTrustPercent), and ratio_smooth how much of the relighting ratio comes from the
 * neighbourhood (see ratioSmoothPercent).
 *
 * composition_frame_settings() returns the defaults, and composition_frame_settings_read() a frame's
 * snapshot.
 */
struct composition_frame_settings {
	uint32_t control_seq;             //!< The header's controlSeq.
	uint32_t tuning_seq;              //!< The header's tuningSeq.
	uint32_t passes;                  //!< The requested number of passes.
	float    transfer_strength;       //!< The detail transfer's strength, 0..4.
	float    colour_strength;         //!< The colour transfer's strength, 0..4.
	float    max_ratio;               //!< The largest relighting ratio.
	float    debug_scale;             //!< The debug views' scale.
	float    white_point_manual;      //!< The slider's white point.
	float    white_point_scale;       //!< The multiplier of what is white.
	float    white_point_trim;        //!< The trim of a measured reading.
	uint32_t white_point_source;      //!< enum WhitePointSource.
	float    compare_split;           //!< Where the compare view splits, 0..1.
	float    compare_zoom;            //!< The compare view's zoom, 1..2.
	float    working_scale;           //!< The model's raster over the frame's, 0.25..1.
	uint32_t native_model_max_width;  //!< The native model's widest raster, or 0.
	uint32_t native_model_max_height; //!< The native model's tallest raster, or 0.
	uint32_t transfer;                //!< The transfer mode, 0..2.
	uint32_t debug_view;              //!< The debug view, 0..5.
	uint32_t compare_mode;            //!< The compare mode, 0..2.
	uint32_t compare_swap;            //!< Whether the compare view swaps its sides.
	uint32_t reversible_mode;         //!< enum ReversibleMode.
	uint32_t apply_model;             //!< Whether the model's answer is applied.
	uint32_t hold_frame;              //!< Whether the frame is held.
	uint32_t composition_bypass;      //!< 1: the model's raw answer is the frame.
	float    ghost_slack;             //!< The running pair's step toward a new answer.
	float    edit_blur;               //!< The stale edit's split radius, in uv.
	float    motion_smooth;           //!< The motion smoothing, 0..1.
	float    colour_trust;            //!< The chroma-agreement gate's share.
	float    ratio_smooth;            //!< The neighbourhood's share of the ratio.
};

/** @brief A surface of the composition.
 *
 * A zeroed image is an empty one. An image owns the handles that are not VK_NULL_HANDLE.
 */
struct composition_image {
	VkImage        image;  //!< The image.
	VkDeviceMemory memory; //!< Its device-local memory.
	VkImageView    view;   //!< Its view.
	VkFormat       format; //!< Its format.
	uint32_t       width;  //!< Its width.
	uint32_t       height; //!< Its height.
	VkImageLayout  layout; //!< The layout that the commands recorded so far leave it in.
};

/** @brief A buffer that the host or the daemon reads or writes.
 *
 * A zeroed buffer is an empty one. A buffer owns the handles that are not VK_NULL_HANDLE.
 */
struct composition_host_buffer {
	VkBuffer       buffer;     //!< The buffer.
	VkDeviceMemory memory;     //!< Its memory.
	void          *mapped;     //!< The memory, mapped; nullptr for exported memory.
	size_t         size;       //!< The buffer's size in bytes.
	VkDeviceSize   allocation; //!< Exported device-local memory: its size; otherwise 0.
};

/** @brief The state that struct composition records in its flags. */
enum composition_flags : uint64_t {
	COMPOSITION_BLIT_SWAPCHAIN      = 1 << 0, //!< work_format is not the swapchain's twin: blit.
	COMPOSITION_LINEAR_HDR          = 1 << 1, //!< The frame holds linear light.
	COMPOSITION_HDR_PROXY           = 1 << 2, //!< The surfaces that cross are float16.
	COMPOSITION_METER_STATE_CLEARED = 1 << 3, //!< The fill that clears the meter's state is recorded.
	COMPOSITION_EXPORT              = 1 << 4, //!< The transport pair is to be exported.
	COMPOSITION_TRANSPORT_READY     = 1 << 5, //!< The daemon imported the exported pair.
	COMPOSITION_CAPTURE_RECORDED    = 1 << 6, //!< This frame's compose recorded a capture pair.
	COMPOSITION_HOLDING             = 1 << 7, //!< The frame is held.
	COMPOSITION_FRAME_CAPTURED      = 1 << 8, //!< frame holds a frame that leg 1 copied.
	COMPOSITION_NETWORK             = 1 << 9  //!< Built for the in-layer network: no transport pair.
};

/** @brief The composition of one swapchain: the pass, the surfaces sized to its frames, and the
 *         transport pair.
 *
 * composition_empty() makes an empty composition: its offer is -1, and every other member is zero.
 * composition() builds the pass, composition_prepare() the surfaces, and composition_fini() destroys
 * what the composition owns and leaves it empty. A failed Vulkan call leaves its output undefined,
 * so a handle is stored only once the call that made it has succeeded. error is the build's
 * failure, or VK_ERROR_FORMAT_NOT_SUPPORTED once composition_prepare() finds that the device cannot
 * write the swapchain's format. The flags are 64 bits wide, which fills the padding that a narrower
 * member would leave. generation tells the surfaces' builds apart where their handles cannot: a
 * destroyed object's handle can come back for a new one.
 *
 * The white point meter: a grid of tile peak luminances measured off the captured frame, and the
 * percentile taken across it -- on the GPU. The reduce pass keeps the percentile, the history and the
 * resolved value in a device-local state buffer; the resolve reads the resolved value through a
 * four-byte copy into its own constant block, so no tensor crosses to the host. The mirror is a
 * 128-byte copy of the state the CPU reads after leg 1's fence, for the frame-hold snapshot and the
 * status field only. See composition_consume_meter(). The state and the mirror exist exactly while
 * the meter runs on the GPU.
 *
 * Frame hold. The freeze point is the raw colour the encode reads, not the proxy: the proxy is
 * derived from it and the resolve reads it, and freezing further down would stop a setting change
 * from re-encoding, which is the whole point of holding.
 */
struct composition {
	struct dlss_nr_pass             pass;                    //!< The composition shader's pass.
	struct capture_writer           capture;                 //!< The matched frames being written.
	struct capture_metadata         capture_metadata;        //!< What the manifest says about the frame.
	struct composition_image        frame;                   //!< The swapchain image, copied.
	struct composition_image        proxy;                   //!< The encode's proxy of the frame.
	struct composition_image        work;                    //!< What the model is handed, if not proxy.
	struct composition_image        model;                   //!< The model's answer.
	struct composition_image        composed;                //!< The resolve's result.
	struct composition_image        meter;                   //!< The meter's grid of tile peaks.
	struct composition_host_buffer  meter_mirror;            //!< The host's copy of the meter's state.
	struct composition_host_buffer  download;                //!< The proxy, for the model.
	struct composition_host_buffer  upload;                  //!< The model's answer.
	struct composition_host_buffer  capture_buf;             //!< A capture pair: frame, then result.
	struct device_table const      *vk;                      //!< The device's next-layer entry points.
	struct instance_table const    *instance;                //!< The instance's next-layer entry points.
	char const                     *reason;                  //!< Why it cannot run; nullptr while it can.
	VkDevice                        device;                  //!< The device.
	VkPhysicalDevice                physical_device;         //!< The device's physical device.
	VkBuffer                        meter_state;             //!< The meter's device-local state.
	VkDeviceMemory                  meter_state_memory;      //!< The state's memory.
	VkPipeline                      meter_pipeline;          //!< The meter's reduce pass.
	VkPipelineLayout                meter_pipeline_layout;   //!< Its layout.
	VkDescriptorSetLayout           meter_descriptor_layout; //!< Its bindings.
	VkDescriptorPool                meter_descriptor_pool;   //!< The pool of its set.
	VkDescriptorSet                 meter_descriptor_set;    //!< Its set.
	VkSampler                       meter_sampler;           //!< The sampler it reads the grid with.
	uint64_t                        flags;                   //!< enum composition_flags.
	uint64_t                        generation;              //!< The surfaces' build, unique; 0: none.
	uint32_t                        width;                   //!< The frame's width.
	uint32_t                        height;                  //!< The frame's height.
	uint32_t                        model_w;                 //!< The model's width.
	uint32_t                        model_h;                 //!< The model's height.
	VkFormat                        swapchain_format;        //!< The swapchain's format.
	VkFormat                        work_format;             //!< Usually the swapchain's UNORM twin.
	uint32_t                        hdr_transfer;            //!< 1: the swapchain carries PQ.
	float                           measured_white_point;    //!< What the meter settled on, or 0.
	float                           held_white_point;        //!< The white point that a held frame keeps.
	uint32_t                        export_family;           //!< The queue family that both legs run on.
	uint32_t                        transport_gen;           //!< The exported pair's generation.
	VkResult                        error;                   //!< Why the composition cannot run.
	int                             offer;                   //!< The connection of an offer, or -1.
};

/** @brief Whether a swapchain format can be composed at all, and what this pass works in when it can.
 *
 * Every internal surface uses the format's UNORM twin rather than the swapchain's own: sampling a
 * _SRGB view would decode to linear on the way in and re-encode on the way out, and the composition
 * wants exactly the display-referred numbers the game already wrote. Copies between the two are
 * byte-for-byte, which is why the final result goes back with vkCmdCopyImage and not a blit -- a blit
 * into an _SRGB image would apply the encode a second time.
 *
 * @param swapchain_format The swapchain's format.
 * @return                 Its UNORM twin, or VK_FORMAT_UNDEFINED if the pass cannot work in it.
 */
extern VkFormat
composition_format (VkFormat swapchain_format);

/** @brief Whether the frame the game presents holds light or a picture.
 *
 * 8-bit and 10-bit formats are already display-referred -- the game tone mapped before it got here --
 * and only a float swapchain is linear.
 *
 * @param swapchain_format The swapchain's format.
 * @param colour_mode      The header's enum ColourMode.
 * @return                 true for linear light.
 */
extern bool
composition_colour_is_linear_hdr (VkFormat swapchain_format,
                                  uint32_t colour_mode);

/** @brief The settings that a frame without a header composes with.
 *
 * @return The defaults.
 */
extern struct composition_frame_settings
composition_frame_settings (void);

/** @brief One frame's settings from the shared header, clamped there rather than trusted, because
 *         the header is a file any process can write.
 *
 * The first call that has a header reads DLSSNR_GHOST_SLACK, DLSSNR_RATIO_SMOOTH,
 * DLSSNR_COLOUR_TRUST, DLSSNR_MOTION_SMOOTH and DLSSNR_EDIT_BLUR, which override their settings
 * from then on.
 *
 * @param h The header, or nullptr.
 * @return  The settings; composition_frame_settings() for nullptr.
 */
extern struct composition_frame_settings
composition_frame_settings_read (struct ShmHeader const *h);

/** @brief An empty composition, which owns nothing.
 *
 * @return A composition whose offer is -1 and whose other members are zero.
 */
static inline struct composition
composition_empty (void)
{
	struct composition ret = {};
	ret.offer = -1;
	return ret;
}

/** @brief Builds a composition's pass on a device.
 *
 * Logs why the device's table cannot run a compute pass; the pass logs its own failures.
 *
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device.
 * @param physical_device The device's physical device.
 * @return                The composition. Its error is VK_SUCCESS if the pass was built; otherwise
 *                        its reason says why not, and it owns nothing.
 */
extern struct composition
composition (struct device_table const   *vk,
             struct instance_table const *instance,
             VkDevice                     device,
             VkPhysicalDevice             physical_device);

/** @brief Builds a composition's pass in place.
 *
 * @param dest            An empty composition, or nullptr.
 * @param vk              The device's next-layer entry points.
 * @param instance        The instance's next-layer entry points.
 * @param device          The device.
 * @param physical_device The device's physical device.
 * @return                The composition's error: VK_SUCCESS if it was built,
 *                        VK_ERROR_INITIALIZATION_FAILED for a null @a dest.
 */
extern VkResult
composition_init (struct composition          *dest,
                  struct device_table const   *vk,
                  struct instance_table const *instance,
                  VkDevice                     device,
                  VkPhysicalDevice             physical_device);

/** @brief Destroys what the composition owns, abandons a capture that is being written, closes the
 *         connection of its offer, then leaves it empty.
 *
 * @param dest The composition, or nullptr.
 */
extern void
composition_fini (struct composition *dest);

/** @brief Whether the composition can run.
 *
 * @param c The composition, or nullptr.
 * @return  true if its pass was built and no composition_prepare() found the swapchain's format
 *          unwritable since.
 */
static inline bool
composition_usable (struct composition const *c)
{
	return c && c->vk && c->error == VK_SUCCESS;
}

/** @brief Why not, when not.
 *
 * @param c The composition, or nullptr.
 * @return  The reason, which is empty while the composition is working; nullptr for nullptr.
 */
static inline char const *
composition_reason (struct composition const *c)
{
	if (!c)
		return nullptr;
	return c->reason ? c->reason : "";
}

/** @brief The raster the model will work at, for a caller that has to size a transport before
 *         composition_prepare() runs. Same arithmetic composition_prepare() uses, in one place so the
 *         two cannot disagree.
 *
 * @param width  The frame's width.
 * @param height The frame's height.
 * @param s      The frame's settings.
 * @return       The model's extent, at most the frame's for a frame of at least kMinW x kMinH.
 */
extern VkExtent2D
composition_model_extent (uint32_t                                 width,
                          uint32_t                                 height,
                          struct composition_frame_settings const *s);

/** @brief Builds or rebuilds everything sized to this frame and this model resolution.
 *
 * Cheap and a no-op when nothing has changed, so it is safe to call every present.
 *
 * @a hdr_proxy asks for the float16 HDR proxy: the crossing surfaces become R16G16B16A16_SFLOAT and
 * the encode writes normalised linear light instead of an sRGB picture. @a hdr_transfer = 1 says the
 * swapchain itself carries PQ. Both are honoured only while the device can hold a float16 proxy; the
 * composition reports the decision it actually made through composition_hdr_proxy_active(). Native
 * HIP is the exception: its network cannot consume the NGX linear-light float proxy, so the proxy
 * stays sRGB-encoded at either precision and @a hdr_transfer is honoured with an RGBA8 proxy too.
 *
 * @a network builds the surfaces for the in-layer network, which samples the proxy and answers into
 * the model image on the device: no transport pair, and a model image that takes storage writes. A
 * build for the other kind rebuilds every surface, keeping a held frame. Each build takes a generation
 * that no other build in the process had.
 *
 * The frame is at least kMinW x kMinH, as the layer passes smaller swapchains through, so the model's
 * raster, composition_model_extent(), is at most the frame's: the composition reduces the frame to
 * it and never enlarges it.
 *
 * @param c                The composition, or nullptr.
 * @param width            The frame's width, at least kMinW.
 * @param height           The frame's height, at least kMinH.
 * @param swapchain_format The swapchain's format.
 * @param s                The frame's settings.
 * @param linear_hdr       Whether the frame holds linear light.
 * @param hdr_proxy        Whether the proxy is to be float16.
 * @param hdr_transfer     1 if the swapchain carries PQ, otherwise 0.
 * @param network          Whether the in-layer network answers the frames.
 * @return                 true if the composition is built for the frame; otherwise its reason says
 *                         why not.
 */
extern bool
composition_prepare (struct composition                      *c,
                     uint32_t                                 width,
                     uint32_t                                 height,
                     VkFormat                                 swapchain_format,
                     struct composition_frame_settings const *s,
                     bool                                     linear_hdr,
                     bool                                     hdr_proxy,
                     uint32_t                                 hdr_transfer,
                     bool                                     network);

/** @brief The build of the composition's surfaces.
 *
 * @param c The composition, or nullptr.
 * @return  A number that no other build in the process had, or 0 while there are no surfaces and for
 *          nullptr.
 */
static inline uint64_t
composition_generation (struct composition const *c)
{
	return c ? c->generation : 0;
}

/** @brief The image that the in-layer network samples: the proxy at the model's raster.
 *
 * After leg 1 of a network build, it is in SHADER_READ_ONLY_OPTIMAL, and compute shaders may read it.
 *
 * @param c The composition, or nullptr.
 * @return  The work image, or the proxy when the model reads it whole; nullptr for nullptr.
 */
static inline struct composition_image const *
composition_network_input (struct composition const *c)
{
	if (!c)
		return nullptr;
	return c->work.image ? &c->work : &c->proxy;
}

/** @brief The image that the in-layer network answers into, which leg 2 composes.
 *
 * After leg 1 of a network build, it is in GENERAL, and compute shaders and transfers may write it.
 *
 * @param c The composition, or nullptr.
 * @return  The model image; nullptr for nullptr.
 */
static inline struct composition_image const *
composition_network_answer (struct composition const *c)
{
	return c ? &c->model : nullptr;
}

/** @brief The model's width.
 *
 * @param c The composition, or nullptr.
 * @return  The width, or 0 for nullptr.
 */
static inline uint32_t
composition_model_width (struct composition const *c)
{
	return c ? c->model_w : 0;
}

/** @brief The model's height.
 *
 * @param c The composition, or nullptr.
 * @return  The height, or 0 for nullptr.
 */
static inline uint32_t
composition_model_height (struct composition const *c)
{
	return c ? c->model_h : 0;
}

/** @brief The size of a frame that crosses: eight bytes a pixel while the float16 proxy is on, four
 *         when it is not -- every transport size in the layer derives from this one number so the two
 *         sides cannot disagree.
 *
 * @param c The composition, or nullptr.
 * @return  The size in bytes, or 0 for nullptr.
 */
static inline size_t
composition_model_bytes (struct composition const *c)
{
	if (!c)
		return 0;
	size_t const pixels = (size_t)c->model_w * c->model_h;
	return pixels * ((c->flags & COMPOSITION_HDR_PROXY) ? 8 : 4);
}

/** @brief Whether the proxy that composition_prepare() built is float16.
 *
 * @param c The composition, or nullptr.
 * @return  true if it is.
 */
static inline bool
composition_hdr_proxy_active (struct composition const *c)
{
	return c && (c->flags & COMPOSITION_HDR_PROXY);
}

/** @brief Native HIP: build the transport pair as exportable device-local memory the worker imports,
 *         so the proxy and the answer never leave VRAM.
 *
 * @param c      The composition, or nullptr.
 * @param family The queue family both legs run on.
 */
static inline void
composition_enable_export (struct composition *c,
                           uint32_t            family)
{
	if (!c)
		return;
	c->flags |= COMPOSITION_EXPORT;
	c->export_family = family;
}

/** @brief Whether the pair is exported memory the worker has not acknowledged: no frame may use it
 *         before an offer succeeds (composition_export_transport(), composition_set_transport_ready())
 *         or composition_disable_export() replaces it.
 *
 * @param c The composition, or nullptr.
 * @return  true while the pair awaits its acknowledgement.
 */
static inline bool
composition_transport_pending (struct composition const *c)
{
	return c && c->download.allocation && !(c->flags & COMPOSITION_TRANSPORT_READY);
}

/** @brief The acknowledged generation that requests name.
 *
 * @param c The composition, or nullptr.
 * @return  The generation, or 0 for the host transport.
 */
static inline uint32_t
composition_transport_generation (struct composition const *c)
{
	return c && (c->flags & COMPOSITION_TRANSPORT_READY) ? c->transport_gen : 0;
}

/** @brief The connection an offer of the pair awaits the daemon's answer on
 *         (composition_await_answer(), composition_withdraw_offer()).
 *
 * @param c The composition, or nullptr.
 * @return  The connection, which the composition closes, or -1 if there is none.
 */
static inline int
composition_offer_connection (struct composition const *c)
{
	return c ? c->offer : -1;
}

/** @brief Withdraws any offer and awaits the daemon's answer to a new one on a connection.
 *
 * @param c          The composition, or nullptr.
 * @param connection The connection, which the composition then owns, or which is closed at once
 *                   for nullptr; a negative value only withdraws.
 */
extern void
composition_await_answer (struct composition *c,
                          int                 connection);

/** @brief Closes the connection of an outstanding offer, if there is one.
 *
 * @param c The composition, or nullptr.
 */
extern void
composition_withdraw_offer (struct composition *c);

/** @brief Ends an outstanding offer with the daemon's answer.
 *
 * @param c     The composition, or nullptr.
 * @param ready true when the daemon imported the pair; false when it holds none, so the pair is
 *              offered again.
 */
extern void
composition_set_transport_ready (struct composition *c,
                                 bool                ready);

/** @brief Two new descriptors for the proxy and answer memory, and the rest of their offer, under a
 *         fresh generation number.
 *
 * @param c     The composition, or nullptr.
 * @param fds   Receives the descriptors, which the caller owns; both are -1 if the export fails.
 * @param offer Receives the device's and driver's UUIDs, the sizes and the generation.
 * @return      true if both descriptors were made; otherwise none is open. A pair that is not
 *              exported memory makes none.
 */
extern bool
composition_export_transport (struct composition       *c,
                              int                       fds[2],
                              struct ShmTransportOffer *offer);

/** @brief The worker cannot import the pair: carry frames through host staging from now on.
 *
 * A staging pair that cannot be made drops every surface, which the next composition_prepare()
 * builds again.
 *
 * @param c The composition, or nullptr.
 */
extern void
composition_disable_export (struct composition *c);

/** @brief Records leg 1.
 *
 * Leaves the proxy the model should see in the download buffer, and the swapchain image back in
 * PRESENT_SRC_KHR so a caller that gives up after this still presents something valid. For the
 * in-layer network it leaves the proxy in its image instead, which it hands to the network together
 * with the model image in one barrier (composition_network_input(), composition_network_answer()).
 *
 * @param c               The composition, or nullptr.
 * @param cb              The command buffer to record into.
 * @param swapchain_image The swapchain image, in PRESENT_SRC_KHR.
 * @param s               The frame's settings.
 * @return                true if leg 1 was recorded. false if a step could not be recorded: what was
 *                        is valid, and the composition's layouts and flags count it as run, so the
 *                        caller submits it.
 */
extern bool
composition_record_capture (struct composition                      *c,
                            VkCommandBuffer                          cb,
                            VkImage                                  swapchain_image,
                            struct composition_frame_settings const *s);

/** @brief The pixels leg 1 produced.
 *
 * @param c The composition, or nullptr.
 * @return  The host-visible proxy, or nullptr.
 */
static inline void const *
composition_proxy_pixels (struct composition const *c)
{
	return c ? c->download.mapped : nullptr;
}

/** @brief Whether the transport pair is exported memory.
 *
 * @param c The composition, or nullptr.
 * @return  true if both buffers are.
 */
static inline bool
composition_transport_exported (struct composition const *c)
{
	return c && c->download.allocation && c->upload.allocation;
}

/** @brief Where the model's answer goes before leg 2.
 *
 * @param c The composition, or nullptr.
 * @return  The host-visible answer, or nullptr.
 */
static inline void *
composition_model_pixels (struct composition *c)
{
	return c ? c->upload.mapped : nullptr;
}

/** @brief Records leg 2.
 *
 * Composes and leaves the swapchain image holding the result, in PRESENT_SRC_KHR. It takes the answer
 * from the upload buffer, or for the in-layer network from the model image, which one barrier takes
 * back from the network.
 *
 * @param c               The composition, or nullptr.
 * @param cb              The command buffer to record into.
 * @param swapchain_image The swapchain image, in PRESENT_SRC_KHR.
 * @param s               The frame's settings.
 * @return                true if leg 2 was recorded. false if a step could not be recorded: what was
 *                        is valid, leaves the swapchain image in PRESENT_SRC_KHR, and the
 *                        composition's layouts and flags count it as run, so the caller submits it.
 */
extern bool
composition_record_compose (struct composition                      *c,
                            VkCommandBuffer                          cb,
                            VkImage                                  swapchain_image,
                            struct composition_frame_settings const *s);

/** @brief Writes this many matched before/after pairs, at most CAPTURE_WRITER_FRAMES, starting with
 *         the next composed frame.
 *
 * @param c           The composition, or nullptr.
 * @param frames      The number of pairs.
 * @param control_seq The control sequence that asked for them.
 */
static inline void
composition_request_capture (struct composition *c,
                             uint32_t            frames,
                             uint32_t            control_seq)
{
	if (c)
		capture_writer_begin(&c->capture, frames, control_seq);
}

/** @brief Records the worker request that answered the frame, for the capture's manifest.
 *
 * @param c   The composition, or nullptr.
 * @param seq The request's sequence number.
 */
static inline void
composition_set_capture_inference (struct composition *c,
                                   uint32_t            seq)
{
	if (c)
		c->capture_metadata.inference_seq = seq;
}

/** @brief Whether this frame's compose recorded a pair; an active capture without its host buffer
 *         records none.
 *
 * @param c The composition, or nullptr.
 * @return  true if it did.
 */
static inline bool
composition_capture_recorded (struct composition const *c)
{
	return c && (c->flags & COMPOSITION_CAPTURE_RECORDED);
}

/** @brief Writes the pair that leg 2 recorded. Called after leg 2's fence, when the readback has
 *         landed.
 *
 * @param c The composition, or nullptr.
 */
extern void
composition_write_captured_frame (struct composition *c);

/** @brief Turns the tile grid the meter wrote into a white point. Called after leg 1's fence.
 *
 * @param c The composition, or nullptr.
 */
extern void
composition_consume_meter (struct composition *c);

/** @brief What the meter settled on. For the interface, so the number in use is visible rather than
 *         inferred.
 *
 * @param c The composition, or nullptr.
 * @return  The white point, or 0 when the meter has not taken a usable reading.
 */
static inline float
composition_measured_white_point (struct composition const *c)
{
	return c ? c->measured_white_point : 0.0f;
}

#endif /* DLSSLOP_AMD_LAYER_COMPOSITION_H_ */
