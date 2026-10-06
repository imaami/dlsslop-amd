/** @file
 *
 * Matched before/after frames, written so questions about this pass get settled by measurement.
 *
 * Upstream's reasoning, which is worth keeping: every comparison of a detail pass tends to be two
 * separate video captures -- different camera path, different exposure, and a codec in between
 * throwing away exactly the high-frequency detail the argument is about. What is wanted instead is
 * the same frames twice. The pass already holds the frame as the game presented it and the frame
 * after the model's edit, so both are written for one run of consecutive frames: a control with
 * nothing varying but the thing under test.
 *
 * Upstream writes raw because a codec is the confound. PNG is lossless, so it is not a confound, and
 * a file you can open is worth a great deal more than one you cannot -- so an 8-bit frame is written
 * as PNG. A 10-bit or float frame has no PNG that can hold it and is written raw, as upstream does,
 * with a manifest saying how to read it. The manifest is written either way.
 */
#ifndef DLSSLOP_AMD_LAYER_CAPTURE_H_
#define DLSSLOP_AMD_LAYER_CAPTURE_H_

#include <stddef.h>
#include <stdint.h>

/** @brief The most frames that one batch writes. */
static constexpr uint32_t CAPTURE_WRITER_FRAMES = 64;

/** @brief What the manifest says about one frame.
 *
 * The renderer's settings come from the frame's snapshot. inference_seq is the worker request that
 * answered the frame; passes is the layer's observed requested count.
 */
struct capture_metadata {
	uint64_t before_hash;       //!< The frame as presented; capture_writer_write_frame() sets it.
	uint32_t frame_control_seq;
	uint32_t tuning_seq;
	uint32_t inference_seq;
	uint32_t passes;
	uint32_t debug_view;
	uint32_t apply_model;
	uint32_t bypass;
	uint32_t hold;
	uint32_t compare;
	uint32_t transfer;
	uint32_t model_width;
	uint32_t model_height;
	uint32_t hdr_proxy;
	uint32_t linear_hdr;
	uint32_t hdr_transfer;
	float    detail;
	float    color;
	float    debug_scale;
};

/** @brief A batch of matched frames that is being written.
 *
 * A zeroed writer is an idle one. While a batch is being written, the writer holds its directory's
 * path on the heap, which capture_writer_fini() frees. control_seq is 64 bits wide, which fills the
 * padding that a narrower member would leave.
 */
struct capture_writer {
	struct capture_metadata metadata[CAPTURE_WRITER_FRAMES]; //!< Each frame written so far.
	uint64_t                control_seq;                     //!< The control sequence of the request.
	char                   *batch_dir;                       //!< The batch's directory, or nullptr.
	size_t                  batch_length;                    //!< The length of batch_dir.
	size_t                  batch_name;                      //!< Where the batch's own name starts in batch_dir.
	uint32_t                remaining;                       //!< Frames still to write; 0: idle.
	uint32_t                index;                           //!< Frames written, and the next one's number.
};

/** @brief Returns a metadata record with the renderer's defaults.
 *
 * @return A record whose debug_scale is 1 and whose other fields are 0.
 */
static inline struct capture_metadata
capture_metadata (void)
{
	struct capture_metadata ret = {0};
	ret.debug_scale = 1.0f;
	return ret;
}

/** @brief Writes where captures go: $XDG_STATE_HOME/dlssnr/captures, or
 *         ~/.local/state/dlssnr/captures, or /tmp/dlssnr-captures.
 *
 * Each request uses a unique batch directory in it. Prior captures are preserved.
 *
 * @param buf  Receives the directory; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the directory's length, which is @a size or more if @a buf
 *             holds only the start of it.
 */
extern int
capture_writer_directory (char   *buf,
                          size_t  size);

/** @brief Starts a batch in a new directory under capture_writer_directory(), which it creates.
 *
 * A batch that is being written is abandoned. If the directory cannot be created, the writer logs
 * that and stays idle.
 *
 * @param w           The writer, or nullptr.
 * @param frames      How many frames to write; at most CAPTURE_WRITER_FRAMES are. 0 does nothing.
 * @param control_seq The control sequence that asked for the batch, for the manifest.
 */
extern void
capture_writer_begin (struct capture_writer *w,
                      uint32_t               frames,
                      uint32_t               control_seq);

/** @brief Whether a batch is being written.
 *
 * @param w The writer, or nullptr.
 * @return  true if frames remain to be written.
 */
static inline bool
capture_writer_active (struct capture_writer const *w)
{
	return w && w->remaining;
}

/** @brief Writes one frame's pair, and the manifest after the batch's last frame.
 *
 * The writer decides between PNG and raw from the format, and reports what it chose in the
 * manifest. The manifest is written in the batch's directory, then linked and renamed to
 * manifest.txt beside it. If a file cannot be written, the batch ends without the manifest.
 *
 * @param w         The writer, or nullptr. An idle writer does nothing.
 * @param before    The frame as the game presented it, or nullptr to do nothing.
 * @param after     The frame with the model's edit composed onto it, or nullptr to do nothing.
 * @param width     The frames' width.
 * @param height    The frames' height.
 * @param vk_format The VkFormat both frames are in.
 * @param metadata  What the manifest says about the frame, or nullptr to do nothing. The writer
 *                  sets the copy's before_hash.
 */
extern void
capture_writer_write_frame (struct capture_writer         *w,
                            void const                    *before,
                            void const                    *after,
                            uint32_t                       width,
                            uint32_t                       height,
                            uint32_t                       vk_format,
                            struct capture_metadata const *metadata);

/** @brief Abandons the batch that is being written and frees the writer's path, then leaves the
 *         writer idle.
 *
 * @param dest The writer, or nullptr.
 */
extern void
capture_writer_fini (struct capture_writer *dest);

#endif /* DLSSLOP_AMD_LAYER_CAPTURE_H_ */
