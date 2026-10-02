/** @file
 *
 * The capture writer: matched frames as PNG or raw, and the manifest that says how to read them.
 */
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "capture.h"
#include "log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb/stb_image_write.h"

/** @brief The size of a path in the batch's directory: the directory, which fits in
 *         CAPTURE_WRITER_DIR_SIZE bytes, then a slash and a file name of at most 13 bytes. A path
 *         that the kernel refuses as too long therefore fails where the file is opened, as
 *         upstream's did.
 */
static constexpr size_t BATCH_PATH_MAX = CAPTURE_WRITER_DIR_SIZE + 16;

/** @brief How the frames of one format are written. */
struct encoding {
	uint8_t bytes_per_pixel; //!< The frame's bytes per pixel.
	bool    png;             //!< Four 8-bit channels, which PNG holds; otherwise raw.
	bool    swap;            //!< Red and blue swap places for PNG's RGBA byte order.
};

/** @brief How a format's frames are written.
 *
 * Only four 8-bit channels make a PNG. A B8G8R8A8 surface has red and blue the other way round from
 * PNG's RGBA. Everything but four 16-bit floats has 4 bytes per pixel.
 *
 * @param vk_format A VkFormat.
 * @return          How its frames are written.
 */
static struct encoding
encoding (uint32_t vk_format)
{
	switch (vk_format) {
	case VK_FORMAT_B8G8R8A8_UNORM:
		return (struct encoding){ .bytes_per_pixel = 4, .png = true, .swap = true };
	case VK_FORMAT_R8G8B8A8_UNORM:
	case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
		return (struct encoding){ .bytes_per_pixel = 4, .png = true };
	case VK_FORMAT_R16G16B16A16_SFLOAT:
		return (struct encoding){ .bytes_per_pixel = 8 };
	default:
		return (struct encoding){ .bytes_per_pixel = 4 };
	}
}

/** @brief Creates a directory and each directory above it, as mkdir -p does, but ignores errors.
 *
 * @param path A nonempty path, which is restored before the function returns.
 */
static void
make_dirs (char *path)
{
	for (char *slash = path; (slash = strchr(slash + 1, '/'));) {
		*slash = '\0';
		mkdir(path, 0700);
		*slash = '/';
	}
	mkdir(path, 0700);
}

/** @brief Writes four 8-bit channels as a PNG.
 *
 * The pixels are copied first, so the encoder reads cached memory rather than the mapped readback
 * buffer, and so red and blue can swap places.
 *
 * @param path   The file.
 * @param pixels The pixels, tightly packed.
 * @param width  Their width.
 * @param height Their height.
 * @param swap   Whether red and blue swap places.
 * @return       true if the file was written.
 */
static bool
write_png (char const *path,
           void const *pixels,
           uint32_t    width,
           uint32_t    height,
           bool        swap)
{
	size_t const bytes = (size_t)width * height * 4;
	uint8_t *rgba = malloc(bytes);
	if (!rgba)
		return false;

	memcpy(rgba, pixels, bytes);
	if (swap) {
		for (size_t i = 0; i < bytes; i += 4) {
			uint8_t const red = rgba[i + 2];
			rgba[i + 2] = rgba[i];
			rgba[i] = red;
		}
	}
	bool const wrote = stbi_write_png(path, (int)width, (int)height, 4, rgba, (int)width * 4);
	free(rgba);
	rgba = nullptr;
	return wrote;
}

/** @brief Writes bytes as they are.
 *
 * @param path   The file.
 * @param pixels The bytes.
 * @param bytes  How many.
 * @return       true if the file was written and closed.
 */
static bool
write_raw (char const *path,
           void const *pixels,
           size_t      bytes)
{
	FILE *const f = fopen(path, "wb");
	if (!f)
		return false;

	bool const wrote = fwrite(pixels, 1, bytes, f) == bytes;
	bool const closed = !fclose(f);
	return wrote && closed;
}

/** @brief Writes one image of the pair, and logs a failure.
 *
 * @param w      The writer; the batch's directory and the frame's number name the file.
 * @param side   "before" or "after".
 * @param pixels The image.
 * @param width  Its width.
 * @param height Its height.
 * @param e      How it is written.
 * @param bytes  Its size.
 * @return       true if the file was written.
 */
static bool
write_image (struct capture_writer const *w,
             char const                  *side,
             void const                  *pixels,
             uint32_t                     width,
             uint32_t                     height,
             struct encoding              e,
             size_t                       bytes)
{
	char path[BATCH_PATH_MAX];
	snprintf(path, sizeof path, "%s/%s_%02u.%s", w->batch_dir, side, w->index,
	         e.png ? "png" : "raw");
	bool const wrote = e.png ? write_png(path, pixels, width, height, e.swap)
	                         : write_raw(path, pixels, bytes);
	if (!wrote)
		log_printf("[capture] could not write %s", path);
	return wrote;
}

/** @brief FNV-1a over 8-byte words, then the tail bytes: only equality matters.
 *
 * @param pixels The bytes.
 * @param bytes  How many.
 * @return       The hash.
 */
static uint64_t
hash_bytes (void const *pixels,
            size_t      bytes)
{
	uint8_t const *const p = pixels;
	uint64_t hash = UINT64_C(14695981039346656037);
	size_t i = 0;
	for (; i + 8 <= bytes; i += 8) {
		uint64_t word;
		memcpy(&word, p + i, sizeof word);
		hash = (hash ^ word) * UINT64_C(1099511628211);
	}
	for (; i < bytes; ++i)
		hash = (hash ^ p[i]) * UINT64_C(1099511628211);
	return hash;
}

/** @brief Writes one frame's metadata to the manifest.
 *
 * @param f      The manifest.
 * @param prefix What precedes each key.
 * @param m      The metadata.
 */
static void
write_metadata (FILE                          *f,
                char const                    *prefix,
                struct capture_metadata const *m)
{
#define CAPTURE_U(name, field) fprintf(f, "%s" name " %u\n", prefix, m->field)
#define CAPTURE_F(name, field) fprintf(f, "%s" name " %.9g\n", prefix, (double)m->field)
	CAPTURE_U("frame_control_seq", frame_control_seq);
	CAPTURE_U("tuning_seq", tuning_seq);
	CAPTURE_U("inference_seq", inference_seq);
	CAPTURE_U("passes", passes);
	CAPTURE_U("debug_view", debug_view);
	CAPTURE_U("apply_model", apply_model);
	CAPTURE_U("bypass", bypass);
	CAPTURE_U("hold", hold);
	CAPTURE_U("compare", compare);
	CAPTURE_U("transfer", transfer);
	CAPTURE_U("model_width", model_width);
	CAPTURE_U("model_height", model_height);
	CAPTURE_U("hdr_proxy", hdr_proxy);
	CAPTURE_U("linear_hdr", linear_hdr);
	CAPTURE_U("hdr_transfer", hdr_transfer);
	CAPTURE_F("detail", detail);
	CAPTURE_F("color", color);
	CAPTURE_F("debug_scale", debug_scale);
#undef CAPTURE_F
#undef CAPTURE_U
	fprintf(f, "%sbefore_hash %016" PRIx64 "\n", prefix, m->before_hash);
}

/** @brief Writes the batch's manifest, then publishes it as manifest.txt beside the batch.
 *
 * The immutable batch is complete before the public completion pointer is replaced. A hard link
 * plus rename avoids partial manifests and never overwrites any previous batch's images.
 *
 * @param w         The writer, which has written at least one frame.
 * @param width     The frames' width.
 * @param height    The frames' height.
 * @param vk_format Their VkFormat.
 * @param e         How they were written.
 * @return          true if the manifest was written and published.
 */
static bool
capture_writer_write_manifest (struct capture_writer const *w,
                               uint32_t                     width,
                               uint32_t                     height,
                               uint32_t                     vk_format,
                               struct encoding              e)
{
	char path[BATCH_PATH_MAX];
	snprintf(path, sizeof path, "%s/manifest.txt", w->batch_dir);
	FILE *const f = fopen(path, "w");
	if (!f)
		return false;

	fprintf(f, "capture_metadata_version 2\n");
	fprintf(f, "capture_control_seq %u\n", w->control_seq);
	fprintf(f, "batch_dir %s\n", w->batch_dir + w->batch_name);
	fprintf(f, "frames %u\n", w->index);
	fprintf(f, "width %u\nheight %u\n", width, height);
	fprintf(f, "vk_format %u\n", vk_format);
	fprintf(f, "encoding %s\n", e.png ? "png" : "raw");
	fprintf(f, "bytes_per_pixel %u\n", e.bytes_per_pixel);
	fprintf(f, "row_pitch %zu\n", (size_t)e.bytes_per_pixel * width);
	// The first frame's metadata without a prefix, then every frame's with one.
	write_metadata(f, "", &w->metadata[0]);
	for (uint32_t i = 0; i < w->index; ++i) {
		char prefix[24];
		snprintf(prefix, sizeof prefix, "frame_%u_", i);
		write_metadata(f, prefix, &w->metadata[i]);
	}
	fputs("\n"
	      "before_NN is the frame as the game presented it; after_NN is the same frame with\n"
	      "the model's edit composed onto it. Same frame, same run, one variable.\n", f);
	bool const written = !ferror(f);
	if (fclose(f) || !written)
		return false;

	char pending[BATCH_PATH_MAX];
	snprintf(pending, sizeof pending, "%s/published.tmp", w->batch_dir);
	if (link(path, pending))
		return false;

	snprintf(path, sizeof path, "%.*smanifest.txt", (int)w->batch_name, w->batch_dir);
	if (rename(pending, path)) {
		unlink(pending);
		return false;
	}
	return true;
}

int
capture_writer_directory (char   *buf,
                          size_t  size)
{
	char const *const state = getenv("XDG_STATE_HOME");
	if (state && *state)
		return snprintf(buf, size, "%s/dlssnr/captures", state);

	char const *const home = getenv("HOME");
	if (home && *home)
		return snprintf(buf, size, "%s/.local/state/dlssnr/captures", home);

	return snprintf(buf, size, "/tmp/dlssnr-captures");
}

/** @brief Creates the batch's directory, named capture-PID-XXXXXX, in the capture directory.
 *
 * @param w The writer, whose batch_dir receives the batch's directory.
 * @return  The capture directory's length, or 0 if the batch's directory was not created; then
 *          batch_dir holds the capture directory, cut to fit.
 */
static size_t
make_batch_dir (struct capture_writer *w)
{
	size_t const size = sizeof w->batch_dir;
	char *const dir = w->batch_dir;
	int const dir_length = capture_writer_directory(dir, size);
	if ((size_t)dir_length >= size)
		return 0;

	make_dirs(dir);
	size_t const length = (size_t)dir_length;
	int const name_length = snprintf(dir + length, size - length, "/capture-%d-XXXXXX",
	                                 (int)getpid());
	if ((size_t)name_length >= size - length || !mkdtemp(dir)) {
		dir[length] = '\0';
		return 0;
	}
	return length;
}

void
capture_writer_begin (struct capture_writer *w,
                      uint32_t               frames,
                      uint32_t               control_seq)
{
	if (!w || !frames)
		return;

	size_t const dir_length = make_batch_dir(w);
	if (!dir_length) {
		log_printf("[capture] cannot create batch directory in %s", w->batch_dir);
		w->remaining = 0;
		return;
	}
	w->batch_name = (uint32_t)dir_length + 1;
	w->control_seq = control_seq;
	w->remaining = frames < CAPTURE_WRITER_FRAMES ? frames : CAPTURE_WRITER_FRAMES;
	w->index = 0;
	log_printf("[capture] capturing %u frames, control %u, to %s", w->remaining, control_seq,
	           w->batch_dir);
}

void
capture_writer_write_frame (struct capture_writer         *w,
                            void const                    *before,
                            void const                    *after,
                            uint32_t                       width,
                            uint32_t                       height,
                            uint32_t                       vk_format,
                            struct capture_metadata const *metadata)
{
	if (!w || !w->remaining || !before || !after || !metadata)
		return;

	struct encoding const e = encoding(vk_format);
	size_t const bytes = (size_t)width * height * e.bytes_per_pixel;
	bool const wrote_before = write_image(w, "before", before, width, height, e, bytes);
	bool const wrote_after = write_image(w, "after", after, width, height, e, bytes);
	if (!wrote_before || !wrote_after) {
		w->remaining = 0;
		log_printf("[capture] batch failed; completion manifest was not published");
		return;
	}

	struct capture_metadata *const m = &w->metadata[w->index++];
	*m = *metadata;
	m->before_hash = hash_bytes(before, bytes);
	if (--w->remaining)
		return;

	if (capture_writer_write_manifest(w, width, height, vk_format, e))
		log_printf("[capture] wrote %u pairs to %s", w->index, w->batch_dir);
	else
		log_printf("[capture] could not publish completion manifest for %s", w->batch_dir);
}
