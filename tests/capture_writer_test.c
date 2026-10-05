/** @file
 *
 * The channel's paths and the capture writer on the host: the shared runtime directory and the
 * channels' and transport's paths, a text field of the channel, where captures go, the batches'
 * directories, manifests and files, a failed write, a cut batch, capture directories that cannot be
 * created, a batch that publishes where it began, finishing, and the log's lines about all that.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "capture.h"
#include "shm_protocol.h"
#include "support.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
#undef STBI_ONLY_PNG
#undef STB_IMAGE_IMPLEMENTATION

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("capture-writer-test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief A file's bytes. */
struct file_text {
	char   *text;   //!< The bytes and a terminating null.
	size_t  length; //!< Their number.
};

/** @brief Reads a file, which must be readable.
 *
 * @param path The file.
 * @return     The file's bytes, which the caller frees with file_text_fini().
 */
static struct file_text
file_text (char const *path)
{
	struct file_text ret = {};
	ret.text = support_read_file(path, &ret.length);
	require(ret.text, "cannot read expected capture file");
	return ret;
}

/** @brief Frees a file's bytes.
 *
 * @param dest The bytes.
 */
static void
file_text_fini (struct file_text *dest)
{
	free(dest->text);
	*dest = (struct file_text){};
}

/** @brief Whether two files' bytes are the same.
 *
 * @param a The first file's bytes.
 * @param b The second's.
 * @return  true if they are.
 */
static bool
same_text (struct file_text const *a,
           struct file_text const *b)
{
	return a->length == b->length && !memcmp(a->text, b->text, a->length);
}

/** @brief Whether a file's bytes are a string.
 *
 * @param f    The file's bytes.
 * @param text The string.
 * @return     true if they are.
 */
static bool
text_is (struct file_text const *f,
         char const             *text)
{
	size_t const length = strlen(text);
	return f->length == length && !memcmp(f->text, text, length);
}

/** @brief A manifest field's value, in the manifest. */
struct field_value {
	char const *start;  //!< Its first byte.
	int         length; //!< Its length, as %.*s takes it.
};

/** @brief A manifest field's value: what follows the first "KEY ", which must start a line, up to
 *         the end of the line.
 *
 * @param manifest The manifest.
 * @param key      The field's name.
 * @return         Its value.
 */
static struct field_value
field (struct file_text const *manifest,
       char const             *key)
{
	size_t const key_length = strlen(key);
	char const *found = manifest->text;
	while ((found = strstr(found, key)) && found[key_length] != ' ')
		++found;
	require(found && (found == manifest->text || found[-1] == '\n'), "missing manifest field");
	char const *const start = found + key_length + 1;
	char const *end = strchr(start, '\n');
	if (!end)
		end = manifest->text + manifest->length;
	return (struct field_value){start, (int)(end - start)};
}

/** @brief Whether a manifest field's value is a string.
 *
 * @param manifest The manifest.
 * @param key      The field's name.
 * @param value    The string.
 * @return         true if it is.
 */
static bool
field_is (struct file_text const *manifest,
          char const             *key,
          char const             *value)
{
	struct field_value const f = field(manifest, key);
	size_t const length = strlen(value);
	return (size_t)f.length == length && !memcmp(f.start, value, length);
}

/** @brief Formats a path on the heap, which must succeed.
 *
 * @param length Receives the path's length, or nullptr.
 * @param fmt    A printf format.
 * @param ...    Its arguments.
 * @return       The path, which the caller frees.
 */
[[gnu::format(printf, 2, 3)]]
static char *
path_of (size_t     *length,
         char const *fmt,
         ...)
{
	va_list args;
	va_start(args, fmt);
	char *const path = support_vformat(length, fmt, args);
	va_end(args);
	require(path, "a path cannot be formatted");
	return path;
}

/** @brief Writes a file, which must succeed.
 *
 * @param path The file.
 * @param text What it holds.
 */
static void
write_file (char const *path,
            char const *text)
{
	FILE *const f = fopen(path, "wbe");
	require(f, "cannot write %s", path);
	bool const written = fputs(text, f) >= 0;
	require(!fclose(f) && written, "cannot write %s", path);
}

/** @brief Whether a path names a directory, following symbolic links.
 *
 * @param path The path.
 * @return     true if it does.
 */
static bool
is_directory (char const *path)
{
	struct stat st;
	return !stat(path, &st) && S_ISDIR(st.st_mode);
}

/** @brief Whether a directory is empty.
 *
 * @param path The directory.
 * @return     true if it holds nothing; false if it holds something or cannot be read.
 */
static bool
is_empty (char const *path)
{
	DIR *const dir = opendir(path);
	if (!dir)
		return false;
	bool empty = true;
	for (struct dirent *entry; empty && (entry = readdir(dir));)
		empty = !strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..");
	return !closedir(dir) && empty;
}

/** @brief The directories in a directory, by name. */
struct directories {
	char   **names; //!< The names, each on the heap.
	size_t   count; //!< Their number.
};

/** @brief The directories in a directory, which must be readable.
 *
 * @param path The directory.
 * @return     Its directories, which the caller frees with directories_fini().
 */
static struct directories
directories (char const *path)
{
	DIR *const dir = opendir(path);
	require(dir, "cannot list the capture directory");
	int const fd = dirfd(dir);
	require(fd >= 0, "cannot list the capture directory");
	struct directories ret = {};
	for (struct dirent *entry; (entry = readdir(dir));) {
		struct stat st;
		if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")
		    || fstatat(fd, entry->d_name, &st, 0) || !S_ISDIR(st.st_mode))
			continue;
		char **const grown = realloc(ret.names, (ret.count + 1) * sizeof *grown);
		require(grown, "out of memory");
		ret.names = grown;
		ret.names[ret.count] = strdup(entry->d_name);
		require(ret.names[ret.count], "out of memory");
		++ret.count;
	}
	require(!closedir(dir), "cannot list the capture directory");
	return ret;
}

/** @brief Whether a list of directories names one.
 *
 * @param d    The list.
 * @param name The name.
 * @return     true if it does.
 */
static bool
directories_have (struct directories const *d,
                  char const               *name)
{
	for (size_t i = 0; i < d->count; ++i)
		if (!strcmp(d->names[i], name))
			return true;
	return false;
}

/** @brief Frees a list of directories.
 *
 * @param dest The list.
 */
static void
directories_fini (struct directories *dest)
{
	for (size_t i = 0; i < dest->count; ++i) {
		free(dest->names[i]);
		dest->names[i] = nullptr;
	}
	free(dest->names);
	*dest = (struct directories){};
}

/** @brief Ends the test unless a C form wrote what fits of a path and returned the length of the
 *         whole path, as snprintf() does.
 *
 * @param length      What the C form returned.
 * @param buffer      What it wrote into eight bytes.
 * @param path        The whole path.
 * @param path_length The length of @a path.
 * @param message     What failed.
 */
static void
check_truncation (int         length,
                  char const  buffer[static 8],
                  char const *path,
                  size_t      path_length,
                  char const *message)
{
	size_t const kept = path_length < 7 ? path_length : 7;
	require(length >= 0 && (size_t)length == path_length && strlen(buffer) == kept
	        && !memcmp(buffer, path, kept), "%s", message);
}

/** @brief The whole string that a C form writes.
 *
 * @param form   A C form, which writes what fits and returns the length of the whole string, as
 *               snprintf() does.
 * @param name   What it writes, for a message.
 * @param length Receives the string's length, or nullptr.
 * @return       The string, which the caller frees.
 */
static char *
whole (int        (*form)(char *, size_t),
       char const  *name,
       size_t      *length)
{
	int const n = form(nullptr, 0);
	require(n >= 0, "%s cannot be formatted", name);
	size_t const size = (size_t)n + 1;
	char *const text = malloc(size);
	require(text, "out of memory");
	require(form(text, size) == n && strlen(text) == size - 1, "%s's length changed", name);
	if (length)
		*length = size - 1;
	return text;
}

/** @brief Ends the test unless the whole string that a C form writes is the one wanted.
 *
 * @param form    The C form.
 * @param want    The string wanted.
 * @param message What failed.
 */
static void
expect_whole (int        (*form)(char *, size_t),
              char const  *want,
              char const  *message)
{
	char *got = whole(form, message, nullptr);
	require(!strcmp(got, want), "%s", message);
	free(got);
	got = nullptr;
}

/** @brief Restores a variable of the environment.
 *
 * @param variable The variable.
 * @param value    Its value, or nullptr to unset it.
 * @param message  What failed.
 */
static void
restore (char const *variable,
         char const *value,
         char const *message)
{
	require(!(value ? setenv(variable, value, 1) : unsetenv(variable)), "%s", message);
}

/** @brief The channel's and the transport's paths. */
static void
check_channel_paths (void)
{
	char const *inherited = getenv("DLSSNR_UID");
	char *original_uid = inherited ? strdup(inherited) : nullptr;
	require(!inherited || original_uid, "out of memory");
	inherited = getenv("DLSSNR_SHM");
	char *original_channel = inherited ? strdup(inherited) : nullptr;
	require(!inherited || original_channel, "out of memory");
	unsigned const uid = (unsigned)getuid();
	char *shared_dir = path_of(nullptr, "/tmp/dlssnr-%u", uid);
	char *shared_path = path_of(nullptr, "/tmp/dlssnr-%u/" kShmChannelName, uid);
	char *native_path = path_of(nullptr, "/tmp/dlsslop-amd-%u/" kShmChannelName, uid);
	char small[8];

	// The default channels' file names the protocol version.
	char *name = path_of(nullptr, "shm-v%u.bin", (unsigned)kShmVersion);
	require(!strcmp(name, kShmChannelName), "the channel's name %s does not name v%u", kShmChannelName,
	        (unsigned)kShmVersion);
	free(name);
	name = nullptr;

	require(!unsetenv("DLSSNR_UID"), "unsetenv failed");
	expect_whole(ShmRuntimeDir, shared_dir, "shared runtime default changed");
	expect_whole(ShmDefaultPath, shared_path, "shared channel default changed");
	expect_whole(ShmNativeDefaultPath, native_path, "native channel default changed");
	require(!setenv("DLSSNR_UID", "12345", 1), "setenv failed");
	expect_whole(ShmRuntimeDir, "/tmp/dlssnr-12345", "shared runtime ignored DLSSNR_UID");
	static char const uid_path[] = "/tmp/dlssnr-12345/" kShmChannelName;
	expect_whole(ShmDefaultPath, uid_path, "shared channel ignored DLSSNR_UID");
	check_truncation(ShmDefaultPath(small, sizeof small), small, uid_path, sizeof uid_path - 1,
	                 "the shared channel's C form did not truncate as snprintf does");
	expect_whole(ShmNativeDefaultPath, native_path, "DLSSNR_UID changed the native channel");
	require(!setenv("DLSSNR_UID", "", 1), "setenv failed");
	expect_whole(ShmRuntimeDir, shared_dir, "empty DLSSNR_UID changed the default");
	restore("DLSSNR_UID", original_uid, "restoring DLSSNR_UID failed");

	// The native channel: nonempty DLSSNR_SHM, at any length, otherwise the native default.
	static char const head[] = "/tmp/";
	static char const tail[] = "/shm.bin";
	size_t const head_length = sizeof head - 1;
	size_t const long_length = head_length + 5000 + sizeof tail - 1;
	char *long_path = malloc(long_length + 1);
	require(long_path, "out of memory");
	memcpy(long_path, head, head_length);
	memset(long_path + head_length, 'x', 5000);
	memcpy(long_path + head_length + 5000, tail, sizeof tail);
	require(!setenv("DLSSNR_SHM", long_path, 1), "setenv failed");
	expect_whole(ShmNativeChannelPath, long_path, "the native channel ignored a long DLSSNR_SHM");
	check_truncation(ShmNativeChannelPath(small, sizeof small), small, long_path, long_length,
	                 "the native channel's C form did not truncate as snprintf does");
	size_t socket_length = 0;
	char *socket = path_of(&socket_length, "%s.sock", long_path);
	int const transport_length = ShmTransportPath(nullptr, 0, long_path);
	require(transport_length >= 0 && (size_t)transport_length == socket_length, "the transport path changed");
	char *transport = malloc(socket_length + 1);
	require(transport, "out of memory");
	require(ShmTransportPath(transport, socket_length + 1, long_path) == transport_length
	        && !strcmp(transport, socket), "the transport path changed");
	check_truncation(ShmTransportPath(small, sizeof small, long_path), small, socket, socket_length,
	                 "the transport path's C form did not truncate as snprintf does");
	require(!setenv("DLSSNR_SHM", "", 1), "setenv failed");
	expect_whole(ShmNativeChannelPath, native_path, "an empty DLSSNR_SHM changed the native channel");
	restore("DLSSNR_SHM", original_channel, "restoring DLSSNR_SHM failed");

	free(transport);
	transport = nullptr;
	free(socket);
	socket = nullptr;
	free(long_path);
	long_path = nullptr;
	free(native_path);
	native_path = nullptr;
	free(shared_path);
	shared_path = nullptr;
	free(shared_dir);
	shared_dir = nullptr;
	free(original_channel);
	original_channel = nullptr;
	free(original_uid);
	original_uid = nullptr;
}

/** @brief A text field's sequence number is bumped after each write, and a load takes the string:
 *         one cut to the field's size, and an empty one for a null string.
 */
static void
check_text_fields (void)
{
	static struct ShmHeader header;
	char text[kReasonBytes];
	ShmStoreString(&header, SHM_TEXT_LAYER_REASON, "first");
	require(atomic_load(&header.layerReasonSeq) == 1
	        && ShmLoadString(&header, SHM_TEXT_LAYER_REASON, text, sizeof text) && !strcmp(text, "first"),
	        "a text field was not published");
	char too_long[kReasonBytes + 1];
	memset(too_long, 'x', kReasonBytes);
	too_long[kReasonBytes] = '\0';
	ShmStoreString(&header, SHM_TEXT_LAYER_REASON, too_long);
	require(atomic_load(&header.layerReasonSeq) == 2
	        && ShmLoadString(&header, SHM_TEXT_LAYER_REASON, text, sizeof text)
	        && strlen(text) == kReasonBytes - 1 && !memcmp(text, too_long, kReasonBytes - 1),
	        "a text field did not cut a string to its size");
	ShmStoreString(&header, SHM_TEXT_LAYER_REASON, nullptr);
	require(atomic_load(&header.layerReasonSeq) == 3
	        && ShmLoadString(&header, SHM_TEXT_LAYER_REASON, text, sizeof text) && !*text
	        && !header.layerReason[0],
	        "a null string did not empty a text field");
}

/** @brief capture_writer_directory() as a whole string. The C form writes what fits and returns the
 *         length of the whole path, as snprintf does.
 *
 * @return The directory, which the caller frees.
 */
static char *
capture_directory (void)
{
	size_t length = 0;
	char *const dir = whole(capture_writer_directory, "the capture directory", &length);
	char small[8];
	check_truncation(capture_writer_directory(small, sizeof small), small, dir, length,
	                 "the capture directory's C form did not truncate as snprintf does");
	return dir;
}

/** @brief Ends the test unless the capture directory is the one wanted.
 *
 * @param want    The directory wanted.
 * @param message What failed.
 */
static void
expect_capture_directory (char const *want,
                          char const *message)
{
	char *dir = capture_directory();
	require(!strcmp(dir, want), "%s", message);
	free(dir);
	dir = nullptr;
}

/** @brief One frame's lines in the first batch's manifest. */
static char const FIRST_FRAME[] =
	"frame_control_seq 123\ntuning_seq 0\ninference_seq 55\npasses 1\ndebug_view 2\napply_model 0\n"
	"bypass 0\nhold 1\ncompare 0\ntransfer 0\nmodel_width 0\nmodel_height 0\nhdr_proxy 0\nlinear_hdr 0\n"
	"hdr_transfer 0\ndetail 0.5\ncolor 0.25\ndebug_scale 1\nbefore_hash c2de31fd48ac5f39\n";

/** @brief What the published manifest of the first batch below holds, with its batch directory.
 *
 * @param batch  The batch directory's name.
 * @param length Receives the manifest's length.
 * @return       The manifest, which the caller frees.
 */
static char *
first_manifest (struct field_value  batch,
                size_t             *length)
{
	// The last frame's lines unprefixed, then each frame's after its prefix; both prefixes are 8 bytes.
	static char const prefixes[][9] = {"frame_0_", "frame_1_"};
	size_t const prefix_count = sizeof prefixes / sizeof *prefixes;
	size_t const prefix_length = sizeof *prefixes - 1;
	size_t const frame_length = sizeof FIRST_FRAME - 1;
	size_t lines = 0;
	for (size_t i = 0; i < frame_length; ++i)
		lines += FIRST_FRAME[i] == '\n';
	size_t const frames_length = frame_length + prefix_count * (frame_length + lines * prefix_length);
	char *frames = malloc(frames_length + 1);
	require(frames, "out of memory");
	memcpy(frames, FIRST_FRAME, frame_length);
	size_t at = frame_length;
	for (size_t p = 0; p < prefix_count; ++p) {
		for (char const *line = FIRST_FRAME; *line;) {
			char const *const end = strchr(line, '\n') + 1;
			size_t const line_length = (size_t)(end - line);
			memcpy(frames + at, prefixes[p], prefix_length);
			memcpy(frames + at + prefix_length, line, line_length);
			at += prefix_length + line_length;
			line = end;
		}
	}
	frames[at] = '\0';
	require(at == frames_length, "the first manifest's frames were miscounted");

	char *const manifest = path_of(length,
	                               "capture_metadata_version 2\ncapture_control_seq 123\nbatch_dir %.*s\n"
	                               "frames 2\nwidth 1\nheight 1\nvk_format 44\nencoding png\n"
	                               "bytes_per_pixel 4\nrow_pitch 4\n%s\n"
	                               "before_NN is the frame as the game presented it; after_NN is the "
	                               "same frame with\n"
	                               "the model's edit composed onto it. Same frame, same run, one variable.\n",
	                               batch.length, batch.start, frames);
	free(frames);
	frames = nullptr;
	return manifest;
}

/** @brief Whether the log holds a line, after the log's prefix.
 *
 * @param log The log.
 * @param cut The most bytes of the line that the log holds; SIZE_MAX for all of them.
 * @param fmt A printf format of the line.
 * @param ... Its arguments.
 * @return    true if it does.
 */
[[gnu::format(printf, 3, 4)]]
static bool
logged (struct file_text const *log,
        size_t                  cut,
        char const             *fmt,
        ...)
{
	va_list args;
	va_start(args, fmt);
	size_t length = 0;
	char *line = support_vformat(&length, fmt, args);
	va_end(args);
	require(line, "a line cannot be formatted");
	if (length > cut)
		line[cut] = '\0';

	size_t needle_length = 0;
	char *needle = path_of(&needle_length, "\n[dlssnr-layer] %s\n", line);
	bool const found = !strncmp(log->text, needle + 1, needle_length - 1) || strstr(log->text, needle);
	free(needle);
	needle = nullptr;
	free(line);
	line = nullptr;
	return found;
}

/** @brief The test's directory, removed with what it holds when the test ends. */
static char *temporary;

/** @brief Removes the test's directory, at exit; a failure ends the test with 1. */
static void
remove_temporary (void)
{
	if (!temporary)
		return;
	if (!support_remove_tree(temporary)) {
		fprintf(stderr, "capture-writer-test: cannot remove %s\n", temporary);
		_exit(1);
	}
	free(temporary);
	temporary = nullptr;
}

/** @brief Appends a slash and a component of a letter to a path on the heap.
 *
 * @param path   The path.
 * @param length Its length, which grows.
 * @param letter The component's letter.
 * @param count  The component's length.
 */
static void
append_component (char   **path,
                  size_t  *length,
                  char     letter,
                  size_t   count)
{
	char *const grown = realloc(*path, *length + 1 + count + 1);
	require(grown, "out of memory");
	grown[*length] = '/';
	memset(grown + *length + 1, letter, count);
	*length += 1 + count;
	grown[*length] = '\0';
	*path = grown;
}

int
main (void)
{
	check_channel_paths();
	check_text_fields();
	temporary = support_temp_dir("/tmp", "dlsslop-amd-capture-test", nullptr);
	require(temporary, "mkdtemp failed");
	// At exit, so a failed check that exits removes it too.
	require(!atexit(remove_temporary), "atexit failed");
	size_t const temporary_length = strlen(temporary);
	// The log reads its variables at its first line, which comes below.
	char *log_path = path_of(nullptr, "%s/layer.log", temporary);
	require(!setenv("DLSSNR_ENABLE", "1", 1) && !setenv("DLSSNR_LOG", log_path, 1), "setenv failed");
	require(!unsetenv("XDG_STATE_HOME"), "unsetenv failed");
	require(!setenv("HOME", temporary, 1), "setenv failed");
	char *home_captures = path_of(nullptr, "%s/.local/state/dlssnr/captures", temporary);
	expect_capture_directory(home_captures, "wrong home capture directory");
	free(home_captures);
	home_captures = nullptr;
	require(!unsetenv("HOME"), "unsetenv failed");
	expect_capture_directory("/tmp/dlssnr-captures", "wrong fallback capture directory");
	require(!setenv("XDG_STATE_HOME", temporary, 1), "setenv failed");
	char *root = capture_directory();
	char *state_captures = path_of(nullptr, "%s/dlssnr/captures", temporary);
	require(!strcmp(root, state_captures), "wrong state capture directory");
	free(state_captures);
	state_captures = nullptr;
	char *state_dlssnr = path_of(nullptr, "%s/dlssnr", temporary);
	require(!mkdir(state_dlssnr, 0700) && !mkdir(root, 0700), "cannot create the capture directory");
	free(state_dlssnr);
	state_dlssnr = nullptr;
	// Legacy unbatched files and unrelated user content must survive a new capture.
	char *legacy = path_of(nullptr, "%s/before_00.png", root);
	char *notes = path_of(nullptr, "%s/notes.txt", root);
	char *published = path_of(nullptr, "%s/manifest.txt", root);
	write_file(legacy, "old capture");
	write_file(notes, "keep");

	// Static: the writer holds its metadata and directory in place.
	static struct capture_writer writer = {};
	require(!capture_writer_active(&writer), "a zeroed writer is active");
	struct capture_metadata metadata = capture_metadata();
	require(metadata.debug_scale == 1.0f, "the metadata's debug scale does not start at 1");
	metadata.frame_control_seq = 123;
	metadata.inference_seq = 55;
	metadata.hold = 1;
	metadata.passes = 1;
	metadata.debug_view = 2;
	metadata.color = 0.25f;
	metadata.detail = 0.5f;
	static unsigned char const bgra[] = {17, 32, 64, 255};
	capture_writer_begin(&writer, 2, 123);
	capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	struct stat st;
	require(stat(published, &st) && errno == ENOENT, "published an incomplete batch");
	require(writer.remaining == 1, "wrong remaining frame count");
	// A missing writer, image or record changes nothing.
	capture_writer_begin(nullptr, 1, 1);
	capture_writer_write_frame(nullptr, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	capture_writer_write_frame(&writer, nullptr, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	capture_writer_write_frame(&writer, bgra, nullptr, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, nullptr);
	require(!capture_writer_active(nullptr) && writer.remaining == 1, "a null pointer changed the batch");
	capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	require(!capture_writer_active(&writer) && !writer.batch_dir, "completed batch remains active");
	struct file_text first = file_text(published);
	require(field_is(&first, "capture_metadata_version", "2"), "wrong metadata version");
	require(field_is(&first, "capture_control_seq", "123"), "wrong capture token");
	require(field_is(&first, "frame_1_before_hash", "c2de31fd48ac5f39"), "wrong input hash");
	require(field_is(&first, "frame_1_inference_seq", "55"), "wrong inference provenance");
	require(field_is(&first, "debug_view", "2"), "wrong renderer view");
	require(field_is(&first, "color", "0.25"), "wrong renderer color");
	// The whole manifest, as the C++ writer wrote it.
	struct field_value const first_batch = field(&first, "batch_dir");
	size_t manifest_length = 0;
	char *manifest = first_manifest(first_batch, &manifest_length);
	require(first.length == manifest_length && !memcmp(first.text, manifest, manifest_length),
	        "the manifest's text changed");
	free(manifest);
	manifest = nullptr;
	char *batch = path_of(nullptr, "%s/%.*s", root, first_batch.length, first_batch.start);
	size_t name_length = 0;
	char *name = path_of(&name_length, "capture-%d-", getpid());
	require(!strncmp(first_batch.start, name, name_length) && (size_t)first_batch.length == name_length + 6,
	        "wrong batch name");
	free(name);
	name = nullptr;
	char *batch_manifest = path_of(nullptr, "%s/manifest.txt", batch);
	struct file_text text = file_text(batch_manifest);
	require(same_text(&text, &first), "batch and published manifests differ");
	file_text_fini(&text);
	char *png_path = path_of(nullptr, "%s/before_00.png", batch);
	int width = 0;
	int height = 0;
	int channels = 0;
	unsigned char *png = stbi_load(png_path, &width, &height, &channels, 4);
	require(png, "cannot decode captured PNG");
	bool const correct = width == 1 && height == 1 && png[0] == 64 && png[1] == 32 && png[2] == 17
	                     && png[3] == 255;
	stbi_image_free(png);
	png = nullptr;
	free(png_path);
	png_path = nullptr;
	require(correct, "BGRA capture channels were not converted to RGBA PNG");
	text = file_text(legacy);
	require(text_is(&text, "old capture"), "overwrote an existing capture");
	file_text_fini(&text);
	text = file_text(notes);
	require(text_is(&text, "keep"), "removed unrelated content");
	file_text_fini(&text);

	capture_writer_begin(&writer, 1, 124);
	text = file_text(published);
	require(same_text(&text, &first), "changed completion before new batch finished");
	file_text_fini(&text);
	capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	text = file_text(published);
	require(field_is(&text, "capture_control_seq", "124"), "new completion not published");
	struct field_value const second_batch = field(&text, "batch_dir");
	require(second_batch.length != first_batch.length
	        || memcmp(second_batch.start, first_batch.start, (size_t)first_batch.length),
	        "reused batch directory");
	file_text_fini(&text);
	text = file_text(batch_manifest);
	require(same_text(&text, &first), "changed prior batch metadata");
	file_text_fini(&text);
	free(batch_manifest);
	batch_manifest = nullptr;

	capture_writer_begin(&writer, 1, 125);
	static uint16_t const fp16[] = {0x3800, 0x3800, 0x3800, 0x3c00};
	capture_writer_write_frame(&writer, fp16, fp16, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, &metadata);
	struct file_text third = file_text(published);
	require(field_is(&third, "encoding", "raw"), "FP16 capture was not raw");
	require(field_is(&third, "bytes_per_pixel", "8"), "FP16 capture byte count incorrect");
	// Eight bytes take the hash's word step; the BGRA pixel above takes its byte tail.
	require(field_is(&third, "before_hash", "736955abadf41fdf"), "wrong FP16 input hash");
	struct field_value const third_batch = field(&third, "batch_dir");
	char *raw = path_of(nullptr, "%s/%.*s/before_00.raw", root, third_batch.length, third_batch.start);
	text = file_text(raw);
	require(text.length == sizeof fp16 && !memcmp(text.text, fp16, sizeof fp16), "FP16 bytes changed");
	file_text_fini(&text);
	free(raw);
	raw = nullptr;

	// A failed image write must leave the last completed batch visible.
	struct directories prior = directories(root);
	capture_writer_begin(&writer, 1, 126);
	struct directories now = directories(root);
	int root_fd = open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	require(root_fd >= 0, "cannot open the capture directory");
	for (size_t i = 0; i < now.count; ++i)
		if (!directories_have(&prior, now.names[i]))
			require(!unlinkat(root_fd, now.names[i], AT_REMOVEDIR), "cannot remove a batch");
	require(!close(root_fd), "cannot close the capture directory");
	root_fd = -1;
	directories_fini(&now);
	directories_fini(&prior);
	capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	require(!capture_writer_active(&writer) && !writer.batch_dir, "failed batch still active");
	text = file_text(published);
	require(same_text(&text, &third), "published failed batch as complete");
	file_text_fini(&text);
	file_text_fini(&third);

	// A request for more frames than a batch holds writes as many as it holds. A float is written
	// with nine significant digits.
	metadata.detail = 0.1f;
	capture_writer_begin(&writer, 100, 128);
	require(writer.remaining == CAPTURE_WRITER_FRAMES, "a batch is not cut to CAPTURE_WRITER_FRAMES");
	for (uint32_t i = 0; i < CAPTURE_WRITER_FRAMES; ++i) {
		metadata.inference_seq = i;
		capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	}
	struct file_text full = file_text(published);
	require(!capture_writer_active(&writer) && field_is(&full, "frames", "64")
	        && field_is(&full, "frame_63_inference_seq", "63")
	        && field_is(&full, "frame_63_detail", "0.100000001"),
	        "a full batch was not written");
	metadata.inference_seq = 55;
	metadata.detail = 0.5f;

	// A batch directory that cannot be created: the writer stays idle and abandons the batch it was
	// writing. Linux refuses a path of PATH_MAX bytes or more, so the directories of a capture
	// directory that long are created as mkdir -p creates them, up to the first that the kernel
	// refuses, even when only its terminating null does not fit; with a capture directory that fits
	// when its batch does not, the directories are created and only the batch's is not.
	capture_writer_begin(&writer, 2, 129);
	require(capture_writer_active(&writer) && writer.batch_dir,
	        "the batch before the failing directories did not start");
	// Components shorter than NAME_MAX, so that only the whole path is too long. PATH_MAX is the
	// value under test here, not a buffer's size.
	size_t const size = PATH_MAX;
	size_t too_long_length = temporary_length;
	char *too_long = strdup(temporary);
	size_t almost_length = temporary_length;
	char *almost = strdup(temporary);
	size_t exact_length = temporary_length;
	char *exact = strdup(temporary);
	require(too_long && almost && exact, "out of memory");
	while (too_long_length < size)
		append_component(&too_long, &too_long_length, 'x', 200);
	while (almost_length + 201 < size - 30)
		append_component(&almost, &almost_length, 'y', 200);
	append_component(&almost, &almost_length, 'y', size - 31 - almost_length);
	// A capture directory of exactly PATH_MAX bytes, "/dlssnr/captures" included.
	while (exact_length + 201 < size - 16)
		append_component(&exact, &exact_length, 'z', 200);
	require(exact_length + 18 <= size, "the temporary directory leaves no room for the exact path");
	append_component(&exact, &exact_length, 'z', size - 17 - exact_length);
	// A capture directory under a file, which mkdir() cannot create.
	char *under_file = path_of(nullptr, "%s/notes", temporary);
	write_file(under_file, "a file");
	char const *const states[] = {too_long, exact, almost, under_file};
	for (size_t i = 0; i < sizeof states / sizeof *states; ++i) {
		require(!setenv("XDG_STATE_HOME", states[i], 1), "setenv failed");
		capture_writer_begin(&writer, 1, 130);
		require(!capture_writer_active(&writer) && !writer.batch_dir,
		        "a batch began in a directory that cannot be created");
		capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
		require(!capture_writer_active(&writer), "an idle writer wrote a frame");
	}
	char *too_long_parent = strndup(too_long, (size_t)(strrchr(too_long, '/') - too_long));
	require(too_long_parent, "out of memory");
	require(is_directory(too_long_parent),
	        "the directories that fit were not created for a capture directory longer than PATH_MAX");
	free(too_long_parent);
	too_long_parent = nullptr;
	char *exact_dlssnr = path_of(nullptr, "%s/dlssnr", exact);
	require(is_directory(exact_dlssnr),
	        "the directories that fit were not created for a capture directory of PATH_MAX bytes");
	free(exact_dlssnr);
	exact_dlssnr = nullptr;
	char *almost_captures = path_of(nullptr, "%s/dlssnr/captures", almost);
	require(is_directory(almost_captures) && is_empty(almost_captures),
	        "the capture directory that fits was not created, or holds a batch");
	require(!setenv("XDG_STATE_HOME", temporary, 1), "setenv failed");

	// A batch publishes where it began, even if the environment moves before it completes.
	capture_writer_begin(&writer, 1, 127);
	char *moved = path_of(nullptr, "%s/moved", temporary);
	require(!setenv("XDG_STATE_HOME", moved, 1), "setenv failed");
	free(moved);
	moved = nullptr;
	capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
	struct file_text fourth = file_text(published);
	require(field_is(&fourth, "capture_control_seq", "127"), "batch did not publish where it began");
	struct field_value const fourth_batch = field(&fourth, "batch_dir");
	char *fourth_manifest = path_of(nullptr, "%s/%.*s/manifest.txt", root, fourth_batch.length,
	                                fourth_batch.start);
	text = file_text(fourth_manifest);
	require(same_text(&text, &fourth), "wrong batch name");
	file_text_fini(&text);
	free(fourth_manifest);
	fourth_manifest = nullptr;

	// Finishing abandons a batch that is being written and frees its directory's path.
	capture_writer_begin(&writer, 2, 131);
	require(capture_writer_active(&writer) && writer.batch_dir, "the batch to abandon did not start");
	capture_writer_fini(&writer);
	require(!capture_writer_active(&writer) && !writer.batch_dir, "finishing kept the batch");
	capture_writer_fini(&writer);
	capture_writer_fini(nullptr);
	text = file_text(published);
	require(same_text(&text, &fourth), "an abandoned batch was published");
	file_text_fini(&text);
	file_text_fini(&fourth);

	// The log's lines, which bench and test scripts read.
	struct file_text log_text = file_text(log_path);
	require(logged(&log_text, SIZE_MAX, "[capture] capturing 2 frames, control 123, to %s/%.*s", root,
	               first_batch.length, first_batch.start),
	        "the first batch's start was not logged");
	require(logged(&log_text, SIZE_MAX, "[capture] wrote 2 pairs to %s/%.*s", root, first_batch.length,
	               first_batch.start),
	        "the first batch's end was not logged");
	struct field_value const full_batch = field(&full, "batch_dir");
	require(logged(&log_text, SIZE_MAX, "[capture] capturing 64 frames, control 128, to %s/%.*s", root,
	               full_batch.length, full_batch.start),
	        "the cut batch's start was not logged");
	char *could_not = path_of(nullptr, "[capture] could not write %s/capture-", root);
	require(strstr(log_text.text, could_not)
	        && logged(&log_text, SIZE_MAX, "[capture] batch failed; completion manifest was not published"),
	        "the failed batch was not logged");
	free(could_not);
	could_not = nullptr;
	// The log cuts a line's text to 2047 bytes.
	static char const failed[] = "[capture] cannot create batch directory in ";
	require(logged(&log_text, 2047, "%s%s", failed, almost_captures),
	        "the batch directory that did not fit was not logged");
	require(logged(&log_text, 2047, "%s%s", failed, too_long),
	        "the capture directory longer than PATH_MAX was not logged");
	require(logged(&log_text, 2047, "%s%s", failed, exact),
	        "the capture directory of PATH_MAX bytes was not logged");
	// The directory that the kernel refused, with its error after it.
	static char const refused[] = "[capture] cannot create ";
	require(logged(&log_text, 2047, "%s%s", refused, too_long)
	        && logged(&log_text, 2047, "%s%s/dlssnr/captures", refused, exact),
	        "the directory that the kernel refused was not logged");
	require(logged(&log_text, SIZE_MAX, "%s%s/dlssnr/captures", failed, under_file),
	        "the capture directory under a file was not logged");

	file_text_fini(&log_text);
	file_text_fini(&full);
	file_text_fini(&first);
	free(almost_captures);
	almost_captures = nullptr;
	free(under_file);
	under_file = nullptr;
	free(exact);
	exact = nullptr;
	free(almost);
	almost = nullptr;
	free(too_long);
	too_long = nullptr;
	free(batch);
	batch = nullptr;
	free(published);
	published = nullptr;
	free(notes);
	notes = nullptr;
	free(legacy);
	legacy = nullptr;
	free(root);
	root = nullptr;
	free(log_path);
	log_path = nullptr;
	if (puts("PASS: runtime and capture paths, text fields, capture publication, provenance, manifest text, "
	         "preserved batches, BGRA PNG, FP16 raw, write failure, full batch, failed directories, moved "
	         "environment, log lines, finishing") == EOF)
		return 1;
	return 0;
}
