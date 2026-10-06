/** @file
 *
 * VK_LAYER_LOCAL_vktrace: an explicit Vulkan layer that logs what an application asks of a device,
 * with every handle renamed by kind and creation order (buf12, img3, pipe40, ...), so that two runs,
 * or two implementations, can be compared line by line or canonically (compare.py). trace.sh
 * enables it for one dlsslopd run; VALIDATION.md describes the procedure.
 *
 * Logged: instance and device creation (enabled extensions and every enabled feature bit), queues,
 * memory allocations (size, type and its property flags, dedicated/import/export), buffers and
 * images (create info) and their binds (allocation + offset), views, samplers, shader modules
 * (FNV-1a 64 of the SPIR-V and the file it matches in VKTRACE_SPIRV), descriptor set layouts,
 * pipeline layouts (push ranges), compute pipelines (module, entry, required subgroup size,
 * specialization map and data), descriptor pools and sets and every descriptor written or copied,
 * under the binding it reaches, command pools and buffers, and each recorded command the network
 * uses (binds, push constants with their bytes, dispatches, barriers v1/v2, copies, blits, fills,
 * updates, clears, queries, timestamps), queue submits and waits.
 *
 * Host data: every copy out of mapped host-visible memory logs the FNV of its source bytes at submit
 * ("Upload"), and every copy into host-visible memory logs the FNV of what arrived once its fence or
 * queue is waited for ("Readback").
 *
 * Content hashes (VKTRACE_HASH, off by default): after a submit completes, the FNV of selected
 * buffers and images, copied out through a staging buffer on the same queue ("Hash"). Selectors,
 * comma-separated:
 *   i2b        the byte range each vkCmdCopyImageToBuffer wrote
 *   copydst    every copy/blit/fill/update/clear destination
 *   storage    every storage buffer range and storage image a dispatch bound
 *   dispatch=N the storage bindings of the submit's dispatch N (0-based)
 *   buf12,img3 those resources, whole
 *   all        every live buffer and image of the device
 * VKTRACE_HASH_SUBMITS limits hashing to submits (per device, 1-based): "A-B,C", or "dispatch" for
 * submits that dispatch anything. Default: all. Items that mean nothing are reported on standard
 * error and ignored. A resource that another thread destroys between the submit and the copies is
 * logged with skip=destroyed, and one whose memory it frees, or has not bound yet, with
 * skip=unbound. A thread that destroys a resource or frees memory while the copies run waits until
 * they are done. Sparse binds are not traced, so a sparse resource is copied as if it were bound. An
 * image bound to the memory of a swapchain image is logged with skip=external, as the swapchain's
 * own images are.
 *
 * Environment: VKTRACE_FILE (the log; %p is the pid; default /tmp/vktrace.%p.log), VKTRACE_SPIRV
 * (':'-separated directories scanned recursively for *.spv), VKTRACE_HASH, VKTRACE_HASH_SUBMITS.
 *
 * The layer stays loaded once the loader has loaded it (CMake links it with -z nodelete): its trace
 * goes on in one file for the life of the process. Every function here is static but
 * vkNegotiateLoaderLayerInterfaceVersion.
 */
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <errno.h>
#include <immintrin.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include "../support.h"
#include "vkt_names.h"

static_assert(VK_USE_64_BIT_PTR_DEFINES, "every Vulkan handle is a pointer");

/** @brief The FNV-1a 64 offset basis. */
static constexpr uint64_t FNV_BASIS = UINT64_C(14695981039346656037);

/** @brief The FNV-1a 64 prime. */
static constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);

/** @brief The bytes of each thread's chunk that fnv_uncached() reads uncached memory into. */
static constexpr size_t SCRATCH_BYTES = 1 << 16;

/** @brief The bytes of a handle's name: "?0x" and 16 hex digits and a null, longer than a kind's
 *         prefix of at most 5 letters and 10 digits.
 */
static constexpr size_t NAME_BYTES = 3 + 16 + 1;

/** @brief The bytes of a number's text: 20 digits, or a sign and 19, and a null. */
static constexpr size_t DIGITS_BYTES = 21;

/** @brief The bytes of an enum value's text: its name from vkt_names.h, or its number. */
static constexpr size_t WORD_BYTES = (VKT_NAME_LENGTH + 1 > DIGITS_BYTES ? VKT_NAME_LENGTH + 1 : DIGITS_BYTES);

/** @brief The bytes of a recorded command's label: the longest, a command buffer's name, '#', 10
 *         digits, '.', 10 digits and a null, takes 42; 48 leaves a host range no padding.
 */
static constexpr size_t LABEL_BYTES = 48;

static_assert(NAME_BYTES - 1 + 1 + 10 + 1 + 10 + 1 <= LABEL_BYTES - 1, "a command's label fits a target's");

/** @brief The bytes of a hash job's resource text: a buffer's name, '+', its offset, '+', its bytes
 *         and a null, at most 62; 63 leaves a job no padding.
 */
static constexpr size_t WHAT_BYTES = 63;

static_assert(NAME_BYTES - 1 + 2 * (1 + 20) + 1 <= WHAT_BYTES, "a buffer region's text fits");

// ---- memory and text ----------------------------------------------------------------------------

/** @brief realloc() for the trace's own memory: running out ends the process, since a trace must not
 *         lose a call.
 *
 * @param p     The memory, or nullptr.
 * @param bytes The bytes it must hold, more than 0.
 * @return      The memory.
 */
static void *
reallocate (void   *p,
            size_t  bytes)
{
	void *const grown = realloc(p, bytes);
	if (!grown) {
		fputs("vktrace: out of memory\n", stderr);
		abort();
	}
	return grown;
}

/** @brief Defines struct NAME, an array of TYPE that grows, empty when zeroed, and NAME_push(), which
 *         adds an element that the caller sets and returns it.
 */
#define VECTOR(name, type) \
	struct name { \
		type   *items;    /* The elements. */ \
		size_t  count;    /* How many. */ \
		size_t  capacity; /* The elements allocated. */ \
	}; \
	\
	[[maybe_unused]] static type * \
	name##_push (struct name *v) \
	{ \
		if (v->count == v->capacity) { \
			v->capacity = v->capacity ? v->capacity * 2 : SUPPORT_MAP_MINIMUM; \
			v->items = reallocate(v->items, v->capacity * sizeof *v->items); \
		} \
		return &v->items[v->count++]; \
	}

/** @brief Text that grows as it is written, empty when zeroed. */
struct text {
	char   *bytes;    //!< The text and a null, or nullptr until something is written.
	size_t  length;   //!< Its length.
	size_t  capacity; //!< The bytes allocated.
};

/** @brief Makes room in a text for more bytes and a null.
 *
 * @param t    The text.
 * @param more The bytes.
 */
static void
text_reserve (struct text *t,
              size_t       more)
{
	size_t const need = t->length + more + 1;
	if (need <= t->capacity)
		return;
	size_t capacity = t->capacity ? t->capacity * 2 : 256;
	while (capacity < need)
		capacity *= 2;
	t->bytes = reallocate(t->bytes, capacity);
	t->capacity = capacity;
}

/** @brief Cuts a text back to a length it had.
 *
 * @param t      The text.
 * @param length The length.
 */
static void
text_truncate (struct text *t,
               size_t       length)
{
	t->length = length;
	if (t->bytes)
		t->bytes[length] = '\0';
}

/** @brief Appends bytes to a text.
 *
 * @param t The text.
 * @param s The bytes.
 * @param n How many.
 */
static void
text_append (struct text *t,
             char const  *s,
             size_t       n)
{
	text_reserve(t, n);
	memcpy(t->bytes + t->length, s, n);
	t->length += n;
	t->bytes[t->length] = '\0';
}

/** @brief Appends formatted words to a text; a format that fails appends nothing.
 *
 * @param t    The text.
 * @param fmt  A printf format.
 * @param args The format's arguments.
 */
[[gnu::format(printf, 2, 0)]]
static void
text_vprintf (struct text *t,
              char const  *fmt,
              va_list      args)
{
	va_list again;
	va_copy(again, args);
	size_t const room = t->capacity - t->length;
	int const n = vsnprintf(t->bytes ? t->bytes + t->length : nullptr, room, fmt, args);
	if (n >= 0 && (size_t)n >= room) {
		// Too long for the room left: grown, and formatted again.
		text_reserve(t, (size_t)n);
		if (vsnprintf(t->bytes + t->length, (size_t)n + 1, fmt, again) != n) {
			va_end(again);
			text_truncate(t, t->length);
			return;
		}
	}
	va_end(again);
	if (n < 0) {
		text_truncate(t, t->length);
		return;
	}
	t->length += (size_t)n;
}

/** @brief Appends formatted words to a text; a format that fails appends nothing.
 *
 * @param t   The text.
 * @param fmt A printf format.
 * @param ... The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
text_printf (struct text *t,
             char const  *fmt,
             ...)
{
	va_list args;
	va_start(args, fmt);
	text_vprintf(t, fmt, args);
	va_end(args);
}

/** @brief Frees a text and empties it.
 *
 * @param t The text.
 */
static void
text_fini (struct text *t)
{
	free(t->bytes);
	t->bytes = nullptr;
	*t = (struct text){};
}

/** @brief A heap string and its length. */
struct string {
	char   *text;   //!< The string, on the heap.
	size_t  length; //!< Its length.
};

VECTOR(strings, struct string)

/** @brief Adds a heap string to an array kept in strcmp() order, after any equal ones.
 *
 * @param list   The array.
 * @param text   The string, which the array takes.
 * @param length Its length.
 */
static void
strings_add_sorted (struct strings *list,
                    char           *text,
                    size_t          length)
{
	size_t low = 0;
	size_t count = list->count;
	while (count) {
		size_t const half = count / 2;
		if (strcmp(text, list->items[low + half].text) < 0) {
			count = half;
		} else {
			low += half + 1;
			count -= half + 1;
		}
	}
	strings_push(list);
	memmove(&list->items[low + 1], &list->items[low], (list->count - 1 - low) * sizeof *list->items);
	list->items[low] = (struct string){text, length};
}

/** @brief Frees an array of heap strings and empties it.
 *
 * @param list The array.
 */
static void
strings_fini (struct strings *list)
{
	for (size_t i = 0; i < list->count; ++i) {
		free(list->items[i].text);
		list->items[i].text = nullptr;
	}
	free(list->items);
	list->items = nullptr;
	*list = (struct strings){};
}

/** @brief A copy of bytes on the heap, with a null after them.
 *
 * @param text   The bytes.
 * @param length How many.
 * @return       The copy, which the caller frees.
 */
static char *
copy_text (char const *text,
           size_t      length)
{
	char *const copy = reallocate(nullptr, length + 1);
	memcpy(copy, text, length);
	copy[length] = '\0';
	return copy;
}

/** @brief Joins two strings on the heap with a '/' between them.
 *
 * @param a        The first string.
 * @param a_length Its length.
 * @param b        The second string.
 * @param b_length Its length.
 * @return         The string, which the caller frees.
 */
static char *
join_path (char const *a,
           size_t      a_length,
           char const *b,
           size_t      b_length)
{
	char *const text = reallocate(nullptr, a_length + 1 + b_length + 1);
	memcpy(text, a, a_length);
	text[a_length] = '/';
	memcpy(text + a_length + 1, b, b_length);
	text[a_length + 1 + b_length] = '\0';
	return text;
}

// ---- hashes -------------------------------------------------------------------------------------

/** @brief Goes on with an FNV-1a 64 over more bytes.
 *
 * @param data The bytes.
 * @param size How many.
 * @param h    The hash so far; FNV_BASIS to start one.
 * @return     The hash.
 */
static uint64_t
fnv (void const *data,
     size_t      size,
     uint64_t    h)
{
	unsigned char const *const p = data;
	for (size_t i = 0; i < size; ++i)
		h = (h ^ p[i]) * FNV_PRIME;
	return h;
}

/** @brief The key of each thread's chunk for fnv_uncached(). Its destructor, free(), releases a
 *         thread's chunk when the thread exits.
 */
static pthread_key_t scratch_key;

/** @brief Whether scratch_key was made; without it, fnv_uncached() reads the bytes where they are. */
static bool scratch_keyed;

/** @brief Makes scratch_key when the layer is loaded. */
[[gnu::constructor]]
static void
initialize (void)
{
	scratch_keyed = !pthread_key_create(&scratch_key, free);
}

/** @brief This thread's chunk for fnv_uncached(), allocated on first use.
 *
 * @return The chunk of SCRATCH_BYTES, or nullptr if there is none.
 */
static unsigned char *
scratch (void)
{
	if (!scratch_keyed)
		return nullptr;
	unsigned char *chunk = pthread_getspecific(scratch_key);
	if (!chunk) {
		chunk = malloc(SCRATCH_BYTES);
		if (chunk && pthread_setspecific(scratch_key, chunk)) {
			free(chunk);
			chunk = nullptr;
		}
	}
	return chunk;
}

/** @brief The FNV-1a 64 of host-visible memory that may be uncached (write-combined VRAM through the
 *         BAR): read in 16-byte streaming loads into a cached chunk first. Byte loads from such
 *         memory each cross the bus, which made a 3.6 MB frame cost ~300 ms to hash.
 *
 * @param data The bytes.
 * @param n    How many.
 * @return     Their hash.
 */
[[gnu::target("sse4.1")]]
static uint64_t
fnv_uncached (void const *data,
              size_t      n)
{
	unsigned char const *p = data;
	unsigned char *const chunk = scratch();
	// Without a chunk, the bytes are read where they are: slower, the same hash.
	if (!chunk)
		return fnv(p, n, FNV_BASIS);
	size_t const misaligned = (16 - ((uintptr_t)p & 15)) & 15;
	size_t const head = n < misaligned ? n : misaligned;
	uint64_t h = fnv(p, head, FNV_BASIS);
	p += head;
	n -= head;
	while (n >= 16) {
		size_t const whole = n & ~(size_t)15;
		size_t const take = whole < SCRATCH_BYTES ? whole : SCRATCH_BYTES;
		for (size_t i = 0; i < take; i += 16) {
			// The intrinsic only reads, though older headers declare a pointer to non-const.
			__m128i const v = _mm_stream_load_si128((__m128i *)(p + i));
			_mm_storeu_si128((__m128i *)(chunk + i), v);
		}
		h = fnv(chunk, take, h);
		p += take;
		n -= take;
	}
	return fnv(p, n, h);
}

// ---- short texts --------------------------------------------------------------------------------

/** @brief A number's text, or a word that stands for a number. */
struct digits {
	char text[DIGITS_BYTES]; //!< The text.
};

/** @brief "0x" and a number's hex digits. */
struct hex_text {
	char text[2 + 16 + 1]; //!< The text.
};

/** @brief An enum value's name, or its number. */
struct word {
	char text[WORD_BYTES]; //!< The text.
};

/** @brief A handle's name in the trace: its kind's prefix and creation number, "null", or "?" and
 *         its address.
 */
struct name {
	char text[NAME_BYTES]; //!< The text.
};

/** @brief A number in hex.
 *
 * @param v The number.
 * @return  "0x" and its digits.
 */
static struct hex_text
hex (uint64_t v)
{
	struct hex_text h;
	snprintf(h.text, sizeof h.text, "0x%" PRIx64, v);
	return h;
}

/** @brief A size, or "WHOLE".
 *
 * @param s The size.
 * @return  Its text.
 */
static struct digits
size_text (VkDeviceSize s)
{
	struct digits d = {"WHOLE"};
	if (s != VK_WHOLE_SIZE)
		snprintf(d.text, sizeof d.text, "%" PRIu64, s);
	return d;
}

/** @brief A queue family index: "ign", "ext", "foreign" or the number.
 *
 * @param f The index.
 * @return  Its text.
 */
static struct digits
family_text (uint32_t f)
{
	struct digits d = {"ign"};
	if (f == VK_QUEUE_FAMILY_EXTERNAL)
		memcpy(d.text, "ext", sizeof "ext");
	else if (f == VK_QUEUE_FAMILY_FOREIGN_EXT)
		memcpy(d.text, "foreign", sizeof "foreign");
	else if (f != VK_QUEUE_FAMILY_IGNORED)
		snprintf(d.text, sizeof d.text, "%" PRIu32, f);
	return d;
}

/** @brief A count of a subresource range, or "all" for the rest of them.
 *
 * @param count The count.
 * @param rest  The value that means the rest: VK_REMAINING_MIP_LEVELS or VK_REMAINING_ARRAY_LAYERS.
 * @return      Its text.
 */
static struct digits
count_text (uint32_t count,
            uint32_t rest)
{
	struct digits d = {"all"};
	if (count != rest)
		snprintf(d.text, sizeof d.text, "%" PRIu32, count);
	return d;
}

/** @brief An API version as "MAJOR.MINOR.PATCH".
 *
 * @param v The version.
 * @return  Its text.
 */
static struct digits
version_text (uint32_t v)
{
	struct digits d;
	snprintf(d.text, sizeof d.text, "%" PRIu32 ".%" PRIu32 ".%" PRIu32, VK_API_VERSION_MAJOR(v),
	         VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
	return d;
}

/** @brief An enum value's text: its name from the generated tables, or its number.
 *
 * @param name  The name, or nullptr for a value without one.
 * @param value The value.
 * @return      Its text.
 */
static struct word
enum_word (char const *name,
           int64_t     value)
{
	struct word w;
	if (name)
		strcpy(w.text, name);
	else
		snprintf(w.text, sizeof w.text, "%" PRId64, value);
	return w;
}

/** @brief A format's text.
 *
 * @param f The format.
 * @return  Its name, or its number.
 */
static struct word
format_word (VkFormat f)
{
	return enum_word(vkt_format_name(f), f);
}

/** @brief An image layout's text.
 *
 * @param l The layout.
 * @return  Its name, or its number.
 */
static struct word
layout_word (VkImageLayout l)
{
	return enum_word(vkt_image_layout_name(l), l);
}

/** @brief A descriptor type's text.
 *
 * @param t The type.
 * @return  Its name, or its number.
 */
static struct word
dtype_word (VkDescriptorType t)
{
	return enum_word(vkt_descriptor_type_name(t), t);
}

/** @brief A structure type's text.
 *
 * @param t The type.
 * @return  Its name, or its number.
 */
static struct word
stype_word (VkStructureType t)
{
	return enum_word(vkt_structure_type_name(t), t);
}

/** @brief A component swizzle's letter.
 *
 * @param s The swizzle.
 * @return  "i", "0", "1", "r", "g", "b", "a", or "?".
 */
static char const *
swizzle_letter (VkComponentSwizzle s)
{
	switch (s) {
	case VK_COMPONENT_SWIZZLE_IDENTITY: return "i";
	case VK_COMPONENT_SWIZZLE_ZERO: return "0";
	case VK_COMPONENT_SWIZZLE_ONE: return "1";
	case VK_COMPONENT_SWIZZLE_R: return "r";
	case VK_COMPONENT_SWIZZLE_G: return "g";
	case VK_COMPONENT_SWIZZLE_B: return "b";
	case VK_COMPONENT_SWIZZLE_A: return "a";
	default: return "?";
	}
}

// ---- handles and what the trace knows of them ---------------------------------------------------

/** @brief The kinds of handles, each named with its own prefix and numbers. */
enum kind : uint8_t {
	KIND_INSTANCE,
	KIND_PHYSICAL,
	KIND_DEVICE,
	KIND_QUEUE,
	KIND_MEMORY,
	KIND_BUFFER,
	KIND_IMAGE,
	KIND_VIEW,
	KIND_BUFFER_VIEW,
	KIND_SAMPLER,
	KIND_SHADER,
	KIND_CACHE,
	KIND_SET_LAYOUT,
	KIND_PIPE_LAYOUT,
	KIND_PIPELINE,
	KIND_DESC_POOL,
	KIND_DESC_SET,
	KIND_CMD_POOL,
	KIND_CMD,
	KIND_FENCE,
	KIND_SEMAPHORE,
	KIND_QUERY_POOL,
	KIND_SWAPCHAIN,
	KIND_COUNT
};

/** @brief Each kind's prefix. */
static char const *const KIND_PREFIX[] = {
	"inst", "pd", "dev", "q", "mem", "buf", "img", "view", "bview", "smp", "shm", "pc", "dsl", "pl", "pipe",
	"dp", "ds", "cp", "cb", "fence", "sem", "qp", "sc",
};

static_assert(sizeof KIND_PREFIX / sizeof *KIND_PREFIX == KIND_COUNT, "a prefix per kind");

/** @brief A handle as a map's key.
 *
 * @param handle The handle.
 * @return       Its address.
 */
static uint64_t
key_of (void const *handle)
{
	static_assert(sizeof (uintptr_t) <= sizeof (uint64_t), "an address fits a key");
	return (uintptr_t)handle;
}

/** @brief The loader's dispatch table pointer, shared by a device and its queues and command buffers
 *         (and by an instance and its physical devices).
 *
 * @param object A dispatchable handle.
 * @return       Its dispatch table pointer.
 */
static void *
dispatch_key (void const *object)
{
	return *(void *const *)object;
}

struct instance_data;
struct device_data;

SUPPORT_SORTED_MAP(u32_map, uint32_t, reallocate)
SUPPORT_SORTED_MAP(handle_map, uint64_t, reallocate)
SUPPORT_SORTED_MAP(instance_map, struct instance_data *, reallocate)
SUPPORT_SORTED_MAP(device_map, struct device_data *, reallocate)

/** @brief A memory allocation the trace knows. */
struct memory_info {
	VkDeviceSize  size;       //!< Its bytes.
	VkDeviceSize  map_offset; //!< Where the mapping starts.
	VkDeviceSize  map_size;   //!< The mapping's bytes.
	void         *mapped;     //!< The application's mapping, or nullptr.
	uint32_t      flags;      //!< Its memory type's VkMemoryPropertyFlags.
	VkBool32      external;   //!< Imported or exportable. 32 bits wide, which fills the padding.
};

/** @brief A buffer the trace knows. */
struct buffer_info {
	VkDeviceMemory      memory;   //!< The memory bound to it, or VK_NULL_HANDLE.
	VkDeviceSize        size;     //!< Its bytes.
	VkDeviceSize        offset;   //!< Where in the memory.
	struct device_data *device;   //!< Its device.
	VkBufferUsageFlags  usage;    //!< Its usage.
	uint32_t            owner;    //!< The queue family the last submitted ownership transfer gave it
	                              //!< to; VK_QUEUE_FAMILY_IGNORED until one is seen.
	VkBool32            external; //!< Created or bound for external memory.
	VkBool32            sparse;   //!< Bound through vkQueueBindSparse, which is not traced: memory
	                              //!< stays null. 32 bits wide, which fills the padding.
};

/** @brief An image the trace knows. */
struct image_info {
	VkDeviceMemory      memory;    //!< The memory bound to it, or VK_NULL_HANDLE.
	struct device_data *device;    //!< Its device.
	VkExtent3D          extent;    //!< Its extent.
	VkFormat            format;    //!< Its format.
	uint32_t            layers;    //!< Its array layers.
	VkImageUsageFlags   usage;     //!< Its usage.
	VkImageLayout       layout;    //!< Its layout after the last submitted barrier.
	uint16_t            sparse;    //!< As buffer_info's. 16 bits wide, which fills the padding.
	bool                swapchain; //!< A swapchain's image, or an image bound to the memory of one.
	bool                external;  //!< Created or bound for external memory.
};

/** @brief A descriptor a set holds. */
struct desc {
	VkBuffer         buffer; //!< A buffer descriptor's buffer.
	VkImageView      view;   //!< An image descriptor's view.
	VkDeviceSize     offset; //!< A buffer descriptor's offset.
	VkDeviceSize     range;  //!< A buffer descriptor's range.
	VkDescriptorType type;   //!< Its type.
};

SUPPORT_SORTED_MAP(slot_map, struct desc, reallocate)

/** @brief A descriptor set layout the trace knows. */
struct layout_info {
	struct u32_map counts;   //!< The descriptor count of each binding number.
	uintptr_t      variable; //!< Nonzero when the last binding's count is given at allocation.
	                         //!< As wide as a pointer, which fills the padding.
};

/** @brief A descriptor set the trace knows. */
struct set_info {
	struct slot_map slots;  //!< Each descriptor written, by binding << 32 | element.
	struct u32_map  counts; //!< Its layout's counts, by binding number; empty when unknown.
};

/** @brief A resource region a hash selector names, as a command recorded it. */
struct target {
	uint64_t     handle;                 //!< The buffer or image.
	VkDeviceSize offset;                 //!< A buffer region's offset.
	VkDeviceSize size;                   //!< Its size, or VK_WHOLE_SIZE.
	char         label[LABEL_BYTES - 1]; //!< The command that named it: "cb3#12.0".
	bool         image;                  //!< An image, which is copied whole.
};

/** @brief Host-visible bytes to hash: at submit (uploads) or once complete (readbacks). */
struct host_range {
	VkDeviceMemory memory;             //!< The memory.
	VkDeviceSize   offset;             //!< Where in the allocation.
	VkDeviceSize   size;               //!< The bytes.
	char           label[LABEL_BYTES]; //!< The command that copies them: "cb3#12.0".
};

/** @brief A layout an image barrier recorded. */
struct image_layout {
	VkImage       image;  //!< The image.
	VkImageLayout layout; //!< Its new layout.
};

/** @brief An ownership transfer a buffer barrier recorded. */
struct buffer_owner {
	VkBuffer buffer; //!< The buffer.
	uint32_t family; //!< The queue family it goes to.
};

VECTOR(key_list, uint64_t)
VECTOR(image_layouts, struct image_layout)
VECTOR(buffer_owners, struct buffer_owner)
VECTOR(host_ranges, struct host_range)
VECTOR(targets, struct target)
VECTOR(ends, size_t)

/** @brief Removes every element of a key list that equals a key, keeping the others' order.
 *
 * @param list The list.
 * @param key  The key.
 */
static void
key_list_remove (struct key_list *list,
                 uint64_t         key)
{
	size_t kept = 0;
	for (size_t i = 0; i < list->count; ++i)
		if (list->items[i] != key)
			list->items[kept++] = list->items[i];
	list->count = kept;
}

/** @brief What a command buffer recorded since it began. */
struct cb_state {
	VkPipeline            compute;          //!< The pipeline bound for compute.
	struct device_data   *device;           //!< Its device.
	struct key_list       sets;             //!< The sets bound for compute, by set number.
	struct image_layouts  layouts;          //!< The layouts its image barriers set, in order.
	struct buffer_owners  owners;           //!< Its ownership transfers of buffers, in order.
	struct host_ranges    uploads;          //!< Its copies out of host-visible memory.
	struct host_ranges    downloads;        //!< Its copies into host-visible memory.
	struct targets        copy_dst;         //!< Every copy, blit, fill, update and clear destination.
	struct targets        i2b_dst;          //!< Every range a copy from an image to a buffer wrote.
	struct targets        dispatch_targets; //!< The storage resources of its dispatches, in order;
	                                        //!< recorded only for the selectors that need them.
	struct ends           dispatch_ends;    //!< Where each dispatch's targets end.
	uint32_t              index;            //!< The commands recorded.
	uint32_t              dispatches;       //!< The dispatches recorded.
};

SUPPORT_SORTED_MAP(memory_map, struct memory_info, reallocate)
SUPPORT_SORTED_MAP(buffer_map, struct buffer_info, reallocate)
SUPPORT_SORTED_MAP(image_map, struct image_info, reallocate)
SUPPORT_SORTED_MAP(layout_map, struct layout_info, reallocate)
SUPPORT_SORTED_MAP(set_map, struct set_info, reallocate)
SUPPORT_SORTED_MAP(list_map, struct key_list, reallocate)
SUPPORT_SORTED_MAP(cb_map, struct cb_state, reallocate)

/** @brief A submission whose host ranges are hashed once it is known to be complete. */
struct pending {
	VkFence            fence;  //!< Its fence, or VK_NULL_HANDLE.
	uint64_t           submit; //!< Its number.
	VkQueue            queue;  //!< Its queue.
	struct host_ranges ranges; //!< What it copies into host-visible memory.
};

VECTOR(pendings, struct pending)

/** @brief A command pool and buffer of the layer's own, for its hash copies on one queue family. */
struct hash_pool {
	VkCommandPool   pool; //!< The pool.
	VkCommandBuffer cb;   //!< Its command buffer.
};

SUPPORT_SORTED_MAP(pool_map, struct hash_pool, reallocate)

/** @brief The instance functions that the layer calls on the next layer. */
#define INSTANCE_CALLS(X) \
	X(vkDestroyInstance) \
	X(vkEnumeratePhysicalDevices) \
	X(vkGetPhysicalDeviceProperties) \
	X(vkGetPhysicalDeviceMemoryProperties)

/** @brief An instance the trace knows. */
struct instance_data {
	VkInstance                instance; //!< The instance.
	PFN_vkGetInstanceProcAddr gipa;     //!< The next layer's vkGetInstanceProcAddr.
	// The next layer's functions.
#define MEMBER(name) PFN_##name name;
	INSTANCE_CALLS(MEMBER)
#undef MEMBER
};

/** @brief The device functions the layer intercepts, as X(Vulkan name, the layer's function). */
#define DEVICE_HOOKS(X) \
	X(vkGetDeviceProcAddr,            get_device_proc_addr) \
	X(vkDestroyDevice,                destroy_device) \
	X(vkGetDeviceQueue,               get_device_queue) \
	X(vkGetDeviceQueue2,              get_device_queue2) \
	X(vkQueueSubmit,                  queue_submit) \
	X(vkQueueSubmit2,                 queue_submit2) \
	X(vkQueueSubmit2KHR,              queue_submit2_khr) \
	X(vkQueueWaitIdle,                queue_wait_idle) \
	X(vkDeviceWaitIdle,               device_wait_idle) \
	X(vkAllocateMemory,               allocate_memory) \
	X(vkFreeMemory,                   free_memory) \
	X(vkMapMemory,                    map_memory) \
	X(vkUnmapMemory,                  unmap_memory) \
	X(vkCreateBuffer,                 create_buffer) \
	X(vkDestroyBuffer,                destroy_buffer) \
	X(vkGetBufferMemoryRequirements,  get_buffer_memory_requirements) \
	X(vkBindBufferMemory,             bind_buffer_memory) \
	X(vkBindBufferMemory2,            bind_buffer_memory2) \
	X(vkCreateImage,                  create_image) \
	X(vkDestroyImage,                 destroy_image) \
	X(vkGetImageMemoryRequirements,   get_image_memory_requirements) \
	X(vkBindImageMemory,              bind_image_memory) \
	X(vkBindImageMemory2,             bind_image_memory2) \
	X(vkCreateImageView,              create_image_view) \
	X(vkDestroyImageView,             destroy_image_view) \
	X(vkCreateSampler,                create_sampler) \
	X(vkDestroySampler,               destroy_sampler) \
	X(vkCreateShaderModule,           create_shader_module) \
	X(vkDestroyShaderModule,          destroy_shader_module) \
	X(vkCreatePipelineCache,          create_pipeline_cache) \
	X(vkDestroyPipelineCache,         destroy_pipeline_cache) \
	X(vkCreateDescriptorSetLayout,    create_descriptor_set_layout) \
	X(vkDestroyDescriptorSetLayout,   destroy_descriptor_set_layout) \
	X(vkCreatePipelineLayout,         create_pipeline_layout) \
	X(vkDestroyPipelineLayout,        destroy_pipeline_layout) \
	X(vkCreateComputePipelines,       create_compute_pipelines) \
	X(vkCreateGraphicsPipelines,      create_graphics_pipelines) \
	X(vkDestroyPipeline,              destroy_pipeline) \
	X(vkCreateDescriptorPool,         create_descriptor_pool) \
	X(vkDestroyDescriptorPool,        destroy_descriptor_pool) \
	X(vkResetDescriptorPool,          reset_descriptor_pool) \
	X(vkAllocateDescriptorSets,       allocate_descriptor_sets) \
	X(vkFreeDescriptorSets,           free_descriptor_sets) \
	X(vkUpdateDescriptorSets,         update_descriptor_sets) \
	X(vkCreateCommandPool,            create_command_pool) \
	X(vkDestroyCommandPool,           destroy_command_pool) \
	X(vkResetCommandPool,             reset_command_pool) \
	X(vkAllocateCommandBuffers,       allocate_command_buffers) \
	X(vkFreeCommandBuffers,           free_command_buffers) \
	X(vkBeginCommandBuffer,           begin_command_buffer) \
	X(vkEndCommandBuffer,             end_command_buffer) \
	X(vkResetCommandBuffer,           reset_command_buffer) \
	X(vkCmdBindPipeline,              cmd_bind_pipeline) \
	X(vkCmdBindDescriptorSets,        cmd_bind_descriptor_sets) \
	X(vkCmdPushConstants,             cmd_push_constants) \
	X(vkCmdDispatch,                  cmd_dispatch) \
	X(vkCmdDispatchBase,              cmd_dispatch_base) \
	X(vkCmdDispatchIndirect,          cmd_dispatch_indirect) \
	X(vkCmdPipelineBarrier,           cmd_pipeline_barrier) \
	X(vkCmdPipelineBarrier2,          cmd_pipeline_barrier2) \
	X(vkCmdPipelineBarrier2KHR,       cmd_pipeline_barrier2_khr) \
	X(vkCmdCopyBuffer,                cmd_copy_buffer) \
	X(vkCmdCopyImage,                 cmd_copy_image) \
	X(vkCmdBlitImage,                 cmd_blit_image) \
	X(vkCmdCopyBufferToImage,         cmd_copy_buffer_to_image) \
	X(vkCmdCopyImageToBuffer,         cmd_copy_image_to_buffer) \
	X(vkCmdFillBuffer,                cmd_fill_buffer) \
	X(vkCmdUpdateBuffer,              cmd_update_buffer) \
	X(vkCmdClearColorImage,           cmd_clear_color_image) \
	X(vkCmdWriteTimestamp,            cmd_write_timestamp) \
	X(vkCmdWriteTimestamp2,           cmd_write_timestamp2) \
	X(vkCmdResetQueryPool,            cmd_reset_query_pool) \
	X(vkCmdBeginQuery,                cmd_begin_query) \
	X(vkCmdEndQuery,                  cmd_end_query) \
	X(vkCmdExecuteCommands,           cmd_execute_commands) \
	X(vkCreateQueryPool,              create_query_pool) \
	X(vkDestroyQueryPool,             destroy_query_pool) \
	X(vkGetQueryPoolResults,          get_query_pool_results) \
	X(vkCreateFence,                  create_fence) \
	X(vkDestroyFence,                 destroy_fence) \
	X(vkResetFences,                  reset_fences) \
	X(vkWaitForFences,                wait_for_fences) \
	X(vkGetFenceStatus,               get_fence_status) \
	X(vkCreateSemaphore,              create_semaphore) \
	X(vkDestroySemaphore,             destroy_semaphore) \
	X(vkGetMemoryFdKHR,               get_memory_fd_khr) \
	X(vkCreateSwapchainKHR,           create_swapchain_khr) \
	X(vkDestroySwapchainKHR,          destroy_swapchain_khr) \
	X(vkGetSwapchainImagesKHR,        get_swapchain_images_khr) \
	X(vkQueuePresentKHR,              queue_present_khr)

/** @brief The device functions the layer calls on the next layer: those it intercepts, and the one
 *         its hashing uses besides.
 */
#define DEVICE_CALLS(X) DEVICE_HOOKS(X) X(vkInvalidateMappedMemoryRanges, -)

/** @brief A device the trace knows. */
struct device_data {
	uint64_t                         submits;          //!< Its submits so far.
	VkPhysicalDeviceMemoryProperties memory;           //!< Its memory types and heaps.
	// Content hashing: a command pool per queue family and a staging buffer, which hash_lock gives
	// to one submitting thread at a time. A thread that destroys a buffer or an image, or frees
	// memory, holds hash_lock too, so that nothing a hash copy reads goes away while the copy is
	// recorded, submitted and waited for.
	VkBuffer                         staging;          //!< The staging buffer, or VK_NULL_HANDLE.
	VkDeviceMemory                   staging_memory;   //!< Its memory, or VK_NULL_HANDLE.
	VkDeviceSize                     staging_size;     //!< Its bytes; 0 while there is none.
	struct pool_map                  pools;            //!< The layer's command pools, by family.
	void                            *staging_mapped;   //!< Its mapping.
	pthread_mutex_t                  hash_lock;        //!< Held while hashing and while destroying.
	VkDevice                         device;           //!< The device.
	PFN_vkSetDeviceLoaderData        set_loader_data;  //!< The loader's, or nullptr.
	struct pendings                  pending;          //!< Its submissions with host ranges to hash.
	// The next layer's functions.
#define MEMBER(name, hook) PFN_##name name;
	DEVICE_CALLS(MEMBER)
#undef MEMBER
	VkBool32                         staging_coherent; //!< The staging memory is host-coherent.
	VkBool32                         staging_cached;   //!< The staging memory is host-cached.
};

// ---- state --------------------------------------------------------------------------------------
// One lock guards all of it and the log, except each device's hashing objects, which its hash_lock
// guards. A thread that takes both takes hash_lock first.

/** @brief The lock of the trace's state and log. */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/** @brief The log, or nullptr until the first instance reads the settings. */
static FILE *g_out;

/** @brief The last log line's number. */
static uint64_t g_seq;

/** @brief The threads that logged so far. */
static uint32_t g_threads;

/** @brief This thread's number in the log, or 0 until it logs. */
static thread_local uint32_t g_thread;

/** @brief The fields of the log line being built. */
static struct text g_line;

/** @brief The instances, by dispatch key. */
static struct instance_map g_instances;

/** @brief The devices, by dispatch key. */
static struct device_map g_devices;

/** @brief Each physical device's instance. */
static struct instance_map g_physical;

/** @brief Each live handle's number, by kind. */
static struct u32_map g_names[KIND_COUNT];

/** @brief The handles named so far, by kind. */
static uint32_t g_counts[KIND_COUNT];

/** @brief The memory allocations. */
static struct memory_map g_mem;

/** @brief The buffers. */
static struct buffer_map g_buf;

/** @brief The images, and at a swapchain's handle ^ 1 the template of its images. */
static struct image_map g_img;

/** @brief Each image view's image. */
static struct handle_map g_view;

/** @brief The descriptor set layouts. */
static struct layout_map g_layout;

/** @brief The descriptor sets. */
static struct set_map g_set;

/** @brief Each descriptor pool's sets. */
static struct list_map g_pool_sets;

/** @brief The command buffers. */
static struct cb_map g_cb;

/** @brief Each command pool's command buffers. */
static struct list_map g_cmdpool_cbs;

/** @brief Each queue's family. */
static struct u32_map g_queue;

/** @brief Each swapchain's images. */
static struct list_map g_swapchain_images;

/** @brief A name that VKTRACE_HASH selects: "img" or "buf" and a number. */
struct hash_name {
	uint64_t    number; //!< The number it names, or UINT64_MAX for none: digits with a leading zero
	                    //!< name nothing.
	char const *text;   //!< The item, in config's hash_items.
	size_t      length; //!< Its length.
};

/** @brief A range of submits that VKTRACE_HASH_SUBMITS selects. */
struct submit_range {
	uint64_t first; //!< The first.
	uint64_t last;  //!< The last.
};

VECTOR(u32_list, uint32_t)
VECTOR(hash_names, struct hash_name)
VECTOR(submit_ranges, struct submit_range)

/** @brief A SPIR-V file's name, or the names of identical files joined with '|'. */
struct spirv_name {
	char   *text;   //!< The names, on the heap.
	size_t  length; //!< Their length.
};

SUPPORT_SORTED_MAP(spirv_map, struct spirv_name, reallocate)

/** @brief The settings, read from the environment when the first instance is created. */
static struct config {
	char                 *spirv_dirs;            //!< VKTRACE_SPIRV, or nullptr.
	char                 *hash_items;            //!< A copy of VKTRACE_HASH, or nullptr.
	struct spirv_map      spirv;                 //!< Each SPIR-V file's name, by FNV-1a 64.
	struct u32_list       hash_dispatch;         //!< dispatch=N selectors.
	struct hash_names     hash_names;            //!< bufN and imgN selectors.
	struct submit_ranges  hash_submits;          //!< VKTRACE_HASH_SUBMITS's ranges.
	bool                  spirv_scanned;         //!< VKTRACE_SPIRV was scanned.
	bool                  hash_i2b;              //!< The i2b selector.
	bool                  hash_copydst;          //!< The copydst selector.
	bool                  hash_storage;          //!< The storage selector.
	bool                  hash_all;              //!< The all selector.
	bool                  hash_any;              //!< Any selector.
	bool                  hash_need_dispatch;    //!< A selector that needs each dispatch's storage.
	bool                  hash_dispatch_submits; //!< VKTRACE_HASH_SUBMITS's dispatch.
} g_cfg;

/** @brief Takes g_lock. A mutex of PTHREAD_MUTEX_INITIALIZER's default type fails no lock or unlock
 *         of a caller that holds it as it should.
 */
static void
lock (void)
{
	pthread_mutex_lock(&g_lock);
}

/** @brief Releases g_lock. */
static void
unlock (void)
{
	pthread_mutex_unlock(&g_lock);
}

// ---- settings -----------------------------------------------------------------------------------

/** @brief The number that a text is, when it is one.
 *
 * @param text   The text.
 * @param length Its length.
 * @return       The number, or UINT64_MAX for a text that is empty, holds anything but digits, or
 *               is too large.
 */
static uint64_t
number (char const *text,
        size_t      length)
{
	uint64_t value = 0;
	for (size_t i = 0; i < length; ++i) {
		uint8_t const digit = (uint8_t)text[i] - '0';
		if (digit > 9 || value > (UINT64_MAX - 9) / 10)
			return UINT64_MAX;
		value = value * 10 + digit;
	}
	return length ? value : UINT64_MAX;
}

/** @brief Reports an item of a variable that means nothing, which the layer ignores. Caller holds
 *         g_lock, and g_out is open.
 *
 * @param variable The variable's name.
 * @param item     The item.
 * @param length   Its length.
 */
static void
ignored (char const *variable,
         char const *item,
         size_t      length)
{
	int const n = (int)length;
	fprintf(stderr, "vktrace: ignoring %s item '%.*s'\n", variable, n, item);
	fprintf(g_out, "# ignored %s item %.*s\n", variable, n, item);
}

/** @brief Opens the log and reads the settings, once. Caller holds g_lock. */
static void
load_config (void)
{
	if (g_out)
		return;
	char const *const file = getenv("VKTRACE_FILE");
	char const *const pattern = file && *file ? file : "/tmp/vktrace.%p.log";
	// The first %p is the process ID.
	char const *const pid = strstr(pattern, "%p");
	struct text path = {};
	if (pid)
		text_printf(&path, "%.*s%d%s", (int)(pid - pattern), pattern, (int)getpid(), pid + 2);
	else
		text_append(&path, pattern, strlen(pattern));
	g_out = fopen(path.bytes, "we");
	if (!g_out) {
		int const err = errno;
		char buf[64];
		fprintf(stderr, "vktrace: cannot write %s (%s); logging to standard error\n", path.bytes,
		        strerror_r(err, buf, sizeof buf));
		g_out = stderr;
	}
	text_fini(&path);
	// Without the bigger buffer, the log is only slower.
	setvbuf(g_out, nullptr, _IOFBF, 1 << 20);
	// Unset and empty variables read alike: "-" in the header.
	char const *hash = getenv("VKTRACE_HASH");
	char const *submits = getenv("VKTRACE_HASH_SUBMITS");
	char const *const spirv = getenv("VKTRACE_SPIRV");
	if (!hash)
		hash = "";
	if (!submits)
		submits = "";
	if (spirv && *spirv)
		g_cfg.spirv_dirs = copy_text(spirv, strlen(spirv));
	fprintf(g_out, "# vktrace 1 pid=%d hash=%s hash_submits=%s spirv=%s\n", (int)getpid(), *hash ? hash : "-",
	        *submits ? submits : "-", g_cfg.spirv_dirs ? g_cfg.spirv_dirs : "-");

#define TEXT_IS(at, n, literal) ((n) == sizeof literal - 1 && !memcmp(at, literal, sizeof literal - 1))
#define TEXT_STARTS(at, n, literal) ((n) >= sizeof literal - 1 && !memcmp(at, literal, sizeof literal - 1))
	g_cfg.hash_items = copy_text(hash, strlen(hash));
	for (char const *at = g_cfg.hash_items; *at;) {
		size_t const n = strcspn(at, ",");
		char const *const item = at;
		at += n;
		if (*at)
			++at;
		if (!n)
			continue;
		bool const dispatch = TEXT_STARTS(item, n, "dispatch=");
		bool const image = TEXT_STARTS(item, n, "img");
		bool const named = TEXT_STARTS(item, n, "buf") || image;
		size_t const skip = dispatch ? sizeof "dispatch=" - 1 : 3;
		uint64_t const value = dispatch || named ? number(item + skip, n - skip) : UINT64_MAX;
		if (TEXT_IS(item, n, "i2b")) {
			g_cfg.hash_i2b = true;
		} else if (TEXT_IS(item, n, "copydst")) {
			g_cfg.hash_copydst = true;
		} else if (TEXT_IS(item, n, "storage")) {
			g_cfg.hash_storage = true;
		} else if (TEXT_IS(item, n, "all")) {
			g_cfg.hash_all = true;
		} else if (value > UINT32_MAX) {
			ignored("VKTRACE_HASH", item, n);
			continue;
		} else if (dispatch) {
			*u32_list_push(&g_cfg.hash_dispatch) = (uint32_t)value;
		} else {
			// A name matches its kind's prefix and a creation number in decimal, without leading zeros.
			bool const canonical = item[skip] != '0' || n == skip + 1;
			*hash_names_push(&g_cfg.hash_names) = (struct hash_name){
				.text = item, .length = n, .number = canonical ? value : UINT64_MAX,
			};
		}
		g_cfg.hash_any = true;
	}
	g_cfg.hash_need_dispatch = g_cfg.hash_storage || g_cfg.hash_dispatch.count;
	for (char const *at = submits; *at;) {
		size_t const n = strcspn(at, ",");
		char const *const item = at;
		at += n;
		if (*at)
			++at;
		if (!n)
			continue;
		size_t dash = 0;
		while (dash < n && item[dash] != '-')
			++dash;
		uint64_t const first = number(item, dash);
		uint64_t const last = dash < n ? number(item + dash + 1, n - dash - 1) : first;
		if (TEXT_IS(item, n, "dispatch"))
			g_cfg.hash_dispatch_submits = true;
		else if (first > last || last == UINT64_MAX)
			ignored("VKTRACE_HASH_SUBMITS", item, n);
		else
			*submit_ranges_push(&g_cfg.hash_submits) = (struct submit_range){first, last};
	}
#undef TEXT_STARTS
#undef TEXT_IS
}

/** @brief Records the SPIR-V files below a directory, by the FNV-1a 64 of their bytes, in name order.
 *         Caller holds g_lock.
 *
 * @param dir        The directory.
 * @param dir_length Its length.
 * @param rel        Its path below the directory VKTRACE_SPIRV names, "" for that one.
 * @param rel_length The path's length.
 */
static void
scan_spirv (char const *dir,
            size_t      dir_length,
            char const *rel,
            size_t      rel_length)
{
	DIR *const d = opendir(dir);
	if (!d)
		return;
	struct strings names = {};
	for (;;) {
		// readdir() reports an error only through errno.
		errno = 0;
		struct dirent const *const entry = readdir(d);
		if (!entry) {
			if (errno) {
				int const err = errno;
				char buf[64];
				fprintf(stderr, "vktrace: cannot read %s: %s\n", dir, strerror_r(err, buf, sizeof buf));
			}
			break;
		}
		size_t const length = strlen(entry->d_name);
		strings_add_sorted(&names, copy_text(entry->d_name, length), length);
	}
	if (closedir(d)) {
		int const err = errno;
		char buf[64];
		fprintf(stderr, "vktrace: cannot close %s: %s\n", dir, strerror_r(err, buf, sizeof buf));
	}
	for (size_t i = 0; i < names.count; ++i) {
		char const *const n = names.items[i].text;
		size_t const length = names.items[i].length;
		if (!strcmp(n, ".") || !strcmp(n, ".."))
			continue;
		char *full = join_path(dir, dir_length, n, length);
		size_t const sub_length = rel_length ? rel_length + 1 + length : length;
		char *sub = rel_length ? join_path(rel, rel_length, n, length) : copy_text(n, length);
		struct stat st;
		if (!stat(full, &st)) {
			if (S_ISDIR(st.st_mode)) {
				scan_spirv(full, dir_length + 1 + length, sub, sub_length);
			} else if (length > 4 && !memcmp(n + length - 4, ".spv", 4)) {
				FILE *const f = fopen(full, "rbe");
				size_t const size = (size_t)st.st_size;
				unsigned char *data = f ? reallocate(nullptr, size ? size : 1) : nullptr;
				size_t const got = f ? fread(data, 1, size, f) : 0;
				// A file read only: its close has nothing to report.
				if (f)
					fclose(f);
				if (f && got == size) {
					uint64_t const h = fnv(data, size, FNV_BASIS);
					bool added;
					struct spirv_name *const known = spirv_map_slot(&g_cfg.spirv, h, &added);
					if (added) {
						*known = (struct spirv_name){sub, sub_length};
						sub = nullptr;
					} else if (!strstr(known->text, sub)) {
						known->text = reallocate(known->text, known->length + 1 + sub_length + 1);
						known->text[known->length] = '|';
						memcpy(known->text + known->length + 1, sub, sub_length + 1);
						known->length += 1 + sub_length;
					}
				}
				free(data);
				data = nullptr;
			}
		}
		free(sub);
		sub = nullptr;
		free(full);
		full = nullptr;
	}
	strings_fini(&names);
}

/** @brief Scans VKTRACE_SPIRV's directories, once, and logs how many files it found. Caller holds
 *         g_lock.
 */
static void
spirv_files (void)
{
	if (g_cfg.spirv_scanned)
		return;
	g_cfg.spirv_scanned = true;
	for (char const *at = g_cfg.spirv_dirs ? g_cfg.spirv_dirs : ""; *at;) {
		size_t const n = strcspn(at, ":");
		if (n) {
			char *dir = copy_text(at, n);
			scan_spirv(dir, n, "", 0);
			free(dir);
			dir = nullptr;
		}
		at += n;
		if (*at)
			++at;
	}
	fprintf(g_out, "# spirv files=%zu\n", g_cfg.spirv.count);
}

// ---- the log ------------------------------------------------------------------------------------

/** @brief This thread's number in the log, given when it first logs. Caller holds g_lock.
 *
 * @return The number, from 1.
 */
static uint32_t
thread_number (void)
{
	if (!g_thread)
		g_thread = ++g_threads;
	return g_thread;
}

/** @brief Appends formatted words to the line being built. Caller holds g_lock.
 *
 * @param fmt A printf format.
 * @param ... The format's arguments.
 */
[[gnu::format(printf, 1, 2)]]
static void
put (char const *fmt,
     ...)
{
	va_list args;
	va_start(args, fmt);
	text_vprintf(&g_line, fmt, args);
	va_end(args);
}

/** @brief Logs the line built so far, "SEQ tN CALL fields", and starts the next. Caller holds
 *         g_lock.
 *
 * @param call The call's name.
 */
static void
emit (char const *call)
{
	if (g_out) {
		fprintf(g_out, "%" PRIu64 " t%" PRIu32 " %s ", ++g_seq, thread_number(), call);
		if (g_line.length)
			fwrite(g_line.bytes, 1, g_line.length, g_out);
		putc('\n', g_out);
	}
	text_truncate(&g_line, 0);
}

/** @brief Appends formatted words to the line being built and logs it. Caller holds g_lock.
 *
 * @param call The call's name.
 * @param fmt  A printf format.
 * @param ...  The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
emitf (char const *call,
       char const *fmt,
       ...)
{
	va_list args;
	va_start(args, fmt);
	text_vprintf(&g_line, fmt, args);
	va_end(args);
	emit(call);
}

/** @brief Writes out what the log holds. A failure sets the stream's error indicator, which
 *         finish() reports. Caller holds g_lock.
 */
static void
flush (void)
{
	if (g_out)
		fflush(g_out);
}

/** @brief Reports a log that could not be written, at exit. */
[[gnu::destructor]]
static void
finish (void)
{
	lock();
	if (g_out && (fflush(g_out) || ferror(g_out)))
		fputs("vktrace: writing the trace failed; it is incomplete\n", stderr);
	unlock();
}

/** @brief Appends a name from an application or a driver, whose spaces would split a field: spaces,
 *         tabs, newlines and '=' become '_', and an empty or missing name "-". Caller holds g_lock.
 *
 * @param s The name, or nullptr.
 */
static void
put_token (char const *s)
{
	size_t const n = s ? strlen(s) : 0;
	if (!n) {
		text_append(&g_line, "-", 1);
		return;
	}
	text_reserve(&g_line, n);
	char *const out = g_line.bytes + g_line.length;
	for (size_t i = 0; i < n; ++i)
		out[i] = s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '=' ? '_' : s[i];
	g_line.length += n;
	g_line.bytes[g_line.length] = '\0';
}

/** @brief Appends bytes in hex. Caller holds g_lock.
 *
 * @param data The bytes.
 * @param n    How many.
 */
static void
put_bytes_hex (void const *data,
               size_t      n)
{
	static char const digits[] = "0123456789abcdef";
	unsigned char const *const p = data;
	text_reserve(&g_line, 2 * n);
	char *const out = g_line.bytes + g_line.length;
	for (size_t i = 0; i < n; ++i) {
		out[2 * i] = digits[p[i] >> 4];
		out[2 * i + 1] = digits[p[i] & 15];
	}
	g_line.length += 2 * n;
	g_line.bytes[g_line.length] = '\0';
}

/** @brief Appends a subresource range: aspect/base mip+levels/base layer+layers. Caller holds g_lock.
 *
 * @param r The range.
 */
static void
put_range (VkImageSubresourceRange const *r)
{
	put("%s/%" PRIu32 "+%s/%" PRIu32 "+%s", hex(r->aspectMask).text, r->baseMipLevel,
	    count_text(r->levelCount, VK_REMAINING_MIP_LEVELS).text, r->baseArrayLayer,
	    count_text(r->layerCount, VK_REMAINING_ARRAY_LAYERS).text);
}

/** @brief Appends subresource layers: aspect/mip/base layer+layers. Caller holds g_lock.
 *
 * @param l The layers.
 */
static void
put_layers (VkImageSubresourceLayers const *l)
{
	put("%s/%" PRIu32 "/%" PRIu32 "+%" PRIu32, hex(l->aspectMask).text, l->mipLevel, l->baseArrayLayer,
	    l->layerCount);
}

/** @brief Appends a structure chain's types: "[A,B]". Caller holds g_lock.
 *
 * @param chain The chain.
 */
static void
put_chain (void const *chain)
{
	put("[");
	for (VkBaseInStructure const *p = chain; p; p = p->pNext)
		put("%s%s", p == chain ? "" : ",", stype_word(p->sType).text);
	put("]");
}

/** @brief A handle's name by its key: "null", its kind's prefix and number, or "?" and the key in
 *         hex for a handle the trace does not know. Caller holds g_lock.
 *
 * @param k   Its kind.
 * @param key Its key.
 * @return    The name.
 */
static struct name
key_name (enum kind k,
          uint64_t  key)
{
	struct name n = {"null"};
	if (!key)
		return n;
	uint32_t const *const number = u32_map_find(&g_names[k], key);
	if (number)
		snprintf(n.text, sizeof n.text, "%s%" PRIu32, KIND_PREFIX[k], *number);
	else
		snprintf(n.text, sizeof n.text, "?0x%" PRIx64, key);
	return n;
}

/** @brief A handle's name. Caller holds g_lock.
 *
 * @param k      Its kind.
 * @param handle The handle.
 * @return       Its name.
 */
static struct name
name_of (enum kind   k,
         void const *handle)
{
	return key_name(k, key_of(handle));
}

/** @brief Whether the trace knows a handle. Caller holds g_lock.
 *
 * @param k      Its kind.
 * @param handle The handle.
 * @return       true if it has a name.
 */
static bool
known (enum kind   k,
       void const *handle)
{
	return u32_map_find(&g_names[k], key_of(handle));
}

/** @brief Names a new handle with its kind's next number. Caller holds g_lock.
 *
 * @param k      Its kind.
 * @param handle The handle.
 * @return       Its name, or "null" for the null handle, which gets no number.
 */
static struct name
fresh (enum kind   k,
       void const *handle)
{
	uint64_t const key = key_of(handle);
	if (key) {
		bool added;
		*u32_map_slot(&g_names[k], key, &added) = ++g_counts[k];
	}
	return key_name(k, key);
}

/** @brief Names the handle a creating call made, if it succeeded. Caller holds g_lock.
 *
 * @param r      The call's result.
 * @param k      The handle's kind.
 * @param handle The handle.
 * @return       Its new name, or "null" after a failure, whose handle is undefined.
 */
static struct name
created (VkResult    r,
         enum kind   k,
         void const *handle)
{
	return r == VK_SUCCESS ? fresh(k, handle) : (struct name){"null"};
}

/** @brief Forgets a handle's name. Caller holds g_lock.
 *
 * @param k      Its kind.
 * @param handle The handle.
 * @return       The name it had.
 */
static struct name
forget (enum kind   k,
        void const *handle)
{
	struct name const n = name_of(k, handle);
	u32_map_remove(&g_names[k], key_of(handle));
	return n;
}

/** @brief An instance by a dispatchable handle of it. Caller holds g_lock.
 *
 * @param dispatchable The handle.
 * @return             The instance, or nullptr.
 */
static struct instance_data *
instance_of (void const *dispatchable)
{
	struct instance_data *const *const data = instance_map_find(&g_instances, key_of(dispatch_key(dispatchable)));
	return data ? *data : nullptr;
}

/** @brief A device by a dispatchable handle of it. Caller holds g_lock.
 *
 * @param dispatchable The handle.
 * @return             The device, or nullptr.
 */
static struct device_data *
device_of (void const *dispatchable)
{
	struct device_data *const *const data = device_map_find(&g_devices, key_of(dispatch_key(dispatchable)));
	return data ? *data : nullptr;
}

/** @brief A physical device's instance. Caller holds g_lock.
 *
 * @param pd The physical device.
 * @return   Its instance, or nullptr.
 */
static struct instance_data *
instance_of_physical (VkPhysicalDevice pd)
{
	struct instance_data *const *const data = instance_map_find(&g_physical, key_of(pd));
	return data ? *data : instance_of(pd);
}

/** @brief A device by a dispatchable handle of it, under g_lock.
 *
 * @param dispatchable The handle.
 * @return             The device, or nullptr.
 */
static struct device_data *
dev (void const *dispatchable)
{
	lock();
	struct device_data *const d = device_of(dispatchable);
	unlock();
	return d;
}

/** @brief A structure of a type in a chain.
 *
 * @param chain The chain.
 * @param type  The type.
 * @return      The first structure of the type, or nullptr.
 */
static void const *
find_in_chain (void const      *chain,
               VkStructureType  type)
{
	for (VkBaseInStructure const *p = chain; p; p = p->pNext)
		if (p->sType == type)
			return p;
	return nullptr;
}

// ---- structures ---------------------------------------------------------------------------------

/** @brief Logs the bits of a feature structure that are on, if any are. Caller holds g_lock.
 *
 * @param device The device's name.
 * @param sname  The structure's name.
 * @param s      The structure.
 * @param bits   Its bits.
 * @param count  How many.
 */
static void
feature_line (struct name const            *device,
              char const                   *sname,
              void const                   *s,
              struct vkt_feature_bit const *bits,
              size_t                        count)
{
	put("dev=%s struct=%s on=[", device->text, sname);
	bool on = false;
	for (size_t i = 0; i < count; ++i) {
		VkBool32 b;
		memcpy(&b, (unsigned char const *)s + bits[i].offset, sizeof b);
		if (b) {
			put("%s%s", on ? "," : "", bits[i].name);
			on = true;
		}
	}
	if (on)
		emitf("DeviceFeatures", "]");
	else
		text_truncate(&g_line, 0);
}

/** @brief The bits of VkPhysicalDeviceFeatures. */
static constexpr size_t CORE_FEATURE_COUNT = sizeof VKT_CORE_FEATURE_BITS / sizeof *VKT_CORE_FEATURE_BITS;

/** @brief The feature structures the tables know. */
static constexpr size_t FEATURE_STRUCT_COUNT = sizeof VKT_FEATURE_STRUCTS / sizeof *VKT_FEATURE_STRUCTS;

/** @brief Whether the feature lines describe a structure type. */
static bool
feature_type (VkStructureType t)
{
	if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
		return true;
	for (size_t i = 0; i < FEATURE_STRUCT_COUNT; ++i)
		if (VKT_FEATURE_STRUCTS[i].type == t)
			return true;
	return false;
}

/** @brief Logs a device's enabled features: a DeviceFeatures line per structure with bits on, in the
 *         order given, and a DeviceChain line with the other structures of the chain, if any. Caller
 *         holds g_lock.
 *
 * @param device The device's name.
 * @param core   VkDeviceCreateInfo's pEnabledFeatures, or nullptr.
 * @param chain  Its pNext chain.
 */
static void
log_features (struct name const              *device,
              VkPhysicalDeviceFeatures const *core,
              void const                     *chain)
{
	if (core)
		feature_line(device, "VkPhysicalDeviceFeatures", core, VKT_CORE_FEATURE_BITS, CORE_FEATURE_COUNT);
	// The chain is read with memcpy(): its structures are of types this file does not know.
	for (void const *p = chain; p;) {
		VkStructureType t;
		memcpy(&t, p, sizeof t);
		if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
			feature_line(device, "VkPhysicalDeviceFeatures", &((VkPhysicalDeviceFeatures2 const *)p)->features,
			             VKT_CORE_FEATURE_BITS, CORE_FEATURE_COUNT);
		for (size_t i = 0; i < FEATURE_STRUCT_COUNT; ++i)
			if (VKT_FEATURE_STRUCTS[i].type == t)
				feature_line(device, VKT_FEATURE_STRUCTS[i].name, p, VKT_FEATURE_STRUCTS[i].bits,
				             VKT_FEATURE_STRUCTS[i].count);
		memcpy(&p, (unsigned char const *)p + offsetof(VkBaseInStructure, pNext), sizeof p);
	}
	put("dev=%s other=[", device->text);
	bool other = false;
	for (void const *p = chain; p;) {
		VkStructureType t;
		memcpy(&t, p, sizeof t);
		if (!feature_type(t)) {
			put("%s%s", other ? "," : "", stype_word(t).text);
			other = true;
		}
		memcpy(&p, (unsigned char const *)p + offsetof(VkBaseInStructure, pNext), sizeof p);
	}
	if (other)
		emitf("DeviceChain", "]");
	else
		text_truncate(&g_line, 0);
}

// ---- layer entry points: instance ---------------------------------------------------------------

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
get_instance_proc_addr (VkInstance  instance,
                        char const *name);

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
get_device_proc_addr (VkDevice    device,
                      char const *name);

/** @brief vkCreateInstance: reads the settings with the first instance. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_instance (VkInstanceCreateInfo const  *ci,
                 VkAllocationCallbacks const *alloc,
                 VkInstance                  *out)
{
	VkLayerInstanceCreateInfo *link = nullptr;
	for (VkBaseInStructure const *p = ci->pNext; p; p = p->pNext)
		if (p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
		    && ((VkLayerInstanceCreateInfo const *)p)->function == VK_LAYER_LINK_INFO)
			// The loader hands the chain on for each layer to take its link off.
			link = (VkLayerInstanceCreateInfo *)p;
	if (!link)
		return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkGetInstanceProcAddr const gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	PFN_vkCreateInstance const create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
	if (!create)
		return VK_ERROR_INITIALIZATION_FAILED;
	VkResult const r = create(ci, alloc, out);
	lock();
	load_config();
	if (r != VK_SUCCESS) {
		emitf("CreateInstance", "rc=%d", r);
		unlock();
		return r;
	}
	struct instance_data *const data = reallocate(nullptr, sizeof *data);
	*data = (struct instance_data){.instance = *out, .gipa = gipa};
#define LOAD(name) data->name = (PFN_##name)gipa(*out, #name);
	INSTANCE_CALLS(LOAD)
#undef LOAD
	bool added;
	*instance_map_slot(&g_instances, key_of(dispatch_key(*out)), &added) = data;
	VkApplicationInfo const *const app = ci->pApplicationInfo;
	put("inst=%s api=%s app=", fresh(KIND_INSTANCE, *out).text, app ? version_text(app->apiVersion).text : "-");
	put_token(app ? app->pApplicationName : nullptr);
	put(" layers=[");
	for (uint32_t i = 0; i < ci->enabledLayerCount; ++i) {
		if (i)
			put(",");
		put_token(ci->ppEnabledLayerNames[i]);
	}
	put("] exts=[");
	for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) {
		if (i)
			put(",");
		put_token(ci->ppEnabledExtensionNames[i]);
	}
	emitf("CreateInstance", "] rc=0");
	flush();
	unlock();
	return r;
}

/** @brief vkDestroyInstance. */
static VKAPI_ATTR void VKAPI_CALL
destroy_instance (VkInstance                   instance,
                  VkAllocationCallbacks const *alloc)
{
	lock();
	struct instance_data *data = instance_of(instance);
	emitf("DestroyInstance", "inst=%s", forget(KIND_INSTANCE, instance).text);
	flush();
	unlock();
	if (!data)
		return;
	uint64_t const key = key_of(dispatch_key(instance));
	data->vkDestroyInstance(instance, alloc);
	lock();
	instance_map_remove(&g_instances, key);
	size_t kept = 0;
	for (size_t i = 0; i < g_physical.count; ++i) {
		if (g_physical.values[i] == data)
			continue;
		g_physical.keys[kept] = g_physical.keys[i];
		g_physical.values[kept++] = g_physical.values[i];
	}
	g_physical.count = kept;
	unlock();
	free(data);
	data = nullptr;
}

/** @brief vkEnumeratePhysicalDevices: names the devices it lists. */
static VKAPI_ATTR VkResult VKAPI_CALL
enumerate_physical_devices (VkInstance        instance,
                            uint32_t         *count,
                            VkPhysicalDevice *out)
{
	lock();
	struct instance_data *const data = instance_of(instance);
	unlock();
	VkResult const r = data->vkEnumeratePhysicalDevices(instance, count, out);
	if (!out || (r != VK_SUCCESS && r != VK_INCOMPLETE))
		return r;
	lock();
	put("inst=%s pds=[", name_of(KIND_INSTANCE, instance).text);
	for (uint32_t i = 0; i < *count; ++i) {
		bool added;
		*instance_map_slot(&g_physical, key_of(out[i]), &added) = data;
		put("%s%s", i ? "," : "",
		    (known(KIND_PHYSICAL, out[i]) ? name_of(KIND_PHYSICAL, out[i]) : fresh(KIND_PHYSICAL, out[i])).text);
	}
	emitf("EnumeratePhysicalDevices", "] rc=%d", r);
	unlock();
	return r;
}

// ---- device -------------------------------------------------------------------------------------

/** @brief vkCreateDevice: logs the device, its queues, extensions, features and memory types. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_device (VkPhysicalDevice             physical,
               VkDeviceCreateInfo const    *ci,
               VkAllocationCallbacks const *alloc,
               VkDevice                    *out)
{
	VkLayerDeviceCreateInfo *link = nullptr;
	PFN_vkSetDeviceLoaderData set_loader_data = nullptr;
	for (VkBaseInStructure const *p = ci->pNext; p; p = p->pNext) {
		if (p->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO)
			continue;
		// The loader hands the chain on for each layer to take its link off.
		VkLayerDeviceCreateInfo *const l = (VkLayerDeviceCreateInfo *)p;
		if (l->function == VK_LAYER_LINK_INFO)
			link = l;
		else if (l->function == VK_LOADER_DATA_CALLBACK)
			set_loader_data = l->u.pfnSetDeviceLoaderData;
	}
	if (!link)
		return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkGetInstanceProcAddr const gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	PFN_vkGetDeviceProcAddr const gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	lock();
	struct instance_data *const inst = instance_of_physical(physical);
	unlock();
	if (!inst)
		return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkCreateDevice const create = (PFN_vkCreateDevice)gipa(inst->instance, "vkCreateDevice");
	VkResult const r = create(physical, ci, alloc, out);
	lock();
	if (r != VK_SUCCESS) {
		emitf("CreateDevice", "pd=%s rc=%d", name_of(KIND_PHYSICAL, physical).text, r);
		unlock();
		return r;
	}
	struct device_data *const d = reallocate(nullptr, sizeof *d);
	*d = (struct device_data){
		.device = *out, .set_loader_data = set_loader_data, .staging_coherent = true, .staging_cached = true,
	};
	if (pthread_mutex_init(&d->hash_lock, nullptr)) {
		fputs("vktrace: cannot make a lock\n", stderr);
		abort();
	}
#define LOAD(name, hook) d->name = (PFN_##name)gdpa(*out, #name);
	DEVICE_CALLS(LOAD)
#undef LOAD
	d->vkGetDeviceProcAddr = gdpa;
	inst->vkGetPhysicalDeviceMemoryProperties(physical, &d->memory);
	bool added;
	*device_map_slot(&g_devices, key_of(dispatch_key(*out)), &added) = d;
	VkPhysicalDeviceProperties props = {};
	inst->vkGetPhysicalDeviceProperties(physical, &props);
	struct name const device = fresh(KIND_DEVICE, *out);
	put("dev=%s pd=%s name=", device.text, name_of(KIND_PHYSICAL, physical).text);
	put_token(props.deviceName);
	put(" api=%s driver=0x%" PRIx32 " vendor=0x%" PRIx32 " device=0x%" PRIx32 " queues=[",
	    version_text(props.apiVersion).text, props.driverVersion, props.vendorID, props.deviceID);
	for (uint32_t i = 0; i < ci->queueCreateInfoCount; ++i)
		put("%sf%" PRIu32 "*%" PRIu32, i ? "," : "", ci->pQueueCreateInfos[i].queueFamilyIndex,
		    ci->pQueueCreateInfos[i].queueCount);
	put("] exts=[");
	// In name order: the order given says nothing about the device.
	struct strings sorted = {};
	for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) {
		size_t const mark = g_line.length;
		put_token(ci->ppEnabledExtensionNames[i]);
		size_t const length = g_line.length - mark;
		strings_add_sorted(&sorted, copy_text(g_line.bytes + mark, length), length);
		text_truncate(&g_line, mark);
	}
	for (size_t i = 0; i < sorted.count; ++i) {
		if (i)
			put(",");
		text_append(&g_line, sorted.items[i].text, sorted.items[i].length);
	}
	strings_fini(&sorted);
	emitf("CreateDevice", "] rc=0");
	log_features(&device, ci->pEnabledFeatures, ci->pNext);
	for (uint32_t i = 0; i < d->memory.memoryTypeCount; ++i)
		emitf("MemoryType", "dev=%s index=%" PRIu32 " flags=%s heap=%" PRIu32 " heapsize=%" PRIu64, device.text, i,
		      hex(d->memory.memoryTypes[i].propertyFlags).text, d->memory.memoryTypes[i].heapIndex,
		      d->memory.memoryHeaps[d->memory.memoryTypes[i].heapIndex].size);
	flush();
	unlock();
	return r;
}

/** @brief Destroys a device's hashing objects.
 *
 * @param d The device.
 */
static void
release_hashing (struct device_data *d)
{
	pthread_mutex_lock(&d->hash_lock);
	for (size_t i = 0; i < d->pools.count; ++i)
		d->vkDestroyCommandPool(d->device, d->pools.values[i].pool, nullptr);
	free(d->pools.keys);
	d->pools.keys = nullptr;
	free(d->pools.values);
	d->pools.values = nullptr;
	d->pools = (struct pool_map){};
	if (d->staging)
		d->vkDestroyBuffer(d->device, d->staging, nullptr);
	if (d->staging_memory)
		d->vkFreeMemory(d->device, d->staging_memory, nullptr);
	d->staging = VK_NULL_HANDLE;
	d->staging_memory = VK_NULL_HANDLE;
	d->staging_mapped = nullptr;
	d->staging_size = 0;
	pthread_mutex_unlock(&d->hash_lock);
}

/** @brief Frees host ranges and empties them.
 *
 * @param ranges The ranges.
 */
static void
host_ranges_fini (struct host_ranges *ranges)
{
	free(ranges->items);
	ranges->items = nullptr;
	*ranges = (struct host_ranges){};
}

/** @brief vkDestroyDevice. */
static VKAPI_ATTR void VKAPI_CALL
destroy_device (VkDevice                     device,
                VkAllocationCallbacks const *alloc)
{
	lock();
	struct device_data *d = device_of(device);
	emitf("DestroyDevice", "dev=%s", forget(KIND_DEVICE, device).text);
	flush();
	unlock();
	if (!d)
		return;
	release_hashing(d);
	uint64_t const key = key_of(dispatch_key(device));
	d->vkDestroyDevice(device, alloc);
	lock();
	device_map_remove(&g_devices, key);
	unlock();
	for (size_t i = 0; i < d->pending.count; ++i)
		host_ranges_fini(&d->pending.items[i].ranges);
	free(d->pending.items);
	d->pending.items = nullptr;
	// Nothing holds it now; a default mutex that nothing holds is destroyed without fail.
	pthread_mutex_destroy(&d->hash_lock);
	free(d);
	d = nullptr;
}

/** @brief Logs a queue that a device gave out. Caller holds g_lock.
 *
 * @param d      The device.
 * @param q      The queue.
 * @param family Its family.
 * @param index  Its index in the family.
 * @param call   The call's name.
 */
static void
note_queue (struct device_data *d,
            VkQueue             q,
            uint32_t            family,
            uint32_t            index,
            char const         *call)
{
	bool const was_known = known(KIND_QUEUE, q);
	bool added;
	*u32_map_slot(&g_queue, key_of(q), &added) = family;
	emitf(call, "dev=%s family=%" PRIu32 " index=%" PRIu32 " queue=%s", name_of(KIND_DEVICE, d->device).text,
	      family, index, (was_known ? name_of(KIND_QUEUE, q) : fresh(KIND_QUEUE, q)).text);
}

/** @brief vkGetDeviceQueue. */
static VKAPI_ATTR void VKAPI_CALL
get_device_queue (VkDevice  device,
                  uint32_t  family,
                  uint32_t  index,
                  VkQueue  *out)
{
	struct device_data *const d = dev(device);
	d->vkGetDeviceQueue(device, family, index, out);
	lock();
	if (*out)
		note_queue(d, *out, family, index, "GetDeviceQueue");
	unlock();
}

/** @brief vkGetDeviceQueue2. */
static VKAPI_ATTR void VKAPI_CALL
get_device_queue2 (VkDevice                  device,
                   VkDeviceQueueInfo2 const *info,
                   VkQueue                  *out)
{
	struct device_data *const d = dev(device);
	d->vkGetDeviceQueue2(device, info, out);
	lock();
	if (*out)
		note_queue(d, *out, info->queueFamilyIndex, info->queueIndex, "GetDeviceQueue2");
	unlock();
}

// ---- memory and resources -----------------------------------------------------------------------

/** @brief vkAllocateMemory. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_memory (VkDevice                     device,
                 VkMemoryAllocateInfo const  *ai,
                 VkAllocationCallbacks const *alloc,
                 VkDeviceMemory              *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkAllocateMemory(device, ai, alloc, out);
	lock();
	VkMemoryPropertyFlags const flags = ai->memoryTypeIndex < d->memory.memoryTypeCount
	                                    ? d->memory.memoryTypes[ai->memoryTypeIndex].propertyFlags : 0;
	if (r == VK_SUCCESS)
		put("mem=%s ", fresh(KIND_MEMORY, *out).text);
	put("size=%" PRIu64 " type=%" PRIu32 " flags=%s", ai->allocationSize, ai->memoryTypeIndex, hex(flags).text);
	bool external = false;
	VkMemoryDedicatedAllocateInfo const *const ded = find_in_chain(ai->pNext,
	                                                               VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
	if (ded)
		put(" dedicated=%s", (ded->buffer ? name_of(KIND_BUFFER, ded->buffer) : name_of(KIND_IMAGE, ded->image)).text);
	VkImportMemoryFdInfoKHR const *const imp = find_in_chain(ai->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
	if (imp) {
		put(" import=fd:%s", hex(imp->handleType).text);
		external = true;
	}
	VkExportMemoryAllocateInfo const *const exp = find_in_chain(ai->pNext, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
	if (exp) {
		put(" export=%s", hex(exp->handleTypes).text);
		external = true;
	}
	VkMemoryAllocateFlagsInfo const *const fl = find_in_chain(ai->pNext, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO);
	if (fl)
		put(" allocflags=%s", hex(fl->flags).text);
	if (r == VK_SUCCESS) {
		bool added;
		*memory_map_slot(&g_mem, key_of(*out), &added) = (struct memory_info){
			.size = ai->allocationSize, .flags = flags, .external = external,
		};
	}
	emitf("AllocateMemory", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkFreeMemory: waits for the device's hashing, and unbinds what was bound to the memory. */
static VKAPI_ATTR void VKAPI_CALL
free_memory (VkDevice                     device,
             VkDeviceMemory               memory,
             VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	pthread_mutex_lock(&d->hash_lock);
	lock();
	if (memory) {
		emitf("FreeMemory", "mem=%s", forget(KIND_MEMORY, memory).text);
		memory_map_remove(&g_mem, key_of(memory));
		// The buffers and images bound to it are bound to nothing now.
		for (size_t i = 0; i < g_buf.count; ++i)
			if (g_buf.values[i].memory == memory)
				g_buf.values[i].memory = VK_NULL_HANDLE;
		for (size_t i = 0; i < g_img.count; ++i)
			if (g_img.values[i].memory == memory)
				g_img.values[i].memory = VK_NULL_HANDLE;
	}
	unlock();
	d->vkFreeMemory(device, memory, alloc);
	pthread_mutex_unlock(&d->hash_lock);
}

/** @brief vkMapMemory: remembers the mapping. */
static VKAPI_ATTR VkResult VKAPI_CALL
map_memory (VkDevice          device,
            VkDeviceMemory    memory,
            VkDeviceSize      offset,
            VkDeviceSize      size,
            VkMemoryMapFlags  flags,
            void            **out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkMapMemory(device, memory, offset, size, flags, out);
	lock();
	struct memory_info *const m = r == VK_SUCCESS ? memory_map_find(&g_mem, key_of(memory)) : nullptr;
	if (m) {
		m->mapped = *out;
		m->map_offset = offset;
		m->map_size = size == VK_WHOLE_SIZE ? m->size - offset : size;
	}
	emitf("MapMemory", "mem=%s offset=%" PRIu64 " size=%s rc=%d", name_of(KIND_MEMORY, memory).text, offset,
	      size_text(size).text, r);
	unlock();
	return r;
}

/** @brief vkUnmapMemory. */
static VKAPI_ATTR void VKAPI_CALL
unmap_memory (VkDevice       device,
              VkDeviceMemory memory)
{
	struct device_data *const d = dev(device);
	lock();
	struct memory_info *const m = memory_map_find(&g_mem, key_of(memory));
	if (m)
		m->mapped = nullptr;
	emitf("UnmapMemory", "mem=%s", name_of(KIND_MEMORY, memory).text);
	unlock();
	d->vkUnmapMemory(device, memory);
}

/** @brief vkGetMemoryFdKHR. */
static VKAPI_ATTR VkResult VKAPI_CALL
get_memory_fd_khr (VkDevice                       device,
                   VkMemoryGetFdInfoKHR const    *info,
                   int                           *fd)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkGetMemoryFdKHR(device, info, fd);
	lock();
	emitf("GetMemoryFdKHR", "mem=%s type=%s rc=%d", name_of(KIND_MEMORY, info->memory).text,
	      hex(info->handleType).text, r);
	unlock();
	return r;
}

/** @brief vkCreateBuffer. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_buffer (VkDevice                     device,
               VkBufferCreateInfo const    *ci,
               VkAllocationCallbacks const *alloc,
               VkBuffer                    *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateBuffer(device, ci, alloc, out);
	lock();
	VkExternalMemoryBufferCreateInfo const *const e = find_in_chain(ci->pNext,
	                                                                VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
	if (r == VK_SUCCESS) {
		bool added;
		*buffer_map_slot(&g_buf, key_of(*out), &added) = (struct buffer_info){
			.device   = d,
			.size     = ci->size,
			.usage    = ci->usage,
			.owner    = VK_QUEUE_FAMILY_IGNORED,
			.external = e != nullptr,
			.sparse   = (ci->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != 0,
		};
	}
	put("buf=%s size=%" PRIu64 " usage=%s flags=%s sharing=%d", created(r, KIND_BUFFER, *out).text, ci->size,
	    hex(ci->usage).text, hex(ci->flags).text, (int)ci->sharingMode);
	if (e)
		put(" external=%s", hex(e->handleTypes).text);
	emitf("CreateBuffer", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroyBuffer: waits for the device's hashing. */
static VKAPI_ATTR void VKAPI_CALL
destroy_buffer (VkDevice                     device,
                VkBuffer                     buffer,
                VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	pthread_mutex_lock(&d->hash_lock);
	lock();
	if (buffer) {
		emitf("DestroyBuffer", "buf=%s", forget(KIND_BUFFER, buffer).text);
		buffer_map_remove(&g_buf, key_of(buffer));
	}
	unlock();
	d->vkDestroyBuffer(device, buffer, alloc);
	pthread_mutex_unlock(&d->hash_lock);
}

/** @brief vkGetBufferMemoryRequirements. */
static VKAPI_ATTR void VKAPI_CALL
get_buffer_memory_requirements (VkDevice              device,
                                VkBuffer              buffer,
                                VkMemoryRequirements *req)
{
	struct device_data *const d = dev(device);
	d->vkGetBufferMemoryRequirements(device, buffer, req);
	lock();
	emitf("GetBufferMemoryRequirements", "buf=%s size=%" PRIu64 " align=%" PRIu64 " types=%s",
	      name_of(KIND_BUFFER, buffer).text, req->size, req->alignment, hex(req->memoryTypeBits).text);
	unlock();
}

/** @brief vkGetImageMemoryRequirements. */
static VKAPI_ATTR void VKAPI_CALL
get_image_memory_requirements (VkDevice              device,
                               VkImage               image,
                               VkMemoryRequirements *req)
{
	struct device_data *const d = dev(device);
	d->vkGetImageMemoryRequirements(device, image, req);
	lock();
	emitf("GetImageMemoryRequirements", "img=%s size=%" PRIu64 " align=%" PRIu64 " types=%s",
	      name_of(KIND_IMAGE, image).text, req->size, req->alignment, hex(req->memoryTypeBits).text);
	unlock();
}

/** @brief Whether memory is imported or exportable. Caller holds g_lock.
 *
 * @param memory The memory.
 * @return       true if it is.
 */
static bool
external_memory (VkDeviceMemory memory)
{
	struct memory_info const *const m = memory_map_find(&g_mem, key_of(memory));
	return m && m->external;
}

/** @brief Records and logs a buffer's bind. Caller holds g_lock.
 *
 * @param buffer The buffer.
 * @param memory The memory.
 * @param offset Where in it.
 * @param r      The bind's result.
 * @param call   The call's name.
 */
static void
bind_buffer (VkBuffer        buffer,
             VkDeviceMemory  memory,
             VkDeviceSize    offset,
             VkResult        r,
             char const     *call)
{
	struct buffer_info *const b = r == VK_SUCCESS ? buffer_map_find(&g_buf, key_of(buffer)) : nullptr;
	if (b) {
		b->memory = memory;
		b->offset = offset;
		if (external_memory(memory))
			b->external = true;
	}
	emitf(call, "buf=%s mem=%s offset=%" PRIu64 " rc=%d", name_of(KIND_BUFFER, buffer).text,
	      name_of(KIND_MEMORY, memory).text, offset, r);
}

/** @brief Records and logs an image's bind. Caller holds g_lock.
 *
 * @param image  The image.
 * @param memory The memory.
 * @param offset Where in it.
 * @param r      The bind's result.
 * @param call   The call's name.
 */
static void
bind_image (VkImage         image,
            VkDeviceMemory  memory,
            VkDeviceSize    offset,
            VkResult        r,
            char const     *call)
{
	struct image_info *const im = r == VK_SUCCESS ? image_map_find(&g_img, key_of(image)) : nullptr;
	if (im) {
		im->memory = memory;
		if (external_memory(memory))
			im->external = true;
	}
	emitf(call, "img=%s mem=%s offset=%" PRIu64 " rc=%d", name_of(KIND_IMAGE, image).text,
	      name_of(KIND_MEMORY, memory).text, offset, r);
}

/** @brief vkBindBufferMemory. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_buffer_memory (VkDevice       device,
                    VkBuffer       buffer,
                    VkDeviceMemory memory,
                    VkDeviceSize   offset)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkBindBufferMemory(device, buffer, memory, offset);
	lock();
	bind_buffer(buffer, memory, offset, r, "BindBufferMemory");
	unlock();
	return r;
}

/** @brief vkBindBufferMemory2. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_buffer_memory2 (VkDevice                      device,
                     uint32_t                      n,
                     VkBindBufferMemoryInfo const *infos)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkBindBufferMemory2(device, n, infos);
	lock();
	for (uint32_t i = 0; i < n; ++i)
		bind_buffer(infos[i].buffer, infos[i].memory, infos[i].memoryOffset, r, "BindBufferMemory2");
	unlock();
	return r;
}

/** @brief vkCreateImage. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_image (VkDevice                     device,
              VkImageCreateInfo const     *ci,
              VkAllocationCallbacks const *alloc,
              VkImage                     *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateImage(device, ci, alloc, out);
	lock();
	VkExternalMemoryImageCreateInfo const *const e = find_in_chain(ci->pNext,
	                                                               VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
	if (r == VK_SUCCESS) {
		// Bound with VkBindImageMemorySwapchainInfoKHR, to the memory of that swapchain's image.
		VkImageSwapchainCreateInfoKHR const *const sc = find_in_chain(ci->pNext,
		                                                              VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR);
		bool added;
		*image_map_slot(&g_img, key_of(*out), &added) = (struct image_info){
			.device    = d,
			.extent    = ci->extent,
			.format    = ci->format,
			.layers    = ci->arrayLayers,
			.usage     = ci->usage,
			.layout    = ci->initialLayout,
			.swapchain = sc && sc->swapchain,
			.external  = e != nullptr,
			.sparse    = (ci->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) != 0,
		};
	}
	put("img=%s type=%s format=%s extent=%" PRIu32 "x%" PRIu32 "x%" PRIu32 " mips=%" PRIu32 " layers=%" PRIu32
	    " samples=%u tiling=%s usage=%s flags=%s sharing=%d initial=%s", created(r, KIND_IMAGE, *out).text,
	    enum_word(vkt_image_type_name(ci->imageType), ci->imageType).text, format_word(ci->format).text,
	    ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers,
	    (unsigned)ci->samples, enum_word(vkt_image_tiling_name(ci->tiling), ci->tiling).text, hex(ci->usage).text,
	    hex(ci->flags).text, (int)ci->sharingMode, layout_word(ci->initialLayout).text);
	if (e)
		put(" external=%s", hex(e->handleTypes).text);
	VkImageFormatListCreateInfo const *const l = find_in_chain(ci->pNext,
	                                                           VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
	if (l) {
		put(" viewformats=[");
		for (uint32_t i = 0; i < l->viewFormatCount; ++i)
			put("%s%s", i ? "," : "", format_word(l->pViewFormats[i]).text);
		put("]");
	}
	emitf("CreateImage", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroyImage: waits for the device's hashing. */
static VKAPI_ATTR void VKAPI_CALL
destroy_image (VkDevice                     device,
               VkImage                      image,
               VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	pthread_mutex_lock(&d->hash_lock);
	lock();
	if (image) {
		emitf("DestroyImage", "img=%s", forget(KIND_IMAGE, image).text);
		image_map_remove(&g_img, key_of(image));
	}
	unlock();
	d->vkDestroyImage(device, image, alloc);
	pthread_mutex_unlock(&d->hash_lock);
}

/** @brief vkBindImageMemory. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_image_memory (VkDevice       device,
                   VkImage        image,
                   VkDeviceMemory memory,
                   VkDeviceSize   offset)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkBindImageMemory(device, image, memory, offset);
	lock();
	bind_image(image, memory, offset, r, "BindImageMemory");
	unlock();
	return r;
}

/** @brief vkBindImageMemory2. */
static VKAPI_ATTR VkResult VKAPI_CALL
bind_image_memory2 (VkDevice                     device,
                    uint32_t                     n,
                    VkBindImageMemoryInfo const *infos)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkBindImageMemory2(device, n, infos);
	lock();
	for (uint32_t i = 0; i < n; ++i)
		bind_image(infos[i].image, infos[i].memory, infos[i].memoryOffset, r, "BindImageMemory2");
	unlock();
	return r;
}

/** @brief vkCreateImageView. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_image_view (VkDevice                     device,
                   VkImageViewCreateInfo const *ci,
                   VkAllocationCallbacks const *alloc,
                   VkImageView                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateImageView(device, ci, alloc, out);
	lock();
	if (r == VK_SUCCESS) {
		bool added;
		*handle_map_slot(&g_view, key_of(*out), &added) = key_of(ci->image);
	}
	put("view=%s img=%s type=%s format=%s range=", created(r, KIND_VIEW, *out).text,
	    name_of(KIND_IMAGE, ci->image).text, enum_word(vkt_image_view_type_name(ci->viewType), ci->viewType).text,
	    format_word(ci->format).text);
	put_range(&ci->subresourceRange);
	put(" swizzle=%s%s%s%s flags=%s", swizzle_letter(ci->components.r), swizzle_letter(ci->components.g),
	    swizzle_letter(ci->components.b), swizzle_letter(ci->components.a), hex(ci->flags).text);
	VkImageViewUsageCreateInfo const *const u = find_in_chain(ci->pNext, VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO);
	if (u)
		put(" viewusage=%s", hex(u->usage).text);
	emitf("CreateImageView", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroyImageView. */
static VKAPI_ATTR void VKAPI_CALL
destroy_image_view (VkDevice                     device,
                    VkImageView                  view,
                    VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (view) {
		emitf("DestroyImageView", "view=%s", forget(KIND_VIEW, view).text);
		handle_map_remove(&g_view, key_of(view));
	}
	unlock();
	d->vkDestroyImageView(device, view, alloc);
}

/** @brief vkCreateSampler. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_sampler (VkDevice                     device,
                VkSamplerCreateInfo const   *ci,
                VkAllocationCallbacks const *alloc,
                VkSampler                   *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateSampler(device, ci, alloc, out);
	lock();
	put("smp=%s mag=%s min=%s mip=%s u=%s v=%s w=%s lodbias=%g aniso=%" PRIu32 ":%g compare=%" PRIu32 ":%s "
	    "lod=%g-%g border=%s unnormalized=%" PRIu32 " flags=%s chain=", created(r, KIND_SAMPLER, *out).text,
	    enum_word(vkt_filter_name(ci->magFilter), ci->magFilter).text,
	    enum_word(vkt_filter_name(ci->minFilter), ci->minFilter).text,
	    enum_word(vkt_sampler_mipmap_mode_name(ci->mipmapMode), ci->mipmapMode).text,
	    enum_word(vkt_sampler_address_mode_name(ci->addressModeU), ci->addressModeU).text,
	    enum_word(vkt_sampler_address_mode_name(ci->addressModeV), ci->addressModeV).text,
	    enum_word(vkt_sampler_address_mode_name(ci->addressModeW), ci->addressModeW).text, ci->mipLodBias,
	    ci->anisotropyEnable, ci->maxAnisotropy, ci->compareEnable,
	    enum_word(vkt_compare_op_name(ci->compareOp), ci->compareOp).text, ci->minLod, ci->maxLod,
	    enum_word(vkt_border_color_name(ci->borderColor), ci->borderColor).text, ci->unnormalizedCoordinates,
	    hex(ci->flags).text);
	put_chain(ci->pNext);
	emitf("CreateSampler", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroySampler. */
static VKAPI_ATTR void VKAPI_CALL
destroy_sampler (VkDevice                     device,
                 VkSampler                    sampler,
                 VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (sampler)
		emitf("DestroySampler", "smp=%s", forget(KIND_SAMPLER, sampler).text);
	unlock();
	d->vkDestroySampler(device, sampler, alloc);
}

/** @brief Appends SPIR-V's size, FNV-1a 64 and the files of VKTRACE_SPIRV that hold it. Caller holds
 *         g_lock.
 *
 * @param code  The SPIR-V.
 * @param bytes Its bytes.
 */
static void
put_spirv (void const *code,
           size_t      bytes)
{
	spirv_files();
	uint64_t const h = fnv(code, bytes, FNV_BASIS);
	struct spirv_name const *const file = spirv_map_find(&g_cfg.spirv, h);
	put("bytes=%zu fnv=%016" PRIx64 " file=%s", bytes, h, file ? file->text : "?");
}

/** @brief vkCreateShaderModule. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_shader_module (VkDevice                        device,
                      VkShaderModuleCreateInfo const *ci,
                      VkAllocationCallbacks const    *alloc,
                      VkShaderModule                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateShaderModule(device, ci, alloc, out);
	lock();
	put("shm=%s ", created(r, KIND_SHADER, *out).text);
	put_spirv(ci->pCode, ci->codeSize);
	emitf("CreateShaderModule", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroyShaderModule. */
static VKAPI_ATTR void VKAPI_CALL
destroy_shader_module (VkDevice                     device,
                       VkShaderModule               module,
                       VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (module)
		emitf("DestroyShaderModule", "shm=%s", forget(KIND_SHADER, module).text);
	unlock();
	d->vkDestroyShaderModule(device, module, alloc);
}

/** @brief vkCreatePipelineCache. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_pipeline_cache (VkDevice                         device,
                       VkPipelineCacheCreateInfo const *ci,
                       VkAllocationCallbacks const     *alloc,
                       VkPipelineCache                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreatePipelineCache(device, ci, alloc, out);
	lock();
	emitf("CreatePipelineCache", "pc=%s initial=%zu flags=%s rc=%d", created(r, KIND_CACHE, *out).text,
	      ci->initialDataSize, hex(ci->flags).text, r);
	unlock();
	return r;
}

/** @brief vkDestroyPipelineCache. */
static VKAPI_ATTR void VKAPI_CALL
destroy_pipeline_cache (VkDevice                     device,
                        VkPipelineCache              cache,
                        VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (cache)
		emitf("DestroyPipelineCache", "pc=%s", forget(KIND_CACHE, cache).text);
	unlock();
	d->vkDestroyPipelineCache(device, cache, alloc);
}

/** @brief Frees a map of counts and empties it.
 *
 * @param map The map.
 */
static void
u32_map_fini (struct u32_map *map)
{
	free(map->keys);
	map->keys = nullptr;
	free(map->values);
	map->values = nullptr;
	*map = (struct u32_map){};
}

/** @brief Frees what a set holds and empties it.
 *
 * @param set The set.
 */
static void
set_info_fini (struct set_info *set)
{
	free(set->slots.keys);
	set->slots.keys = nullptr;
	free(set->slots.values);
	set->slots.values = nullptr;
	u32_map_fini(&set->counts);
	*set = (struct set_info){};
}

/** @brief vkCreateDescriptorSetLayout: remembers each binding's count. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_descriptor_set_layout (VkDevice                               device,
                              VkDescriptorSetLayoutCreateInfo const *ci,
                              VkAllocationCallbacks const           *alloc,
                              VkDescriptorSetLayout                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateDescriptorSetLayout(device, ci, alloc, out);
	lock();
	VkDescriptorSetLayoutBindingFlagsCreateInfo const *const flags =
		find_in_chain(ci->pNext, VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
	put("dsl=%s flags=%s bindings=[", created(r, KIND_SET_LAYOUT, *out).text, hex(ci->flags).text);
	struct layout_info info = {};
	for (uint32_t i = 0; i < ci->bindingCount; ++i) {
		VkDescriptorSetLayoutBinding const *const b = &ci->pBindings[i];
		bool added;
		*u32_map_slot(&info.counts, b->binding, &added) = b->descriptorCount;
		bool const flagged = flags && i < flags->bindingCount;
		if (flagged && (flags->pBindingFlags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT))
			info.variable = 1;
		put("%s%" PRIu32 ":%s*%" PRIu32 "@%s", i ? "," : "", b->binding, dtype_word(b->descriptorType).text,
		    b->descriptorCount, hex(b->stageFlags).text);
		if (b->pImmutableSamplers) {
			put(":imm=");
			for (uint32_t k = 0; k < b->descriptorCount; ++k)
				put("%s%s", k ? "|" : "", name_of(KIND_SAMPLER, b->pImmutableSamplers[k]).text);
		}
		if (flagged)
			put(":f%s", hex(flags->pBindingFlags[i]).text);
	}
	if (r == VK_SUCCESS) {
		bool added;
		struct layout_info *const slot = layout_map_slot(&g_layout, key_of(*out), &added);
		if (!added)
			u32_map_fini(&slot->counts);
		*slot = info;
	} else {
		u32_map_fini(&info.counts);
	}
	emitf("CreateDescriptorSetLayout", "] rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroyDescriptorSetLayout. */
static VKAPI_ATTR void VKAPI_CALL
destroy_descriptor_set_layout (VkDevice                     device,
                               VkDescriptorSetLayout        layout,
                               VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (layout)
		emitf("DestroyDescriptorSetLayout", "dsl=%s", forget(KIND_SET_LAYOUT, layout).text);
	struct layout_info *const info = layout_map_find(&g_layout, key_of(layout));
	if (info) {
		u32_map_fini(&info->counts);
		layout_map_remove(&g_layout, key_of(layout));
	}
	unlock();
	d->vkDestroyDescriptorSetLayout(device, layout, alloc);
}

/** @brief vkCreatePipelineLayout. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_pipeline_layout (VkDevice                          device,
                        VkPipelineLayoutCreateInfo const *ci,
                        VkAllocationCallbacks const      *alloc,
                        VkPipelineLayout                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreatePipelineLayout(device, ci, alloc, out);
	lock();
	put("pl=%s sets=[", created(r, KIND_PIPE_LAYOUT, *out).text);
	for (uint32_t i = 0; i < ci->setLayoutCount; ++i)
		put("%s%s", i ? "," : "", name_of(KIND_SET_LAYOUT, ci->pSetLayouts[i]).text);
	put("] push=[");
	for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i)
		put("%s%s:%" PRIu32 "+%" PRIu32, i ? "," : "", hex(ci->pPushConstantRanges[i].stageFlags).text,
		    ci->pPushConstantRanges[i].offset, ci->pPushConstantRanges[i].size);
	emitf("CreatePipelineLayout", "] flags=%s rc=%d", hex(ci->flags).text, r);
	unlock();
	return r;
}

/** @brief vkDestroyPipelineLayout. */
static VKAPI_ATTR void VKAPI_CALL
destroy_pipeline_layout (VkDevice                     device,
                         VkPipelineLayout             layout,
                         VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (layout)
		emitf("DestroyPipelineLayout", "pl=%s", forget(KIND_PIPE_LAYOUT, layout).text);
	unlock();
	d->vkDestroyPipelineLayout(device, layout, alloc);
}

/** @brief Appends a shader stage: its module, entry, flags, required subgroup size, other chained
 *         structures and specialization. Caller holds g_lock.
 *
 * @param s The stage.
 */
static void
put_stage (VkPipelineShaderStageCreateInfo const *s)
{
	put("stage=%s", hex(s->stage).text);
	VkShaderModuleCreateInfo const *const inl = s->module ? nullptr
	                                          : find_in_chain(s->pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
	if (s->module) {
		put(" module=%s", name_of(KIND_SHADER, s->module).text);
	} else if (inl) {
		put(" module=inline ");
		put_spirv(inl->pCode, inl->codeSize);
	} else {
		put(" module=null");
	}
	put(" entry=");
	put_token(s->pName);
	put(" stageflags=%s", hex(s->flags).text);
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo const *const sub =
		find_in_chain(s->pNext, VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
	if (sub)
		put(" subgroup=%" PRIu32, sub->requiredSubgroupSize);
	size_t const mark = g_line.length;
	put(" stagechain=[");
	bool other = false;
	for (VkBaseInStructure const *p = s->pNext; p; p = p->pNext) {
		if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
		    || p->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
			continue;
		put("%s%s", other ? "," : "", stype_word(p->sType).text);
		other = true;
	}
	if (other)
		put("]");
	else
		text_truncate(&g_line, mark);
	VkSpecializationInfo const *const sp = s->pSpecializationInfo;
	if (sp) {
		put(" spec=[");
		for (uint32_t i = 0; i < sp->mapEntryCount; ++i)
			put("%s%" PRIu32 ":%" PRIu32 "+%zu", i ? "," : "", sp->pMapEntries[i].constantID,
			    sp->pMapEntries[i].offset, sp->pMapEntries[i].size);
		put("] specdata=");
		put_bytes_hex(sp->pData, sp->dataSize);
	} else {
		put(" spec=none");
	}
}

/** @brief vkCreateComputePipelines. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_compute_pipelines (VkDevice                           device,
                          VkPipelineCache                    cache,
                          uint32_t                           n,
                          VkComputePipelineCreateInfo const *ci,
                          VkAllocationCallbacks const       *alloc,
                          VkPipeline                        *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateComputePipelines(device, cache, n, ci, alloc, out);
	lock();
	for (uint32_t i = 0; i < n; ++i) {
		put("pipe=%s cache=%s layout=%s flags=%s ", fresh(KIND_PIPELINE, out[i]).text, name_of(KIND_CACHE, cache).text,
		    name_of(KIND_PIPE_LAYOUT, ci[i].layout).text, hex(ci[i].flags).text);
		put_stage(&ci[i].stage);
		put(" chain=");
		put_chain(ci[i].pNext);
		emitf("CreateComputePipeline", " rc=%d", r);
	}
	unlock();
	return r;
}

/** @brief vkCreateGraphicsPipelines. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_graphics_pipelines (VkDevice                            device,
                           VkPipelineCache                     cache,
                           uint32_t                            n,
                           VkGraphicsPipelineCreateInfo const *ci,
                           VkAllocationCallbacks const        *alloc,
                           VkPipeline                         *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateGraphicsPipelines(device, cache, n, ci, alloc, out);
	lock();
	for (uint32_t i = 0; i < n; ++i) {
		put("pipe=%s cache=%s layout=%s stages={", fresh(KIND_PIPELINE, out[i]).text,
		    name_of(KIND_CACHE, cache).text, name_of(KIND_PIPE_LAYOUT, ci[i].layout).text);
		for (uint32_t k = 0; k < ci[i].stageCount; ++k) {
			if (k)
				put("|");
			// A stage's fields are joined with '/' here: the whole is one field.
			size_t const mark = g_line.length;
			put_stage(&ci[i].pStages[k]);
			for (size_t c = mark; c < g_line.length; ++c)
				if (g_line.bytes[c] == ' ')
					g_line.bytes[c] = '/';
		}
		emitf("CreateGraphicsPipeline", "} rc=%d", r);
	}
	unlock();
	return r;
}

/** @brief vkDestroyPipeline. */
static VKAPI_ATTR void VKAPI_CALL
destroy_pipeline (VkDevice                     device,
                  VkPipeline                   pipeline,
                  VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (pipeline)
		emitf("DestroyPipeline", "pipe=%s", forget(KIND_PIPELINE, pipeline).text);
	unlock();
	d->vkDestroyPipeline(device, pipeline, alloc);
}

// ---- descriptors --------------------------------------------------------------------------------

/** @brief vkCreateDescriptorPool. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_descriptor_pool (VkDevice                          device,
                        VkDescriptorPoolCreateInfo const *ci,
                        VkAllocationCallbacks const      *alloc,
                        VkDescriptorPool                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateDescriptorPool(device, ci, alloc, out);
	lock();
	put("dp=%s flags=%s maxsets=%" PRIu32 " sizes=[", created(r, KIND_DESC_POOL, *out).text, hex(ci->flags).text,
	    ci->maxSets);
	for (uint32_t i = 0; i < ci->poolSizeCount; ++i)
		put("%s%s*%" PRIu32, i ? "," : "", dtype_word(ci->pPoolSizes[i].type).text,
		    ci->pPoolSizes[i].descriptorCount);
	emitf("CreateDescriptorPool", "] rc=%d", r);
	unlock();
	return r;
}

/** @brief Frees a list of keys and empties it.
 *
 * @param list The list.
 */
static void
key_list_fini (struct key_list *list)
{
	free(list->items);
	list->items = nullptr;
	*list = (struct key_list){};
}

/** @brief Forgets a descriptor set. Caller holds g_lock.
 *
 * @param key The set's key.
 */
static void
drop_set (uint64_t key)
{
	u32_map_remove(&g_names[KIND_DESC_SET], key);
	struct set_info *const set = set_map_find(&g_set, key);
	if (set) {
		set_info_fini(set);
		set_map_remove(&g_set, key);
	}
}

/** @brief Forgets the sets of a descriptor pool. Caller holds g_lock.
 *
 * @param pool The pool.
 */
static void
drop_pool_sets (VkDescriptorPool pool)
{
	struct key_list *const sets = list_map_find(&g_pool_sets, key_of(pool));
	if (!sets)
		return;
	for (size_t i = 0; i < sets->count; ++i)
		drop_set(sets->items[i]);
	key_list_fini(sets);
	list_map_remove(&g_pool_sets, key_of(pool));
}

/** @brief vkDestroyDescriptorPool. */
static VKAPI_ATTR void VKAPI_CALL
destroy_descriptor_pool (VkDevice                     device,
                         VkDescriptorPool             pool,
                         VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (pool) {
		emitf("DestroyDescriptorPool", "dp=%s", forget(KIND_DESC_POOL, pool).text);
		drop_pool_sets(pool);
	}
	unlock();
	d->vkDestroyDescriptorPool(device, pool, alloc);
}

/** @brief vkResetDescriptorPool. */
static VKAPI_ATTR VkResult VKAPI_CALL
reset_descriptor_pool (VkDevice                   device,
                       VkDescriptorPool           pool,
                       VkDescriptorPoolResetFlags flags)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkResetDescriptorPool(device, pool, flags);
	lock();
	emitf("ResetDescriptorPool", "dp=%s rc=%d", name_of(KIND_DESC_POOL, pool).text, r);
	drop_pool_sets(pool);
	unlock();
	return r;
}

/** @brief A set's entry, made empty if it is new. Caller holds g_lock.
 *
 * @param key The set's key.
 * @return    The entry, which holds until the next set is added or removed.
 */
static struct set_info *
set_entry (uint64_t key)
{
	bool added;
	struct set_info *const set = set_map_slot(&g_set, key, &added);
	if (added)
		*set = (struct set_info){};
	return set;
}

/** @brief A list's entry in a map of lists, made empty if it is new. Caller holds g_lock.
 *
 * @param map The map.
 * @param key The list's key.
 * @return    The list, which holds until the next list is added to or removed from the map.
 */
static struct key_list *
list_entry (struct list_map *map,
            uint64_t         key)
{
	bool added;
	struct key_list *const list = list_map_slot(map, key, &added);
	if (added)
		*list = (struct key_list){};
	return list;
}

/** @brief vkAllocateDescriptorSets: gives each set its layout's counts. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_descriptor_sets (VkDevice                           device,
                          VkDescriptorSetAllocateInfo const *ai,
                          VkDescriptorSet                   *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkAllocateDescriptorSets(device, ai, out);
	lock();
	VkDescriptorSetVariableDescriptorCountAllocateInfo const *const variable =
		find_in_chain(ai->pNext, VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);
	put("dp=%s sets=[", name_of(KIND_DESC_POOL, ai->descriptorPool).text);
	for (uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
		bool const ok = r == VK_SUCCESS && out[i];
		if (ok) {
			struct set_info *const set = set_entry(key_of(out[i]));
			set_info_fini(set);
			struct layout_info const *const l = layout_map_find(&g_layout, key_of(ai->pSetLayouts[i]));
			if (l && l->counts.count) {
				size_t const n = l->counts.count;
				set->counts = (struct u32_map){
					.keys     = reallocate(nullptr, n * sizeof *set->counts.keys),
					.values   = reallocate(nullptr, n * sizeof *set->counts.values),
					.count    = n,
					.capacity = n,
				};
				memcpy(set->counts.keys, l->counts.keys, n * sizeof *set->counts.keys);
				memcpy(set->counts.values, l->counts.values, n * sizeof *set->counts.values);
				// The variable-count binding is the last one; without a count given, it has none.
				if (l->variable)
					set->counts.values[n - 1] = variable && i < variable->descriptorSetCount
					                            ? variable->pDescriptorCounts[i] : 0;
			}
			*key_list_push(list_entry(&g_pool_sets, key_of(ai->descriptorPool))) = key_of(out[i]);
		}
		put("%s%s:%s", i ? "," : "", (ok ? fresh(KIND_DESC_SET, out[i]) : (struct name){"null"}).text,
		    name_of(KIND_SET_LAYOUT, ai->pSetLayouts[i]).text);
	}
	put("] chain=");
	put_chain(ai->pNext);
	emitf("AllocateDescriptorSets", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkFreeDescriptorSets. */
static VKAPI_ATTR VkResult VKAPI_CALL
free_descriptor_sets (VkDevice               device,
                      VkDescriptorPool       pool,
                      uint32_t               n,
                      VkDescriptorSet const *sets)
{
	struct device_data *const d = dev(device);
	lock();
	struct key_list *const owned = list_map_find(&g_pool_sets, key_of(pool));
	put("dp=%s sets=[", name_of(KIND_DESC_POOL, pool).text);
	for (uint32_t i = 0; i < n; ++i) {
		put("%s%s", i ? "," : "", forget(KIND_DESC_SET, sets[i]).text);
		drop_set(key_of(sets[i]));
		if (owned)
			key_list_remove(owned, key_of(sets[i]));
	}
	emitf("FreeDescriptorSets", "]");
	unlock();
	return d->vkFreeDescriptorSets(device, pool, n, sets);
}

/** @brief Whether a descriptor type holds a buffer.
 *
 * @param t The type.
 * @return  true for (dynamic) storage and uniform buffers.
 */
static bool
is_buffer_type (VkDescriptorType t)
{
	return t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
	       || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
}

/** @brief Whether a descriptor type holds an image or a sampler.
 *
 * @param t The type.
 * @return  true for storage, sampled and combined images, input attachments and samplers.
 */
static bool
is_image_type (VkDescriptorType t)
{
	return t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
	       || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
	       || t == VK_DESCRIPTOR_TYPE_SAMPLER;
}

/** @brief Moves a binding and an element past the end of a binding of a set to the next binding that
 *         has descriptors, from its first element, as consecutive binding updates do. Without the
 *         set's layout, or past its last binding, they stay.
 *
 * @param set     The set.
 * @param binding The binding.
 * @param element The element.
 */
static void
consecutive (struct set_info const *set,
             uint32_t              *binding,
             uint32_t              *element)
{
	struct u32_map const *const counts = &set->counts;
	bool found;
	size_t at = support_keys_find(counts->keys, counts->count, *binding, &found);
	if (!found)
		return;
	uint32_t b = *binding;
	uint32_t e = *element;
	while (at < counts->count && e >= counts->values[at]) {
		e -= counts->values[at];
		do
			++at;
		while (at < counts->count && !counts->values[at]);
		if (at == counts->count)
			return;
		b = (uint32_t)counts->keys[at];
	}
	*binding = b;
	*element = e;
}

/** @brief A descriptor's slot key.
 *
 * @param binding Its binding.
 * @param element Its element.
 * @return        binding << 32 | element.
 */
static uint64_t
slot_key (uint32_t binding,
          uint32_t element)
{
	return (uint64_t)binding << 32 | element;
}

/** @brief vkUpdateDescriptorSets: logs every descriptor written or copied under the binding it
 *         reaches.
 */
static VKAPI_ATTR void VKAPI_CALL
update_descriptor_sets (VkDevice                    device,
                        uint32_t                    nw,
                        VkWriteDescriptorSet const *w,
                        uint32_t                    nc,
                        VkCopyDescriptorSet const  *c)
{
	struct device_data *const d = dev(device);
	lock();
	emitf("UpdateDescriptorSets", "writes=%" PRIu32 " copies=%" PRIu32, nw, nc);
	for (uint32_t i = 0; i < nw; ++i) {
		VkWriteDescriptorSet const *const wr = &w[i];
		struct set_info *const set = set_entry(key_of(wr->dstSet));
		uint32_t binding = wr->dstBinding;
		uint32_t element = wr->dstArrayElement;
		for (uint32_t k = 0; k < wr->descriptorCount; ++k, ++element) {
			consecutive(set, &binding, &element);
			struct desc desc = {.type = wr->descriptorType};
			put("set=%s binding=%" PRIu32 " elem=%" PRIu32 " type=%s ", name_of(KIND_DESC_SET, wr->dstSet).text,
			    binding, element, dtype_word(wr->descriptorType).text);
			if (is_buffer_type(wr->descriptorType)) {
				VkDescriptorBufferInfo const *const b = &wr->pBufferInfo[k];
				desc.buffer = b->buffer;
				desc.offset = b->offset;
				desc.range = b->range;
				put("buf=%s off=%" PRIu64 " range=%s", name_of(KIND_BUFFER, b->buffer).text, b->offset,
				    size_text(b->range).text);
			} else if (is_image_type(wr->descriptorType)) {
				VkDescriptorImageInfo const *const im = &wr->pImageInfo[k];
				desc.view = im->imageView;
				put("view=%s layout=%s smp=%s", name_of(KIND_VIEW, im->imageView).text,
				    layout_word(im->imageLayout).text, name_of(KIND_SAMPLER, im->sampler).text);
			} else if (wr->pTexelBufferView) {
				put("bview=%s", name_of(KIND_BUFFER_VIEW, wr->pTexelBufferView[k]).text);
			} else {
				put("chain=");
				put_chain(wr->pNext);
			}
			bool added;
			*slot_map_slot(&set->slots, slot_key(binding, element), &added) = desc;
			emit("DescriptorWrite");
		}
	}
	// One line per descriptor copied, with the bindings it came from and went to.
	for (uint32_t i = 0; i < nc; ++i) {
		VkCopyDescriptorSet const *const cp = &c[i];
		set_entry(key_of(cp->srcSet));
		set_entry(key_of(cp->dstSet));
		// Both are there now, and stay while the copy runs.
		struct set_info const *const src = set_map_find(&g_set, key_of(cp->srcSet));
		struct set_info *const dst = set_map_find(&g_set, key_of(cp->dstSet));
		uint32_t sb = cp->srcBinding;
		uint32_t se = cp->srcArrayElement;
		uint32_t db = cp->dstBinding;
		uint32_t de = cp->dstArrayElement;
		for (uint32_t k = 0; k < cp->descriptorCount; ++k, ++se, ++de) {
			consecutive(src, &sb, &se);
			consecutive(dst, &db, &de);
			struct desc const *const from = slot_map_find(&src->slots, slot_key(sb, se));
			if (from) {
				// The same set's slots may move when the copy lands in them.
				struct desc const copy = *from;
				bool added;
				*slot_map_slot(&dst->slots, slot_key(db, de), &added) = copy;
			}
			emitf("DescriptorCopy", "src=%s:%" PRIu32 ":%" PRIu32 " dst=%s:%" PRIu32 ":%" PRIu32,
			      name_of(KIND_DESC_SET, cp->srcSet).text, sb, se, name_of(KIND_DESC_SET, cp->dstSet).text, db, de);
		}
	}
	unlock();
	d->vkUpdateDescriptorSets(device, nw, w, nc, c);
}

// ---- command pools and buffers ------------------------------------------------------------------

/** @brief vkCreateCommandPool. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_command_pool (VkDevice                       device,
                     VkCommandPoolCreateInfo const *ci,
                     VkAllocationCallbacks const   *alloc,
                     VkCommandPool                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateCommandPool(device, ci, alloc, out);
	lock();
	emitf("CreateCommandPool", "cp=%s family=%" PRIu32 " flags=%s rc=%d", created(r, KIND_CMD_POOL, *out).text,
	      ci->queueFamilyIndex, hex(ci->flags).text, r);
	unlock();
	return r;
}

/** @brief Empties what a command buffer recorded, keeping its device and its arrays' memory.
 *
 * @param s The command buffer's state.
 */
static void
reset_cb (struct cb_state *s)
{
	s->compute = VK_NULL_HANDLE;
	s->sets.count = 0;
	s->layouts.count = 0;
	s->owners.count = 0;
	s->uploads.count = 0;
	s->downloads.count = 0;
	s->copy_dst.count = 0;
	s->i2b_dst.count = 0;
	s->dispatch_targets.count = 0;
	s->dispatch_ends.count = 0;
	s->index = 0;
	s->dispatches = 0;
}

/** @brief Frees what a command buffer's state holds and empties it.
 *
 * @param s The state.
 */
static void
cb_state_fini (struct cb_state *s)
{
	free(s->sets.items);
	s->sets.items = nullptr;
	free(s->layouts.items);
	s->layouts.items = nullptr;
	free(s->owners.items);
	s->owners.items = nullptr;
	free(s->uploads.items);
	s->uploads.items = nullptr;
	free(s->downloads.items);
	s->downloads.items = nullptr;
	free(s->copy_dst.items);
	s->copy_dst.items = nullptr;
	free(s->i2b_dst.items);
	s->i2b_dst.items = nullptr;
	free(s->dispatch_targets.items);
	s->dispatch_targets.items = nullptr;
	free(s->dispatch_ends.items);
	s->dispatch_ends.items = nullptr;
	*s = (struct cb_state){};
}

/** @brief Forgets a command buffer. Caller holds g_lock.
 *
 * @param key The command buffer's key.
 */
static void
drop_cb (uint64_t key)
{
	u32_map_remove(&g_names[KIND_CMD], key);
	struct cb_state *const s = cb_map_find(&g_cb, key);
	if (s) {
		cb_state_fini(s);
		cb_map_remove(&g_cb, key);
	}
}

/** @brief vkDestroyCommandPool: forgets its command buffers. */
static VKAPI_ATTR void VKAPI_CALL
destroy_command_pool (VkDevice                     device,
                      VkCommandPool                pool,
                      VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (pool) {
		emitf("DestroyCommandPool", "cp=%s", forget(KIND_CMD_POOL, pool).text);
		struct key_list *const cbs = list_map_find(&g_cmdpool_cbs, key_of(pool));
		if (cbs) {
			for (size_t i = 0; i < cbs->count; ++i)
				drop_cb(cbs->items[i]);
			key_list_fini(cbs);
			list_map_remove(&g_cmdpool_cbs, key_of(pool));
		}
	}
	unlock();
	d->vkDestroyCommandPool(device, pool, alloc);
}

/** @brief vkResetCommandPool. */
static VKAPI_ATTR VkResult VKAPI_CALL
reset_command_pool (VkDevice                device,
                    VkCommandPool           pool,
                    VkCommandPoolResetFlags flags)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkResetCommandPool(device, pool, flags);
	lock();
	emitf("ResetCommandPool", "cp=%s flags=%s rc=%d", name_of(KIND_CMD_POOL, pool).text, hex(flags).text, r);
	unlock();
	return r;
}

/** @brief vkAllocateCommandBuffers. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_command_buffers (VkDevice                           device,
                          VkCommandBufferAllocateInfo const *ai,
                          VkCommandBuffer                   *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkAllocateCommandBuffers(device, ai, out);
	lock();
	put("cp=%s level=%s cbs=[", name_of(KIND_CMD_POOL, ai->commandPool).text,
	    enum_word(vkt_command_buffer_level_name(ai->level), ai->level).text);
	for (uint32_t i = 0; r == VK_SUCCESS && i < ai->commandBufferCount; ++i) {
		put("%s%s", i ? "," : "", fresh(KIND_CMD, out[i]).text);
		bool added;
		struct cb_state *const s = cb_map_slot(&g_cb, key_of(out[i]), &added);
		if (!added)
			cb_state_fini(s);
		*s = (struct cb_state){.device = d};
		*key_list_push(list_entry(&g_cmdpool_cbs, key_of(ai->commandPool))) = key_of(out[i]);
	}
	emitf("AllocateCommandBuffers", "] rc=%d", r);
	unlock();
	return r;
}

/** @brief vkFreeCommandBuffers. */
static VKAPI_ATTR void VKAPI_CALL
free_command_buffers (VkDevice               device,
                      VkCommandPool          pool,
                      uint32_t               n,
                      VkCommandBuffer const *cbs)
{
	struct device_data *const d = dev(device);
	lock();
	struct key_list *const owned = list_map_find(&g_cmdpool_cbs, key_of(pool));
	put("cp=%s cbs=[", name_of(KIND_CMD_POOL, pool).text);
	bool listed = false;
	for (uint32_t i = 0; i < n; ++i) {
		if (!cbs[i])
			continue;
		put("%s%s", listed ? "," : "", name_of(KIND_CMD, cbs[i]).text);
		listed = true;
		drop_cb(key_of(cbs[i]));
		if (owned)
			key_list_remove(owned, key_of(cbs[i]));
	}
	emitf("FreeCommandBuffers", "]");
	unlock();
	d->vkFreeCommandBuffers(device, pool, n, cbs);
}

/** @brief A command buffer's recording state, made empty for one the trace does not know. Caller
 *         holds g_lock.
 *
 * @param cb The command buffer.
 * @return   Its state, which holds until the next command buffer is added or removed.
 */
static struct cb_state *
cb_state (VkCommandBuffer cb)
{
	bool added;
	struct cb_state *const s = cb_map_slot(&g_cb, key_of(cb), &added);
	if (added)
		*s = (struct cb_state){};
	return s;
}

/** @brief vkBeginCommandBuffer. */
static VKAPI_ATTR VkResult VKAPI_CALL
begin_command_buffer (VkCommandBuffer                 cb,
                      VkCommandBufferBeginInfo const *bi)
{
	struct device_data *const d = dev(cb);
	lock();
	reset_cb(cb_state(cb));
	emitf("BeginCommandBuffer", "cb=%s flags=%s", name_of(KIND_CMD, cb).text, hex(bi->flags).text);
	unlock();
	return d->vkBeginCommandBuffer(cb, bi);
}

/** @brief vkEndCommandBuffer. */
static VKAPI_ATTR VkResult VKAPI_CALL
end_command_buffer (VkCommandBuffer cb)
{
	struct device_data *const d = dev(cb);
	VkResult const r = d->vkEndCommandBuffer(cb);
	lock();
	struct cb_state const *const s = cb_state(cb);
	emitf("EndCommandBuffer", "cb=%s cmds=%" PRIu32 " dispatches=%" PRIu32 " rc=%d", name_of(KIND_CMD, cb).text,
	      s->index, s->dispatches, r);
	unlock();
	return r;
}

/** @brief vkResetCommandBuffer. */
static VKAPI_ATTR VkResult VKAPI_CALL
reset_command_buffer (VkCommandBuffer           cb,
                      VkCommandBufferResetFlags flags)
{
	struct device_data *const d = dev(cb);
	lock();
	reset_cb(cb_state(cb));
	emitf("ResetCommandBuffer", "cb=%s flags=%s", name_of(KIND_CMD, cb).text, hex(flags).text);
	unlock();
	return d->vkResetCommandBuffer(cb, flags);
}

/** @brief Starts a command's line, "cb=cbN i=K ", and counts the command. Caller holds g_lock.
 *
 * @param cb The command buffer.
 * @param at Receives the command's index.
 * @return   The command buffer's state, which holds until the next command buffer is added or
 *           removed.
 */
static struct cb_state *
cmd (VkCommandBuffer  cb,
     uint32_t        *at)
{
	struct cb_state *const s = cb_state(cb);
	*at = s->index++;
	put("cb=%s i=%" PRIu32 " ", name_of(KIND_CMD, cb).text, *at);
	return s;
}

/** @brief Labels what a command named: "cbN#AT", or "cbN#AT.I" for its region I. Caller holds
 *         g_lock.
 *
 * @param dest   Where the label goes.
 * @param size   Its bytes, at least LABEL_BYTES - 1.
 * @param cb     The command buffer.
 * @param at     The command's index.
 * @param region The region, or -1 for none.
 */
static void
label_command (char            *dest,
               size_t           size,
               VkCommandBuffer  cb,
               uint32_t         at,
               int64_t          region)
{
	struct name const n = name_of(KIND_CMD, cb);
	if (region < 0)
		snprintf(dest, size, "%s#%" PRIu32, n.text, at);
	else
		snprintf(dest, size, "%s#%" PRIu32 ".%" PRId64, n.text, at, region);
}

/** @brief vkCmdBindPipeline. */
static VKAPI_ATTR void VKAPI_CALL
cmd_bind_pipeline (VkCommandBuffer     cb,
                   VkPipelineBindPoint bp,
                   VkPipeline          pipeline)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	if (bp == VK_PIPELINE_BIND_POINT_COMPUTE)
		s->compute = pipeline;
	emitf("CmdBindPipeline", "bind=%s pipe=%s", enum_word(vkt_pipeline_bind_point_name(bp), bp).text,
	      name_of(KIND_PIPELINE, pipeline).text);
	unlock();
	d->vkCmdBindPipeline(cb, bp, pipeline);
}

/** @brief vkCmdBindDescriptorSets: remembers the sets bound for compute. */
static VKAPI_ATTR void VKAPI_CALL
cmd_bind_descriptor_sets (VkCommandBuffer        cb,
                          VkPipelineBindPoint    bp,
                          VkPipelineLayout       layout,
                          uint32_t               first,
                          uint32_t               n,
                          VkDescriptorSet const *sets,
                          uint32_t               ndyn,
                          uint32_t const        *dyn)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("bind=%s layout=%s first=%" PRIu32 " sets=[", enum_word(vkt_pipeline_bind_point_name(bp), bp).text,
	    name_of(KIND_PIPE_LAYOUT, layout).text, first);
	for (uint32_t i = 0; i < n; ++i) {
		put("%s%s", i ? "," : "", name_of(KIND_DESC_SET, sets[i]).text);
		if (bp != VK_PIPELINE_BIND_POINT_COMPUTE)
			continue;
		size_t const number = (size_t)first + i;
		while (s->sets.count <= number)
			*key_list_push(&s->sets) = 0;
		s->sets.items[number] = key_of(sets[i]);
	}
	put("] dyn=[");
	for (uint32_t i = 0; i < ndyn; ++i)
		put("%s%" PRIu32, i ? "," : "", dyn[i]);
	emitf("CmdBindDescriptorSets", "]");
	unlock();
	d->vkCmdBindDescriptorSets(cb, bp, layout, first, n, sets, ndyn, dyn);
}

/** @brief vkCmdPushConstants: logs the bytes. */
static VKAPI_ATTR void VKAPI_CALL
cmd_push_constants (VkCommandBuffer     cb,
                    VkPipelineLayout    layout,
                    VkShaderStageFlags  stages,
                    uint32_t            offset,
                    uint32_t            size,
                    void const         *data)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	put("layout=%s stages=%s offset=%" PRIu32 " size=%" PRIu32 " data=", name_of(KIND_PIPE_LAYOUT, layout).text,
	    hex(stages).text, offset, size);
	put_bytes_hex(data, size);
	emit("CmdPushConstants");
	unlock();
	d->vkCmdPushConstants(cb, layout, stages, offset, size, data);
}

/** @brief Counts a dispatch and records the storage resources the bound compute sets hold, for the
 *         selectors that need them. Caller holds g_lock.
 *
 * @param s The command buffer's state.
 */
static void
note_dispatch (struct cb_state *s)
{
	++s->dispatches;
	if (!g_cfg.hash_need_dispatch)
		return;
	for (size_t number = 0; number < s->sets.count; ++number) {
		struct set_info const *const set = set_map_find(&g_set, s->sets.items[number]);
		if (!set)
			continue;
		for (size_t i = 0; i < set->slots.count; ++i) {
			struct desc const *const desc = &set->slots.values[i];
			struct target t = {.size = VK_WHOLE_SIZE};
			if (desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && desc->buffer) {
				t.handle = key_of(desc->buffer);
				t.offset = desc->offset;
				t.size = desc->range;
			} else if (desc->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && desc->view) {
				uint64_t const *const image = handle_map_find(&g_view, key_of(desc->view));
				if (!image)
					continue;
				t.handle = *image;
				t.image = true;
			} else {
				continue;
			}
			snprintf(t.label, sizeof t.label, "set%zu.b%" PRIu64, number, set->slots.keys[i] >> 32);
			*targets_push(&s->dispatch_targets) = t;
		}
	}
	*ends_push(&s->dispatch_ends) = s->dispatch_targets.count;
}

/** @brief vkCmdDispatch. */
static VKAPI_ATTR void VKAPI_CALL
cmd_dispatch (VkCommandBuffer cb,
              uint32_t        x,
              uint32_t        y,
              uint32_t        z)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	emitf("CmdDispatch", "n=%" PRIu32 " pipe=%s x=%" PRIu32 " y=%" PRIu32 " z=%" PRIu32, s->dispatches,
	      name_of(KIND_PIPELINE, s->compute).text, x, y, z);
	note_dispatch(s);
	unlock();
	d->vkCmdDispatch(cb, x, y, z);
}

/** @brief vkCmdDispatchBase. */
static VKAPI_ATTR void VKAPI_CALL
cmd_dispatch_base (VkCommandBuffer cb,
                   uint32_t        bx,
                   uint32_t        by,
                   uint32_t        bz,
                   uint32_t        x,
                   uint32_t        y,
                   uint32_t        z)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	emitf("CmdDispatchBase", "n=%" PRIu32 " pipe=%s base=%" PRIu32 ",%" PRIu32 ",%" PRIu32 " x=%" PRIu32
	      " y=%" PRIu32 " z=%" PRIu32, s->dispatches, name_of(KIND_PIPELINE, s->compute).text, bx, by, bz, x, y, z);
	note_dispatch(s);
	unlock();
	d->vkCmdDispatchBase(cb, bx, by, bz, x, y, z);
}

/** @brief vkCmdDispatchIndirect. */
static VKAPI_ATTR void VKAPI_CALL
cmd_dispatch_indirect (VkCommandBuffer cb,
                       VkBuffer        buffer,
                       VkDeviceSize    offset)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	emitf("CmdDispatchIndirect", "n=%" PRIu32 " pipe=%s buf=%s offset=%" PRIu64, s->dispatches,
	      name_of(KIND_PIPELINE, s->compute).text, name_of(KIND_BUFFER, buffer).text, offset);
	note_dispatch(s);
	unlock();
	d->vkCmdDispatchIndirect(cb, buffer, offset);
}

/** @brief Appends the start of an image barrier, "imgN:OLD>NEW:", before its access masks. Caller
 *         holds g_lock.
 *
 * @param image The image.
 * @param from  Its old layout.
 * @param to    Its new layout.
 */
static void
put_barrier_head (VkImage       image,
                  VkImageLayout from,
                  VkImageLayout to)
{
	put("%s:%s>%s:", name_of(KIND_IMAGE, image).text, layout_word(from).text, layout_word(to).text);
}

/** @brief Appends the end of an image barrier, after its access masks: ":SRC>DST:" queue families and
 *         the range. Caller holds g_lock.
 *
 * @param sq The source queue family.
 * @param dq The destination queue family.
 * @param r  The range.
 */
static void
put_barrier_tail (uint32_t                       sq,
                  uint32_t                       dq,
                  VkImageSubresourceRange const *r)
{
	put(":%s>%s:", family_text(sq).text, family_text(dq).text);
	put_range(r);
}

/** @brief vkCmdPipelineBarrier: remembers ownership transfers and image layouts. */
static VKAPI_ATTR void VKAPI_CALL
cmd_pipeline_barrier (VkCommandBuffer              cb,
                      VkPipelineStageFlags         src,
                      VkPipelineStageFlags         dst,
                      VkDependencyFlags            dep,
                      uint32_t                     nm,
                      VkMemoryBarrier const       *mb,
                      uint32_t                     nb,
                      VkBufferMemoryBarrier const *bb,
                      uint32_t                     ni,
                      VkImageMemoryBarrier const  *ib)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("src=%s dst=%s dep=%s mem=[", hex(src).text, hex(dst).text, hex(dep).text);
	for (uint32_t i = 0; i < nm; ++i)
		put("%s%s>%s", i ? "," : "", hex(mb[i].srcAccessMask).text, hex(mb[i].dstAccessMask).text);
	put("] buf=[");
	for (uint32_t i = 0; i < nb; ++i) {
		put("%s%s@%" PRIu64 "+%s:%s>%s:%s>%s", i ? "," : "", name_of(KIND_BUFFER, bb[i].buffer).text, bb[i].offset,
		    size_text(bb[i].size).text, hex(bb[i].srcAccessMask).text, hex(bb[i].dstAccessMask).text,
		    family_text(bb[i].srcQueueFamilyIndex).text, family_text(bb[i].dstQueueFamilyIndex).text);
		if (bb[i].srcQueueFamilyIndex != bb[i].dstQueueFamilyIndex)
			*buffer_owners_push(&s->owners) = (struct buffer_owner){bb[i].buffer, bb[i].dstQueueFamilyIndex};
	}
	put("] img=[");
	for (uint32_t i = 0; i < ni; ++i) {
		if (i)
			put(",");
		put_barrier_head(ib[i].image, ib[i].oldLayout, ib[i].newLayout);
		put("%s>%s", hex(ib[i].srcAccessMask).text, hex(ib[i].dstAccessMask).text);
		put_barrier_tail(ib[i].srcQueueFamilyIndex, ib[i].dstQueueFamilyIndex, &ib[i].subresourceRange);
		*image_layouts_push(&s->layouts) = (struct image_layout){ib[i].image, ib[i].newLayout};
	}
	emitf("CmdPipelineBarrier", "]");
	unlock();
	d->vkCmdPipelineBarrier(cb, src, dst, dep, nm, mb, nb, bb, ni, ib);
}

/** @brief Appends a synchronization2 barrier's stages and accesses: "SS:SA>DS:DA". Caller holds
 *         g_lock.
 *
 * @param ss The source stages.
 * @param sa The source accesses.
 * @param ds The destination stages.
 * @param da The destination accesses.
 */
static void
put_access2 (VkPipelineStageFlags2 ss,
             VkAccessFlags2        sa,
             VkPipelineStageFlags2 ds,
             VkAccessFlags2        da)
{
	put("%s:%s>%s:%s", hex(ss).text, hex(sa).text, hex(ds).text, hex(da).text);
}

/** @brief Logs a synchronization2 barrier and remembers its ownership transfers and image layouts.
 *         Caller holds g_lock.
 *
 * @param cb The command buffer.
 * @param di The dependency.
 */
static void
barrier2 (VkCommandBuffer         cb,
          VkDependencyInfo const *di)
{
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("dep=%s mem=[", hex(di->dependencyFlags).text);
	for (uint32_t i = 0; i < di->memoryBarrierCount; ++i) {
		VkMemoryBarrier2 const *const x = &di->pMemoryBarriers[i];
		if (i)
			put(",");
		put_access2(x->srcStageMask, x->srcAccessMask, x->dstStageMask, x->dstAccessMask);
	}
	put("] buf=[");
	for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; ++i) {
		VkBufferMemoryBarrier2 const *const x = &di->pBufferMemoryBarriers[i];
		put("%s%s@%" PRIu64 "+%s:", i ? "," : "", name_of(KIND_BUFFER, x->buffer).text, x->offset,
		    size_text(x->size).text);
		put_access2(x->srcStageMask, x->srcAccessMask, x->dstStageMask, x->dstAccessMask);
		put(":%s>%s", family_text(x->srcQueueFamilyIndex).text, family_text(x->dstQueueFamilyIndex).text);
		if (x->srcQueueFamilyIndex != x->dstQueueFamilyIndex)
			*buffer_owners_push(&s->owners) = (struct buffer_owner){x->buffer, x->dstQueueFamilyIndex};
	}
	put("] img=[");
	for (uint32_t i = 0; i < di->imageMemoryBarrierCount; ++i) {
		VkImageMemoryBarrier2 const *const x = &di->pImageMemoryBarriers[i];
		if (i)
			put(",");
		put_barrier_head(x->image, x->oldLayout, x->newLayout);
		put_access2(x->srcStageMask, x->srcAccessMask, x->dstStageMask, x->dstAccessMask);
		put_barrier_tail(x->srcQueueFamilyIndex, x->dstQueueFamilyIndex, &x->subresourceRange);
		*image_layouts_push(&s->layouts) = (struct image_layout){x->image, x->newLayout};
	}
	emitf("CmdPipelineBarrier2", "]");
}

/** @brief vkCmdPipelineBarrier2. */
static VKAPI_ATTR void VKAPI_CALL
cmd_pipeline_barrier2 (VkCommandBuffer         cb,
                       VkDependencyInfo const *di)
{
	struct device_data *const d = dev(cb);
	lock();
	barrier2(cb, di);
	unlock();
	d->vkCmdPipelineBarrier2(cb, di);
}

/** @brief vkCmdPipelineBarrier2KHR, logged as vkCmdPipelineBarrier2. */
static VKAPI_ATTR void VKAPI_CALL
cmd_pipeline_barrier2_khr (VkCommandBuffer         cb,
                           VkDependencyInfo const *di)
{
	struct device_data *const d = dev(cb);
	lock();
	barrier2(cb, di);
	unlock();
	d->vkCmdPipelineBarrier2KHR(cb, di);
}

/** @brief The FNV-1a 64 of mapped memory: through streaming loads unless the memory is host-cached.
 *
 * @param m    The memory.
 * @param data The bytes.
 * @param n    How many.
 * @return     Their hash.
 */
static uint64_t
host_fnv (struct memory_info const *m,
          void const               *data,
          size_t                    n)
{
	return m->flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? fnv(data, n, FNV_BASIS) : fnv_uncached(data, n);
}

/** @brief The host range of a buffer region, if the buffer's memory is host-visible. Caller holds
 *         g_lock.
 *
 * @param buffer The buffer.
 * @param offset The region's offset in the buffer.
 * @param size   Its bytes.
 * @param out    Receives the range, without a label.
 * @return       false if the buffer is not bound to host-visible memory.
 */
static bool
host_range (VkBuffer           buffer,
            VkDeviceSize       offset,
            VkDeviceSize       size,
            struct host_range *out)
{
	struct buffer_info const *const b = buffer_map_find(&g_buf, key_of(buffer));
	if (!b || !b->memory)
		return false;
	struct memory_info const *const m = memory_map_find(&g_mem, key_of(b->memory));
	if (!m || !(m->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
		return false;
	*out = (struct host_range){.memory = b->memory, .offset = b->offset + offset, .size = size};
	return true;
}

/** @brief The bytes a buffer<->image region spans in the buffer.
 *
 * @param r      The region.
 * @param format The image's format.
 * @return       The bytes, or 0 for a format of no known texel size or an empty region.
 */
static VkDeviceSize
region_bytes (VkBufferImageCopy const *r,
              VkFormat                 format)
{
	VkDeviceSize const texel = vkt_format_bytes(format);
	VkDeviceSize const row = r->bufferRowLength ? r->bufferRowLength : r->imageExtent.width;
	VkDeviceSize const height = r->bufferImageHeight ? r->bufferImageHeight : r->imageExtent.height;
	VkDeviceSize const slices = (VkDeviceSize)r->imageExtent.depth * r->imageSubresource.layerCount;
	if (!texel || !r->imageExtent.width || !r->imageExtent.height || !slices)
		return 0;
	return texel * ((slices - 1) * row * height + (r->imageExtent.height - 1) * row + r->imageExtent.width);
}

/** @brief Appends buffer<->image regions. Caller holds g_lock.
 *
 * @param n The regions.
 * @param r Their array.
 */
static void
put_buffer_image_regions (uint32_t                 n,
                          VkBufferImageCopy const *r)
{
	for (uint32_t i = 0; i < n; ++i) {
		put("%s%" PRIu64 ":%" PRIu32 ":%" PRIu32 ":", i ? "," : "", r[i].bufferOffset, r[i].bufferRowLength,
		    r[i].bufferImageHeight);
		put_layers(&r[i].imageSubresource);
		put(":%" PRId32 ",%" PRId32 ",%" PRId32 ":%" PRIu32 "x%" PRIu32 "x%" PRIu32, r[i].imageOffset.x,
		    r[i].imageOffset.y, r[i].imageOffset.z, r[i].imageExtent.width, r[i].imageExtent.height,
		    r[i].imageExtent.depth);
	}
}

/** @brief An image's format. Caller holds g_lock.
 *
 * @param image The image.
 * @return      Its format, or VK_FORMAT_UNDEFINED for an image the trace does not know.
 */
static VkFormat
image_format (VkImage image)
{
	struct image_info const *const im = image_map_find(&g_img, key_of(image));
	return im ? im->format : VK_FORMAT_UNDEFINED;
}

/** @brief Records a destination for the selectors. Caller holds g_lock.
 *
 * @param list   The list.
 * @param image  An image, which is copied whole.
 * @param handle The buffer or image.
 * @param offset A buffer region's offset.
 * @param size   Its size, or VK_WHOLE_SIZE.
 * @param cb     The command buffer.
 * @param at     The command's index.
 * @param region The region, or -1 for none.
 */
static void
add_target (struct targets  *list,
            bool             image,
            void const      *handle,
            VkDeviceSize     offset,
            VkDeviceSize     size,
            VkCommandBuffer  cb,
            uint32_t         at,
            int64_t          region)
{
	struct target *const t = targets_push(list);
	*t = (struct target){.handle = key_of(handle), .offset = offset, .size = size, .image = image};
	label_command(t->label, sizeof t->label, cb, at, region);
}

/** @brief vkCmdCopyBuffer: notes host ranges and destinations. */
static VKAPI_ATTR void VKAPI_CALL
cmd_copy_buffer (VkCommandBuffer     cb,
                 VkBuffer            src,
                 VkBuffer            dst,
                 uint32_t            n,
                 VkBufferCopy const *r)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("src=%s dst=%s regions=[", name_of(KIND_BUFFER, src).text, name_of(KIND_BUFFER, dst).text);
	for (uint32_t i = 0; i < n; ++i)
		put("%s%" PRIu64 ">%" PRIu64 "+%" PRIu64, i ? "," : "", r[i].srcOffset, r[i].dstOffset, r[i].size);
	emitf("CmdCopyBuffer", "]");
	for (uint32_t i = 0; i < n; ++i) {
		struct host_range h;
		if (host_range(src, r[i].srcOffset, r[i].size, &h)) {
			label_command(h.label, sizeof h.label, cb, at, i);
			*host_ranges_push(&s->uploads) = h;
		}
		if (host_range(dst, r[i].dstOffset, r[i].size, &h)) {
			label_command(h.label, sizeof h.label, cb, at, i);
			*host_ranges_push(&s->downloads) = h;
		}
		add_target(&s->copy_dst, false, dst, r[i].dstOffset, r[i].size, cb, at, i);
	}
	unlock();
	d->vkCmdCopyBuffer(cb, src, dst, n, r);
}

/** @brief vkCmdCopyImage. */
static VKAPI_ATTR void VKAPI_CALL
cmd_copy_image (VkCommandBuffer    cb,
                VkImage            src,
                VkImageLayout      sl,
                VkImage            dst,
                VkImageLayout      dl,
                uint32_t           n,
                VkImageCopy const *r)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("src=%s:%s dst=%s:%s regions=[", name_of(KIND_IMAGE, src).text, layout_word(sl).text,
	    name_of(KIND_IMAGE, dst).text, layout_word(dl).text);
	for (uint32_t i = 0; i < n; ++i) {
		if (i)
			put(",");
		put_layers(&r[i].srcSubresource);
		put(":%" PRId32 ",%" PRId32 ",%" PRId32 ">", r[i].srcOffset.x, r[i].srcOffset.y, r[i].srcOffset.z);
		put_layers(&r[i].dstSubresource);
		put(":%" PRId32 ",%" PRId32 ",%" PRId32 ":%" PRIu32 "x%" PRIu32 "x%" PRIu32, r[i].dstOffset.x,
		    r[i].dstOffset.y, r[i].dstOffset.z, r[i].extent.width, r[i].extent.height, r[i].extent.depth);
	}
	emitf("CmdCopyImage", "]");
	add_target(&s->copy_dst, true, dst, 0, VK_WHOLE_SIZE, cb, at, -1);
	unlock();
	d->vkCmdCopyImage(cb, src, sl, dst, dl, n, r);
}

/** @brief vkCmdBlitImage. */
static VKAPI_ATTR void VKAPI_CALL
cmd_blit_image (VkCommandBuffer    cb,
                VkImage            src,
                VkImageLayout      sl,
                VkImage            dst,
                VkImageLayout      dl,
                uint32_t           n,
                VkImageBlit const *r,
                VkFilter           filter)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("src=%s:%s dst=%s:%s filter=%s regions=[", name_of(KIND_IMAGE, src).text, layout_word(sl).text,
	    name_of(KIND_IMAGE, dst).text, layout_word(dl).text, enum_word(vkt_filter_name(filter), filter).text);
	for (uint32_t i = 0; i < n; ++i) {
		if (i)
			put(",");
		put_layers(&r[i].srcSubresource);
		put(":%" PRId32 ",%" PRId32 ",%" PRId32 "-%" PRId32 ",%" PRId32 ",%" PRId32 ">", r[i].srcOffsets[0].x,
		    r[i].srcOffsets[0].y, r[i].srcOffsets[0].z, r[i].srcOffsets[1].x, r[i].srcOffsets[1].y,
		    r[i].srcOffsets[1].z);
		put_layers(&r[i].dstSubresource);
		put(":%" PRId32 ",%" PRId32 ",%" PRId32 "-%" PRId32 ",%" PRId32 ",%" PRId32, r[i].dstOffsets[0].x,
		    r[i].dstOffsets[0].y, r[i].dstOffsets[0].z, r[i].dstOffsets[1].x, r[i].dstOffsets[1].y,
		    r[i].dstOffsets[1].z);
	}
	emitf("CmdBlitImage", "]");
	add_target(&s->copy_dst, true, dst, 0, VK_WHOLE_SIZE, cb, at, -1);
	unlock();
	d->vkCmdBlitImage(cb, src, sl, dst, dl, n, r, filter);
}

/** @brief vkCmdCopyBufferToImage: notes uploads and the destination. */
static VKAPI_ATTR void VKAPI_CALL
cmd_copy_buffer_to_image (VkCommandBuffer          cb,
                          VkBuffer                 src,
                          VkImage                  dst,
                          VkImageLayout            dl,
                          uint32_t                 n,
                          VkBufferImageCopy const *r)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("src=%s dst=%s:%s regions=[", name_of(KIND_BUFFER, src).text, name_of(KIND_IMAGE, dst).text,
	    layout_word(dl).text);
	put_buffer_image_regions(n, r);
	emitf("CmdCopyBufferToImage", "]");
	VkFormat const f = image_format(dst);
	for (uint32_t i = 0; i < n; ++i) {
		struct host_range h;
		if (host_range(src, r[i].bufferOffset, region_bytes(&r[i], f), &h)) {
			label_command(h.label, sizeof h.label, cb, at, i);
			*host_ranges_push(&s->uploads) = h;
		}
	}
	add_target(&s->copy_dst, true, dst, 0, VK_WHOLE_SIZE, cb, at, -1);
	unlock();
	d->vkCmdCopyBufferToImage(cb, src, dst, dl, n, r);
}

/** @brief vkCmdCopyImageToBuffer: notes readbacks and the ranges written. */
static VKAPI_ATTR void VKAPI_CALL
cmd_copy_image_to_buffer (VkCommandBuffer          cb,
                          VkImage                  src,
                          VkImageLayout            sl,
                          VkBuffer                 dst,
                          uint32_t                 n,
                          VkBufferImageCopy const *r)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("src=%s:%s dst=%s regions=[", name_of(KIND_IMAGE, src).text, layout_word(sl).text,
	    name_of(KIND_BUFFER, dst).text);
	put_buffer_image_regions(n, r);
	emitf("CmdCopyImageToBuffer", "]");
	VkFormat const f = image_format(src);
	for (uint32_t i = 0; i < n; ++i) {
		VkDeviceSize const bytes = region_bytes(&r[i], f);
		struct host_range h;
		if (host_range(dst, r[i].bufferOffset, bytes, &h)) {
			label_command(h.label, sizeof h.label, cb, at, i);
			*host_ranges_push(&s->downloads) = h;
		}
		add_target(&s->i2b_dst, false, dst, r[i].bufferOffset, bytes, cb, at, i);
		add_target(&s->copy_dst, false, dst, r[i].bufferOffset, bytes, cb, at, i);
	}
	unlock();
	d->vkCmdCopyImageToBuffer(cb, src, sl, dst, n, r);
}

/** @brief vkCmdFillBuffer. */
static VKAPI_ATTR void VKAPI_CALL
cmd_fill_buffer (VkCommandBuffer cb,
                 VkBuffer        buffer,
                 VkDeviceSize    offset,
                 VkDeviceSize    size,
                 uint32_t        data)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	emitf("CmdFillBuffer", "buf=%s offset=%" PRIu64 " size=%s data=0x%08" PRIx32, name_of(KIND_BUFFER, buffer).text,
	      offset, size_text(size).text, data);
	add_target(&s->copy_dst, false, buffer, offset, size, cb, at, -1);
	unlock();
	d->vkCmdFillBuffer(cb, buffer, offset, size, data);
}

/** @brief vkCmdUpdateBuffer: logs the bytes' FNV, and the bytes up to 1 KiB. */
static VKAPI_ATTR void VKAPI_CALL
cmd_update_buffer (VkCommandBuffer  cb,
                   VkBuffer         buffer,
                   VkDeviceSize     offset,
                   VkDeviceSize     size,
                   void const      *data)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("buf=%s offset=%" PRIu64 " size=%" PRIu64 " fnv=%016" PRIx64 " data=", name_of(KIND_BUFFER, buffer).text,
	    offset, size, fnv(data, size, FNV_BASIS));
	if (size <= 1024)
		put_bytes_hex(data, size);
	else
		put("long");
	emit("CmdUpdateBuffer");
	add_target(&s->copy_dst, false, buffer, offset, size, cb, at, -1);
	unlock();
	d->vkCmdUpdateBuffer(cb, buffer, offset, size, data);
}

/** @brief vkCmdClearColorImage. */
static VKAPI_ATTR void VKAPI_CALL
cmd_clear_color_image (VkCommandBuffer                cb,
                       VkImage                        image,
                       VkImageLayout                  layout,
                       VkClearColorValue const       *color,
                       uint32_t                       n,
                       VkImageSubresourceRange const *ranges)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	struct cb_state *const s = cmd(cb, &at);
	put("img=%s:%s color=", name_of(KIND_IMAGE, image).text, layout_word(layout).text);
	put_bytes_hex(color, sizeof *color);
	put(" ranges=[");
	for (uint32_t i = 0; i < n; ++i) {
		if (i)
			put(",");
		put_range(&ranges[i]);
	}
	emitf("CmdClearColorImage", "]");
	add_target(&s->copy_dst, true, image, 0, VK_WHOLE_SIZE, cb, at, -1);
	unlock();
	d->vkCmdClearColorImage(cb, image, layout, color, n, ranges);
}

/** @brief vkCmdWriteTimestamp. */
static VKAPI_ATTR void VKAPI_CALL
cmd_write_timestamp (VkCommandBuffer         cb,
                     VkPipelineStageFlagBits stage,
                     VkQueryPool             pool,
                     uint32_t                query)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	emitf("CmdWriteTimestamp", "stage=%s qp=%s query=%" PRIu32, hex(stage).text, name_of(KIND_QUERY_POOL, pool).text,
	      query);
	unlock();
	d->vkCmdWriteTimestamp(cb, stage, pool, query);
}

/** @brief vkCmdWriteTimestamp2. */
static VKAPI_ATTR void VKAPI_CALL
cmd_write_timestamp2 (VkCommandBuffer       cb,
                      VkPipelineStageFlags2 stage,
                      VkQueryPool           pool,
                      uint32_t              query)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	emitf("CmdWriteTimestamp2", "stage=%s qp=%s query=%" PRIu32, hex(stage).text,
	      name_of(KIND_QUERY_POOL, pool).text, query);
	unlock();
	d->vkCmdWriteTimestamp2(cb, stage, pool, query);
}

/** @brief vkCmdResetQueryPool. */
static VKAPI_ATTR void VKAPI_CALL
cmd_reset_query_pool (VkCommandBuffer cb,
                      VkQueryPool     pool,
                      uint32_t        first,
                      uint32_t        count)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	emitf("CmdResetQueryPool", "qp=%s first=%" PRIu32 " count=%" PRIu32, name_of(KIND_QUERY_POOL, pool).text,
	      first, count);
	unlock();
	d->vkCmdResetQueryPool(cb, pool, first, count);
}

/** @brief vkCmdBeginQuery. */
static VKAPI_ATTR void VKAPI_CALL
cmd_begin_query (VkCommandBuffer     cb,
                 VkQueryPool         pool,
                 uint32_t            query,
                 VkQueryControlFlags flags)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	emitf("CmdBeginQuery", "qp=%s query=%" PRIu32 " flags=%s", name_of(KIND_QUERY_POOL, pool).text, query,
	      hex(flags).text);
	unlock();
	d->vkCmdBeginQuery(cb, pool, query, flags);
}

/** @brief vkCmdEndQuery. */
static VKAPI_ATTR void VKAPI_CALL
cmd_end_query (VkCommandBuffer cb,
               VkQueryPool     pool,
               uint32_t        query)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	emitf("CmdEndQuery", "qp=%s query=%" PRIu32, name_of(KIND_QUERY_POOL, pool).text, query);
	unlock();
	d->vkCmdEndQuery(cb, pool, query);
}

/** @brief vkCmdExecuteCommands. */
static VKAPI_ATTR void VKAPI_CALL
cmd_execute_commands (VkCommandBuffer        cb,
                      uint32_t               n,
                      VkCommandBuffer const *cbs)
{
	struct device_data *const d = dev(cb);
	lock();
	uint32_t at;
	cmd(cb, &at);
	put("cbs=[");
	for (uint32_t i = 0; i < n; ++i)
		put("%s%s", i ? "," : "", name_of(KIND_CMD, cbs[i]).text);
	emitf("CmdExecuteCommands", "]");
	unlock();
	d->vkCmdExecuteCommands(cb, n, cbs);
}

// ---- queries, fences, semaphores, swapchains ----------------------------------------------------

/** @brief vkCreateQueryPool. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_query_pool (VkDevice                     device,
                   VkQueryPoolCreateInfo const *ci,
                   VkAllocationCallbacks const *alloc,
                   VkQueryPool                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateQueryPool(device, ci, alloc, out);
	lock();
	emitf("CreateQueryPool", "qp=%s type=%s count=%" PRIu32 " rc=%d", created(r, KIND_QUERY_POOL, *out).text,
	      enum_word(vkt_query_type_name(ci->queryType), ci->queryType).text, ci->queryCount, r);
	unlock();
	return r;
}

/** @brief vkDestroyQueryPool. */
static VKAPI_ATTR void VKAPI_CALL
destroy_query_pool (VkDevice                     device,
                    VkQueryPool                  pool,
                    VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (pool)
		emitf("DestroyQueryPool", "qp=%s", forget(KIND_QUERY_POOL, pool).text);
	unlock();
	d->vkDestroyQueryPool(device, pool, alloc);
}

/** @brief vkGetQueryPoolResults. */
static VKAPI_ATTR VkResult VKAPI_CALL
get_query_pool_results (VkDevice           device,
                        VkQueryPool        pool,
                        uint32_t           first,
                        uint32_t           count,
                        size_t             size,
                        void              *data,
                        VkDeviceSize       stride,
                        VkQueryResultFlags flags)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkGetQueryPoolResults(device, pool, first, count, size, data, stride, flags);
	lock();
	emitf("GetQueryPoolResults", "qp=%s first=%" PRIu32 " count=%" PRIu32 " flags=%s rc=%d",
	      name_of(KIND_QUERY_POOL, pool).text, first, count, hex(flags).text, r);
	unlock();
	return r;
}

/** @brief vkCreateFence. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_fence (VkDevice                     device,
              VkFenceCreateInfo const     *ci,
              VkAllocationCallbacks const *alloc,
              VkFence                     *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateFence(device, ci, alloc, out);
	lock();
	put("fence=%s flags=%s chain=", created(r, KIND_FENCE, *out).text, hex(ci->flags).text);
	put_chain(ci->pNext);
	emitf("CreateFence", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroyFence. */
static VKAPI_ATTR void VKAPI_CALL
destroy_fence (VkDevice                     device,
               VkFence                      fence,
               VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (fence)
		emitf("DestroyFence", "fence=%s", forget(KIND_FENCE, fence).text);
	unlock();
	d->vkDestroyFence(device, fence, alloc);
}

/** @brief vkResetFences. */
static VKAPI_ATTR VkResult VKAPI_CALL
reset_fences (VkDevice       device,
              uint32_t       n,
              VkFence const *fences)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkResetFences(device, n, fences);
	lock();
	put("fences=[");
	for (uint32_t i = 0; i < n; ++i)
		put("%s%s", i ? "," : "", name_of(KIND_FENCE, fences[i]).text);
	emitf("ResetFences", "] rc=%d", r);
	unlock();
	return r;
}

/** @brief Hashes and logs the host ranges of completed work. Caller holds g_lock.
 *
 * @param d   The device.
 * @param p   The work.
 * @param why What showed it complete: "fence", "queue" or "device".
 */
static void
hash_host (struct device_data   *d,
           struct pending const *p,
           char const           *why)
{
	for (size_t i = 0; i < p->ranges.count; ++i) {
		struct host_range const *const h = &p->ranges.items[i];
		struct memory_info const *const m = memory_map_find(&g_mem, key_of(h->memory));
		if (!m)
			continue;
		unsigned char const *base = nullptr;
		void *mine = nullptr;
		if (m->mapped && h->offset >= m->map_offset && h->offset + h->size <= m->map_offset + m->map_size) {
			base = (unsigned char const *)m->mapped + (h->offset - m->map_offset);
			if (!(m->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
				VkMappedMemoryRange const range = {
					.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
					.memory = h->memory,
					.offset = m->map_offset,
					.size   = VK_WHOLE_SIZE,
				};
				d->vkInvalidateMappedMemoryRanges(d->device, 1, &range);
			}
		} else if (!m->mapped) {
			void *mapped;
			if (d->vkMapMemory(d->device, h->memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
				mine = mapped;
				base = (unsigned char const *)mapped + h->offset;
			}
		}
		if (base)
			emitf("Readback", "sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=%016" PRIx64 " via=%s",
			      p->submit, h->label, name_of(KIND_MEMORY, h->memory).text, h->offset, h->size,
			      host_fnv(m, base, h->size), why);
		else
			emitf("Readback", "sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=unmapped", p->submit,
			      h->label, name_of(KIND_MEMORY, h->memory).text, h->offset, h->size);
		if (mine)
			d->vkUnmapMemory(d->device, h->memory);
	}
}

/** @brief What completes a device's pending work. */
enum completion : uint8_t {
	COMPLETE_FENCE, //!< The work with a fence.
	COMPLETE_QUEUE, //!< The work on a queue.
	COMPLETE_ALL,   //!< All of it.
};

/** @brief Hashes the host ranges of a device's work that is known complete, and forgets that work.
 *         Caller holds g_lock.
 *
 * @param d      The device.
 * @param how    Which work.
 * @param handle The fence or the queue, or nullptr for all.
 * @param why    What showed it complete.
 */
static void
complete (struct device_data *d,
          enum completion     how,
          void const         *handle,
          char const         *why)
{
	size_t kept = 0;
	for (size_t i = 0; i < d->pending.count; ++i) {
		struct pending *const p = &d->pending.items[i];
		uint64_t const key = how == COMPLETE_FENCE ? key_of(p->fence) : key_of(p->queue);
		bool const done = how == COMPLETE_ALL || key == key_of(handle);
		if (done) {
			hash_host(d, p, why);
			host_ranges_fini(&p->ranges);
		} else {
			d->pending.items[kept++] = *p;
		}
	}
	d->pending.count = kept;
}

/** @brief vkWaitForFences: hashes the readbacks of what the fences show complete. */
static VKAPI_ATTR VkResult VKAPI_CALL
wait_for_fences (VkDevice       device,
                 uint32_t       n,
                 VkFence const *fences,
                 VkBool32       all,
                 uint64_t       timeout)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkWaitForFences(device, n, fences, all, timeout);
	lock();
	put("fences=[");
	for (uint32_t i = 0; i < n; ++i)
		put("%s%s", i ? "," : "", name_of(KIND_FENCE, fences[i]).text);
	emitf("WaitForFences", "] all=%" PRIu32 " timeout=%" PRIu64 " rc=%d", all, timeout, r);
	if (r == VK_SUCCESS)
		for (uint32_t i = 0; i < n; ++i)
			if (all || n == 1 || d->vkGetFenceStatus(device, fences[i]) == VK_SUCCESS)
				complete(d, COMPLETE_FENCE, fences[i], "fence");
	flush();
	unlock();
	return r;
}

/** @brief vkGetFenceStatus: hashes the readbacks of what a signaled fence shows complete. */
static VKAPI_ATTR VkResult VKAPI_CALL
get_fence_status (VkDevice device,
                  VkFence  fence)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkGetFenceStatus(device, fence);
	lock();
	emitf("GetFenceStatus", "fence=%s rc=%d", name_of(KIND_FENCE, fence).text, r);
	if (r == VK_SUCCESS)
		complete(d, COMPLETE_FENCE, fence, "fence");
	unlock();
	return r;
}

/** @brief vkCreateSemaphore. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_semaphore (VkDevice                     device,
                  VkSemaphoreCreateInfo const *ci,
                  VkAllocationCallbacks const *alloc,
                  VkSemaphore                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateSemaphore(device, ci, alloc, out);
	lock();
	put("sem=%s chain=", created(r, KIND_SEMAPHORE, *out).text);
	put_chain(ci->pNext);
	emitf("CreateSemaphore", " rc=%d", r);
	unlock();
	return r;
}

/** @brief vkDestroySemaphore. */
static VKAPI_ATTR void VKAPI_CALL
destroy_semaphore (VkDevice                     device,
                   VkSemaphore                  sem,
                   VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (sem)
		emitf("DestroySemaphore", "sem=%s", forget(KIND_SEMAPHORE, sem).text);
	unlock();
	d->vkDestroySemaphore(device, sem, alloc);
}

/** @brief The key of a swapchain's image template in g_img.
 *
 * @param sc The swapchain.
 * @return   Its key ^ 1, which no image has.
 */
static uint64_t
template_key (VkSwapchainKHR sc)
{
	return key_of(sc) ^ 1;
}

/** @brief vkCreateSwapchainKHR: keeps a template of its images. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_swapchain_khr (VkDevice                        device,
                      VkSwapchainCreateInfoKHR const *ci,
                      VkAllocationCallbacks const    *alloc,
                      VkSwapchainKHR                 *out)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkCreateSwapchainKHR(device, ci, alloc, out);
	lock();
	emitf("CreateSwapchainKHR", "sc=%s format=%s colorspace=%d extent=%" PRIu32 "x%" PRIu32 " usage=%s images=%" PRIu32
	      " present=%d old=%s rc=%d", created(r, KIND_SWAPCHAIN, *out).text, format_word(ci->imageFormat).text,
	      (int)ci->imageColorSpace, ci->imageExtent.width, ci->imageExtent.height, hex(ci->imageUsage).text,
	      ci->minImageCount, (int)ci->presentMode, name_of(KIND_SWAPCHAIN, ci->oldSwapchain).text, r);
	if (r == VK_SUCCESS) {
		bool added;
		*image_map_slot(&g_img, template_key(*out), &added) = (struct image_info){
			.device    = d,
			.extent    = {ci->imageExtent.width, ci->imageExtent.height, 1},
			.format    = ci->imageFormat,
			.layers    = ci->imageArrayLayers,
			.usage     = ci->imageUsage,
			.swapchain = true,
		};
	}
	unlock();
	return r;
}

/** @brief vkDestroySwapchainKHR: forgets its images. */
static VKAPI_ATTR void VKAPI_CALL
destroy_swapchain_khr (VkDevice                     device,
                       VkSwapchainKHR               sc,
                       VkAllocationCallbacks const *alloc)
{
	struct device_data *const d = dev(device);
	lock();
	if (sc) {
		emitf("DestroySwapchainKHR", "sc=%s", forget(KIND_SWAPCHAIN, sc).text);
		struct key_list *const images = list_map_find(&g_swapchain_images, key_of(sc));
		if (images) {
			for (size_t i = 0; i < images->count; ++i) {
				u32_map_remove(&g_names[KIND_IMAGE], images->items[i]);
				image_map_remove(&g_img, images->items[i]);
			}
			key_list_fini(images);
			list_map_remove(&g_swapchain_images, key_of(sc));
		}
		image_map_remove(&g_img, template_key(sc));
	}
	unlock();
	d->vkDestroySwapchainKHR(device, sc, alloc);
}

/** @brief vkGetSwapchainImagesKHR: names the images and gives each the swapchain's template. */
static VKAPI_ATTR VkResult VKAPI_CALL
get_swapchain_images_khr (VkDevice        device,
                          VkSwapchainKHR  sc,
                          uint32_t       *count,
                          VkImage        *images)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkGetSwapchainImagesKHR(device, sc, count, images);
	if (!images || (r != VK_SUCCESS && r != VK_INCOMPLETE))
		return r;
	lock();
	struct key_list *const owned = list_entry(&g_swapchain_images, key_of(sc));
	put("sc=%s imgs=[", name_of(KIND_SWAPCHAIN, sc).text);
	for (uint32_t i = 0; i < *count; ++i) {
		bool const was_known = known(KIND_IMAGE, images[i]);
		put("%s%s", i ? "," : "", (was_known ? name_of(KIND_IMAGE, images[i]) : fresh(KIND_IMAGE, images[i])).text);
		if (was_known)
			continue;
		*key_list_push(owned) = key_of(images[i]);
		struct image_info const *const t = image_map_find(&g_img, template_key(sc));
		if (t) {
			// The template may move when the image's entry is added.
			struct image_info const copy = *t;
			bool added;
			*image_map_slot(&g_img, key_of(images[i]), &added) = copy;
		}
	}
	emitf("GetSwapchainImagesKHR", "] rc=%d", r);
	unlock();
	return r;
}

// ---- submission and content hashes --------------------------------------------------------------

/** @brief A region to hash after a submission: what the selectors named, and what it resolved to. */
struct job {
	uint64_t      handle;           //!< The buffer or image.
	VkDeviceSize  offset;           //!< A buffer region's offset.
	VkDeviceSize  size;             //!< Its size as named, or VK_WHOLE_SIZE.
	VkDeviceSize  bytes;            //!< The bytes to copy.
	uint64_t      fnv;              //!< The FNV-1a 64 of what was copied.
	struct text   label;            //!< The selectors that named it, joined with '|'.
	char const   *skip;             //!< Why it is not copied, or nullptr.
	VkExtent3D    extent;           //!< An image's extent.
	VkFormat      format;           //!< An image's format.
	VkImageLayout layout;           //!< An image's layout after the submission.
	uint32_t      layers;           //!< An image's array layers.
	uint32_t      owner;            //!< A buffer that another queue family owns (external memory the
	                                //!< application released to VK_QUEUE_FAMILY_EXTERNAL): that
	                                //!< family, from which the copy acquires it and to which it
	                                //!< releases it again; else VK_QUEUE_FAMILY_IGNORED.
	uint32_t      number;           //!< The resource's creation number, which a later resource with
	                                //!< the same handle does not have.
	char          what[WHAT_BYTES]; //!< "buf12+OFFSET+BYTES" or "img3".
	bool          image;            //!< An image, which is copied whole.
};

VECTOR(jobs, struct job)

/** @brief Frees jobs and empties them.
 *
 * @param jobs The jobs.
 */
static void
jobs_fini (struct jobs *jobs)
{
	for (size_t i = 0; i < jobs->count; ++i)
		text_fini(&jobs->items[i].label);
	free(jobs->items);
	jobs->items = nullptr;
	*jobs = (struct jobs){};
}

/** @brief Whether the selectors hash after a submit.
 *
 * @param sub        The submit's number.
 * @param dispatches Whether it dispatches anything.
 * @return           true if they do.
 */
static bool
hash_this_submit (uint64_t sub,
                  bool     dispatches)
{
	if (!g_cfg.hash_any)
		return false;
	if (g_cfg.hash_dispatch_submits && !dispatches)
		return false;
	if (!g_cfg.hash_submits.count)
		return true;
	for (size_t i = 0; i < g_cfg.hash_submits.count; ++i)
		if (sub >= g_cfg.hash_submits.items[i].first && sub <= g_cfg.hash_submits.items[i].last)
			return true;
	return false;
}

/** @brief A command buffer of the device's own on a queue family, begun. Caller holds the device's
 *         hash_lock and not g_lock.
 *
 * @param d      The device.
 * @param family The queue family.
 * @return       The command buffer, or VK_NULL_HANDLE if it cannot be made.
 */
static VkCommandBuffer
hash_cb (struct device_data *d,
         uint32_t            family)
{
	struct hash_pool *pool = pool_map_find(&d->pools, family);
	if (!pool) {
		VkCommandPoolCreateInfo const ci = {
			.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = family,
		};
		VkCommandPool made;
		if (d->vkCreateCommandPool(d->device, &ci, nullptr, &made) != VK_SUCCESS)
			return VK_NULL_HANDLE;
		VkCommandBufferAllocateInfo const ai = {
			.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool        = made,
			.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkCommandBuffer cb;
		if (d->vkAllocateCommandBuffers(d->device, &ai, &cb) != VK_SUCCESS) {
			d->vkDestroyCommandPool(d->device, made, nullptr);
			return VK_NULL_HANDLE;
		}
		// A command buffer made below the loader carries no dispatch pointer yet.
		if (d->set_loader_data)
			d->set_loader_data(d->device, cb);
		else
			*(void **)cb = dispatch_key(d->device);
		bool added;
		pool = pool_map_slot(&d->pools, family, &added);
		*pool = (struct hash_pool){made, cb};
	}
	VkCommandBuffer const cb = pool->cb;
	d->vkResetCommandBuffer(cb, 0);
	VkCommandBufferBeginInfo const bi = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	d->vkBeginCommandBuffer(cb, &bi);
	return cb;
}

/** @brief Makes the device's staging buffer at least a size, mapped. Caller holds the device's
 *         hash_lock.
 *
 * @param d     The device.
 * @param bytes The size.
 * @return      false if it cannot be made.
 */
static bool
staging (struct device_data *d,
         VkDeviceSize        bytes)
{
	if (d->staging_size >= bytes)
		return true;
	if (d->staging)
		d->vkDestroyBuffer(d->device, d->staging, nullptr);
	if (d->staging_memory)
		d->vkFreeMemory(d->device, d->staging_memory, nullptr);
	d->staging = VK_NULL_HANDLE;
	d->staging_memory = VK_NULL_HANDLE;
	d->staging_mapped = nullptr;
	d->staging_size = 0;
	VkBufferCreateInfo const bi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size  = bytes,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	VkBuffer buffer;
	if (d->vkCreateBuffer(d->device, &bi, nullptr, &buffer) != VK_SUCCESS)
		return false;
	d->staging = buffer;
	VkMemoryRequirements req;
	d->vkGetBufferMemoryRequirements(d->device, buffer, &req);
	// Cached system memory reads fast; any host-visible type does. A device with nothing else
	// (lavapipe) has only device-local types.
	static struct {
		VkMemoryPropertyFlags want;   // The flags a type must have.
		VkMemoryPropertyFlags reject; // The flags it must not have.
	} const order[] = {
		{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
		{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
		{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0},
	};
	uint32_t type = UINT32_MAX;
	for (size_t o = 0; o < sizeof order / sizeof *order && type == UINT32_MAX; ++o) {
		for (uint32_t i = 0; i < d->memory.memoryTypeCount && type == UINT32_MAX; ++i) {
			VkMemoryPropertyFlags const f = d->memory.memoryTypes[i].propertyFlags;
			if ((req.memoryTypeBits & (1u << i)) && (f & order[o].want) == order[o].want && !(f & order[o].reject))
				type = i;
		}
	}
	if (type == UINT32_MAX)
		return false;
	VkMemoryAllocateInfo const ai = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory memory;
	if (d->vkAllocateMemory(d->device, &ai, nullptr, &memory) != VK_SUCCESS)
		return false;
	d->staging_memory = memory;
	if (d->vkBindBufferMemory(d->device, buffer, memory, 0) != VK_SUCCESS)
		return false;
	void *mapped;
	if (d->vkMapMemory(d->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
		return false;
	d->staging_mapped = mapped;
	VkMemoryPropertyFlags const flags = d->memory.memoryTypes[type].propertyFlags;
	d->staging_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	d->staging_cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
	d->staging_size = bytes;
	return true;
}

/** @brief Resolves a job's resource against what the trace knows. Caller holds g_lock.
 *
 * @param j The job, with its label, resource and region set.
 */
static void
resolve (struct job *j)
{
	enum kind const k = j->image ? KIND_IMAGE : KIND_BUFFER;
	uint32_t const *const number = u32_map_find(&g_names[k], j->handle);
	if (number)
		j->number = *number;
	struct name const n = key_name(k, j->handle);
	if (!j->image) {
		struct buffer_info const *const b = buffer_map_find(&g_buf, j->handle);
		if (!b) {
			snprintf(j->what, sizeof j->what, "%s", n.text);
			j->skip = "unknown";
			return;
		}
		j->bytes = j->size == VK_WHOLE_SIZE ? b->size - (j->offset < b->size ? j->offset : b->size) : j->size;
		snprintf(j->what, sizeof j->what, "%s+%" PRIu64 "+%" PRIu64, n.text, j->offset, j->bytes);
		if (!(b->usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
			j->skip = "no-transfer-src";
		} else if (b->external) {
			// Memory shared with another device or process: only after a submitted transfer says
			// who holds it, and then through the same acquire and release.
			j->owner = b->owner;
			if (j->owner == VK_QUEUE_FAMILY_IGNORED)
				j->skip = "external";
		}
		return;
	}
	snprintf(j->what, sizeof j->what, "%s", n.text);
	struct image_info const *const im = image_map_find(&g_img, j->handle);
	if (!im) {
		j->skip = "unknown";
		return;
	}
	uint32_t const texel = vkt_format_bytes(im->format);
	j->format = im->format;
	j->extent = im->extent;
	j->layers = im->layers;
	j->layout = im->layout;
	j->bytes = (VkDeviceSize)texel * im->extent.width * im->extent.height * im->extent.depth * im->layers;
	if (!(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
		j->skip = "no-transfer-src";
	else if (!texel || (im->format >= VK_FORMAT_D16_UNORM && im->format <= VK_FORMAT_D32_SFLOAT_S8_UINT))
		j->skip = "format";
	else if (im->external || im->swapchain)
		j->skip = "external";
	else if (im->layout == VK_IMAGE_LAYOUT_UNDEFINED || im->layout == VK_IMAGE_LAYOUT_PREINITIALIZED)
		j->skip = "layout";
}

/** @brief Why the resource a job resolved to can no longer be copied. Caller holds g_lock.
 *
 * @param j The job.
 * @return  "destroyed" when its handle names nothing or a newer resource; "unbound" when its
 *          memory was freed or never bound (a sparse resource counts as bound, because its binds
 *          are not traced); nullptr when it can be copied.
 */
static char const *
gone (struct job const *j)
{
	uint32_t const *const number = u32_map_find(&g_names[j->image ? KIND_IMAGE : KIND_BUFFER], j->handle);
	if (!number || *number != j->number)
		return "destroyed";
	if (j->image) {
		struct image_info const *const im = image_map_find(&g_img, j->handle);
		return im && (im->memory || im->sparse) ? nullptr : "unbound";
	}
	struct buffer_info const *const b = buffer_map_find(&g_buf, j->handle);
	return b && (b->memory || b->sparse) ? nullptr : "unbound";
}

/** @brief Copies each job's bytes out on a queue and logs their FNV. Called without g_lock, on the
 *         thread that submitted, with the queue idle. The hashing objects are the device's, so
 *         threads that submit to its queues take turns. Another thread may have destroyed a job's
 *         resource, or freed its memory, since the job was resolved; such a job is skipped.
 *         Destruction waits for hash_lock, so the rest stay until their copies are done.
 *
 * @param d      The device.
 * @param queue  The queue.
 * @param family Its family.
 * @param sub    The last submit's number.
 * @param jobs   The jobs.
 */
static void
run_jobs (struct device_data *d,
          VkQueue             queue,
          uint32_t            family,
          uint64_t            sub,
          struct jobs        *jobs)
{
	pthread_mutex_lock(&d->hash_lock);
	lock();
	for (size_t i = 0; i < jobs->count; ++i)
		if (!jobs->items[i].skip)
			jobs->items[i].skip = gone(&jobs->items[i]);
	unlock();
	// The jobs before this one have their lines: a job that cannot be recorded or submitted ends
	// the hashing.
	size_t lines = 0;
	for (size_t i = 0; i < jobs->count; ++i) {
		struct job *const j = &jobs->items[i];
		if (!j->skip && !j->bytes)
			j->skip = "empty";
		if (!j->skip && !staging(d, j->bytes))
			j->skip = "staging";
		if (j->skip) {
			lines = i + 1;
			continue;
		}
		VkCommandBuffer const cb = hash_cb(d, family);
		if (!cb)
			break;
		VkMemoryBarrier const before = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		};
		d->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before,
		                        0, nullptr, 0, nullptr);
		if (!j->image) {
			VkBuffer const buffer = (VkBuffer)(uintptr_t)j->handle;
			bool const foreign = j->owner != VK_QUEUE_FAMILY_IGNORED && j->owner != family;
			VkBufferMemoryBarrier own = {
				.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
				.srcQueueFamilyIndex = j->owner,
				.dstQueueFamilyIndex = family,
				.buffer              = buffer,
				.size                = VK_WHOLE_SIZE,
			};
			if (foreign)
				d->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
				                        nullptr, 1, &own, 0, nullptr);
			VkBufferCopy const region = {j->offset, 0, j->bytes};
			d->vkCmdCopyBuffer(cb, buffer, d->staging, 1, &region);
			if (foreign) {
				own.srcQueueFamilyIndex = family;
				own.dstQueueFamilyIndex = j->owner;
				own.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				own.dstAccessMask = 0;
				d->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
				                        nullptr, 1, &own, 0, nullptr);
			}
		} else {
			VkImage const image = (VkImage)(uintptr_t)j->handle;
			bool const direct = j->layout == VK_IMAGE_LAYOUT_GENERAL || j->layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			VkImageLayout const copy_layout = direct ? j->layout : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			VkImageMemoryBarrier to = {
				.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT,
				.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
				.oldLayout           = j->layout,
				.newLayout           = copy_layout,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image               = image,
				.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, j->layers},
			};
			if (!direct)
				d->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
				                        nullptr, 0, nullptr, 1, &to);
			VkBufferImageCopy const region = {
				.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, j->layers},
				.imageExtent      = j->extent,
			};
			d->vkCmdCopyImageToBuffer(cb, image, copy_layout, d->staging, 1, &region);
			if (!direct) {
				to.oldLayout = copy_layout;
				to.newLayout = j->layout;
				to.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				to.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
				d->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
				                        nullptr, 0, nullptr, 1, &to);
			}
		}
		VkMemoryBarrier const host = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		};
		d->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
		                        0, nullptr);
		d->vkEndCommandBuffer(cb);
		VkSubmitInfo const si = {
			.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers    = &cb,
		};
		lines = i + 1;
		if (d->vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS || d->vkQueueWaitIdle(queue) != VK_SUCCESS) {
			j->skip = "submit";
			break;
		}
		if (!d->staging_coherent) {
			VkMappedMemoryRange const range = {
				.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = d->staging_memory,
				.size   = VK_WHOLE_SIZE,
			};
			d->vkInvalidateMappedMemoryRanges(d->device, 1, &range);
		}
		j->fnv = d->staging_cached ? fnv(d->staging_mapped, j->bytes, FNV_BASIS)
		                           : fnv_uncached(d->staging_mapped, j->bytes);
	}
	pthread_mutex_unlock(&d->hash_lock);
	lock();
	for (size_t i = 0; i < lines; ++i) {
		struct job const *const j = &jobs->items[i];
		if (j->skip)
			emitf("Hash", "sub=%" PRIu64 " sel=%s res=%s skip=%s", sub, j->label.bytes, j->what, j->skip);
		else
			emitf("Hash", "sub=%" PRIu64 " sel=%s res=%s bytes=%" PRIu64 " fnv=%016" PRIx64, sub, j->label.bytes,
			      j->what, j->bytes, j->fnv);
	}
	flush();
	unlock();
}

/** @brief Adds a region the selectors name to a submission's jobs; a region that an earlier selector
 *         of the submission named gets this label too. Caller holds g_lock.
 *
 * @param jobs   The jobs.
 * @param first  The submission's first job.
 * @param image  An image, which is copied whole.
 * @param handle The buffer or image.
 * @param offset A buffer region's offset.
 * @param size   Its size, or VK_WHOLE_SIZE.
 * @param fmt    A printf format for the label.
 * @param ...    The format's arguments.
 */
[[gnu::format(printf, 7, 8)]]
static void
select_region (struct jobs  *jobs,
               size_t        first,
               bool          image,
               uint64_t      handle,
               VkDeviceSize  offset,
               VkDeviceSize  size,
               char const   *fmt,
               ...)
{
	struct job *j = nullptr;
	for (size_t i = first; i < jobs->count && !j; ++i) {
		struct job *const u = &jobs->items[i];
		if (u->image == image && u->handle == handle && u->offset == offset && u->size == size)
			j = u;
	}
	if (j) {
		text_append(&j->label, "|", 1);
	} else {
		j = jobs_push(jobs);
		*j = (struct job){
			.handle = handle, .offset = offset, .size = size, .owner = VK_QUEUE_FAMILY_IGNORED, .layers = 1,
			.image = image,
		};
	}
	va_list args;
	va_start(args, fmt);
	text_vprintf(&j->label, fmt, args);
	va_end(args);
}

/** @brief A live resource of a device, in the order that the all selector hashes them. */
struct ordered {
	uint64_t handle; //!< The buffer or image.
	uint32_t order;  //!< Its creation number; with the top bit set for an image.
	VkBool32 image;  //!< An image.
};

/** @brief Orders resources for the all selector.
 *
 * @param a A struct ordered.
 * @param b Another.
 * @return  Their order.
 */
static int
ordered_compare (void const *a,
                 void const *b)
{
	uint32_t const x = ((struct ordered const *)a)->order;
	uint32_t const y = ((struct ordered const *)b)->order;
	return (x > y) - (x < y);
}

VECTOR(ordered_list, struct ordered)

/** @brief Adds the regions the selectors name in a submission's command buffers to its jobs, each
 *         region once. Caller holds g_lock.
 *
 * @param d    The device.
 * @param cbs  The command buffers.
 * @param ncb  How many.
 * @param jobs The jobs.
 */
static void
select_targets (struct device_data const *d,
                VkCommandBuffer const    *cbs,
                uint32_t                  ncb,
                struct jobs              *jobs)
{
	size_t const first = jobs->count;
	uint32_t dispatch = 0;
	for (uint32_t c = 0; c < ncb; ++c) {
		struct cb_state const *const s = cb_map_find(&g_cb, key_of(cbs[c]));
		if (!s)
			continue;
		if (g_cfg.hash_i2b)
			for (size_t i = 0; i < s->i2b_dst.count; ++i) {
				struct target const *const t = &s->i2b_dst.items[i];
				select_region(jobs, first, t->image, t->handle, t->offset, t->size, "i2b:%s", t->label);
			}
		if (g_cfg.hash_copydst)
			for (size_t i = 0; i < s->copy_dst.count; ++i) {
				struct target const *const t = &s->copy_dst.items[i];
				select_region(jobs, first, t->image, t->handle, t->offset, t->size, "copydst:%s", t->label);
			}
		for (size_t k = 0; k < s->dispatch_ends.count; ++k, ++dispatch) {
			bool wanted = g_cfg.hash_storage;
			for (size_t i = 0; i < g_cfg.hash_dispatch.count && !wanted; ++i)
				wanted = g_cfg.hash_dispatch.items[i] == dispatch;
			if (!wanted)
				continue;
			for (size_t i = k ? s->dispatch_ends.items[k - 1] : 0; i < s->dispatch_ends.items[k]; ++i) {
				struct target const *const t = &s->dispatch_targets.items[i];
				select_region(jobs, first, t->image, t->handle, t->offset, t->size, "dispatch%" PRIu32 ":%s",
				              dispatch, t->label);
			}
		}
	}
	for (size_t i = 0; i < g_cfg.hash_names.count; ++i) {
		struct hash_name const *const n = &g_cfg.hash_names.items[i];
		bool const image = n->text[0] == 'i';
		struct u32_map const *const names = &g_names[image ? KIND_IMAGE : KIND_BUFFER];
		for (size_t h = 0; h < names->count; ++h)
			if (names->values[h] == n->number)
				select_region(jobs, first, image, names->keys[h], 0, VK_WHOLE_SIZE, "%.*s", (int)n->length, n->text);
	}
	if (g_cfg.hash_all) {
		struct ordered_list all = {};
		for (size_t i = 0; i < g_buf.count; ++i) {
			uint32_t const *const number = u32_map_find(&g_names[KIND_BUFFER], g_buf.keys[i]);
			if (g_buf.values[i].device == d && number)
				*ordered_list_push(&all) = (struct ordered){g_buf.keys[i], *number, false};
		}
		for (size_t i = 0; i < g_img.count; ++i) {
			uint32_t const *const number = u32_map_find(&g_names[KIND_IMAGE], g_img.keys[i]);
			if (g_img.values[i].device == d && number)
				*ordered_list_push(&all) = (struct ordered){g_img.keys[i], *number | 0x80000000u, true};
		}
		if (all.count)
			qsort(all.items, all.count, sizeof *all.items, ordered_compare);
		for (size_t i = 0; i < all.count; ++i)
			select_region(jobs, first, all.items[i].image, all.items[i].handle, 0, VK_WHOLE_SIZE, "all");
		free(all.items);
		all.items = nullptr;
	}
}

/** @brief Starts a submit line, up to its wait semaphores: "sub=N q=qN batch=B cbs=[...] wait=[".
 *         Caller holds g_lock.
 *
 * @param d     The device.
 * @param queue The queue.
 * @param batch The batch's index in the call.
 * @param cbs   Its command buffers.
 * @param ncb   How many.
 * @return      The submit's number.
 */
static uint64_t
batch_begin (struct device_data    *d,
             VkQueue                queue,
             uint32_t               batch,
             VkCommandBuffer const *cbs,
             uint32_t               ncb)
{
	uint64_t const sub = ++d->submits;
	put("sub=%" PRIu64 " q=%s batch=%" PRIu32 " cbs=[", sub, name_of(KIND_QUEUE, queue).text, batch);
	for (uint32_t i = 0; i < ncb; ++i)
		put("%s%s", i ? "," : "", name_of(KIND_CMD, cbs[i]).text);
	put("] wait=[");
	return sub;
}

/** @brief Logs a batch, after its semaphores, and does its host work: the ownership transfers and
 *         layouts its command buffers recorded, its uploads, and its readbacks for later; adds the
 *         regions to hash once it completes to the jobs. Caller holds g_lock.
 *
 * @param d     The device.
 * @param queue The queue.
 * @param fence The call's fence.
 * @param sub   The batch's submit number.
 * @param cbs   Its command buffers.
 * @param ncb   How many.
 * @param call  The call's name.
 * @param jobs  The jobs.
 */
static void
batch_end (struct device_data    *d,
           VkQueue                queue,
           VkFence                fence,
           uint64_t               sub,
           VkCommandBuffer const *cbs,
           uint32_t               ncb,
           char const            *call,
           struct jobs           *jobs)
{
	emitf(call, "] fence=%s", name_of(KIND_FENCE, fence).text);
	struct pending p = {.queue = queue, .fence = fence, .submit = sub};
	bool dispatches = false;
	for (uint32_t c = 0; c < ncb; ++c) {
		struct cb_state const *const s = cb_map_find(&g_cb, key_of(cbs[c]));
		if (!s)
			continue;
		dispatches |= s->dispatches != 0;
		for (size_t i = 0; i < s->owners.count; ++i) {
			struct buffer_info *const b = buffer_map_find(&g_buf, key_of(s->owners.items[i].buffer));
			if (b)
				b->owner = s->owners.items[i].family;
		}
		for (size_t i = 0; i < s->layouts.count; ++i) {
			struct image_info *const im = image_map_find(&g_img, key_of(s->layouts.items[i].image));
			if (im)
				im->layout = s->layouts.items[i].layout;
		}
		for (size_t i = 0; i < s->uploads.count; ++i) {
			struct host_range const *const h = &s->uploads.items[i];
			struct memory_info const *const m = memory_map_find(&g_mem, key_of(h->memory));
			if (!m)
				continue;
			if (m->mapped && h->offset >= m->map_offset && h->offset + h->size <= m->map_offset + m->map_size)
				emitf("Upload", "sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=%016" PRIx64, sub,
				      h->label, name_of(KIND_MEMORY, h->memory).text, h->offset, h->size,
				      host_fnv(m, (unsigned char const *)m->mapped + (h->offset - m->map_offset), h->size));
			else
				emitf("Upload", "sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=unmapped", sub,
				      h->label, name_of(KIND_MEMORY, h->memory).text, h->offset, h->size);
		}
		for (size_t i = 0; i < s->downloads.count; ++i)
			*host_ranges_push(&p.ranges) = s->downloads.items[i];
	}
	if (p.ranges.count)
		*pendings_push(&d->pending) = p;
	if (hash_this_submit(sub, dispatches)) {
		size_t const first = jobs->count;
		select_targets(d, cbs, ncb, jobs);
		for (size_t i = first; i < jobs->count; ++i)
			resolve(&jobs->items[i]);
	}
}

/** @brief The family of a queue. Caller holds g_lock.
 *
 * @param queue The queue.
 * @return      Its family, or 0 for a queue the trace does not know.
 */
static uint32_t
queue_family (VkQueue queue)
{
	uint32_t const *const family = u32_map_find(&g_queue, key_of(queue));
	return family ? *family : 0;
}

/** @brief vkQueueSubmit: logs each batch, then hashes what the selectors name once the queue is
 *         idle.
 */
static VKAPI_ATTR VkResult VKAPI_CALL
queue_submit (VkQueue             queue,
              uint32_t            n,
              VkSubmitInfo const *s,
              VkFence             fence)
{
	struct device_data *const d = dev(queue);
	VkResult const r = d->vkQueueSubmit(queue, n, s, fence);
	struct jobs jobs = {};
	uint64_t last = 0;
	lock();
	uint32_t const family = queue_family(queue);
	for (uint32_t b = 0; b < n; ++b) {
		uint64_t const sub = batch_begin(d, queue, b, s[b].pCommandBuffers, s[b].commandBufferCount);
		for (uint32_t i = 0; i < s[b].waitSemaphoreCount; ++i)
			put("%s%s@%s", i ? "," : "", name_of(KIND_SEMAPHORE, s[b].pWaitSemaphores[i]).text,
			    hex(s[b].pWaitDstStageMask[i]).text);
		VkTimelineSemaphoreSubmitInfo const *const tl = find_in_chain(s[b].pNext,
		                                                              VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
		for (uint32_t i = 0; tl && i < tl->waitSemaphoreValueCount; ++i)
			put(":%" PRIu64, tl->pWaitSemaphoreValues[i]);
		put("] signal=[");
		for (uint32_t i = 0; i < s[b].signalSemaphoreCount; ++i)
			put("%s%s", i ? "," : "", name_of(KIND_SEMAPHORE, s[b].pSignalSemaphores[i]).text);
		for (uint32_t i = 0; tl && i < tl->signalSemaphoreValueCount; ++i)
			put(":%" PRIu64, tl->pSignalSemaphoreValues[i]);
		batch_end(d, queue, fence, sub, s[b].pCommandBuffers, s[b].commandBufferCount, "QueueSubmit", &jobs);
		last = sub;
	}
	if (!n)
		emitf("QueueSubmit", "q=%s batches=0 fence=%s", name_of(KIND_QUEUE, queue).text,
		      name_of(KIND_FENCE, fence).text);
	emitf("QueueSubmitResult", "q=%s last=%" PRIu64 " rc=%d", name_of(KIND_QUEUE, queue).text, last, r);
	flush();
	unlock();
	if (r == VK_SUCCESS && jobs.count && d->vkQueueWaitIdle(queue) == VK_SUCCESS)
		run_jobs(d, queue, family, last, &jobs);
	jobs_fini(&jobs);
	return r;
}

VECTOR(cb_list, VkCommandBuffer)

/** @brief The command buffers of a vkQueueSubmit2 batch, gathered for batch_begin() and batch_end().
 *         Caller holds g_lock.
 */
static struct cb_list g_submit2_cbs;

/** @brief vkQueueSubmit2 and vkQueueSubmit2KHR, both logged as vkQueueSubmit2.
 *
 * @param next  The next layer's function.
 * @param queue The queue.
 * @param n     The batches.
 * @param s     Their array.
 * @param fence The fence.
 * @return      What the next layer returned.
 */
static VkResult
submit2 (PFN_vkQueueSubmit2   next,
         VkQueue              queue,
         uint32_t             n,
         VkSubmitInfo2 const *s,
         VkFence              fence)
{
	struct device_data *const d = dev(queue);
	VkResult const r = next(queue, n, s, fence);
	struct jobs jobs = {};
	uint64_t last = 0;
	lock();
	uint32_t const family = queue_family(queue);
	for (uint32_t b = 0; b < n; ++b) {
		uint32_t const ncb = s[b].commandBufferInfoCount;
		g_submit2_cbs.count = 0;
		for (uint32_t i = 0; i < ncb; ++i)
			*cb_list_push(&g_submit2_cbs) = s[b].pCommandBufferInfos[i].commandBuffer;
		VkCommandBuffer const *const cbs = g_submit2_cbs.items;
		uint64_t const sub = batch_begin(d, queue, b, cbs, ncb);
		for (uint32_t i = 0; i < s[b].waitSemaphoreInfoCount; ++i) {
			VkSemaphoreSubmitInfo const *const w = &s[b].pWaitSemaphoreInfos[i];
			put("%s%s@%s:%" PRIu64, i ? "," : "", name_of(KIND_SEMAPHORE, w->semaphore).text, hex(w->stageMask).text,
			    w->value);
		}
		put("] signal=[");
		for (uint32_t i = 0; i < s[b].signalSemaphoreInfoCount; ++i) {
			VkSemaphoreSubmitInfo const *const g = &s[b].pSignalSemaphoreInfos[i];
			put("%s%s@%s:%" PRIu64, i ? "," : "", name_of(KIND_SEMAPHORE, g->semaphore).text, hex(g->stageMask).text,
			    g->value);
		}
		batch_end(d, queue, fence, sub, cbs, ncb, "QueueSubmit2", &jobs);
		last = sub;
	}
	emitf("QueueSubmitResult", "q=%s last=%" PRIu64 " rc=%d", name_of(KIND_QUEUE, queue).text, last, r);
	flush();
	unlock();
	if (r == VK_SUCCESS && jobs.count && d->vkQueueWaitIdle(queue) == VK_SUCCESS)
		run_jobs(d, queue, family, last, &jobs);
	jobs_fini(&jobs);
	return r;
}

/** @brief vkQueueSubmit2. */
static VKAPI_ATTR VkResult VKAPI_CALL
queue_submit2 (VkQueue              queue,
               uint32_t             n,
               VkSubmitInfo2 const *s,
               VkFence              fence)
{
	return submit2(dev(queue)->vkQueueSubmit2, queue, n, s, fence);
}

/** @brief vkQueueSubmit2KHR. */
static VKAPI_ATTR VkResult VKAPI_CALL
queue_submit2_khr (VkQueue              queue,
                   uint32_t             n,
                   VkSubmitInfo2 const *s,
                   VkFence              fence)
{
	return submit2(dev(queue)->vkQueueSubmit2KHR, queue, n, s, fence);
}

/** @brief vkQueueWaitIdle: hashes the readbacks of the queue's work. */
static VKAPI_ATTR VkResult VKAPI_CALL
queue_wait_idle (VkQueue queue)
{
	struct device_data *const d = dev(queue);
	VkResult const r = d->vkQueueWaitIdle(queue);
	lock();
	emitf("QueueWaitIdle", "q=%s rc=%d", name_of(KIND_QUEUE, queue).text, r);
	if (r == VK_SUCCESS)
		complete(d, COMPLETE_QUEUE, queue, "queue");
	flush();
	unlock();
	return r;
}

/** @brief vkDeviceWaitIdle: hashes the readbacks of all the device's work. */
static VKAPI_ATTR VkResult VKAPI_CALL
device_wait_idle (VkDevice device)
{
	struct device_data *const d = dev(device);
	VkResult const r = d->vkDeviceWaitIdle(device);
	lock();
	emitf("DeviceWaitIdle", "dev=%s rc=%d", name_of(KIND_DEVICE, device).text, r);
	if (r == VK_SUCCESS)
		complete(d, COMPLETE_ALL, nullptr, "device");
	flush();
	unlock();
	return r;
}

/** @brief vkQueuePresentKHR. */
static VKAPI_ATTR VkResult VKAPI_CALL
queue_present_khr (VkQueue                 queue,
                   VkPresentInfoKHR const *pi)
{
	struct device_data *const d = dev(queue);
	lock();
	put("q=%s images=[", name_of(KIND_QUEUE, queue).text);
	for (uint32_t i = 0; i < pi->swapchainCount; ++i)
		put("%s%s:%" PRIu32, i ? "," : "", name_of(KIND_SWAPCHAIN, pi->pSwapchains[i]).text, pi->pImageIndices[i]);
	put("] wait=[");
	for (uint32_t i = 0; i < pi->waitSemaphoreCount; ++i)
		put("%s%s", i ? "," : "", name_of(KIND_SEMAPHORE, pi->pWaitSemaphores[i]).text);
	emitf("QueuePresentKHR", "]");
	unlock();
	VkResult const r = d->vkQueuePresentKHR(queue, pi);
	lock();
	emitf("QueuePresentResult", "q=%s rc=%d", name_of(KIND_QUEUE, queue).text, r);
	flush();
	unlock();
	return r;
}

// ---- lookup -------------------------------------------------------------------------------------

/** @brief A function the layer answers for. */
struct entry {
	char const         *name; //!< The function's name.
	PFN_vkVoidFunction  fn;   //!< The layer's function.
};

/** @brief An entry; a function that does not have its Vulkan function's type does not compile. */
#define ENTRY(vk, hook) {#vk, _Generic(hook, PFN_##vk: (PFN_vkVoidFunction)hook)},

/** @brief The instance functions the layer answers for. */
static struct entry const INSTANCE_ENTRIES[] = {
	ENTRY(vkGetInstanceProcAddr,      get_instance_proc_addr)
	ENTRY(vkCreateInstance,           create_instance)
	ENTRY(vkDestroyInstance,          destroy_instance)
	ENTRY(vkEnumeratePhysicalDevices, enumerate_physical_devices)
	ENTRY(vkCreateDevice,             create_device)
};

/** @brief The device functions the layer answers for. */
static struct entry const DEVICE_ENTRIES[] = {DEVICE_HOOKS(ENTRY)};

#undef ENTRY
#undef DEVICE_CALLS
#undef DEVICE_HOOKS
#undef INSTANCE_CALLS

/** @brief The layer's function for a device function, if the device's next layer has that function
 *         too: an intercept for a function the device lacks must not be handed out.
 *
 * @param d    The device, or nullptr to check nothing.
 * @param name The function's name.
 * @return     The layer's function, or nullptr.
 */
static PFN_vkVoidFunction
device_entry (struct device_data const *d,
              char const               *name)
{
	for (size_t i = 0; i < sizeof DEVICE_ENTRIES / sizeof *DEVICE_ENTRIES; ++i) {
		if (strcmp(DEVICE_ENTRIES[i].name, name))
			continue;
		if (d && !d->vkGetDeviceProcAddr(d->device, name))
			return nullptr;
		return DEVICE_ENTRIES[i].fn;
	}
	return nullptr;
}

/** @brief vkGetDeviceProcAddr. */
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
get_device_proc_addr (VkDevice    device,
                      char const *name)
{
	struct device_data const *const d = dev(device);
	if (!d)
		return nullptr;
	PFN_vkVoidFunction const f = device_entry(d, name);
	return f ? f : d->vkGetDeviceProcAddr(device, name);
}

/** @brief vkGetInstanceProcAddr. */
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
get_instance_proc_addr (VkInstance  instance,
                        char const *name)
{
	for (size_t i = 0; i < sizeof INSTANCE_ENTRIES / sizeof *INSTANCE_ENTRIES; ++i)
		if (!strcmp(INSTANCE_ENTRIES[i].name, name))
			return INSTANCE_ENTRIES[i].fn;
	if (!instance)
		return nullptr;
	lock();
	struct instance_data const *const data = instance_of(instance);
	unlock();
	if (!data)
		return nullptr;
	// Device functions through the instance: ours, when the driver has them at all.
	PFN_vkVoidFunction const f = device_entry(nullptr, name);
	if (f)
		return data->gipa(instance, name) ? f : nullptr;
	return data->gipa(instance, name);
}

VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion (VkNegotiateLayerInterface *v)
{
	if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
		return VK_ERROR_INITIALIZATION_FAILED;
	if (v->loaderLayerInterfaceVersion > 2)
		v->loaderLayerInterfaceVersion = 2;
	v->pfnGetInstanceProcAddr = get_instance_proc_addr;
	v->pfnGetDeviceProcAddr = get_device_proc_addr;
	v->pfnGetPhysicalDeviceProcAddr = nullptr;
	return VK_SUCCESS;
}

#undef VECTOR
