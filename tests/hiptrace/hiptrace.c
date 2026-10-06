/** @file
 *
 * A tracing HIP runtime for dlsslopd: DLSSLOP_HIP_LIBRARY=libhiptrace.so (tests/hiptrace/trace.sh;
 * VALIDATION.md describes the procedure).
 *
 * Exports every entry point dlsslopd's loader (backend/hip.h) resolves, and those upstream's host
 * code resolved, so that builds from before the port trace too. Forwards each call to the real
 * runtime and logs it to HIPTRACE_FILE (no file: forward only). Device and host pointers are logged
 * as allocation id + byte offset, so traces of two runs, or of two implementations, compare with
 * diff. Kernel arguments are decoded with the AMDGPU metadata note of the code object passed to
 * hipModuleLoadData.
 *
 * Environment:
 *   HIPTRACE_FILE          the log; unset or empty: no log
 *   HIPTRACE_REAL          the real runtime; unset or empty: the first of dlsslopd's loader's
 *                          candidates that loads
 *   HIPTRACE_MODULES       directory whose *.hsaco name the loaded images (matched by FNV-1a 64 of
 *                          the image); unset or empty: DLSSLOP_MODULES
 *   HIPTRACE_DEEP          comma-separated ranges "A-B" and ordinals "A", such as "0-10,20": after
 *                          launches A..B-1, or A (0-based launch ordinals), wait for the stream and
 *                          log the FNV of every non-uploaded buffer argument, from the argument to
 *                          the end of its allocation; "all": after every launch; unset or empty:
 *                          none; anything else, a number above 2^64 - 1 included: none, with a
 *                          message on stderr and deep=invalid in the log
 *   HIPTRACE_DEEP_KERNELS  comma-separated kernel names deep-hashed always; unset or empty: none
 *
 * Every function with external linkage here is a HIP entry point; the rest is static.
 */
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../backend/hip.h"
#include "../support.h"

/** @brief The FNV-1a 64 offset basis. */
static constexpr uint64_t FNV_BASIS = UINT64_C(14695981039346656037);

/** @brief The FNV-1a 64 prime. */
static constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);

/** @brief hipErrorNotFound, the answer for an entry point that the real runtime lacks. */
static constexpr int HIP_ERROR_NOT_FOUND = 500;

/** @brief hipMemcpyKind's names, by value. */
static char const *const COPY_KINDS[] = {"H2H", "H2D", "D2H", "D2D", "DEFAULT"};

/** @brief The number of COPY_KINDS. */
static constexpr int COPY_KIND_COUNT = (int)(sizeof COPY_KINDS / sizeof *COPY_KINDS);

/** @brief The deepest MessagePack nesting read: the root is at depth 0. */
static constexpr unsigned MSG_DEPTH = 32;

/** @brief The most decimal digits of a uint64_t. */
static constexpr size_t DECIMAL_DIGITS = 20;

/** @brief The bytes of a short name in the trace: a kind letter, a 32-bit id, '+', a 64-bit offset
 *         and the null, the longest that where() writes.
 */
static constexpr size_t NAME_BYTES = 1 + 10 + 1 + DECIMAL_DIGITS + 1;

static_assert(UINT_MAX <= UINT32_MAX, "an allocation's id has at most 10 digits");

/** @brief NT_AMDGPU_METADATA, the type of the note that holds a code object's metadata. */
static constexpr uint32_t NOTE_AMDGPU_METADATA = 32;

/** @brief The name of the AMDGPU notes' owner. */
static char const NOTE_AMDGPU[] = "AMDGPU";

/** @brief The metadata's key of its kernels' list. */
static char const AMDHSA_KERNELS[] = "amdhsa.kernels";

/** @brief The suffix of a module file's name. */
static char const HSACO[] = ".hsaco";

/** @brief The length of HSACO. */
static constexpr size_t HSACO_LENGTH = sizeof HSACO - 1;

/** @brief Every entry point that dlsslopd or upstream's host code resolves, as ROCm 7 (and 6.4)
 *         declare them, as X(return type, name, parameter types).
 */
#define HIPTRACE_FUNCTIONS(X) \
	X(int,          hipGetDevicePropertiesR0600,      (struct hip_device_properties *, int)) \
	X(int,          hipModuleLoadData,                (void **, void const *)) \
	X(int,          hipInit,                          (unsigned)) \
	X(int,          hipRuntimeGetVersion,             (int *)) \
	X(int,          hipGetDeviceCount,                (int *)) \
	X(int,          hipDeviceGetName,                 (char *, int, int)) \
	X(int,          hipSetDevice,                     (int)) \
	X(int,          hipSetDeviceFlags,                (unsigned)) \
	X(int,          hipMemGetInfo,                    (size_t *, size_t *)) \
	X(int,          hipMalloc,                        (void **, size_t)) \
	X(int,          hipHostMalloc,                    (void **, size_t, unsigned)) \
	X(int,          hipHostFree,                      (void *)) \
	X(int,          hipHostRegister,                  (void *, size_t, unsigned)) \
	X(int,          hipHostUnregister,                (void *)) \
	X(int,          hipFree,                          (void *)) \
	X(int,          hipMemcpy,                        (void *, void const *, size_t, int)) \
	X(int,          hipMemcpyAsync,                   (void *, void const *, size_t, int, void *)) \
	X(int,          hipMemsetAsync,                   (void *, int, size_t, void *)) \
	X(int,          hipEventCreate,                   (void **)) \
	X(int,          hipEventRecord,                   (void *, void *)) \
	X(int,          hipEventElapsedTime,              (float *, void *, void *)) \
	X(int,          hipEventDestroy,                  (void *)) \
	X(int,          hipEventSynchronize,              (void *)) \
	X(int,          hipDeviceSynchronize,             (void)) \
	X(int,          hipStreamCreate,                  (void **)) \
	X(int,          hipStreamSynchronize,             (void *)) \
	X(int,          hipStreamDestroy,                 (void *)) \
	X(int,          hipImportExternalMemory,          (void **, struct hip_memory_desc const *)) \
	X(int,          hipExternalMemoryGetMappedBuffer, (void **, void *, struct hip_buffer_desc const *)) \
	X(int,          hipDestroyExternalMemory,         (void *)) \
	X(int,          hipImportExternalSemaphore,       (void **, void const *)) \
	X(int,          hipSignalExternalSemaphoresAsync, (void *const *, void const *, unsigned, void *)) \
	X(int,          hipWaitExternalSemaphoresAsync,   (void *const *, void const *, unsigned, void *)) \
	X(int,          hipDestroyExternalSemaphore,      (void *)) \
	X(int,          hipStreamBeginCapture,            (void *, int)) \
	X(int,          hipStreamEndCapture,              (void *, void **)) \
	X(int,          hipGraphInstantiate,              (void **, void *, void **, char *, size_t)) \
	X(int,          hipGraphLaunch,                   (void *, void *)) \
	X(int,          hipGraphDestroy,                  (void *)) \
	X(int,          hipGraphExecDestroy,              (void *)) \
	X(int,          hipMemAddressReserve,             (void **, size_t, size_t, void *, unsigned long long)) \
	X(int,          hipMemAddressFree,                (void *, size_t)) \
	X(int,          hipMemCreate,                     (void **, size_t, void const *, unsigned long long)) \
	X(int,          hipMemRelease,                    (void *)) \
	X(int,          hipMemMap,                        (void *, size_t, size_t, void *, unsigned long long)) \
	X(int,          hipMemUnmap,                      (void *, size_t)) \
	X(int,          hipMemSetAccess,                  (void *, size_t, void const *, size_t)) \
	X(int,          hipMemGetAllocationGranularity,   (size_t *, void const *, unsigned)) \
	X(int,          hipModuleLoad,                    (void **, char const *)) \
	X(int,          hipModuleGetFunction,             (void **, void *, char const *)) \
	X(int,          hipModuleLaunchKernel,            (void *, unsigned, unsigned, unsigned, unsigned, unsigned, \
	                                                   unsigned, unsigned, void *, void **, void **)) \
	X(int,          hipExtModuleLaunchKernel,         (void *, unsigned, unsigned, unsigned, unsigned, unsigned, \
	                                                   unsigned, size_t, void *, void **, void **, void *, void *, \
	                                                   unsigned)) \
	X(int,          hipModuleUnload,                  (void *)) \
	X(char const *, hipGetErrorName,                  (int))

// The entry points this file defines, each as the real runtime's.
#define DECLARE(ret, name, params) extern ret name params;
HIPTRACE_FUNCTIONS(DECLARE)
#undef DECLARE

/** @brief The real runtime: its library, and each entry point, null for one it does not export. */
struct real {
	void *dll; //!< The library, or nullptr.
#define FIELD(ret, name, params) ret (*name) params;
	HIPTRACE_FUNCTIONS(FIELD)
#undef FIELD
};

/** @brief hipModuleLaunchKernel's markers in its extra list, as HIP defines them. */
enum launch_param : uintptr_t {
	LAUNCH_PARAM_BUFFER_POINTER = 1, //!< HIP_LAUNCH_PARAM_BUFFER_POINTER: the next item is the arguments' buffer.
	LAUNCH_PARAM_BUFFER_SIZE    = 2, //!< HIP_LAUNCH_PARAM_BUFFER_SIZE: the next item points to its size.
	LAUNCH_PARAM_END            = 3, //!< HIP_LAUNCH_PARAM_END: the list's end.
};

// ---- msgpack, as much as the AMDGPU metadata note uses ------------------------------------------

/** @brief A MessagePack value's type. */
enum msg_type : uint32_t {
	MSG_NIL,   //!< nil.
	MSG_BOOL,  //!< A boolean.
	MSG_UINT,  //!< A nonnegative integer.
	MSG_INT,   //!< A negative integer, or one of a signed encoding.
	MSG_FLOAT, //!< A float32 or float64.
	MSG_STR,   //!< A string.
	MSG_BIN,   //!< Binary bytes.
	MSG_ARRAY, //!< An array.
	MSG_MAP,   //!< A map.
};

/** @brief Where the next MessagePack value is read, and where the bytes end. */
struct msg_reader {
	uint8_t const *p;   //!< The next byte.
	uint8_t const *end; //!< The end of the bytes.
};

/** @brief A MessagePack value that msg_head() or msg_read() read: a number, a string's bytes, or
 *         where a sequence's items are.
 */
struct msg {
	uint64_t       number; //!< An integer, an MSG_INT's bits as uint64_t; 0 for any other type.
	uint8_t const *at;     //!< A string's or binary's bytes, or a sequence's first item.
	uint8_t const *end;    //!< The end of the bytes that the sequence's items are read from.
	size_t         n;      //!< A string's or binary's bytes, an array's items or a map's keys and values.
	enum msg_type  type;   //!< Its type.
	unsigned       depth;  //!< The depth a sequence was read at.
};

/** @brief The bytes of a string that a map does not have. */
static uint8_t const NO_BYTES[1];

/** @brief A big-endian number from a reader that has its bytes.
 *
 * @param r The reader, which moves past the number.
 * @param w The number's bytes, at most 8.
 * @return  The number.
 */
static uint64_t
msg_be (struct msg_reader *r,
        unsigned           w)
{
	uint64_t v = 0;
	for (unsigned k = 0; k < w; ++k)
		v = v << 8 | r->p[k];
	r->p += w;
	return v;
}

/** @brief Whether a reader has bytes left.
 *
 * @param r The reader.
 * @param n The bytes needed.
 * @return  true if at least @a n bytes are left.
 */
static bool
msg_need (struct msg_reader const *r,
          uint64_t                 n)
{
	return (size_t)(r->end - r->p) >= n;
}

/** @brief Reads a string's or binary's bytes.
 *
 * @param r    The reader, which moves past them.
 * @param out  Receives the value.
 * @param type MSG_STR or MSG_BIN.
 * @param n    The bytes.
 * @return     false if fewer are left.
 */
static bool
msg_bytes (struct msg_reader *r,
           struct msg        *out,
           enum msg_type      type,
           uint64_t           n)
{
	if (!msg_need(r, n))
		return false;
	out->type = type;
	out->at = r->p;
	out->n = n;
	r->p += n;
	return true;
}

/** @brief Reads an array's or map's head: its count, which the bytes left must allow.
 *
 * @param r       The reader, which moves to the first item.
 * @param out     Receives the value.
 * @param type    MSG_ARRAY or MSG_MAP.
 * @param entries The array's items, or the map's keys.
 * @param depth   The sequence's depth.
 * @return        false if fewer bytes are left than items.
 */
static bool
msg_sequence (struct msg_reader *r,
              struct msg        *out,
              enum msg_type      type,
              uint64_t           entries,
              unsigned           depth)
{
	out->type = type;
	uint64_t const count = type == MSG_MAP ? entries * 2 : entries;
	// Every item takes at least one byte.
	if (!msg_need(r, count))
		return false;
	out->at = r->p;
	out->end = r->end;
	out->n = count;
	out->depth = depth;
	return true;
}

/** @brief Reads a MessagePack value's head: all of a scalar, a string's or binary's bytes, or an
 *         array's or map's count; the reader is then at the sequence's first item.
 *
 * @param r     The reader, which moves past the head.
 * @param out   Receives the value.
 * @param depth Its depth, at most MSG_DEPTH.
 * @return      false if it is cut short, too deep or of an ext type, which AMDGPU metadata has not.
 */
static bool
msg_head (struct msg_reader *r,
          struct msg        *out,
          unsigned           depth)
{
	if (depth > MSG_DEPTH || !msg_need(r, 1))
		return false;
	*out = (struct msg){};
	uint8_t const c = *r->p++;
	if (c <= 0x7f) {
		out->type = MSG_UINT;
		out->number = c;
		return true;
	}
	if (c >= 0xe0) {
		out->type = MSG_INT;
		out->number = (uint64_t)(int64_t)(int8_t)c;
		return true;
	}
	if ((c & 0xf0) == 0x80)
		return msg_sequence(r, out, MSG_MAP, c & 15, depth);
	if ((c & 0xf0) == 0x90)
		return msg_sequence(r, out, MSG_ARRAY, c & 15, depth);
	if ((c & 0xe0) == 0xa0)
		return msg_bytes(r, out, MSG_STR, c & 31);
	switch (c) {
	case 0xc0:
		out->type = MSG_NIL;
		return true;
	case 0xc2: case 0xc3:
		out->type = MSG_BOOL;
		return true;
	case 0xc4: case 0xc5: case 0xc6: {
		unsigned const w = 1u << (c - 0xc4);
		return msg_need(r, w) && msg_bytes(r, out, MSG_BIN, msg_be(r, w));
	}
	case 0xca: case 0xcb: {
		// Its value is not used.
		unsigned const w = c == 0xca ? 4 : 8;
		if (!msg_need(r, w))
			return false;
		r->p += w;
		out->type = MSG_FLOAT;
		return true;
	}
	case 0xcc: case 0xcd: case 0xce: case 0xcf: {
		unsigned const w = 1u << (c - 0xcc);
		if (!msg_need(r, w))
			return false;
		out->type = MSG_UINT;
		out->number = msg_be(r, w);
		return true;
	}
	case 0xd0: case 0xd1: case 0xd2: case 0xd3: {
		unsigned const w = 1u << (c - 0xd0);
		if (!msg_need(r, w))
			return false;
		uint64_t const v = msg_be(r, w);
		out->type = MSG_INT;
		// Sign-extended from its width.
		out->number = w == 1 ? (uint64_t)(int64_t)(int8_t)v
		            : w == 2 ? (uint64_t)(int64_t)(int16_t)v
		            : w == 4 ? (uint64_t)(int64_t)(int32_t)v
		            : v;
		return true;
	}
	case 0xd9: case 0xda: case 0xdb: {
		unsigned const w = 1u << (c - 0xd9);
		return msg_need(r, w) && msg_bytes(r, out, MSG_STR, msg_be(r, w));
	}
	case 0xdc: case 0xdd: {
		unsigned const w = c == 0xdc ? 2 : 4;
		return msg_need(r, w) && msg_sequence(r, out, MSG_ARRAY, msg_be(r, w), depth);
	}
	case 0xde: case 0xdf: {
		unsigned const w = c == 0xde ? 2 : 4;
		return msg_need(r, w) && msg_sequence(r, out, MSG_MAP, msg_be(r, w), depth);
	}
	default:
		// ext types: not in AMDGPU metadata.
		return false;
	}
}

static bool
msg_read (struct msg_reader *r,
          struct msg        *out,
          unsigned           depth);

/** @brief Reads what follows a value's head: an array's or map's items, each of which must read.
 *
 * @param r     The reader, at the first item of a sequence; it moves past the items.
 * @param value The value, which msg_head() read.
 * @return      false if an item does not read.
 */
static bool
msg_rest (struct msg_reader *r,
          struct msg const  *value)
{
	if (value->type != MSG_ARRAY && value->type != MSG_MAP)
		return true;
	for (size_t k = 0; k < value->n; ++k) {
		struct msg item;
		if (!msg_read(r, &item, value->depth + 1))
			return false;
	}
	return true;
}

/** @brief Reads a MessagePack value, and checks all of it: an array's or map's items too.
 *
 * @param r     The reader, which moves past the value.
 * @param out   Receives the value.
 * @param depth Its depth, at most MSG_DEPTH.
 * @return      false if it or an item is cut short, too deep or of an ext type.
 */
static bool
msg_read (struct msg_reader *r,
          struct msg        *out,
          unsigned           depth)
{
	return msg_head(r, out, depth) && msg_rest(r, out);
}

/** @brief The items of an array or map: a reader at the first.
 *
 * @param sequence The array or map.
 * @return         The reader.
 */
static struct msg_reader
msg_items (struct msg const *sequence)
{
	return (struct msg_reader){sequence->at, sequence->end};
}

/** @brief A map's value for a key: that of its first string key with the key's bytes.
 *
 * @param map    The value to look in, which msg_read() read; any other type than a map has no keys.
 * @param key    The key.
 * @param length The key's length.
 * @param dest   Receives the value's head (msg_head()): a sequence's items are read from
 *               msg_items().
 * @return       false if the map has no such key.
 */
static bool
msg_get (struct msg const *map,
         char const       *key,
         size_t            length,
         struct msg       *dest)
{
	if (map->type != MSG_MAP)
		return false;
	struct msg_reader items = msg_items(map);
	for (size_t k = 0; k + 1 < map->n; k += 2) {
		struct msg name;
		if (!msg_read(&items, &name, map->depth + 1))
			return false;
		if (name.type == MSG_STR && name.n == length && !memcmp(name.at, key, length))
			return msg_head(&items, dest, map->depth + 1);
		if (!msg_read(&items, dest, map->depth + 1))
			return false;
	}
	return false;
}

// ---- code objects -------------------------------------------------------------------------------

/** @brief A kernel argument's kind; its value is the letter that a function's layout shows. */
enum arg_kind : uint32_t {
	ARG_VALUE   = 'v', //!< Any other kind: a value.
	ARG_POINTER = 'p', //!< global_buffer or dynamic_shared_pointer.
	ARG_HIDDEN  = 'h', //!< hidden_*: the runtime's.
};

/** @brief A kernel argument as the metadata describes it. */
struct arg {
	uint32_t      offset; //!< .offset.
	uint32_t      size;   //!< .size.
	enum arg_kind kind;   //!< From .value_kind.
};

/** @brief The numbers of a kernel's metadata that a function's log line shows, in its order. */
enum kernel_number : uint8_t {
	KERNEL_KERNARG,    //!< .kernarg_segment_size.
	KERNEL_LDS,        //!< .group_segment_fixed_size.
	KERNEL_SCRATCH,    //!< .private_segment_fixed_size.
	KERNEL_VGPR,       //!< .vgpr_count.
	KERNEL_SGPR,       //!< .sgpr_count.
	KERNEL_VGPR_SPILL, //!< .vgpr_spill_count.
	KERNEL_SGPR_SPILL, //!< .sgpr_spill_count.
	KERNEL_WAVE,       //!< .wavefront_size.
	KERNEL_MAX_GROUP,  //!< .max_flat_workgroup_size.
	KERNEL_NUMBER_COUNT
};

/** @brief A metadata key of a kernel's number and its name in the log. */
struct kernel_key {
	char const *key;    //!< The metadata key.
	char const *label;  //!< The log's name for it.
	size_t      length; //!< The key's length.
};

/** @brief struct kernel_key of a string literal key. */
#define KEY(key, label) {key, label, sizeof key - 1}

/** @brief Each number's key and label, by enum kernel_number. */
static struct kernel_key const KERNEL_KEYS[] = {
	[KERNEL_KERNARG]    = KEY(".kernarg_segment_size", "kernarg"),
	[KERNEL_LDS]        = KEY(".group_segment_fixed_size", "lds"),
	[KERNEL_SCRATCH]    = KEY(".private_segment_fixed_size", "scratch"),
	[KERNEL_VGPR]       = KEY(".vgpr_count", "vgpr"),
	[KERNEL_SGPR]       = KEY(".sgpr_count", "sgpr"),
	[KERNEL_VGPR_SPILL] = KEY(".vgpr_spill_count", "vgpr_spill"),
	[KERNEL_SGPR_SPILL] = KEY(".sgpr_spill_count", "sgpr_spill"),
	[KERNEL_WAVE]       = KEY(".wavefront_size", "wave"),
	[KERNEL_MAX_GROUP]  = KEY(".max_flat_workgroup_size", "max_group"),
};

#undef KEY

static_assert(sizeof KERNEL_KEYS / sizeof *KERNEL_KEYS == KERNEL_NUMBER_COUNT, "a key per number");

/** @brief A kernel that a code object's metadata describes. */
struct kernel {
	uint64_t    numbers[KERNEL_NUMBER_COUNT]; //!< Its numbers, by enum kernel_number.
	char       *name;                         //!< .name, on the heap.
	struct arg *args;                         //!< .args, explicit ones first as the metadata lists them.
	uint32_t    name_length;                  //!< The name's length: a MessagePack string's fits.
	uint32_t    arg_count;                    //!< The arguments: a MessagePack array's fit.
	uint32_t    explicit_args;                //!< Those that are not hidden.
	uint32_t    deep;                         //!< Whether HIPTRACE_DEEP_KERNELS lists it. 32 bits
	                                          //!< wide, which fills the padding.
};

// ---- state --------------------------------------------------------------------------------------

/** @brief realloc() for the trace's own memory: running out ends the process, since a trace must
 *         not lose a call.
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
		fputs("hiptrace: out of memory\n", stderr);
		abort();
	}
	return grown;
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

/** @brief Joins two strings on the heap with a character between them.
 *
 * @param a         The first string.
 * @param a_length  Its length.
 * @param separator The character between them.
 * @param b         The second string.
 * @param b_length  Its length.
 * @return          The string, which the caller frees.
 */
static char *
join (char const *a,
      size_t      a_length,
      char        separator,
      char const *b,
      size_t      b_length)
{
	char *const text = reallocate(nullptr, a_length + 1 + b_length + 1);
	memcpy(text, a, a_length);
	text[a_length] = separator;
	memcpy(text + a_length + 1, b, b_length);
	text[a_length + 1 + b_length] = '\0';
	return text;
}

/** @brief An allocation the trace names: what made it, its ordinal and its size. The last two bytes
 *         are padding that no member needs.
 */
struct range {
	size_t   size;     //!< Its bytes.
	unsigned id;       //!< Its ordinal among its kind's, from 1.
	char     kind;     //!< d: hipMalloc, h: hipHostMalloc, r: hipHostRegister, x: an external memory's
	                   //!< mapped buffer.
	bool     uploaded; //!< The destination of a host-to-device copy (weights, maps, constants).
};

/** @brief A module the trace knows. */
struct module {
	uint64_t       id;           //!< Its ordinal, from 1; 64 bits wide, as an unsigned would leave
	                             //!< padding.
	struct kernel *kernels;      //!< Its kernels, sorted by name, or nullptr.
	char          *file;         //!< The module file's name, on the heap.
	size_t         file_length;  //!< The name's length.
	size_t         kernel_count; //!< Its kernels.
};

/** @brief A function the trace knows. */
struct function {
	uint64_t             id;     //!< Its ordinal, from 1; 64 bits wide, as an unsigned would leave
	                             //!< padding.
	void                *module; //!< The module it came from.
	struct kernel const *kernel; //!< Its metadata in the module, or nullptr.
	char                *label;  //!< file:kernel, on the heap.
};

/** @brief A module file's name. */
struct file_name {
	char   *text;   //!< The name, on the heap.
	size_t  length; //!< Its length.
};

/** @brief HIPTRACE_DEEP's range of launch ordinals: [begin, end). */
struct deep_range {
	uint64_t begin; //!< The first ordinal.
	uint64_t end;   //!< The ordinal after the last.
};

/** @brief A name that HIPTRACE_DEEP_KERNELS lists. */
struct deep_kernel {
	char const *name;   //!< The name, not null-terminated.
	size_t      length; //!< Its length.
};

SUPPORT_SORTED_MAP(ranges, struct range, reallocate)
SUPPORT_SORTED_MAP(ids, unsigned, reallocate)
SUPPORT_SORTED_MAP(modules, struct module, reallocate)
SUPPORT_SORTED_MAP(functions, struct function, reallocate)
SUPPORT_SORTED_MAP(files, struct file_name, reallocate)

/** @brief The trace. */
struct state {
	uint64_t            sequence;                 //!< The next log line's number.
	uint64_t            launches;                 //!< The launches so far.
	uint64_t            threads;                  //!< The threads that logged so far.
	pthread_mutex_t     lock;                     //!< Held by every reader and writer of the rest.
	struct real         real;                     //!< The real runtime.
	struct ranges       ranges;                   //!< Every allocation, by its base address.
	struct ids          streams;                  //!< Each stream's id, by handle.
	struct ids          events;                   //!< Each event's id, by handle.
	struct ids          memories;                 //!< Each external memory's id, by handle.
	struct modules      modules;                  //!< Each module, by handle.
	struct functions    functions;                //!< Each function, by handle.
	struct files        files;                    //!< Each module file's name, by FNV-1a 64 of its bytes.
	FILE               *log;                      //!< HIPTRACE_FILE, or nullptr.
	struct deep_range  *deep_ranges;              //!< HIPTRACE_DEEP's ranges, or nullptr.
	struct deep_kernel *deep_kernels;             //!< HIPTRACE_DEEP_KERNELS's names, each once, or nullptr.
	char               *deep_kernel_list;         //!< A copy of HIPTRACE_DEEP_KERNELS, which they point into.
	size_t              deep_range_count;         //!< The ranges.
	size_t              deep_kernel_count;        //!< The names.
	unsigned            counters[UCHAR_MAX + 1];  //!< The ordinals given so far, by kind letter.
};

/** @brief The one trace. It is never torn down: the runtime's threads may call in during exit. */
static struct state state = {.lock = PTHREAD_MUTEX_INITIALIZER};

/** @brief This thread's ordinal in the log plus 1, or 0 until it logs. */
static thread_local uint64_t thread_number;

/** @brief Takes the trace's lock. A mutex of PTHREAD_MUTEX_INITIALIZER's default type fails no lock
 *         or unlock of a caller that holds it as it should.
 *
 * @return The trace.
 */
static struct state *
lock (void)
{
	pthread_mutex_lock(&state.lock);
	return &state;
}

/** @brief Releases the trace's lock.
 *
 * @param s The trace.
 */
static void
unlock (struct state *s)
{
	pthread_mutex_unlock(&s->lock);
}

/** @brief The FNV-1a 64 of bytes.
 *
 * @param data The bytes.
 * @param size How many.
 * @return     Their hash.
 */
static uint64_t
fnv (void const *data,
     size_t      size)
{
	uint8_t const *const p = data;
	uint64_t h = FNV_BASIS;
	for (size_t i = 0; i < size; ++i)
		h = (h ^ p[i]) * FNV_PRIME;
	return h;
}

/** @brief A pointer or handle as a map's key.
 *
 * @param p The pointer.
 * @return  Its address.
 */
static uint64_t
key_of (void const *p)
{
	static_assert(sizeof (uintptr_t) <= sizeof (uint64_t), "an address fits a key");
	return (uintptr_t)p;
}

/** @brief Starts a log line with its number and its thread's ordinal, if there is a log. The lock is
 *         held, as for every function that takes the trace.
 *
 * @param s The trace.
 * @return  The log, or nullptr when there is none.
 */
static FILE *
line (struct state *s)
{
	if (!s->log)
		return nullptr;
	if (!thread_number)
		thread_number = ++s->threads;
	fprintf(s->log, "%" PRIu64 " t%" PRIu64 " ", s->sequence++, thread_number - 1);
	return s->log;
}

/** @brief Logs a line, if there is a log.
 *
 * @param s   The trace.
 * @param fmt A printf format for what follows the line's number and thread.
 * @param ... The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
emit (struct state *s,
      char const   *fmt,
      ...)
{
	FILE *const log = line(s);
	if (!log)
		return;
	va_list args;
	va_start(args, fmt);
	vfprintf(log, fmt, args);
	va_end(args);
	fputc('\n', log);
}

/** @brief Writes out what the log holds, so that a crash after a synchronous call loses no line.
 *
 * @param s The trace.
 */
static void
flush (struct state *s)
{
	// A failure sets the stream's error indicator, which finish() reports.
	if (s->log)
		fflush(s->log);
}

/** @brief A short name in the trace: an allocation and offset, a handle's name, "null" or "?". */
struct name {
	char text[NAME_BYTES]; //!< The name.
};

/** @brief Writes a number's decimal digits, with no null after them.
 *
 * @param at Where they go, with room for DECIMAL_DIGITS.
 * @param v  The number.
 * @return   Where they end.
 */
static char *
decimal (char     *at,
         uint64_t  v)
{
	char digits[DECIMAL_DIGITS];
	size_t first = sizeof digits;
	do
		digits[--first] = "0123456789"[v % 10];
	while (v /= 10);
	size_t const n = sizeof digits - first;
	memcpy(at, &digits[first], n);
	return at + n;
}

/** @brief The name of an object by its kind letter and ordinal.
 *
 * @param kind The letter.
 * @param id   The ordinal.
 * @return     The name, such as "m3".
 */
static struct name
numbered (char     kind,
          uint64_t id)
{
	struct name name;
	name.text[0] = kind;
	*decimal(&name.text[1], id) = '\0';
	return name;
}

/** @brief The next ordinal of a kind.
 *
 * @param s    The trace.
 * @param kind The kind's letter.
 * @return     The ordinal, from 1.
 */
static unsigned
next_id (struct state *s,
         char          kind)
{
	return ++s->counters[(unsigned char)kind];
}

/** @brief The allocation that holds an address: the last whose base is not above it, if the address
 *         is below its end.
 *
 * @param ranges  The allocations.
 * @param address The address.
 * @return        Its index, or the allocations' count when none holds the address.
 */
static size_t
range_at (struct ranges const *ranges,
          uint64_t             address)
{
	size_t const above = support_keys_upper_bound(ranges->keys, ranges->count, address);
	if (!above)
		return ranges->count;
	size_t const at = above - 1;
	return address - ranges->keys[at] < ranges->values[at].size ? at : ranges->count;
}

/** @brief A pointer's name: its allocation and the offset in it, "null", or "?" and its address.
 *
 * @param s       The trace.
 * @param pointer The pointer.
 * @return        The name, such as "d3+64".
 */
static struct name
where (struct state const *s,
       void const         *pointer)
{
	uint64_t const address = key_of(pointer);
	if (!address)
		return (struct name){"null"};
	struct name name;
	size_t const at = range_at(&s->ranges, address);
	if (at < s->ranges.count) {
		struct range const *const r = &s->ranges.values[at];
		name.text[0] = r->kind;
		char *const plus = decimal(&name.text[1], r->id);
		*plus = '+';
		*decimal(plus + 1, address - s->ranges.keys[at]) = '\0';
	} else {
		snprintf(name.text, sizeof name.text, "?%" PRIx64, address);
	}
	return name;
}

/** @brief A copy's side: unregistered host memory has no stable name.
 *
 * @param s       The trace.
 * @param pointer The side's pointer.
 * @return        Its name, or "host" for one that no allocation holds.
 */
static struct name
side (struct state const *s,
      void const         *pointer)
{
	struct name const name = where(s, pointer);
	return name.text[0] == '?' ? (struct name){"host"} : name;
}

/** @brief A handle's name, for a handle that is not null.
 *
 * @param ids    The handles of its kind.
 * @param kind   The kind's letter.
 * @param handle The handle.
 * @return       The kind's letter and the handle's ordinal, or "?" for a handle the trace does not
 *               know.
 */
static struct name
id_name (struct ids const *ids,
         char              kind,
         void const       *handle)
{
	bool found;
	size_t const at = support_keys_find(ids->keys, ids->count, key_of(handle), &found);
	return found ? numbered(kind, ids->values[at]) : (struct name){"?"};
}

/** @brief A handle's name: "null" for the null handle.
 *
 * @param ids    The handles of its kind.
 * @param kind   The kind's letter.
 * @param handle The handle.
 * @return       The name.
 */
static struct name
handle_name (struct ids const *ids,
             char              kind,
             void const       *handle)
{
	return handle ? id_name(ids, kind, handle) : (struct name){"null"};
}

/** @brief A stream's name: "s0" for the null stream.
 *
 * @param s      The trace.
 * @param stream The stream.
 * @return       The name.
 */
static struct name
stream_name (struct state const *s,
             void const         *stream)
{
	return stream ? id_name(&s->streams, 's', stream) : (struct name){"s0"};
}

/** @brief Names a new handle: its kind's next ordinal.
 *
 * @param s      The trace.
 * @param ids    The handles of its kind.
 * @param kind   The kind's letter.
 * @param handle The handle.
 * @return       The ordinal.
 */
static unsigned
add_id (struct state *s,
        struct ids   *ids,
        char          kind,
        void const   *handle)
{
	unsigned const id = next_id(s, kind);
	bool found;
	size_t const at = support_keys_find(ids->keys, ids->count, key_of(handle), &found);
	*(found ? &ids->values[at] : ids_insert(ids, at, key_of(handle))) = id;
	return id;
}

/** @brief Forgets a handle, if the trace knows it.
 *
 * @param ids    The handles of its kind.
 * @param handle The handle.
 */
static void
drop_id (struct ids *ids,
         void const *handle)
{
	bool found;
	size_t const at = support_keys_find(ids->keys, ids->count, key_of(handle), &found);
	if (found)
		ids_erase(ids, at);
}

/** @brief Records an allocation under its kind's next ordinal; one at the same base address is
 *         replaced.
 *
 * @param s       The trace.
 * @param kind    The kind's letter.
 * @param pointer Its base address.
 * @param size    Its bytes.
 * @return        The ordinal.
 */
static unsigned
add_range (struct state *s,
           char          kind,
           void const   *pointer,
           size_t        size)
{
	unsigned const id = next_id(s, kind);
	struct range const range = {.size = size, .id = id, .kind = kind};
	bool found;
	size_t const at = support_keys_find(s->ranges.keys, s->ranges.count, key_of(pointer), &found);
	*(found ? &s->ranges.values[at] : ranges_insert(&s->ranges, at, key_of(pointer))) = range;
	return id;
}

/** @brief Logs a release and forgets the allocation that starts at the pointer.
 *
 * @param s       The trace.
 * @param pointer The pointer released.
 * @param call    The call's name.
 */
static void
drop_range (struct state *s,
            void const   *pointer,
            char const   *call)
{
	struct name const name = where(s, pointer);
	bool found;
	size_t const at = support_keys_find(s->ranges.keys, s->ranges.count, key_of(pointer), &found);
	size_t size = 0;
	if (found) {
		size = s->ranges.values[at].size;
		ranges_erase(&s->ranges, at);
	}
	emit(s, "%s %s bytes=%zu", call, name.text, size);
}

/** @brief hipMemcpyKind's name.
 *
 * @param kind The kind.
 * @return     Its name, or "?" for no kind.
 */
static char const *
kind_name (int kind)
{
	return kind >= 0 && kind < COPY_KIND_COUNT ? COPY_KINDS[kind] : "?";
}

/** @brief Whether an address is device memory this trace knows of.
 *
 * @param s       The trace.
 * @param pointer The address.
 * @return        true if a hipMalloc() or a mapped buffer holds it.
 */
static bool
device_memory (struct state const *s,
               void const         *pointer)
{
	size_t const at = range_at(&s->ranges, key_of(pointer));
	if (at == s->ranges.count)
		return false;
	char const kind = s->ranges.values[at].kind;
	return kind == 'd' || kind == 'x';
}

/** @brief Marks the allocation that holds an address as a host-to-device copy's destination.
 *
 * @param s       The trace.
 * @param pointer The address.
 */
static void
mark_uploaded (struct state *s,
               void const   *pointer)
{
	size_t const at = range_at(&s->ranges, key_of(pointer));
	if (at < s->ranges.count)
		s->ranges.values[at].uploaded = true;
}

/** @brief Whether HIPTRACE_DEEP_KERNELS lists a name.
 *
 * @param s      The trace.
 * @param name   The name.
 * @param length Its length.
 * @return       true if it does.
 */
static bool
deep_listed (struct state const *s,
             char const         *name,
             size_t              length)
{
	for (size_t i = 0; i < s->deep_kernel_count; ++i)
		if (s->deep_kernels[i].length == length && !memcmp(s->deep_kernels[i].name, name, length))
			return true;
	return false;
}

/** @brief How two names order, as std::string compares them: byte by byte, then the shorter first.
 *
 * @param a        A name.
 * @param a_length Its length.
 * @param b        Another name.
 * @param b_length Its length.
 * @return         Less than, equal to or greater than 0 as @a a orders before, with or after @a b.
 */
static int
name_order (char const *a,
            size_t      a_length,
            char const *b,
            size_t      b_length)
{
	int const order = memcmp(a, b, a_length < b_length ? a_length : b_length);
	if (order)
		return order;
	return (a_length > b_length) - (a_length < b_length);
}

/** @brief Where a kernel is in a module's kernels, by name, or where it goes.
 *
 * @param kernels The kernels, sorted by name.
 * @param count   How many.
 * @param name    The name.
 * @param length  Its length.
 * @param found   Receives whether the kernel is there.
 * @return        Its index, or the index it goes to.
 */
static size_t
kernel_find (struct kernel const *kernels,
             size_t               count,
             char const          *name,
             size_t               length,
             bool                *found)
{
	size_t low = 0;
	size_t high = count;
	while (low < high) {
		size_t const middle = low + (high - low) / 2;
		int const order = name_order(kernels[middle].name, kernels[middle].name_length, name, length);
		if (!order) {
			*found = true;
			return middle;
		}
		if (order < 0)
			low = middle + 1;
		else
			high = middle;
	}
	*found = false;
	return low;
}

/** @brief The image's extent from its headers.
 *
 * @param image The image; its headers are trusted.
 * @return      The extent, or 0 when it is not a 64-bit ELF.
 */
static size_t
elf_size (uint8_t const *image)
{
	Elf64_Ehdr eh;
	memcpy(&eh, image, sizeof eh);
	if (memcmp(eh.e_ident, ELFMAG, SELFMAG) || eh.e_ident[EI_CLASS] != ELFCLASS64)
		return 0;
	size_t size = eh.e_ehsize;
	size_t const programs = eh.e_phoff + (size_t)eh.e_phnum * eh.e_phentsize;
	size_t const sections = eh.e_shoff + (size_t)eh.e_shnum * eh.e_shentsize;
	size = size < programs ? programs : size;
	size = size < sections ? sections : size;
	for (size_t k = 0; k < eh.e_phnum; ++k) {
		Elf64_Phdr ph;
		memcpy(&ph, image + eh.e_phoff + k * eh.e_phentsize, sizeof ph);
		size_t const end = ph.p_offset + ph.p_filesz;
		size = size < end ? end : size;
	}
	for (size_t k = 0; k < eh.e_shnum; ++k) {
		Elf64_Shdr sh;
		memcpy(&sh, image + eh.e_shoff + k * eh.e_shentsize, sizeof sh);
		size_t const end = sh.sh_offset + sh.sh_size;
		if (sh.sh_type != SHT_NOBITS)
			size = size < end ? end : size;
	}
	return size;
}

/** @brief Marks a key of a map seen: the first value of a key counts, as msg_get() finds it.
 *
 * @param seen The keys seen so far, a bit each.
 * @param bit  The key's bit.
 * @return     true if the key was not seen before.
 */
static bool
first_seen (uint32_t *seen,
            unsigned  bit)
{
	uint32_t const mask = UINT32_C(1) << bit;
	bool const first = !(*seen & mask);
	*seen |= mask;
	return first;
}

/** @brief Whether a MessagePack string has a string literal's bytes. */
#define TEXT_IS(text, literal) ((text).n == sizeof literal - 1 && !memcmp((text).at, literal, sizeof literal - 1))

/** @brief Whether a MessagePack string begins with a string literal's bytes. */
#define TEXT_STARTS(text, literal) ((text).n >= sizeof literal - 1 && !memcmp((text).at, literal, sizeof literal - 1))

/** @brief The bits of the keys that read_arg() reads, in its first_seen() mask. */
enum arg_key_bit : uint8_t {
	ARG_KEY_KIND,   //!< .value_kind.
	ARG_KEY_OFFSET, //!< .offset.
	ARG_KEY_SIZE,   //!< .size.
};

/** @brief Reads a kernel argument's map, or any other value in its place, in one pass.
 *
 * @param r     The reader, at the value, which a checked note holds; it moves past it.
 * @param depth The value's depth.
 * @param dest  Receives the argument: a value argument of no bytes at offset 0, unless a map's
 *              .value_kind, .offset and .size tell otherwise.
 * @return      false if the value does not read.
 */
static bool
read_arg (struct msg_reader *r,
          unsigned           depth,
          struct arg        *dest)
{
	struct msg a;
	if (!msg_head(r, &a, depth))
		return false;
	*dest = (struct arg){.kind = ARG_VALUE};
	if (a.type != MSG_MAP)
		return msg_rest(r, &a);
	uint32_t seen = 0;
	for (size_t k = 0; k < a.n; k += 2) {
		struct msg key;
		struct msg value;
		if (!msg_read(r, &key, depth + 1) || !msg_read(r, &value, depth + 1))
			return false;
		if (key.type != MSG_STR)
			continue;
		if (TEXT_IS(key, ".value_kind")) {
			if (first_seen(&seen, ARG_KEY_KIND) && value.type == MSG_STR)
				dest->kind = TEXT_STARTS(value, "hidden_") ? ARG_HIDDEN
				           : TEXT_IS(value, "global_buffer") || TEXT_IS(value, "dynamic_shared_pointer")
				           ? ARG_POINTER
				           : ARG_VALUE;
		} else if (TEXT_IS(key, ".offset")) {
			if (first_seen(&seen, ARG_KEY_OFFSET))
				dest->offset = (uint32_t)value.number;
		} else if (TEXT_IS(key, ".size")) {
			if (first_seen(&seen, ARG_KEY_SIZE))
				dest->size = (uint32_t)value.number;
		}
	}
	return true;
}

#undef TEXT_STARTS

/** @brief Reads a kernel's .args value: an array of arguments, or any other value, which gives none.
 *
 * @param r      The reader, at the value, which a checked note holds; it moves past it.
 * @param depth  The value's depth.
 * @param kernel Receives the arguments, on the heap.
 * @return       false if the value does not read.
 */
static bool
read_args (struct msg_reader *r,
           unsigned           depth,
           struct kernel     *kernel)
{
	struct msg args;
	if (!msg_head(r, &args, depth))
		return false;
	if (args.type != MSG_ARRAY || !args.n)
		return msg_rest(r, &args);
	kernel->args = reallocate(nullptr, args.n * sizeof *kernel->args);
	for (; kernel->arg_count < args.n; ++kernel->arg_count) {
		struct arg *const arg = &kernel->args[kernel->arg_count];
		if (!read_arg(r, depth + 1, arg))
			return false;
		kernel->explicit_args += arg->kind != ARG_HIDDEN;
	}
	return true;
}

/** @brief The bits of the keys that read_kernel() reads, in its first_seen() mask: a number's is its
 *         enum kernel_number, and these follow.
 */
enum kernel_key_bit : uint8_t {
	KERNEL_KEY_NAME = KERNEL_NUMBER_COUNT, //!< .name.
	KERNEL_KEY_ARGS,                       //!< .args.
};

static_assert(KERNEL_KEY_ARGS < 32, "a bit per key in a uint32_t");

/** @brief Reads a kernel's map, or any other value in its place, in one pass.
 *
 * @param r      The reader, at the value, which a checked note holds; it moves past it.
 * @param depth  The value's depth.
 * @param kernel Receives the map's numbers and arguments, the arguments on the heap, also when the
 *               value does not read; its name is not set.
 * @param name   Receives .name's string; it is left as it is without one.
 * @return       false if the value does not read.
 */
static bool
read_kernel (struct msg_reader *r,
             unsigned           depth,
             struct kernel     *kernel,
             struct msg        *name)
{
	struct msg m;
	if (!msg_head(r, &m, depth))
		return false;
	if (m.type != MSG_MAP)
		return msg_rest(r, &m);
	uint32_t seen = 0;
	for (size_t k = 0; k < m.n; k += 2) {
		struct msg key;
		if (!msg_read(r, &key, depth + 1))
			return false;
		bool const text = key.type == MSG_STR;
		if (text && TEXT_IS(key, ".args") && first_seen(&seen, KERNEL_KEY_ARGS)) {
			if (!read_args(r, depth + 1, kernel))
				return false;
			continue;
		}
		struct msg value;
		if (!msg_read(r, &value, depth + 1))
			return false;
		if (!text)
			continue;
		if (TEXT_IS(key, ".name")) {
			if (first_seen(&seen, KERNEL_KEY_NAME) && value.type == MSG_STR)
				*name = value;
			continue;
		}
		for (unsigned i = 0; i < KERNEL_NUMBER_COUNT; ++i) {
			if (key.n == KERNEL_KEYS[i].length && !memcmp(key.at, KERNEL_KEYS[i].key, key.n)) {
				if (first_seen(&seen, i))
					kernel->numbers[i] = value.number;
				break;
			}
		}
	}
	return true;
}

#undef TEXT_IS

/** @brief Adds the kernel of a metadata list's item to a module's, unless the module has one of its
 *         name: the first of a name stays.
 *
 * @param s      The trace.
 * @param module The module, whose kernels have room for one more.
 * @param r      The reader, at the item, which a checked note holds; it moves past it.
 * @param depth  The item's depth.
 * @return       false if the item does not read.
 */
static bool
add_kernel (struct state const *s,
            struct module      *module,
            struct msg_reader  *r,
            unsigned            depth)
{
	struct kernel kernel = {};
	struct msg name = {.at = NO_BYTES};
	if (!read_kernel(r, depth, &kernel, &name)) {
		free(kernel.args);
		kernel.args = nullptr;
		return false;
	}
	char const *const text = (char const *)name.at;
	bool found;
	size_t const at = kernel_find(module->kernels, module->kernel_count, text, name.n, &found);
	if (found) {
		free(kernel.args);
		kernel.args = nullptr;
		return true;
	}
	kernel.name = copy_text(text, name.n);
	kernel.name_length = (uint32_t)name.n;
	kernel.deep = deep_listed(s, text, name.n);
	memmove(&module->kernels[at + 1], &module->kernels[at], (module->kernel_count - at) * sizeof *module->kernels);
	module->kernels[at] = kernel;
	++module->kernel_count;
	return true;
}

/** @brief Reads the kernels that the image's NT_AMDGPU_METADATA notes describe into a module.
 *
 * @param s      The trace.
 * @param module The module, with no kernels yet.
 * @param image  The image, a 64-bit ELF; its headers are trusted.
 */
static void
add_kernels (struct state const *s,
             struct module      *module,
             uint8_t const      *image)
{
	Elf64_Ehdr eh;
	memcpy(&eh, image, sizeof eh);
	for (size_t k = 0; k < eh.e_shnum; ++k) {
		Elf64_Shdr sh;
		memcpy(&sh, image + eh.e_shoff + k * eh.e_shentsize, sizeof sh);
		if (sh.sh_type != SHT_NOTE)
			continue;
		uint8_t const *const notes = image + sh.sh_offset;
		for (size_t at = 0; at + sizeof (Elf64_Nhdr) <= sh.sh_size;) {
			Elf64_Nhdr note;
			memcpy(&note, notes + at, sizeof note);
			size_t const name_at = at + sizeof note;
			size_t const desc_at = name_at + ((note.n_namesz + 3) & ~3u);
			at = desc_at + ((note.n_descsz + 3) & ~3u);
			if (at > sh.sh_size)
				break;
			if (note.n_type != NOTE_AMDGPU_METADATA || note.n_namesz != sizeof NOTE_AMDGPU
			    || memcmp(notes + name_at, NOTE_AMDGPU, sizeof NOTE_AMDGPU))
				continue;
			// The whole note is checked first: one that does not read adds no kernel.
			struct msg_reader reader = {notes + desc_at, notes + desc_at + note.n_descsz};
			struct msg root;
			struct msg list;
			if (!msg_read(&reader, &root, 0) || !msg_get(&root, AMDHSA_KERNELS, sizeof AMDHSA_KERNELS - 1, &list)
			    || list.type != MSG_ARRAY || !list.n)
				continue;
			// Room for every kernel of the list; those of names already there are not added.
			size_t const room = module->kernel_count + list.n;
			module->kernels = reallocate(module->kernels, room * sizeof *module->kernels);
			struct msg_reader items = msg_items(&list);
			for (size_t i = 0; i < list.n; ++i)
				if (!add_kernel(s, module, &items, list.depth + 1))
					break;
		}
	}
}

/** @brief Frees what a module holds.
 *
 * @param module The module.
 */
static void
module_fini (struct module *module)
{
	for (size_t k = 0; k < module->kernel_count; ++k) {
		free(module->kernels[k].name);
		module->kernels[k].name = nullptr;
		free(module->kernels[k].args);
		module->kernels[k].args = nullptr;
	}
	free(module->kernels);
	module->kernels = nullptr;
	free(module->file);
	module->file = nullptr;
	module->file_length = 0;
	module->kernel_count = 0;
}

/** @brief Records a module under the next ordinal and logs it: its file, by the hash of its image, and
 *         its kernels. One of the same handle is replaced.
 *
 * @param s      The trace.
 * @param handle The module's handle.
 * @param image  Its image.
 * @param call   The call's name.
 * @param path   The file it was loaded from, or nullptr.
 */
static void
add_module (struct state  *s,
            void          *handle,
            uint8_t const *image,
            char const    *call,
            char const    *path)
{
	unsigned const id = next_id(s, 'm');
	size_t const size = elf_size(image);
	uint64_t const hash = size ? fnv(image, size) : 0;
	bool known_file;
	size_t const file_at = support_keys_find(s->files.keys, s->files.count, hash, &known_file);
	char const *file = "?";
	size_t file_length = sizeof "?" - 1;
	if (known_file) {
		file = s->files.values[file_at].text;
		file_length = s->files.values[file_at].length;
	} else if (path) {
		file = path;
		file_length = strlen(path);
	}
	char *const copy = copy_text(file, file_length);
	struct module module = {.file = copy, .file_length = file_length, .id = id};
	if (size)
		add_kernels(s, &module, image);
	emit(s, "%s m%u file=%s bytes=%zu fnv=%016" PRIx64 " kernels=%zu", call, id, module.file, size, hash,
	     module.kernel_count);

	bool found;
	size_t const at = support_keys_find(s->modules.keys, s->modules.count, key_of(handle), &found);
	if (found) {
		module_fini(&s->modules.values[at]);
		s->modules.values[at] = module;
	} else {
		*modules_insert(&s->modules, at, key_of(handle)) = module;
	}
}

/** @brief A module the trace knows.
 *
 * @param s      The trace.
 * @param handle Its handle.
 * @return       The module, until the modules change, or nullptr.
 */
static struct module *
module_of (struct state *s,
           void const   *handle)
{
	bool found;
	size_t const at = support_keys_find(s->modules.keys, s->modules.count, key_of(handle), &found);
	return found ? &s->modules.values[at] : nullptr;
}

/** @brief Reads a whole file.
 *
 * @param path    The file.
 * @param minimum The bytes to allocate at least; those past the file's end are zero.
 * @param size    Receives the file's size.
 * @return        Its bytes, which the caller frees, or nullptr if it cannot be read.
 */
static uint8_t *
read_file (char const *path,
           size_t      minimum,
           size_t     *size)
{
	FILE *const file = fopen(path, "rbe");
	if (!file)
		return nullptr;
	size_t capacity = 65536;
	size_t used = 0;
	uint8_t *bytes = reallocate(nullptr, capacity);
	for (;;) {
		used += fread(bytes + used, 1, capacity - used, file);
		if (used < capacity)
			break;
		capacity *= 2;
		bytes = reallocate(bytes, capacity);
	}
	bool const failed = ferror(file);
	if (fclose(file) || failed) {
		free(bytes);
		bytes = nullptr;
		return nullptr;
	}
	if (capacity < minimum)
		bytes = reallocate(bytes, minimum);
	if (used < minimum)
		memset(bytes + used, 0, minimum - used);
	*size = used;
	return bytes;
}

/** @brief Names the module files in a directory, each by the hash of its bytes; a later file of the
 *         same hash names it instead.
 *
 * @param s         The trace.
 * @param directory The directory, or nullptr.
 */
static void
load_module_files (struct state *s,
                   char const   *directory)
{
	if (!directory || !*directory)
		return;
	DIR *const dir = opendir(directory);
	if (!dir)
		return;
	size_t const directory_length = strlen(directory);
	for (;;) {
		// readdir() reports an error only through errno.
		errno = 0;
		struct dirent const *const entry = readdir(dir);
		if (!entry) {
			if (errno)
				fprintf(stderr, "hiptrace: cannot read %s: %s\n", directory, strerror(errno));
			break;
		}
		size_t const length = strlen(entry->d_name);
		if (length < HSACO_LENGTH || memcmp(entry->d_name + length - HSACO_LENGTH, HSACO, HSACO_LENGTH))
			continue;
		char *path = join(directory, directory_length, '/', entry->d_name, length);
		size_t size;
		uint8_t *bytes = read_file(path, 0, &size);
		free(path);
		path = nullptr;
		if (!bytes)
			continue;
		uint64_t const hash = fnv(bytes, size);
		free(bytes);
		bytes = nullptr;
		char *const copy = copy_text(entry->d_name, length);
		struct file_name const name = {copy, length};
		bool found;
		size_t const at = support_keys_find(s->files.keys, s->files.count, hash, &found);
		if (found) {
			// The old name is replaced at once.
			free(s->files.values[at].text);
			s->files.values[at] = name;
		} else {
			*files_insert(&s->files, at, hash) = name;
		}
	}
	if (closedir(dir))
		fprintf(stderr, "hiptrace: cannot close %s: %s\n", directory, strerror(errno));
}

/** @brief Reads a decimal number of HIPTRACE_DEEP.
 *
 * @param text Where it starts.
 * @param end  Receives where it ends.
 * @param dest Receives the number.
 * @return     false if @a text starts with no digit or the number is above 2^64 - 1.
 */
static bool
parse_ordinal (char const  *text,
               char       **end,
               uint64_t    *dest)
{
	if (*text < '0' || *text > '9')
		return false;
	// strtoull() reports a number out of range only through errno.
	errno = 0;
	unsigned long long const value = strtoull(text, end, 10);
	if (errno)
		return false;
	*dest = value;
	return true;
}

/** @brief Reads HIPTRACE_DEEP's launch ordinals as [begin, end) ranges into the trace.
 *
 * @param s    The trace, with no ranges yet.
 * @param text "all", or comma-separated "A-B" ranges and "A" ordinals.
 * @return     false if @a text is neither; the trace then has no ranges.
 */
static bool
parse_deep (struct state *s,
            char const   *text)
{
	if (!strcmp(text, "all")) {
		s->deep_ranges = reallocate(nullptr, sizeof *s->deep_ranges);
		s->deep_ranges[0] = (struct deep_range){0, UINT64_MAX};
		s->deep_range_count = 1;
		return true;
	}
	// A range per comma, and one more.
	size_t count = 1;
	for (char const *at = text; (at = strchr(at, ',')); ++at)
		++count;
	s->deep_ranges = reallocate(nullptr, count * sizeof *s->deep_ranges);
	for (char const *at = text;; ++at) {
		char *end;
		uint64_t begin;
		if (!parse_ordinal(at, &end, &begin))
			break;
		uint64_t stop = begin + 1;
		if (*end == '-' && !parse_ordinal(end + 1, &end, &stop))
			break;
		s->deep_ranges[s->deep_range_count++] = (struct deep_range){begin, stop};
		if (!*end)
			return true;
		if (*end != ',')
			break;
		at = end;
	}
	free(s->deep_ranges);
	s->deep_ranges = nullptr;
	s->deep_range_count = 0;
	return false;
}

/** @brief Reads HIPTRACE_DEEP_KERNELS's comma-separated names into the trace, each once.
 *
 * @param s    The trace, with no names yet.
 * @param list The names.
 */
static void
parse_deep_kernels (struct state *s,
                    char const   *list)
{
	size_t const length = strlen(list);
	s->deep_kernel_list = copy_text(list, length);
	char const *const stop = s->deep_kernel_list + length;
	// A name per comma, and one more.
	size_t count = 1;
	for (char const *at = s->deep_kernel_list; (at = memchr(at, ',', (size_t)(stop - at))); ++at)
		++count;
	s->deep_kernels = reallocate(nullptr, count * sizeof *s->deep_kernels);
	for (char const *name = s->deep_kernel_list; name < stop;) {
		char const *comma = memchr(name, ',', (size_t)(stop - name));
		if (!comma)
			comma = stop;
		size_t const n = (size_t)(comma - name);
		if (n && !deep_listed(s, name, n))
			s->deep_kernels[s->deep_kernel_count++] = (struct deep_kernel){name, n};
		name = comma + 1;
	}
}

/** @brief Whether a launch ordinal is in one of HIPTRACE_DEEP's ranges.
 *
 * @param s       The trace.
 * @param ordinal The ordinal.
 * @return        true if it is.
 */
static bool
deep_ordinal (struct state const *s,
              uint64_t            ordinal)
{
	for (size_t i = 0; i < s->deep_range_count; ++i)
		if (ordinal >= s->deep_ranges[i].begin && ordinal < s->deep_ranges[i].end)
			return true;
	return false;
}

/** @brief Loads the real runtime, names the module files and reads the settings; then logs them. */
[[gnu::constructor]]
static void
initialize (void)
{
	struct state *const s = &state;
	// HIPTRACE_REAL, else backend/hip.c's candidates.
	static char const *const candidates[] = {
		"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so.7",
		"/opt/rocm/lib/libamdhip64.so.6", "/opt/rocm/lib/libamdhip64.so",
	};
	char const *const chosen = getenv("HIPTRACE_REAL");
	char const *real = nullptr;
	if (chosen && *chosen) {
		if ((s->real.dll = dlopen(chosen, RTLD_NOW | RTLD_LOCAL)))
			real = chosen;
	} else {
		for (size_t i = 0; i < sizeof candidates / sizeof *candidates && !real; ++i)
			if ((s->real.dll = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL)))
				real = candidates[i];
	}
	if (s->real.dll) {
		// POSIX makes the address dlsym() gives for a function a valid pointer to it.
#define LOAD(ret, name, params) \
		{ \
			void *const symbol = dlsym(s->real.dll, #name); \
			static_assert(sizeof s->real.name == sizeof symbol, "a function pointer is a pointer"); \
			memcpy(&s->real.name, &symbol, sizeof symbol); \
		}
		HIPTRACE_FUNCTIONS(LOAD)
#undef LOAD
	} else {
		fprintf(stderr, "hiptrace: no real HIP runtime: %s\n", dlerror());
	}

	char const *modules = getenv("HIPTRACE_MODULES");
	if (!modules || !*modules)
		modules = getenv("DLSSLOP_MODULES");
	load_module_files(s, modules);
	char const *const deep = getenv("HIPTRACE_DEEP");
	bool const deep_invalid = deep && *deep && !parse_deep(s, deep);
	if (deep_invalid)
		fprintf(stderr, "hiptrace: HIPTRACE_DEEP=%s is not all or comma-separated A-B ranges and A ordinals up "
		        "to 2^64 - 1; no launch is hashed by ordinal\n", deep);
	char const *const list = getenv("HIPTRACE_DEEP_KERNELS");
	if (list && *list)
		parse_deep_kernels(s, list);
	char const *const path = getenv("HIPTRACE_FILE");
	if (path && *path) {
		s->log = fopen(path, "we");
		// Without the bigger buffer, the log is only slower.
		if (!s->log)
			fprintf(stderr, "hiptrace: cannot write %s\n", path);
		else
			setvbuf(s->log, nullptr, _IOFBF, 1 << 20);
	}

	lock();
	FILE *const log = line(s);
	if (log) {
		fprintf(log, "trace real=%s module_files=%zu modules_dir=%s deep=", real ? real : "none", s->files.count,
		        modules ? modules : "");
		if (deep_invalid)
			fputs("invalid", log);
		else if (!s->deep_range_count)
			fputs("none", log);
		for (size_t i = 0; i < s->deep_range_count; ++i)
			fprintf(log, "%s%" PRIu64 "-%" PRIu64, i ? "," : "", s->deep_ranges[i].begin, s->deep_ranges[i].end);
		fprintf(log, " deep_kernels=%zu missing=", s->deep_kernel_count);
		bool missing = false;
#define MISSING(ret, name, params) \
		if (s->real.dll && !s->real.name) { \
			fputs(" " #name, log); \
			missing = true; \
		}
		HIPTRACE_FUNCTIONS(MISSING)
#undef MISSING
		fputs(missing ? "\n" : "none\n", log);
		fflush(log);
	}
	unlock(s);
}

#undef HIPTRACE_FUNCTIONS

/** @brief Logs the launch count at exit and reports a log that could not be written. */
[[gnu::destructor]]
static void
finish (void)
{
	struct state *const s = lock();
	emit(s, "exit launches=%" PRIu64, s->launches);
	if (s->log && (fflush(s->log) || ferror(s->log)))
		fputs("hiptrace: writing HIPTRACE_FILE failed; the trace is incomplete\n", stderr);
	unlock(s);
}

/** @brief Logs each kernel argument as the runtime copies it: pointers by their names, other values
 *         of 4 bytes in decimal and of any other size in hex.
 *
 * @param s      The trace.
 * @param log    The log.
 * @param kernel The kernel.
 * @param params Its explicit arguments' values.
 */
static void
log_arguments (struct state const  *s,
               FILE                *log,
               struct kernel const *kernel,
               void *const         *params)
{
	uint32_t index = 0;
	for (uint32_t k = 0; k < kernel->arg_count; ++k) {
		struct arg const *const arg = &kernel->args[k];
		if (arg->kind == ARG_HIDDEN)
			continue;
		if (index)
			fputc(',', log);
		uint8_t const *const value = params[index++];
		if (arg->kind == ARG_POINTER && arg->size == 8) {
			void *pointer;
			memcpy(&pointer, value, sizeof pointer);
			fputs(where(s, pointer).text, log);
		} else if (arg->size == 4) {
			uint32_t v;
			memcpy(&v, value, sizeof v);
			fprintf(log, "%" PRIu32, v);
		} else {
			// Little-endian bytes, the most significant first.
			fputs("0x", log);
			for (uint32_t b = arg->size; b--;)
				fprintf(log, "%02x", value[b]);
		}
	}
}

/** @brief After a deep launch: waits for the stream, then logs the FNV of every buffer argument, from
 *         the argument to the end of its allocation. The real runtime is called directly.
 *
 * @param s      The trace.
 * @param log    The log, or nullptr: the stream is waited for and the buffers read all the same.
 * @param kernel The kernel.
 * @param params Its explicit arguments' values.
 * @param stream The launch's stream.
 */
static void
log_deep_hashes (struct state        *s,
                 FILE                *log,
                 struct kernel const *kernel,
                 void *const         *params,
                 void                *stream)
{
	int const rc = s->real.hipStreamSynchronize(stream);
	if (rc) {
		if (log)
			fprintf(log, "sync_rc=%d", rc);
		return;
	}
	uint8_t *copy = nullptr;
	size_t capacity = 0;
	uint32_t index = 0;
	for (uint32_t k = 0; k < kernel->arg_count; ++k) {
		struct arg const *const arg = &kernel->args[k];
		if (arg->kind == ARG_HIDDEN)
			continue;
		uint32_t const position = index++;
		if (arg->kind != ARG_POINTER || arg->size != 8)
			continue;
		void *pointer;
		memcpy(&pointer, params[position], sizeof pointer);
		uint64_t const address = key_of(pointer);
		size_t const at = range_at(&s->ranges, address);
		if (at == s->ranges.count)
			continue;
		struct range const *const r = &s->ranges.values[at];
		size_t const bytes = r->size - (address - s->ranges.keys[at]);
		if (r->uploaded) {
			if (log)
				fprintf(log, " a%" PRIu32 "=uploaded", position);
			continue;
		}
		void const *data = pointer;
		if (r->kind == 'd' || r->kind == 'x') {
			if (capacity < bytes) {
				copy = reallocate(copy, bytes);
				capacity = bytes;
			}
			int const e = s->real.hipMemcpy(copy, pointer, bytes, 2);
			if (e) {
				if (log)
					fprintf(log, " a%" PRIu32 "=copy_rc%d", position, e);
				continue;
			}
			data = copy;
		}
		uint64_t const hash = fnv(data, bytes);
		if (log)
			fprintf(log, " a%" PRIu32 "=%zu:%016" PRIx64, position, bytes, hash);
	}
	free(copy);
	copy = nullptr;
}

/** @brief Logs a launch, decoding its arguments with its kernel's metadata, and after a deep one the
 *         hashes of its buffers.
 *
 * @param call   The log's name for the call.
 * @param f      The function.
 * @param gx     The grid's groups in x.
 * @param gy     In y.
 * @param gz     In z.
 * @param bx     The block's threads in x.
 * @param by     In y.
 * @param bz     In z.
 * @param shared The dynamic LDS bytes.
 * @param stream The stream.
 * @param params The explicit arguments' values, or nullptr.
 * @param extra  The arguments' buffer as HIP_LAUNCH_PARAM items, or nullptr.
 * @param rc     The real runtime's result.
 * @return       @a rc.
 */
static int
launch (char const *call,
        void       *f,
        unsigned    gx,
        unsigned    gy,
        unsigned    gz,
        unsigned    bx,
        unsigned    by,
        unsigned    bz,
        size_t      shared,
        void       *stream,
        void      **params,
        void      **extra,
        int         rc)
{
	struct state *const s = lock();
	uint64_t const ordinal = s->launches++;
	bool found;
	size_t const at = support_keys_find(s->functions.keys, s->functions.count, key_of(f), &found);
	struct function const *const fn = found ? &s->functions.values[at] : nullptr;
	struct kernel const *const kernel = fn ? fn->kernel : nullptr;
	bool const deep = !rc && kernel && params && (deep_ordinal(s, ordinal) || kernel->deep);
	FILE *const log = line(s);
	if (log) {
		fprintf(log, "%s n=%" PRIu64 " f=%s k=%s grid=%ux%ux%u block=%ux%ux%u lds=%zu s=%s args=", call, ordinal,
		        fn ? numbered('f', fn->id).text : "?", fn ? fn->label : "?", gx, gy, gz, bx, by, bz, shared,
		        stream_name(s, stream).text);
		if (kernel && params) {
			log_arguments(s, log, kernel, params);
		} else if (extra) {
			uint8_t const *buffer = nullptr;
			size_t size = 0;
			for (void **e = extra;; e += 2) {
				uintptr_t const marker = (uintptr_t)e[0];
				if (marker == LAUNCH_PARAM_END)
					break;
				if (marker == LAUNCH_PARAM_BUFFER_POINTER)
					buffer = e[1];
				if (marker == LAUNCH_PARAM_BUFFER_SIZE)
					memcpy(&size, e[1], sizeof size);
			}
			fputs("buffer:", log);
			for (size_t k = 0; buffer && k < size; ++k)
				fprintf(log, "%02x", buffer[k]);
		} else {
			fputc('?', log);
		}
		fprintf(log, " rc=%d%s", rc, deep ? " deep" : "");
	}
	if (deep)
		log_deep_hashes(s, log, kernel, params, stream);
	if (log)
		fputc('\n', log);
	unlock(s);
	return rc;
}

// ---- exports ------------------------------------------------------------------------------------

/** @brief Calls the real runtime's entry point, or answers hipErrorNotFound when it lacks one. */
#define REAL(name, ...) (state.real.name ? state.real.name(__VA_ARGS__) : HIP_ERROR_NOT_FOUND)

/** @brief hipGetErrorName: the real runtime's name, or "hipErrorNotFound" without one. */
char const *
hipGetErrorName (int error)
{
	return state.real.hipGetErrorName ? state.real.hipGetErrorName(error) : "hipErrorNotFound";
}

/** @brief hipInit, logged. */
int
hipInit (unsigned flags)
{
	int const rc = REAL(hipInit, flags);
	struct state *const s = lock();
	emit(s, "hipInit flags=%u rc=%d", flags, rc);
	unlock(s);
	return rc;
}

/** @brief hipRuntimeGetVersion, logged with the version. */
int
hipRuntimeGetVersion (int *version)
{
	int const rc = REAL(hipRuntimeGetVersion, version);
	struct state *const s = lock();
	emit(s, "hipRuntimeGetVersion version=%d rc=%d", rc ? 0 : *version, rc);
	unlock(s);
	return rc;
}

/** @brief hipGetDeviceCount, logged with the count. */
int
hipGetDeviceCount (int *count)
{
	int const rc = REAL(hipGetDeviceCount, count);
	struct state *const s = lock();
	emit(s, "hipGetDeviceCount count=%d rc=%d", rc ? 0 : *count, rc);
	unlock(s);
	return rc;
}

/** @brief hipGetDevicePropertiesR0600, logged with the device's name and architecture. */
int
hipGetDevicePropertiesR0600 (struct hip_device_properties *p,
                             int                           device)
{
	int const rc = REAL(hipGetDevicePropertiesR0600, p, device);
	struct state *const s = lock();
	emit(s, "hipGetDevicePropertiesR0600 device=%d name=\"%s\" arch=%s rc=%d", device, rc ? "" : p->name,
	     rc ? "" : p->gcnArchName, rc);
	unlock(s);
	return rc;
}

/** @brief hipDeviceGetName, logged. */
int
hipDeviceGetName (char *name,
                  int   length,
                  int   device)
{
	int const rc = REAL(hipDeviceGetName, name, length, device);
	struct state *const s = lock();
	emit(s, "hipDeviceGetName device=%d rc=%d", device, rc);
	unlock(s);
	return rc;
}

/** @brief hipSetDevice, logged. */
int
hipSetDevice (int device)
{
	int const rc = REAL(hipSetDevice, device);
	struct state *const s = lock();
	emit(s, "hipSetDevice device=%d rc=%d", device, rc);
	unlock(s);
	return rc;
}

/** @brief hipSetDeviceFlags, logged. */
int
hipSetDeviceFlags (unsigned flags)
{
	int const rc = REAL(hipSetDeviceFlags, flags);
	struct state *const s = lock();
	emit(s, "hipSetDeviceFlags flags=%u rc=%d", flags, rc);
	unlock(s);
	return rc;
}

/** @brief hipMemGetInfo, logged. */
int
hipMemGetInfo (size_t *available,
               size_t *total)
{
	int const rc = REAL(hipMemGetInfo, available, total);
	struct state *const s = lock();
	emit(s, "hipMemGetInfo rc=%d", rc);
	unlock(s);
	return rc;
}

/** @brief hipMalloc: the allocation is named dN. */
int
hipMalloc (void   **pointer,
           size_t   size)
{
	int const rc = REAL(hipMalloc, pointer, size);
	struct state *const s = lock();
	if (rc) {
		emit(s, "hipMalloc bytes=%zu rc=%d", size, rc);
	} else {
		unsigned const id = add_range(s, 'd', *pointer, size);
		emit(s, "hipMalloc d%u bytes=%zu flags=0", id, size);
	}
	unlock(s);
	return rc;
}

/** @brief hipFree, logged before the real call. */
int
hipFree (void *pointer)
{
	struct state *const s = lock();
	drop_range(s, pointer, "hipFree");
	unlock(s);
	return REAL(hipFree, pointer);
}

/** @brief hipHostMalloc: the allocation is named hN. */
int
hipHostMalloc (void     **pointer,
               size_t     size,
               unsigned   flags)
{
	int const rc = REAL(hipHostMalloc, pointer, size, flags);
	struct state *const s = lock();
	if (rc) {
		emit(s, "hipHostMalloc bytes=%zu flags=%u rc=%d", size, flags, rc);
	} else {
		unsigned const id = add_range(s, 'h', *pointer, size);
		emit(s, "hipHostMalloc h%u bytes=%zu flags=%u", id, size, flags);
	}
	unlock(s);
	return rc;
}

/** @brief hipHostFree, logged before the real call. */
int
hipHostFree (void *pointer)
{
	struct state *const s = lock();
	drop_range(s, pointer, "hipHostFree");
	unlock(s);
	return REAL(hipHostFree, pointer);
}

/** @brief hipHostRegister: the registered memory is named rN. */
int
hipHostRegister (void     *pointer,
                 size_t    size,
                 unsigned  flags)
{
	int const rc = REAL(hipHostRegister, pointer, size, flags);
	struct state *const s = lock();
	if (rc) {
		emit(s, "hipHostRegister bytes=%zu flags=%u rc=%d", size, flags, rc);
	} else {
		unsigned const id = add_range(s, 'r', pointer, size);
		emit(s, "hipHostRegister r%u bytes=%zu flags=%u", id, size, flags);
	}
	unlock(s);
	return rc;
}

/** @brief hipHostUnregister, logged before the real call. */
int
hipHostUnregister (void *pointer)
{
	struct state *const s = lock();
	drop_range(s, pointer, "hipHostUnregister");
	unlock(s);
	return REAL(hipHostUnregister, pointer);
}

/** @brief hipMemcpy, logged with the FNV of the payload: the source's from the host, which marks the
 *         destination uploaded, or the destination's to the host.
 */
int
hipMemcpy (void       *dst,
           void const *src,
           size_t      size,
           int         kind)
{
	struct state *s = lock();
	struct name const where_dst = side(s, dst);
	struct name const where_src = side(s, src);
	bool const from_host = kind == 1 || (kind == 4 && !device_memory(s, src) && device_memory(s, dst));
	uint64_t payload = 0;
	if (from_host) {
		payload = fnv(src, size);
		mark_uploaded(s, dst);
	}
	unlock(s);
	int const rc = REAL(hipMemcpy, dst, src, size, kind);
	s = lock();
	bool const to_host = kind == 2 || (kind == 4 && !device_memory(s, dst));
	if (!rc && to_host)
		payload = fnv(dst, size);
	emit(s, "hipMemcpy kind=%s dst=%s src=%s bytes=%zu %s=%016" PRIx64 " rc=%d", kind_name(kind), where_dst.text,
	     where_src.text, size, from_host ? "src_fnv" : to_host ? "dst_fnv" : "fnv",
	     from_host || to_host ? payload : 0, rc);
	flush(s);
	unlock(s);
	return rc;
}

/** @brief hipMemcpyAsync, logged with the FNV of a payload from the host, which marks the destination
 *         uploaded.
 */
int
hipMemcpyAsync (void       *dst,
                void const *src,
                size_t      size,
                int         kind,
                void       *stream)
{
	struct state *s = lock();
	struct name const where_dst = side(s, dst);
	struct name const where_src = side(s, src);
	bool const from_host = kind == 1 || (kind == 4 && !device_memory(s, src) && device_memory(s, dst));
	uint64_t payload = 0;
	if (from_host) {
		payload = fnv(src, size);
		mark_uploaded(s, dst);
	}
	unlock(s);
	int const rc = REAL(hipMemcpyAsync, dst, src, size, kind, stream);
	s = lock();
	if (from_host)
		emit(s, "hipMemcpyAsync kind=%s dst=%s src=%s bytes=%zu s=%s src_fnv=%016" PRIx64 " rc=%d",
		     kind_name(kind), where_dst.text, where_src.text, size, stream_name(s, stream).text, payload, rc);
	else
		emit(s, "hipMemcpyAsync kind=%s dst=%s src=%s bytes=%zu s=%s rc=%d", kind_name(kind), where_dst.text,
		     where_src.text, size, stream_name(s, stream).text, rc);
	unlock(s);
	return rc;
}

/** @brief hipMemsetAsync, logged. */
int
hipMemsetAsync (void   *dst,
                int     value,
                size_t  size,
                void   *stream)
{
	int const rc = REAL(hipMemsetAsync, dst, value, size, stream);
	struct state *const s = lock();
	emit(s, "hipMemsetAsync dst=%s value=%d bytes=%zu s=%s rc=%d", where(s, dst).text, value, size,
	     stream_name(s, stream).text, rc);
	unlock(s);
	return rc;
}

/** @brief hipEventCreate: the event is named eN. */
int
hipEventCreate (void **event)
{
	int const rc = REAL(hipEventCreate, event);
	struct state *const s = lock();
	if (rc) {
		emit(s, "hipEventCreate ? rc=%d", rc);
	} else {
		unsigned const id = add_id(s, &s->events, 'e', *event);
		emit(s, "hipEventCreate e%u rc=%d", id, rc);
	}
	unlock(s);
	return rc;
}

/** @brief hipEventRecord, logged. */
int
hipEventRecord (void *event,
                void *stream)
{
	int const rc = REAL(hipEventRecord, event, stream);
	struct state *const s = lock();
	emit(s, "hipEventRecord %s s=%s rc=%d", handle_name(&s->events, 'e', event).text,
	     stream_name(s, stream).text, rc);
	unlock(s);
	return rc;
}

/** @brief hipEventElapsedTime, logged without the time. */
int
hipEventElapsedTime (float *ms,
                     void  *begin,
                     void  *end)
{
	int const rc = REAL(hipEventElapsedTime, ms, begin, end);
	struct state *const s = lock();
	emit(s, "hipEventElapsedTime %s %s rc=%d", handle_name(&s->events, 'e', begin).text,
	     handle_name(&s->events, 'e', end).text, rc);
	unlock(s);
	return rc;
}

/** @brief hipEventDestroy, logged before the real call. */
int
hipEventDestroy (void *event)
{
	struct state *const s = lock();
	emit(s, "hipEventDestroy %s", handle_name(&s->events, 'e', event).text);
	drop_id(&s->events, event);
	unlock(s);
	return REAL(hipEventDestroy, event);
}

/** @brief hipEventSynchronize, logged, and the log written out. */
int
hipEventSynchronize (void *event)
{
	int const rc = REAL(hipEventSynchronize, event);
	struct state *const s = lock();
	emit(s, "hipEventSynchronize %s rc=%d", handle_name(&s->events, 'e', event).text, rc);
	flush(s);
	unlock(s);
	return rc;
}

/** @brief hipDeviceSynchronize, logged, and the log written out. */
int
hipDeviceSynchronize (void)
{
	int const rc = REAL(hipDeviceSynchronize);
	struct state *const s = lock();
	emit(s, "hipDeviceSynchronize rc=%d", rc);
	flush(s);
	unlock(s);
	return rc;
}

/** @brief hipStreamCreate: the stream is named sN. */
int
hipStreamCreate (void **stream)
{
	int const rc = REAL(hipStreamCreate, stream);
	struct state *const s = lock();
	if (rc) {
		emit(s, "hipStreamCreate ? rc=%d", rc);
	} else {
		unsigned const id = add_id(s, &s->streams, 's', *stream);
		emit(s, "hipStreamCreate s%u rc=%d", id, rc);
	}
	unlock(s);
	return rc;
}

/** @brief hipStreamSynchronize, logged, and the log written out. */
int
hipStreamSynchronize (void *stream)
{
	int const rc = REAL(hipStreamSynchronize, stream);
	struct state *const s = lock();
	emit(s, "hipStreamSynchronize %s rc=%d", stream_name(s, stream).text, rc);
	flush(s);
	unlock(s);
	return rc;
}

/** @brief hipStreamDestroy, logged before the real call. */
int
hipStreamDestroy (void *stream)
{
	struct state *const s = lock();
	emit(s, "hipStreamDestroy %s", stream_name(s, stream).text);
	drop_id(&s->streams, stream);
	unlock(s);
	return REAL(hipStreamDestroy, stream);
}

/** @brief hipImportExternalMemory: the memory is named XN. */
int
hipImportExternalMemory (void                         **memory,
                         struct hip_memory_desc const  *d)
{
	int const rc = REAL(hipImportExternalMemory, memory, d);
	struct state *const s = lock();
	struct name name = {"?"};
	if (!rc) {
		unsigned const id = add_id(s, &s->memories, 'X', *memory);
		name = numbered('X', id);
	}
	emit(s, "hipImportExternalMemory %s type=%d bytes=%llu flags=%u rc=%d", name.text, d->type, d->size, d->flags,
	     rc);
	unlock(s);
	return rc;
}

/** @brief hipExternalMemoryGetMappedBuffer: the buffer is named xN. */
int
hipExternalMemoryGetMappedBuffer (void                         **pointer,
                                  void                          *memory,
                                  struct hip_buffer_desc const  *d)
{
	int const rc = REAL(hipExternalMemoryGetMappedBuffer, pointer, memory, d);
	struct state *const s = lock();
	struct name const of = handle_name(&s->memories, 'X', memory);
	if (rc) {
		emit(s, "hipExternalMemoryGetMappedBuffer of=%s rc=%d", of.text, rc);
	} else {
		size_t const size = d->size;
		unsigned const id = add_range(s, 'x', *pointer, size);
		emit(s, "hipExternalMemoryGetMappedBuffer of=%s offset=%llu x%u bytes=%zu flags=%u", of.text, d->offset,
		     id, size, d->flags);
	}
	unlock(s);
	return rc;
}

/** @brief hipDestroyExternalMemory, logged before the real call. */
int
hipDestroyExternalMemory (void *memory)
{
	struct state *const s = lock();
	emit(s, "hipDestroyExternalMemory %s", handle_name(&s->memories, 'X', memory).text);
	drop_id(&s->memories, memory);
	unlock(s);
	return REAL(hipDestroyExternalMemory, memory);
}

/** @brief Defines an entry point that upstream's host code resolved but never called in dlsslopd:
 *         forwarded, with a log line.
 */
#define PLAIN(name, params, args) \
	int \
	name params \
	{ \
		int const rc = REAL(name, UNPAREN args); \
		struct state *const s = lock(); \
		emit(s, #name " rc=%d", rc); \
		unlock(s); \
		return rc; \
	}
#define UNPAREN(...) __VA_ARGS__
PLAIN(hipImportExternalSemaphore, (void **a, void const *b), (a, b))
PLAIN(hipSignalExternalSemaphoresAsync, (void *const *a, void const *b, unsigned c, void *d), (a, b, c, d))
PLAIN(hipWaitExternalSemaphoresAsync, (void *const *a, void const *b, unsigned c, void *d), (a, b, c, d))
PLAIN(hipDestroyExternalSemaphore, (void *a), (a))
PLAIN(hipStreamBeginCapture, (void *a, int b), (a, b))
PLAIN(hipStreamEndCapture, (void *a, void **b), (a, b))
PLAIN(hipGraphInstantiate, (void **a, void *b, void **c, char *d, size_t e), (a, b, c, d, e))
PLAIN(hipGraphLaunch, (void *a, void *b), (a, b))
PLAIN(hipGraphDestroy, (void *a), (a))
PLAIN(hipGraphExecDestroy, (void *a), (a))
PLAIN(hipMemAddressReserve, (void **a, size_t b, size_t c, void *d, unsigned long long e), (a, b, c, d, e))
PLAIN(hipMemAddressFree, (void *a, size_t b), (a, b))
PLAIN(hipMemCreate, (void **a, size_t b, void const *c, unsigned long long d), (a, b, c, d))
PLAIN(hipMemRelease, (void *a), (a))
PLAIN(hipMemMap, (void *a, size_t b, size_t c, void *d, unsigned long long e), (a, b, c, d, e))
PLAIN(hipMemUnmap, (void *a, size_t b), (a, b))
PLAIN(hipMemSetAccess, (void *a, size_t b, void const *c, size_t d), (a, b, c, d))
PLAIN(hipMemGetAllocationGranularity, (size_t *a, void const *b, unsigned c), (a, b, c))
#undef UNPAREN
#undef PLAIN

/** @brief hipModuleLoadData: the module is named mN, its file by the hash of its image. */
int
hipModuleLoadData (void       **module,
                   void const  *image)
{
	int const rc = REAL(hipModuleLoadData, module, image);
	struct state *const s = lock();
	if (rc)
		emit(s, "hipModuleLoadData rc=%d", rc);
	else
		add_module(s, *module, image, "hipModuleLoadData", nullptr);
	unlock(s);
	return rc;
}

/** @brief hipModuleLoad: the module is named mN, its file by the hash of the file's bytes, read
 *         again here; a file that cannot be read is an empty image.
 */
int
hipModuleLoad (void       **module,
               char const  *path)
{
	int const rc = REAL(hipModuleLoad, module, path);
	struct state *const s = lock();
	if (rc || !path) {
		emit(s, "hipModuleLoad path=%s rc=%d", path ? path : "null", rc);
		unlock(s);
		return rc;
	}
	static uint8_t const empty[sizeof (Elf64_Ehdr)];
	size_t size;
	uint8_t *bytes = read_file(path, sizeof empty, &size);
	add_module(s, *module, bytes ? bytes : empty, "hipModuleLoad", path);
	free(bytes);
	bytes = nullptr;
	unlock(s);
	return rc;
}

/** @brief Logs a function that hipModuleGetFunction() gave, with its kernel's metadata.
 *
 * @param log    The log, at a line's start.
 * @param f      The function.
 * @param module Its module's name.
 */
static void
log_function (FILE                  *log,
              struct function const *f,
              char const            *module)
{
	struct kernel const *const k = f->kernel;
	if (!k) {
		fprintf(log, "hipModuleGetFunction f%" PRIu64 " %s k=%s metadata=none rc=0\n", f->id, module, f->label);
		return;
	}
	fprintf(log, "hipModuleGetFunction f%" PRIu64 " %s k=%s args=%" PRIu32, f->id, module, f->label,
	        k->explicit_args);
	for (size_t i = 0; i < KERNEL_NUMBER_COUNT; ++i)
		fprintf(log, " %s=%" PRIu64, KERNEL_KEYS[i].label, k->numbers[i]);
	fputs(" layout=", log);
	for (uint32_t i = 0; i < k->arg_count; ++i)
		fprintf(log, "%s%c%" PRIu32 "@%" PRIu32, i ? "," : "", k->args[i].kind, k->args[i].size, k->args[i].offset);
	fputs(" rc=0\n", log);
}

/** @brief hipModuleGetFunction: the function is named fN, or keeps the name it had, and is logged
 *         with its kernel's metadata.
 */
int
hipModuleGetFunction (void       **function,
                      void        *module,
                      char const  *name)
{
	int const rc = REAL(hipModuleGetFunction, function, module, name);
	struct state *const s = lock();
	struct module const *const m = module_of(s, module);
	struct name const module_name = m ? numbered('m', m->id) : (struct name){"?"};
	if (rc) {
		emit(s, "hipModuleGetFunction %s k=%s rc=%d", module_name.text, name, rc);
		unlock(s);
		return rc;
	}

	size_t const length = strlen(name);
	struct function f = {.module = module};
	if (m) {
		f.label = join(m->file, m->file_length, ':', name, length);
		bool listed;
		size_t const k = kernel_find(m->kernels, m->kernel_count, name, length, &listed);
		if (listed)
			f.kernel = &m->kernels[k];
	} else {
		f.label = join("?", sizeof "?" - 1, ':', name, length);
	}
	bool known;
	size_t const at = support_keys_find(s->functions.keys, s->functions.count, key_of(*function), &known);
	f.id = known ? s->functions.values[at].id : next_id(s, 'f');

	FILE *const log = line(s);
	if (log)
		log_function(log, &f, module_name.text);

	if (known) {
		// The old label is replaced at once.
		free(s->functions.values[at].label);
		s->functions.values[at] = f;
	} else {
		*functions_insert(&s->functions, at, key_of(*function)) = f;
	}
	unlock(s);
	return rc;
}

/** @brief hipModuleLaunchKernel, logged. */
int
hipModuleLaunchKernel (void      *f,
                       unsigned   gx,
                       unsigned   gy,
                       unsigned   gz,
                       unsigned   bx,
                       unsigned   by,
                       unsigned   bz,
                       unsigned   shared,
                       void      *stream,
                       void     **params,
                       void     **extra)
{
	int const rc = REAL(hipModuleLaunchKernel, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
	return launch("launch", f, gx, gy, gz, bx, by, bz, shared, stream, params, extra, rc);
}

/** @brief hipExtModuleLaunchKernel, logged: its global sizes are threads, logged as groups like
 *         hipModuleLaunchKernel's.
 */
int
hipExtModuleLaunchKernel (void      *f,
                          unsigned   gx,
                          unsigned   gy,
                          unsigned   gz,
                          unsigned   lx,
                          unsigned   ly,
                          unsigned   lz,
                          size_t     shared,
                          void      *stream,
                          void     **params,
                          void     **extra,
                          void      *start,
                          void      *stop,
                          unsigned   flags)
{
	int const rc = REAL(hipExtModuleLaunchKernel, f, gx, gy, gz, lx, ly, lz, shared, stream, params, extra, start,
	                    stop, flags);
	return launch(flags ? "ext_launch_anyorder" : "ext_launch", f, lx ? gx / lx : gx, ly ? gy / ly : gy,
	              lz ? gz / lz : gz, lx, ly, lz, shared, stream, params, extra, rc);
}

/** @brief hipModuleUnload, logged before the real call; the module's functions are forgotten. */
int
hipModuleUnload (void *module)
{
	struct state *const s = lock();
	bool found;
	size_t const at = support_keys_find(s->modules.keys, s->modules.count, key_of(module), &found);
	emit(s, "hipModuleUnload %s", found ? numbered('m', s->modules.values[at].id).text : "?");
	size_t kept = 0;
	for (size_t i = 0; i < s->functions.count; ++i) {
		struct function *const f = &s->functions.values[i];
		if (f->module == module) {
			free(f->label);
			f->label = nullptr;
			continue;
		}
		s->functions.keys[kept] = s->functions.keys[i];
		s->functions.values[kept++] = *f;
	}
	s->functions.count = kept;
	if (found) {
		module_fini(&s->modules.values[at]);
		modules_erase(&s->modules, at);
	}
	unlock(s);
	return REAL(hipModuleUnload, module);
}

#undef REAL
