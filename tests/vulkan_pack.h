/** @file
 *
 * Model packs for the Vulkan network's host tests, in memory: NRMODEL1 files of given entries, the
 * synthetic entries that upstream's graph build packed for the tests' digests, and synthetic models
 * of the entries that a plan reads. vulkan_pack.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_TESTS_VULKAN_PACK_H_
#define DLSSLOP_AMD_TESTS_VULKAN_PACK_H_

#include <stddef.h>
#include <stdint.h>

#include "vulkan_plan.h"
#include "vulkan_weights.h"

/** @brief FNV-1a 64's offset basis, the hash of no bytes. */
#define VULKAN_PACK_FNV1A_BASIS UINT64_C(0xcbf29ce484222325)

/** @brief Hashes bytes with FNV-1a 64, on from a hash.
 *
 * @param hash  The hash so far: VULKAN_PACK_FNV1A_BASIS for the first bytes.
 * @param data  The bytes.
 * @param bytes Their number.
 * @return      The hash with them.
 */
extern uint64_t
vulkan_pack_fnv1a (uint64_t    hash,
                   void const *data,
                   size_t      bytes);

/** @brief splitmix64's state. */
struct vulkan_pack_random {
	uint64_t state; //!< The state.
};

/** @brief splitmix64's next output.
 *
 * @param random The state, which advances.
 * @return       The output.
 */
extern uint64_t
vulkan_pack_random_next (struct vulkan_pack_random *random);

/** @brief Writes a synthetic entry: splitmix64 outputs, little-endian, seeded with the FNV-1a 64 of
 *         its name.
 *
 * Every residual_scale in unpacked/ but those of blocks 48, 56 and 62 starts with eight zero values,
 * as in the real model, so that both of upstream's readings of it are taken.
 *
 * @param name   The entry's name.
 * @param length The name's length.
 * @param data   Receives the entry.
 * @param bytes  Its bytes.
 */
extern void
vulkan_pack_synthetic_entry (char const *name,
                             size_t      length,
                             uint8_t    *data,
                             size_t      bytes);

/** @brief A model pack in memory, which vulkan_model_open() reads through /proc. */
struct vulkan_pack {
	char path[32]; //!< /proc/self/fd/FD.
	int  fd;       //!< The memory file, or -1.
	bool ok;       //!< Whether its path fit and every byte reached the file.
};

/** @brief Makes an empty model pack in memory.
 *
 * @param dest Receives the pack: its fd is -1 when no memory file could be made, and its ok says
 *             whether its path fit. A write to a pack without a file clears its ok.
 */
extern void
vulkan_pack_init (struct vulkan_pack *dest);

/** @brief Closes a model pack.
 *
 * @param pack The pack, or nullptr.
 */
extern void
vulkan_pack_fini (struct vulkan_pack *pack);

/** @brief Appends bytes to a model pack; one that cannot be written clears its ok.
 *
 * @param pack The pack.
 * @param data The bytes.
 * @param size Their number.
 */
extern void
vulkan_pack_write (struct vulkan_pack *pack,
                   void const         *data,
                   size_t              size);

/** @brief Appends the start of an NRMODEL1 file to a model pack.
 *
 * @param pack  The pack.
 * @param count The file's entries.
 */
extern void
vulkan_pack_header (struct vulkan_pack *pack,
                    uint32_t            count);

/** @brief The bytes of an index entry: its name's length, its name, its offset and its size.
 *
 * @param name   The name.
 * @param length Its length, below VULKAN_WEIGHTS_NAME_BYTES.
 * @param offset The entry's data's offset.
 * @param size   Their bytes.
 * @param out    Receives the entry.
 * @return       Its bytes: 4 + length + 16.
 */
extern size_t
vulkan_pack_index_entry (char const *name,
                         size_t      length,
                         uint64_t    offset,
                         uint64_t    size,
                         uint8_t     out[4 + VULKAN_WEIGHTS_NAME_BYTES + 16]);

/** @brief An entry of an NRMODEL1 file to write. */
struct vulkan_pack_entry {
	char const    *name;   //!< Its name.
	uint8_t const *data;   //!< Its data.
	size_t         length; //!< The name's length.
	size_t         size;   //!< The data's bytes.
};

/** @brief Appends the NRMODEL1 file of entries to a model pack, their data after the index in their
 *         order.
 *
 * @param pack    The pack.
 * @param entries The entries.
 * @param count   Their number.
 */
extern void
vulkan_pack_entries (struct vulkan_pack             *pack,
                     struct vulkan_pack_entry const *entries,
                     size_t                          count);

/** @brief The size of an entry in the real model, the pack that linux/package/model-tools extracts
 *         from nvngx_dlssnr 310.8.0.
 *
 * @param source The entry.
 * @return       Its bytes.
 */
extern uint32_t
vulkan_pack_model_bytes (struct vulkan_source const *source);

/** @brief An entry that a plan reads, by name. */
struct vulkan_pack_source {
	char                 name[VULKAN_WEIGHTS_NAME_BYTES]; //!< Its name.
	size_t               length;                          //!< The name's length.
	struct vulkan_source source;                          //!< The entry.
};

/** @brief The entries that a plan's segments read, each once, sorted by name.
 *
 * @param plan  The plan.
 * @param count Receives their number.
 * @return      The entries, which the caller frees, or nullptr without memory for them.
 */
extern struct vulkan_pack_source *
vulkan_pack_plan_entries (struct vulkan_plan const *plan,
                          size_t                   *count);

/** @brief Writes a model pack of synthetic entries (vulkan_pack_synthetic_entry()), of the real
 *         model's names and sizes, holding every entry that a plan reads.
 *
 * It is the pack that upstream's synthetic goldens came from, but for the entries no plan reads. Its
 * weights free no Swin head of the exponent's upper clamp; unclamped, every position bias and head
 * scale is zero, which frees every head.
 *
 * @param plan      The plan.
 * @param pack      The pack, empty.
 * @param unclamped Whether the biases and scales are zero.
 * @return          Whether the pack was written.
 */
extern bool
vulkan_pack_synthetic_model (struct vulkan_plan const *plan,
                             struct vulkan_pack       *pack,
                             bool                      unclamped);

#endif /* DLSSLOP_AMD_TESTS_VULKAN_PACK_H_ */
