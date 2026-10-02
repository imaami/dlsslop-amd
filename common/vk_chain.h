/** @file
 *
 * A Vulkan structure chain's links, read and written as bytes.
 *
 * Through VkBaseInStructure or VkBaseOutStructure they would be accesses of another type than the
 * structures', which the compiler may assume never alias: GCC at -O3 dropped feature bits written
 * through such a walk. memcpy() reads and writes the sType and pNext of any structure.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_VK_CHAIN_H_
#define DLSSLOP_AMD_COMMON_VK_CHAIN_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstring>
# define VK_CHAIN_STD(x) std::x
#else
# include <stddef.h>
# include <string.h>
# define VK_CHAIN_STD(x) x
#endif

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief The type of a structure.
 *
 * @param structure The structure.
 * @return          Its sType.
 */
static inline VkStructureType
vk_chain_type (void const *structure)
{
	VkStructureType type;
	VK_CHAIN_STD(memcpy)(&type, structure, sizeof type);
	return type;
}

/** @brief The structure that follows a structure in its chain.
 *
 * @param structure The structure.
 * @return          Its pNext.
 */
static inline void *
vk_chain_next (void const *structure)
{
	void *next;
	VK_CHAIN_STD(memcpy)(&next, (unsigned char const *)structure + offsetof(VkBaseInStructure, pNext),
	                     sizeof next);
	return next;
}

/** @brief Makes a structure's pNext point at another.
 *
 * @param structure The structure.
 * @param next      What follows it now, or nullptr.
 */
static inline void
vk_chain_link (void       *structure,
               void const *next)
{
	VK_CHAIN_STD(memcpy)((unsigned char *)structure + offsetof(VkBaseOutStructure, pNext), &next,
	                     sizeof next);
}

/** @brief The structure of a type in a chain.
 *
 * @param first The chain's first structure, or nullptr.
 * @param type  The type.
 * @return      The first structure of @a type from @a first on, or nullptr.
 */
static inline void *
vk_chain_find (void const      *first,
               VkStructureType  type)
{
	while (first && vk_chain_type(first) != type)
		first = vk_chain_next(first);
	return (void *)first;
}

/** @brief A feature bit of a structure.
 *
 * @param structure The structure.
 * @param offset    The bit's offset in it.
 * @return          The bit.
 */
static inline VkBool32 *
vk_chain_bit (void const           *structure,
              VK_CHAIN_STD(size_t)  offset)
{
	return (VkBool32 *)((unsigned char *)structure + offset);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef VK_CHAIN_STD

#endif /* DLSSLOP_AMD_COMMON_VK_CHAIN_H_ */
