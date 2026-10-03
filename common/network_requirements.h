/** @file
 *
 * What DLSSNR-AMD's Vulkan network needs of a device: one table for dlsslopd's own device and for a
 * game's device the layer adds it to. network_requirements.c defines the table and the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_NETWORK_REQUIREMENTS_H_
#define DLSSLOP_AMD_COMMON_NETWORK_REQUIREMENTS_H_

#ifdef __cplusplus
# include <cstdint>
#else
# include <stdint.h>
#endif

#include <vulkan/vulkan.h>

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief A feature bit that the network's SPIR-V uses.
 *
 * Upstream listed them in DLSSNR-AMD's linux/src/core/nrvk.hpp, Context::create and adopt.
 */
struct network_feature {
	char const      *name;        //!< The bit's name.
	char const      *extension;   //!< The extension that provides it; nullptr in core Vulkan 1.3.
	VkStructureType  type;        //!< The structure that carries the bit alone.
	STD(uint32_t)    offset;      //!< The bit's offset in that structure.
	VkStructureType  core;        //!< The VkPhysicalDeviceVulkan1xFeatures that carries it too.
	STD(uint32_t)    core_offset; //!< The bit's offset in that structure.
};

/** @brief The core type of a bit that no VkPhysicalDeviceVulkan1xFeatures carries. */
#define NETWORK_FEATURE_NO_CORE VK_STRUCTURE_TYPE_MAX_ENUM

/** @brief The number of NETWORK_FEATURES, the rows of network_requirements.c's table. */
#define NETWORK_FEATURE_COUNT UINT32_C(13)

/** @brief The feature bits that the network's SPIR-V uses. */
extern struct network_feature const NETWORK_FEATURES[NETWORK_FEATURE_COUNT];

/** @brief One structure of each type that NETWORK_FEATURES names, zeroed and chained behind a
 *         VkPhysicalDeviceFeatures2: for a support query or a vkCreateDevice.
 *
 * Its structures point at each other, so it is only ever initialized in place, by
 * network_feature_chain_init().
 */
struct network_feature_chain {
	VkPhysicalDeviceFeatures2                                head;      //!< The chain's first structure.
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR             coop;      //!< Cooperative matrices.
	VkPhysicalDeviceShaderFloat8FeaturesEXT                  fp8;       //!< FP8.
	VkPhysicalDevice16BitStorageFeatures                     storage16; //!< 16-bit storage.
	VkPhysicalDevice8BitStorageFeatures                      storage8;  //!< 8-bit storage.
	VkPhysicalDeviceShaderFloat16Int8Features                float16;   //!< FP16 and Int8 arithmetic.
	VkPhysicalDeviceVulkanMemoryModelFeatures                memory;    //!< The memory model.
	VkPhysicalDeviceSubgroupSizeControlFeatures              subgroup;  //!< Subgroup size control.
	VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR layout;    //!< Explicit workgroup layout.
};

/** @brief What stops the network when the host's memory runs out, named like the other requirements. */
#define NETWORK_REQUIREMENTS_NO_MEMORY "host memory"

/** @brief What stops the network on a device that lists no FP8 matrices of the network's shape. */
#define NETWORK_REQUIREMENTS_NO_MATRICES "e4m3 16x16x16 cooperative matrices"

/** @brief Zeroes a chain's structures, sets their types and links them in the chain's order.
 *
 * @param dest The chain, or nullptr.
 */
extern void
network_feature_chain_init (struct network_feature_chain *dest);

/** @brief A chain's structure of a type, whatever a caller has since linked it to.
 *
 * @param c    The chain, or nullptr.
 * @param type The type.
 * @return     The structure, or nullptr if the chain has none of @a type.
 */
extern void *
network_feature_chain_structure (struct network_feature_chain *c,
                                 VkStructureType               type);

/** @brief A chain's bit of a feature.
 *
 * @param c The chain, or nullptr.
 * @param f The feature.
 * @return  The bit in the structure that carries it alone, or nullptr if the chain has none.
 */
extern VkBool32 *
network_feature_chain_bit (struct network_feature_chain *c,
                           struct network_feature const *f);

/** @brief Whether a list holds a name.
 *
 * @param list  The list.
 * @param count The names in it.
 * @param name  The name.
 * @return      true if @a list holds @a name.
 */
extern bool
network_requirements_listed (char const *const *list,
                             STD(uint32_t)      count,
                             char const        *name);

/** @brief Appends the network's extensions that a list lacks.
 *
 * @param list  The list, with room for NETWORK_FEATURE_COUNT more names.
 * @param count The names in it.
 * @return      The names in it now: more than @a count if any was appended.
 */
extern STD(uint32_t)
network_requirements_append_extensions (char const    **list,
                                        STD(uint32_t)   count);

#undef STD

/** @brief What stops the network running on a device.
 *
 * The functions are the caller's route to the device: the loader's, or the next layer's.
 *
 * @param physical    The device.
 * @param properties2 vkGetPhysicalDeviceProperties2.
 * @param features2   vkGetPhysicalDeviceFeatures2.
 * @param extensions  vkEnumerateDeviceExtensionProperties.
 * @param matrices    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, or nullptr.
 * @param also        An extension the caller needs besides, or nullptr.
 * @return            nullptr if nothing does. Otherwise "Vulkan 1.3", an extension's or a feature's
 *                    name, "32-lane compute subgroups", the name of @a matrices if it is nullptr,
 *                    NETWORK_REQUIREMENTS_NO_MATRICES, or NETWORK_REQUIREMENTS_NO_MEMORY.
 */
extern char const *
network_requirements_unsupported (VkPhysicalDevice                                       physical,
                                  PFN_vkGetPhysicalDeviceProperties2                     properties2,
                                  PFN_vkGetPhysicalDeviceFeatures2                       features2,
                                  PFN_vkEnumerateDeviceExtensionProperties               extensions,
                                  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR  matrices,
                                  char const                                            *also);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_NETWORK_REQUIREMENTS_H_ */
