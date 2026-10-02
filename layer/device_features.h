/** @file
 *
 * What the layer adds to a game's vkCreateDevice: formatless storage writes for the composition,
 * and on request the in-layer network's features and extensions. And the ledger, which tells from
 * the request that vkCreateDevice accepted whether the network may run on the device.
 *
 * Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_DEVICE_FEATURES_H_
#define DLSSLOP_AMD_LAYER_DEVICE_FEATURES_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
# include <cstdlib>
# include <cstring>
# define DEVICE_FEATURES_STD(x) std::x
#else
# include <stddef.h>
# include <stdint.h>
# include <stdlib.h>
# include <string.h>
# define DEVICE_FEATURES_STD(x) x
#endif

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include "../common/network_requirements.h"
#include "../common/vk_chain.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VK_VERSION_1_4
# define DEVICE_FEATURES_VULKAN_1_4(X) \
	X(VkPhysicalDeviceVulkan14Features, PHYSICAL_DEVICE_VULKAN_1_4_FEATURES)
#else
# define DEVICE_FEATURES_VULKAN_1_4(X)
#endif

// The structures that struct device_features copies: the loader's, the core feature structures and
// the others of the usual Proton chains, and the network's own, which a game may chain itself.
#define DEVICE_FEATURES_COPYABLE_LIST(X) \
	X(VkLayerDeviceCreateInfo, LOADER_DEVICE_CREATE_INFO) \
	X(VkPhysicalDeviceFeatures2, PHYSICAL_DEVICE_FEATURES_2) \
	X(VkPhysicalDeviceVulkan11Features, PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) \
	X(VkPhysicalDeviceVulkan12Features, PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) \
	X(VkPhysicalDeviceVulkan13Features, PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) \
	DEVICE_FEATURES_VULKAN_1_4(X) \
	X(VkDeviceGroupDeviceCreateInfo, DEVICE_GROUP_DEVICE_CREATE_INFO) \
	X(VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT, \
	  PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT) \
	X(VkPhysicalDeviceSynchronization2Features, PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES) \
	X(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR) \
	X(VkPhysicalDeviceShaderFloat8FeaturesEXT, PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT) \
	X(VkPhysicalDevice16BitStorageFeatures, PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) \
	X(VkPhysicalDevice8BitStorageFeatures, PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) \
	X(VkPhysicalDeviceShaderFloat16Int8Features, PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) \
	X(VkPhysicalDeviceVulkanMemoryModelFeatures, PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES) \
	X(VkPhysicalDeviceSubgroupSizeControlFeatures, PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES) \
	X(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, \
	  PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR)

/** @brief Room for a private copy of a structure that struct device_features copies.
 *
 * Its members are named after their structures' types.
 */
union device_features_copy {
#define DEVICE_FEATURES_MEMBER(type, tag) type tag;
	DEVICE_FEATURES_COPYABLE_LIST(DEVICE_FEATURES_MEMBER)
#undef DEVICE_FEATURES_MEMBER
};

/** @brief A structure that struct device_features copies. */
struct device_features_copyable {
	VkStructureType               type; //!< The structure's type.
	DEVICE_FEATURES_STD(uint32_t) size; //!< Its size.
};

/** @brief The structures that struct device_features copies. */
static struct device_features_copyable const DEVICE_FEATURES_COPYABLE[] = {
#define DEVICE_FEATURES_ROW(type, tag) {VK_STRUCTURE_TYPE_##tag, sizeof (type)},
	DEVICE_FEATURES_COPYABLE_LIST(DEVICE_FEATURES_ROW)
#undef DEVICE_FEATURES_ROW
};

#undef DEVICE_FEATURES_COPYABLE_LIST
#undef DEVICE_FEATURES_VULKAN_1_4

/** @brief The number of DEVICE_FEATURES_COPYABLE. */
static constexpr DEVICE_FEATURES_STD(uint32_t) DEVICE_FEATURES_COPYABLE_COUNT =
	sizeof DEVICE_FEATURES_COPYABLE / sizeof *DEVICE_FEATURES_COPYABLE;

/** @brief The most structures of a game's chain that struct device_features copies.
 *
 * A valid chain needs at most 20 copies: each structure that DEVICE_FEATURES_COPYABLE names but the
 * loader's once (VUID-VkDeviceCreateInfo-sType-unique), and the loader's once per VkLayerFunction.
 */
static constexpr DEVICE_FEATURES_STD(uint32_t) DEVICE_FEATURES_COPIES = 32;

/** @brief What the layer adds to a game's vkCreateDevice.
 *
 * A bit is set in the structure of the game's chain that carries it, else in a structure of its own
 * put at the head of the chain. The structures up to the last one changed are private copies: the
 * game's const chain is never written. A structure to change behind one that this cannot copy, or
 * behind more than DEVICE_FEATURES_COPIES of them, fails safely, and the game then creates its device
 * as it asked.
 *
 * A zeroed object is an empty one, and the object owns nothing. A request that
 * device_features_enable() changed points into the object, which must outlive the request's
 * vkCreateDevice.
 */
struct device_features {
	union device_features_copy   copies[DEVICE_FEATURES_COPIES]; //!< Copies of the game's structures.
	struct network_feature_chain added;                          //!< The network's own structures.
	VkPhysicalDeviceFeatures     legacy;                         //!< A copy of pEnabledFeatures.
};

/** @brief Where a feature bit is. */
struct device_features_place {
	void const                  *node;   //!< The structure, or nullptr if no structure carries the bit.
	DEVICE_FEATURES_STD(size_t)  offset; //!< The bit's offset in it.
};

/** @brief The VkPhysicalDeviceFeatures2 of a request.
 *
 * @param info The request.
 * @return     The structure in its chain, or nullptr if it has none.
 */
static inline VkPhysicalDeviceFeatures2 const *
device_features_core_features2 (VkDeviceCreateInfo const *info)
{
	return (VkPhysicalDeviceFeatures2 const *)vk_chain_find(info->pNext,
	                                                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
}

/** @brief Whether a request enables formatless storage writes.
 *
 * @param info The request, or nullptr.
 * @return     true if its VkPhysicalDeviceFeatures2, or without one its pEnabledFeatures, enables
 *             shaderStorageImageWriteWithoutFormat.
 */
static inline bool
device_features_has_formatless_storage_writes (VkDeviceCreateInfo const *info)
{
	if (!info)
		return false;
	VkPhysicalDeviceFeatures2 const *const features2 = device_features_core_features2(info);
	if (features2)
		return features2->features.shaderStorageImageWriteWithoutFormat;
	return info->pEnabledFeatures && info->pEnabledFeatures->shaderStorageImageWriteWithoutFormat;
}

/** @brief Where a network feature is enabled in a chain.
 *
 * @param chain The chain's first structure, or nullptr.
 * @param f     The feature.
 * @return      The structure that carries the bit alone, else the core structure that carries it.
 */
static inline struct device_features_place
device_features_network_feature_in (void const                   *chain,
                                    struct network_feature const *f)
{
	struct device_features_place const alone = {vk_chain_find(chain, f->type), f->offset};
	if (alone.node || f->core == NETWORK_FEATURE_NO_CORE)
		return alone;
	struct device_features_place const core = {vk_chain_find(chain, f->core), f->core_offset};
	return core;
}

/** @brief The ledger: whether a device has every feature and extension that the in-layer network
 *         needs.
 *
 * A device's supported features are no proof: only what was enabled may be used.
 *
 * @param info The request that vkCreateDevice accepted, or nullptr.
 * @return     true if @a info enables them all.
 */
static inline bool
device_features_network_enabled (VkDeviceCreateInfo const *info)
{
	if (!info)
		return false;
	for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		struct network_feature const *const f = &NETWORK_FEATURES[i];
		if (f->extension && !network_requirements_listed(info->ppEnabledExtensionNames,
		                                                 info->enabledExtensionCount, f->extension))
			return false;
		struct device_features_place const place = device_features_network_feature_in(info->pNext, f);
		if (!place.node || !*vk_chain_bit(place.node, place.offset))
			return false;
	}
	return true;
}

/** @brief Whether the in-layer network is asked for, while it is being developed.
 *
 * @return true if DLSSLOP_LAYER_NETWORK is 1.
 */
static inline bool
device_features_network_requested (void)
{
	char const *const value = DEVICE_FEATURES_STD(getenv)("DLSSLOP_LAYER_NETWORK");
	return value && !DEVICE_FEATURES_STD(strcmp)(value, "1");
}

/** @brief What keeps the in-layer network off a game's device.
 *
 * The functions are the next layer's.
 *
 * @param physical         The device.
 * @param instance_version The instance's API version.
 * @param properties2      vkGetPhysicalDeviceProperties2, or nullptr.
 * @param features2        vkGetPhysicalDeviceFeatures2, or nullptr.
 * @param extensions       vkEnumerateDeviceExtensionProperties.
 * @param matrices         vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, or nullptr.
 * @return                 nullptr if nothing does. Otherwise "a Vulkan 1.3 instance", or what the
 *                         device lacks (network_requirements_unsupported()).
 */
static inline char const *
device_features_network_unavailable (VkPhysicalDevice                                      physical,
                                     DEVICE_FEATURES_STD(uint32_t)                         instance_version,
                                     PFN_vkGetPhysicalDeviceProperties2                    properties2,
                                     PFN_vkGetPhysicalDeviceFeatures2                      features2,
                                     PFN_vkEnumerateDeviceExtensionProperties              extensions,
                                     PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrices)
{
	if (instance_version < VK_API_VERSION_1_3 || !properties2 || !features2)
		return "a Vulkan 1.3 instance";
	return network_requirements_unsupported(physical, properties2, features2, extensions, matrices,
	                                        nullptr);
}

/** @brief Adds the network's extensions that a request lacks.
 *
 * @param info The request, or nullptr.
 * @param list Room for the request's extensions and NETWORK_FEATURE_COUNT more. It may be the
 *             request's own list if that has the room.
 * @return     true if any was added; the request then names the extensions in @a list.
 */
static inline bool
device_features_add_network_extensions (VkDeviceCreateInfo  *info,
                                        char const         **list)
{
	if (!info || !list)
		return false;
	DEVICE_FEATURES_STD(uint32_t) const count = info->enabledExtensionCount;
	// The request's own list is in place already.
	if (list != info->ppEnabledExtensionNames) {
		for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < count; ++i)
			list[i] = info->ppEnabledExtensionNames[i];
	}
	DEVICE_FEATURES_STD(uint32_t) const total = network_requirements_append_extensions(list, count);
	if (total == count)
		return false;
	info->enabledExtensionCount = total;
	info->ppEnabledExtensionNames = list;
	return true;
}

/** @brief The size of a structure that struct device_features copies.
 *
 * @param structure The structure.
 * @return          Its size, or 0 if it is not one that struct device_features copies.
 */
static inline DEVICE_FEATURES_STD(uint32_t)
device_features_copy_size (void const *structure)
{
	VkStructureType const type = vk_chain_type(structure);
	for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < DEVICE_FEATURES_COPYABLE_COUNT; ++i) {
		if (DEVICE_FEATURES_COPYABLE[i].type == type)
			return DEVICE_FEATURES_COPYABLE[i].size;
	}
	return 0;
}

/** @brief Sets the bits of a structure's changes in its copy.
 *
 * @param copy    The copy.
 * @param node    The structure.
 * @param changes The bits to set, in any structure.
 * @param count   The bits in @a changes.
 * @return        The bits of @a node that were set.
 */
static inline DEVICE_FEATURES_STD(uint32_t)
device_features_apply (void                               *copy,
                       void const                         *node,
                       struct device_features_place const *changes,
                       DEVICE_FEATURES_STD(uint32_t)       count)
{
	DEVICE_FEATURES_STD(uint32_t) applied = 0;
	for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < count; ++i) {
		if (changes[i].node != node)
			continue;
		*vk_chain_bit(copy, changes[i].offset) = VK_TRUE;
		++applied;
	}
	return applied;
}

/** @brief Copies a chain up to the last structure to change, and changes the copies.
 *
 * The last copy's pNext still leads to the rest of the chain.
 *
 * @param r       The request's copies.
 * @param head    The chain's first structure; then the first copy.
 * @param changes The bits to set.
 * @param count   The bits in @a changes.
 * @return        false, with @a head unchanged, if a structure to copy is not one that struct
 *                device_features copies, or is one more than DEVICE_FEATURES_COPIES.
 */
static inline bool
device_features_copy_prefix (struct device_features             *r,
                             void const                        **head,
                             struct device_features_place const *changes,
                             DEVICE_FEATURES_STD(uint32_t)       count)
{
	// The copies are linked behind an anchor, whose pNext is the chain's first structure.
	VkBaseOutStructure anchor = {};
	vk_chain_link(&anchor, *head);
	void *previous = &anchor;
	DEVICE_FEATURES_STD(uint32_t) copies = 0;
	DEVICE_FEATURES_STD(uint32_t) left = count;
	for (void const *node = *head; left; node = vk_chain_next(node)) {
		DEVICE_FEATURES_STD(uint32_t) const size = device_features_copy_size(node);
		if (!size || copies == DEVICE_FEATURES_COPIES)
			return false;
		void *const copy = DEVICE_FEATURES_STD(memcpy)(&r->copies[copies++], node, size);
		vk_chain_link(previous, copy);
		previous = copy;
		left -= device_features_apply(copy, node, changes, count);
	}
	*head = vk_chain_next(&anchor);
	return true;
}

/** @brief Whether a list holds a structure.
 *
 * @param list      The list.
 * @param count     The structures in it.
 * @param structure The structure.
 * @return          true if @a list holds @a structure.
 */
static inline bool
device_features_holds (void *const                   *list,
                       DEVICE_FEATURES_STD(uint32_t)  count,
                       void const                    *structure)
{
	for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < count; ++i) {
		if (list[i] == structure)
			return true;
	}
	return false;
}

/** @brief Puts structures of the request's own ahead of a chain, for the network features that no
 *         structure of the chain carries.
 *
 * @param r       The request.
 * @param head    The chain's first structure.
 * @param missing The features.
 * @param count   The features in @a missing.
 * @return        The new chain's first structure.
 */
static inline void const *
device_features_add_own (struct device_features              *r,
                         void const                          *head,
                         struct network_feature const *const *missing,
                         DEVICE_FEATURES_STD(uint32_t)        count)
{
	network_feature_chain_init(&r->added);
	void *own[NETWORK_FEATURE_COUNT];
	DEVICE_FEATURES_STD(uint32_t) owned = 0;
	for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < count; ++i) {
		void *const structure = network_feature_chain_structure(&r->added, missing[i]->type);
		*vk_chain_bit(structure, missing[i]->offset) = VK_TRUE;
		if (!device_features_holds(own, owned, structure))
			own[owned++] = structure;
	}
	// The structures go ahead in the order the features first asked for them.
	while (owned--) {
		vk_chain_link(own[owned], head);
		head = own[owned];
	}
	return head;
}

/** @brief Formatless storage writes in a request's pEnabledFeatures, for a request without a
 *         VkPhysicalDeviceFeatures2.
 *
 * @param r       The request's storage.
 * @param enabled The request's pEnabledFeatures, or nullptr.
 * @return        @a enabled if it enables them, otherwise the copy of it in @a r that does.
 */
static inline VkPhysicalDeviceFeatures const *
device_features_enable_legacy (struct device_features         *r,
                               VkPhysicalDeviceFeatures const *enabled)
{
	VkPhysicalDeviceFeatures const none = {};
	VkPhysicalDeviceFeatures const *const from = enabled ? enabled : &none;
	if (from->shaderStorageImageWriteWithoutFormat)
		return enabled;
	r->legacy = *from;
	r->legacy.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	return &r->legacy;
}

/** @brief Adds the layer's features to a game's vkCreateDevice request.
 *
 * Formatless storage writes always; with @a network the in-layer network's features too.
 *
 * @param r       The request's storage, or nullptr. A zeroed one, or one that this declined.
 * @param info    The game's request, or nullptr; or the layer's copy of it.
 * @param network Whether to add the network's features.
 * @return        false, leaving @a info as it was, if a structure to change follows one this cannot
 *                copy, or more than DEVICE_FEATURES_COPIES structures.
 */
static inline bool
device_features_enable (struct device_features *r,
                        VkDeviceCreateInfo     *info,
                        bool                    network)
{
	if (!r || !info)
		return false;
	// The bits to set in the game's structures, and the network's that no structure carries.
	struct device_features_place changes[1 + NETWORK_FEATURE_COUNT];
	struct network_feature const *missing[NETWORK_FEATURE_COUNT];
	DEVICE_FEATURES_STD(uint32_t) change_count = 0;
	DEVICE_FEATURES_STD(uint32_t) missing_count = 0;
	VkPhysicalDeviceFeatures2 const *const features2 = device_features_core_features2(info);
	VkPhysicalDeviceFeatures const *enabled = info->pEnabledFeatures;
	// Without a VkPhysicalDeviceFeatures2, formatless storage writes go in a copy of pEnabledFeatures.
	if (!features2) {
		enabled = device_features_enable_legacy(r, enabled);
	} else if (!features2->features.shaderStorageImageWriteWithoutFormat) {
		struct device_features_place const place = {
			features2,
			offsetof(VkPhysicalDeviceFeatures2, features.shaderStorageImageWriteWithoutFormat)
		};
		changes[change_count++] = place;
	}
	DEVICE_FEATURES_STD(uint32_t) const asked = network ? NETWORK_FEATURE_COUNT : 0;
	for (DEVICE_FEATURES_STD(uint32_t) i = 0; i < asked; ++i) {
		struct device_features_place const place =
			device_features_network_feature_in(info->pNext, &NETWORK_FEATURES[i]);
		if (!place.node)
			missing[missing_count++] = &NETWORK_FEATURES[i];
		else if (!*vk_chain_bit(place.node, place.offset))
			changes[change_count++] = place;
	}
	void const *head = info->pNext;
	if (!device_features_copy_prefix(r, &head, changes, change_count))
		return false;
	info->pNext = device_features_add_own(r, head, missing, missing_count);
	info->pEnabledFeatures = enabled;
	return true;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef DEVICE_FEATURES_STD

#endif /* DLSSLOP_AMD_LAYER_DEVICE_FEATURES_H_ */
