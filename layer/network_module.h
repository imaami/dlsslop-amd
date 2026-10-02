/** @file
 *
 * The in-layer network's module, libdlsslop-network.so, beside the layer. The layer loads it for a
 * device whose ledger enabled the network, and reaches it through these C functions only: the
 * network's code and the C++ runtime it links stay out of the layer.
 *
 * The functions' names, network_module.map, the layout of struct dlsslop_network_device and the
 * values of enum dlsslop_network_state are the contract between the layer and the module. Each
 * function has a function type, which declares it and types the loader's pointer to it.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_LAYER_NETWORK_MODULE_H_
#define DLSSLOP_AMD_LAYER_NETWORK_MODULE_H_

#ifdef __cplusplus
# include <cstdint>
# include <cstdio>
# define NETWORK_MODULE_STD(x) std::x
#else
# include <stdint.h>
# include <stdio.h>
# define NETWORK_MODULE_STD(x) x
#endif

#include <dlfcn.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct DlsslopNetwork;
struct ShmHeader;

/** @brief A function that the module calls with the device's context around a submit.
 *
 * @param context The device's context.
 */
typedef void
dlsslop_network_queue_fn (void *context);

/** @brief A function that takes the network's log lines.
 *
 * @param line A line, without its newline.
 */
typedef void
dlsslop_network_log_fn (char const *line);

/** @brief The game's device, as the layer knows it.
 *
 * The network's uploads go to the queue from a thread of the module's own while the network
 * builds; lock_queue and unlock_queue, with context, are called around each of those submits.
 * physical_dispatch is the next layer's, for physical-device queries through the layer's handles,
 * and the module looks up its functions only in dlsslop_network_open().
 */
struct dlsslop_network_device {
	VkInstance                       instance;          //!< The game's instance.
	VkPhysicalDevice                 physical;          //!< The device's physical device.
	VkDevice                         device;            //!< The device.
	VkQueue                          queue;             //!< The queue that takes the build's uploads.
	NETWORK_MODULE_STD(uint32_t)     family;            //!< The queue's family.
	dlsslop_network_queue_fn        *lock_queue;        //!< Called before each submit of the build.
	dlsslop_network_queue_fn        *unlock_queue;      //!< Called after each submit of the build.
	void                            *context;           //!< What lock_queue and unlock_queue take.
	PFN_vkGetInstanceProcAddr        physical_dispatch; //!< The next layer's vkGetInstanceProcAddr.
	VkPhysicalDeviceMemoryProperties memory;            //!< The physical device's memory.
	dlsslop_network_log_fn          *log;               //!< Where the network's log lines go.
};

/** @brief What dlsslop_network_prepare() says of the next frame. */
enum dlsslop_network_state {
	DLSSLOP_NETWORK_READY,    //!< dlsslop_network_record() records it.
	DLSSLOP_NETWORK_BUILDING, //!< The network is being built for it in the background.
	DLSSLOP_NETWORK_REJECTED, //!< Its settings are out of range; dlsslop_network_error() says which.
	DLSSLOP_NETWORK_FAILED,   //!< The network cannot run; dlsslop_network_error() says why.
};

/** @brief The network on a device, not yet built.
 *
 * @param device The device.
 * @return       The network, or nullptr without memory.
 */
typedef struct DlsslopNetwork *
dlsslop_network_open_fn (struct dlsslop_network_device const *device);

/** @brief Readies the network for the next frame.
 *
 * A frame of another extent starts a build in the background; one of another shape of the same
 * extent reshapes the network here, in a fraction of a millisecond. Neither may overlap the
 * network's recorded work: the caller has waited for its last frame.
 *
 * @param network The network.
 * @param channel The channel, whose settings the frame takes.
 * @param width   The frame's width.
 * @param height  The frame's height.
 * @param fp16    Nonzero for RGBA16F frames, 0 for RGBA8.
 * @return        An enum dlsslop_network_state.
 */
typedef int
dlsslop_network_prepare_fn (struct DlsslopNetwork        *network,
                            struct ShmHeader const       *channel,
                            NETWORK_MODULE_STD(uint32_t)  width,
                            NETWORK_MODULE_STD(uint32_t)  height,
                            int                           fp16);

/** @brief Records the frame that dlsslop_network_prepare() readied: the proxy that the composition
 *         captured, through the network, into the answer for the composition.
 *
 * @param network  The network.
 * @param cmd      The command buffer. If the frame failed, it still holds valid commands, which give
 *                 the pair back as they found it.
 * @param proxy    The composition's transfer buffer that holds the proxy.
 * @param answer   The composition's transfer buffer that takes the answer.
 * @param family   The queue's family.
 * @param exported Nonzero if both buffers belong to VK_QUEUE_FAMILY_EXTERNAL between uses.
 * @return         DLSSLOP_NETWORK_READY or DLSSLOP_NETWORK_FAILED.
 */
typedef int
dlsslop_network_record_fn (struct DlsslopNetwork        *network,
                           VkCommandBuffer               cmd,
                           VkBuffer                      proxy,
                           VkBuffer                      answer,
                           NETWORK_MODULE_STD(uint32_t)  family,
                           int                           exported);

/** @brief Says that the frame that dlsslop_network_record() recorded last was submitted.
 *
 * The next frame then follows it in the motion history. A frame that is recorded and not submitted
 * leaves the history as it was.
 *
 * @param network The network.
 */
typedef void
dlsslop_network_submitted_fn (struct DlsslopNetwork *network);

/** @brief Why the network rejected a frame or failed.
 *
 * @param network The network.
 * @return        The reason.
 */
typedef char const *
dlsslop_network_error_fn (struct DlsslopNetwork const *network);

/** @brief Waits for a build to end and frees the network.
 *
 * The device must have finished the network's work.
 *
 * @param network The network, or nullptr.
 */
typedef void
dlsslop_network_close_fn (struct DlsslopNetwork *network);

extern dlsslop_network_open_fn      dlsslop_network_open;
extern dlsslop_network_prepare_fn   dlsslop_network_prepare;
extern dlsslop_network_record_fn    dlsslop_network_record;
extern dlsslop_network_submitted_fn dlsslop_network_submitted;
extern dlsslop_network_error_fn     dlsslop_network_error;
extern dlsslop_network_close_fn     dlsslop_network_close;

/** @brief The module's functions, as the layer finds them.
 *
 * A zeroed object is an empty one: it has loaded nothing. Its failure holds at most what one log
 * line shows.
 */
struct network_module {
	void                         *library;        //!< The module, or nullptr.
	dlsslop_network_open_fn      *open;           //!< dlsslop_network_open().
	dlsslop_network_prepare_fn   *prepare;        //!< dlsslop_network_prepare().
	dlsslop_network_record_fn    *record;         //!< dlsslop_network_record().
	dlsslop_network_submitted_fn *submitted;      //!< dlsslop_network_submitted().
	dlsslop_network_error_fn     *error;          //!< dlsslop_network_error().
	dlsslop_network_close_fn     *close;          //!< dlsslop_network_close().
	char                          failure[2048];  //!< Why network_module_load() failed.
};

/** @brief Records why network_module_load() failed, before anything else can reset dlerror().
 *
 * @param m    The module.
 * @param path The module's path, the reason when dlerror() has none.
 * @return     false.
 */
static inline bool
network_module_refuse (struct network_module *m,
                       char const            *path)
{
	char const *const why = dlerror();
	NETWORK_MODULE_STD(snprintf)(m->failure, sizeof m->failure, "%s", why ? why : path);
	return false;
}

/** @brief Loads the module at a path.
 *
 * A module without the functions stays out of the game's process.
 *
 * @param m    An empty module, or nullptr.
 * @param path The module's path.
 * @return     true if @a m holds the module's functions; otherwise it holds no module, and its
 *             failure says why.
 */
static inline bool
network_module_load (struct network_module *m,
                     char const            *path)
{
	if (!m)
		return false;
	// Looked up in a local first: a refused module leaves no pointer into it in m.
	struct network_module found = {.library = dlopen(path, RTLD_NOW | RTLD_LOCAL)};
	if (!found.library)
		return network_module_refuse(m, path);
	// The lookups stop at the first that fails, whose dlerror() a later lookup that succeeds would
	// clear.
#define NETWORK_MODULE_FIND(name) \
	(found.name = (dlsslop_network_##name##_fn *)dlsym(found.library, "dlsslop_network_" #name))
	if (NETWORK_MODULE_FIND(open) && NETWORK_MODULE_FIND(prepare) && NETWORK_MODULE_FIND(record)
	    && NETWORK_MODULE_FIND(submitted) && NETWORK_MODULE_FIND(error) && NETWORK_MODULE_FIND(close)) {
		*m = found;
		return true;
	}
#undef NETWORK_MODULE_FIND
	network_module_refuse(m, path);
	dlclose(found.library);
	return false;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef NETWORK_MODULE_STD

#endif /* DLSSLOP_AMD_LAYER_NETWORK_MODULE_H_ */
