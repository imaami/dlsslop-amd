/** @file
 *
 * The in-layer network's module, libdlsslop-network.so, beside the layer. The layer loads it for a
 * device whose ledger enabled the network, and reaches it through these C functions only: the
 * network's code and the C++ runtime it links stay out of the layer.
 *
 * The functions' names, network_module.map, the layouts of struct dlsslop_network_device and struct
 * dlsslop_network_images and the values of enum dlsslop_network_state are the contract between the
 * layer and the module. Each function has a function type, which declares it and types the loader's
 * pointer to it. The module exports DLSSLOP_NETWORK_INTERFACE as dlsslop_network_interface, and the
 * layer puts it first in struct dlsslop_network_device: each side refuses the other's of another
 * interface, and a change to any part of the contract takes a new DLSSLOP_NETWORK_INTERFACE.
 * dlsslop_network.cpp is the module's source, and network_module.c is the layer's loader of it.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_LAYER_NETWORK_MODULE_H_
#define DLSSLOP_AMD_LAYER_NETWORK_MODULE_H_

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

struct DlsslopNetwork;
struct ShmHeader;

/** @brief The interface that the layer and the module share.
 *
 * It is odd and has its top bit set, so no pointer equals it: an older layer's struct
 * dlsslop_network_device begins with an aligned VkInstance where this one begins with this number.
 * Bits 8-31 count the versions: 1 with buffers, 2 with images.
 */
#define DLSSLOP_NETWORK_INTERFACE UINT64_C(0xd155100000000201)

/** @brief The module's DLSSLOP_NETWORK_INTERFACE, which network_module_load() requires to equal the
 *         layer's. */
extern STD(uint64_t) const dlsslop_network_interface;

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
	STD(uint64_t)                     interface;         //!< DLSSLOP_NETWORK_INTERFACE.
	VkInstance                        instance;          //!< The game's instance.
	VkPhysicalDevice                  physical;          //!< The device's physical device.
	VkDevice                          device;            //!< The device.
	VkQueue                           queue;             //!< The queue that takes the build's uploads.
	STD(uint32_t)                     family;            //!< The queue's family.
	dlsslop_network_queue_fn         *lock_queue;        //!< Called before each submit of the build.
	dlsslop_network_queue_fn         *unlock_queue;      //!< Called after each submit of the build.
	void                             *context;           //!< What lock_queue and unlock_queue take.
	PFN_vkGetInstanceProcAddr         physical_dispatch; //!< The next layer's vkGetInstanceProcAddr.
	VkPhysicalDeviceMemoryProperties  memory;            //!< The physical device's memory.
	dlsslop_network_log_fn           *log;               //!< Where the network's log lines go.
};

/** @brief The composition's images that a frame goes through.
 *
 * Both are width x height and of format, which is R8G8B8A8_UNORM or R16G16B16A16_SFLOAT. generation is
 * nonzero, and new whenever any handle may be: a destroyed image's handle can come back for a new one,
 * so the module never compares handles. The 32-bit members leave 4 bytes of trailing padding, which a
 * 64-bit height would only fill with casts.
 */
struct dlsslop_network_images {
	STD(uint64_t) generation;  //!< The composition's build of these images.
	VkImageView   input_view;  //!< The frame that the network samples.
	VkImage       answer;      //!< Takes the answer; storage and transfer-target usage.
	VkImageView   answer_view; //!< The answer's view.
	VkFormat      format;      //!< Both images' format.
	STD(uint32_t) width;       //!< Both images' width.
	STD(uint32_t) height;      //!< Both images' height.
};

#undef STD

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
 * @return       The network, or nullptr without memory or for a device of another interface.
 */
typedef struct DlsslopNetwork *
dlsslop_network_open_fn (struct dlsslop_network_device const *device);

/** @brief Readies the network for the next frame, in the composition's images.
 *
 * A frame of another extent starts a build in the background, which binds no images. Another shape
 * of the same extent reshapes the network here, and images of another generation are bound here.
 * Both take a fraction of a millisecond, and neither may overlap the network's recorded work: the
 * caller has waited for its last frame.
 *
 * @param network The network.
 * @param channel The channel, whose settings the frame takes.
 * @param images  The images that the frame goes through.
 * @return        What the network can do with the frame. Images of another format fail the network.
 */
typedef enum dlsslop_network_state
dlsslop_network_prepare_fn (struct DlsslopNetwork               *network,
                            struct ShmHeader const              *channel,
                            struct dlsslop_network_images const *images);

/** @brief Records the frame that dlsslop_network_prepare() readied, from its input image through the
 *         network into its answer image; if a wait of the frame runs out, the answer is the input.
 *
 * Before it, the caller's barriers make the input readable by compute shaders in
 * SHADER_READ_ONLY_OPTIMAL and the answer writable by compute shaders and transfers in GENERAL. After
 * it, the caller's barrier takes the answer's compute-shader and transfer writes. The network leaves
 * both images in those layouts and records no barrier on them.
 *
 * @param network The network.
 * @param cmd     The command buffer.
 * @return        DLSSLOP_NETWORK_READY or DLSSLOP_NETWORK_FAILED.
 */
typedef enum dlsslop_network_state
dlsslop_network_record_fn (struct DlsslopNetwork *network,
                           VkCommandBuffer        cmd);

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

/** @brief Loads the module at a path.
 *
 * A module of another interface, or without the functions, stays out of the game's process: its
 * interface is read first, and none of its functions is called.
 *
 * @param m    An empty module, or nullptr.
 * @param path The module's path.
 * @return     true if @a m holds the module's functions; otherwise it holds no module, and its
 *             failure says why.
 */
extern bool
network_module_load (struct network_module *m,
                     char const            *path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_LAYER_NETWORK_MODULE_H_ */
