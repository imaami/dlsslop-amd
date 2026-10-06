/** @file
 *
 * dlsslopd's command line and config file. options.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_OPTIONS_H_
#define DLSSLOP_AMD_BACKEND_OPTIONS_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"

struct paths_home;

/** @brief Where the network runs (--backend). */
enum options_backend : uint8_t {
	OPTIONS_BACKEND_AUTO,   //!< Vulkan when its model is there and a device can run it, else HIP.
	OPTIONS_BACKEND_VULKAN, //!< The Vulkan network.
	OPTIONS_BACKEND_HIP,    //!< The HIP network.
};

/** @brief dlsslopd's options: options_init() makes the defaults, options_parse() reads them from
 *         the config file and the command line, and options_fini() frees them.
 *
 * Each string is a heap string with its length, or nullptr and 0 when it is unset; an empty one
 * counts as unset everywhere but in shm. No option sets shaders: it is found from the binary's
 * path, as the default of modules is.
 */
struct options {
	char                 *assets;              //!< --assets: the HIP model weights' directory.
	char                 *modules;             //!< --modules: the HIP modules' directory.
	char                 *shm;                 //!< --shm: the channel file.
	char                 *vulkan_model;        //!< --vulkan-model: the Vulkan network's model.
	char                 *shaders;             //!< The Vulkan network's SPIR-V directory.
	char                 *input;               //!< --input: the offline mode's input file.
	char                 *output;              //!< --output: the offline or self-test output file.
	char                 *trace_dir;           //!< --trace-dir: the trace directory; never empty.
	size_t                assets_length;       //!< The length of assets.
	size_t                modules_length;      //!< The length of modules.
	size_t                shm_length;          //!< The length of shm.
	size_t                vulkan_model_length; //!< The length of vulkan_model.
	size_t                shaders_length;      //!< The length of shaders.
	size_t                input_length;        //!< The length of input.
	size_t                output_length;       //!< The length of output.
	size_t                trace_dir_length;    //!< The length of trace_dir.
	uint32_t              width;               //!< --width: the offline image's width; 0 when unset.
	uint32_t              height;              //!< --height: the offline image's height; 0 when unset.
	uint32_t              self_test_runs;      //!< --self-test-runs.
	uint32_t              self_test_drops;     //!< --self-test-drops.
	uint32_t              idle_exit;           //!< --idle-exit: seconds; 0 for never.
	/** @brief --tier; 0 when unset: then a serving daemon keeps the channel's live tier. */
	uint32_t              tier;
	/** @brief --passes; 0 when unset: then a serving daemon keeps the channel's live count. */
	uint32_t              passes;
	int                   device;              //!< --device: the device's index, or -1 for the first.
	enum options_backend  backend;             //!< --backend.
	bool                  diagnose;            //!< --diagnose.
	bool                  self_test;           //!< --self-test.
	bool                  test_identity;       //!< --test-identity.
	bool                  once;                //!< --once.
	bool                  cpu_codec;           //!< --cpu-codec.
	bool                  performance;         //!< --performance.
	bool                  help;                //!< --help: the help is printed, and nothing else is set.
};

/** @brief Makes the options' defaults.
 *
 * @param dest Receives the defaults; nothing to free on a failure.
 * @param home The home, from paths_home().
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
options_init (struct options          *dest,
              struct paths_home const *home,
              struct error            *e);

/** @brief Reads the options from the config file and the command line.
 *
 * The defaults come first, then the config file's settings, then the command line's options.
 * --help prints the help to stdout and ends the parsing; an invalid option prints it to stderr.
 *
 * @param dest Receives the options, or only help for --help; nothing to free on a failure.
 * @param argc The number of arguments.
 * @param argv The arguments, which getopt_long() may reorder.
 * @param e    Receives the words for what is wrong, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
options_parse (struct options  *dest,
               int              argc,
               char           **argv,
               struct error    *e);

/** @brief Frees the options and empties them.
 *
 * @param options The options, or nullptr.
 */
extern void
options_fini (struct options *options);

/** @brief The first option set that only the HIP network serves: auto takes HIP for it and the
 *         Vulkan backend refuses it.
 *
 * @param options The options, or nullptr.
 * @return        The option's name, such as "--cpu-codec", or nullptr for none and for no options.
 */
extern char const *
options_hip_only (struct options const *options);

/** @brief The Vulkan model that dlsslopd loads without options: vulkan-model from its default
 *         config file, else the default path.
 *
 * @param dest   Receives the model's path, which the caller frees; nullptr for none or on a
 *               failure.
 * @param length Receives its length; 0 without a path.
 * @param home   The home, from paths_home().
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED for a config file that cannot be read or holds an
 *               invalid setting, or without memory.
 */
extern enum error_code
options_configured_vulkan_model (char                    **dest,
                                 size_t                   *length,
                                 struct paths_home const  *home,
                                 struct error             *e);

#endif /* DLSSLOP_AMD_BACKEND_OPTIONS_H_ */
