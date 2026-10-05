/** @file
 *
 * Which engine runs the network: the one choice made at run time. Everything that an engine runs is
 * compiled for each engine (run_engine.inc), into run_vulkan_engine(), run_hip_engine() and
 * run_identity_engine(); open_run_engine() opens the engine that --backend selects and calls its
 * function. open.c defines open_run_engine().
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_OPEN_H_
#define DLSSLOP_AMD_BACKEND_OPEN_H_

#include <stdint.h>

#include "error.h"

struct hip_engine;
struct identity_engine;
struct options;
struct serving;
struct vulkan_engine;

/** @brief Opens the engine that --backend selects and runs it: serving, or the mode the options
 *         ask for (--diagnose, --self-test or the offline mode).
 *
 * The Vulkan engine is opened prepared; the HIP engine, which binds its own addresses, is made where
 * it runs, with the runtime that it opens; --test-identity takes neither. auto takes the Vulkan
 * network when its model is there, a device can run it, it builds and no option needs HIP, and
 * says why not before taking HIP.
 *
 * @param o       The options; the HIP device selected is recorded in them.
 * @param tier    The tier.
 * @param serving What serving serves with, or nullptr for the other modes.
 * @param e       Receives the words for what failed, or nullptr.
 * @return        ERROR_NONE, or the code of what failed.
 */
extern enum error_code
open_run_engine (struct options       *o,
                 uint32_t              tier,
                 struct serving const *serving,
                 struct error         *e);

/** @brief Runs the Vulkan engine (run_vulkan.c).
 *
 * @param o       The options.
 * @param engine  The engine, prepared.
 * @param serving What serving serves with, or nullptr for the mode the options ask for.
 * @param e       Receives the words for what failed, or nullptr.
 * @return        ERROR_NONE, or the code of what failed.
 */
extern enum error_code
run_vulkan_engine (struct options const *o,
                   struct vulkan_engine *engine,
                   struct serving const *serving,
                   struct error         *e);

/** @brief Runs the HIP engine (run_hip.c).
 *
 * @param o       The options.
 * @param engine  The engine, which this prepares.
 * @param serving What serving serves with, or nullptr for the mode the options ask for.
 * @param e       Receives the words for what failed, or nullptr.
 * @return        ERROR_NONE, or the code of what failed.
 */
extern enum error_code
run_hip_engine (struct options const *o,
                struct hip_engine    *engine,
                struct serving const *serving,
                struct error         *e);

/** @brief Runs the identity engine (run_identity.c).
 *
 * @param o       The options.
 * @param engine  The engine.
 * @param serving What serving serves with, or nullptr for the mode the options ask for.
 * @param e       Receives the words for what failed, or nullptr.
 * @return        ERROR_NONE, or the code of what failed.
 */
extern enum error_code
run_identity_engine (struct options const   *o,
                     struct identity_engine *engine,
                     struct serving const   *serving,
                     struct error           *e);

#endif /* DLSSLOP_AMD_BACKEND_OPEN_H_ */
