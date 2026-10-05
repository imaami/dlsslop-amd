/** @file
 *
 * The HIP self-test's checks of the linux_native module's kernels on synthetic inputs, independent
 * of the model's weights: each kernel against its CPU reference, bit for bit, and the motion
 * history's behaviour. control_selftest.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_CONTROL_SELFTEST_H_
#define DLSSLOP_AMD_BACKEND_CONTROL_SELFTEST_H_

#include <stddef.h>

#include "error.h"
#include "native_kernels.h"

/** @brief Runs the checks of the native tuning, the motion history and the codec, printing a line
 *         to stdout for each that passes.
 *
 * @param kernels The kernels; the stream is idle before and after.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, ERROR_FAILED, or ERROR_REJECTED for a reference that refused its input.
 */
extern enum error_code
control_selftest_run (struct native_kernels const *kernels,
                      struct error                *e);

/** @brief Waits for the kernels' stream, then reads floats back from the device.
 *
 * @param kernels The kernels.
 * @param pointer The device memory, or nullptr, which fails.
 * @param samples Receives the floats.
 * @param count   Their number.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
control_selftest_read (struct native_kernels const *kernels,
                       void const                  *pointer,
                       float                       *samples,
                       size_t                       count,
                       struct error                *e);

/** @brief Compares a kernel's floats with its CPU reference's, bit for bit: both compute the same
 *         IEEE operations without contraction. A difference is printed to stderr with its first
 *         sample.
 *
 * @param actual   The kernel's floats.
 * @param expected The reference's.
 * @param count    Their number.
 * @param name     What made them, for the words.
 * @param e        Receives the words for a difference, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
control_selftest_compare (float const  *actual,
                          float const  *expected,
                          size_t        count,
                          char const   *name,
                          struct error *e);

#endif /* DLSSLOP_AMD_BACKEND_CONTROL_SELFTEST_H_ */
