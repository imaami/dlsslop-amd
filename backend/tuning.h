/** @file
 *
 * The check of a native tuning (kernel_args.h) that a frame asks for, which dlsslopd and the
 * in-layer network's module both make. tuning.c defines it.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_TUNING_H_
#define DLSSLOP_AMD_BACKEND_TUNING_H_

#include "error.h"
#include "kernel_args.h"

/** @brief Checks that a tuning is inside the controls' ranges.
 *
 * @param tuning The tuning.
 * @param e      Receives the words of a rejection, or nullptr.
 * @return       ERROR_NONE, or ERROR_REJECTED for an intensity, tone or structure that is not
 *               finite and within 0..4, or a sharpness that is not finite and within 0..1.
 */
extern enum error_code
tuning_validate (struct native_tuning const *tuning,
                 struct error               *e);

#endif /* DLSSLOP_AMD_BACKEND_TUNING_H_ */
