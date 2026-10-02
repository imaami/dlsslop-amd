/** @file
 *
 * One log sink for the whole layer.
 *
 * Split out of layer.cpp because the composition needs it too, and fixed
 * while moving: the previous implementation allocated a fresh std::mutex
 * on every single line and leaked it, which at present rates is a leak
 * per frame.
 *
 * The first call of log_printf(), log_verbose(), log_time_enabled() or
 * log_time_interval() reads the environment. Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_LOG_H_
#define DLSSLOP_AMD_LAYER_LOG_H_

#ifdef __cplusplus
# include <cstdint>
# define LOG_STD(x) std::x
extern "C" {
#else
# include <stdint.h>
# define LOG_STD(x) x
#endif

/** @brief Writes one line to the log.
 *
 * Writes nothing unless VKLayer_DLSS5, VKLAYER_DLSS5 or DLSSNR_ENABLE
 * starts with 1. The line goes to the file that DLSSNR_LOG names, opened
 * for appending, or to stderr if DLSSNR_LOG is unset or empty or the file
 * cannot be opened. The formatted text is cut to 2047 bytes, then written
 * after "[dlssnr-layer] " and before a newline, and flushed; a line whose
 * format fails is not written. Lines that threads write at the same time
 * do not mix. When the layer is unloaded or the process exits, the log
 * closes the file it opened and drops the lines that follow. If a thread
 * holds the log's lock at that moment, the file stays open until the
 * process ends. That happens at exit while another thread writes a line,
 * and in a child that fork() made while a thread wrote one. The programs
 * that the process executes do not inherit the file.
 *
 * @param fmt A printf format.
 * @param ... The format's arguments.
 */
[[gnu::format(printf, 1, 2)]]
extern void
log_printf (char const *fmt,
            ...);

/** @brief Whether DLSSNR_VERBOSE starts with 1.
 *
 * @return true if the layer logs every present.
 */
extern bool
log_verbose (void);

/** @brief Whether DLSSNR_TIME starts with 1.
 *
 * @return true if the layer logs its timings.
 */
extern bool
log_time_enabled (void);

/** @brief The number of frames between two timing lines.
 *
 * @return DLSSNR_TIME_EVERY if it is a positive number of at most 32 bits, otherwise 30.
 */
extern LOG_STD(uint32_t)
log_time_interval (void);

/** @brief The time of CLOCK_MONOTONIC.
 *
 * @return The time in milliseconds, or 0 if the clock cannot be read.
 */
extern double
log_now_ms (void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef LOG_STD

#endif /* DLSSLOP_AMD_LAYER_LOG_H_ */
