/** @file
 *
 * Helpers for the project's C: force_inline, container_of(), min_d() and now_ns().
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_UTIL_H_
#define DLSSLOP_AMD_COMMON_UTIL_H_

#include <stddef.h>
#include <stdint.h>
#include <time.h>

/** @brief Declares a function of a header that every call inlines. */
#define force_inline [[gnu::always_inline]] static inline

/** @brief The struct that holds a member.
 *
 * A pointer of another type than the member's draws a warning.
 * A pointer to a const member gives a pointer to a struct that is not const.
 *
 * @param ptr    A pointer to the member.
 * @param type   The struct's type.
 * @param member The member's name.
 * @return       A pointer to the struct.
 */
#define container_of(ptr, type, member) \
	((void)sizeof ((ptr) == &((type *)nullptr)->member), \
	 (type *)(void *)((char *)(ptr) - offsetof(type, member)))

/** @brief The smaller of two numbers, as std::min compares them: @a a unless @a b is less.
 *
 * @param a A number.
 * @param b Another number.
 * @return  @a b if it is less than @a a, otherwise @a a.
 */
force_inline double
min_d (double a,
       double b)
{
	return b < a ? b : a;
}

/** @brief The time of CLOCK_MONOTONIC.
 *
 * @return The time in nanoseconds, or 0 if the clock cannot be read.
 */
static inline uint64_t
now_ns (void)
{
	struct timespec now;
	if (clock_gettime(CLOCK_MONOTONIC, &now))
		return 0;
	return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

#endif /* DLSSLOP_AMD_COMMON_UTIL_H_ */
