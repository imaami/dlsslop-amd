/** @file
 *
 * Helpers for the project's C: force_inline, container_of() and min_d().
 *
 * Plain C, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_UTIL_H_
#define DLSSLOP_AMD_COMMON_UTIL_H_

#ifdef __cplusplus
# include <cstddef>
#else
# include <stddef.h>
#endif

/** @brief Declares a function of a header that every call inlines. */
#define force_inline [[gnu::always_inline]] static inline

/** @brief The struct that holds a member.
 *
 * A pointer of another type than the member's does not compile in C++, and draws a warning in C.
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

#endif /* DLSSLOP_AMD_COMMON_UTIL_H_ */
