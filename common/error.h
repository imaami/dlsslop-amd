/** @file
 *
 * Errors as returned codes, with the words that the log and the channel need. A fallible function
 * returns an enum error_code and puts its words in the caller's struct error. error.c defines the
 * functions that put them there.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_ERROR_H_
#define DLSSLOP_AMD_COMMON_ERROR_H_

#ifdef __cplusplus
# include <cstdint>
#else
# include <stdint.h>
#endif

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief How a fallible function ended. */
enum error_code : STD(uint8_t) {
	ERROR_NONE,     //!< It did what it should.
	ERROR_FAILED,   //!< A fault: serving ends, the daemon exits 1.
	ERROR_REJECTED, //!< The request was at fault: the frame fails, serving goes on.
	ERROR_DROPPED,  //!< A rejection serving does not report: a wait of the Vulkan network ran out.
};

#undef STD

/** @brief The bytes of struct error's words, their terminating null included. */
#define ERROR_WHAT_BYTES 2048

/** @brief Why a function failed, in words for the log and the channel. */
struct error {
	char what[ERROR_WHAT_BYTES]; //!< The words and a null; cut to fit and ending in "..." if longer.
};

/** @brief Puts the words of a fault in an error.
 *
 * @param e   The error, or nullptr where no caller needs the words.
 * @param fmt A printf format.
 * @param ... The format's arguments.
 * @return    ERROR_FAILED.
 */
[[gnu::cold, gnu::format(printf, 2, 3)]]
extern enum error_code
error_fail (struct error *e,
            char const   *fmt,
            ...);

/** @brief Puts the words of a rejected request in an error.
 *
 * @param e   The error, or nullptr where no caller needs the words.
 * @param fmt A printf format.
 * @param ... The format's arguments.
 * @return    ERROR_REJECTED.
 */
[[gnu::cold, gnu::format(printf, 2, 3)]]
extern enum error_code
error_reject (struct error *e,
              char const   *fmt,
              ...);

/** @brief Puts the words of a dropped frame in an error.
 *
 * @param e   The error, or nullptr where no caller needs the words.
 * @param fmt A printf format.
 * @param ... The format's arguments.
 * @return    ERROR_DROPPED.
 */
[[gnu::cold, gnu::format(printf, 2, 3)]]
extern enum error_code
error_drop (struct error *e,
            char const   *fmt,
            ...);

/** @brief Puts a prefix before the words in an error.
 *
 * @param e   The error, holding words, or nullptr.
 * @param fmt A printf format for the prefix.
 * @param ... The format's arguments.
 */
[[gnu::cold, gnu::format(printf, 2, 3)]]
extern void
error_wrap (struct error *e,
            char const   *fmt,
            ...);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_ERROR_H_ */
