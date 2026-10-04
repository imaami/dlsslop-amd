/** @file
 *
 * The CPU references' goldens: deterministic fixtures, and the FNV-1a 64 of what a reference makes
 * of them. A build with another compiler, or a port of a reference, must reproduce every golden bit
 * for bit.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_TESTS_GOLDEN_H_
#define DLSSLOP_AMD_TESTS_GOLDEN_H_

#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/** @brief A value of a fixture: splitmix64's finalizer over both numbers.
 *
 * @param seed The fixture.
 * @param i    The value's index.
 * @return     The value.
 */
static inline uint64_t
golden_bits (uint32_t seed,
             uint64_t i)
{
	uint64_t z = (((uint64_t)seed << 40) ^ i) + UINT64_C(0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
	return z ^ (z >> 31);
}

/** @brief A value of a fixture, uniform in [0, 1) with 24 bits.
 *
 * @param seed The fixture.
 * @param i    The value's index.
 * @return     The value.
 */
static inline float
golden_unit (uint32_t seed,
             uint64_t i)
{
	return (float)(golden_bits(seed, i) >> 40) * 0x1p-24f;
}

/** @brief A value of a fixture as a network's answer: half of the values in [-0.25, 1.25), the
 *         other half of either sign, from 2^-27 to below 2^15, with every fraction bit set at random.
 *         Each rounds to a finite binary16.
 *
 * @param seed The fixture.
 * @param i    The value's index.
 * @return     The value.
 */
static inline float
golden_neural (uint32_t seed,
               uint64_t i)
{
	uint64_t const v = golden_bits(seed, i);
	if (v & 1)
		return (float)(v >> 40) * 0x1p-24f * 1.5f - 0.25f;
	uint32_t const word = (uint32_t)((v >> 32) & 0x80000000u) |
	                      ((uint32_t)(100 + ((v >> 1) & 0xffff) % 42) << 23) |
	                      (uint32_t)((v >> 17) & 0x7fffffu);
	float value;
	memcpy(&value, &word, sizeof value);
	return value;
}

/** @brief A value of a fixture as a binary16 of any sign, magnitude and fraction but a nonfinite
 *         one.
 *
 * @param seed The fixture.
 * @param i    The value's index.
 * @return     The binary16's bits.
 */
static inline uint16_t
golden_half (uint32_t seed,
             uint64_t i)
{
	uint16_t const h = (uint16_t)golden_bits(seed, i);
	return (h & 0x7c00) == 0x7c00 ? (uint16_t)(h ^ 0x0400) : h;
}

/** @brief The FNV-1a 64 of bytes.
 *
 * @param data  The bytes.
 * @param bytes Their number.
 * @return      The hash.
 */
static inline uint64_t
golden_fnv1a64 (void const *data,
                size_t      bytes)
{
	uint64_t hash = UINT64_C(0xcbf29ce484222325);
	for (unsigned char const *at = data; bytes--; ++at)
		hash = (hash ^ *at) * UINT64_C(0x100000001b3);
	return hash;
}

/** @brief Whether bytes hash to their golden. When they do not, it says which golden moved and to
 *         what, so that a change shows every golden it moves.
 *
 * @param data     The bytes.
 * @param bytes    Their number.
 * @param expected The golden.
 * @param fmt      A printf format for the golden's name.
 * @param ...      The format's arguments.
 * @return         true if they do.
 */
[[gnu::format(printf, 4, 5)]]
static inline bool
golden_check (void const *data,
              size_t      bytes,
              uint64_t    expected,
              char const *fmt,
              ...)
{
	uint64_t const actual = golden_fnv1a64(data, bytes);
	if (actual == expected)
		return true;

	va_list args;
	va_start(args, fmt);
	fputs("golden ", stderr);
	vfprintf(stderr, fmt, args);
	va_end(args);
	fprintf(stderr, ": fnv1a64=%016" PRIx64 ", not %016" PRIx64 "\n", actual, expected);
	return false;
}

#endif /* DLSSLOP_AMD_TESTS_GOLDEN_H_ */
