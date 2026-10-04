/** @file
 *
 * Host test of files_replace(), through which dlsslopd and every game's in-layer network save the
 * one pipeline cache they share. Two writers replace a file at once while a reader reads it: every
 * read must find the file, holding one writer's data whole, and no temporary file may stay behind.
 * A replacement that fails must leave the file and its directory as they were.
 */
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "error.h"
#include "files.h"
#include "support.h"

/** @brief The checks that failed. */
static unsigned failures;

/** @brief Fails the test with a message unless a condition holds.
 *
 * @param ok  The condition.
 * @param fmt A printf format for the message.
 * @param ... The format's arguments.
 * @return    @a ok.
 */
[[gnu::format(printf, 2, 3)]]
static bool
expect (bool        ok,
        char const *fmt,
        ...)
{
	if (ok)
		return true;

	va_list args;
	va_start(args, fmt);
	fputs("files test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	++failures;
	return false;
}

/** @brief Expects a call to have failed with given words.
 *
 * @param code  What the call returned.
 * @param e     Its error.
 * @param words The words it should have put there.
 */
static void
expect_error (enum error_code     code,
              struct error const *e,
              char const         *words)
{
	char const *const what = code ? e->what : "no error";
	expect(!strcmp(what, words), "expected \"%s\", got \"%s\"", words, what);
}

/** @brief Ends the test without memory for a value.
 *
 * @param p The value's memory, or nullptr.
 * @return  @a p.
 */
static void *
allocated (void *p)
{
	if (!p) {
		fputs("files test: out of memory\n", stderr);
		exit(1);
	}
	return p;
}

/** @brief Whether a directory entry is neither . nor ..: scandir()'s filter.
 *
 * @param entry The entry.
 * @return      Nonzero to list it.
 */
static int
not_dots (struct dirent const *entry)
{
	return strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..");
}

/** @brief Orders two directory entries by their names' bytes: scandir()'s comparator.
 *
 * @param a The first entry.
 * @param b The second.
 * @return  Less than, equal to or greater than 0 as @a a's name sorts before, with or after @a b's.
 */
static int
by_name (struct dirent const **a,
         struct dirent const **b)
{
	return strcmp((*a)->d_name, (*b)->d_name);
}

/** @brief The names in a directory but . and .., sorted and joined by spaces.
 *
 * @param directory The directory.
 * @return          The names, empty if the directory cannot be read; the caller frees them.
 */
static char *
listing (char const *directory)
{
	char *text = nullptr;
	size_t length = 0;
	FILE *const out = allocated(open_memstream(&text, &length));
	struct dirent **entries = nullptr;
	int const count = scandir(directory, &entries, not_dots, by_name);
	for (int i = 0; i < count; ++i) {
		fprintf(out, "%s%s", i ? " " : "", entries[i]->d_name);
		free(entries[i]);
		entries[i] = nullptr;
	}
	free(entries);
	entries = nullptr;
	// fclose() puts the names in the text, which may need memory.
	return allocated(fclose(out) ? nullptr : text);
}

/** @brief Expects a directory to hold given names and nothing else.
 *
 * @param directory The directory.
 * @param names     The names, sorted and joined by spaces.
 */
static void
expect_listing (char const *directory,
                char const *names)
{
	char *text = listing(directory);
	expect(!strcmp(text, names), "left in the directory: %s", text);
	free(text);
	text = nullptr;
}

/** @brief Bytes that a seed sets apart from another writer's.
 *
 * @param size The number of bytes; more than 0.
 * @param seed The seed.
 * @return     The bytes, which the caller frees.
 */
static uint8_t *
pattern (size_t   size,
         uint32_t seed)
{
	uint8_t *const data = allocated(malloc(size));
	for (size_t i = 0; i < size; ++i) {
		seed = seed * 1664525u + 1013904223u;
		data[i] = (uint8_t)(seed >> 24);
	}
	return data;
}

/** @brief Whether a file's bytes are given bytes.
 *
 * @param file The file's bytes.
 * @param data The bytes.
 * @param size Their number.
 * @return     true if they are the same.
 */
static bool
holds (struct files_data const *file,
       uint8_t const           *data,
       size_t                   size)
{
	return file->size == size && !memcmp(file->bytes, data, size);
}

/** @brief A path in a directory, which must have memory.
 *
 * @param directory        The directory.
 * @param directory_length The length of its path.
 * @param name             The name in it: a string literal, which sizeof measures; the empty
 *                         literal before it lets nothing else through.
 * @param length           Receives the path's length, or nullptr.
 * @return                 The path, which the caller frees.
 */
#define PATH_IN(directory, directory_length, name, length) \
	allocated(files_join(directory, directory_length, "" name, sizeof "" name - 1, length))

/** @brief A replaced file holds the new data, and only its owner may read it.
 *
 * @param directory An empty directory to work in.
 * @param length    The length of its path.
 */
static void
check_replace (char const *directory,
               size_t      length)
{
	size_t file_length;
	char *file = PATH_IN(directory, length, "file", &file_length);
	uint8_t *data = pattern(1000, 1);
	size_t const sizes[] = {1000, 0};
	bool replaced;
	for (size_t i = 0; i < sizeof sizes / sizeof *sizes; ++i) {
		replaced = expect(!files_replace(file, file_length, data, sizes[i], nullptr), "cannot replace %s",
		                  file);
		if (!replaced)
			break;
		struct files_data got;
		bool const ok = !files_read(&got, file, nullptr) && holds(&got, data, sizes[i]);
		expect(ok, "%s does not hold the %zu bytes written", file, sizes[i]);
		files_data_fini(&got);
	}
	if (replaced) {
		struct stat st;
		unsigned const mode = stat(file, &st) ? 0 : st.st_mode & 0777;
		expect(mode == 0600, "%s has mode %o, not 600", file, mode);
		expect_listing(directory, "file");
	}
	free(data);
	data = nullptr;
	free(file);
	file = nullptr;
}

/** @brief Replacements that fail leave nothing behind: in a missing directory, and in place of a
 *         directory, which rename(2) does not replace with a file.
 *
 * @param directory An empty directory to work in.
 * @param length    The length of its path.
 */
static void
check_failures (char const *directory,
                size_t      length)
{
	static char const data[] = "data";
	struct error e;
	size_t missing_length;
	char *missing = PATH_IN(directory, length, "missing/file", &missing_length);
	expect_error(files_replace(missing, missing_length, data, sizeof data - 1, &e), &e,
	             "No such file or directory");
	free(missing);
	missing = nullptr;

	size_t occupied_length;
	char *occupied = PATH_IN(directory, length, "occupied", &occupied_length);
	if (expect(!mkdir(occupied, 0700), "cannot create %s", occupied)) {
		expect_error(files_replace(occupied, occupied_length, data, sizeof data - 1, &e), &e,
		             "Is a directory");
		expect(files_is_directory(occupied), "%s is no longer a directory", occupied);
		expect_listing(directory, "occupied");
	}
	free(occupied);
	occupied = nullptr;
}

/** @brief A writer that replaces one file again and again. */
struct writer {
	char const    *file;    //!< The file.
	uint8_t const *data;    //!< What it writes.
	atomic_uint   *writing; //!< The writers still writing, which it leaves at its end.
	size_t         length;  //!< The length of the file's path.
	size_t         size;    //!< The number of bytes it writes.
	long           rounds;  //!< How many times it replaces the file.
	struct error   error;   //!< Its first failure's words; empty if none.
};

/** @brief Replaces a writer's file its number of rounds: a thread's start routine.
 *
 * @param arg The writer.
 * @return    nullptr.
 */
static void *
write_rounds (void *arg)
{
	struct writer *const w = arg;
	for (long round = 0; round < w->rounds; ++round) {
		struct error e;
		if (files_replace(w->file, w->length, w->data, w->size, &e) && !w->error.what[0])
			w->error = e;
	}
	atomic_fetch_sub_explicit(w->writing, 1, memory_order_release);
	return nullptr;
}

/** @brief Two writers replace one file a number of times each, one with 256 KiB and the other with
 *         384 KiB, while this thread reads it.
 *
 * @param directory An empty directory to work in.
 * @param length    The length of its path.
 * @param rounds    The replacements by each writer.
 */
static void
check_concurrent (char const *directory,
                  size_t      length,
                  long        rounds)
{
	size_t file_length;
	char *file = PATH_IN(directory, length, "vulkan-pipelines.cache", &file_length);
	size_t const sizes[2] = {256 << 10, 384 << 10};
	uint8_t *data[2] = {pattern(sizes[0], 1), pattern(sizes[1], 2)};
	if (expect(!files_replace(file, file_length, data[0], sizes[0], nullptr), "cannot write %s", file)) {
		atomic_uint writing = 2;
		struct writer writers[2];
		pthread_t threads[2];
		bool started[2];
		for (unsigned w = 0; w < 2; ++w) {
			writers[w] = (struct writer){.file = file, .data = data[w], .writing = &writing,
			                             .length = file_length, .size = sizes[w], .rounds = rounds};
			started[w] = expect(!pthread_create(&threads[w], nullptr, write_rounds, &writers[w]),
			                    "cannot start writer %u", w + 1);
			if (!started[w])
				atomic_fetch_sub_explicit(&writing, 1, memory_order_release);
		}

		size_t reads = 0, missing = 0, partial = 0, partial_bytes = 0;
		struct error missing_error;
		while (atomic_load_explicit(&writing, memory_order_acquire)) {
			++reads;
			struct files_data got;
			struct error e;
			if (files_read(&got, file, &e)) {
				if (!missing++)
					missing_error = e;
			} else if (!holds(&got, data[0], sizes[0]) && !holds(&got, data[1], sizes[1])) {
				if (!partial++)
					partial_bytes = got.size;
			}
			files_data_fini(&got);
		}
		for (unsigned w = 0; w < 2; ++w) {
			if (!started[w])
				continue;
			expect(!pthread_join(threads[w], nullptr), "cannot join writer %u", w + 1);
			expect(!writers[w].error.what[0], "writer %u failed: %s", w + 1, writers[w].error.what);
		}
		expect(!missing, "%zu of %zu reads found no file, the first with \"%s\"", missing, reads,
		       missing_error.what);
		expect(!partial, "%zu of %zu reads found neither writer's data whole, the first %zu bytes",
		       partial, reads, partial_bytes);

		struct files_data last;
		bool const whole = !files_read(&last, file, nullptr)
		                   && (holds(&last, data[0], sizes[0]) || holds(&last, data[1], sizes[1]));
		expect(whole, "the last file is neither writer's data whole");
		files_data_fini(&last);
		expect_listing(directory, "vulkan-pipelines.cache");
		printf("files test: %zu reads while 2 writers replaced the file %ld times each\n", reads, rounds);
	}
	for (unsigned w = 0; w < 2; ++w) {
		free(data[w]);
		data[w] = nullptr;
	}
	free(file);
	file = nullptr;
}

int
main (int    argc,
      char **argv)
{
	char const *const tmpdir = getenv("TMPDIR");
	char const *parent = tmpdir && *tmpdir ? tmpdir : "/tmp";
	long rounds = 1000;
	static struct option const options[] = {
		{"directory", required_argument, nullptr, 'd'},
		{"rounds",    required_argument, nullptr, 'r'},
		{"help",      no_argument,       nullptr, 'h'},
		{},
	};
	for (int code; (code = getopt_long(argc, argv, "+d:r:h", options, nullptr)) != -1;) {
		switch (code) {
		case 'd':
			parent = optarg;
			break;
		case 'r': {
			// strtol() reports a value out of range only through errno.
			char *end;
			errno = 0;
			rounds = strtol(optarg, &end, 10);
			if (*optarg && !*end && rounds > 0 && errno != ERANGE)
				break;
			fprintf(stderr, "files-test: --rounds wants a positive count, not \"%s\"\n", optarg);
			return 2;
		}
		case 'h':
			puts("Usage: files-test [OPTION]...\n"
			     "Checks that files_replace() replaces a file whole while other writers replace it too.\n"
			     " -d, --directory DIR  Work in a new directory in DIR (default: $TMPDIR, or /tmp without it)\n"
			     " -r, --rounds N       Replacements by each of the 2 writers (default: 1000)\n"
			     " -h, --help           Show help (default: off)");
			return 0;
		default:
			return 2;
		}
	}
	if (optind != argc)
		return 2;

	size_t root_length;
	char *root = support_temp_dir(parent, "files-test", &root_length);
	if (!root) {
		fprintf(stderr, "files test: cannot create a directory in %s\n", parent);
		return 1;
	}
	size_t replace_length, fail_length, concurrent_length;
	char *replace = PATH_IN(root, root_length, "replace", &replace_length);
	char *fail = PATH_IN(root, root_length, "fail", &fail_length);
	char *concurrent = PATH_IN(root, root_length, "concurrent", &concurrent_length);
	if (expect(!mkdir(replace, 0700) && !mkdir(fail, 0700) && !mkdir(concurrent, 0700),
	           "cannot create directories in %s", root)) {
		check_replace(replace, replace_length);
		check_failures(fail, fail_length);
		check_concurrent(concurrent, concurrent_length, rounds);
	}
	expect(support_remove_tree(root), "cannot remove %s", root);
	free(concurrent);
	concurrent = nullptr;
	free(fail);
	fail = nullptr;
	free(replace);
	replace = nullptr;
	free(root);
	root = nullptr;
	if (!failures)
		puts("files test: every check passed");
	return failures ? 1 : 0;
}

#undef PATH_IN
