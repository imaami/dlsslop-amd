/** @file
 *
 * The log's sink, its switches and its clock.
 */
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "log.h"

/** @brief What a log file owns. */
enum log_file_flags {
	LOG_FILE_STREAM_OPEN = 1 //!< log_file() opened the stream; log_file_fini() closes it.
};

/** @brief A stream that the log's lines go to. */
struct log_file {
	FILE      *stream; //!< Where lines go; nullptr: nowhere.
	uintptr_t  flags;  //!< enum log_file_flags, as wide as a pointer.
};

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER; //!< Guards log_sink.
static pthread_once_t  log_once  = PTHREAD_ONCE_INIT;         //!< Runs log_open().

static struct log_file log_sink;       //!< Where log_printf() writes.
static uint32_t        log_interval;   //!< What log_time_interval() returns.
static bool            log_verbose_on; //!< What log_verbose() returns.
static bool            log_time_on;    //!< What log_time_enabled() returns.

/** @brief Opens a log file.
 *
 * @param path The file to append to, or nullptr.
 * @return     A log file that owns the opened stream, or one that writes
 *             to stderr if @a path is nullptr or empty or does not open.
 */
static struct log_file
log_file (char const *path)
{
	FILE *stream = path && *path ? fopen(path, "a") : nullptr;
	if (!stream)
		return (struct log_file){ .stream = stderr };

	return (struct log_file){
		.stream = stream,
		.flags  = LOG_FILE_STREAM_OPEN
	};
}

/** @brief Closes the stream that a log file opened, if any, and leaves
 *         the log file empty.
 *
 * @param dest The log file, or nullptr.
 */
static void
log_file_fini (struct log_file *dest)
{
	if (dest) {
		if (dest->flags & LOG_FILE_STREAM_OPEN)
			fclose(dest->stream);

		*dest = (struct log_file){0};
	}
}

/** @brief Whether an environment variable starts with 1.
 *
 * @param name The variable's name.
 * @return     true if it does.
 */
static bool
log_env_on (char const *name)
{
	char const *value = getenv(name);
	return value && *value == '1';
}

/** @brief Whether the user asked for the layer, and so for its log.
 *
 * @return true if VKLayer_DLSS5, VKLAYER_DLSS5 or DLSSNR_ENABLE starts
 *         with 1.
 */
static bool
layer_requested (void)
{
	return log_env_on("VKLayer_DLSS5") || log_env_on("VKLAYER_DLSS5")
	       || log_env_on("DLSSNR_ENABLE");
}

/** @brief Reads the switches and, if the user asked for the layer, opens
 *         the sink.
 */
static void
log_open (void)
{
	log_verbose_on = log_env_on("DLSSNR_VERBOSE");
	log_time_on = log_env_on("DLSSNR_TIME");

	char const *every = getenv("DLSSNR_TIME_EVERY");
	int interval = every ? atoi(every) : 0;
	log_interval = interval > 0 ? (uint32_t)interval : 30;

	if (!layer_requested())
		return;

	struct log_file sink = log_file(getenv("DLSSNR_LOG"));

	// log_close() can run on another thread when the process exits.
	pthread_mutex_lock(&log_mutex);
	log_sink = sink;
	pthread_mutex_unlock(&log_mutex);
}

/** @brief Closes the sink when the layer is unloaded or the process
 *         exits. Lines that are logged after that are dropped.
 *
 * Leaves the sink alone if the mutex is held. No thread can hold it when
 * the layer is unloaded. At exit another thread can, and in a child that
 * fork() made while a thread wrote a line, the mutex stays held by a
 * thread that the child does not have. The process closes the file when
 * it ends.
 */
[[gnu::destructor]]
static void
log_close (void)
{
	if (pthread_mutex_trylock(&log_mutex))
		return;

	log_file_fini(&log_sink);
	pthread_mutex_unlock(&log_mutex);
}

void
log_printf (char const *fmt,
            ...)
{
	pthread_once(&log_once, log_open);
	pthread_mutex_lock(&log_mutex);
	if (log_sink.stream) {
		char line[2048];
		va_list args;
		va_start(args, fmt);
		vsnprintf(line, sizeof line, fmt, args);
		va_end(args);
		fprintf(log_sink.stream, "[dlssnr-layer] %s\n", line);
		fflush(log_sink.stream);
	}
	pthread_mutex_unlock(&log_mutex);
}

bool
log_verbose (void)
{
	pthread_once(&log_once, log_open);
	return log_verbose_on;
}

bool
log_time_enabled (void)
{
	pthread_once(&log_once, log_open);
	return log_time_on;
}

uint32_t
log_time_interval (void)
{
	pthread_once(&log_once, log_open);
	return log_interval;
}

double
log_now_ms (void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (double)(now.tv_sec * INT64_C(1000000000) + now.tv_nsec) / 1e6;
}
