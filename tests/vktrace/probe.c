/** @file
 *
 * A small Vulkan compute program for checking the tracing layer (vktrace-test.py). A setup
 * submission uploads a buffer through a mapped staging buffer and fills another; a one-shot
 * submission, from a command pool made and destroyed around it as a build's are, runs probe.spv
 * once; then two frames run it with different push constants, specialization and descriptor sets,
 * and read the result and a storage image back. It prints the FNV-1a 64 of what it wrote and read,
 * so that the trace's Upload, Readback and Hash lines can be checked against it. One descriptor
 * write and one copy each span two bindings, the result binding has an offset, and the storage
 * image ends frame 0 in SHADER_READ_ONLY_OPTIMAL, from which the layer's hashing has to move it and
 * to which it has to return it. With --destroy-during-hash, a second thread then submits a batch
 * that waits for the host, and the probe destroys a buffer and an image and frees another buffer's
 * memory while the layer waits for that batch before hashing what it selected at the submit. A
 * sparse buffer, where the device supports one, stays and is hashed. Then a second thread creates,
 * binds, destroys and frees buffers while the probe submits empty batches, which the layer hashes.
 * It runs on a CPU device only, with the Khronos validation layer enabled below the tracing layer,
 * and exits 77 without either.
 */
// SPDX-License-Identifier: MIT
#include <getopt.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <vulkan/vulkan.h>

#include "../support.h"

/** @brief The FNV-1a 64 offset basis. */
static constexpr uint64_t FNV_BASIS = UINT64_C(14695981039346656037);

/** @brief The elements of the probe's buffers: 32-bit words, one per invocation. */
static constexpr uint32_t ELEMENTS = 4096;

/** @brief The storage image's width. */
static constexpr uint32_t WIDTH = 64;

/** @brief The storage image's height: a texel per element. */
static constexpr uint32_t HEIGHT = ELEMENTS / WIDTH;

/** @brief The bytes of the input and of each half of the result buffer. */
static constexpr VkDeviceSize BYTES = ELEMENTS * 4;

/** @brief The word the setup fills the result buffer with. */
static constexpr uint32_t FILL = 0x5a0f3c01;

/** @brief The most words of SPIR-V the probe reads. */
static constexpr size_t CODE_WORDS = 64 << 10;

/** @brief The usage line. */
static char const USAGE[] = "Usage: vktrace-probe [OPTION]... PROBE.SPV\n";

/** @brief Goes on with an FNV-1a 64 over more bytes.
 *
 * @param data The bytes.
 * @param n    How many.
 * @param h    The hash so far; FNV_BASIS to start one.
 * @return     The hash.
 */
static uint64_t
fnv (void const *data,
     size_t      n,
     uint64_t    h)
{
	unsigned char const *const p = data;
	for (size_t i = 0; i < n; ++i)
		h = (h ^ p[i]) * UINT64_C(1099511628211);
	return h;
}

/** @brief Ends the probe if a Vulkan call failed.
 *
 * @param r    The call's result.
 * @param call The call, as written.
 */
static void
check (VkResult    r,
       char const *call)
{
	if (r != VK_SUCCESS) {
		fprintf(stderr, "%s failed: %d\n", call, r);
		exit(1);
	}
}

#define CHECK(x) check((x), #x)

/** @brief Memory on the heap, or the end of the probe.
 *
 * @param bytes The bytes.
 * @return      The memory, which the caller frees.
 */
static void *
allocate (size_t bytes)
{
	void *const p = malloc(bytes);
	if (!p) {
		fputs("probe: out of memory\n", stderr);
		exit(1);
	}
	return p;
}

/** @brief The device and what the probe needs of it on every thread. */
struct probe {
	VkPhysicalDeviceMemoryProperties memory; //!< Its memory types.
	VkDevice                         device; //!< The device.
};

/** @brief An allocation and its mapping. */
struct mem {
	VkDeviceMemory  memory; //!< The memory.
	void           *mapped; //!< Its mapping, or nullptr.
};

/** @brief Allocates memory of the first type that requirements allow and that has flags.
 *
 * @param p    The device.
 * @param req  The requirements.
 * @param want The flags.
 * @param map  Whether to map it.
 * @return     The memory.
 */
static struct mem
allocate_memory (struct probe const    *p,
                 VkMemoryRequirements   req,
                 VkMemoryPropertyFlags  want,
                 bool                   map)
{
	struct mem m = {};
	uint32_t type = 0;
	while (type < p->memory.memoryTypeCount
	       && !((req.memoryTypeBits >> type & 1) && (p->memory.memoryTypes[type].propertyFlags & want) == want))
		++type;
	VkMemoryAllocateInfo const ai = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = type,
	};
	CHECK(vkAllocateMemory(p->device, &ai, nullptr, &m.memory));
	if (map)
		CHECK(vkMapMemory(p->device, m.memory, 0, VK_WHOLE_SIZE, 0, &m.mapped));
	return m;
}

/** @brief Makes a buffer bound to memory of its own.
 *
 * @param p     The device.
 * @param size  Its bytes.
 * @param usage Its usage.
 * @param want  The memory's flags.
 * @param map   Whether to map the memory.
 * @param m     Receives the memory.
 * @return      The buffer.
 */
static VkBuffer
make_buffer (struct probe const    *p,
             VkDeviceSize           size,
             VkBufferUsageFlags     usage,
             VkMemoryPropertyFlags  want,
             bool                   map,
             struct mem            *m)
{
	VkBufferCreateInfo const bi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size  = size,
		.usage = usage,
	};
	VkBuffer b;
	CHECK(vkCreateBuffer(p->device, &bi, nullptr, &b));
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(p->device, b, &req);
	*m = allocate_memory(p, req, want, map);
	CHECK(vkBindBufferMemory(p->device, b, m->memory, 0));
	return b;
}

/** @brief Submits a command buffer with a fence and waits for it.
 *
 * @param p     The device.
 * @param queue The queue.
 * @param fence The fence.
 * @param c     The command buffer.
 */
static void
submit (struct probe const *p,
        VkQueue             queue,
        VkFence             fence,
        VkCommandBuffer     c)
{
	VkSubmitInfo const si = {
		.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers    = &c,
	};
	CHECK(vkResetFences(p->device, 1, &fence));
	CHECK(vkQueueSubmit(queue, 1, &si, fence));
	CHECK(vkWaitForFences(p->device, 1, &fence, VK_TRUE, UINT64_MAX));
}

/** @brief The trace's path, as the tracing layer finds it: VKTRACE_FILE or its default, with the first
 *         %p replaced by the process ID.
 *
 * @return The path, which the caller frees.
 */
static char *
trace_path (void)
{
	char const *const file = getenv("VKTRACE_FILE");
	char const *const pattern = file && *file ? file : "/tmp/vktrace.%p.log";
	char const *const pid = strstr(pattern, "%p");
	char *const path = pid ? support_format(nullptr, "%.*s%d%s", (int)(pid - pattern), pattern, (int)getpid(), pid + 2)
	                       : support_format(nullptr, "%s", pattern);
	if (!path) {
		fputs("probe: out of memory\n", stderr);
		exit(1);
	}
	return path;
}

/** @brief Waits until a trace holds a QueueSubmitResult line after the first submit line that waits
 *         for a semaphore, which the layer writes out before it waits for that submission.
 *
 * @param path The trace.
 * @return     false after 60 s.
 */
static bool
submitted (char const *path)
{
	for (uint32_t tries = 0; tries < 60000; ++tries) {
		char *text = support_read_file(path, nullptr);
		if (text) {
			char const *const wait = strstr(text, " wait=[sem");
			bool const found = wait && strstr(wait, " QueueSubmitResult ");
			free(text);
			text = nullptr;
			if (found)
				return true;
		}
		support_sleep_ms(1);
	}
	return false;
}

/** @brief What the thread that submits the waiting batch needs. */
struct waiter {
	struct probe const *probe;     //!< The device.
	VkQueue             queue;     //!< The queue.
	VkSemaphore         semaphore; //!< The timeline semaphore the batch waits for.
	VkFence             fence;     //!< The batch's fence.
};

/** @brief Submits a batch that waits for a timeline semaphore to reach 1.
 *
 * @param arg The struct waiter.
 * @return    nullptr.
 */
static void *
wait_batch (void *arg)
{
	struct waiter const *const w = arg;
	uint64_t const one = 1;
	VkTimelineSemaphoreSubmitInfo const values = {
		.sType                   = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.waitSemaphoreValueCount = 1,
		.pWaitSemaphoreValues    = &one,
	};
	VkPipelineStageFlags const stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkSubmitInfo const si = {
		.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext              = &values,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores    = &w->semaphore,
		.pWaitDstStageMask  = &stage,
	};
	CHECK(vkQueueSubmit(w->queue, 1, &si, w->fence));
	return nullptr;
}

/** @brief What the thread that churns buffers needs. */
struct churn {
	struct probe const *probe; //!< The device.
	atomic_bool         stop;  //!< Set when it should stop.
};

/** @brief Makes buffers that live for up to 300 us each, half of them losing their memory first,
 *         until told to stop.
 *
 * @param arg The struct churn.
 * @return    nullptr.
 */
static void *
churn_buffers (void *arg)
{
	struct churn *const c = arg;
	VkBufferUsageFlags const all = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	                               | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	for (uint32_t seed = 1; !atomic_load(&c->stop); seed = seed * 1103515245u + 12345u) {
		struct mem m;
		VkBuffer const b = make_buffer(c->probe, 4 << 20, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &m);
		struct timespec const hold = {.tv_nsec = (seed >> 16 & 0xffff) % 301 * 1000};
		// A sleep cut short only shortens the buffer's life.
		nanosleep(&hold, nullptr);
		if (seed & 1u << 16) {
			vkFreeMemory(c->probe->device, m.memory, nullptr);
			vkDestroyBuffer(c->probe->device, b, nullptr);
		} else {
			vkDestroyBuffer(c->probe->device, b, nullptr);
			vkFreeMemory(c->probe->device, m.memory, nullptr);
		}
	}
	return nullptr;
}

/** @brief Starts a thread, or ends the probe.
 *
 * @param thread Receives the thread.
 * @param run    Its function.
 * @param arg    Its argument.
 */
static void
start (pthread_t  *thread,
       void     *(*run)(void *),
       void       *arg)
{
	int const err = pthread_create(thread, nullptr, run, arg);
	if (err) {
		char buf[64];
		fprintf(stderr, "probe: cannot start a thread: %s\n", strerror_r(err, buf, sizeof buf));
		exit(1);
	}
}

/** @brief Waits for a thread to finish, or ends the probe.
 *
 * @param thread The thread.
 */
static void
join (pthread_t thread)
{
	int const err = pthread_join(thread, nullptr);
	if (err) {
		char buf[64];
		fprintf(stderr, "probe: cannot join a thread: %s\n", strerror_r(err, buf, sizeof buf));
		exit(1);
	}
}

/** @brief Three resources that VKTRACE_HASH=all selects at the next submission, after a setup
 *         submission that leaves the image in GENERAL, from which the layer would copy it, are
 *         destroyed while the layer waits for that submission; then buffers are made and destroyed
 *         at any moment while the layer hashes. The setup also fills the sparse buffer, which stays.
 *
 * @param p       The device.
 * @param image   How the storage image was made.
 * @param barrier A barrier of the storage image's.
 * @param filled  The setup's fill, ELEMENTS words.
 * @param queue   The queue.
 * @param fence   The fence.
 * @param cb      The frames' command buffer.
 * @param sparse  Whether the device has sparse buffers.
 */
static void
destroy_during_hash (struct probe const         *p,
                     VkImageCreateInfo const    *image,
                     VkImageMemoryBarrier const *barrier,
                     uint32_t const             *filled,
                     VkQueue                     queue,
                     VkFence                     fence,
                     VkCommandBuffer             cb,
                     bool                        sparse)
{
	VkDevice const device = p->device;
	VkBufferUsageFlags const all = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	                               | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	struct mem gone_memory;
	struct mem unbound_memory;
	struct mem sparse_memory = {};
	VkBuffer const gone = make_buffer(p, BYTES, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &gone_memory);
	VkBuffer const unbound = make_buffer(p, BYTES, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &unbound_memory);
	VkImage gone_image;
	CHECK(vkCreateImage(device, image, nullptr, &gone_image));
	VkMemoryRequirements ireq;
	vkGetImageMemoryRequirements(device, gone_image, &ireq);
	struct mem const gone_image_memory = allocate_memory(p, ireq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
	CHECK(vkBindImageMemory(device, gone_image, gone_image_memory.memory, 0));
	// Bound through the queue, which the layer does not trace: it has to copy the buffer as if it
	// were bound.
	VkBuffer sparse_buffer = VK_NULL_HANDLE;
	if (sparse) {
		VkBufferCreateInfo const sbi = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
			.size  = BYTES,
			.usage = all,
		};
		CHECK(vkCreateBuffer(device, &sbi, nullptr, &sparse_buffer));
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, sparse_buffer, &req);
		sparse_memory = allocate_memory(p, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
		VkSparseMemoryBind const bind = {0, req.size, sparse_memory.memory, 0, 0};
		VkSparseBufferMemoryBindInfo const binds = {sparse_buffer, 1, &bind};
		VkBindSparseInfo const bsi = {
			.sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
			.bufferBindCount = 1,
			.pBufferBinds    = &binds,
		};
		CHECK(vkResetFences(device, 1, &fence));
		CHECK(vkQueueBindSparse(queue, 1, &bsi, fence));
		CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
	}
	VkCommandBufferBeginInfo const once = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	CHECK(vkResetCommandBuffer(cb, 0));
	CHECK(vkBeginCommandBuffer(cb, &once));
	if (sparse)
		vkCmdFillBuffer(cb, sparse_buffer, 0, BYTES, FILL);
	VkImageMemoryBarrier general = *barrier;
	general.srcAccessMask = 0;
	general.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	general.image = gone_image;
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
	                     nullptr, 1, &general);
	CHECK(vkEndCommandBuffer(cb));
	submit(p, queue, fence, cb);
	// The layer selects them when the waiting batch is submitted and copies them once it has
	// completed: they are gone by then.
	VkSemaphoreTypeCreateInfo const timeline = {
		.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
	};
	VkSemaphoreCreateInfo const sci = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &timeline,
	};
	VkSemaphore semaphore;
	CHECK(vkCreateSemaphore(device, &sci, nullptr, &semaphore));
	CHECK(vkResetFences(device, 1, &fence));
	struct waiter const w = {.probe = p, .queue = queue, .semaphore = semaphore, .fence = fence};
	pthread_t waiter;
	start(&waiter, wait_batch, (void *)&w);
	char *trace = trace_path();
	if (!submitted(trace)) {
		fprintf(stderr, "probe: no waiting submission in %s\n", trace);
		exit(1);
	}
	free(trace);
	trace = nullptr;
	vkDestroyBuffer(device, gone, nullptr);
	vkFreeMemory(device, gone_memory.memory, nullptr);
	vkFreeMemory(device, unbound_memory.memory, nullptr);
	vkDestroyImage(device, gone_image, nullptr);
	vkFreeMemory(device, gone_image_memory.memory, nullptr);
	VkSemaphoreSignalInfo const signal = {
		.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
		.semaphore = semaphore,
		.value     = 1,
	};
	CHECK(vkSignalSemaphore(device, &signal));
	join(waiter);
	CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
	vkDestroyBuffer(device, unbound, nullptr);
	vkDestroyBuffer(device, sparse_buffer, nullptr);
	vkFreeMemory(device, sparse_memory.memory, nullptr);
	vkDestroySemaphore(device, semaphore, nullptr);
	// The same at any moment, while each submission is hashed.
	struct churn ch = {.probe = p, .stop = false};
	pthread_t churner;
	start(&churner, churn_buffers, &ch);
	VkSubmitInfo const empty = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
	for (uint32_t i = 0; i < 300; ++i) {
		CHECK(vkResetFences(device, 1, &fence));
		CHECK(vkQueueSubmit(queue, 1, &empty, fence));
		CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
	}
	atomic_store(&ch.stop, true);
	join(churner);
	if (sparse)
		printf("destroyed during hash, sparse=%016" PRIx64 "\n", fnv(filled, BYTES, FNV_BASIS));
	else
		puts("destroyed during hash, sparse=none");
}

int
main (int    argc,
      char **argv)
{
	bool split = false;
	bool destroy = false;
	struct option const options[] = {
		{"split-setup",         no_argument, nullptr, 's'},
		{"destroy-during-hash", no_argument, nullptr, 'd'},
		{"help",                no_argument, nullptr, 'h'},
		{},
	};
	for (int opt; (opt = getopt_long(argc, argv, "+sdh", options, nullptr)) != -1;) {
		if (opt == 's') {
			split = true;
		} else if (opt == 'd') {
			destroy = true;
		} else if (opt == 'h') {
			printf("%s", USAGE);
			puts("Runs probe.spv on a CPU Vulkan device as vktrace-test.py describes, printing the FNV-1a 64\n"
			     "of what it uploads and reads back. Exits 77 without a CPU device or the Khronos validation\n"
			     "layer.\n"
			     " -s, --split-setup          Upload from a staging buffer of its own in a submission of its\n"
			     "                            own, fill in another, and destroy the staging buffer after the\n"
			     "                            one-shot submission (default: off; one setup submission from\n"
			     "                            the frames' upload buffer)\n"
			     " -d, --destroy-during-hash  After the frames, submit from a second thread a batch that\n"
			     "                            waits for a timeline semaphore; once the trace shows the\n"
			     "                            submission, destroy a buffer and an image, free a third\n"
			     "                            buffer's memory, then signal the semaphore. A sparse buffer,\n"
			     "                            if the device supports one, stays. Then submit empty batches\n"
			     "                            while a second thread creates, binds, destroys and frees\n"
			     "                            buffers. It reads the trace where the tracing layer writes\n"
			     "                            it: VKTRACE_FILE, or /tmp/vktrace.%p.log when that is unset,\n"
			     "                            with %p as the process ID (default: off)\n"
			     " -h, --help                 Show this help and exit (default: off)");
			return 0;
		} else {
			fprintf(stderr, "%s(see --help)\n", USAGE);
			return 2;
		}
	}
	if (optind + 1 != argc) {
		fprintf(stderr, "%s(see --help)\n", USAGE);
		return 2;
	}
	uint32_t *code = allocate(CODE_WORDS * sizeof *code);
	FILE *const f = fopen(argv[optind], "rbe");
	size_t const words = f ? fread(code, sizeof *code, CODE_WORDS, f) : 0;
	bool const read = f && !ferror(f) && feof(f) && words;
	if ((f && fclose(f)) || !read) {
		fprintf(stderr, "probe: cannot read %s\n", argv[optind]);
		free(code);
		code = nullptr;
		return 1;
	}
	// The layers the loader adds from the environment sit above those the application enables:
	// validation checks the tracing layer's work too.
	char const *const validation = "VK_LAYER_KHRONOS_validation";
	uint32_t count = 0;
	CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
	VkLayerProperties *layers = allocate((count ? count : 1) * sizeof *layers);
	CHECK(vkEnumerateInstanceLayerProperties(&count, layers));
	bool installed = false;
	for (uint32_t i = 0; i < count && !installed; ++i)
		installed = !strcmp(layers[i].layerName, validation);
	free(layers);
	layers = nullptr;
	if (!installed) {
		fprintf(stderr, "probe: skipped, %s is not installed\n", validation);
		free(code);
		code = nullptr;
		return 77;
	}
	VkApplicationInfo const app = {
		.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "vktrace probe",
		.apiVersion       = VK_API_VERSION_1_3,
	};
	VkInstanceCreateInfo const ici = {
		.sType               = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo    = &app,
		.enabledLayerCount   = 1,
		.ppEnabledLayerNames = &validation,
	};
	VkInstance instance;
	CHECK(vkCreateInstance(&ici, nullptr, &instance));
	CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
	VkPhysicalDevice *devices = allocate((count ? count : 1) * sizeof *devices);
	CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < count && !physical; ++i) {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(devices[i], &properties);
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
			physical = devices[i];
	}
	free(devices);
	devices = nullptr;
	if (!physical) {
		fputs("probe: skipped, no CPU Vulkan device\n", stderr);
		vkDestroyInstance(instance, nullptr);
		free(code);
		code = nullptr;
		return 77;
	}
	struct probe p = {};
	vkGetPhysicalDeviceMemoryProperties(physical, &p.memory);
	VkPhysicalDeviceFeatures supported;
	vkGetPhysicalDeviceFeatures(physical, &supported);
	VkQueueFamilyProperties family;
	count = 1;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, &family);
	bool const sparse = destroy && supported.sparseBinding && (family.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
	float const priority = 1;
	VkDeviceQueueCreateInfo const qci = {
		.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = 0,
		.queueCount       = 1,
		.pQueuePriorities = &priority,
	};
	VkPhysicalDeviceVulkan13Features f13 = {
		.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.synchronization2 = VK_TRUE,
	};
	VkPhysicalDeviceVulkan12Features f12 = {
		.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext             = &f13,
		.shaderInt8        = VK_TRUE,
		.timelineSemaphore = destroy,
	};
	VkPhysicalDeviceFeatures2 f2 = {
		.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext    = &f12,
		.features = {.shaderInt64 = VK_TRUE, .sparseBinding = sparse},
	};
	VkDeviceCreateInfo const dci = {
		.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext                = &f2,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos    = &qci,
	};
	CHECK(vkCreateDevice(physical, &dci, nullptr, &p.device));
	VkDevice const device = p.device;
	VkQueue queue;
	vkGetDeviceQueue(device, 0, 0, &queue);

	VkBufferUsageFlags const all = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	                               | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags const host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	// The input a, and b, whose second half the shader writes; its first half keeps the setup's fill.
	struct mem mu;
	struct mem ma;
	struct mem mb;
	struct mem md;
	VkBuffer const upload = make_buffer(&p, BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, true, &mu);
	VkBuffer const a = make_buffer(&p, BYTES, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &ma);
	VkBuffer const b = make_buffer(&p, BYTES * 2, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &mb);
	VkBuffer const download = make_buffer(&p, BYTES * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, true, &md);
	uint32_t *const u = mu.mapped;
	for (uint32_t i = 0; i < ELEMENTS; ++i)
		u[i] = i * 2654435761u;

	VkImageCreateInfo const ii = {
		.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType   = VK_IMAGE_TYPE_2D,
		.format      = VK_FORMAT_R8G8B8A8_UNORM,
		.extent      = {WIDTH, HEIGHT, 1},
		.mipLevels   = 1,
		.arrayLayers = 1,
		.samples     = VK_SAMPLE_COUNT_1_BIT,
		.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
	};
	VkImage image;
	CHECK(vkCreateImage(device, &ii, nullptr, &image));
	VkMemoryRequirements ireq;
	vkGetImageMemoryRequirements(device, image, &ireq);
	struct mem const mi = allocate_memory(&p, ireq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
	CHECK(vkBindImageMemory(device, image, mi.memory, 0));
	VkImageViewCreateInfo const vi = {
		.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image            = image,
		.viewType         = VK_IMAGE_VIEW_TYPE_2D,
		.format           = ii.format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	VkImageView view;
	CHECK(vkCreateImageView(device, &vi, nullptr, &view));

	VkShaderModuleCreateInfo const smi = {
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = words * sizeof *code,
		.pCode    = code,
	};
	VkShaderModule module;
	CHECK(vkCreateShaderModule(device, &smi, nullptr, &module));
	free(code);
	code = nullptr;
	VkDescriptorSetLayoutBinding const lb[3] = {
		{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
		{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
		{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
	};
	VkDescriptorSetLayoutCreateInfo const dli = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings    = lb,
	};
	VkDescriptorSetLayout dsl;
	CHECK(vkCreateDescriptorSetLayout(device, &dli, nullptr, &dsl));
	VkPushConstantRange const pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
	VkPipelineLayoutCreateInfo const pli = {
		.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount         = 1,
		.pSetLayouts            = &dsl,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &pcr,
	};
	VkPipelineLayout layout;
	CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &layout));
	VkPipeline pipes[2];
	for (uint32_t k = 0; k < 2; ++k) {
		uint32_t const scale = 2 + k;
		VkSpecializationMapEntry const entry = {0, 0, 4};
		VkSpecializationInfo const spec = {1, &entry, 4, &scale};
		VkComputePipelineCreateInfo const cpi = {
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = {
				.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage               = VK_SHADER_STAGE_COMPUTE_BIT,
				.module              = module,
				.pName               = "main",
				.pSpecializationInfo = &spec,
			},
			.layout = layout,
		};
		CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipes[k]));
	}
	VkDescriptorPoolSize const sizes[2] = {
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
	};
	VkDescriptorPoolCreateInfo const dpi = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets       = 2,
		.poolSizeCount = 2,
		.pPoolSizes    = sizes,
	};
	VkDescriptorPool pool;
	CHECK(vkCreateDescriptorPool(device, &dpi, nullptr, &pool));
	VkDescriptorSetLayout const dsls[2] = {dsl, dsl};
	VkDescriptorSetAllocateInfo const dai = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool     = pool,
		.descriptorSetCount = 2,
		.pSetLayouts        = dsls,
	};
	VkDescriptorSet sets[2];
	CHECK(vkAllocateDescriptorSets(device, &dai, sets));
	// Bindings 0 and 1 in one write: its second descriptor goes to binding 1.
	VkDescriptorBufferInfo const bi[2] = {{a, 0, VK_WHOLE_SIZE}, {b, BYTES, BYTES}};
	VkDescriptorImageInfo const imi = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
	VkWriteDescriptorSet const w[2] = {
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = sets[0],
			.dstBinding      = 0,
			.descriptorCount = 2,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = bi,
		},
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = sets[0],
			.dstBinding      = 2,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.pImageInfo      = &imi,
		},
	};
	vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
	// Frame 1's set is a copy, bindings 0 and 1 again in one.
	VkCopyDescriptorSet const copies[2] = {
		{
			.sType           = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
			.srcSet          = sets[0],
			.dstSet          = sets[1],
			.descriptorCount = 2,
		},
		{
			.sType           = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
			.srcSet          = sets[0],
			.srcBinding      = 2,
			.dstSet          = sets[1],
			.dstBinding      = 2,
			.descriptorCount = 1,
		},
	};
	vkUpdateDescriptorSets(device, 0, nullptr, 2, copies);

	VkCommandPoolCreateInfo const cpci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
	};
	VkCommandPool cpool;
	CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cpool));
	VkCommandBufferAllocateInfo const cai = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = cpool,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cb;
	CHECK(vkAllocateCommandBuffers(device, &cai, &cb));
	VkFenceCreateInfo const fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	CHECK(vkCreateFence(device, &fci, nullptr, &fence));
	VkQueryPoolCreateInfo const qpi = {
		.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType  = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = 2,
	};
	VkQueryPool qp;
	CHECK(vkCreateQueryPool(device, &qpi, nullptr, &qp));
	VkCommandBufferBeginInfo const once = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	// Setup, which dispatches nothing: the input uploaded and b filled.
	uint32_t *filled = allocate(ELEMENTS * 2 * sizeof *filled);
	for (uint32_t i = 0; i < ELEMENTS * 2; ++i)
		filled[i] = FILL;
	VkBufferCopy const whole = {0, 0, BYTES};
	struct mem ms = {};
	VkBuffer staging = VK_NULL_HANDLE;
	CHECK(vkBeginCommandBuffer(cb, &once));
	if (split) {
		staging = make_buffer(&p, BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, true, &ms);
		memcpy(ms.mapped, u, BYTES);
		vkCmdCopyBuffer(cb, staging, a, 1, &whole);
		CHECK(vkEndCommandBuffer(cb));
		submit(&p, queue, fence, cb);
		CHECK(vkResetCommandBuffer(cb, 0));
		CHECK(vkBeginCommandBuffer(cb, &once));
	} else {
		vkCmdCopyBuffer(cb, upload, a, 1, &whole);
	}
	vkCmdFillBuffer(cb, b, 0, BYTES * 2, FILL);
	CHECK(vkEndCommandBuffer(cb));
	submit(&p, queue, fence, cb);
	printf("setup upload=%016" PRIx64 " fill=%016" PRIx64 "\n", fnv(u, BYTES, FNV_BASIS),
	       fnv(filled, BYTES * 2, FNV_BASIS));

	// What the shader writes to b's second half with scale 2 and ADD, and the FNV of all of b then.
	uint64_t const fill_fnv = fnv(filled, BYTES, FNV_BASIS);
	uint32_t *expected = allocate(ELEMENTS * sizeof *expected);
	for (uint32_t i = 0; i < ELEMENTS; ++i)
		expected[i] = u[i] * 2 + 5;

	// A build's one-shot dispatch: its command pool lives for this submission alone.
	VkCommandPoolCreateInfo const opci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	VkCommandPool opool;
	CHECK(vkCreateCommandPool(device, &opci, nullptr, &opool));
	VkCommandBufferAllocateInfo const oai = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = opool,
		.commandBufferCount = 1,
	};
	VkCommandBuffer ocb;
	CHECK(vkAllocateCommandBuffers(device, &oai, &ocb));
	CHECK(vkBeginCommandBuffer(ocb, &once));
	VkMemoryBarrier const filled_barrier = {
		.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	};
	VkImageMemoryBarrier ib = {
		.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
		.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout           = VK_IMAGE_LAYOUT_GENERAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = image,
		.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	vkCmdPipelineBarrier(ocb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
	                     &filled_barrier, 0, nullptr, 1, &ib);
	vkCmdBindPipeline(ocb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
	vkCmdBindDescriptorSets(ocb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[0], 0, nullptr);
	uint32_t const first[2] = {5, WIDTH};
	vkCmdPushConstants(ocb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, first);
	vkCmdDispatch(ocb, ELEMENTS / 64, 1, 1);
	CHECK(vkEndCommandBuffer(ocb));
	submit(&p, queue, fence, ocb);
	vkDestroyCommandPool(device, opool, nullptr);
	if (split) {
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, ms.memory, nullptr);
	}
	printf("oneshot result=%016" PRIx64 " b=%016" PRIx64 "\n", fnv(expected, BYTES, FNV_BASIS),
	       fnv(expected, BYTES, fill_fnv));
	free(expected);
	expected = nullptr;

	VkImageLayout current = VK_IMAGE_LAYOUT_GENERAL;
	for (uint32_t frame = 0; frame < 2; ++frame) {
		CHECK(vkResetCommandBuffer(cb, 0));
		CHECK(vkBeginCommandBuffer(cb, &once));
		vkCmdResetQueryPool(cb, qp, 0, 2);
		vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
		vkCmdCopyBuffer(cb, upload, a, 1, &whole);
		ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		ib.oldLayout = current;
		VkMemoryBarrier const mbar = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
		vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbar, 0, nullptr, 1, &ib);
		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[frame]);
		vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[frame], 0, nullptr);
		uint32_t const push[2] = {7 + frame, WIDTH};
		vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
		vkCmdDispatch(cb, ELEMENTS / 64, 1, 1);
		VkMemoryBarrier2 const m2 = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
		};
		VkImageMemoryBarrier2 const i2 = {
			.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
			.dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT,
			.oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image               = image,
			.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
		};
		VkDependencyInfo const dep = {
			.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.memoryBarrierCount      = 1,
			.pMemoryBarriers         = &m2,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers    = &i2,
		};
		vkCmdPipelineBarrier2(cb, &dep);
		VkBufferCopy const result = {BYTES, 0, BYTES};
		vkCmdCopyBuffer(cb, b, download, 1, &result);
		VkBufferImageCopy const region = {
			.bufferOffset     = BYTES,
			.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.imageExtent      = {WIDTH, HEIGHT, 1},
		};
		vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, download, 1, &region);
		VkMemoryBarrier const to_host = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		};
		vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
		                     nullptr, 0, nullptr);
		current = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		if (!frame) {
			// Left where the layer's hashing cannot copy from directly.
			VkImageMemoryBarrier sampled = ib;
			sampled.srcAccessMask = 0;
			sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sampled.oldLayout = current;
			sampled.newLayout = current = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
			                     nullptr, 0, nullptr, 1, &sampled);
		}
		vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
		CHECK(vkEndCommandBuffer(cb));
		submit(&p, queue, fence, cb);
		unsigned char const *const read_back = md.mapped;
		printf("frame %" PRIu32 " upload=%016" PRIx64 " result=%016" PRIx64 " image=%016" PRIx64 " b=%016" PRIx64
		       "\n", frame, fnv(mu.mapped, BYTES, FNV_BASIS), fnv(read_back, BYTES, FNV_BASIS),
		       fnv(read_back + BYTES, BYTES, FNV_BASIS), fnv(read_back, BYTES, fill_fnv));
	}

	if (destroy)
		destroy_during_hash(&p, &ii, &ib, filled, queue, fence, cb, sparse);
	free(filled);
	filled = nullptr;
	vkDeviceWaitIdle(device);
	vkDestroyQueryPool(device, qp, nullptr);
	vkDestroyFence(device, fence, nullptr);
	vkDestroyCommandPool(device, cpool, nullptr);
	vkDestroyDescriptorPool(device, pool, nullptr);
	for (uint32_t k = 0; k < 2; ++k)
		vkDestroyPipeline(device, pipes[k], nullptr);
	vkDestroyPipelineLayout(device, layout, nullptr);
	vkDestroyDescriptorSetLayout(device, dsl, nullptr);
	vkDestroyShaderModule(device, module, nullptr);
	vkDestroyImageView(device, view, nullptr);
	vkDestroyImage(device, image, nullptr);
	VkBuffer const buffers[] = {upload, a, b, download};
	for (size_t i = 0; i < sizeof buffers / sizeof *buffers; ++i)
		vkDestroyBuffer(device, buffers[i], nullptr);
	struct mem const *const memories[] = {&mu, &ma, &mb, &md, &mi};
	for (size_t i = 0; i < sizeof memories / sizeof *memories; ++i)
		vkFreeMemory(device, memories[i]->memory, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);
	return 0;
}

#undef CHECK
