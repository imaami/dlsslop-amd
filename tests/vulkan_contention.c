/** @file
 *
 * vulkan-contention: bounded compute work on a Vulkan device from a process of its own, for the
 * hardware check of the Vulkan network's waits (VALIDATION.md). Another client whose workgroups
 * hold the compute units' shared memory for milliseconds can make the network's waits run out;
 * this is such a client, bounded: one dispatch a submission, of at most WORK workgroup
 * iterations, until the time is up or a submission takes longer than LONGEST_MS.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

// The load's SPIR-V, the uint32_t array kContentionSpv.
#include "tests/vulkan_contention.h"

/** @brief A submission that takes longer than this many milliseconds stops the load: the check
 *         wants work of milliseconds, far from amdgpu's job timeout of 2 s.
 */
static constexpr uint32_t LONGEST_MS = 100;

/** @brief The nanoseconds in a millisecond. */
static constexpr int64_t NS_PER_MS = 1000000;

/** @brief The nanoseconds in a second. */
static constexpr int64_t NS_PER_SECOND = 1000000000;

/** @brief Invocations a workgroup (vulkan_contention.comp). */
static constexpr uint32_t INVOCATIONS = 256;

/** @brief --device's value when not given: the first device with a queue of the kind asked for. */
static constexpr uint32_t DEVICE_ANY = UINT32_MAX;

/** @brief A queue family's index when there is none. */
static constexpr uint32_t FAMILY_NONE = UINT32_MAX;

/** @brief A number option's range and its value when not given. */
struct limit {
	uint32_t low;     //!< The smallest value.
	uint32_t high;    //!< The largest value.
	uint32_t initial; //!< The value when the option is not given.
};

/** @brief --seconds. */
static constexpr struct limit SECONDS = {1, 60, 40};

/** @brief --workgroups. */
static constexpr struct limit WORKGROUPS = {1, 4096, 64};

/** @brief --lds-bytes. */
static constexpr struct limit LDS_BYTES = {4, 65536, 65536};

/** @brief --iterations. */
static constexpr struct limit ITERATIONS = {1, 2000000, 1000000};

/** @brief --device. */
static constexpr struct limit DEVICE = {0, 255, DEVICE_ANY};

/** @brief A dispatch's work, workgroups times iterations, at most the defaults'. On an otherwise
 *         idle RX 9070 XT the defaults took 9-19 ms a submission, and the extremes of these
 *         limits, one workgroup or 32 of ITERATIONS.high, or 4096 of 4 bytes or 64 KiB, at most
 *         19 ms; beside the network's frames the defaults took up to 65 ms. No option asks for a
 *         dispatch near LONGEST_MS before the first one runs.
 */
static constexpr uint64_t WORK = (uint64_t)WORKGROUPS.initial * ITERATIONS.initial;

/** @brief The command line. */
struct settings {
	uint32_t     seconds;    //!< How long to load the device.
	uint32_t     workgroups; //!< Workgroups a dispatch.
	uint32_t     lds_bytes;  //!< Shared memory a workgroup holds.
	uint32_t     iterations; //!< Dependent multiply-adds an invocation runs.
	uint32_t     device;     //!< The device's index, or DEVICE_ANY.
	VkQueueFlags graphics;   //!< VK_QUEUE_GRAPHICS_BIT for a graphics queue, 0 for a compute-only one.
};

/** @brief A uint32_t's conversion in the help. */
#define U32 "%" PRIu32

/** @brief Prints the help, with the limits and defaults above. */
static void
usage (void)
{
	printf("Usage: vulkan-contention [OPTION]...\n"
	       "Loads a Vulkan device from a process of its own with bounded compute work, as another GPU\n"
	       "client can, for the hardware check of the Vulkan network's waits (VALIDATION.md): one\n"
	       "dispatch a submission, of workgroups that each hold shared memory and run a bounded loop,\n"
	       "until the time is up. A submission that takes longer than " U32 " ms stops the load.\n"
	       " -q, --queue KIND      compute: a queue of a compute-only family, the asynchronous compute\n"
	       "                       engine; graphics: one of a graphics family (default: compute)\n"
	       " -s, --seconds N       How long to load the device, " U32 ".." U32 " (default: " U32 ")\n"
	       " -w, --workgroups N    Workgroups of " U32 " invocations a dispatch, " U32 ".." U32
	       " (default: " U32 ")\n"
	       " -l, --lds-bytes N     Shared memory a workgroup holds, a multiple of 4 from " U32 " to " U32 " bytes\n"
	       "                       and at most the device's limit (default: " U32 ")\n"
	       " -i, --iterations N    Dependent multiply-adds an invocation runs, " U32 ".." U32 " and at most\n"
	       "                       %" PRIu64 " / workgroups (default: " U32 ")\n"
	       " -d, --device N        The Vulkan device by its index, " U32 ".." U32 " (default: the first with a\n"
	       "                       queue of KIND)\n"
	       " -h, --help            Show help (default: off)\n",
	       LONGEST_MS, SECONDS.low, SECONDS.high, SECONDS.initial, INVOCATIONS, WORKGROUPS.low,
	       WORKGROUPS.high, WORKGROUPS.initial, LDS_BYTES.low, LDS_BYTES.high, LDS_BYTES.initial,
	       ITERATIONS.low, ITERATIONS.high, WORK, ITERATIONS.initial, DEVICE.low, DEVICE.high);
}

#undef U32

/** @brief Reads the number of an option.
 *
 * @param name  The option's long name, for the message.
 * @param text  The option's value.
 * @param limit The number's range.
 * @param dest  Receives the number; untouched on a failure.
 * @return      Whether the value is a number in the range; false after a message.
 */
static bool
number (char const   *name,
        char const   *text,
        struct limit  limit,
        uint32_t     *dest)
{
	char *end;
	// strtoul() reports a value out of its range only through errno.
	errno = 0;
	unsigned long const n = strtoul(text, &end, 10);
	if (!errno && *text >= '0' && *text <= '9' && !*end && n >= limit.low && n <= limit.high) {
		*dest = (uint32_t)n;
		return true;
	}
	fprintf(stderr, "vulkan-contention: --%s takes a number from %" PRIu32 " to %" PRIu32 ", not \"%s\"\n",
	        name, limit.low, limit.high, text);
	return false;
}

/** @brief Set by SIGINT and SIGTERM: the load stops after its submission. */
static volatile sig_atomic_t stopping;

/** @brief Stops the load: the handler of SIGINT and SIGTERM. */
static void
stop (int)
{
	stopping = 1;
}

/** @brief Reads the monotonic clock.
 *
 * @param ns Receives its time in nanoseconds.
 * @return   Whether it could be read; false after a message.
 */
static bool
now (int64_t *ns)
{
	struct timespec t;
	if (clock_gettime(CLOCK_MONOTONIC, &t)) {
		perror("vulkan-contention: clock_gettime");
		return false;
	}
	*ns = t.tv_sec * NS_PER_SECOND + t.tv_nsec;
	return true;
}

/** @brief Says whether a Vulkan call succeeded.
 *
 * @param result What the call returned.
 * @param what   The call's name, for the message.
 * @return       Whether it succeeded; false after a message.
 */
static bool
check (VkResult    result,
       char const *what)
{
	if (result == VK_SUCCESS)
		return true;
	fprintf(stderr, "vulkan-contention: %s failed (VkResult %d)\n", what, (int)result);
	return false;
}

/** @brief What the load makes, destroyed once the device is idle. */
struct load {
	VkBuffer              buffer;     //!< The results, which nothing reads.
	VkDeviceMemory        memory;     //!< Their memory.
	VkDescriptorSetLayout set_layout; //!< The results' set's layout.
	VkPipelineLayout      layout;     //!< The pipeline's layout.
	VkShaderModule        module;     //!< The load's shader.
	VkPipeline            pipeline;   //!< The load's pipeline.
	VkDescriptorPool      pool;       //!< The pool of the results' set.
	VkCommandPool         commands;   //!< The pool of the one command buffer.
	VkFence               fence;      //!< Signalled by each submission.
	VkInstance            instance;   //!< The instance.
	VkDevice              device;     //!< The device.
};

/** @brief Destroys what a load holds once its device is idle, then leaves it empty.
 *
 * @param l The load.
 */
static void
load_fini (struct load *l)
{
	if (l->device) {
		(void)vkDeviceWaitIdle(l->device);
		vkDestroyFence(l->device, l->fence, nullptr);
		vkDestroyCommandPool(l->device, l->commands, nullptr);
		vkDestroyDescriptorPool(l->device, l->pool, nullptr);
		vkDestroyPipeline(l->device, l->pipeline, nullptr);
		vkDestroyShaderModule(l->device, l->module, nullptr);
		vkDestroyPipelineLayout(l->device, l->layout, nullptr);
		vkDestroyDescriptorSetLayout(l->device, l->set_layout, nullptr);
		vkDestroyBuffer(l->device, l->buffer, nullptr);
		vkFreeMemory(l->device, l->memory, nullptr);
		vkDestroyDevice(l->device, nullptr);
	}
	if (l->instance)
		vkDestroyInstance(l->instance, nullptr);
	*l = (struct load){};
}

/** @brief Finds the first queue family of a device that the load asks for.
 *
 * @param physical The device.
 * @param graphics VK_QUEUE_GRAPHICS_BIT for a graphics family, 0 for a compute-only one.
 * @param family   Receives the family's index, or FAMILY_NONE when the device has none.
 * @return         Whether the families could be read; false after a message.
 */
static bool
family_of (VkPhysicalDevice  physical,
           VkQueueFlags      graphics,
           uint32_t         *family)
{
	*family = FAMILY_NONE;
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
	if (!count)
		return true;
	VkQueueFamilyProperties *families = malloc(count * sizeof *families);
	if (!families) {
		fputs("vulkan-contention: out of memory\n", stderr);
		return false;
	}
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
	// A compute family that has graphics exactly when the load asks for it.
	VkQueueFlags const kinds = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
	for (uint32_t i = 0; i < count && *family == FAMILY_NONE; ++i)
		if ((families[i].queueFlags & kinds) == (VK_QUEUE_COMPUTE_BIT | graphics))
			*family = i;
	free(families);
	families = nullptr;
	return true;
}

/** @brief Says that no device has what the settings ask for.
 *
 * @param s     The settings.
 * @param count The devices there are.
 * @return      FAMILY_NONE.
 */
static uint32_t
no_device (struct settings const *s,
           uint32_t               count)
{
	if (s->device != DEVICE_ANY && s->device >= count)
		fprintf(stderr, "vulkan-contention: no Vulkan device %" PRIu32 "\n", s->device);
	else
		fprintf(stderr, "vulkan-contention: no Vulkan device with a %s queue\n",
		        s->graphics ? "graphics" : "compute-only");
	return FAMILY_NONE;
}

/** @brief Picks the device and the queue family that the settings ask for.
 *
 * @param instance The instance.
 * @param s        The settings.
 * @param physical Receives the device when one is picked.
 * @return         The family's index, or FAMILY_NONE after a message.
 */
static uint32_t
pick (VkInstance             instance,
      struct settings const *s,
      VkPhysicalDevice      *physical)
{
	uint32_t count = 0;
	if (!check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices"))
		return FAMILY_NONE;
	if (!count)
		return no_device(s, count);
	VkPhysicalDevice *devices = malloc(count * sizeof *devices);
	if (!devices) {
		fputs("vulkan-contention: out of memory\n", stderr);
		return FAMILY_NONE;
	}
	uint32_t family = FAMILY_NONE;
	bool ok = check(vkEnumeratePhysicalDevices(instance, &count, devices), "vkEnumeratePhysicalDevices");
	for (uint32_t i = 0; ok && family == FAMILY_NONE && i < count; ++i) {
		if (s->device != DEVICE_ANY && s->device != i)
			continue;
		*physical = devices[i];
		ok = family_of(devices[i], s->graphics, &family);
	}
	free(devices);
	devices = nullptr;
	if (ok && family == FAMILY_NONE)
		return no_device(s, count);
	return family;
}

/** @brief Makes the load and runs it until the time is up, a signal stops it or a submission takes
 *         longer than LONGEST_MS.
 *
 * @param l The load, empty; load_fini() destroys what it holds after any return.
 * @param s The settings.
 * @return  Whether the load ran to its end; false after a message.
 */
static bool
load_run (struct load           *l,
          struct settings const *s)
{
	VkApplicationInfo const app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "vulkan-contention",
		.apiVersion = VK_API_VERSION_1_3,
	};
	VkInstanceCreateInfo const instance_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
	};
	VkInstance instance;
	if (!check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance"))
		return false;
	l->instance = instance;
	VkPhysicalDevice physical;
	uint32_t const family = pick(l->instance, s, &physical);
	if (family == FAMILY_NONE)
		return false;
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(physical, &properties);
	if (s->lds_bytes > properties.limits.maxComputeSharedMemorySize) {
		fprintf(stderr, "vulkan-contention: %s holds at most %" PRIu32 " bytes of shared memory a workgroup\n",
		        properties.deviceName, properties.limits.maxComputeSharedMemorySize);
		return false;
	}
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo const queue_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = family,
		.queueCount = 1,
		.pQueuePriorities = &priority,
	};
	VkDeviceCreateInfo const device_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queue_info,
	};
	VkDevice device;
	if (!check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice"))
		return false;
	l->device = device;
	VkQueue queue;
	vkGetDeviceQueue(l->device, family, 0, &queue);

	// The results, which nothing reads, in device-local memory.
	VkBufferCreateInfo const buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = (VkDeviceSize)s->workgroups * INVOCATIONS * sizeof (float),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	};
	VkBuffer buffer;
	if (!check(vkCreateBuffer(l->device, &buffer_info, nullptr, &buffer), "vkCreateBuffer"))
		return false;
	l->buffer = buffer;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(l->device, l->buffer, &req);
	VkPhysicalDeviceMemoryProperties memory_properties;
	vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
	uint32_t type = 0;
	while (type < memory_properties.memoryTypeCount
	       && !((req.memoryTypeBits >> type & 1)
	            && (memory_properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
		++type;
	if (type == memory_properties.memoryTypeCount) {
		fprintf(stderr, "vulkan-contention: %s has no device-local memory for the results\n",
		        properties.deviceName);
		return false;
	}
	VkMemoryAllocateInfo const allocation = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = req.size,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory memory;
	if (!check(vkAllocateMemory(l->device, &allocation, nullptr, &memory), "vkAllocateMemory"))
		return false;
	l->memory = memory;
	if (!check(vkBindBufferMemory(l->device, l->buffer, l->memory, 0), "vkBindBufferMemory"))
		return false;

	// The pipeline: its shared memory sized by a specialization constant.
	VkDescriptorSetLayoutBinding const binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
	};
	VkDescriptorSetLayoutCreateInfo const set_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	VkDescriptorSetLayout set_layout;
	if (!check(vkCreateDescriptorSetLayout(l->device, &set_layout_info, nullptr, &set_layout),
	           "vkCreateDescriptorSetLayout"))
		return false;
	l->set_layout = set_layout;
	VkPushConstantRange const push = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof s->iterations,
	};
	VkPipelineLayoutCreateInfo const layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &l->set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push,
	};
	VkPipelineLayout layout;
	if (!check(vkCreatePipelineLayout(l->device, &layout_info, nullptr, &layout), "vkCreatePipelineLayout"))
		return false;
	l->layout = layout;
	VkShaderModuleCreateInfo const module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof kContentionSpv,
		.pCode = kContentionSpv,
	};
	VkShaderModule module;
	if (!check(vkCreateShaderModule(l->device, &module_info, nullptr, &module), "vkCreateShaderModule"))
		return false;
	l->module = module;
	uint32_t const shared_words = s->lds_bytes / 4;
	VkSpecializationMapEntry const entry = {
		.constantID = 0,
		.offset = 0,
		.size = sizeof shared_words,
	};
	VkSpecializationInfo const specialization = {
		.mapEntryCount = 1,
		.pMapEntries = &entry,
		.dataSize = sizeof shared_words,
		.pData = &shared_words,
	};
	VkComputePipelineCreateInfo const pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = l->module,
			.pName = "main",
			.pSpecializationInfo = &specialization,
		},
		.layout = l->layout,
	};
	VkPipeline pipeline;
	if (!check(vkCreateComputePipelines(l->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
	           "vkCreateComputePipelines"))
		return false;
	l->pipeline = pipeline;

	// Its set and the one dispatch that every submission runs.
	VkDescriptorPoolSize const size = {
		.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
	};
	VkDescriptorPoolCreateInfo const pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &size,
	};
	VkDescriptorPool pool;
	if (!check(vkCreateDescriptorPool(l->device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool"))
		return false;
	l->pool = pool;
	VkDescriptorSetAllocateInfo const set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = l->pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &l->set_layout,
	};
	VkDescriptorSet set;
	if (!check(vkAllocateDescriptorSets(l->device, &set_info, &set), "vkAllocateDescriptorSets"))
		return false;
	VkDescriptorBufferInfo const results = {
		.buffer = l->buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &results,
	};
	vkUpdateDescriptorSets(l->device, 1, &write, 0, nullptr);
	VkCommandPoolCreateInfo const commands_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.queueFamilyIndex = family,
	};
	VkCommandPool commands;
	if (!check(vkCreateCommandPool(l->device, &commands_info, nullptr, &commands), "vkCreateCommandPool"))
		return false;
	l->commands = commands;
	VkCommandBufferAllocateInfo const cmd_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = l->commands,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBufferBeginInfo const begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};
	VkCommandBuffer cmd;
	if (!check(vkAllocateCommandBuffers(l->device, &cmd_info, &cmd), "vkAllocateCommandBuffers")
	    || !check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer"))
		return false;
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->layout, 0, 1, &set, 0, nullptr);
	vkCmdPushConstants(cmd, l->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof s->iterations, &s->iterations);
	vkCmdDispatch(cmd, s->workgroups, 1, 1);
	if (!check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer"))
		return false;
	VkFenceCreateInfo const fence_info = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};
	VkFence fence;
	if (!check(vkCreateFence(l->device, &fence_info, nullptr, &fence), "vkCreateFence"))
		return false;
	l->fence = fence;

	fprintf(stderr,
	        "vulkan-contention: %" PRIu32 " workgroups of %" PRIu32 " bytes of shared memory and %" PRIu32
	        " iterations a dispatch on the %s queue of %s, for %" PRIu32 " s\n",
	        s->workgroups, s->lds_bytes, s->iterations, s->graphics ? "graphics" : "compute",
	        properties.deviceName, s->seconds);
	VkSubmitInfo const submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	int64_t start;
	if (!now(&start))
		return false;
	int64_t const end = start + s->seconds * NS_PER_SECOND;
	// When the last submission ended, or the load started.
	int64_t at = start;
	int64_t longest = 0;
	uint32_t submissions = 0;
	bool ok = true;
	while (ok && !stopping && at < end) {
		int64_t const submitted = at;
		if (!check(vkQueueSubmit(queue, 1, &submit, l->fence), "vkQueueSubmit")
		    || !check(vkWaitForFences(l->device, 1, &l->fence, VK_TRUE, 5 * NS_PER_SECOND), "vkWaitForFences")
		    || !check(vkResetFences(l->device, 1, &l->fence), "vkResetFences")
		    || !now(&at))
			return false;
		int64_t const took = at - submitted;
		if (took > longest)
			longest = took;
		++submissions;
		if (took > LONGEST_MS * NS_PER_MS) {
			fprintf(stderr,
			        "vulkan-contention: a submission took %.1f ms, longer than %" PRIu32
			        " ms; stopping (lower --iterations or --workgroups)\n",
			        took / 1e6, LONGEST_MS);
			ok = false;
		}
	}
	double const seconds = (at - start) / 1e9;
	fprintf(stderr, "vulkan-contention: %" PRIu32 " submissions in %.1f s, %.3f ms each, the longest %.3f ms\n",
	        submissions, seconds, submissions ? seconds * 1e3 / submissions : 0.0, longest / 1e6);
	return ok;
}

/** @brief Makes the load, runs it and destroys it.
 *
 * @param s The settings.
 * @return  Whether the load ran to its end.
 */
static bool
run (struct settings const *s)
{
	struct load l = {};
	bool const ok = load_run(&l, s);
	load_fini(&l);
	return ok;
}

int
main (int    argc,
      char **argv)
{
	static struct option const options[] = {
		{"queue",      required_argument, nullptr, 'q'},
		{"seconds",    required_argument, nullptr, 's'},
		{"workgroups", required_argument, nullptr, 'w'},
		{"lds-bytes",  required_argument, nullptr, 'l'},
		{"iterations", required_argument, nullptr, 'i'},
		{"device",     required_argument, nullptr, 'd'},
		{"help",       no_argument,       nullptr, 'h'},
		{}
	};
	struct settings s = {
		.seconds = SECONDS.initial,
		.workgroups = WORKGROUPS.initial,
		.lds_bytes = LDS_BYTES.initial,
		.iterations = ITERATIONS.initial,
		.device = DEVICE.initial,
	};
	for (int code; (code = getopt_long(argc, argv, "+q:s:w:l:i:d:h", options, nullptr)) != -1;) {
		bool ok;
		switch (code) {
		case 'q':
			s.graphics = strcmp(optarg, "graphics") ? 0 : VK_QUEUE_GRAPHICS_BIT;
			ok = s.graphics || !strcmp(optarg, "compute");
			if (!ok)
				fprintf(stderr, "vulkan-contention: --queue takes compute or graphics, not \"%s\"\n",
				        optarg);
			break;
		case 's':
			ok = number("seconds", optarg, SECONDS, &s.seconds);
			break;
		case 'w':
			ok = number("workgroups", optarg, WORKGROUPS, &s.workgroups);
			break;
		case 'l':
			ok = number("lds-bytes", optarg, LDS_BYTES, &s.lds_bytes);
			break;
		case 'i':
			ok = number("iterations", optarg, ITERATIONS, &s.iterations);
			break;
		case 'd':
			ok = number("device", optarg, DEVICE, &s.device);
			break;
		case 'h':
			usage();
			return 0;
		default:
			return 2;
		}
		if (!ok)
			return 2;
	}
	if (optind != argc) {
		fprintf(stderr, "vulkan-contention: unexpected argument \"%s\"; see --help\n", argv[optind]);
		return 2;
	}
	if (s.lds_bytes % 4) {
		fprintf(stderr, "vulkan-contention: --lds-bytes takes a multiple of 4, not %" PRIu32 "\n", s.lds_bytes);
		return 2;
	}
	if ((uint64_t)s.workgroups * s.iterations > WORK) {
		fprintf(stderr,
		        "vulkan-contention: --iterations takes at most %" PRIu64 " / workgroups, not %" PRIu32
		        " with %" PRIu32 "\n",
		        WORK, s.iterations, s.workgroups);
		return 2;
	}
	if (signal(SIGINT, stop) == SIG_ERR || signal(SIGTERM, stop) == SIG_ERR) {
		perror("vulkan-contention: signal");
		return 1;
	}
	return run(&s) ? 0 : 1;
}
