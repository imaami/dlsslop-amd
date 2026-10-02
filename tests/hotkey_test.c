/** @file
 *
 * The hotkeys on the host, without a real keyboard: hotkeys_fini() with fake keyboards and loaded X
 * libraries; the evdev reads through pipes; the sweeps of directories that stand in for /dev/input,
 * whose plain files are not keyboards unless this test's ioctl() makes them answer as devices do;
 * the choice of a backend when none is available; and the XInput2 backend on a private Xvfb,
 * pressed through XTEST, when Xvfb and libXtst are installed. Pressing a real key through either
 * backend is a check by hand (VALIDATION.md).
 */
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "hotkey_priv.h"
#include "log.h"

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("hotkey-test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief The size of the test's paths: TMPDIR, the test's directory in it, and the names in that. */
static constexpr size_t PATH_SIZE = 1024;

/** @brief Writes a directory's path and a name in it, which must fit. */
static void
join (char        path[static PATH_SIZE],
      char const *dir,
      char const *name)
{
	require(snprintf(path, PATH_SIZE, "%s/%s", dir, name) < (int)PATH_SIZE, "%s/%s is too long", dir, name);
}

/** @brief Writes eventN. */
static void
event_name (char     name[static 16],
            uint16_t event)
{
	require(snprintf(name, 16, "event%u", event) < 16, "event%u is too long", event);
}

/** @brief What the layer's log must hold when the test ends. */
static char expected_log[16384];

/** @brief The length of expected_log. */
static size_t expected_length;

/** @brief Adds a line to what the layer's log must hold, after the log's prefix. */
[[gnu::format(printf, 1, 2)]]
static void
expect_log (char const *fmt,
            ...)
{
	char line[PATH_SIZE + 256];
	va_list args;
	va_start(args, fmt);
	int const length = vsnprintf(line, sizeof line, fmt, args);
	va_end(args);
	require(length >= 0 && length < (int)sizeof line, "an expected log line is too long");

	size_t const room = sizeof expected_log - expected_length;
	int const added = snprintf(expected_log + expected_length, room, "[dlssnr-layer] %s\n", line);
	require(added >= 0, "an expected log line cannot be formatted");
	size_t const bytes = (size_t)added;
	require(bytes < room, "the expected log is full");
	expected_length += bytes;
}

/** @brief The line that the hotkeys log when no backend opens. */
static char const NO_WAY[] = "[hotkey] no way to read the keyboard here. /dev/input needs the 'input' "
                             "group or a uaccess ACL; without it only an X11 session can be read, and "
                             "never one inside gamescope.";

/** @brief What a fake device claims to be. */
enum fake_kind {
	FAKE_KEYBOARD,     //!< Keys, every key code among them.
	FAKE_WITHOUT_A,    //!< A keyboard without KEY_A.
	FAKE_WITHOUT_Z,    //!< A keyboard without KEY_Z.
	FAKE_POWER_BUTTON, //!< Keys: KEY_POWER only.
	FAKE_MOUSE,        //!< Relative axes and no keys, although EVIOCGBIT(EV_KEY) answers every code.
	FAKE_MUTE          //!< Keys, but EVIOCGBIT(EV_KEY) fails.
};

/** @brief What a kind of fake device answers. */
struct fake_answers {
	uint16_t type;    //!< Its event type, EV_KEY or EV_REL.
	uint16_t only;    //!< Its only key code, if not 0.
	uint16_t without; //!< The one key code it lacks, if not 0.
	bool     mute;    //!< EVIOCGBIT(EV_KEY) fails.
};

/** @brief What each kind of fake device answers. */
static struct fake_answers const FAKE_KINDS[] = {
	[FAKE_KEYBOARD]     = { .type = EV_KEY },
	[FAKE_WITHOUT_A]    = { .type = EV_KEY, .without = KEY_A },
	[FAKE_WITHOUT_Z]    = { .type = EV_KEY, .without = KEY_Z },
	[FAKE_POWER_BUTTON] = { .type = EV_KEY, .only = KEY_POWER },
	[FAKE_MOUSE]        = { .type = EV_REL },
	[FAKE_MUTE]         = { .type = EV_KEY, .mute = true }
};

/** @brief A file that answers evdev's ioctls as a device does. */
struct fake_device {
	dev_t          dev;    //!< The file's device.
	ino_t          ino;    //!< The file's inode.
	uint32_t       probes; //!< How many times its event types were asked for.
	enum fake_kind kind;   //!< What it claims to be.
	bool           dead;   //!< Unplugged: every ioctl fails with ENODEV.
};

/** @brief The fake devices. */
static struct fake_device fake_devices[16];

/** @brief The entries of fake_devices in use. */
static size_t fake_device_count;

/** @brief The fake device of a file, or nullptr. */
static struct fake_device *
fake_device_at (dev_t dev,
                ino_t ino)
{
	for (size_t i = 0; i < fake_device_count; ++i)
		if (fake_devices[i].dev == dev && fake_devices[i].ino == ino)
			return &fake_devices[i];
	return nullptr;
}

/** @brief Creates a file that answers evdev's ioctls as a device of a kind does.
 *
 * A file that takes the inode of a removed one takes its place among the fake devices.
 *
 * @param dir    The directory.
 * @param name   The file's name.
 * @param kind   What it claims to be.
 * @param events What a read of it returns, or nullptr.
 * @param size   The size of @a events.
 * @return       The device.
 */
static struct fake_device *
fake_device (char const     *dir,
             char const     *name,
             enum fake_kind  kind,
             void const     *events,
             size_t          size)
{
	char path[PATH_SIZE];
	join(path, dir, name);
	int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
	require(fd >= 0, "cannot create %s", path);
	struct stat st;
	require((!size || write(fd, events, size) == (ssize_t)size) && !fstat(fd, &st) && !close(fd),
	        "cannot write %s", path);
	fd = -1;

	struct fake_device *device = fake_device_at(st.st_dev, st.st_ino);
	if (!device) {
		require(fake_device_count < sizeof fake_devices / sizeof *fake_devices, "too many fake devices");
		device = &fake_devices[fake_device_count++];
	}
	*device = (struct fake_device){ .dev = st.st_dev, .ino = st.st_ino, .kind = kind };
	return device;
}

/** @brief The bits of an unsigned long. */
static constexpr uint32_t LONG_BITS = 8 * sizeof (unsigned long);

/** @brief Sets a bit of an evdev bit mask. */
static void
set_bit (unsigned long bits[],
         uint32_t      bit)
{
	bits[bit / LONG_BITS] |= 1UL << (bit % LONG_BITS);
}

/** @brief evdev's ioctls, answered for the fake devices and passed to the kernel for everything
 *         else.
 *
 * Defined in this program, it takes the C library's place for hotkey.c. EVIOCGVERSION answers 0, as
 * the kernel's does, and EVIOCGBIT the number of bytes it wrote.
 */
int
ioctl (int           fd,
       unsigned long request,
       ...)
{
	va_list args;
	va_start(args, request);
	void *const arg = va_arg(args, void *);
	va_end(args);

	struct stat st;
	struct fake_device *const device = fstat(fd, &st) ? nullptr : fake_device_at(st.st_dev, st.st_ino);
	if (!device)
		return (int)syscall(SYS_ioctl, fd, request, arg);

	if (device->dead) {
		errno = ENODEV;
		return -1;
	}
	if (request == EVIOCGVERSION) {
		*(int *)arg = EV_VERSION;
		return 0;
	}

	struct fake_answers const *const answers = &FAKE_KINDS[device->kind];
	size_t const size = _IOC_SIZE(request);
	unsigned long bits[(KEY_MAX + LONG_BITS) / LONG_BITS] = {0};
	if (request == EVIOCGBIT(0, size)) {
		++device->probes;
		set_bit(bits, answers->type);
	} else if (request == EVIOCGBIT(EV_KEY, size) && !answers->mute) {
		for (uint32_t key = 1; key <= KEY_MAX; ++key)
			if (answers->only ? key == answers->only : key != answers->without)
				set_bit(bits, key);
	} else {
		errno = EINVAL;
		return -1;
	}

	size_t const copied = size < sizeof bits ? size : sizeof bits;
	memcpy(arg, bits, copied);
	return (int)copied;
}

/** @brief Whether a descriptor is open. */
static bool
is_open (int fd)
{
	return fcntl(fd, F_GETFD) >= 0 || errno != EBADF;
}

/** @brief Whether a descriptor is a fake device's file. */
static bool
is_device (int                       fd,
           struct fake_device const *device)
{
	struct stat st;
	return !fstat(fd, &st) && st.st_dev == device->dev && st.st_ino == device->ino;
}

/** @brief How many descriptors the process has open. */
static uint32_t
open_descriptors (void)
{
	DIR *const dir = opendir("/proc/self/fd");
	require(dir, "cannot list /proc/self/fd");
	uint32_t n = 0;
	for (struct dirent *e; (e = readdir(dir));)
		n += e->d_name[0] != '.';
	closedir(dir);
	// Without the listing's own.
	return n - 1;
}

/** @brief Whether every byte of an object is zero. */
static bool
is_zero (void const *object,
         size_t      size)
{
	unsigned char const *const bytes = object;
	for (size_t i = 0; i < size; ++i)
		if (bytes[i])
			return false;
	return true;
}

/** @brief The file that descriptor 0 holds, which nothing may close. */
static struct stat descriptor_0;

/** @brief Whether descriptor 0 is still the file the test put there. */
static bool
descriptor_0_intact (void)
{
	struct stat st;
	return !fstat(0, &st) && st.st_dev == descriptor_0.st_dev && st.st_ino == descriptor_0.st_ino;
}

/** @brief Whether a library is loaded. */
static bool
loaded (char const *library)
{
	void *const handle = dlopen(library, RTLD_LAZY | RTLD_NOLOAD);
	if (handle)
		dlclose(handle);
	return handle;
}

/** @brief A pipe whose ends are close-on-exec and do not block. */
static void
make_pipe (int fds[2])
{
	require(!pipe2(fds, O_CLOEXEC | O_NONBLOCK), "pipe2 failed");
}

/** @brief Writes @a count copies of one input event into a pipe. */
static void
write_events (int      fd,
              uint16_t type,
              uint16_t code,
              int32_t  value,
              uint32_t count)
{
	struct input_event const event = { .type = type, .code = code, .value = value };
	for (uint32_t i = 0; i < count; ++i)
		require(write(fd, &event, sizeof event) == (ssize_t)sizeof event, "cannot write an event");
}

/** @brief How many times in a row a key reads as pressed. */
static uint32_t
presses (struct hotkeys *h,
         uint32_t        key)
{
	uint32_t n = 0;
	while (hotkeys_pressed(h, key))
		++n;
	return n;
}

/** @brief hotkeys_fini() closes the keyboards it opened, and nothing else, once. */
static void
check_fini (void)
{
	struct hotkeys empty = {0};
	hotkeys_fini(&empty);
	hotkeys_fini(nullptr);
	require(descriptor_0_intact(), "finishing empty hotkeys closed descriptor 0");

	int first[2] = { -1, -1 };
	int second[2] = { -1, -1 };
	make_pipe(first);
	make_pipe(second);
	int other = open("/dev/null", O_RDONLY | O_CLOEXEC);
	require(other >= 0, "cannot open /dev/null");
	struct hotkey_node *const nodes = malloc(4 * sizeof *nodes);
	require(nodes, "out of memory");
	// Every node whose fd is not -1 is a keyboard, whatever its flags say; the others own nothing.
	nodes[0] = (struct hotkey_node){ .fd = first[0], .event = 3 };
	nodes[1] = (struct hotkey_node){ .ino = 7, .fd = -1, .event = 4, .flags = HOTKEY_NODE_REJECTED };
	nodes[2] = (struct hotkey_node){ .ino = 8, .fd = second[0], .event = 9, .flags = HOTKEY_NODE_REJECTED };
	nodes[3] = (struct hotkey_node){ .fd = -1, .event = 12 };
	struct hotkeys h = {
		.nodes         = nodes,
		.last_scan     = 1.0,
		.node_count    = 4,
		.keyboards     = 2,
		.pending_total = 3,
		.flags         = HOTKEYS_OPENED | HOTKEYS_ANNOUNCED
	};
	h.pending[KEY_F10] = 3;
	hotkeys_fini(&h);
	require(!is_open(first[0]) && !is_open(second[0]), "the keyboards stayed open");
	first[0] = -1;
	second[0] = -1;
	require(is_open(first[1]) && is_open(second[1]) && is_open(other) && descriptor_0_intact(),
	        "hotkeys_fini() closed a descriptor that no keyboard owned");
	require(is_zero(&h, sizeof h), "finished hotkeys are not empty");

	// The keyboards' descriptors, reused: finishing again closes nothing.
	int reused[2] = {
		open("/dev/null", O_RDONLY | O_CLOEXEC),
		open("/dev/null", O_RDONLY | O_CLOEXEC)
	};
	require(reused[0] >= 0 && reused[1] >= 0, "cannot open /dev/null");
	hotkeys_fini(&h);
	require(is_open(reused[0]) && is_open(reused[1]) && is_open(other) && descriptor_0_intact(),
	        "finishing hotkeys twice closed descriptors");
	int *const opened[] = { &reused[0], &reused[1], &first[1], &second[1], &other };
	for (size_t i = 0; i < sizeof opened / sizeof *opened; ++i) {
		close(*opened[i]);
		*opened[i] = -1;
	}
}

/** @brief hotkeys_fini() unloads the X libraries that the XInput2 backend loaded. */
static void
check_fini_x11 (void)
{
	void *const probe = dlopen("libXi.so.6", RTLD_LAZY | RTLD_LOCAL);
	if (!probe) {
		printf("hotkey-test: libXi.so.6 does not load, so its unloading is not checked\n");
		return;
	}
	dlclose(probe);
	if (loaded("libXi.so.6")) {
		printf("hotkey-test: libXi.so.6 stays loaded, so its unloading is not checked\n");
		return;
	}

	struct hotkey_x11 *const x = calloc(1, sizeof *x);
	require(x, "out of memory");
	x->x11 = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
	x->xi = dlopen("libXi.so.6", RTLD_LAZY | RTLD_LOCAL);
	require(x->x11 && x->xi, "cannot load libX11 and libXi");
	struct hotkeys h = { .x11 = x, .flags = HOTKEYS_OPENED };
	hotkeys_fini(&h);
	bool const xi = loaded("libXi.so.6");
	bool const x11 = loaded("libX11.so.6");
	require(!xi && !x11, "hotkeys_fini() left %s%s%s loaded", xi ? "libXi" : "", xi && x11 ? " and " : "",
	        x11 ? "libX11" : "");
	require(is_zero(&h, sizeof h), "finished hotkeys are not empty");
}

/** @brief The evdev backend's reads, through a pipe that stands in for a keyboard. */
static void
check_reads (void)
{
	struct hotkeys h = {0};
	require(!hotkeys_pressed(nullptr, KEY_F10), "no hotkeys read a press");
	require(!hotkeys_pressed(&h, 0) && is_zero(&h, sizeof h), "an unbound key opened a backend");

	int keyboard[2] = { -1, -1 };
	make_pipe(keyboard);
	h.nodes = malloc(sizeof *h.nodes);
	require(h.nodes, "out of memory");
	*h.nodes = (struct hotkey_node){ .fd = keyboard[0] };
	h.node_count = 1;
	h.keyboards = 1;
	// Opened, and /dev/input looked at forever from now: no sweep while this runs.
	h.flags = HOTKEYS_OPENED | HOTKEYS_ANNOUNCED;
	h.last_scan = INFINITY;
	require(!hotkeys_pressed(&h, KEY_F10), "a key read as pressed with no event");

	// Only a key going down counts: not its release, its repeats or other event types.
	write_events(keyboard[1], EV_KEY, KEY_F10, 0, 1);
	write_events(keyboard[1], EV_KEY, KEY_F10, 2, 3);
	write_events(keyboard[1], EV_MSC, KEY_F10, 1, 1);
	write_events(keyboard[1], EV_SYN, SYN_REPORT, 0, 1);
	require(!presses(&h, KEY_F10), "a release, a repeat or another event read as a press");
	write_events(keyboard[1], EV_KEY, KEY_F10, 1, 1);
	write_events(keyboard[1], EV_KEY, KEY_A, 1, 2);
	write_events(keyboard[1], EV_KEY, KEY_F10, 1, 1);
	require(presses(&h, KEY_F10) == 2 && presses(&h, KEY_A) == 2, "presses of two keys did not read back");

	// More presses than one read takes: one call drains them all.
	write_events(keyboard[1], EV_KEY, KEY_F10, 1, 150);
	require(hotkeys_pressed(&h, KEY_F10) && h.pending_total == 149, "one call did not drain 150 presses");
	require(presses(&h, KEY_F10) == 149, "150 presses did not read back");

	// The presses of other keys are cleared once more than 256 wait while the key is not pressed.
	write_events(keyboard[1], EV_KEY, KEY_B, 1, 200);
	write_events(keyboard[1], EV_KEY, KEY_C, 1, 56);
	require(!hotkeys_pressed(&h, KEY_F10) && h.pending_total == 256, "256 waiting presses were not kept");
	write_events(keyboard[1], EV_KEY, KEY_C, 1, 1);
	require(!hotkeys_pressed(&h, KEY_F10) && !h.pending_total && is_zero(h.pending, sizeof h.pending),
	        "257 waiting presses were not cleared");
	require(!presses(&h, KEY_B) && !presses(&h, KEY_C), "cleared presses read back");

	// One key answers at most 255 presses that wait together.
	write_events(keyboard[1], EV_KEY, KEY_D, 1, 256);
	require(presses(&h, KEY_D) == 255, "a key did not answer 255 of 256 waiting presses");

	// Codes beyond the key codes count nothing, and are never pressed.
	h.pending_total = 0;
	write_events(keyboard[1], EV_KEY, KEY_MAX, 1, 1);
	write_events(keyboard[1], EV_KEY, KEY_CNT, 1, 1);
	write_events(keyboard[1], EV_KEY, UINT16_MAX, 1, 1);
	require(presses(&h, KEY_MAX) == 1 && !presses(&h, KEY_CNT) && !presses(&h, UINT16_MAX)
	        && !presses(&h, UINT32_MAX) && !h.pending_total, "a code beyond the key codes counted");

	close(keyboard[1]);
	keyboard[1] = -1;
	hotkeys_fini(&h);
	require(!is_open(keyboard[0]), "the keyboard stayed open");
	keyboard[0] = -1;
}

/** @brief Creates a plain file. */
static void
make_file (char const *dir,
           char const *name)
{
	char path[PATH_SIZE];
	join(path, dir, name);
	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
	require(fd >= 0 && !close(fd), "cannot create %s", path);
	fd = -1;
}

/** @brief Removes a file. */
static void
remove_file (char const *dir,
             char const *name)
{
	char path[PATH_SIZE];
	join(path, dir, name);
	require(!remove(path), "cannot remove %s", path);
}

/** @brief Takes away a file's permissions. */
static void
lock_file (char const *dir,
           char const *name)
{
	char path[PATH_SIZE];
	join(path, dir, name);
	require(!chmod(path, 0), "cannot chmod %s", path);
}

/** @brief The inode of a file. */
static ino_t
inode (char const *dir,
       char const *name)
{
	char path[PATH_SIZE];
	join(path, dir, name);
	struct stat st;
	require(!stat(path, &st), "cannot stat %s", path);
	return st.st_ino;
}

/** @brief The node of eventN, or nullptr. */
static struct hotkey_node const *
node_of (struct hotkeys const *h,
         uint16_t              event)
{
	for (uint32_t i = 0; i < h->node_count; ++i)
		if (h->nodes[i].event == event)
			return &h->nodes[i];
	return nullptr;
}

/** @brief Whether eventN is known as rejected with its file's inode. */
static bool
rejected (struct hotkeys const *h,
          char const           *dir,
          uint16_t              event)
{
	char name[16];
	event_name(name, event);
	struct hotkey_node const *const node = node_of(h, event);
	return node && node->flags == HOTKEY_NODE_REJECTED && node->fd == -1 && node->ino == inode(dir, name);
}

/** @brief Whether eventN is open as a keyboard, on a fake device's file. */
static bool
keyboard (struct hotkeys const     *h,
          uint16_t                  event,
          struct fake_device const *device)
{
	struct hotkey_node const *const node = node_of(h, event);
	return node && !node->flags && node->fd >= 0 && is_device(node->fd, device);
}

/** @brief The sweeps of a directory that stands in for /dev/input. Plain files open, but they are
 *         not keyboards.
 */
static void
check_sweeps (char const *dir)
{
	struct hotkeys h = {0};
	hotkeys_rescan_evdev(&h, "/nonexistent/input");
	require(is_zero(&h, sizeof h), "a sweep of a missing directory changed the hotkeys");

	// A keyboard that has gone away, and whose node has left the directory.
	int gone[2] = { -1, -1 };
	make_pipe(gone);
	h.nodes = malloc(sizeof *h.nodes);
	require(h.nodes, "out of memory");
	*h.nodes = (struct hotkey_node){ .fd = gone[0], .event = 20 };
	h.node_count = 1;
	h.keyboards = 1;

	static char const *const SWEPT_NAMES[] = { "event3", "event10", "event65535", "event65536", "eventX",
	                                           "event", "event1a", "event-1", "mouse0", "js0" };
	for (size_t i = 0; i < sizeof SWEPT_NAMES / sizeof *SWEPT_NAMES; ++i)
		make_file(dir, SWEPT_NAMES[i]);
	hotkeys_rescan_evdev(&h, dir);
	require(!is_open(gone[0]) && !h.keyboards, "a keyboard that went away stayed open");
	gone[0] = -1;
	require(h.flags == HOTKEYS_ANNOUNCED, "a sweep did not mark the hotkeys announced");
	require(h.node_count == 3 && rejected(&h, dir, 3) && rejected(&h, dir, 10) && rejected(&h, dir, 65535),
	        "a sweep found %u nodes, not events 3, 10 and 65535 rejected", h.node_count);
	close(gone[1]);
	gone[1] = -1;

	// Again: the same nodes, the same verdicts.
	hotkeys_rescan_evdev(&h, dir);
	require(h.node_count == 3 && rejected(&h, dir, 3) && rejected(&h, dir, 10) && rejected(&h, dir, 65535),
	        "a second sweep changed the nodes");

	// A new node at a path: another inode, judged again. The old file still exists when the new one
	// is made, so the inode differs.
	ino_t const old = inode(dir, "event3");
	make_file(dir, "event3.new");
	char from[PATH_SIZE], to[PATH_SIZE];
	join(from, dir, "event3.new");
	join(to, dir, "event3");
	require(!rename(from, to), "cannot replace event3");
	hotkeys_rescan_evdev(&h, dir);
	require(inode(dir, "event3") != old && rejected(&h, dir, 3), "a replaced node kept its old verdict");

	// A rejection is trusted while the inode stays: the node is not opened again, which a file that
	// no longer opens shows. Root opens it anyway.
	lock_file(dir, "event10");
	hotkeys_rescan_evdev(&h, dir);
	require(rejected(&h, dir, 10), "a rejected node was opened again");

	// A node that leaves the directory is forgotten.
	remove_file(dir, "event10");
	hotkeys_rescan_evdev(&h, dir);
	require(h.node_count == 2 && !node_of(&h, 10), "a removed node was not forgotten");

	// A node that does not open is known, but not rejected; one that is a directory opens.
	make_file(dir, "event7");
	lock_file(dir, "event7");
	char sub[PATH_SIZE];
	join(sub, dir, "event8");
	require(!mkdir(sub, 0700), "cannot create event8");
	hotkeys_rescan_evdev(&h, dir);
	struct hotkey_node const *const seven = node_of(&h, 7);
	require(seven && seven->fd == -1 && (geteuid() ? !seven->flags : seven->flags == HOTKEY_NODE_REJECTED),
	        "a node that does not open was remembered wrongly");
	require(rejected(&h, dir, 8) && h.node_count == 4, "a directory node was not rejected");

	// The table follows the directory as it grows and shrinks.
	for (uint16_t i = 100; i < 140; ++i) {
		char name[16];
		event_name(name, i);
		make_file(dir, name);
	}
	hotkeys_rescan_evdev(&h, dir);
	require(h.node_count == 44 && rejected(&h, dir, 139), "the table did not grow to 44 nodes");
	for (uint16_t i = 100; i < 140; ++i) {
		char name[16];
		event_name(name, i);
		remove_file(dir, name);
	}
	hotkeys_rescan_evdev(&h, dir);
	require(h.node_count == 4 && !h.keyboards, "the table did not shrink to 4 nodes");
	hotkeys_fini(&h);

	require(!rmdir(sub), "cannot remove event8");
	remove_file(dir, "event7");
	remove_file(dir, "event3");
	for (size_t i = 2; i < sizeof SWEPT_NAMES / sizeof *SWEPT_NAMES; ++i)
		remove_file(dir, SWEPT_NAMES[i]);
}

/** @brief The sweeps of a directory of fake devices: which nodes are keyboards, that a live keyboard
 *         stays open and is not judged again, that one that is gone closes, and the log of a keyboard
 *         that appears later.
 */
static void
check_keyboards (char const *parent)
{
	char dir[PATH_SIZE];
	join(dir, parent, "input");
	require(!mkdir(dir, 0700), "cannot create %s", dir);
	uint32_t const descriptors = open_descriptors();

	// A keyboard with a press of F10 to read, devices that are not keyboards, and a plain file.
	struct input_event const f10 = { .type = EV_KEY, .code = KEY_F10, .value = 1 };
	struct fake_device *const one = fake_device(dir, "event1", FAKE_KEYBOARD, &f10, sizeof f10);
	static struct {
		char const     *what;
		enum fake_kind  kind;
	} const OTHERS[] = {
		{ "keyboard without KEY_A", FAKE_WITHOUT_A },
		{ "keyboard without KEY_Z", FAKE_WITHOUT_Z },
		{ "power button", FAKE_POWER_BUTTON },
		{ "mouse", FAKE_MOUSE },
		{ "device whose keys are not told", FAKE_MUTE }
	};
	static constexpr size_t OTHER_COUNT = sizeof OTHERS / sizeof *OTHERS;
	struct fake_device *others[OTHER_COUNT];
	for (uint16_t i = 0; i < OTHER_COUNT; ++i) {
		char name[16];
		event_name(name, i + 2);
		others[i] = fake_device(dir, name, OTHERS[i].kind, nullptr, 0);
	}
	make_file(dir, "event7");

	struct hotkeys h = {0};
	hotkeys_rescan_evdev(&h, dir);
	require(h.keyboards == 1 && keyboard(&h, 1, one) && one->probes == 1, "the keyboard was not opened");
	for (uint16_t i = 0; i < OTHER_COUNT; ++i)
		require(rejected(&h, dir, i + 2) && others[i]->probes == 1, "a %s was not rejected",
		        OTHERS[i].what);
	require(rejected(&h, dir, 7) && h.node_count == 7, "a plain file was not rejected");

	// The keyboard's press reads back, once.
	h.flags |= HOTKEYS_OPENED;
	h.last_scan = INFINITY;
	require(hotkeys_pressed(&h, KEY_F10) && !hotkeys_pressed(&h, KEY_F10),
	        "the press on the keyboard did not read back once");

	// Again: the live keyboard stays open, and nothing is judged again.
	int const fd = node_of(&h, 1)->fd;
	hotkeys_rescan_evdev(&h, dir);
	require(h.keyboards == 1 && keyboard(&h, 1, one) && node_of(&h, 1)->fd == fd && one->probes == 1,
	        "a live keyboard was closed, or judged again");
	for (uint16_t i = 0; i < OTHER_COUNT; ++i)
		require(rejected(&h, dir, i + 2) && others[i]->probes == 1, "a %s was judged again",
		        OTHERS[i].what);

	// A keyboard that appears later is opened, and logged.
	struct fake_device *const late = fake_device(dir, "event21", FAKE_KEYBOARD, nullptr, 0);
	hotkeys_rescan_evdev(&h, dir);
	expect_log("[hotkey] picked up a keyboard that appeared later: %s/event21", dir);
	require(h.keyboards == 2 && keyboard(&h, 21, late) && keyboard(&h, 1, one),
	        "a later keyboard was not opened");

	// A live keyboard whose node has left the directory stays open.
	remove_file(dir, "event1");
	hotkeys_rescan_evdev(&h, dir);
	require(h.keyboards == 2 && keyboard(&h, 1, one) && node_of(&h, 1)->fd == fd,
	        "a live keyboard whose node left was dropped");

	// Once it is gone, it is closed and forgotten. Nothing else opens during the sweep, so its
	// descriptor stays closed.
	one->dead = true;
	hotkeys_rescan_evdev(&h, dir);
	require(h.keyboards == 1 && !node_of(&h, 1) && !is_open(fd), "a keyboard that is gone was kept");

	// One keyboard unplugged, and another plugged in at its path: the new one is opened and logged.
	struct fake_device *const replug = fake_device(dir, "event21.new", FAKE_KEYBOARD, nullptr, 0);
	char from[PATH_SIZE], to[PATH_SIZE];
	join(from, dir, "event21.new");
	join(to, dir, "event21");
	require(!rename(from, to), "cannot replace event21");
	late->dead = true;
	hotkeys_rescan_evdev(&h, dir);
	expect_log("[hotkey] picked up a keyboard that appeared later: %s/event21", dir);
	require(h.keyboards == 1 && keyboard(&h, 21, replug) && replug->probes == 1,
	        "a keyboard plugged in at the path of another was not opened");

	// A keyboard that is gone while its node is still there closes, and its node is rejected.
	replug->dead = true;
	hotkeys_rescan_evdev(&h, dir);
	require(!h.keyboards && rejected(&h, dir, 21), "a keyboard that is gone was not closed and rejected");

	hotkeys_fini(&h);
	require(open_descriptors() == descriptors, "the hotkeys did not close every descriptor they opened");
	static char const *const LEFT[] = { "event2", "event3", "event4", "event5", "event6", "event7",
	                                    "event21" };
	for (size_t i = 0; i < sizeof LEFT / sizeof *LEFT; ++i)
		remove_file(dir, LEFT[i]);
	require(!rmdir(dir), "cannot remove %s", dir);
	fake_device_count = 0;
}

/** @brief The process of the Xvfb that check_x11() started, or 0. */
static pid_t xvfb;

/** @brief Stops the Xvfb that check_x11() started, if it runs; also at exit, after a failed check. */
static void
stop_xvfb (void)
{
	if (!xvfb)
		return;

	kill(xvfb, SIGTERM);
	waitpid(xvfb, nullptr, 0);
	xvfb = 0;
}

/** @brief Runs Xvfb in a child of the test, which the kernel stops when the test ends, also by a
 *         crash. GLX is off, so that Xvfb loads no GPU driver.
 *
 * @param log    The file that receives Xvfb's output.
 * @param fd     The descriptor to which Xvfb writes its display's number.
 * @param parent The test's process.
 */
[[noreturn]] static void
exec_xvfb (char const *log,
           int         fd,
           pid_t       parent)
{
	char number[16];
	int const out = open(log, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
	if (prctl(PR_SET_PDEATHSIG, SIGTERM) || getppid() != parent || out < 0 || dup2(out, 1) < 0
	    || dup2(out, 2) < 0 || fcntl(fd, F_SETFD, 0) || snprintf(number, sizeof number, "%d", fd) < 0)
		_exit(126);

	char *const argv[] = { "Xvfb", "-displayfd", number, "-nolisten", "tcp", "-extension", "GLX",
	                       "-screen", "0", "64x64x24", nullptr };
	execvp("Xvfb", argv);
	_exit(errno == ENOENT ? 127 : 126);
}

/** @brief Starts Xvfb on a display that it chooses.
 *
 * @param log     The file that receives Xvfb's output.
 * @param display Receives the display's name, such as ":3".
 * @return        false if there is no Xvfb to start.
 */
static bool
start_xvfb (char const *log,
            char        display[static 16])
{
	int fds[2] = { -1, -1 };
	require(!pipe2(fds, O_CLOEXEC), "cannot make Xvfb's pipe");
	pid_t const parent = getpid();
	xvfb = fork();
	require(xvfb >= 0, "fork failed");
	if (!xvfb)
		exec_xvfb(log, fds[1], parent);
	close(fds[1]);
	fds[1] = -1;

	// Xvfb writes the display's number and a newline once it takes connections, and the child
	// writes nothing if it does not run Xvfb.
	char number[8] = {0};
	size_t length = 0;
	while (length < sizeof number - 1 && !memchr(number, '\n', length)) {
		struct pollfd ready = { .fd = fds[0], .events = POLLIN };
		require(poll(&ready, 1, 10000) == 1, "Xvfb did not tell its display in 10 s; see %s", log);
		ssize_t const n = read(fds[0], number + length, sizeof number - 1 - length);
		if (n <= 0)
			break;
		length += (size_t)n;
	}
	close(fds[0]);
	fds[0] = -1;
	if (!length) {
		int status = 0;
		require(waitpid(xvfb, &status, 0) == xvfb, "waitpid failed");
		xvfb = 0;
		require(WIFEXITED(status) && WEXITSTATUS(status) == 127, "Xvfb exited before it told its display; see %s",
		        log);
		return false;
	}
	require(number[length - 1] == '\n', "Xvfb told \"%s\", not a display's number", number);
	number[length - 1] = '\0';
	require(snprintf(display, 16, ":%s", number) < 16, "the display's name is too long");
	return true;
}

/** @brief libXtst's XTestFakeKeyEvent(), which this test loads rather than links. */
extern int
XTestFakeKeyEvent (Display       *display,
                   unsigned int   keycode,
                   Bool           is_press,
                   unsigned long  delay);

/** @brief Whether a press of a key reads back within five seconds. */
static bool
arrives (struct hotkeys *h,
         uint32_t        key)
{
	for (uint32_t ms = 0; ms < 5000; ++ms) {
		if (hotkeys_pressed(h, key))
			return true;
		nanosleep(&(struct timespec){ .tv_nsec = 1000000 }, nullptr);
	}
	return false;
}

/** @brief The XInput2 backend on a private Xvfb, pressed through XTEST: a press counts while the key
 *         is down, the presses count in the order they were sent, and the releases do not. Skipped
 *         without Xvfb or libXtst.
 */
static void
check_x11 (char const *dir)
{
	void *const x11 = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
	void *const xtst = x11 ? dlopen("libXtst.so.6", RTLD_LAZY | RTLD_LOCAL) : nullptr;
	if (!xtst) {
		printf("hotkey-test: libX11.so.6 or libXtst.so.6 does not load, so XInput2 is not checked\n");
		if (x11)
			dlclose(x11);
		return;
	}

#define X_SYMBOL(lib, name) ((typeof (name) *)dlsym(lib, #name))
	typeof (XOpenDisplay) *const open_display = X_SYMBOL(x11, XOpenDisplay);
	typeof (XSync) *const sync_display = X_SYMBOL(x11, XSync);
	typeof (XCloseDisplay) *const close_display = X_SYMBOL(x11, XCloseDisplay);
	typeof (XTestFakeKeyEvent) *const fake_key_event = X_SYMBOL(xtst, XTestFakeKeyEvent);
#undef X_SYMBOL
	require(open_display && sync_display && close_display && fake_key_event,
	        "libX11 or libXtst lacks a function");

	char log[PATH_SIZE];
	join(log, dir, "xvfb.log");
	char display[16];
	if (!start_xvfb(log, display)) {
		printf("hotkey-test: Xvfb is not installed, so XInput2 is not checked\n");
		require(!remove(log), "cannot remove %s", log);
		dlclose(xtst);
		dlclose(x11);
		return;
	}

	require(!setenv("DISPLAY", display, 1) && !setenv("DLSSNR_HOTKEY_BACKEND", "x11", 1), "setenv failed");
	struct hotkeys h = {0};
	require(!hotkeys_pressed(&h, KEY_F10) && h.x11 && h.flags == HOTKEYS_OPENED && !h.nodes && !h.keyboards,
	        "the XInput2 backend did not open on Xvfb %s", display);
	expect_log("[hotkey] watching XInput2 raw keys on %s", display);

	// F10 goes down, and its press arrives while it stays down. X key codes are evdev's plus 8.
	Display *xtest = open_display(display);
	require(xtest, "cannot connect to Xvfb %s", display);
	require(fake_key_event(xtest, KEY_F10 + 8u, True, 0), "XTEST did not press F10");
	sync_display(xtest, False);
	require(arrives(&h, KEY_F10), "the press of F10 did not arrive");

	// Then its release, two presses of A and one of F10, with their releases, which do not count.
	static struct {
		uint16_t key;  // the key's evdev code
		bool     down; // whether the key goes down or comes up
	} const SENT[] = {
		{ KEY_F10, false }, { KEY_A, true }, { KEY_A, false }, { KEY_A, true }, { KEY_A, false },
		{ KEY_F10, true }, { KEY_F10, false }
	};
	for (size_t i = 0; i < sizeof SENT / sizeof *SENT; ++i)
		require(fake_key_event(xtest, SENT[i].key + 8u, SENT[i].down, 0), "XTEST did not send a key");
	sync_display(xtest, False);
	close_display(xtest);
	xtest = nullptr;

	// The raw events arrive shortly after the sync, in order: once the second press of F10 has
	// arrived, so have those of A.
	require(arrives(&h, KEY_F10), "the second press of F10 did not arrive");
	require(!hotkeys_pressed(&h, KEY_F10) && presses(&h, KEY_A) == 2, "the presses did not read back as sent");

	// hotkeys_fini() leaves the display open, as the layer does; the test closes it, so that leak
	// and descriptor checkers find nothing.
	close_display(h.x11->display);
	h.x11->display = nullptr;
	hotkeys_fini(&h);
	require(is_zero(&h, sizeof h), "finished hotkeys are not empty");
	stop_xvfb();
	require(!remove(log), "cannot remove %s", log);
	dlclose(xtst);
	dlclose(x11);
}

/** @brief The first press with no backend available: the hotkeys log so, open nothing, and unload
 *         what they loaded.
 */
static void
check_no_backend (void)
{
	static struct {
		char const *backend; // DLSSNR_HOTKEY_BACKEND
		char const *display; // DISPLAY, or nullptr for none
	} const CHOICES[] = {
		{ "none", nullptr },
		{ "x11", nullptr },
		// A display that no X server serves: the libraries load, and the display does not open.
		{ "x11", ":99999" }
	};
	for (size_t i = 0; i < sizeof CHOICES / sizeof *CHOICES; ++i) {
		char const *const display = CHOICES[i].display;
		require(!setenv("DLSSNR_HOTKEY_BACKEND", CHOICES[i].backend, 1)
		        && !(display ? setenv("DISPLAY", display, 1) : unsetenv("DISPLAY")), "setenv failed");
		bool const xi = loaded("libXi.so.6");
		struct hotkeys h = {0};
		require(!hotkeys_pressed(&h, KEY_F10) && h.flags == HOTKEYS_OPENED && !h.nodes && !h.x11
		        && !h.keyboards, "DLSSNR_HOTKEY_BACKEND=%s opened something", CHOICES[i].backend);
		expect_log("%s", NO_WAY);
		require(xi || !loaded("libXi.so.6"), "a backend that did not open left libXi loaded");
		require(!hotkeys_pressed(&h, KEY_F10) && h.flags == HOTKEYS_OPENED, "a second press opened again");
		hotkeys_fini(&h);
	}
}

/** @brief Reads a whole file. */
static char *
read_file (char const *path)
{
	FILE *const f = fopen(path, "rb");
	require(f, "cannot read %s", path);
	char *text = nullptr;
	size_t size = 0;
	for (size_t n; !feof(f) && !ferror(f); size += n) {
		char *const grown = realloc(text, size + 4097);
		require(grown, "out of memory");
		text = grown;
		n = fread(text + size, 1, 4096, f);
	}
	fclose(f);
	text[size] = '\0';
	return text;
}

int
main (void)
{
	// Descriptor 0 holds a file of the test's own, so that a close of it shows.
	int null = open("/dev/null", O_RDONLY | O_CLOEXEC);
	require(null >= 0 && dup2(null, 0) == 0 && !fstat(0, &descriptor_0), "cannot replace descriptor 0");
	if (null) {
		close(null);
		null = -1;
	}
	require(!atexit(stop_xvfb), "atexit failed");

	char const *const tmp = getenv("TMPDIR");
	char dir[PATH_SIZE];
	require(snprintf(dir, sizeof dir, "%s/dlsslop-amd-hotkey-XXXXXX", tmp && *tmp ? tmp : "/tmp")
	        < (int)sizeof dir, "TMPDIR is too long");
	require(mkdtemp(dir), "mkdtemp failed");
	char log[PATH_SIZE];
	join(log, dir, "layer.log");
	// The log reads its variables at its first line, which opens its file before any check counts
	// descriptors.
	require(!setenv("DLSSNR_ENABLE", "1", 1) && !setenv("DLSSNR_LOG", log, 1), "setenv failed");
	log_printf("[hotkey-test] begins");
	expect_log("[hotkey-test] begins");

	check_fini();
	check_fini_x11();
	check_reads();
	check_sweeps(dir);
	check_keyboards(dir);
	check_x11(dir);
	check_no_backend();
	require(descriptor_0_intact(), "descriptor 0 changed");

	char *text = read_file(log);
	require(!strcmp(text, expected_log), "the log holds\n%s\nnot\n%s", text, expected_log);
	free(text);
	text = nullptr;
	require(!remove(log) && !rmdir(dir), "cannot remove %s", dir);
	printf("hotkey-test: finishing, reads, sweeps, keyboards, XInput2 and the backend choice hold\n");
	return 0;
}
