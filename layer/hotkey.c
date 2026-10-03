/** @file
 *
 * The hotkeys' two backends, evdev and XInput2, and the names of the keys.
 */
#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/input.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hotkey_priv.h"
#include "log.h"

static_assert(HOTKEYS_CODES == KEY_CNT, "HOTKEYS_CODES is not the number of key codes");

/** @brief A key that a name binds. */
struct named_key {
	uint32_t code;     //!< Its KEY_* code.
	char     name[12]; //!< Its KEY_ name without the prefix, in upper case.
};

/** @brief The keys anyone would plausibly bind.
 *
 * Not the whole table: a name that is not here simply does not resolve, which is better than
 * accepting something nobody can type.
 */
static struct named_key const HOTKEY_KEYS[] = {
	{ KEY_F1, "F1" },   { KEY_F2, "F2" },   { KEY_F3, "F3" },   { KEY_F4, "F4" },
	{ KEY_F5, "F5" },   { KEY_F6, "F6" },   { KEY_F7, "F7" },   { KEY_F8, "F8" },
	{ KEY_F9, "F9" },   { KEY_F10, "F10" }, { KEY_F11, "F11" }, { KEY_F12, "F12" },
	{ KEY_HOME, "HOME" }, { KEY_END, "END" }, { KEY_INSERT, "INSERT" }, { KEY_DELETE, "DELETE" },
	{ KEY_PAGEUP, "PAGEUP" }, { KEY_PAGEDOWN, "PAGEDOWN" }, { KEY_PAUSE, "PAUSE" },
	{ KEY_SCROLLLOCK, "SCROLLLOCK" }, { KEY_SYSRQ, "SYSRQ" }, { KEY_GRAVE, "GRAVE" },
	{ KEY_A, "A" }, { KEY_B, "B" }, { KEY_C, "C" }, { KEY_D, "D" }, { KEY_E, "E" }, { KEY_F, "F" },
	{ KEY_G, "G" }, { KEY_H, "H" }, { KEY_I, "I" }, { KEY_J, "J" }, { KEY_K, "K" }, { KEY_L, "L" },
	{ KEY_M, "M" }, { KEY_N, "N" }, { KEY_O, "O" }, { KEY_P, "P" }, { KEY_Q, "Q" }, { KEY_R, "R" },
	{ KEY_S, "S" }, { KEY_T, "T" }, { KEY_U, "U" }, { KEY_V, "V" }, { KEY_W, "W" }, { KEY_X, "X" },
	{ KEY_Y, "Y" }, { KEY_Z, "Z" },
	{ KEY_0, "0" }, { KEY_1, "1" }, { KEY_2, "2" }, { KEY_3, "3" }, { KEY_4, "4" },
	{ KEY_5, "5" }, { KEY_6, "6" }, { KEY_7, "7" }, { KEY_8, "8" }, { KEY_9, "9" },
};

/** @brief The number of named keys. */
static constexpr size_t HOTKEY_KEY_COUNT = sizeof HOTKEY_KEYS / sizeof *HOTKEY_KEYS;

uint32_t
hotkey_key_code_from_name (char const *name)
{
	if (!name || !*name)
		return 0;

	// A bare number is a Linux key code, so a key with no name here is still reachable. The
	// only names in the table that begin with a digit are the digits, which read as numbers.
	if (isdigit((unsigned char)*name)) {
		char *end;
		// Only a number below HOTKEYS_CODES is a key's, and strtoul() saturates one that overflows.
		unsigned long const code = strtoul(name, &end, 10);
		return *end || code >= HOTKEYS_CODES ? 0 : (uint32_t)code;
	}

	// The name in upper case, each byte put there with toupper() as upstream did. A name that
	// does not fit is longer than "KEY_" and any name in the table.
	char upper[sizeof "KEY_" - 1 + sizeof HOTKEY_KEYS->name];
	size_t length = 0;
	for (; name[length]; ++length) {
		if (length == sizeof upper - 1)
			return 0;
		upper[length] = (char)toupper((unsigned char)name[length]);
	}
	upper[length] = '\0';

	char const *const key = strncmp(upper, "KEY_", 4) ? upper : upper + 4;
	for (size_t i = 0; i < HOTKEY_KEY_COUNT; ++i)
		if (!strcmp(key, HOTKEY_KEYS[i].name))
			return HOTKEY_KEYS[i].code;
	return 0;
}

char const *
hotkey_key_name_from_code (uint32_t code)
{
	for (size_t i = 0; i < HOTKEY_KEY_COUNT; ++i)
		if (HOTKEY_KEYS[i].code == code)
			return HOTKEY_KEYS[i].name;
	return "?";
}

/** @brief Counts a press of a key.
 *
 * A key's own count stops at 255, but the total counts every press, so that hotkeys_take() clears
 * the counts after the same number of presses whichever keys they were.
 *
 * @param h    The hotkeys.
 * @param code The key's code; one beyond the key codes counts nothing.
 */
static void
hotkeys_count (struct hotkeys *h,
               uint32_t        code)
{
	if (code >= HOTKEYS_CODES)
		return;

	++h->pending_total;
	if (h->pending[code] < UINT8_MAX)
		++h->pending[code];
}

/** @brief Answers a press of a key, if one is pending.
 *
 * @param h        The hotkeys.
 * @param key_code The key's code.
 * @return         true if a press was pending; it is no longer.
 */
static bool
hotkeys_take (struct hotkeys *h,
              uint32_t        key_code)
{
	if (key_code < HOTKEYS_CODES && h->pending[key_code]) {
		--h->pending[key_code];
		--h->pending_total;
		return true;
	}

	// Nothing this caller wanted; keep the counts short rather than growing them forever.
	if (h->pending_total > 256) {
		memset(h->pending, 0, sizeof h->pending);
		h->pending_total = 0;
	}
	return false;
}

/** @brief Whether bit @a bit is set in an evdev bit mask. */
static bool
test_bit (unsigned long const *bits,
          uint32_t             bit)
{
	return (bits[bit / (8 * sizeof (long))] >> (bit % (8 * sizeof (long)))) & 1UL;
}

/** @brief Whether an event node is a keyboard, as opposed to a mouse, a lid switch, a power button
 *         or a gamepad.
 *
 * Asking for letters is enough: nothing else on a normal system claims to produce KEY_A through
 * KEY_Z.
 *
 * @param fd The node.
 * @return   true if it has keys, among them every code from KEY_A to KEY_Z.
 */
static bool
looks_like_a_keyboard (int fd)
{
	unsigned long evbits[(EV_MAX + 8 * sizeof (long)) / (8 * sizeof (long))] = {0};
	if (ioctl(fd, EVIOCGBIT(0, sizeof evbits), evbits) < 0 || !test_bit(evbits, EV_KEY))
		return false;

	unsigned long keybits[(KEY_MAX + 8 * sizeof (long)) / (8 * sizeof (long))] = {0};
	if (ioctl(fd, EVIOCGBIT(EV_KEY, sizeof keybits), keybits) < 0)
		return false;

	for (uint32_t k = KEY_A; k <= KEY_Z; ++k)
		if (!test_bit(keybits, k))
			return false;
	return true;
}

/** @brief N of a directory entry named eventN.
 *
 * The kernel names its event nodes eventN, and the node table keeps N in 16 bits, so any other name
 * is not looked at. Upstream probed every name that begins with "event".
 *
 * @param name  The entry's name.
 * @param event Receives N.
 * @return      true if the name is event and a number that fits in 16 bits.
 */
static bool
event_number (char const *name,
              uint16_t   *event)
{
	if (strncmp(name, "event", 5) || !isdigit((unsigned char)name[5]))
		return false;

	char *end;
	unsigned long const n = strtoul(name + 5, &end, 10);
	if (*end || n > UINT16_MAX)
		return false;

	*event = (uint16_t)n;
	return true;
}

/** @brief The node of eventN, which is added if there is none.
 *
 * @param h     The hotkeys.
 * @param event N.
 * @return      The node, or nullptr if there was none and no memory for one.
 */
static struct hotkey_node *
hotkeys_node (struct hotkeys *h,
              uint16_t        event)
{
	for (uint32_t i = 0; i < h->node_count; ++i)
		if (h->nodes[i].event == event)
			return &h->nodes[i];

	struct hotkey_node *const nodes = realloc(h->nodes, (h->node_count + 1) * sizeof *nodes);
	if (!nodes)
		return nullptr;

	h->nodes = nodes;
	nodes[h->node_count] = (struct hotkey_node){ .fd = -1, .event = event };
	return &nodes[h->node_count++];
}

/** @brief Opens a node, and keeps it open if it is a keyboard.
 *
 * @param h    The hotkeys.
 * @param node The node, which is not open.
 * @param dir  The directory's descriptor.
 * @param path The directory's path, for the log.
 * @param name The node's name in it.
 * @return     true if the node opened and is not a keyboard.
 */
static bool
hotkeys_probe (struct hotkeys     *h,
               struct hotkey_node *node,
               int                 dir,
               char const         *path,
               char const         *name)
{
	int fd = openat(dir, name, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
	if (fd < 0)
		return false;

	if (!looks_like_a_keyboard(fd)) {
		close(fd);
		fd = -1;
		return true;
	}
	node->fd = fd;
	if (h->flags & HOTKEYS_ANNOUNCED)
		log_printf("[hotkey] picked up a keyboard that appeared later: %s/%s", path, name);
	return false;
}

/** @brief Looks at one directory entry: opens it if it is a keyboard that is not open yet.
 *
 * @param h    The hotkeys.
 * @param dir  The directory's descriptor.
 * @param path The directory's path, for the log.
 * @param name The entry's name.
 */
static void
hotkeys_look_at (struct hotkeys *h,
                 int             dir,
                 char const     *path,
                 char const     *name)
{
	uint16_t event;
	if (!event_number(name, &event))
		return;

	struct hotkey_node *const node = hotkeys_node(h, event);
	if (!node)
		return;

	node->flags |= HOTKEY_NODE_SEEN;
	if (node->fd >= 0)
		return;

	// Already judged not to be a keyboard? Only trust that verdict if it is still the same node: a
	// stat is two orders of magnitude cheaper than the open/ioctl/close it replaces. A node that
	// cannot be stat'ed is probed, and not remembered.
	struct stat st;
	if (fstatat(dir, name, &st, 0)) {
		hotkeys_probe(h, node, dir, path, name);
		return;
	}
	if ((node->flags & HOTKEY_NODE_REJECTED) && node->ino == st.st_ino)
		return;

	// The same path with another device, or a new one: look again.
	node->flags &= ~HOTKEY_NODE_REJECTED;
	if (hotkeys_probe(h, node, dir, path, name)) {
		node->ino = st.st_ino;
		node->flags |= HOTKEY_NODE_REJECTED;
	}
}

/** @brief Closes a keyboard if it has gone away, asked with an ioctl rather than a read.
 *
 * A read looked like the obvious test -- a removed device answers ENODEV -- but it answers with a
 * queued event first if there is one, so a dead device with unread input looked alive and its path
 * stayed known. The next device to take that path was then never opened, and the key silently
 * stopped working. EVIOCGVERSION asks the same question without consuming anything.
 *
 * @param node A node.
 */
static void
hotkeys_check (struct hotkey_node *node)
{
	int version;
	if (node->fd < 0 || ioctl(node->fd, EVIOCGVERSION, &version) >= 0)
		return;

	close(node->fd);
	node->fd = -1;
}

/** @brief Forgets the nodes that have left the directory, so that the table tracks the directory
 *         rather than growing without bound on a machine that plugs and unplugs a lot.
 *
 * A node that has left keeps only an open keyboard, which stays until hotkeys_check() finds it gone.
 *
 * @param h The hotkeys, after a look at every entry of the directory.
 */
static void
hotkeys_forget (struct hotkeys *h)
{
	uint32_t kept = 0;
	for (uint32_t i = 0; i < h->node_count; ++i) {
		struct hotkey_node node = h->nodes[i];
		if (!(node.flags & HOTKEY_NODE_SEEN))
			node.flags = 0;
		if (!node.flags && node.fd < 0)
			continue;

		node.flags &= ~HOTKEY_NODE_SEEN;
		h->nodes[kept++] = node;
	}
	h->node_count = kept;
}

void
hotkeys_rescan_evdev (struct hotkeys *h,
                      char const     *path)
{
	for (uint32_t i = 0; i < h->node_count; ++i)
		hotkeys_check(&h->nodes[i]);

	DIR *const dir = opendir(path);
	if (!dir)
		return;
	// The stream's own descriptor, which closedir() closes.
	int fd = dirfd(dir);
	if (fd < 0) {
		closedir(dir);
		return;
	}

	for (struct dirent *e; (e = readdir(dir));)
		hotkeys_look_at(h, fd, path, e->d_name);
	closedir(dir);
	fd = -1;

	hotkeys_forget(h);
	h->flags |= HOTKEYS_ANNOUNCED;
}

/** @brief Unloads what an XInput2 backend loaded, frees it and empties the caller's pointer. The
 *         display stays open.
 *
 * @param p_dest The address of a backend's pointer, or nullptr; a null backend pointer is ignored.
 */
static void
hotkey_x11_destroy (struct hotkey_x11 **p_dest)
{
	if (!p_dest || !*p_dest)
		return;

	struct hotkey_x11 *x = *p_dest;
	*p_dest = nullptr;
	if (x->xi)
		dlclose(x->xi);
	if (x->x11)
		dlclose(x->x11);
	free(x);
	x = nullptr;
}

/** @brief Loads libX11 and libXi and selects raw key presses on the root window.
 *
 * Raw events are delivered whatever has focus, which is the whole point: the layer has no window and
 * the game may not even be an X client.
 *
 * @param x    A zeroed backend, which receives what was loaded even when this fails.
 * @param name The display's name.
 * @return     true if raw key presses are selected.
 */
static bool
hotkey_x11_open (struct hotkey_x11 *x,
                 char const        *name)
{
	x->x11 = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
	x->xi = dlopen("libXi.so.6", RTLD_LAZY | RTLD_LOCAL);
	if (!x->x11 || !x->xi)
		return false;

#define X_SYMBOL(lib, name) ((typeof (name) *)dlsym(lib, #name))
	typeof (XOpenDisplay) *const open_display = X_SYMBOL(x->x11, XOpenDisplay);
	typeof (XCloseDisplay) *const close_display = X_SYMBOL(x->x11, XCloseDisplay);
	typeof (XQueryExtension) *const query_extension = X_SYMBOL(x->x11, XQueryExtension);
	typeof (XFlush) *const flush = X_SYMBOL(x->x11, XFlush);
	typeof (XIQueryVersion) *const query_version = X_SYMBOL(x->xi, XIQueryVersion);
	typeof (XISelectEvents) *const select_events = X_SYMBOL(x->xi, XISelectEvents);
	x->pending = X_SYMBOL(x->x11, XPending);
	x->next_event = X_SYMBOL(x->x11, XNextEvent);
	x->get_event_data = X_SYMBOL(x->x11, XGetEventData);
	x->free_event_data = X_SYMBOL(x->x11, XFreeEventData);
#undef X_SYMBOL
	if (!open_display || !close_display || !query_extension || !flush || !query_version || !select_events
	    || !x->pending || !x->next_event || !x->get_event_data || !x->free_event_data)
		return false;

	Display *display = open_display(name);
	if (!display)
		return false;

	// A display that cannot serve the backend is closed before its libraries are unloaded. The server
	// has just answered, so the close does not meet one that has gone.
	int event;
	int error;
	if (!query_extension(display, "XInputExtension", &x->opcode, &event, &error)) {
		close_display(display);
		display = nullptr;
		return false;
	}

	// The version the server speaks, which is at most the one asked for. Raw events reach a client
	// that has no grab from 2.1 on, and the backend asks for 2.2.
	int major = 2;
	int minor = 2;
	if (query_version(display, &major, &minor) != Success || major < 2 || (major == 2 && minor < 2)) {
		close_display(display);
		display = nullptr;
		return false;
	}

	unsigned char mask[4] = {0};
	XISetMask(mask, XI_RawKeyPress);
	XIEventMask em = {
		.deviceid = XIAllMasterDevices,
		.mask_len = sizeof mask,
		.mask     = mask
	};
	select_events(display, DefaultRootWindow(display), &em, 1);
	flush(display);
	x->display = display;
	return true;
}

/** @brief Opens an XInput2 backend on a display.
 *
 * A display that has no XInput 2.2 gives no backend, and what was loaded is unloaded at once.
 * Upstream went on polling such a display, on which it had selected no events.
 *
 * @param name The display's name.
 * @return     The backend, or nullptr if raw key presses could not be selected.
 */
static struct hotkey_x11 *
hotkey_x11_create (char const *name)
{
	struct hotkey_x11 *x = calloc(1, sizeof *x);
	if (x && !hotkey_x11_open(x, name))
		hotkey_x11_destroy(&x);
	return x;
}

/** @brief Opens the XInput2 backend if DISPLAY names a display.
 *
 * @param h    The hotkeys.
 * @param name DISPLAY's value, or nullptr if it is unset.
 * @return     true if the backend works.
 */
static bool
hotkeys_open_x11 (struct hotkeys *h,
                  char const     *name)
{
	if (!name)
		return false;

	h->x11 = hotkey_x11_create(name);
	return h->x11;
}

/** @brief Counts the nodes that are open as keyboards: those whose fd is not -1.
 *
 * @param h The hotkeys.
 * @return  The number of keyboards.
 */
static uint32_t
hotkeys_keyboards (struct hotkeys const *h)
{
	uint32_t keyboards = 0;
	for (uint32_t i = 0; i < h->node_count; ++i)
		if (h->nodes[i].fd >= 0)
			++keyboards;
	return keyboards;
}

/** @brief Opens whichever backend is available, once.
 *
 * Forced backends exist for testing: the fallback is otherwise unreachable on a machine where evdev
 * works, which is the machine it is hardest to get wrong on.
 *
 * @param h The hotkeys.
 */
static void
hotkeys_open (struct hotkeys *h)
{
	if (h->flags & HOTKEYS_OPENED)
		return;

	h->flags |= HOTKEYS_OPENED;
	char const *const forced = getenv("DLSSNR_HOTKEY_BACKEND");
	bool const want_evdev = !forced || !strcmp(forced, "evdev");
	bool const want_x11 = !forced || !strcmp(forced, "x11");

	if (want_evdev) {
		hotkeys_rescan_evdev(h, "/dev/input");
		// The first press's look at /dev/input is this one.
		h->last_scan = log_now_ms();
		uint32_t const keyboards = hotkeys_keyboards(h);
		if (keyboards) {
			h->flags |= HOTKEYS_EVDEV;
			log_printf("[hotkey] watching %u keyboard(s) through evdev", keyboards);
			return;
		}
	}
	// DISPLAY read once, for the check, the connection and the log.
	char const *const display = want_x11 ? getenv("DISPLAY") : nullptr;
	if (hotkeys_open_x11(h, display)) {
		log_printf("[hotkey] watching XInput2 raw keys on %s", display);
		return;
	}

	log_printf("[hotkey] no way to read the keyboard here. /dev/input needs the 'input' group or a "
	           "uaccess ACL; without it only an X11 session can be read, and never one inside "
	           "gamescope.");
}

/** @brief Reads one buffer of a keyboard's events and counts its key presses.
 *
 * @param h  The hotkeys.
 * @param fd The keyboard.
 * @return   true if the buffer filled, so more events may wait.
 */
static bool
hotkeys_read (struct hotkeys *h,
              int             fd)
{
	struct input_event events[64];
	ssize_t const n = read(fd, events, sizeof events);
	if (n <= 0)
		return false;

	ssize_t const count = n / (ssize_t)sizeof *events;
	for (ssize_t i = 0; i < count; ++i)
		if (events[i].type == EV_KEY && events[i].value == 1)
			hotkeys_count(h, events[i].code);
	return n == (ssize_t)sizeof events;
}

/** @brief Drains everything the keyboards have to say, and answers whether the key went down.
 *
 * Draining rather than looking for one code matters: the queue is per device and a read that leaves
 * events behind makes the next frame's answer stale, and eventually the buffer overruns.
 *
 * @param h        The hotkeys.
 * @param key_code The key.
 * @return         true if a press of it was pending.
 */
static bool
hotkeys_pressed_evdev (struct hotkeys *h,
                       uint32_t        key_code)
{
	// Keyboards appear after the game starts: someone plugs one in, a wireless one wakes, or a
	// controller's keyboard half enumerates late. Scanning once at startup meant those were
	// invisible for the life of the process, which is a very confusing way for a key to not work.
	// Timed rather than counted in frames: a frame count means a different rescan interval at 30 fps
	// than at 300, and the thing being waited for is a person plugging something in. Once a second
	// is far below what anyone would notice.
	double const now = log_now_ms();
	if (now - h->last_scan >= 1000.0) {
		h->last_scan = now;
		hotkeys_rescan_evdev(h, "/dev/input");
	}

	for (uint32_t i = 0; i < h->node_count; ++i) {
		if (h->nodes[i].fd < 0)
			continue;
		while (hotkeys_read(h, h->nodes[i].fd))
			continue;
	}
	return hotkeys_take(h, key_code);
}

/** @brief Drains the X connection's raw key presses, and answers whether the key went down.
 *
 * Raw key presses are already edges, so there is no held state to track. The X keycode for an evdev
 * code is that code plus eight on every X server driven by evdev, which is all of them on Linux.
 *
 * @param h        The hotkeys.
 * @param key_code The key.
 * @return         true if a press of it was pending.
 */
static bool
hotkeys_pressed_x11 (struct hotkeys *h,
                     uint32_t        key_code)
{
	struct hotkey_x11 const *const x = h->x11;
	while (x->pending(x->display) > 0) {
		XEvent event;
		x->next_event(x->display, &event);
		XGenericEventCookie *const cookie = &event.xcookie;
		if (cookie->type != GenericEvent || cookie->extension != x->opcode
		    || !x->get_event_data(x->display, cookie))
			continue;

		XIRawEvent const *const raw = cookie->data;
		if (cookie->evtype == XI_RawKeyPress && raw->detail >= 8)
			hotkeys_count(h, (uint32_t)raw->detail - 8);
		x->free_event_data(x->display, cookie);
	}
	return hotkeys_take(h, key_code);
}

bool
hotkeys_pressed (struct hotkeys *h,
                 uint32_t        key_code)
{
	if (!h || !key_code)
		return false;

	hotkeys_open(h);
	// By the backend, not by the keyboards open now: evdev looks for new ones while it has none.
	if (h->flags & HOTKEYS_EVDEV)
		return hotkeys_pressed_evdev(h, key_code);
	if (h->x11)
		return hotkeys_pressed_x11(h, key_code);
	return false;
}

void
hotkeys_fini (struct hotkeys *dest)
{
	if (!dest)
		return;

	for (uint32_t i = 0; i < dest->node_count; ++i) {
		if (dest->nodes[i].fd >= 0) {
			close(dest->nodes[i].fd);
			dest->nodes[i].fd = -1;
		}
	}
	free(dest->nodes);
	dest->nodes = nullptr;
	hotkey_x11_destroy(&dest->x11);
	*dest = (struct hotkeys){0};
}
