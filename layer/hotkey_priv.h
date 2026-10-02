/** @file
 *
 * The hotkeys' internals, which hotkey.c and its test share. C only.
 */
#ifndef DLSSLOP_AMD_LAYER_HOTKEY_PRIV_H_
#define DLSSLOP_AMD_LAYER_HOTKEY_PRIV_H_

#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <stdint.h>
#include <sys/types.h>

#include "hotkey.h"

/** @brief What the hotkeys know about an event node. */
enum hotkey_node_flags {
	HOTKEY_NODE_REJECTED = 1, //!< Not a keyboard, as long as the node's inode is ino.
	HOTKEY_NODE_SEEN     = 2  //!< In the directory, during a look at it.
};

/** @brief One of /dev/input's event nodes.
 *
 * Remembering the rejections is what makes a rescan cheap: without them every mouse, audio jack and
 * lid switch is re-opened and re-closed every second, and closing an evdev node is not free --
 * measured at 4-16 ms each on a machine with 24 of them, all of it on the present thread.
 *
 * A rejection holds for an inode, not just a name. devtmpfs hands out a fresh inode when a node is
 * destroyed and recreated, so an unplug/replug that reuses "event6" still looks new here and is
 * probed again. Name alone would cache the verdict for whatever device lands on that path next,
 * which is the same class of bug the EVIOCGVERSION check exists to avoid.
 *
 * A node is open as a keyboard exactly while its fd is not -1, and the hotkeys close that fd.
 */
struct hotkey_node {
	ino_t    ino;   //!< The inode it had when it was rejected.
	int      fd;    //!< The keyboard's descriptor, or -1.
	uint16_t event; //!< N in eventN.
	uint16_t flags; //!< enum hotkey_node_flags.
};

/** @brief The XInput2 backend: libX11 and libXi, loaded rather than linked, so a layer on a machine
 *         with no X at all still starts.
 */
struct hotkey_x11 {
	void                    *x11;             //!< libX11's handle, if it loaded.
	void                    *xi;              //!< libXi's handle, if it loaded.
	Display                 *display;         //!< The connection, which is never closed.
	typeof (XPending)       *pending;         //!< libX11's XPending().
	typeof (XNextEvent)     *next_event;      //!< libX11's XNextEvent().
	typeof (XGetEventData)  *get_event_data;  //!< libX11's XGetEventData().
	typeof (XFreeEventData) *free_event_data; //!< libX11's XFreeEventData().
	int                      opcode;          //!< XInput's major opcode.
};

/** @brief Opens every keyboard in a directory of event nodes that is not open yet, and closes every
 *         keyboard that has gone away.
 *
 * Cheap: a readdir, and an ioctl only for a node nobody has looked at yet. Nodes that turn out not
 * to be keyboards are remembered as such, and forgotten when they leave the directory. A keyboard
 * found after the first look is logged.
 *
 * @param h    The hotkeys.
 * @param path The directory: /dev/input, or a test's.
 */
extern void
hotkeys_rescan_evdev (struct hotkeys *h,
                      char const     *path);

#endif /* DLSSLOP_AMD_LAYER_HOTKEY_PRIV_H_ */
