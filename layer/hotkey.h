/** @file
 *
 * An in-game key, in every environment this layer runs in.
 *
 * This is harder than it looks, because the layer has no window. It is a shared object inside someone
 * else's process, and the three environments it has to work in do not agree on what "the keyboard" is:
 *
 *   X11 / XWayland   an X server exists, but not every way of asking it still works
 *   winewayland      the game is a Wayland client; the layer is not a client at all
 *   gamescope        a nested compositor with its own Xwayland inside it
 *
 * Two backends, measured rather than assumed on an XWayland session with a real key press:
 *
 *   evdev    reads the kernel's own input devices. Works everywhere -- a Proton game is still a Linux
 *            process and gamescope does not sit between a process and /dev/input -- and has the lowest
 *            latency. It needs read access to /dev/input/event*, which is root:input. Keyboards
 *            deliberately get no uaccess ACL, unlike joysticks, because that would let any program
 *            keylog, so this means the 'input' group and most people are not in it.
 *
 *   XInput2  raw key events selected on the root window. Needs no permission, and raw events arrive
 *            regardless of which client has focus -- including when the focused client is a Wayland
 *            one and no X client is focused at all. Verified with a real key from a virtual keyboard,
 *            not with XTEST, which injects at the server and would have made this pass falsely.
 *
 * XQueryKeymap, which is what vkBasalt uses, was tried first and does not work here: on an XWayland
 * session it reports nothing for a real key press. It is fine on a true Xorg session and that is
 * increasingly not what people run.
 *
 * Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_HOTKEY_H_
#define DLSSLOP_AMD_LAYER_HOTKEY_H_

#ifdef __cplusplus
# include <cstdint>
# define HOTKEY_STD(x) std::x
extern "C" {
#else
# include <stdint.h>
# define HOTKEY_STD(x) x
#endif

/** @brief The number of Linux key codes, KEY_CNT: the codes a press can have. */
static constexpr HOTKEY_STD(uint32_t) HOTKEYS_CODES = 0x300;

struct hotkey_node;
struct hotkey_x11;

/** @brief The state that struct hotkeys records in its flags. */
enum hotkeys_flags : HOTKEY_STD(uint32_t) {
	HOTKEYS_OPENED    = 1, //!< The first hotkeys_pressed() chose a backend.
	HOTKEYS_ANNOUNCED = 2, //!< /dev/input was read once: a keyboard found later is logged.
	HOTKEYS_EVDEV     = 4  //!< The backend is evdev, also while no keyboard is open.
};

/** @brief The keyboards that the layer reads its key from, and the presses it has not answered.
 *
 * A zeroed object is an empty one: the first hotkeys_pressed() opens a backend. hotkeys_fini()
 * closes what that opened.
 */
struct hotkeys {
	struct hotkey_node  *nodes;                   //!< /dev/input's event nodes; hotkey_priv.h.
	struct hotkey_x11   *x11;                     //!< The XInput2 backend, if it works; hotkey_priv.h.
	double               last_scan;               //!< log_now_ms() of the last look at /dev/input.
	HOTKEY_STD(uint32_t) node_count;              //!< The entries of nodes.
	HOTKEY_STD(uint32_t) keyboards;               //!< The nodes open as keyboards.
	HOTKEY_STD(uint32_t) pending_total;           //!< Presses not answered yet, of every key.
	HOTKEY_STD(uint32_t) flags;                   //!< enum hotkeys_flags.
	HOTKEY_STD(uint8_t)  pending[HOTKEYS_CODES];  //!< Presses not answered yet, per key, at most 255.
};

/** @brief A Linux KEY_* code from a name.
 *
 * Names are the KEY_ names without the prefix, case-insensitive: "F10", "Home", "N". A bare number
 * below HOTKEYS_CODES is a key code, so a key with no name here is still reachable.
 *
 * @param name The name, or nullptr.
 * @return     The code, or 0 (unbound) for anything unrecognised, and for a number that is no key
 *             code.
 */
extern HOTKEY_STD(uint32_t)
hotkey_key_code_from_name (char const *name);

/** @brief The name of a Linux KEY_* code.
 *
 * @param code The code.
 * @return     Its name in upper case, or "?" for a key that has none here.
 */
extern char const *
hotkey_key_name_from_code (HOTKEY_STD(uint32_t) code);

/** @brief Whether the key went down since the last time this was asked.
 *
 * True once per press. The first call opens whichever backend is available, evdev first, or the one
 * that DLSSNR_HOTKEY_BACKEND names ("evdev" or "x11"). Cheap enough for the present hook: the evdev
 * backend drains a non-blocking read and looks at /dev/input again once a second, and the X11 one
 * drains the events of a cached connection.
 *
 * Not thread-safe: the calls on one struct hotkeys must not overlap. The layer keeps its devices'
 * calls apart with a mutex (g_hotkeys in layer.c).
 *
 * @param h        The hotkeys, or nullptr.
 * @param key_code A Linux KEY_* code; 0 is unbound.
 * @return         true if a press of the key is pending; it is then answered.
 */
extern bool
hotkeys_pressed (struct hotkeys       *h,
                 HOTKEY_STD(uint32_t)  key_code);

/** @brief Closes the keyboards and unloads libX11 and libXi, then leaves the hotkeys empty.
 *
 * The X display stays open, as upstream left it: closing it at exit, after the X server has gone,
 * would run Xlib's I/O error handler, which exits the process.
 *
 * @param dest The hotkeys, or nullptr.
 */
extern void
hotkeys_fini (struct hotkeys *dest);

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef HOTKEY_STD

#endif /* DLSSLOP_AMD_LAYER_HOTKEY_H_ */
