# Standalone graphical controller

A native C++17 **Qt 6 Widgets** application with no QML, JavaScript or web engine.
It controls an existing worker channel; the worker and Vulkan layer do not
require Qt. All 41 settings share definitions with the CLI and use protocol 23.

## Build and launch

First fetch and prepare upstream sources as described in [README.md](../README.md).
Install Qt 6 development libraries, CMake and a C++ compiler (`qt6-base-dev`,
`cmake` and `g++` on Debian/Ubuntu). From the repository root:

```bash
cmake -S gui -B build-gui -DCMAKE_BUILD_TYPE=Release
cmake --build build-gui --parallel
ctest --test-dir build-gui --output-on-failure
cmake --install build-gui --prefix "$HOME/.local"
dlsslop-gui
```

The tests need Qt 6.8 or newer; configure with `-DBUILD_TESTING=OFF` on older Qt.

This independent build installs only the GUI. The full project enables the
same target with `-DDLSSLOP_BUILD_GUI=ON` (default ON); use
`-DDLSSLOP_BUILD_GUI=OFF` for a headless core source build. The application links
the installed Qt 6 libraries; it does not package a Qt runtime.

Select another channel with `-s PATH` / `--shm PATH`. The default is nonempty
`DLSSNR_SHM`, otherwise `/tmp/dlsslop-amd-UID/shm.bin`, using the current user's
numeric ID. This native channel default is independent of the upstream
`DLSSNR_UID` override for the shared `/tmp/dlssnr-UID/` directory. `-h` / `--help`
works without a graphical session. `QT_QPA_PLATFORM=xcb` selects X11 when needed.

## Controls

Settings are grouped into neural passes, composition, image/HDR, motion,
comparison/debug and model configuration. The four fixed model settings are
visible and disabled. Writable settings have individual reset buttons.

- Changes apply live. An edit is written at once; further edits within 40 ms
  are coalesced into one batch, with a final write on slider release. Clicking
  anywhere along a slider places its handle at that position immediately; keep
  the mouse button down to drag. Only edited settings are written. Keyboard
  focus highlights the handle; arrow keys, Page Up/Down and Home/End remain
  available.
- The mouse wheel scrolls the page. It edits a slider, numeric field or menu
  only after that control has focus.
- Numeric fields accept six decimal places; arrow steps are 0.001, 0.01
  (default) or 0.1. Typing commits on Enter or focus loss. Wide positive ranges
  have logarithmic sliders and linear numeric entry.
- Start the worker separately and use **Connect / refresh**. Opening and
  refreshing are read-only; the GUI never creates or initializes a channel.
- Refresh reads changes from other controllers. Status is a snapshot with no
  background polling. Closing flushes pending edits and leaves the worker running.
- Actions include reset-all, capture 0–64 frames, request stop and clear stop.
  Stop/reset ask for confirmation. Clear stop does not restart an exited worker.
- Channel replacement or protocol mismatch disables editing until reconnect;
  queued edits are not replayed into a replacement channel.

**Color preserve**, under Neural passes, anchors broad chroma changes to the
original proxy before per-pass history/feedback. Default 0 disables it. The
worker selects the HIP module or logs the slower CPU fallback; see
[COLOR-PRESERVATION.md](../COLOR-PRESERVATION.md).

The CI workflow builds the GUI and runs its option-parsing, channel and
offscreen slider and window tests. For testing instructions, see
[VALIDATION.md](../VALIDATION.md).
