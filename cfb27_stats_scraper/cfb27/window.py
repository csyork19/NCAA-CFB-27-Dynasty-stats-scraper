"""Find, launch, focus and screenshot the College Football 27 window (Windows only)."""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import time
from ctypes import wintypes
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32 if os.name == "nt" else None


def set_dpi_aware() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def find_window(title_substr: str) -> Optional[int]:
    found = []
    want = title_substr.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.lower()
        if want in title and "launcher" not in title and "steam" not in title:
            found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(_cb, 0)
    found.sort(key=lambda h: len(h[1]))  # shortest title = the game, not a browser tab
    return found[0][0] if found else None


def client_rect(hwnd: int) -> Tuple[int, int, int, int]:
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, r.right - r.left, r.bottom - r.top


def focus(hwnd: int) -> None:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.keybd_event(0x12, 0, 0, 0)  # tap ALT so Windows allows SetForegroundWindow
    user32.keybd_event(0x12, 0, 0x0002, 0)
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    time.sleep(0.3)


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def launch(cfg: dict) -> None:
    how = cfg.get("launch", "steam").lower()
    if how == "steam":
        app = int(cfg.get("steam_app_id", 4032350))
        log.info("Launching via Steam (app %s)", app)
        os.startfile(f"steam://rungameid/{app}")  # type: ignore[attr-defined]
    elif how == "exe":
        exe = cfg.get("exe_path", "")
        if not exe or not os.path.exists(exe):
            raise SystemExit(f"[game].exe_path not found: {exe!r}")
        log.info("Launching %s", exe)
        subprocess.Popen([exe], cwd=os.path.dirname(exe))
    elif how == "uri":
        os.startfile(cfg["launch_uri"])  # type: ignore[attr-defined]
    else:
        raise SystemExit("Game isn't running and [game].launch = 'none'")


def close_window(hwnd: int) -> None:
    user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE


class Capturer:
    """Grabs the game's client area as a BGR array. dxcam (DXGI) is ~5 ms per frame, mss ~25 ms."""

    def __init__(self, hwnd: int, backend: str = "dxcam"):
        self.hwnd = hwnd
        self.backend = backend
        self._last = None
        if backend == "dxcam":
            try:
                import dxcam  # type: ignore

                self._cam = dxcam.create(output_color="BGR")
            except Exception as e:
                log.warning("dxcam unavailable (%s); using mss", e)
                self.backend = "mss"
        if self.backend == "mss":
            import mss  # type: ignore

            self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        left, top, w, h = client_rect(self.hwnd)
        if w <= 0 or h <= 0:
            raise RuntimeError("Game window has no client area (minimised?)")
        if self.backend == "dxcam":
            frame = self._cam.grab(region=(left, top, left + w, top + h))
            if frame is None:  # nothing changed since the last grab
                if self._last is not None:
                    return self._last
                time.sleep(0.02)
                return self.grab()
            self._last = frame
            return frame
        shot = self._sct.grab({"left": left, "top": top, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3].copy()
