"""Virtual Xbox 360 controller (ViGEmBus driver + the `vgamepad` package)."""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

log = logging.getLogger(__name__)

STICK_DIRS = {"UP": (0.0, 1.0), "DOWN": (0.0, -1.0), "LEFT": (-1.0, 0.0), "RIGHT": (1.0, 0.0)}


class Gamepad:
    """Button names: A B X Y LB RB LT RT START BACK UP DOWN LEFT RIGHT LS RS,
    plus stick flicks such as LS_DOWN or RS_RIGHT."""

    def __init__(self, hold: float = 0.06, gap: float = 0.035, tap_hold: float = 0.035):
        import vgamepad as vg  # Windows only

        self.pad = vg.VX360Gamepad()
        self.hold = hold
        self.gap = gap
        self.tap_hold = tap_hold
        B = vg.XUSB_BUTTON
        self.buttons = {
            "A": B.XUSB_GAMEPAD_A, "B": B.XUSB_GAMEPAD_B, "X": B.XUSB_GAMEPAD_X, "Y": B.XUSB_GAMEPAD_Y,
            "LB": B.XUSB_GAMEPAD_LEFT_SHOULDER, "RB": B.XUSB_GAMEPAD_RIGHT_SHOULDER,
            "START": B.XUSB_GAMEPAD_START, "BACK": B.XUSB_GAMEPAD_BACK,
            "UP": B.XUSB_GAMEPAD_DPAD_UP, "DOWN": B.XUSB_GAMEPAD_DPAD_DOWN,
            "LEFT": B.XUSB_GAMEPAD_DPAD_LEFT, "RIGHT": B.XUSB_GAMEPAD_DPAD_RIGHT,
            "LS": B.XUSB_GAMEPAD_LEFT_THUMB, "RS": B.XUSB_GAMEPAD_RIGHT_THUMB,
        }
        self.pad.reset()
        self.pad.update()
        time.sleep(1.5)  # let Windows and the game see the new controller
        log.info("Virtual Xbox 360 controller connected")

    def _down(self, name: str) -> None:
        if name in ("LT", "RT"):
            (self.pad.left_trigger if name == "LT" else self.pad.right_trigger)(value=255)
        elif name[:3] in ("LS_", "RS_"):
            x, y = STICK_DIRS[name[3:]]
            fn = self.pad.left_joystick_float if name.startswith("LS") else self.pad.right_joystick_float
            fn(x_value_float=x, y_value_float=y)
        else:
            self.pad.press_button(button=self.buttons[name])
        self.pad.update()

    def _up(self, name: str) -> None:
        if name in ("LT", "RT"):
            (self.pad.left_trigger if name == "LT" else self.pad.right_trigger)(value=0)
        elif name[:3] in ("LS_", "RS_"):
            fn = self.pad.left_joystick_float if name.startswith("LS") else self.pad.right_joystick_float
            fn(x_value_float=0.0, y_value_float=0.0)
        else:
            self.pad.release_button(button=self.buttons[name])
        self.pad.update()

    def press(self, name: str, hold: Optional[float] = None) -> None:
        name = name.upper()
        self._down(name)
        time.sleep(self.hold if hold is None else hold)
        self._up(name)

    def tap(self, name: str, times: int, hold: Optional[float] = None, gap: Optional[float] = None) -> None:
        """Quick burst of presses (used to jump a whole page of rows/columns at once)."""
        for i in range(times):
            self.press(name, self.tap_hold if hold is None else hold)
            if i < times - 1:
                time.sleep(self.gap if gap is None else gap)

    def sequence(self, names: Iterable[str], pause: float = 0.35) -> None:
        for n in names:
            self.press(n)
            time.sleep(pause)

    def close(self) -> None:
        try:
            self.pad.reset()
            self.pad.update()
        except Exception:
            pass
