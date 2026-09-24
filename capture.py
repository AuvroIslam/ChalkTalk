"""Screen capture of the monitor under the mouse cursor, in physical pixels."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import mss
from PIL import Image


@dataclass
class Shot:
    image: Image.Image  # physical pixels
    left: int  # monitor origin in physical desktop coordinates
    top: int


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def cursor_pos() -> tuple[int, int]:
    pt = _POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def grab_monitor_under_cursor() -> Shot:
    with mss.MSS() as sct:  # mss makes the process per-monitor DPI aware
        x, y = cursor_pos()
        mon = sct.monitors[1]
        for m in sct.monitors[1:]:
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                mon = m
                break
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return Shot(img, mon["left"], mon["top"])
