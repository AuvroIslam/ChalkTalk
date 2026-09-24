"""System-wide hotkey via Win32 RegisterHotKey (works whatever app is focused)."""
from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable

MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "f1": 0x70}
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312


def parse(combo: str) -> tuple[int, int]:
    mods, vk = MOD_NOREPEAT, 0
    for part in combo.lower().replace(" ", "").split("+"):
        if part in MODS:
            mods |= MODS[part]
        elif part in KEYS:
            vk = KEYS[part]
        elif len(part) == 1:
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            vk = 0x70 + int(part[1:]) - 1
    if not vk:
        raise ValueError(f"Can't understand hotkey {combo!r}")
    return mods, vk


def listen(combo: str, callback: Callable[[], None], hotkey_id: int = 1) -> threading.Thread:
    mods, vk = parse(combo)
    ok = threading.Event()

    def run():
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(None, hotkey_id, mods, vk):
            which = "CHALK_VOICE_HOTKEY" if hotkey_id == 2 else "CHALK_HOTKEY"
            print(f"Could not register hotkey {combo} (another app uses it). Pick another with {which} in .env.")
            ok.set()
            return
        ok.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                callback()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    ok.wait(2)
    return t
