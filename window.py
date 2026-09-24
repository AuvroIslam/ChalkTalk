"""What app is in front: window title, process, and (for browsers) the exact URL."""
from __future__ import annotations

import ctypes
import os
import re
from ctypes import wintypes
from dataclasses import dataclass

BROWSERS = {"chrome.exe", "msedge.exe", "brave.exe", "firefox.exe", "opera.exe", "vivaldi.exe", "arc.exe"}


@dataclass
class Foreground:
    hwnd: int
    title: str
    exe: str

    @property
    def is_browser(self) -> bool:
        return self.exe in BROWSERS


def foreground() -> Foreground:
    """Call before our overlay appears, so this is the user's app."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    exe = ""
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if h:
        path = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, path, ctypes.byref(size)):
            exe = os.path.basename(path.value).lower()
        ctypes.windll.kernel32.CloseHandle(h)
    return Foreground(hwnd, buf.value, exe)


def browser_url(hwnd: int) -> str | None:
    """Read the address bar through UI Automation (exact, unlike OCR). ~100-400 ms."""
    import uiautomation as auto

    with auto.UIAutomationInitializerInThread():
        win = auto.ControlFromHandle(hwnd)
        if win is None:
            return None
        edit = win.EditControl(searchDepth=14, Name="Address and search bar")  # Chrome / Edge / Brave
        if not edit.Exists(0, 0):
            edit = win.EditControl(searchDepth=14, AutomationId="urlbar-input")  # Firefox
        if not edit.Exists(0, 0):
            edit = win.EditControl(searchDepth=14)  # anything that looks like an address bar
            if not edit.Exists(0, 0):
                return None
        try:
            value = edit.GetValuePattern().Value.strip()
        except Exception:
            return None
    if not value:
        return None
    # Local files show as "D:/folder/file.pdf" (or file:///...) in the address bar
    if re.match(r"^[A-Za-z]:[/\\]", value):
        return "file:///" + value.replace("\\", "/").replace(" ", "%20")
    if " " in value:
        return None
    if "://" not in value:
        value = "https://" + value
    return value
