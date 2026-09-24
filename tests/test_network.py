"""Tests that need the internet (YouTube, Firecrawl) or a real Chrome window.
Run: python -m pytest tests/test_network.py -q"""
import ctypes
import subprocess
import time
from ctypes import wintypes

import pytest

import config


def test_youtube_real_video_loads_captions_and_details():
    from context import SearchVideoTranscript, YouTubeSource

    yt = YouTubeSource("aircAruvnKk", "https://www.youtube.com/watch?v=aircAruvnKk",
                       "But what is a neural network? - YouTube - Google Chrome")
    assert yt.wait(40), yt.load_error()
    assert len(yt.segments) > 100 and yt.channel == "3Blue1Brown" and yt.length > 1000
    assert "10:" in yt.call(SearchVideoTranscript(query="sigmoid squish", reason="r"))[:8]


def test_youtube_video_without_captions_degrades_gracefully():
    from context import YouTubeSource

    yt = YouTubeSource("xxxxxxxxxxx", "https://www.youtube.com/watch?v=xxxxxxxxxxx", "x - YouTube")
    yt.wait(40)
    assert yt.tools() == [] and "not available" in yt.digest("")


@pytest.mark.skipif(not config.FIRECRAWL_KEY, reason="no Firecrawl key")
def test_firecrawl_search_and_scrape():
    import context

    res = context.web_search("gradient descent learning rate")
    assert res.count("http") >= 2
    page = context.read_url("https://en.wikipedia.org/wiki/Gradient_descent")
    assert "Gradient descent" in page and len(page) > 5000


def _find_window(fragment: str, timeout: float = 15):
    end = time.time() + timeout
    while time.time() < end:
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def cb(h, _):
            b = ctypes.create_unicode_buffer(400)
            ctypes.windll.user32.GetWindowTextW(h, b, 400)
            if ctypes.windll.user32.IsWindowVisible(h) and fragment in b.value:
                found.append((h, b.value))
            return True

        ctypes.windll.user32.EnumWindows(cb, 0)
        if found:
            return found[0]
        time.sleep(0.5)
    return None


CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def test_detect_real_chrome_youtube_tab():
    import context
    from window import Foreground

    subprocess.Popen([CHROME, "--new-window", "https://www.youtube.com/watch?v=aircAruvnKk&t=65s"])
    hit = _find_window("YouTube")
    assert hit, "Chrome window with YouTube didn't appear"
    time.sleep(2)
    src = context.detect(Foreground(hit[0], hit[1], "chrome.exe"))
    assert isinstance(src, context.YouTubeSource) and src.video_id == "aircAruvnKk" and src.start_hint == 65
    ctypes.windll.user32.PostMessageW(hit[0], 0x0010, 0, 0)  # WM_CLOSE the test window
