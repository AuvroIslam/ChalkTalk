"""Speaks each teaching step aloud, like a teacher talking while drawing.

Uses the voices built into Windows (offline, instant). The drawing waits for the
voice: the next step starts only after the current sentence has been spoken.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
import threading

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

import config

_SYMBOLS = [("→", " leads to "), ("->", " leads to "), ("⇒", " so "), ("×", " times "), ("·", " times "),
            ("÷", " divided by "), ("⊥", " perpendicular "), ("≈", " about "), ("≠", " not equal to "),
            ("≤", " at most "), ("≥", " at least "), ("∞", " infinity "), ("=", " equals "), ("+", " plus "),
            ("−", " minus "), ("√", " root "), ("²", " squared "), ("³", " cubed "), ("τ", " tau "),
            ("θ", " theta "), ("Δ", " change in "), ("π", " pi "), ("★", " "), ("/", " or ")]


def spoken(text: str) -> str:
    """Written notes into something a voice can read: symbols as words, no source brackets."""
    t = re.sub(r"\((?:video|p\.|page|slide)\s*[^)]*\)|\((?:[\w-]+\.)+[a-z]{2,}\)", "", str(text or ""))
    t = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", t)
    t = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " over ", t)
    for sym, word in _SYMBOLS:
        t = t.replace(sym, word)
    return re.sub(r"\s+", " ", t).strip(" ;")


def fallback_say(action: dict) -> str:
    """What to say for a step the model forgot to narrate: read out what it writes."""
    op = action.get("op")
    if op in ("note", "summary", "tag"):
        return str(action.get("text", ""))
    if op == "diagram":
        nodes = [str(n) for n in action.get("nodes") or []]
        return ". ".join(filter(None, [str(action.get("title", "")), ", then ".join(nodes)]))
    if op == "arrow":
        return str(action.get("label", ""))
    return ""


def _synthesize(text: str) -> bytes:
    from winrt.windows.media.speechsynthesis import SpeechSynthesizer
    from winrt.windows.storage.streams import DataReader

    async def run() -> bytes:
        s = SpeechSynthesizer()
        wanted = config.VOICE_NAME.lower()
        for v in SpeechSynthesizer.all_voices:
            if wanted and wanted in v.display_name.lower():
                s.voice = v
                break
        s.options.speaking_rate = config.VOICE_RATE
        stream = await s.synthesize_text_to_stream_async(text)
        size = stream.size
        reader = DataReader(stream.get_input_stream_at(0))
        await reader.load_async(size)
        buf = bytearray(size)
        reader.read_bytes(buf)
        return bytes(buf)

    return asyncio.run(run())


class Narrator(QObject):
    ready = Signal(int, bytes)  # utterance id, WAV (from the synthesis thread)
    done = Signal()             # finished speaking (or was stopped)

    def __init__(self):
        super().__init__()
        self.enabled = config.VOICE_OUT
        self.busy = False
        self._id = 0
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(1.0)
        self._player.setAudioOutput(self._audio)
        self._player.mediaStatusChanged.connect(self._status)
        self._player.errorOccurred.connect(lambda *_: self._finish())
        self.ready.connect(self._play)
        self._file = os.path.join(tempfile.gettempdir(), "chalktalk_voice_{}.wav")

    def say(self, text: str) -> None:
        text = spoken(text)
        if not self.enabled or not text:
            return
        self.stop()
        self._id += 1
        uid = self._id
        self.busy = True

        def work():
            try:
                self.ready.emit(uid, _synthesize(text))
            except Exception as e:  # never let the voice break the drawing
                print(f"  voice failed: {type(e).__name__}: {e}")
                self.ready.emit(uid, b"")

        threading.Thread(target=work, daemon=True).start()

    def _play(self, uid: int, wav: bytes) -> None:
        if uid != self._id:
            return  # a newer sentence (or a stop) replaced this one
        if not wav:
            self._finish()
            return
        path = self._file.format(uid % 4)
        with open(path, "wb") as f:
            f.write(wav)
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def _status(self, status) -> None:
        if status in (QMediaPlayer.EndOfMedia, QMediaPlayer.InvalidMedia):
            self._finish()

    def _finish(self) -> None:
        if self.busy:
            self.busy = False
            self.done.emit()

    def stop(self) -> None:
        self._id += 1
        self._player.stop()
        self._finish()

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self.stop()
