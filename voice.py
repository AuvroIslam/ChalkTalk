"""Voice input: record from the microphone until you stop talking, then turn it
into text with a Microsoft Foundry speech-to-text deployment (English).

Voice commands: "stop" / "cancel" / "clear" / "never mind" close ChalkTalk;
anything else is your question.
"""
from __future__ import annotations

import io
import re
import time
import wave

import numpy as np
from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

import config

SILENCE_AFTER_SPEECH = 1.2  # seconds of quiet that end the question
NO_SPEECH_TIMEOUT = 7.0
MAX_SECONDS = 20.0

_COMMANDS = {
    "stop": r"(stop|cancel|clear( (it|the screen|everything))?|close|never ?mind|dismiss|go away|that'?s all|thank you|thanks)",
}


def command_for(text: str) -> str | None:
    """'stop' if the whole utterance is a stop/clear command, else None."""
    t = re.sub(r"[^\w' ]+", "", text.lower()).strip()
    for name, pattern in _COMMANDS.items():
        if re.fullmatch(rf"(please |ok |okay |hey chalk ?talk )?{pattern}( please)?", t):
            return name
    return None


class Recorder(QObject):
    """Records 16 kHz mono speech; stops by itself after a pause."""

    level = Signal(float)          # 0..1 loudness, for the mascot
    finished = Signal(bytes)       # WAV bytes
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._src: QAudioSource | None = None
        self._io = None
        self._buf = bytearray()
        self._fmt: QAudioFormat | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

    @property
    def active(self) -> bool:
        return self._src is not None

    @Slot()
    def finish(self) -> None:
        self.stop(deliver=True)

    @Slot()
    def cancel(self) -> None:
        self.stop(deliver=False)

    @Slot()
    def start(self) -> None:
        dev = QMediaDevices.defaultAudioInput()
        if dev.isNull():
            self.failed.emit("No microphone found.")
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)
        if not dev.isFormatSupported(fmt):
            fmt = dev.preferredFormat()
        self._fmt = fmt
        self._src = QAudioSource(dev, fmt, self)
        self._buf.clear()
        self._io = self._src.start()
        self._t0 = time.monotonic()
        self._speech_at: float | None = None
        self._last_loud = self._t0
        self._noise = 0.01
        self._noise_set = False
        self._peak = 0.0
        self._timer.start()

    def stop(self, deliver: bool = True) -> None:
        if not self._src:
            return
        self._poll(final=True)
        self._timer.stop()
        self._src.stop()
        self._src = None
        self.level.emit(0.0)
        if deliver:
            if self._speech_at is None and self._peak < 1e-4:
                self.failed.emit("Your microphone sends silence. Check it's not muted, and that "
                                 "Settings > Privacy > Microphone allows desktop apps.")
            elif self._speech_at is None:
                self.failed.emit("I didn't hear anything.")
            else:
                self.finished.emit(self._wav())

    def _samples(self, raw: bytes) -> np.ndarray:
        f = self._fmt
        if f.sampleFormat() == QAudioFormat.Float:
            a = np.frombuffer(raw, dtype=np.float32)
        elif f.sampleFormat() == QAudioFormat.Int32:
            a = np.frombuffer(raw, dtype=np.int32) / 2**31
        else:
            a = np.frombuffer(raw, dtype=np.int16) / 32768.0
        ch = max(1, f.channelCount())
        if ch > 1:
            a = a[: len(a) // ch * ch].reshape(-1, ch).mean(axis=1)
        return a.astype(np.float32)

    def _poll(self, final: bool = False) -> None:
        if not self._io:
            return
        raw = bytes(self._io.readAll())
        if raw:
            self._buf += raw
            a = self._samples(raw)
            rms = float(np.sqrt(np.mean(a * a))) if len(a) else 0.0
            now = time.monotonic()
            if now - self._t0 < 0.6:
                # Room noise = the quietest moment early on. (Averaging would learn the user's
                # voice as "noise" when they start talking right away.)
                self._noise = min(self._noise, max(0.0003, rms)) if self._noise_set else max(0.0003, rms)
                self._noise_set = True
            # Relative to this mic's own floor, so a quiet (low-gain) mic still works.
            loud = rms > max(0.003, min(self._noise, 0.03) * 4)
            if loud:
                self._last_loud = now
                self._speech_at = self._speech_at or now
            self._peak = max(self._peak, rms)
            self.level.emit(min(1.0, rms / max(self._noise * 12, 0.02)))  # mascot's mouth follows your voice
        if final:
            return
        now = time.monotonic()
        if self._speech_at and now - self._last_loud > SILENCE_AFTER_SPEECH:
            self.stop()
        elif not self._speech_at and now - self._t0 > NO_SPEECH_TIMEOUT:
            self.stop()
        elif now - self._t0 > MAX_SECONDS:
            self.stop()

    def _wav(self) -> bytes:
        a = self._samples(bytes(self._buf))
        peak = float(np.abs(a).max()) if len(a) else 0.0
        if 0 < peak < 0.5:
            a = a * min(0.9 / peak, 60.0)  # quiet mic: boost so the transcriber hears it clearly
        pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._fmt.sampleRate())
            w.writeframes(pcm)
        return out.getvalue()


class Mic(QObject):
    """The Recorder on its own thread: opening a microphone takes ~1 s on Windows
    and would freeze the bar and the mascot if done on the UI thread."""

    def __init__(self):
        super().__init__()
        self.rec = Recorder()
        self.level, self.finished, self.failed = self.rec.level, self.rec.finished, self.rec.failed
        self._thread = QThread()
        self._thread.setObjectName("mic")
        self.rec.moveToThread(self._thread)
        self._thread.start()
        self.active = False  # tracked here so the UI never waits on the mic thread
        self.rec.finished.connect(self._idle)
        self.rec.failed.connect(self._idle)

    def _idle(self, *_):
        self.active = False

    def start(self) -> None:
        self.active = True
        QMetaObject.invokeMethod(self.rec, "start", Qt.QueuedConnection)

    def stop(self, deliver: bool = True) -> None:
        """deliver=True: transcribe what was said (active until the audio arrives). False: discard."""
        if not self.active:
            return
        if not deliver:
            self.active = False
        QMetaObject.invokeMethod(self.rec, "finish" if deliver else "cancel", Qt.QueuedConnection)

    def shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(2000)


_stt = None


def transcribe(wav: bytes) -> str:
    """Speech to text on Microsoft Foundry (English)."""
    global _stt
    import openai

    if not (config.STT_ENDPOINT and config.STT_KEY):
        raise RuntimeError("Voice needs CHALK_STT_ENDPOINT and CHALK_STT_KEY in .env")
    if _stt is None:
        # Audio models are served on the classic per-deployment route, not /openai/v1/ (it returns 404).
        base = config.STT_ENDPOINT.split("/openai/")[0]
        _stt = openai.AzureOpenAI(azure_endpoint=base, api_key=config.STT_KEY, api_version="2025-03-01-preview",
                                  timeout=30.0, max_retries=1)
    r = _stt.audio.transcriptions.create(
        model=config.STT_MODEL, file=("question.wav", wav, "audio/wav"), language="en",
        prompt="A student asking a tutor about what is on their screen: a slide, video, PDF or web page.")
    return r.text.strip()


def warm_up() -> None:
    """Open the connection early with a fraction of a second of silence."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 4000)
    transcribe(out.getvalue())
