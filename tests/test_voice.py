"""Voice pipeline without a real microphone: a recorded sentence is streamed in
real time into the recorder, which must stop by itself when the speech ends.
Run: python -m pytest tests/test_voice.py -q   (the transcription part needs the Foundry keys)"""
import time
import wave

import numpy as np
import pytest

import config


def _speech_wav(tmp_path, gain=1.0):
    path = tmp_path / "q.wav"
    import subprocess

    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    f"$s.SetOutputToWaveFile('{path}'); $s.Speak('What does the learning rate mean on this slide?');"
                    "$s.Dispose()"], check=True, capture_output=True)
    with wave.open(str(path)) as w:
        rate, pcm = w.getframerate(), w.readframes(w.getnframes())
    a = (np.frombuffer(pcm, dtype=np.int16) * gain).astype(np.int16)
    return rate, a.tobytes()


def _run(qapp, rate, pcm, lead_silence=0.0):
    from PySide6.QtMultimedia import QAudioFormat
    import voice

    stream = b"\0\0" * int(rate * lead_silence) + pcm
    noise = (np.random.default_rng(0).normal(0, 3, int(rate * 12))).astype(np.int16).tobytes()  # faint hiss

    class FakeIO:
        def __init__(self):
            self.t0, self.sent = time.monotonic(), 0

        def readAll(self):
            due = int((time.monotonic() - self.t0) * rate) * 2
            chunk = stream[self.sent:due] if self.sent < len(stream) else noise[:int(rate * 0.05) * 2]
            self.sent += len(chunk)
            return chunk

    class Src:
        def stop(self):
            pass

    r = voice.Recorder()
    out = {}
    r.finished.connect(lambda wav: out.setdefault("wav", wav))
    r.failed.connect(lambda m: out.setdefault("err", m))
    fmt = QAudioFormat()
    fmt.setSampleRate(rate)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.Int16)
    r._fmt, r._io, r._src, r._buf = fmt, FakeIO(), Src(), bytearray()
    r._t0 = time.monotonic()
    r._speech_at, r._last_loud, r._noise, r._noise_set, r._peak = None, r._t0, 0.01, False, 0.0
    r._timer.start()
    t0 = time.monotonic()
    while not out and time.monotonic() - t0 < 15:
        qapp.processEvents()
        time.sleep(0.01)
    return out, time.monotonic() - t0


@pytest.mark.parametrize("gain,lead", [(1.0, 0.0), (1.0, 1.0), (0.02, 0.5)])  # loud / pause first / very quiet mic
def test_recorder_stops_after_speech(qapp, tmp_path, gain, lead):
    rate, pcm = _speech_wav(tmp_path, gain)
    out, took = _run(qapp, rate, pcm, lead)
    speech = len(pcm) / 2 / rate + lead
    assert "wav" in out, out.get("err")
    assert took < speech + 2.5  # stopped by itself shortly after the talking ended


def test_silence_times_out(qapp):
    out, took = _run(qapp, 16000, b"")
    assert "err" in out and 6 < took < 9


def test_voice_commands():
    from voice import command_for

    for s in ["Stop.", "okay, clear the screen", "Never mind!", "Thank you.", "cancel please"]:
        assert command_for(s) == "stop", s
    for s in ["stop explaining the bias?", "What does this formula mean?", "Why did it stop here?"]:
        assert command_for(s) is None, s


@pytest.mark.skipif(not config.STT_ENDPOINT, reason="no speech-to-text deployment configured")
def test_transcription_is_accurate(qapp, tmp_path):
    import voice

    rate, pcm = _speech_wav(tmp_path, 0.02)  # even from a very quiet mic (gets boosted)
    out, _ = _run(qapp, rate, pcm)
    text = voice.transcribe(out["wav"]).lower()
    assert "learning rate" in text and "slide" in text
