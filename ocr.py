"""Local, fast OCR using the built-in Windows OCR engine (Windows.Media.Ocr).

Returns words and lines with pixel-exact bounding boxes. These boxes are what
make drawing placement accurate: the LLM refers to line ids, and we draw on the
boxes OCR measured, instead of trusting the model to guess pixel coordinates.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import PySide6.QtCore  # noqa: F401 - must load before winrt: the other order crashes (clashing C++ runtime DLLs)
from PIL import Image
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def union(self, other: "Box") -> "Box":
        x, y = min(self.x, other.x), min(self.y, other.y)
        return Box(x, y, max(self.x2, other.x2) - x, max(self.y2, other.y2) - y)

    def pad(self, p: float) -> "Box":
        return Box(self.x - p, self.y - p, self.w + 2 * p, self.h + 2 * p)

    def scaled(self, s: float) -> "Box":
        return Box(self.x * s, self.y * s, self.w * s, self.h * s)


@dataclass
class Word:
    text: str
    box: Box


@dataclass
class Line:
    id: str
    text: str
    box: Box
    words: list[Word] = field(default_factory=list)


_engine: OcrEngine | None = None


def _get_engine() -> OcrEngine:
    global _engine
    if _engine is None:
        _engine = OcrEngine.try_create_from_user_profile_languages()
        if _engine is None:
            raise RuntimeError(
                "Windows OCR is unavailable. Install an OCR language pack: "
                "Settings > Time & language > Language > English > Language options > OCR."
            )
    return _engine


async def _recognize(img: Image.Image):
    rgba = img.convert("RGBA")
    # Windows wants BGRA byte order.
    r, g, b, a = rgba.split()
    bgra = Image.merge("RGBA", (b, g, r, a)).tobytes()
    writer = DataWriter()
    writer.write_bytes(bgra)
    bitmap = SoftwareBitmap.create_copy_from_buffer(
        writer.detach_buffer(), BitmapPixelFormat.BGRA8, img.width, img.height
    )
    return await _get_engine().recognize_async(bitmap)


def ocr_lines(img: Image.Image) -> list[Line]:
    """OCR an image and return its text lines (coordinates in image pixels)."""
    _get_engine()
    limit = OcrEngine.max_image_dimension
    scale = 1.0
    if max(img.size) > limit:
        scale = limit / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    result = asyncio.run(_recognize(img))
    lines: list[Line] = []
    for i, ln in enumerate(result.lines):
        words = []
        for w in ln.words:
            r = w.bounding_rect
            words.append(Word(w.text, Box(r.x, r.y, r.width, r.height).scaled(1 / scale)))
        if not words:
            continue
        box = words[0].box
        for w in words[1:]:
            box = box.union(w.box)
        lines.append(Line(id=f"L{i + 1}", text=ln.text, box=box, words=words))
    return lines
