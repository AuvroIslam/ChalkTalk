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
    kind: str = ""  # "node": a label inside a drawn circle (the box is the whole circle)


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


def _words_of(result, scale: float = 1.0, dx: float = 0, dy: float = 0) -> list[Word]:
    out = []
    for ln in result.lines:
        for w in ln.words:
            r = w.bounding_rect
            out.append(Word(w.text, Box(r.x / scale + dx, r.y / scale + dy, r.width / scale, r.height / scale)))
    return out


def recover_small_text(img: Image.Image, lines: list[Line]) -> list[Line]:
    """OCR engines skip isolated single characters: a graph's node letters (A, B…)
    and edge weights (5, 7…) come back empty. Find those small marks, line them up
    on one strip (in a row, OCR reads them fine), and map what it read back to
    where each mark really is. Nodes drawn as circles get the circle as their box."""
    import cv2
    import numpy as np

    gray = np.asarray(img.convert("L"))
    H, Wd = gray.shape
    dark_bg = gray.mean() < 110
    ink = (gray > 140) if dark_bg else (gray < 120)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    known = [w.box.pad(3) for ln in lines for w in ln.words]

    def overlaps_known(b: Box) -> bool:
        return any(b.x < k.x2 and k.x < b.x2 and b.y < k.y2 and k.y < b.y2 for k in known)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.009 * H <= h <= 0.07 * H and w <= 2.2 * h and area >= 10 and w >= 2):
            continue
        b = Box(float(x), float(y), float(w), float(h))
        if not overlaps_known(b):
            comps.append(b)
    comps.sort(key=lambda b: (b.cy, b.x))
    # glue characters of one token together ("1" "8" -> "18")
    tokens: list[Box] = []
    for b in sorted(comps, key=lambda b: b.x):
        for k, t in enumerate(tokens):
            same_row = min(t.y2, b.y2) - max(t.y, b.y) > 0.5 * min(t.h, b.h)
            if same_row and 0 <= b.x - t.x2 < 0.45 * max(t.h, b.h):
                tokens[k] = t.union(b)
                break
        else:
            tokens.append(b)
    tokens = [t for t in tokens if t.w <= 3.5 * t.h][:120]
    if not tokens:
        return []

    # A sheet of rows: "is H is 2 is 9 is ...". OCR drops lone characters but reads a
    # line of real words perfectly; the separator word tells us where each token ends.
    # (Symbols like "#" or "," as separators make Windows OCR return nothing at all.)
    from PIL import ImageDraw, ImageFont, ImageOps

    th, pad, sp = 56, 4, 16
    SEP = "is"
    font = ImageFont.truetype("arial.ttf", int(th * 0.8))
    hash_w = int(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(SEP, font=font))
    crops = []
    for t in tokens:
        c = img.convert("L").crop((int(t.x) - pad, int(t.y) - pad, int(t.x2) + pad, int(t.y2) + pad))
        c = c.resize((max(1, int(c.width * th / c.height)), th), Image.LANCZOS)
        crops.append(ImageOps.invert(c) if dark_bg else c)
    found: dict[int, str] = {}
    # Two passes: tokens missed the first time get a second sheet of their own.
    for pass_tokens in (list(range(len(crops))), None):
        todo = pass_tokens if pass_tokens is not None else [k for k in range(len(crops)) if k not in found]
        if not todo:
            break
        found.update(_read_sheet(crops, todo, th, sp, SEP, font, hash_w))
    circles = _circles(gray)
    out = []
    for k, text in found.items():
        text = text.strip(".,;:'\"`")
        if not text or len(text) > 3 or not text.isalnum():
            continue  # only short labels (node names, numbers); anything else is likely an icon
        t = tokens[k]
        box, in_circle = t, False
        for (ccx, ccy, r) in circles:  # a letter inside a circle stands for the whole node
            if (t.cx - ccx) ** 2 + (t.cy - ccy) ** 2 < (0.7 * r) ** 2 and r < 4 * max(t.w, t.h):
                box, in_circle = Box(ccx - r, ccy - r, 2 * r, 2 * r), True
                break
        if not in_circle and text in ("I", "l", "|", "i"):
            text = "1"  # a lone stroke outside a node is the digit one (an edge weight)
        out.append(Line(id="", text=text, box=box, words=[Word(text, box)], kind="node" if in_circle else ""))
    return out


def _read_sheet(crops, todo, th, sp, SEP, font, sep_w) -> dict[int, str]:
    from PIL import ImageDraw

    n_rows = max(1, -(-len(todo) // 8))
    per_row = -(-len(todo) // n_rows)  # rows of even length: OCR skips very short lines
    rows = [todo[i:i + per_row] for i in range(0, len(todo), per_row)]
    hash_w = sep_w
    row_h = th + 60
    sheet_w = 40 + max(sum(crops[k].width + 2 * sp + hash_w for k in r) for r in rows) + hash_w + 40
    sheet = Image.new("L", (sheet_w, row_h * len(rows) + 40), 255)
    d = ImageDraw.Draw(sheet)
    cells: dict[int, tuple[int, int, int, int]] = {}  # token -> where it sits on the sheet
    for ri, r in enumerate(rows):
        x, y = 40, 30 + ri * row_h
        for k in r:
            d.text((x, y + th * 0.08), SEP, font=font, fill=0)
            x += hash_w + sp
            sheet.paste(crops[k], (x, y))
            cells[k] = (x, y, x + crops[k].width, y + th)
            x += crops[k].width + sp
        d.text((x, y + th * 0.08), SEP, font=font, fill=0)
    result = asyncio.run(_recognize(sheet.convert("RGB")))

    found: dict[int, str] = {}
    for ln in result.lines:
        for w in ln.words:
            if w.text.lower() in (SEP, "1s", "ls", "is.", "is,"):
                continue
            r = w.bounding_rect
            cx, cy = r.x + r.width / 2, r.y + r.height / 2
            for k, (x0, y0, x1, y1) in cells.items():  # whichever token this word sits on
                if x0 - sp <= cx <= x1 + sp and y0 - 20 <= cy <= y1 + 20:
                    found[k] = found.get(k, "") + w.text
                    break
    return found


def _circles(gray) -> list[tuple[float, float, float]]:
    import cv2

    H, W = gray.shape
    g = cv2.medianBlur(gray, 5)
    found = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.03 * W, param1=120, param2=38,
                             minRadius=int(0.008 * W), maxRadius=int(0.05 * W))
    return [] if found is None else [tuple(map(float, c)) for c in found[0]]


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
    try:
        lines += recover_small_text(img, lines)
    except Exception as e:  # a bonus pass; normal OCR still stands
        print(f"  small-text recovery skipped: {type(e).__name__}: {e}")
    return lines
