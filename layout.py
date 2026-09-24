"""Geometry of the captured screen: where things are, where empty space is,
and which ink colours stay readable.

Everything here is in *logical* overlay coordinates (the units Qt draws in), so
a 300%-scaled laptop screen and a 100% monitor behave the same.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ocr import Box, Line, Word

CELL = 6  # logical px per occupancy-grid cell


@dataclass
class Region:
    id: str
    box: Box
    color: str = ""  # figure parts: the ink colour ("red", "white"...), to match "the red 50 N arrow"
    tip: tuple[float, float] | None = None  # figure parts: the free end (arrowhead, label)
    hue: float | None = None    # figure parts: mean OpenCV hue (0-179) of its ink; None for white/black/grey
    angle: float | None = None  # figure parts: direction from its crowded end to its free end (degrees, y down)
    straight: float = 0.0       # figure parts: 1 = a straight stroke (an arrow, a rod), 0 = a scribble (handwriting)


# The words people use for a colour, as OpenCV hue ranges (overlapping on purpose: one person's
# purple is another's magenta).
_HUE_WORDS = {
    "red": [(165, 180), (0, 10)], "crimson": [(165, 180), (0, 8)], "orange": [(6, 24)], "brown": [(4, 24)],
    "peach": [(4, 24)], "yellow": [(20, 38)], "gold": [(18, 34)], "lime": [(30, 50)], "green": [(35, 88)],
    "teal": [(78, 100)], "cyan": [(80, 104)], "aqua": [(80, 104)], "turquoise": [(78, 100)],
    "light blue": [(85, 112)], "sky blue": [(88, 112)], "blue": [(90, 132)], "navy": [(105, 130)],
    "indigo": [(115, 138)], "violet": [(122, 155)], "purple": [(122, 158)], "lavender": [(120, 150)],
    "magenta": [(138, 172)], "pink": [(140, 178)], "fuchsia": [(140, 172)],
}
_DIRS = {"right": 0, "down-right": 45, "down": 90, "down-left": 135, "left": 180, "up-left": 225, "up": 270,
         "up-right": 315, "east": 0, "south": 90, "west": 180, "north": 270}


# ink colour names by OpenCV hue (0-179)
_HUES = [(8, "red"), (22, "orange"), (35, "yellow"), (85, "green"), (100, "cyan"), (130, "blue"), (150, "purple"),
         (170, "pink"), (180, "red")]


def _seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from point p to the segment a-b."""
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _direction(angle: float | None) -> str:
    """Which way a mark points (screen y grows downward); "" for a short or round mark."""
    if angle is None:
        return ""
    names = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
    return names[int(((angle + 22.5) % 360) // 45)]


def _hue_name(hue: float | None, plain: str) -> str:
    if hue is None:
        return plain
    lo = 0
    for hi, name in _HUES:
        if lo <= hue < hi:
            return name
        lo = hi
    return "red"


def _hue_matches(word: str, hue: float | None) -> float:
    """1 if the colour word fits this hue, 0.5 near the edge of its range, else 0."""
    word = word.lower().strip()
    if hue is None:
        return 1.0 if word in ("white", "black", "grey", "gray", "chalk") else 0.0
    best = 0.0
    for key, ranges in _HUE_WORDS.items():
        if key in word:
            for lo, hi in ranges:
                if lo <= hue <= hi:
                    return 1.0
                if lo - 6 <= hue <= hi + 6:
                    best = 0.5
    return best


def _scale_line(line: Line, s: float) -> Line:
    return Line(line.id, line.text, line.box.scaled(s), [Word(w.text, w.box.scaled(s)) for w in line.words], line.kind)


def _reading_order(lines: list[Line]) -> list[Line]:
    """Sort top-to-bottom, left-to-right, and renumber L1..Ln."""
    lines = sorted(lines, key=lambda l: l.box.cy)
    rows: list[list[Line]] = []
    for ln in lines:
        if rows and abs(rows[-1][0].box.cy - ln.box.cy) < 0.5 * min(rows[-1][0].box.h, ln.box.h):
            rows[-1].append(ln)
        else:
            rows.append([ln])
    ordered = [ln for row in rows for ln in sorted(row, key=lambda l: l.box.x)]
    for i, ln in enumerate(ordered):
        ln.id = f"L{i + 1}"
    return ordered


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


# ---- colour -----------------------------------------------------------------

INKS = {  # (for light backgrounds, for dark backgrounds)
    "red": ("#D62828", "#FF7A7A"),
    "blue": ("#1D4ED8", "#7CB8FF"),
    "green": ("#137A3A", "#5BE08A"),
    "purple": ("#7C3AED", "#C9A7FF"),
    "orange": ("#C2410C", "#FFB067"),
}


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def luminance(rgb) -> float:
    def ch(c: float) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


class Scene:
    def __init__(self, image: Image.Image, lines: list[Line], scale: float):
        """image: physical-pixel screenshot. lines: OCR result in physical px.
        scale: logical px per physical px (1/devicePixelRatio)."""
        self.image = image
        self.scale = scale
        self.w = image.width * scale
        self.h = image.height * scale
        self.lines = _reading_order([_scale_line(l, scale) for l in lines])
        self.line_by_id = {l.id: l for l in self.lines}
        self.gw = math.ceil(self.w / CELL)
        self.gh = math.ceil(self.h / CELL)

        gray = np.asarray(image.convert("L"))
        edges = cv2.Canny(gray, 40, 120)
        density = cv2.resize(edges.astype(np.float32) / 255, (self.gw, self.gh), interpolation=cv2.INTER_AREA)
        occ = (density > 0.015).astype(np.float32)
        for ln in self.lines:
            self._mark(occ, ln.box.pad(3), 1.0)
        self.occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        self.reserved = np.zeros_like(self.occ)
        self.colors = np.asarray(image.convert("RGB").resize((self.gw, self.gh), Image.BOX)).astype(np.float32)

        self.regions = self._find_regions(edges)
        self.region_by_id = {r.id: r for r in self.regions}
        self.parts = self._find_parts(np.asarray(image.convert("RGB")))
        self.part_by_id = {p.id: p for p in self.parts}

        self.hidden: list[Box] = []  # covered by our own bar / the taskbar: never mark things there

        # Mapping between the image the model saw and logical coordinates.
        self.view = Box(0, 0, self.w, self.h)
        self.view_px = (self.w, self.h)

    # ---- grid helpers -----------------------------------------------------

    def _cells(self, b: Box):
        x0 = max(0, int(b.x // CELL))
        y0 = max(0, int(b.y // CELL))
        x1 = min(self.gw, math.ceil(b.x2 / CELL))
        y1 = min(self.gh, math.ceil(b.y2 / CELL))
        return y0, y1, x0, x1

    def _mark(self, grid: np.ndarray, b: Box, value: float) -> None:
        y0, y1, x0, x1 = self._cells(b)
        if y1 > y0 and x1 > x0:
            grid[y0:y1, x0:x1] = np.maximum(grid[y0:y1, x0:x1], value)

    def reserve(self, b: Box, weight: float = 1.0) -> None:
        """Mark space as used by a drawing so later notes avoid it."""
        self._mark(self.reserved, b, weight)

    def unreserve(self, b: Box) -> None:
        y0, y1, x0, x1 = self._cells(b)
        if y1 > y0 and x1 > x0:
            self.reserved[y0:y1, x0:x1] = 0

    def bg_color(self, b: Box) -> tuple[float, float, float]:
        y0, y1, x0, x1 = self._cells(b.pad(CELL))
        patch = self.colors[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            patch = self.colors.reshape(-1, 3)
        return tuple(np.median(patch, axis=0))  # type: ignore[return-value]

    def is_dark(self, b: Box) -> bool:
        return luminance(self.bg_color(b)) < 0.3

    def ink(self, name: str | None, b: Box) -> str:
        """An ink colour that contrasts with the background around b."""
        bg = self.bg_color(b)
        on_light, on_dark = INKS.get((name or "red").lower(), INKS["red"])
        first, second = (on_light, on_dark) if luminance(bg) > 0.3 else (on_dark, on_light)
        if contrast(hex_rgb(first), bg) >= 3 or contrast(hex_rgb(first), bg) >= contrast(hex_rgb(second), bg):
            return first
        return second

    # ---- regions (figures, images, charts) --------------------------------

    def _find_regions(self, edges: np.ndarray) -> list[Region]:
        mask = edges.copy()
        inv = 1 / self.scale
        for ln in self.lines:
            b = ln.box.pad(4).scaled(inv)
            mask[max(0, int(b.y)) : int(b.y2) + 1, max(0, int(b.x)) : int(b.x2) + 1] = 0
        k = max(3, int(14 * inv))
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = self.w * self.h
        found = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            b = Box(x, y, w, h).scaled(self.scale)
            if 0.01 * area <= b.w * b.h <= 0.8 * area and min(b.w, b.h) >= 40:
                found.append(b)
        found = sorted(found, key=lambda b: -b.w * b.h)[:10]
        found.sort(key=lambda b: (round(b.y / 40), b.x))
        return [Region(f"R{i + 1}", b) for i, b in enumerate(found)]

    def _find_parts(self, rgb: np.ndarray) -> list[Region]:
        """The separate marks inside a drawing (a whiteboard, chalk video, diagram):
        the pivot, the rod, the force arrow, a hand-written "F". OCR can't name these,
        so each gets an id the tutor can point at. Photos and thumbnails are skipped."""
        inv = 1 / self.scale
        found: list[tuple[Box, str]] = []
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for r in self.regions:
            if min(r.box.w, r.box.h) < 150:
                continue  # toolbars and strips: their icons aren't a figure
            pb = r.box.scaled(inv)
            x0, y0, x1, y1 = int(pb.x), int(pb.y), int(pb.x2), int(pb.y2)
            patch = rgb[y0:y1, x0:x1].astype(np.int16)
            if patch.size == 0:
                continue
            bg = np.median(patch.reshape(-1, 3), axis=0)
            ink = np.abs(patch - bg).max(axis=2) > 70
            if ink.mean() > 0.15:
                continue  # busy all over: a photo or a thumbnail, not a drawing
            for ln in self.lines:  # text is already listed as lines
                b = ln.box.pad(3).scaled(inv)
                ink[max(0, int(b.y) - y0):max(0, int(b.y2) - y0 + 1), max(0, int(b.x) - x0):max(0, int(b.x2) - x0 + 1)] = False
            # Split by ink colour first: in a multi-colour sketch each force, its arrow and its
            # label ("F1 = 50N") share a colour, and they all touch at the object in the middle.
            h_, s_, v_ = (hsv[y0:y1, x0:x1, c].astype(np.int16) for c in range(3))
            names = np.full(ink.shape, "plain", dtype=object)
            coloured = (s_ > 80) & (v_ > 80)
            lo = 0
            for hi, name in _HUES:
                names[coloured & (h_ >= lo) & (h_ < hi)] = name
                lo = hi
            ra = (x1 - x0) * (y1 - y0)
            for name in dict.fromkeys(names[ink].tolist()):
                sel = ink & (names == name)
                if sel.sum() < 30:
                    continue
                plain = "white" if bg.mean() < 110 else "black"
                hues = h_ if name != "plain" else None
                # A filled shape (the object a diagram's forces act on) is not the thin arrow of the
                # same colour touching it: split off what's thick.
                dt = cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 3)
                core = dt > 6 * inv
                if core.sum() > (20 * inv) ** 2:
                    kk = max(3, int(14 * inv))
                    blob = cv2.dilate(core.astype(np.uint8), np.ones((kk, kk), np.uint8)).astype(bool) & sel
                    crowd = np.pad((ink & ~blob).astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                    for b, t, _a, hue, _s in self._stroke_parts(blob, x0, y0, ra, inv, crowd, hues):
                        found.append(Region("", b, _hue_name(hue, plain) + ", filled shape", t, hue, None))
                    sel = sel & ~blob
                crowd = np.pad((ink & ~sel).astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                for b, t, angle, hue, straight in self._stroke_parts(sel, x0, y0, ra, inv, crowd, hues):
                    d = _direction(angle)
                    round_ = 0.8 <= b.w / max(b.h, 1) <= 1.25 and max(b.w, b.h) <= 60 and \
                        (angle is None or math.hypot(b.w, b.h) < 70)
                    desc = _hue_name(hue, plain) + (", round" if round_ else "") + (f", points {d}" if d and not round_ else "")
                    found.append(Region("", b, desc, t, hue, None if round_ else angle, straight))
            found += self._rings(ink, x0, y0, inv, plain, h_, coloured)
        # Keep the most prominent marks. Rank by length, not area: a force arrow is long and thin.
        rings: list[Region] = []
        for p in found:  # overlapping regions find the same ring twice
            if p.color.endswith("round") and not any(abs(p.box.cx - q.box.cx) < p.box.w / 2 and
                                                     abs(p.box.cy - q.box.cy) < p.box.h / 2 for q in rings):
                rings.append(p)
        rings = rings[:4]
        strokes = [p for p in found if not p.color.endswith("round")]
        found = sorted(strokes, key=lambda p: -max(p.box.w, p.box.h))[:36] + rings
        found.sort(key=lambda p: (round(p.box.y / 40), p.box.x))
        for i, p in enumerate(found):
            p.id = f"P{i + 1}"
        return found

    def graph_edges(self) -> list[tuple[Line, Line, str]]:
        """The edges of a drawn graph, read from the picture: two nodes are joined when a line
        of ink runs between their circles; its weight is the number label sitting on that line.
        The tutor gets this list, so it can't invent an edge that isn't there."""
        if hasattr(self, "_edges"):
            return self._edges
        nodes = [l for l in self.lines if l.kind == "node"]
        labels = [l for l in self.lines if l.kind != "node" and re.fullmatch(r"-?\d{1,3}", l.text.strip())]
        self._edges: list[tuple[Line, Line, str]] = []
        if len(nodes) < 3:
            return self._edges
        gray = np.asarray(self.image.convert("L"))
        dark = gray.mean() < 110
        ink = ((gray > 100) if dark else (gray < 170)).astype(np.uint8)  # thin anti-aliased lines are light grey
        inv = 1 / self.scale
        # Only the lines: blank out the node circles first. (Not the weight labels: one sitting on a
        # line would cut it in two, and digits are too short to pass for a line themselves.)
        for n in nodes:
            cv2.circle(ink, (int(n.box.cx * inv), int(n.box.cy * inv)), int(n.box.w / 2 * inv * 1.15), 0, -1)
        r = float(np.median([n.box.w / 2 for n in nodes])) * inv
        segs = cv2.HoughLinesP(ink * 255, 1, np.pi / 180, threshold=int(r), minLineLength=int(r), maxLineGap=int(r / 3))
        if segs is None:
            return self._edges
        centres = [(n, n.box.cx * inv, n.box.cy * inv, n.box.w / 2 * inv) for n in nodes]
        wide = cv2.dilate(ink, np.ones((5, 5), np.uint8))
        H, W = ink.shape

        def node_at(x, y, away_x, away_y):
            # Follow the line on from this end (the detector often breaks one long line into
            # pieces) until it reaches a node's circle; small gaps (a weight label) are bridged.
            dx, dy = x - away_x, y - away_y
            n_ = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n_, dy / n_
            px, py, gap = float(x), float(y), 0.0
            for _ in range(int(40 * r)):
                for n, cx, cy, rad in centres:
                    if math.hypot(px - cx, py - cy) < rad * 1.25:
                        return n
                px, py = px + dx * 2, py + dy * 2
                if not (0 <= px < W and 0 <= py < H):
                    return None
                gap = 0.0 if wide[int(py), int(px)] else gap + 2
                if gap > r * 0.8:
                    return None  # the line ends in empty space
            return None

        found: dict[tuple[str, str], tuple[Line, Line]] = {}
        for x1, y1, x2, y2 in segs[:, 0]:
            a, b = node_at(x1, y1, x2, y2), node_at(x2, y2, x1, y1)
            if a is None or b is None or a is b:
                continue
            key = tuple(sorted((a.text, b.text)))
            found.setdefault(key, (a, b))
        claimed: set[int] = set()
        pairs = []
        for a, b in found.values():
            ax, ay, bx, by = a.box.cx, a.box.cy, b.box.cx, b.box.cy
            d = math.hypot(bx - ax, by - ay)
            cands = []
            for k, lab in enumerate(labels):
                t = ((lab.box.cx - ax) * (bx - ax) + (lab.box.cy - ay) * (by - ay)) / (d * d)
                dist = _seg_dist(lab.box.cx, lab.box.cy, ax, ay, bx, by)
                if 0.1 < t < 0.9 and dist < max(3.0 * lab.box.h, 0.3 * d):
                    cands.append((dist, k))
            pairs.append((a, b, sorted(cands)))
        # each weight label belongs to the one edge it sits closest to
        for a, b, cands in sorted(pairs, key=lambda p: p[2][0][0] if p[2] else 1e9):
            k = next((k for _, k in cands if k not in claimed), None)
            if k is not None:
                claimed.add(k)
            self._edges.append((a, b, labels[k].text.strip() if k is not None else ""))
        return self._edges

    def part_for(self, target) -> Region | None:
        """The figure part a target names: "P7", or a description {"ink": "red", "points": "right"}."""
        if isinstance(target, str):
            if target.strip().upper() in self.part_by_id:
                return self.part_by_id[target.strip().upper()]
            # a description copied as text: "red, points right" / "white, round"
            t = target.lower()
            ink = next((w for w in sorted(_HUE_WORDS, key=len, reverse=True) + ["white", "black", "grey", "gray"]
                        if re.search(rf"\b{w}\b", t)), None)
            m = re.search(r"points? ([a-z-]+(?: [a-z-]+)?)", t)
            shape = "round" if "round" in t else "filled" if "filled" in t else None
            if ink is None or len(t.split()) > 6:
                return None  # not a colour description (or a whole sentence)
            return self.find_part(ink, m.group(1).replace(" ", "-") if m else None, shape)
        if isinstance(target, dict) and ("ink" in target or "points" in target or "shape" in target):
            return self.find_part(target.get("ink"), target.get("points"), target.get("shape"))
        return None

    def find_part(self, ink: str | None = None, points: str | None = None, shape: str | None = None,
                  near: Box | None = None) -> Region | None:
        """The part a description means: "the purple arrow pointing down". Matching by what
        the model can see (colour, direction, shape) instead of by reading a tiny id label."""
        want = _DIRS.get(str(points or "").lower().strip().replace(" ", "-").replace("upper", "up").replace("lower", "down"))
        shape = str(shape or "").lower()
        best, best_score = None, 0.0
        for p in self.visible_parts():
            # Colour is what the tutor names most reliably (it mixes up left and right more often):
            # an exact colour outweighs any direction, a near-miss colour counts for little.
            fit = _hue_matches(ink, p.hue) if ink else 1.0
            if ink and fit == 0:
                continue
            score = 3.0 if fit == 1.0 else 0.5
            for word in ("round", "filled"):
                if word in shape:
                    score += 1.5 if word in p.color else -1.0
            if want is not None:
                if p.angle is None:
                    score += 0.2
                else:
                    # "points" means crowded end -> free end, right for arrows drawn out from an object;
                    # an arrow pushing INTO something has its head at the crowded end, so the reverse
                    # also counts, a little less.
                    diff = abs((p.angle - want + 180) % 360 - 180)
                    fwd = max(0.0, 1.5 - diff / 45)
                    back = 0.7 * max(0.0, 1.5 - (180 - diff) / 45)
                    score += max(fwd, back) - (1.0 if diff > 67.5 and back == 0 else 0.0)  # the wrong way is wrong
            score += min(0.5, max(p.box.w, p.box.h) / 600)  # the prominent one, all else equal
            if want is not None:
                score += 0.4 * p.straight  # something that "points" is a stroke, not handwriting
            if near is not None:  # a rough guess of where it is breaks ties
                score += max(0.0, 0.8 - math.hypot(p.box.cx - near.cx, p.box.cy - near.cy) / 300)
            if score > best_score:
                best, best_score = p, score
        if best is None and ink and not _hue_matches(ink, None):
            # no mark has exactly that colour ("blue" for a cyan arrow): take the nearest hue
            centre = next((sum(r[0]) / 2 for k, r in _HUE_WORDS.items() if k in ink.lower()), None)
            coloured = [p for p in self.visible_parts() if p.hue is not None]
            if centre is not None and coloured:
                dist = lambda h: min(abs(h - centre), 180 - abs(h - centre))
                close = [p for p in coloured if dist(p.hue) <= 25]
                if close:
                    def fit(p):
                        s = -dist(p.hue) / 25
                        if want is not None and p.angle is not None:
                            s += max(0.0, 1.5 - abs((p.angle - want + 180) % 360 - 180) / 45)
                        return s + min(0.5, max(p.box.w, p.box.h) / 600)
                    return max(close, key=fit)
        return best

    def _rings(self, ink: np.ndarray, x0: int, y0: int, inv: float, plain: str, hues: np.ndarray,
               coloured: np.ndarray) -> list[Region]:
        """Small drawn circles (a pivot, a wheel, a node) as their own marks, even when they
        touch a rod or a line: a ring whose outline really is inked all round."""
        g = cv2.GaussianBlur(ink.astype(np.uint8) * 255, (5, 5), 0)
        found = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, int(20 * inv)), param1=100,
                                 param2=18, minRadius=max(6, int(14 * inv)), maxRadius=max(8, int(45 * inv)))
        out: list[Region] = []
        if found is None:
            return out
        H, W = ink.shape
        for cx, cy, r in found[0][:12]:
            ts = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            xs = np.clip((cx + r * np.cos(ts)).astype(int), 0, W - 1)
            ys = np.clip((cy + r * np.sin(ts)).astype(int), 0, H - 1)
            near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8))[ys, xs]
            inside = ink[max(0, int(cy - r * 0.5)):int(cy + r * 0.5), max(0, int(cx - r * 0.5)):int(cx + r * 0.5)]
            if near_ink.mean() < 0.85 or inside.size == 0 or inside.mean() > 0.15:
                continue  # not a drawn ring (a letter's loop, a filled blob, noise)
            on = coloured[ys, xs] & (near_ink > 0)
            hue = None
            if on.sum() > len(ts) / 2:
                hs = hues[ys[on], xs[on]].astype(np.float64) * (np.pi / 90)
                hue = float(math.degrees(math.atan2(np.sin(hs).mean(), np.cos(hs).mean())) / 2) % 180
            b = Box(float(x0 + cx - r), float(y0 + cy - r), float(2 * r), float(2 * r)).scaled(self.scale)
            if any(abs(b.cx - o.box.cx) < b.w / 2 and abs(b.cy - o.box.cy) < b.h / 2 for o in out):
                continue
            out.append(Region("", b, _hue_name(hue, plain) + ", round", (b.cx, b.cy), hue, None))
            if len(out) >= 4:
                break
        return out

    def _stroke_parts(self, ink: np.ndarray, x0: int, y0: int, ra: int, inv: float, crowd: np.ndarray,
                      hues: np.ndarray | None) -> list[tuple[Box, tuple[float, float], float | None, float | None, float]]:
        """Connected marks of one colour; a long stroke is split by position into pieces.
        Each comes with its tip: of its two ends, the one with less OTHER ink around it
        (`crowd`: integral image of the other marks, patch pixels). Arrows in a diagram
        meet at a crowded junction; the free end, with the label, is where a teacher
        writes "1"."""
        k = max(3, int(7 * inv))  # joins the letters of a label and the label to its arrow
        mask = cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out: list[tuple[Box, tuple[float, float], float | None, float | None, float]] = []
        H, W = ink.shape
        r = int(30 * inv)

        def busy(px: float, py: float) -> int:
            xa, xb = max(0, int(px) - r), min(W, int(px) + r)
            ya, yb = max(0, int(py) - r), min(H, int(py) + r)
            return int(crowd[yb, xb] - crowd[ya, xb] - crowd[yb, xa] + crowd[ya, xa])

        def ends_of(q: np.ndarray, ox: int, oy: int):
            """(free end, crowded end): the extremes along its longest direction."""
            c = q - q.mean(axis=0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0] if len(q) > 2 else np.array([1.0, 0.0])
            proj = c @ axis
            ends = [q[int(np.argmin(proj))], q[int(np.argmax(proj))]]
            ends.sort(key=lambda e: busy(e[0] + ox - x0, e[1] + oy - y0))
            return ends[0], ends[1]

        def piece(q: np.ndarray, ox: int, oy: int, pad: float, hub=None):
            qx, qy = q[:, 0].min(), q[:, 1].min()
            b = Box(float(ox + qx), float(oy + qy), float(q[:, 0].max() - qx + 1), float(q[:, 1].max() - qy + 1))
            end, base = ends_of(q, ox, oy)
            if hub is not None:
                # a chunk of a bigger shape points away from that shape's crowded end (an arrow
                # cut in pieces still points away from the object it starts at)
                c = q - q.mean(axis=0)
                axis = np.linalg.svd(c, full_matrices=False)[2][0] if len(q) > 2 else np.array([1.0, 0.0])
                proj = c @ axis
                two = [q[int(np.argmin(proj))], q[int(np.argmax(proj))]]
                end = max(two, key=lambda e: (e[0] - hub[0]) ** 2 + (e[1] - hub[1]) ** 2)
                base = hub
            dx, dy = float(end[0] - base[0]), float(end[1] - base[1])
            angle = math.degrees(math.atan2(dy, dx)) % 360 if math.hypot(dx, dy) >= 40 * inv else None
            hue = None
            if hues is not None:  # circular mean: red sits at both ends of the hue circle
                hs = hues[(q[:, 1] + oy - y0).astype(int), (q[:, 0] + ox - x0).astype(int)].astype(np.float64) * (np.pi / 90)
                hue = float(math.degrees(math.atan2(np.sin(hs).mean(), np.cos(hs).mean())) / 2) % 180
            sv = np.linalg.svd(q - q.mean(axis=0), compute_uv=False) if len(q) > 2 else np.array([1.0, 1.0])
            straight = float(max(0.0, 1 - (sv[1] / max(sv[0], 1e-6)) / 0.5))
            out.append((b.scaled(self.scale).pad(pad), ((ox + end[0]) * self.scale, (oy + end[1]) * self.scale), angle, hue,
                        straight))

        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            b = Box(float(x + x0), float(y + y0), float(w), float(h)).scaled(self.scale)
            if max(b.w, b.h) < 18 or min(b.w, b.h) < 10 and max(b.w, b.h) < 45 or w * h > 0.5 * ra:
                continue  # specks and dash fragments, or the frame itself
            ys, xs = np.nonzero((labels[y:y + h, x:x + w] == i) & ink[y:y + h, x:x + w])
            pts = np.column_stack([xs, ys]).astype(np.float32)[:: max(1, len(xs) // 4000)]
            k = int(min(6, round((b.w + b.h) / 120)))
            if k >= 2 and len(pts) > 8:
                sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
                if sv[1] < 0.2 * sv[0]:
                    k = 1  # one straight stroke (an arrow) is one thing: don't cut it up
            if k < 2 or len(pts) < 4 * k:
                if len(pts):
                    piece(pts, x + x0, y + y0, 0)
                continue
            # One long connected stroke (a rod with its pivot and the force arrow) is
            # really several things: split it by position so each piece gets a name.
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _, lab, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            hub = ends_of(pts, x + x0, y + y0)[1]
            for c in range(k):
                q = pts[lab.ravel() == c]
                if len(q):
                    piece(q, x + x0, y + y0, 3, hub)
        return out

    # ---- the model's view -------------------------------------------------

    def set_view(self, view: Box, px_w: int, px_h: int) -> None:
        """The model sees `view` (logical) as an image of px_w x px_h pixels."""
        self.view = view
        self.view_px = (px_w, px_h)

    def to_model(self, b: Box) -> list[int]:
        sx = self.view_px[0] / self.view.w
        sy = self.view_px[1] / self.view.h
        return [round((b.x - self.view.x) * sx), round((b.y - self.view.y) * sy), round(b.w * sx), round(b.h * sy)]

    def from_model(self, xywh) -> Box:
        if not isinstance(xywh, (list, tuple)) or len(xywh) != 4:
            raise ValueError(f"box must be [x, y, w, h], got {xywh!r}")
        x, y, w, h = (float(v) for v in xywh)
        sx = self.view.w / self.view_px[0]
        sy = self.view.h / self.view_px[1]
        return Box(self.view.x + x * sx, self.view.y + y * sy, w * sx, h * sy)

    def visible_lines(self) -> list[Line]:
        v = self.view
        return [l for l in self.lines if l.box.cx >= v.x and l.box.cx <= v.x2 and l.box.cy >= v.y and l.box.cy <= v.y2]

    def visible_regions(self) -> list[Region]:
        v = self.view
        return [r for r in self.regions if r.box.cx >= v.x and r.box.cx <= v.x2 and r.box.cy >= v.y and r.box.cy <= v.y2]

    def visible_parts(self) -> list[Region]:
        v = self.view
        return [r for r in self.parts if r.box.cx >= v.x and r.box.cx <= v.x2 and r.box.cy >= v.y and r.box.cy <= v.y2]

    # ---- resolving what the model points at -------------------------------

    def is_hidden(self, b: Box) -> bool:
        return any(h.x <= b.cx <= h.x2 and h.y <= b.cy <= h.y2 for h in self.hidden)

    def _target_lines(self, target) -> tuple[list[Line], list[Box]]:
        if isinstance(target, list):
            lines, boxes = [], []
            for t in target:
                l2, b2 = self._target_lines(t)
                lines += l2
                boxes += b2
            return lines, boxes
        if isinstance(target, dict) and ("ink" in target or "points" in target or "shape" in target):
            p = self.part_for(target)
            return [], [p.box] if p is not None else []
        if isinstance(target, dict) and "box" in target:
            try:
                b = self.from_model(target["box"])
            except (TypeError, ValueError):
                return [], []  # malformed box from the model: ignore it rather than crash
            return [], [b] if b.w > 0 and b.h > 0 else []
        if isinstance(target, str):
            t = target.strip().upper()
            m = re.fullmatch(r"L(\d+)\s*[-–]\s*L?(\d+)", t)
            if m:
                a, b = sorted((int(m.group(1)), int(m.group(2))))
                lines = [self.line_by_id[f"L{i}"] for i in range(a, b + 1) if f"L{i}" in self.line_by_id]
                if len(lines) > 2:
                    # Ids run row by row across columns; keep the column(s) the range starts and ends in,
                    # so a box around a formula doesn't swallow the sidebar next to it.
                    ends = (lines[0].box, lines[-1].box)
                    lines = [l for l in lines if any(l.box.x < e.x2 and e.x < l.box.x2 for e in ends)]
                return lines, [l.box for l in lines]
            if t in self.line_by_id:
                return [self.line_by_id[t]], [self.line_by_id[t].box]
            if t in self.region_by_id:
                r = self.region_by_id[t].box
                inside = [l for l in self.lines if r.x <= l.box.cx <= r.x2 and r.y <= l.box.cy <= r.y2]
                return inside, [r]
            p = self.part_for(target)  # a part id, or a description copied as text
            if p is not None:
                return [], [p.box]
        return [], []

    def resolve(self, target=None, phrase: str | None = None) -> Box | None:
        lines, boxes = self._target_lines(target)
        if phrase:
            hit = self.find_phrase(phrase, lines) if lines else None
            if hit is None:
                # The model sometimes gets a line id wrong but the words right: search all text.
                # But when it points into a figure, the same words in a far-off title are a
                # different thing (the "why is it pointing at the title" bug): search nearby only.
                pool = self.visible_lines() or self.lines
                text_target = bool(lines) and not (isinstance(target, str) and target.strip().upper()[:1] in "RP")
                if boxes and not text_target:
                    near = boxes[0]
                    for b in boxes[1:]:
                        near = near.union(b)
                    m = max(40.0, 3 * max((l.box.h for l in lines), default=20.0))
                    pool = [l for l in pool if near.x - m <= l.box.cx <= near.x2 + m and near.y - m <= l.box.cy <= near.y2 + m]
                hit = self.find_phrase(phrase, pool)
            if hit is not None:
                return hit
        if not boxes:
            return None
        out = boxes[0]
        for b in boxes[1:]:
            out = out.union(b)
        return out

    def find_phrase(self, phrase: str, lines: list[Line]) -> Box | None:
        want = _norm(phrase)
        if not want:
            return None
        best_score, best = 0.0, None
        for ln in lines:
            norms = [_norm(w.text) for w in ln.words]
            exact_line = want in _norm(ln.text)
            for i in range(len(norms)):
                acc = ""
                for j in range(i, len(norms)):
                    acc += norms[j]
                    if len(acc) > len(want) * 1.6 + 3:
                        break
                    if exact_line and want in acc:
                        score = 1.0 + len(want) / max(len(acc), 1)  # exact, tightest span wins
                    elif abs(len(acc) - len(want)) <= max(3, len(want) // 2):
                        score = difflib.SequenceMatcher(None, acc, want).ratio()
                    else:
                        continue
                    if score > best_score:
                        best_score, best = score, (ln, i, j)
        if best is None or best_score < 0.72:
            return None
        ln, i, j = best
        out = ln.words[i].box
        for w in ln.words[i + 1 : j + 1]:
            out = out.union(w.box)
        return out

    # ---- finding space for notes -------------------------------------------

    def space_cost(self, b: Box) -> float:
        """How much a drawing at b would cover: content, and (much worse) other drawings."""
        y0, y1, x0, x1 = self._cells(b)
        if b.x < 0 or b.y < 0 or b.x2 > self.w or b.y2 > self.h or y1 <= y0 or x1 <= x0:
            return 1e9
        return float((self.occ[y0:y1, x0:x1] + self.reserved[y0:y1, x0:x1] * 40).sum())

    def place(self, w: float, h: float, near: Box | None = None) -> tuple[Box, bool]:
        """Best spot for a w x h drawing: empty, on screen, close to `near`.
        Returns (box, covers_content)."""
        w = min(w, self.w - 2 * CELL)
        h = min(h, self.h - 2 * CELL)
        cw, ch = max(1, math.ceil(w / CELL)), max(1, math.ceil(h / CELL))
        # Covering content is OK-ish (the note gets a card); covering another drawing is not.
        cost = self.occ + self.reserved * 40
        if near is not None:
            cost = cost.copy()
            self._mark(cost, near.pad(4), 50.0)  # never cover what we're explaining
        ii = np.pad(cost, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = ii[ch:, cw:] - ii[:-ch, cw:] - ii[ch:, :-cw] + ii[:-ch, :-cw]

        ys = (np.arange(sums.shape[0]) * CELL)[:, None].astype(np.float32)
        xs = (np.arange(sums.shape[1]) * CELL)[None, :].astype(np.float32)
        score = sums * 50.0
        # keep one cell away from the screen edges
        edge = (ys < CELL) | (xs < CELL) | (ys + h > self.h - CELL) | (xs + w > self.w - CELL)
        score = score + edge * 1e7
        if near is not None:
            dx = np.maximum(0, np.maximum(near.x - (xs + w), xs - near.x2))
            dy = np.maximum(0, np.maximum(near.y - (ys + h), ys - near.y2))
            dist = np.hypot(dx, dy)
            score = score + dist + (dist < 8) * 30 + ((xs + w) <= near.x) * 25
        else:
            cx, cy = self.view.cx, self.view.y + self.view.h * 0.65
            score = score + 0.15 * np.hypot(xs + w / 2 - cx, ys + h / 2 - cy)

        if near is None:
            i, j = np.unravel_index(int(np.argmin(score)), score.shape)
        else:
            i, j = self._best_leader_spot(score, w, h, near)
        box = Box(float(j * CELL), float(i * CELL), w, h)
        covers = float(sums[i, j]) > 0.5
        self.reserve(box.pad(4), 1.0)
        return box, covers

    def _crossings(self, busy: np.ndarray, p: tuple[float, float], q: tuple[float, float]) -> int:
        n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / CELL))
        crossed = 0
        for f in np.linspace(0.1, 0.9, n):
            gx = int((p[0] + (q[0] - p[0]) * f) // CELL)
            gy = int((p[1] + (q[1] - p[1]) * f) // CELL)
            if 0 <= gy < self.gh and 0 <= gx < self.gw and busy[gy, gx]:
                crossed += 1
        return crossed

    def leader(self, note: Box, target: Box, busy: np.ndarray | None = None):
        """Cleanest pointer from a note to its target: tries attaching at the
        target's top, bottom, left and right. Returns (crossings, start, end)."""
        if busy is None:
            busy = (self.occ + self.reserved) > 0
        anchors = [(target.cx, target.y - 4), (target.cx, target.y2 + 4),
                   (target.x - 4, target.cy), (target.x2 + 4, target.cy)]
        best = None
        for qx, qy in anchors:
            px = min(max(qx, note.x), note.x2)
            py = min(max(qy, note.y), note.y2)
            c = self._crossings(busy, (px, py), (qx, qy))
            key = (c, math.hypot(qx - px, qy - py))
            if best is None or key < best[0]:
                best = (key, (px, py), (qx, qy))
        (c, _), p, q = best
        return c, p, q

    def _best_leader_spot(self, score: np.ndarray, w: float, h: float, near: Box) -> tuple[int, int]:
        """Among the best few spots, prefer one whose pointer line to the target
        doesn't cut through text or other drawings."""
        flat = score.ravel()
        k = min(150, flat.size)
        cand = np.argpartition(flat, k - 1)[:k]
        busy = (self.occ + self.reserved) > 0
        best, best_s = None, math.inf
        for idx in cand:
            i, j = divmod(int(idx), score.shape[1])
            crossed, _, _ = self.leader(Box(j * CELL, i * CELL, w, h), near, busy)
            s = float(flat[idx]) + crossed * 80
            if s < best_s:
                best, best_s = (i, j), s
        return best
