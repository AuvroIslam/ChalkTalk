"""Offline check of placement and readability, no API key or overlay needed.

    python render_test.py [image.png] [--scale 0.5] [--actions actions.jsonl] [--out out.png]

Runs OCR + layout + composer on an image and saves the finished drawing on
top of it. With no image, a sample lecture slide is generated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from compose import Composer
from layout import Scene
from ocr import ocr_lines

SAMPLE_ACTIONS = """
{"op":"circle","target":"L3","phrase":"learning rate","color":"red","say":"This is the knob that matters most."}
{"op":"note","target":"L3","phrase":"learning rate","text":"Step size. Too big = you jump over the valley; too small = forever.","color":"red"}
{"op":"highlight","target":"L6"}
{"op":"note","target":"L6","text":"New weight = old weight minus a small step downhill.","color":"blue"}
{"op":"arrow","from":"L6","to":"R1","label":"repeat until flat","color":"blue"}
{"op":"diagram","target":"R1","title":"One training loop","nodes":["Guess w","Measure error","Step downhill"],"edges":[[0,1,""],[1,2,""],[2,0,"repeat"]],"color":"purple"}
{"op":"summary","text":"Walk downhill in small steps until the error stops shrinking."}
"""


def sample_slide(dark: bool = False) -> Image.Image:
    bg, fg, acc = ((24, 28, 38), (235, 238, 245), (120, 170, 255)) if dark else ((255, 255, 255), (30, 30, 40), (40, 90, 200))
    img = Image.new("RGB", (1920, 1080), bg)
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype("segoeuib.ttf", 64)
    f = ImageFont.truetype("segoeui.ttf", 36)
    f_math = ImageFont.truetype("cambria.ttc", 44)
    d.rectangle((0, 0, 1920, 14), fill=acc)
    d.text((110, 70), "Gradient Descent", font=f_title, fill=fg)
    d.text((110, 200), "Goal: minimise the loss function L(w)", font=f, fill=fg)
    d.text((110, 270), "Update rule uses a learning rate η > 0", font=f, fill=fg)
    d.text((110, 340), "Repeat until convergence:", font=f, fill=fg)
    d.text((170, 430), "w ← w − η ∇L(w)", font=f_math, fill=fg)
    d.text((110, 540), "Small η: slow. Large η: may diverge.", font=f, fill=fg)
    # a loss-curve figure on the right
    x0, y0, x1, y1 = 1180, 230, 1780, 760
    d.rectangle((x0, y0, x1, y1), outline=fg, width=3)
    pts = [(x0 + 20 + i * 5.6, y0 + 40 + ((i - 50) / 50) ** 2 * 420) for i in range(101)]
    d.line(pts, fill=acc, width=5)
    for k, i in enumerate((8, 22, 34, 44, 50)):
        px, py = pts[i]
        d.ellipse((px - 9, py - 9, px + 9, py + 9), fill=(220, 60, 60))
    d.text((x0 + 230, y1 + 12), "weight w", font=ImageFont.truetype("segoeui.ttf", 26), fill=fg)
    d.text((110, 960), "Lecture 4 • Optimisation", font=ImageFont.truetype("segoeui.ttf", 24), fill=(130, 130, 140))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--scale", type=float, default=1.0, help="logical px per physical px (0.5 = 200%% display)")
    ap.add_argument("--actions")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--out", default="render_out.png")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    img = Image.open(args.image).convert("RGB") if args.image else sample_slide(args.dark)
    t = time.perf_counter()
    lines = ocr_lines(img)
    t_ocr = time.perf_counter()
    scene = Scene(img, lines, args.scale)
    t_scene = time.perf_counter()
    print(f"OCR {1000 * (t_ocr - t):.0f} ms, scene {1000 * (t_scene - t_ocr):.0f} ms")
    for ln in scene.lines:
        print(" ", ln.id, [round(v) for v in (ln.box.x, ln.box.y, ln.box.w, ln.box.h)], ln.text)
    for r in scene.regions:
        print(" ", r.id, [round(v) for v in (r.box.x, r.box.y, r.box.w, r.box.h)])

    text = open(args.actions, encoding="utf-8").read() if args.actions else SAMPLE_ACTIONS
    comp = Composer(scene)
    items = []
    for raw in text.strip().splitlines():
        if raw.strip():
            items += comp.build(json.loads(raw))
    print(f"compose {1000 * (time.perf_counter() - t_scene):.0f} ms, {len(items)} items")

    dpr = 1 / args.scale
    out = QImage(img.width, img.height, QImage.Format_ARGB32)
    out.setDevicePixelRatio(dpr)
    qimg = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.drawImage(QRectF(0, 0, scene.w, scene.h), qimg)
    for it in items:
        it.paint(p, 1.0)
    p.end()
    out.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
