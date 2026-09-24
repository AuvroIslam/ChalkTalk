"""Fast tests with no network and no model. Run: python -m pytest tests/test_offline.py -q"""
import http.server
import json
import threading
from functools import partial

import pytest
from PIL import Image

from conftest import make_pdf


# ---- streaming parser ------------------------------------------------------

def test_parser_handles_chunks_fences_nesting_and_bad_lines():
    from planner import ActionStream

    text = ('```json\n{"op":"circle","target":"L3"}\n{"op":"note","target":{"box":[1,2,3,4]},"text":"a {b} c"}\n'
            '{broken line\n{"op":"summary","text":"end"}\n```')
    p = ActionStream()
    out = []
    for i in range(0, len(text), 5):
        out += list(p.feed(text[i:i + 5]))
    assert [a["op"] for a in out] == ["circle", "note", "summary"]
    assert out[1]["target"] == {"box": [1, 2, 3, 4]}


# ---- geometry: ids, phrases, display scaling --------------------------------

def test_reading_order_ids_are_sequential(slide, slide_lines):
    from layout import Scene

    s = Scene(slide, slide_lines, 0.5)
    assert [l.id for l in s.lines] == [f"L{i + 1}" for i in range(len(s.lines))]
    ys = [l.box.cy for l in s.lines]
    assert ys == sorted(ys) or all(abs(a - b) < 20 for a, b in zip(ys, sorted(ys)))


def _line_with(scene, words):
    return next(l for l in scene.lines if words in l.text.lower())


def test_phrase_resolves_to_exact_words(slide, slide_lines):
    from layout import Scene

    s = Scene(slide, slide_lines, 1.0)
    line = _line_with(s, "learning rate")
    b = s.resolve(line.id, "learning rate")
    assert b is not None and b.w < line.box.w * 0.6          # just the phrase, not the line
    assert line.box.x <= b.x and b.x2 <= line.box.x2 + 1


def test_wrong_id_right_words_still_found(slide, slide_lines):
    from layout import Scene

    s = Scene(slide, slide_lines, 1.0)
    right = _line_with(s, "learning rate")
    b = s.resolve("L1", "learning rate")                      # model named the wrong line
    assert b is not None and right.box.y <= b.cy <= right.box.y2


def test_fuzzy_phrase_tolerates_ocr_typos(slide, slide_lines):
    from layout import Scene

    s = Scene(slide, slide_lines, 1.0)
    assert s.resolve(None, "leaming rale") is not None        # typical OCR confusions
    assert s.resolve(None, "quantum chromodynamics") is None   # not on screen -> nothing drawn


def test_targets_ranges_regions_and_bad_ids(slide, slide_lines):
    from layout import Scene

    s = Scene(slide, slide_lines, 1.0)
    rng = s.resolve("L2-L4")
    assert rng.h > s.line_by_id["L2"].box.h * 2
    assert s.resolve(["L2", "L4"]) is not None
    assert s.regions and s.resolve(s.regions[0].id) is not None   # the chart is found as a region
    assert s.resolve("L999") is None and s.resolve(42) is None and s.resolve({"box": "x"}) is None


@pytest.mark.parametrize("scale", [1.0, 1 / 1.5, 0.5, 1 / 3])
def test_display_scaling_maps_back_to_the_same_pixels(slide, slide_lines, scale):
    from layout import Scene

    s = Scene(slide, slide_lines, scale)
    b = s.resolve(_line_with(s, "learning rate").id, "learning rate").scaled(1 / scale)
    ref = Scene(slide, slide_lines, 1.0)
    r = ref.resolve(_line_with(ref, "learning rate").id, "learning rate")
    assert abs(b.x - r.x) < 2 and abs(b.y - r.y) < 2 and abs(b.w - r.w) < 2


def test_model_coordinates_round_trip(slide, slide_lines):
    from layout import Scene
    from ocr import Box

    s = Scene(slide, slide_lines, 0.5)
    s.set_view(Box(100, 50, 600, 400), 1200, 800)
    b = Box(150, 120, 80, 30)
    back = s.from_model(s.to_model(b))
    assert abs(back.x - b.x) < 1 and abs(back.w - b.w) < 1


# ---- composition: placement, overlap, readability ---------------------------

SAMPLE = [
    {"op": "circle", "target": "L3", "phrase": "learning rate", "color": "red"},
    {"op": "note", "target": "L3", "phrase": "learning rate", "text": "Step size. Too big = you overshoot the valley.", "color": "red"},
    {"op": "highlight", "target": "L6"},
    {"op": "note", "target": "L6", "text": "New weight = old weight minus a small step downhill.", "color": "blue"},
    {"op": "arrow", "from": "L6", "to": "R1", "label": "repeat", "color": "blue"},
    {"op": "diagram", "target": "R1", "title": "Loop", "nodes": ["Guess", "Measure", "Step"], "edges": [[0, 1, ""], [1, 2, ""], [2, 0, "again"]]},
    {"op": "note", "text": "A third note with no target, placed anywhere free.", "color": "green"},
    {"op": "summary", "text": "Walk downhill in small steps."},
]


def _compose(img, lines, scale):
    from compose import Composer
    from layout import Scene

    s = Scene(img, lines, scale)
    c = Composer(s)
    items = [it for a in SAMPLE for it in c.build(a)]
    return s, items


def _boxes(items):
    from items import Group, Note

    out = []
    for it in items:
        if isinstance(it, Note):
            out.append((it.box, it.card is not None))
        elif isinstance(it, Group) and it.backdrop is not None:
            out.append((it.backdrop, it.card is not None))
    return out


@pytest.mark.parametrize("scale", [1.0, 0.5, 1 / 3])
def test_notes_stay_on_screen_and_never_overlap_each_other(qapp, slide, slide_lines, scale):
    s, items = _compose(slide, slide_lines, scale)
    boxes = _boxes(items)
    assert len(boxes) == 5  # 3 notes + diagram + summary
    for b, _ in boxes:
        assert b.x >= 0 and b.y >= 0 and b.x2 <= s.w and b.y2 <= s.h
    for i, (a, _) in enumerate(boxes):
        for b, _ in boxes[i + 1:]:
            assert a.x2 <= b.x or b.x2 <= a.x or a.y2 <= b.y or b.y2 <= a.y, (a, b)


def test_notes_on_free_space_do_not_cover_text(qapp, slide, slide_lines):
    s, items = _compose(slide, slide_lines, 0.5)
    for b, carded in _boxes(items):
        if carded:
            continue  # over busy content it deliberately sits on a readable card
        for ln in s.lines:
            l = ln.box
            assert l.x2 <= b.x or b.x2 <= l.x or l.y2 <= b.y or b.y2 <= l.y, (ln.text, b)


@pytest.mark.parametrize("dark", [False, True])
def test_ink_contrasts_with_background(qapp, slide, dark_slide, dark):
    from layout import Scene, contrast, hex_rgb
    from ocr import ocr_lines

    img = dark_slide if dark else slide
    s = Scene(img, ocr_lines(img), 0.5)
    for name in ("red", "blue", "green", "purple", "orange"):
        for ln in s.lines[:4]:
            ink = s.ink(name, ln.box.pad(10))
            assert contrast(hex_rgb(ink), s.bg_color(ln.box.pad(10))) >= 3, (name, dark)


def test_crowded_screen_uses_readable_cards(qapp):
    """A screen full of text: notes must land on sticky-note cards."""
    from PIL import ImageDraw, ImageFont
    from compose import Composer
    from layout import Scene
    from ocr import ocr_lines

    img = Image.new("RGB", (1280, 720), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("segoeui.ttf", 22)
    for y in range(10, 710, 30):
        d.text((10, y), "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor " * 2, font=f, fill="black")
    s = Scene(img, ocr_lines(img), 1.0)
    items = Composer(s).build({"op": "note", "target": "L5", "text": "This note has nowhere empty to go."})
    assert items and items[0].card is not None


def test_garbage_actions_never_crash(qapp, slide, slide_lines):
    from compose import Composer
    from layout import Scene

    c = Composer(Scene(slide, slide_lines, 1.0))
    junk = [{}, {"op": "explode"}, {"op": "circle"}, {"op": "circle", "target": "L999"}, {"op": "note", "text": ""},
            {"op": "arrow", "from": "L1"}, {"op": "diagram", "nodes": "not a list"}, {"op": "diagram", "nodes": []},
            {"op": "diagram", "nodes": ["a", "b"], "edges": [["x", "y"], [0, 9], "bad"]}, {"op": "number", "target": "L1", "n": None},
            {"op": "highlight", "target": {"box": [1, 2]}}, {"op": "note", "target": ["L1", 5, None], "text": "ok"}]
    for a in junk:
        c.build(a)  # must not raise


def test_one_emphasis_per_spot(qapp, slide, slide_lines):
    from compose import Composer
    from layout import Scene

    c = Composer(Scene(slide, slide_lines, 1.0))
    assert c.build({"op": "box", "target": "L6"})
    assert not c.build({"op": "highlight", "target": "L6"})     # same line again: skipped
    assert not c.build({"op": "underline", "target": "L6"})
    assert c.build({"op": "circle", "target": "L3", "phrase": "learning rate"})  # different spot: fine


def test_every_item_paints_at_every_stage(qapp, slide, slide_lines):
    from PySide6.QtGui import QImage, QPainter

    _, items = _compose(slide, slide_lines, 1.0)
    img = QImage(1920, 1080, QImage.Format_ARGB32)
    p = QPainter(img)
    for it in items:
        for t in (0.0, 0.05, 0.4, 0.99, 1.0):
            it.paint(p, t)
    p.end()


# ---- typed tools -------------------------------------------------------------

def test_tool_schemas_are_strict_and_validated():
    from pydantic import ValidationError
    from context import SearchDocument, Toolbox, _name

    class Fake:
        def tools(self):
            return [SearchDocument]

    tb = Toolbox(Fake())
    for spec in tb.specs():
        fn = spec["function"]
        assert fn["strict"] is True
        assert fn["parameters"]["additionalProperties"] is False
        assert set(fn["parameters"]["required"]) == set(fn["parameters"]["properties"])
    args = tb.parse("search_document", json.dumps({"query": "q", "reason": "r"}))
    assert args.query == "q"
    with pytest.raises(ValidationError):
        tb.parse("search_document", '{"query": 3}')
    assert _name(SearchDocument) == "search_document"


# ---- documents ---------------------------------------------------------------

PAGES = ["Chapter one\nPhotosynthesis turns light into sugar", "Chapter two\nThe Krebs cycle releases energy",
         "Chapter three\nMitochondria are the powerhouse of the cell"]


def test_pdf_pages_current_page_and_search(qapp, tmp_path):
    from context import DocumentSource, ReadDocumentPages, SearchDocument

    doc = DocumentSource(str(make_pdf(tmp_path / "bio.pdf", PAGES)))
    assert doc.wait(20) and len(doc.pages) == 3
    assert doc.current_page("Chapter two The Krebs cycle releases energy") == 2
    assert "page 3" in doc.call(SearchDocument(query="mitochondria powerhouse", reason="r")).split("\n")[0]
    assert "Krebs" in doc.call(ReadDocumentPages(start_page=2, end_page=2, reason="r"))
    assert "page 2" in doc.digest("Chapter two The Krebs cycle releases energy")


def test_pptx_slides_include_speaker_notes(tmp_path):
    from pptx import Presentation
    from context import DocumentSource

    prs = Presentation()
    for title, note in [("Supply and demand", "Price rises when demand beats supply"), ("Elasticity", "How much demand reacts")]:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        s.notes_slide.notes_text_frame.text = note
    path = tmp_path / "econ.pptx"
    prs.save(path)
    doc = DocumentSource(str(path))
    assert doc.wait(20) and len(doc.pages) == 2 and doc.unit == "slide"
    assert "Speaker notes: How much demand reacts" in doc.pages[1]


def test_docx_and_txt(tmp_path):
    from docx import Document
    from context import DocumentSource

    d = Document()
    for i in range(400):
        d.add_paragraph(f"Paragraph {i} about thermodynamics and entropy.")
    d.save(tmp_path / "notes.docx")
    doc = DocumentSource(str(tmp_path / "notes.docx"))
    assert doc.wait(20) and len(doc.pages) > 3
    (tmp_path / "t.txt").write_text("hello world", encoding="utf-8")
    txt = DocumentSource(str(tmp_path / "t.txt"))
    assert txt.wait(5) and txt.pages == ["hello world"]


def test_detect_file_url_web_pdf_and_desktop_apps(qapp, tmp_path, monkeypatch):
    import context
    from window import Foreground

    pdf = make_pdf(tmp_path / "lecture 4.pdf", PAGES)
    # PDF opened in the browser from disk
    monkeypatch.setattr(context, "browser_url", lambda h: "file:///" + str(pdf).replace("\\", "/").replace(" ", "%20"))
    src = context.detect(Foreground(1, "lecture 4.pdf - Google Chrome", "chrome.exe"))
    assert isinstance(src, context.DocumentSource) and src.wait(20) and len(src.pages) == 3

    # PDF opened from a web URL -> downloaded
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(context, "browser_url", lambda h: f"http://127.0.0.1:{srv.server_port}/lecture%204.pdf")
    src = context.detect(Foreground(1, "lecture 4.pdf", "msedge.exe"))
    assert isinstance(src, context.DocumentSource) and src.wait(20) and len(src.pages) == 3
    srv.shutdown()

    # a normal page and a YouTube video
    monkeypatch.setattr(context, "browser_url", lambda h: "https://example.com/article")
    assert isinstance(context.detect(Foreground(1, "Article - Google Chrome", "chrome.exe")), context.WebPageSource)

    # PowerPoint title with extensions hidden -> found on disk
    from pptx import Presentation

    docs = tmp_path / "Documents"
    docs.mkdir()
    Presentation().save(docs / "Week 3 Slides.pptx")
    monkeypatch.setattr(context.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "nowhere"))
    src = context.detect(Foreground(1, "Week 3 Slides - PowerPoint", "powerpnt.exe"))
    assert isinstance(src, context.DocumentSource) and src.path.endswith("Week 3 Slides.pptx")
    assert context.detect(Foreground(1, "Untitled - Paint", "mspaint.exe")) is None


# ---- YouTube (no network) ------------------------------------------------------

def _offline_video(monkeypatch):
    import context

    monkeypatch.setattr(context.YouTubeSource, "_load", lambda self: None)
    yt = context.YouTubeSource("abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk&t=1m30s", "Neural nets - YouTube - Google Chrome")
    yt.wait(5)
    yt.segments = [(float(t), f"sentence at {t} seconds" + (" about the bias term" if t == 400 else "")) for t in range(0, 600, 5)]
    return yt


def test_youtube_ids_and_times(monkeypatch):
    import context

    for url in ["https://www.youtube.com/watch?v=aircAruvnKk", "https://youtu.be/aircAruvnKk?t=3",
                "https://www.youtube.com/watch?list=x&v=aircAruvnKk", "https://www.youtube.com/shorts/aircAruvnKk"]:
        assert context._youtube_id(url) == "aircAruvnKk"
    assert context._youtube_id("https://www.youtube.com/") is None
    yt = _offline_video(monkeypatch)
    assert yt.label == "Neural nets" and yt.start_hint == 90
    assert yt.current_time("12:05 / 18:40") == 725          # player controls win over the URL
    assert yt.current_time("no clock visible") == 90


def test_youtube_digest_and_tools(monkeypatch):
    from context import GetVideoTranscript, SearchVideoTranscript

    yt = _offline_video(monkeypatch)
    d = yt.digest("5:00 / 10:00")
    assert "about 5:00" in d and "[2:30]" in d and "[8:" not in d   # only the last 2.5 min before now
    assert yt.call(SearchVideoTranscript(query="bias", reason="r")).startswith("[6:")
    out = yt.call(GetVideoTranscript(start_sec=60, end_sec=90, reason="r"))
    assert out.startswith("[1:00]") and "[2:" not in out


def test_long_page_find_returns_the_relevant_section(monkeypatch):
    import context

    filler = "\n\n".join(f"## Section {i}\n" + "unrelated filler text " * 200 for i in range(30))
    md = "Intro\n\n" + filler + "\n\n## Extensions\nMomentum and Nesterov accelerated gradient speed things up."
    monkeypatch.setitem(context._pages, "https://x.test/p", ("Page", md))
    assert len(md) > context.MAX_TOOL_CHARS * 2
    assert "Nesterov" not in context.read_url("https://x.test/p")               # the start doesn't reach it
    found = context.read_url("https://x.test/p", find="momentum nesterov extensions")
    assert "Nesterov accelerated" in found and "Sections:" in found


def test_address_bar_local_file_becomes_file_url(monkeypatch):
    import window

    class Edit:
        def Exists(self, *a):
            return True

        def GetValuePattern(self):
            class V:
                Value = "D:/AiAgenthon/My Rules.pdf"
            return V()

    class Win:
        def EditControl(self, **k):
            return Edit()

    import uiautomation as auto

    monkeypatch.setattr(auto, "ControlFromHandle", lambda h: Win())
    assert window.browser_url(1) == "file:///D:/AiAgenthon/My%20Rules.pdf"


def test_range_box_stays_in_its_column(qapp):
    from PIL import ImageDraw, ImageFont
    from layout import Scene
    from ocr import ocr_lines

    img = Image.new("RGB", (1600, 900), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("segoeui.ttf", 30)
    for k in range(4):
        d.text((60, 100 + k * 70), f"Main column formula line {k}", font=f, fill="black")
        d.text((1150, 100 + k * 70), f"Sidebar video {k}", font=f, fill="black")
    s = Scene(img, ocr_lines(img), 1.0)
    first = next(l for l in s.lines if "line 0" in l.text)
    last = next(l for l in s.lines if "line 3" in l.text)
    box = s.resolve(f"{first.id}-{last.id}")
    assert box.x2 < 1000, "range box swallowed the sidebar"


def test_hotkey_parsing():
    import hotkey

    assert hotkey.parse("ctrl+shift+space") == (0x4000 | 0x2 | 0x4, 0x20)
    assert hotkey.parse("alt+q")[1] == ord("Q")
    with pytest.raises(ValueError):
        hotkey.parse("ctrl+shift")
