import os
import sys
from pathlib import Path

import pytest

# Unit tests stay offline: no embedding calls (keyword search) unless asked for.
if os.environ.get("CHALK_TEST_EMBED") != "1":
    os.environ["CHALK_EMBED_MODEL"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session")
def slide():
    from render_test import sample_slide

    return sample_slide()


@pytest.fixture(scope="session")
def dark_slide():
    from render_test import sample_slide

    return sample_slide(dark=True)


@pytest.fixture(scope="session")
def slide_lines(slide):
    from ocr import ocr_lines

    return ocr_lines(slide)


def make_pdf(path: Path, pages: list[str]) -> Path:
    """A real text PDF (selectable text), drawn with Qt."""
    from PySide6.QtCore import QMarginsF
    from PySide6.QtGui import QFont, QPageSize, QPainter, QPdfWriter

    w = QPdfWriter(str(path))
    w.setPageSize(QPageSize(QPageSize.A4))
    w.setPageMargins(QMarginsF(20, 20, 20, 20))
    p = QPainter(w)
    p.setFont(QFont("Segoe UI", 12))
    for i, text in enumerate(pages):
        if i:
            w.newPage()
        for k, line in enumerate(text.splitlines()):
            p.drawText(200, 400 + k * 300, line)
    p.end()
    return path
