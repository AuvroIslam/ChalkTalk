"""Settings, read from chalktalk/.env (see .env.example) and the project's .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")  # e.g. the Firecrawl key kept at the project root


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


# Azure OpenAI in Microsoft Foundry (used when set); otherwise OPENAI_API_KEY for api.openai.com.
AZURE_ENDPOINT = _get("AZURE_OPENAI_ENDPOINT", "")  # https://<resource>.openai.azure.com/openai/v1/
AZURE_KEY = _get("AZURE_OPENAI_API_KEY", "")
MODEL = _get("CHALK_MODEL", "gpt-5-mini")
# none / minimal / low / medium. Lower = faster first stroke. Empty for non-reasoning models.
# Measured on Foundry (Southeast Asia): gpt-5-mini minimal ~3.5-5 s to first drawing, low ~5-7 s.
REASONING = _get("CHALK_REASONING", "minimal")
# Typed "is there enough context?" check, run in parallel with drawing (1 = on).
CONTEXT_CHECK = _get("CHALK_CONTEXT_CHECK", "1") == "1"
# Measured: minimal passed 10/10 scenarios with a 3.0 s median first drawing; low was slower and no better.
CHECK_REASONING = _get("CHALK_CHECK_REASONING", "minimal")
# Reasoning for the answer written after reading more context (only on that slower path).
AFTER_LOOKUP_REASONING = _get("CHALK_AFTER_LOOKUP_REASONING", REASONING)
# How many rounds of tool calls (reading more context) before it must answer.
MAX_TOOL_ROUNDS = int(_get("CHALK_MAX_TOOL_ROUNDS", "3"))
IMAGE_DETAIL = _get("CHALK_IMAGE_DETAIL", "auto")

# Meaning-based search in pages/slides/transcripts ("variants" finds "Momentum", "Extensions").
# Empty = keyword search only.
EMBED_MODEL = _get("CHALK_EMBED_MODEL", "text-embedding-3-small")

FIRECRAWL_KEY = _get("FIRECRAWL_API_KEY", os.environ.get("fireCrawl_api", ""))
# Scrape the open web page in the background while you type (1 Firecrawl credit per question).
PREFETCH_PAGES = _get("CHALK_PREFETCH_PAGES", "1") == "1"

HOTKEY = _get("CHALK_HOTKEY", "ctrl+shift+space")
# Press, then just talk: captures the screen and listens straight away.
VOICE_HOTKEY = _get("CHALK_VOICE_HOTKEY", "ctrl+g")

# Teaching voice: each step is spoken aloud, and drawing waits for the voice.
VOICE_OUT = _get("CHALK_VOICE_OUT", "1") == "1"
VOICE_NAME = _get("CHALK_VOICE_NAME", "Zira")  # a Windows voice: Zira, David, Mark...
VOICE_RATE = float(_get("CHALK_VOICE_RATE", "1.1"))
# Drawing speed: 1.0 = quick; lower = slower, like a hand writing on a board.
DRAW_SPEED = float(_get("CHALK_DRAW_SPEED", "0.5"))

# Speech to text (Microsoft Foundry deployment)
STT_ENDPOINT = _get("CHALK_STT_ENDPOINT", "")
STT_KEY = _get("CHALK_STT_KEY", "")
STT_MODEL = _get("CHALK_STT_MODEL", "gpt-4o-mini-transcribe")

NOTE_FONT = _get("CHALK_FONT", "Segoe Print")
NOTE_FONT_PX = int(_get("CHALK_FONT_PX", "18"))

# Longest side of the image sent to the model. Smaller = faster upload and fewer tokens.
MODEL_IMAGE_MAX = int(_get("CHALK_IMAGE_MAX", "1568"))
