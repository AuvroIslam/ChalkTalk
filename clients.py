"""The one shared model client (Microsoft Foundry / Azure OpenAI, or OpenAI)."""
from __future__ import annotations

import threading

import openai

import config

_client: openai.OpenAI | None = None
_lock = threading.Lock()


def make_client() -> openai.OpenAI:
    """One shared client, so the TLS connection stays open between requests."""
    global _client
    with _lock:
        if _client is None:
            if config.AZURE_ENDPOINT and config.AZURE_KEY:  # Microsoft Foundry deployment
                _client = openai.OpenAI(base_url=config.AZURE_ENDPOINT, api_key=config.AZURE_KEY,
                                        timeout=60.0, max_retries=1)
            else:
                _client = openai.OpenAI(timeout=60.0, max_retries=1)  # OPENAI_API_KEY
        return _client
