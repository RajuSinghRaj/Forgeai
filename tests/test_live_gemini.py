"""Explicit live Gemini API connectivity test.

Skipped during ordinary `pytest` runs so tests never depend on live API
availability or consume tokens unless `FORGEAI_RUN_LIVE_TESTS=1` is set.
"""

from __future__ import annotations

import os

import pytest

from forgeai.config import load_settings
from forgeai.llm.gemini_client import GeminiClient


@pytest.mark.live_gemini
@pytest.mark.skipif(
    os.getenv("FORGEAI_RUN_LIVE_TESTS") != "1",
    reason="Set FORGEAI_RUN_LIVE_TESTS=1 to run live Gemini connectivity tests.",
)
def test_live_gemini_connectivity() -> None:
    settings = load_settings()
    settings.require_gemini_config()
    client = GeminiClient(settings)
    reply = client.generate_text("Reply with exactly: OK", temperature=0.0)
    assert "OK" in reply.upper()

