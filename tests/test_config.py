"""Tests for ForgeAI configuration, lazy Gemini initialization, and error messages."""

from __future__ import annotations

import pytest

from forgeai.config import ConfigurationError, ForgeSettings, load_settings
from forgeai.llm.gemini_client import GeminiClient


def test_missing_gemini_config_reports_actionable_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    settings = load_settings(env_file=None)
    ready, issues = settings.gemini_readiness()

    assert ready is False
    assert len(issues) == 2
    assert "GEMINI_API_KEY" in issues[0]
    assert "GEMINI_MODEL" in issues[1]

    with pytest.raises(ConfigurationError) as exc_info:
        settings.require_gemini_config()
    assert "GEMINI_API_KEY" in str(exc_info.value)
    assert "GEMINI_MODEL" in str(exc_info.value)


def test_placeholder_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    settings = load_settings(env_file=None)
    ready, issues = settings.gemini_readiness()

    assert ready is False
    assert len(issues) == 1
    assert "placeholder" in issues[0].lower()


def test_valid_env_config_and_lazy_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key-value")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    settings = ForgeSettings(_env_file=None)
    key, model = settings.require_gemini_config()

    assert key == "test-only-key-value"
    assert model == "gemini-2.5-flash"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"

    # Constructing GeminiClient must not make network calls or instantiate SDK eagerly.
    client = GeminiClient(settings=settings)
    assert client._client is None
    assert client.model_name == "gemini-2.5-flash"
    assert client._client is None

    summary = settings.status_summary()
    assert summary["gemini_configured"] is True
    assert summary["api_key_present"] is True
    assert "test-only-key-value" not in str(summary)

