"""Application configuration and environment validation for ForgeAI.

Configuration is loaded lazily from environment variables and an optional `.env`
file. Importing this module never triggers Gemini API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_KEYS = {
    "",
    "your_gemini_api_key_here",
    "changeme",
    "replace_me",
}


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


class ForgeSettings(BaseSettings):
    """Runtime settings for the ForgeAI agent factory."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str | None = Field(default=None, alias="GEMINI_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chroma_persist_dir: Path = Field(default=Path(".forgeai/chroma"), alias="CHROMA_PERSIST_DIR")
    artifacts_dir: Path = Field(default=Path("artifacts/runs"), alias="ARTIFACTS_DIR")

    # Assignment guardrails & approved defaults
    max_clarifying_questions: int = Field(default=5, ge=1, le=5)
    max_optimization_iterations: int = Field(
        default=3, ge=0, le=3, alias="MAX_OPTIMIZATION_ITERATIONS"
    )
    target_pass_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    min_eval_cases: int = Field(default=20, ge=1)
    judge_batch_size: int = Field(default=5, ge=1, le=20, alias="JUDGE_BATCH_SIZE")
    csv_rows_per_unit: int = Field(default=1, ge=1, le=500, alias="CSV_ROWS_PER_UNIT")
    default_chunk_size: int = Field(default=800, ge=100, le=4000)
    default_chunk_overlap: int = Field(default=120, ge=0, le=1000)

    def gemini_readiness(self) -> tuple[bool, list[str]]:
        """Check whether Gemini credentials and model configuration are usable."""
        issues: list[str] = []
        key = (self.gemini_api_key or "").strip()
        if key.lower() in _PLACEHOLDER_KEYS:
            issues.append(
                "GEMINI_API_KEY is missing or still set to a placeholder. "
                "Set a valid key in your environment or .env file."
            )
        model = (self.gemini_model or "").strip()
        if not model:
            issues.append(
                "GEMINI_MODEL is missing. Set GEMINI_MODEL (for example, gemini-3.8-flash) "
                "in your environment or .env file."
            )
        return (len(issues) == 0, issues)

    def require_gemini_config(self) -> tuple[str, str]:
        """Return validated `(api_key, model_name)` or raise `ConfigurationError`."""
        ready, issues = self.gemini_readiness()
        if not ready:
            raise ConfigurationError(" ".join(issues))
        assert self.gemini_api_key is not None
        assert self.gemini_model is not None
        return self.gemini_api_key.strip(), self.gemini_model.strip()

    def status_summary(self) -> dict[str, str | bool]:
        """Return a safe, non-secret summary suitable for UI display and diagnostics."""
        ready, _ = self.gemini_readiness()
        return {
            "gemini_configured": ready,
            "gemini_model": (self.gemini_model or "").strip() or "(unset)",
            "embedding_model": self.embedding_model,
            "chroma_persist_dir": str(self.chroma_persist_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "api_key_present": bool((self.gemini_api_key or "").strip())
            and (self.gemini_api_key or "").strip().lower() not in _PLACEHOLDER_KEYS,
        }


def _read_hosting_secret(key: str) -> str | None:
    """Safely read a secret from Streamlit hosting secrets when available."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None and key in secrets:
            value = str(secrets[key]).strip()
            return value or None
    except Exception:
        return None
    return None


def load_settings(env_file: str | Path | None = ".env") -> ForgeSettings:
    """Load settings from environment variables, optional `.env`, and hosting secrets."""
    settings = ForgeSettings(_env_file=None) if env_file is None else ForgeSettings(_env_file=env_file)
    updates: dict[str, str] = {}
    if not (settings.gemini_api_key or "").strip():
        secret_key = _read_hosting_secret("GEMINI_API_KEY")
        if secret_key:
            updates["gemini_api_key"] = secret_key
    if not (settings.gemini_model or "").strip():
        secret_model = _read_hosting_secret("GEMINI_MODEL")
        if secret_model:
            updates["gemini_model"] = secret_model
    if updates:
        return settings.model_copy(update=updates)
    return settings


SupportedDocExtension = Literal[".pdf", ".csv", ".txt", ".md"]

