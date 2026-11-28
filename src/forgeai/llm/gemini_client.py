"""Thin wrapper around the official Google GenAI SDK (`google-genai`).

Client construction is deferred until an LLM call is explicitly made so that
importing modules or starting the UI without credentials never fails or incurs
network requests.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from forgeai.config import ForgeSettings, load_settings
from forgeai.llm.telemetry import (
    ApiTelemetryTracker,
    extract_token_usage,
    get_current_stage,
    sanitize_error_summary,
)

TModel = TypeVar("TModel", bound=BaseModel)
_RETRY_DELAY_RE = re.compile(r"(?:Please retry in\s+|'retryDelay':\s*'?)(\d+(?:\.\d+)?)s?")


class StructuredSchemaValidationError(ValueError):
    """Raised when structured LLM JSON fails Pydantic validation, preserving raw JSON for bounded repair."""

    def __init__(
        self,
        *,
        schema_name: str,
        raw_text: str,
        validation_error: ValidationError,
    ) -> None:
        self.schema_name = schema_name
        self.raw_text = raw_text
        self.validation_error = validation_error
        super().__init__(f"{validation_error}")


def _classify_retry(
    exc: Exception,
    *,
    attempt: int,
    base_backoff: float,
) -> tuple[str, float]:
    """Classify an exception as `'none'`, `'rate_limit'`, or `'transient'` with wait seconds."""
    message = str(exc)
    status_code = getattr(exc, "code", None)
    lower_msg = message.lower()

    is_rate_limit = status_code == 429 or "429" in message or "resource_exhausted" in lower_msg
    if is_rate_limit:
        is_daily_quota = (
            "perday" in lower_msg
            or "per day" in lower_msg
            or "requests per day" in lower_msg
        )
        match = _RETRY_DELAY_RE.search(message)
        parsed = float(match.group(1)) if match else 60.0
        if is_daily_quota or parsed > 120.0:
            return "none", 0.0
        delay = min(max(parsed + 2.0, 5.0), 125.0)
        return "rate_limit", delay

    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "none", 0.0

    is_transient_server = (
        isinstance(exc, TimeoutError)
        or status_code in {503, 504}
        or "503" in message
        or "unavailable" in lower_msg
        or "504" in message
        or "deadline_exceeded" in lower_msg
        or "timed out" in lower_msg
        or "timeout" in lower_msg
    )
    if is_transient_server:
        delay = min(base_backoff * (2**attempt), 30.0)
        return "transient", delay

    return "none", 0.0


class GeminiClient:
    """Lazy, configurable Gemini API client backed by `google-genai`."""

    def __init__(
        self,
        settings: ForgeSettings | None = None,
        *,
        max_retries: int | None = None,
        max_transient_retries: int = 2,
        max_rate_limit_retries: int = 3,
        base_backoff_seconds: float = 2.0,
        request_timeout_ms: int = 60000,
        sleep_fn: Callable[[float], None] | None = None,
        telemetry: ApiTelemetryTracker | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._client = None
        self.max_transient_retries = max(0, max_transient_retries)
        self.max_rate_limit_retries = max(0, max_rate_limit_retries)
        if max_retries is not None:
            cap = max(0, max_retries)
            self.max_transient_retries = min(self.max_transient_retries, cap)
            self.max_rate_limit_retries = min(self.max_rate_limit_retries, cap)
            self.max_retries = cap
        else:
            self.max_retries = self.max_transient_retries + self.max_rate_limit_retries
        self.base_backoff_seconds = max(0.0, base_backoff_seconds)
        self.request_timeout_ms = max(1000, request_timeout_ms)
        self._sleep = sleep_fn or time.sleep
        self.telemetry = telemetry or ApiTelemetryTracker(artifacts_dir=self._settings.artifacts_dir)

    @property
    def model_name(self) -> str:
        _, model = self._settings.require_gemini_config()
        return model

    def _get_client(self):
        api_key, _ = self._settings.require_gemini_config()
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=self.request_timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def _generate_content_with_retry(self, prompt: str, config):
        """Invoke `generate_content` with bounded retry protection and stage telemetry."""
        client = self._get_client()
        model = self.model_name
        stage = get_current_stage()
        transient_retries = 0
        rate_limit_retries = 0
        total_retries = 0
        attempts = 0

        while True:
            attempts += 1
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                prompt_tok, comp_tok, total_tok, tok_avail = extract_token_usage(response)
                self.telemetry.record(
                    stage=stage,
                    model=model,
                    attempts=attempts,
                    retries=total_retries,
                    success=True,
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    total_tokens=total_tok,
                    tokens_available=tok_avail,
                )
                return response
            except Exception as exc:
                kind, delay = _classify_retry(
                    exc,
                    attempt=transient_retries if "503" in str(exc) or "504" in str(exc) or isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() or "unavailable" in str(exc).lower() else total_retries,
                    base_backoff=self.base_backoff_seconds,
                )
                can_retry = False
                if kind == "transient" and transient_retries < self.max_transient_retries and total_retries < self.max_retries:
                    transient_retries += 1
                    can_retry = True
                elif kind == "rate_limit" and rate_limit_retries < self.max_rate_limit_retries and total_retries < self.max_retries:
                    rate_limit_retries += 1
                    can_retry = True

                if not can_retry:
                    self.telemetry.record(
                        stage=stage,
                        model=model,
                        attempts=attempts,
                        retries=total_retries,
                        success=False,
                        error=sanitize_error_summary(exc),
                        tokens_available=False,
                    )
                    raise

                total_retries += 1
                self._sleep(delay)

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a plain-text response using the configured Gemini model."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        response = self._generate_content_with_retry(prompt, config)
        return (response.text or "").strip()

    def generate_structured(
        self,
        prompt: str,
        response_schema: type[TModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.1,
    ) -> TModel:
        """Generate a JSON response validated against a Pydantic schema."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        response = self._generate_content_with_retry(prompt, config)
        if not response.text:
            raise ValueError(f"Gemini returned an empty response for schema {response_schema.__name__}.")
        try:
            return response_schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise StructuredSchemaValidationError(
                schema_name=response_schema.__name__,
                raw_text=response.text,
                validation_error=exc,
            ) from exc
