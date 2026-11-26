"""Thin wrapper around the official Google GenAI SDK (`google-genai`).

Client construction is deferred until an LLM call is explicitly made so that
importing modules or starting the UI without credentials never fails or incurs
network requests.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from forgeai.config import ForgeSettings, load_settings

TModel = TypeVar("TModel", bound=BaseModel)


class GeminiClient:
    """Lazy, configurable Gemini API client backed by `google-genai`."""

    def __init__(self, settings: ForgeSettings | None = None) -> None:
        self._settings = settings or load_settings()
        self._client = None

    @property
    def model_name(self) -> str:
        _, model = self._settings.require_gemini_config()
        return model

    def _get_client(self):
        api_key, _ = self._settings.require_gemini_config()
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a plain-text response using the configured Gemini model."""
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
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

        client = self._get_client()
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        if not response.text:
            raise ValueError(f"Gemini returned an empty response for schema {response_schema.__name__}.")
        return response_schema.model_validate_json(response.text)

