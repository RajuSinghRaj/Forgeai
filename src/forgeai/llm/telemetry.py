"""Zero-cost API telemetry tracking requests, retries, errors, and token usage per stage."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ACTIVE_STAGE: ContextVar[str] = ContextVar("forgeai_llm_stage", default="unknown")
_ACTIVE_CONTEXT: ContextVar[str] = ContextVar("forgeai_llm_context", default="")
_API_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{20,}")


@contextmanager
def telemetry_stage(stage: str) -> Iterator[None]:
    """Bind the active ForgeAI pipeline stage name for nested Gemini client calls."""
    token = _ACTIVE_STAGE.set(stage)
    try:
        yield
    finally:
        _ACTIVE_STAGE.reset(token)


@contextmanager
def telemetry_context(context: str) -> Iterator[None]:
    """Bind an orthogonal execution context tag (e.g., optimization_candidate_evaluation)."""
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_CONTEXT.reset(token)


def get_current_stage() -> str:
    """Return the active pipeline stage label."""
    return _ACTIVE_STAGE.get()


def get_current_context() -> str:
    """Return the active orthogonal telemetry context tag, if any."""
    return _ACTIVE_CONTEXT.get()


def sanitize_error_summary(exc: Exception) -> str:
    """Return a concise, secret-free summary of an LLM API failure."""
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    raw = f"{type(exc).__name__}: {exc}"
    cleaned = _API_KEY_RE.sub("[REDACTED_KEY]", " ".join(raw.split()))
    prefix_parts = []
    if code is not None:
        prefix_parts.append(str(code))
    if status:
        prefix_parts.append(str(status))
    prefix = " ".join(prefix_parts)
    if prefix and prefix not in cleaned[:30]:
        cleaned = f"{prefix} | {cleaned}"
    return cleaned[:180]


def extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None, bool]:
    """Safely extract prompt, completion, and total token counts when present."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None, None, False

    def _read_int(field: str) -> int | None:
        val = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        return int(val) if isinstance(val, int) else None

    prompt_tokens = _read_int("prompt_token_count")
    completion_tokens = _read_int("candidates_token_count")
    total_tokens = _read_int("total_token_count")
    available = any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens))
    return prompt_tokens, completion_tokens, total_tokens, available


class ApiTelemetryTracker:
    """Thread-safe append-only JSONL recorder for Gemini API calls."""

    def __init__(self, artifacts_dir: Path | str = Path("artifacts/runs")) -> None:
        self.telemetry_dir = Path(artifacts_dir) / "telemetry"
        self.telemetry_file = self.telemetry_dir / "gemini_requests.jsonl"
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_attempts = 0
        self._total_retries = 0
        self._total_errors = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._optimization_candidate_requests = 0
        self._agent_generation_requests_saved = 0
        self._optimization_early_stops = 0
        self._early_stopped_cases = 0
        self._requests_saved_early_stopping = 0
        self._by_stage: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "requests": 0,
                "attempts": 0,
                "retries": 0,
                "errors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )

    def record(
        self,
        *,
        stage: str | None = None,
        context: str | None = None,
        model: str,
        attempts: int,
        retries: int,
        success: bool,
        error: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        tokens_available: bool = False,
    ) -> dict[str, Any]:
        """Record one logical Gemini request to memory and append-only JSONL."""
        stage_name = stage or get_current_stage()
        context_name = context if context is not None else get_current_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "request",
            "stage": stage_name,
            "context": context_name,
            "model": model,
            "request_count": 1,
            "attempts": attempts,
            "retries": retries,
            "success": success,
            "error": error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_available": tokens_available,
        }

        line = json.dumps(payload) + "\n"
        with self._lock:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            with self.telemetry_file.open("a", encoding="utf-8") as handle:
                handle.write(line)

            self._total_requests += 1
            self._total_attempts += attempts
            self._total_retries += retries
            if context_name == "optimization_candidate_evaluation":
                self._optimization_candidate_requests += 1
            if not success:
                self._total_errors += 1
            if prompt_tokens is not None:
                self._prompt_tokens += prompt_tokens
            if completion_tokens is not None:
                self._completion_tokens += completion_tokens
            if total_tokens is not None:
                self._total_tokens += total_tokens

            bucket = self._by_stage[stage_name]
            bucket["requests"] += 1
            bucket["attempts"] += attempts
            bucket["retries"] += retries
            if not success:
                bucket["errors"] += 1
            if prompt_tokens is not None:
                bucket["prompt_tokens"] += prompt_tokens
            if completion_tokens is not None:
                bucket["completion_tokens"] += completion_tokens
            if total_tokens is not None:
                bucket["total_tokens"] += total_tokens

        return payload

    def record_efficiency(
        self,
        *,
        stage: str,
        requests_saved: int = 0,
        early_stopped_cases: int = 0,
        early_stops: int = 0,
    ) -> dict[str, Any]:
        """Record request-savings and early-stopping telemetry without keys or secrets."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "efficiency",
            "stage": stage,
            "request_count": 0,
            "requests_saved": max(0, int(requests_saved)),
            "early_stopped_cases": max(0, int(early_stopped_cases)),
            "early_stops": max(0, int(early_stops)),
        }
        line = json.dumps(payload) + "\n"
        with self._lock:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            with self.telemetry_file.open("a", encoding="utf-8") as handle:
                handle.write(line)

            if stage == "agent_generation":
                self._agent_generation_requests_saved += payload["requests_saved"]
            elif stage == "optimization_early_stopping":
                self._requests_saved_early_stopping += payload["requests_saved"]
                self._early_stopped_cases += payload["early_stopped_cases"]
                self._optimization_early_stops += payload["early_stops"]

        return payload

    def summary(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of accumulated telemetry counters."""
        with self._lock:
            agent_gen_reqs = self._by_stage.get("agent_generation", {}).get("requests", 0)
            total_saved = self._agent_generation_requests_saved + self._requests_saved_early_stopping
            return {
                "telemetry_file": str(self.telemetry_file),
                "total_requests": self._total_requests,
                "total_attempts": self._total_attempts,
                "total_retries": self._total_retries,
                "total_errors": self._total_errors,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "agent_generation_requests": agent_gen_reqs,
                "agent_generation_requests_saved": self._agent_generation_requests_saved,
                "optimization_candidate_requests": self._optimization_candidate_requests,
                "optimization_early_stops": self._optimization_early_stops,
                "early_stopped_cases": self._early_stopped_cases,
                "requests_saved_early_stopping": self._requests_saved_early_stopping,
                "total_requests_saved": total_saved,
                "by_stage": {k: dict(v) for k, v in self._by_stage.items()},
            }


def read_telemetry_summary(artifacts_dir: Path | str = Path("artifacts/runs")) -> dict[str, Any]:
    """Read aggregated telemetry counters from the append-only JSONL log without API calls."""
    telemetry_file = Path(artifacts_dir) / "telemetry" / "gemini_requests.jsonl"
    totals: dict[str, Any] = {
        "telemetry_file": str(telemetry_file),
        "total_requests": 0,
        "total_retries": 0,
        "total_errors": 0,
        "total_tokens": 0,
        "agent_generation_requests": 0,
        "agent_generation_requests_saved": 0,
        "optimization_candidate_requests": 0,
        "optimization_early_stops": 0,
        "early_stopped_cases": 0,
        "requests_saved_early_stopping": 0,
        "total_requests_saved": 0,
        "by_stage": {},
    }
    if not telemetry_file.exists():
        return totals

    by_stage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "retries": 0, "errors": 0, "total_tokens": 0}
    )
    for line in telemetry_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        stage = str(item.get("stage") or "unknown")
        if item.get("event_type") == "efficiency":
            saved = int(item.get("requests_saved") or 0)
            stopped_cases = int(item.get("early_stopped_cases") or 0)
            early_stops = int(item.get("early_stops") or 0)
            if stage == "agent_generation":
                totals["agent_generation_requests_saved"] += saved
                totals["total_requests_saved"] += saved
            elif stage == "optimization_early_stopping":
                totals["requests_saved_early_stopping"] += saved
                totals["early_stopped_cases"] += stopped_cases
                totals["optimization_early_stops"] += early_stops
                totals["total_requests_saved"] += saved
            continue

        req = int(item.get("request_count") or 0)
        ret = int(item.get("retries") or 0)
        err = 0 if item.get("success", True) else 1
        tok = int(item.get("total_tokens") or 0)
        totals["total_requests"] += req
        totals["total_retries"] += ret
        totals["total_errors"] += err
        totals["total_tokens"] += tok
        if stage == "agent_generation":
            totals["agent_generation_requests"] += req
        if item.get("context") == "optimization_candidate_evaluation":
            totals["optimization_candidate_requests"] += req
        bucket = by_stage[stage]
        bucket["requests"] += req
        bucket["retries"] += ret
        bucket["errors"] += err
        bucket["total_tokens"] += tok

    totals["by_stage"] = {k: dict(v) for k, v in by_stage.items()}
    return totals
