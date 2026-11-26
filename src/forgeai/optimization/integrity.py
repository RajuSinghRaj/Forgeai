"""Immutable evaluation suite integrity verification helpers."""

from __future__ import annotations

import hashlib
import json

from forgeai.schemas.evaluation import EvalCase


class EvalIntegrityError(ValueError):
    """Raised if evaluation cases were mutated, dropped, or tampered with across iterations."""


def compute_eval_suite_hash(eval_cases: list[EvalCase]) -> str:
    """Compute a stable SHA-256 fingerprint over the canonical evaluation suite."""
    canonical = json.dumps(
        [case.model_dump(mode="json") for case in eval_cases],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_eval_suite_integrity(eval_cases: list[EvalCase], expected_hash: str) -> None:
    """Verify the current evaluation suite matches the baseline SHA-256 fingerprint."""
    actual = compute_eval_suite_hash(eval_cases)
    if actual != expected_hash:
        raise EvalIntegrityError(
            "Evaluation suite integrity check failed: expected hash "
            f"{expected_hash[:12]}..., got {actual[:12]}... "
            "Evaluation cases must remain immutable during optimization."
        )


def snapshot_eval_suite(eval_cases: list[EvalCase]) -> tuple[list[EvalCase], str]:
    """Return a deep copy of `eval_cases` paired with its SHA-256 fingerprint."""
    frozen = [case.model_copy(deep=True) for case in eval_cases]
    return frozen, compute_eval_suite_hash(frozen)

