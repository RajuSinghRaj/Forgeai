"""Duplicate-execution locking and completed-evaluation guards for ForgeAI."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from typing import Any

from forgeai.schemas.agent import DeclarativeAgentDefinition

_GLOBAL_EXECUTION_LOCK = threading.Lock()


@contextmanager
def execution_lock_guard(lock: threading.Lock | None = None) -> Iterator[bool]:
    """Try to acquire an in-flight execution lock without blocking; release safely on exit."""
    target = lock or _GLOBAL_EXECUTION_LOCK
    acquired = target.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            target.release()


def compute_evaluation_signature(
    agent: DeclarativeAgentDefinition | None,
    eval_suite_sha256: str | None,
) -> tuple[str, int, str] | None:
    """Build a canonical fingerprint identifying an evaluated `(agent, suite)` pair."""
    if agent is None or not eval_suite_sha256:
        return None
    return (agent.agent_id, agent.version, eval_suite_sha256)


def is_evaluation_already_completed(
    state: Mapping[str, Any] | None,
    agent: DeclarativeAgentDefinition | None = None,
) -> bool:
    """Return True when `state` already contains completed evaluations for `agent` and suite hash."""
    if not state:
        return False
    target_agent = agent or state.get("agent_definition")
    suite_hash = state.get("eval_suite_sha256")
    eval_results = state.get("eval_results") or []
    scorecards = state.get("scorecards") or []
    if target_agent is None or not suite_hash or not eval_results or not scorecards:
        return False

    current_sig = compute_evaluation_signature(target_agent, suite_hash)
    recorded_sig = state.get("evaluated_signature")
    if recorded_sig is not None:
        return tuple(recorded_sig) == current_sig

    baseline_agent = state.get("baseline_agent")
    if baseline_agent is not None:
        return (
            baseline_agent.agent_id == target_agent.agent_id
            and baseline_agent.version == target_agent.version
        )
    return False


def is_optimization_already_completed(
    state: Mapping[str, Any] | None,
    agent: DeclarativeAgentDefinition | None = None,
) -> bool:
    """Return True when optimization has already finished for the current agent/suite."""
    if not state or state.get("final_report") is None:
        return False
    target_agent = agent or state.get("agent_definition")
    suite_hash = state.get("eval_suite_sha256")
    current_sig = compute_evaluation_signature(target_agent, suite_hash)
    recorded_sig = state.get("optimized_signature")
    if recorded_sig is not None and current_sig is not None:
        return tuple(recorded_sig) == current_sig
    return True

