"""Zero-cost tests for ForgeAI Phase A API consumption protection."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from forgeai.config import ForgeSettings
from forgeai.llm.gemini_client import GeminiClient
from forgeai.llm.telemetry import ApiTelemetryTracker, read_telemetry_summary, telemetry_stage
from forgeai.schemas.agent import (
    ChunkingConfig,
    DeclarativeAgentDefinition,
    OutputContract,
    RetrievalConfig,
    ToolSelection,
)
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseCategory,
    EvalCaseResult,
    ExpectedBehavior,
    FailureClassification,
    IterationScorecard,
)
from forgeai.schemas.intake import (
    ClarifyingQuestion,
    DocumentInput,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    StructuredRequirements,
)
from forgeai.workflow.protection import (
    compute_evaluation_signature,
    execution_lock_guard,
    is_evaluation_already_completed,
)


def _sample_requirements() -> StructuredRequirements:
    return StructuredRequirements(
        domain_name="Test Domain",
        purpose="Test agent purpose",
        scope="Answer supported questions cleanly",
        audience="Staff",
        tone="Professional",
        must_answer_topics=["Leave policy"],
        must_refuse_topics=["Payroll secrets"],
    )


def _sample_groundability() -> GroundabilityReport:
    return GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        approval_required=False,
        findings=[
            GroundabilityFinding(
                topic="Leave policy",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["policy.md"],
                evidence_summary="Covered in policy.md.",
            )
        ],
    )


def _sample_agent(version: int = 1) -> DeclarativeAgentDefinition:
    req = _sample_requirements()
    return DeclarativeAgentDefinition(
        agent_id="test_agent_001",
        name="Test Agent",
        description="Test agent definition",
        version=version,
        domain_summary="Test domain",
        system_prompt="Follow grounded policy strictly.",
        chunking=ChunkingConfig(chunk_size=600, chunk_overlap=100, csv_rows_per_unit=1),
        retrieval=RetrievalConfig(collection_name="test_col", top_k=4),
        tools=[ToolSelection(name="retrieve_documents", purpose="Retrieve docs")],
        output_contract=OutputContract(),
        must_answer_topics=req.must_answer_topics,
        must_refuse_topics=req.must_refuse_topics,
        source_documents=["policy.md"],
        approved_requirements=req,
    )


def _sample_eval_cases(count: int = 20) -> list[EvalCase]:
    return [
        EvalCase(
            case_id=f"case_{i:02d}",
            topic="Leave policy",
            category=EvalCaseCategory.ANSWERABLE,
            question=f"Question {i}?",
            expected_behavior=ExpectedBehavior.ANSWER,
            expected_facts=["fact"],
            source_documents=["policy.md"],
        )
        for i in range(1, count + 1)
    ]


def _sample_results(count: int = 20) -> list[EvalCaseResult]:
    return [
        EvalCaseResult(
            case_id=f"case_{i:02d}",
            question=f"Question {i}?",
            category=EvalCaseCategory.ANSWERABLE,
            expected_behavior=ExpectedBehavior.ANSWER,
            agent_answer="Answer citing policy.md.",
            refused=False,
            cited_sources=["policy.md"],
            grounding_score=1.0,
            correctness_score=1.0,
            refusal_correct=True,
            citation_valid=True,
            contract_compliant=True,
            passed=True,
            failure_classification=FailureClassification.NONE,
            judge_reasoning="Grounded and accurate.",
        )
        for i in range(1, count + 1)
    ]


def _sample_scorecard() -> IterationScorecard:
    return IterationScorecard(
        iteration=0,
        total_cases=20,
        passed_cases=20,
        failed_cases=0,
        pass_rate=1.0,
        answerable_accuracy=1.0,
        out_of_scope_refusal_accuracy=1.0,
        edge_case_accuracy=1.0,
        grounding_rate=1.0,
        citation_accuracy=1.0,
        contract_compliance_rate=1.0,
        avg_grounding_score=1.0,
        avg_correctness_score=1.0,
        refusal_accuracy=1.0,
    )


def test_execution_lock_guard_prevents_overlap_and_releases() -> None:
    lock = threading.Lock()
    with execution_lock_guard(lock) as first:
        assert first is True
        with execution_lock_guard(lock) as second:
            assert second is False

    # Reacquiring must succeed cleanly once outer scope exits.
    with pytest.raises(RuntimeError, match="boom"):
        with execution_lock_guard(lock) as third:
            assert third is True
            raise RuntimeError("boom")

    with execution_lock_guard(lock) as fourth:
        assert fourth is True


def test_completed_evaluation_detection_uses_agent_version_and_suite_hash() -> None:
    agent_v1 = _sample_agent(version=1)
    agent_v2 = _sample_agent(version=2)
    suite_hash = "abc123def456"
    state = {
        "agent_definition": agent_v1,
        "eval_suite_sha256": suite_hash,
        "eval_results": _sample_results(1),
        "scorecards": [_sample_scorecard()],
        "evaluated_signature": compute_evaluation_signature(agent_v1, suite_hash),
    }

    assert is_evaluation_already_completed(state, agent_v1) is True
    assert is_evaluation_already_completed(state, agent_v2) is False

    changed_suite_state = {**state, "eval_suite_sha256": "different_hash"}
    assert is_evaluation_already_completed(changed_suite_state, agent_v1) is False


def test_503_and_timeout_retries_capped_at_2_and_sdk_retries_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key-value")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    settings = ForgeSettings(_env_file=None, artifacts_dir=tmp_path / "artifacts")
    sleeps: list[float] = []
    calls = 0

    class _Always503Models:
        def generate_content(self, **_kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise RuntimeError("503 UNAVAILABLE")

    class _FakeSDK:
        models = _Always503Models()

    client = GeminiClient(settings=settings, sleep_fn=sleeps.append)
    client._client = _FakeSDK()

    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        client.generate_text("hello")
    assert calls == 3  # 1 initial attempt + max 2 transient retries
    assert sleeps == [2.0, 4.0]

    # Timeout errors obey the same max 2 transient retries cap
    calls = 0
    sleeps.clear()

    class _AlwaysTimeoutModels:
        def generate_content(self, **_kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise TimeoutError("request timed out")

    client_timeout = GeminiClient(settings=settings, sleep_fn=sleeps.append)
    client_timeout._client = types.SimpleNamespace(models=_AlwaysTimeoutModels())
    with pytest.raises(TimeoutError):
        client_timeout.generate_text("hello")
    assert calls == 3
    assert sleeps == [2.0, 4.0]

    # Verify SDK-level retries are explicitly disabled (attempts=1) via HttpOptions
    captured: dict[str, Any] = {}

    class _FakeGenAIClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            captured["api_key"] = api_key
            captured["http_options"] = http_options

    fake_genai_mod = types.ModuleType("google.genai")
    fake_genai_mod.Client = _FakeGenAIClient  # type: ignore[attr-defined]
    fake_types_mod = types.ModuleType("google.genai.types")
    fake_types_mod.HttpRetryOptions = lambda *, attempts: {"attempts": attempts}  # type: ignore[attr-defined]
    fake_types_mod.HttpOptions = lambda *, timeout, retry_options: {  # type: ignore[attr-defined]
        "timeout": timeout,
        "retry_options": retry_options,
    }
    fake_genai_mod.types = fake_types_mod  # type: ignore[attr-defined]
    fake_google_mod = types.ModuleType("google")
    fake_google_mod.genai = fake_genai_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_mod)

    fresh_client = GeminiClient(settings=settings)
    fresh_client._get_client()
    assert captured["http_options"]["retry_options"]["attempts"] == 1


def test_immediate_failure_for_daily_quota_and_4xx_and_respects_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key-value")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    settings = ForgeSettings(_env_file=None, artifacts_dir=tmp_path / "artifacts")

    for err_msg in (
        "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier 'retryDelay': '49463s'",
        "400 INVALID_ARGUMENT bad schema",
        "403 PERMISSION_DENIED invalid key",
    ):
        sleeps: list[float] = []
        calls = 0

        class _ErrModels:
            def __init__(self, msg: str) -> None:
                self.msg = msg

            def generate_content(self, **_kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                raise RuntimeError(self.msg)

        c = GeminiClient(settings=settings, sleep_fn=sleeps.append)
        c._client = types.SimpleNamespace(models=_ErrModels(err_msg))
        with pytest.raises(RuntimeError):
            c.generate_text("hello")
        assert calls == 1
        assert sleeps == []

    # Per-minute 429 respects RetryInfo retryDelay and succeeds on retry
    sleeps = []
    calls = 0

    class _MinuteRateLimitModels:
        def generate_content(self, **_kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED. 'retryDelay': '15s'")
            return types.SimpleNamespace(text="ok", usage_metadata=None)

    c_rpm = GeminiClient(settings=settings, sleep_fn=sleeps.append)
    c_rpm._client = types.SimpleNamespace(models=_MinuteRateLimitModels())
    assert c_rpm.generate_text("hello") == "ok"
    assert calls == 2
    assert sleeps == [17.0]


def test_telemetry_records_stages_tokens_retries_errors_and_missing_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret_key = "AIzaSyDummySecretValueForTesting123456"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    artifacts_dir = tmp_path / "artifacts"
    settings = ForgeSettings(_env_file=None, artifacts_dir=artifacts_dir)
    tracker = ApiTelemetryTracker(artifacts_dir=artifacts_dir)
    sleeps: list[float] = []
    call_idx = 0

    class _TelemetryModels:
        def generate_content(self, **_kwargs: Any) -> Any:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                usage = types.SimpleNamespace(
                    prompt_token_count=120,
                    candidates_token_count=30,
                    total_token_count=150,
                )
                return types.SimpleNamespace(text="answer_1", usage_metadata=usage)
            if call_idx == 2:
                return types.SimpleNamespace(text="answer_no_usage")
            if call_idx == 3:
                raise RuntimeError("503 UNAVAILABLE temporary")
            if call_idx == 4:
                return types.SimpleNamespace(text="judged_after_retry", usage_metadata=None)
            raise RuntimeError(f"404 NOT_FOUND with leaked key {secret_key} and SECRET_DOC_TEXT")

    client = GeminiClient(settings=settings, sleep_fn=sleeps.append, telemetry=tracker)
    client._client = types.SimpleNamespace(models=_TelemetryModels())

    with telemetry_stage("agent_runner"):
        assert client.generate_text("DOC_CONTENT_SHOULD_NOT_APPEAR") == "answer_1"
        assert client.generate_text("second") == "answer_no_usage"

    with telemetry_stage("hybrid_judge"):
        assert client.generate_text("third") == "judged_after_retry"
        with pytest.raises(RuntimeError):
            client.generate_text("fourth")

    telemetry_path = artifacts_dir / "telemetry" / "gemini_requests.jsonl"
    assert telemetry_path.exists()
    raw_log = telemetry_path.read_text(encoding="utf-8")
    assert secret_key not in raw_log
    assert "DOC_CONTENT_SHOULD_NOT_APPEAR" not in raw_log

    lines = [json.loads(line) for line in raw_log.splitlines() if line.strip()]
    assert len(lines) == 4
    assert lines[0]["stage"] == "agent_runner"
    assert lines[0]["model"] == "gemini-3.5-flash-lite"
    assert lines[0]["request_count"] == 1
    assert lines[0]["tokens_available"] is True
    assert lines[0]["total_tokens"] == 150

    assert lines[1]["stage"] == "agent_runner"
    assert lines[1]["tokens_available"] is False
    assert lines[1]["total_tokens"] is None

    assert lines[2]["stage"] == "hybrid_judge"
    assert lines[2]["retries"] == 1
    assert lines[2]["attempts"] == 2
    assert lines[2]["success"] is True

    assert lines[3]["stage"] == "hybrid_judge"
    assert lines[3]["success"] is False
    assert lines[3]["retries"] == 0
    assert "[REDACTED_KEY]" in lines[3]["error"]

    summary = read_telemetry_summary(artifacts_dir)
    assert summary["total_requests"] == 4
    assert summary["total_retries"] == 1
    assert summary["total_errors"] == 1
    assert summary["total_tokens"] == 150
    assert summary["by_stage"]["agent_runner"]["requests"] == 2
    assert summary["by_stage"]["hybrid_judge"]["requests"] == 2


def test_apptest_duplicate_lock_and_completed_eval_protection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key-value")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    invocations: list[dict[str, Any]] = []
    agent = _sample_agent(version=1)
    req = _sample_requirements()
    gb = _sample_groundability()
    cases = _sample_eval_cases(20)
    results = _sample_results(20)
    scorecard = _sample_scorecard()
    suite_hash = "deadbeef12345678"

    class _MockGraph:
        def invoke(self, state_in: dict[str, Any]) -> dict[str, Any]:
            invocations.append(dict(state_in))
            return {
                **state_in,
                "eval_results": results,
                "scorecards": [scorecard],
                "phase4_status": "completed",
            }

    monkeypatch.setattr("forgeai.workflow.build_factory_graph", lambda **_kwargs: _MockGraph())

    # 1. Completed evaluations are locked against accidental rerun until confirmed
    completed_state = {
        "documents": [
            DocumentInput(
                doc_id="doc1",
                file_name="policy.md",
                file_type=".md",
                file_path="policy.md",
                unit_count=1,
                size_bytes=120,
            )
        ],
        "document_units": [],
        "clarifying_questions": [
            ClarifyingQuestion(
                question_id="q1",
                question="Any special constraints?",
                rationale="Clarify scope",
                target_field="constraints",
            )
        ],
        "requirements": req,
        "groundability": gb,
        "agent_definition": agent,
        "eval_cases": cases,
        "eval_suite_sha256": suite_hash,
        "eval_results": results,
        "scorecards": [scorecard],
        "evaluated_signature": compute_evaluation_signature(agent, suite_hash),
    }

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(app_path)
    at.session_state["factory_state"] = completed_state
    at.run(timeout=30)
    assert not at.exception

    eval_btn = next(b for b in at.button if b.label == "Run Evaluations & Judge")
    assert eval_btn.disabled is True
    intake_btn = next(b for b in at.button if b.label == "Start Intake Interview")
    assert intake_btn.disabled is True
    assert any("Evaluation already completed" in info.value for info in at.info)
    assert len(invocations) == 0

    # Checking the explicit confirmation box enables Run Evaluations & Judge cleanly
    confirm_box = next(c for c in at.checkbox if c.label == "Confirm rerun of completed evaluations")
    confirm_box.check().run(timeout=30)
    eval_btn_after = next(b for b in at.button if b.label == "Run Evaluations & Judge")
    assert eval_btn_after.disabled is False

    eval_btn_after.click().run(timeout=30)
    assert len(invocations) == 1
    eval_btn_post = next(b for b in at.button if b.label == "Run Evaluations & Judge")
    assert eval_btn_post.disabled is True

    # 2. In-flight active_operation disables all expensive action buttons simultaneously
    at_busy = AppTest.from_file(app_path)
    at_busy.session_state["factory_state"] = completed_state
    at_busy.session_state["active_operation"] = "run_evaluations"
    at_busy.run(timeout=30)
    assert not at_busy.exception
    assert any("Operation in progress" in w.value for w in at_busy.warning)

    for label in (
        "Start Intake Interview",
        "Generate Agent",
        "Generate Evaluation Set",
        "Run Evaluations & Judge",
        "Optimize Agent",
    ):
        btn = next(b for b in at_busy.button if b.label == label)
        assert btn.disabled is True
    submit_btn = next(b for b in at_busy.button if "Synthesize Requirements" in b.label)
    assert submit_btn.disabled is True
