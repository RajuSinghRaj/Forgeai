"""Deterministic tests for PDF evaluation report generation, download, and deployment readiness."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from forgeai.config import load_settings
from forgeai.evaluation import (
    build_evaluation_report_pdf,
    build_evaluation_report_pdf_from_state,
)
from forgeai.optimization.report import build_final_report
from forgeai.schemas import (
    ChunkingConfig,
    DeclarativeAgentDefinition,
    EvalCase,
    EvalCaseCategory,
    EvalCaseResult,
    ExpectedBehavior,
    FailureClassification,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    IterationScorecard,
    OptimizationIterationLog,
    OutputContract,
    RetrievalConfig,
    StructuredRequirements,
)


def _sample_agent(version: int = 2) -> DeclarativeAgentDefinition:
    return DeclarativeAgentDefinition(
        agent_id="policy_helper",
        name="Policy Helper Agent",
        version=version,
        domain_summary="HR policy assistant answering employee leave and conduct questions.",
        system_prompt="You are Policy Helper. Answer strictly from retrieved sources.",
        chunking=ChunkingConfig(chunk_size=800, chunk_overlap=120, csv_rows_per_unit=1),
        retrieval=RetrievalConfig(collection_name="policy_col", top_k=5),
        output_contract=OutputContract(response_format="Markdown"),
        source_documents=["handbook.pdf"],
        groundability_limitations=["Travel reimbursement caps are not specified in handbook.pdf."],
    )


def _sample_requirements() -> StructuredRequirements:
    return StructuredRequirements(
        domain_name="HR Policy",
        purpose="Answer HR policy questions.",
        scope="Leave, benefits, and conduct.",
        audience="Employees",
        tone="Professional",
        must_answer_topics=["Leave policy"],
        must_refuse_topics=["Legal advice"],
    )


def _sample_groundability() -> GroundabilityReport:
    return GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Leave policy",
                verdict=GroundabilityVerdict.SUPPORTED,
                evidence_summary="Covered in handbook.pdf.",
                supporting_sources=["handbook.pdf"],
            )
        ],
    )


def _sample_results() -> list[EvalCaseResult]:
    return [
        EvalCaseResult(
            case_id="eval_001",
            topic="Leave policy",
            category=EvalCaseCategory.ANSWERABLE,
            question="How many annual leave days are granted?",
            expected_behavior=ExpectedBehavior.ANSWER,
            agent_answer="Employees receive 20 annual leave days [handbook.pdf].",
            refused=False,
            cited_sources=["handbook.pdf"],
            passed=True,
            grounding_score=1.0,
            correctness_score=1.0,
            citation_accurate=True,
            refusal_correct=True,
            contract_compliant=True,
            failure_classification=FailureClassification.NONE,
            judge_reasoning="Fully grounded and accurate.",
        ),
        EvalCaseResult(
            case_id="eval_002",
            category=EvalCaseCategory.EDGE_CASE,
            question="What is the bereavement leave duration for contractors?",
            expected_behavior=ExpectedBehavior.ANSWER,
            agent_answer="Contractors receive 5 days of bereavement leave.",
            refused=False,
            cited_sources=[],
            passed=False,
            grounding_score=0.2,
            correctness_score=0.0,
            citation_accurate=False,
            refusal_correct=True,
            contract_compliant=False,
            failure_classification=FailureClassification.HALLUCINATION,
            judge_reasoning="Unsupported contractor leave claim not present in handbook.pdf.",
            deterministic_issues=["Missing required citation"],
        ),
    ]


def _sample_scorecard(
    *,
    iteration: int = 1,
    total_cases: int = 20,
    passed_cases: int = 17,
    is_partial: bool = False,
) -> IterationScorecard:
    failed_cases = total_cases - passed_cases
    pass_rate = round(passed_cases / total_cases, 4) if total_cases else 0.0
    return IterationScorecard(
        iteration=iteration,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        answerable_accuracy=0.90,
        out_of_scope_refusal_accuracy=1.0,
        edge_case_accuracy=0.60,
        grounding_rate=0.95,
        citation_accuracy=0.88,
        contract_compliance_rate=0.92,
        avg_grounding_score=0.93,
        avg_correctness_score=0.86,
        refusal_accuracy=1.0,
        is_partial=is_partial,
    )


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def test_pdf_report_contains_exact_retained_best_scorecard_and_failures() -> None:
    agent = _sample_agent(version=2)
    sc = _sample_scorecard(iteration=1, total_cases=20, passed_cases=17)
    results = _sample_results()
    baseline = _sample_scorecard(iteration=0, total_cases=20, passed_cases=14)
    opt_log = OptimizationIterationLog(
        iteration=1,
        agent_version_before=1,
        agent_version_after=2,
        diagnosis_summary="Tightened citation rules.",
        failed_case_ids=["eval_002"],
        before_scorecard=baseline,
        after_scorecard=sc,
        accepted=True,
        acceptance_reason="Pass rate improved from 70% to 85%.",
        remaining_failed_case_ids=["eval_002"],
    )
    eval_dt = datetime(2026, 11, 28, 10, 30, tzinfo=timezone.utc)
    suite_hash = "abc123def4567890" * 4

    pdf_bytes = build_evaluation_report_pdf(
        agent=agent,
        scorecard=sc,
        results=results,
        eval_suite_sha256=suite_hash,
        evaluation_date=eval_dt,
        optimization_logs=[opt_log],
        stop_reason="max_iterations_reached",
        honest_limitations=[
            "Travel reimbursement caps are not specified in handbook.pdf.",
            "Remaining failure eval_002 [hallucination]: Contractor bereavement leave.",
        ],
        run_id="run_test_001",
        gemini_model="gemini-test-model",
    )

    assert pdf_bytes.startswith(b"%PDF-1.4")
    text = _extract_pdf_text(pdf_bytes)

    assert "Policy Helper Agent (policy_helper) v2" in text
    assert "2026-11-28 10:30 UTC" in text
    assert suite_hash in text
    assert "17/20 passed" in text
    assert "Total Cases: 20" in text
    assert "Passed Cases: 17" in text
    assert "Failed Cases: 3" in text
    assert "85%" in text  # pass rate
    assert "95%" in text  # grounding rate
    assert "88%" in text  # citation accuracy
    assert "92%" in text  # contract compliance
    assert "eval_002 [hallucination]" in text
    assert "Unsupported contractor leave claim not present in handbook.pdf." in text
    assert "Optimization Iterations: 1" in text
    assert "max_iterations_reached" in text
    assert "Travel reimbursement caps are not specified in handbook.pdf." in text


def test_pdf_from_state_exports_retained_best_not_partial_candidate() -> None:
    best_agent = _sample_agent(version=2)
    best_sc = _sample_scorecard(iteration=1, total_cases=20, passed_cases=16, is_partial=False)
    partial_sc = _sample_scorecard(iteration=2, total_cases=5, passed_cases=1, is_partial=True)
    partial_sc.expected_total_cases = 20

    state = {
        "run_id": "run_partial_guard",
        "agent_definition": best_agent,
        "best_agent": best_agent,
        "best_scorecard": best_sc,
        "best_results": _sample_results(),
        "scorecards": [
            _sample_scorecard(iteration=0, total_cases=20, passed_cases=12),
            best_sc,
            partial_sc,
        ],
        "latest_candidate_scorecard": partial_sc,
        "latest_candidate_agent": _sample_agent(version=3),
        "eval_suite_sha256": "deadbeef" * 8,
        "stop_reason": "early_stopped_candidate_rejected",
    }

    pdf_bytes = build_evaluation_report_pdf_from_state(state)
    text = _extract_pdf_text(pdf_bytes)

    assert "Policy Helper Agent (policy_helper) v2" in text
    assert "16/20 passed" in text
    assert "80%" in text
    assert "1/5 passed" not in text
    assert "v3" not in text


def test_streamlit_apptest_renders_pdf_download_button_and_valid_pdf_bytes() -> None:
    agent = _sample_agent(version=2)
    baseline = _sample_scorecard(iteration=0, total_cases=20, passed_cases=14)
    best_sc = _sample_scorecard(iteration=1, total_cases=20, passed_cases=18)
    results = _sample_results()
    eval_cases = [
        EvalCase(
            case_id="eval_001",
            topic="Leave policy",
            category=EvalCaseCategory.ANSWERABLE,
            question="How many annual leave days are granted?",
            expected_behavior=ExpectedBehavior.ANSWER,
            expected_facts=["20 annual leave days"],
            source_documents=["handbook.pdf"],
        )
    ]
    report = build_final_report(
        run_id="run_ui_pdf",
        gemini_model="gemini-test",
        embedding_model="all-MiniLM-L6-v2",
        requirements=_sample_requirements(),
        groundability=_sample_groundability(),
        final_agent=agent,
        eval_cases=eval_cases,
        final_results=results,
        baseline_scorecard=baseline,
        final_scorecard=best_sc,
        scorecards=[baseline, best_sc],
        optimization_logs=[],
        target_pass_rate=0.90,
        stop_reason="target_pass_rate_met",
        eval_suite_sha256="cafe1234" * 8,
    )

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.session_state["factory_state"] = {
        "run_id": "run_ui_pdf",
        "requirements": _sample_requirements(),
        "groundability": _sample_groundability(),
        "agent_definition": agent,
        "best_agent": agent,
        "eval_cases": eval_cases,
        "eval_results": results,
        "best_results": results,
        "best_scorecard": best_sc,
        "scorecards": [baseline, best_sc],
        "optimization_logs": [],
        "final_report": report,
        "eval_suite_sha256": "cafe1234" * 8,
    }
    at.run()
    assert not at.exception

    download_buttons = at.get("download_button")
    pdf_buttons = [b for b in download_buttons if b.label == "Download Evaluation Report (PDF)"]
    assert len(pdf_buttons) == 1


def test_config_reads_env_and_hosting_secrets_without_exposing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-env-secret-value")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-model")
    settings = load_settings(env_file=None)
    summary = settings.status_summary()

    assert settings.gemini_api_key == "test-env-secret-value"
    assert settings.gemini_model == "gemini-env-model"
    assert summary["api_key_present"] is True
    assert "test-env-secret-value" not in str(summary)

    # Verify fallback to hosting secrets when env vars are absent
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    fake_st = SimpleNamespace(
        secrets={"GEMINI_API_KEY": "hosting-secret-key", "GEMINI_MODEL": "gemini-hosted-model"}
    )
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_st)
    hosted_settings = load_settings(env_file=None)
    assert hosted_settings.gemini_api_key == "hosting-secret-key"
    assert hosted_settings.gemini_model == "gemini-hosted-model"
