"""Tests for ToolRegistry, scorecard computation, and LangGraph workflow orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeai.evaluation import compute_scorecard
from forgeai.schemas import (
    BriefInput,
    EvalCaseResult,
    ExpectedBehavior,
    FailureClassification,
    GroundabilityReport,
    GroundabilityVerdict,
)
from forgeai.tools import build_default_registry
from forgeai.workflow import FactoryState, build_factory_graph, should_continue_optimization


def _make_result(case_id: str, passed: bool, failure: FailureClassification) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=case_id,
        question=f"Question {case_id}?",
        expected_behavior=ExpectedBehavior.ANSWER,
        agent_answer="Sample answer.",
        refused=False,
        grounding_score=1.0 if passed else 0.3,
        correctness_score=1.0 if passed else 0.2,
        refusal_correct=True,
        passed=passed,
        failure_classification=failure,
        judge_reasoning="Evaluated.",
    )


def test_tool_registry_csv_lookup_and_error_handling(tmp_path: Path) -> None:
    (tmp_path / "items.csv").write_text(
        "sku,category,status\nA1,valve,in_stock\nB2,gasket,backorder\n",
        encoding="utf-8",
    )
    registry = build_default_registry(documents_dir=tmp_path)
    specs = registry.list_specs()
    assert {"retrieve_documents", "inspect_csv_records"}.issubset({s["name"] for s in specs})

    rows = registry.execute("inspect_csv_records", file_name="items.csv", column="sku", contains="b2")
    assert len(rows) == 1
    assert rows[0]["status"] == "backorder"

    with pytest.raises(KeyError):
        registry.execute("nonexistent_tool")


def test_compute_scorecard_and_optimization_routing() -> None:
    results = [
        _make_result("1", True, FailureClassification.NONE),
        _make_result("2", False, FailureClassification.RETRIEVAL_MISS),
        _make_result("3", False, FailureClassification.HALLUCINATION),
        _make_result("4", True, FailureClassification.NONE),
    ]
    scorecard = compute_scorecard(results, iteration=1)

    assert scorecard.total_cases == 4
    assert scorecard.passed_cases == 2
    assert scorecard.pass_rate == 0.5
    assert scorecard.failure_counts[FailureClassification.RETRIEVAL_MISS] == 1
    assert scorecard.failure_counts[FailureClassification.HALLUCINATION] == 1

    assert (
        should_continue_optimization(
            {"scorecards": [scorecard], "iteration": 1, "max_iterations": 3, "target_pass_rate": 0.9}
        )
        == "optimize"
    )
    assert (
        should_continue_optimization(
            {"scorecards": [scorecard], "iteration": 3, "max_iterations": 3, "target_pass_rate": 0.9}
        )
        == "report"
    )


def test_langgraph_workflow_runs_loop_and_stops_at_max_iterations(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("Reference content for ingestion.", encoding="utf-8")

    def fake_intake(_state: FactoryState) -> FactoryState:
        return {}

    def fake_groundability(_state: FactoryState) -> FactoryState:
        return {
            "groundability": GroundabilityReport(
                overall_verdict=GroundabilityVerdict.SUPPORTED,
                can_proceed=True,
            )
        }

    def fake_generate_agent(_state: FactoryState) -> FactoryState:
        return {}

    def fake_generate_evals(_state: FactoryState) -> FactoryState:
        return {}

    def fake_run_and_judge(state: FactoryState) -> FactoryState:
        iteration = state.get("iteration", 0)
        scorecards = list(state.get("scorecards") or [])
        # Pass rate hits target on iteration 2
        passed = iteration >= 2
        res = _make_result(
            "c1",
            passed=passed,
            failure=FailureClassification.NONE if passed else FailureClassification.PROMPT_DEFECT,
        )
        scorecards.append(compute_scorecard([res], iteration=iteration))
        return {"eval_results": [res], "scorecards": scorecards}

    def fake_optimize(state: FactoryState) -> FactoryState:
        return {"iteration": state.get("iteration", 0) + 1}

    def fake_report(state: FactoryState) -> FactoryState:
        return {"stop_reason": f"completed_at_iteration_{state.get('iteration', 0)}"}

    compiled = build_factory_graph(
        node_handlers={
            "intake_interview": fake_intake,
            "groundability_guard": fake_groundability,
            "generate_agent": fake_generate_agent,
            "generate_evals": fake_generate_evals,
            "run_and_judge": fake_run_and_judge,
            "optimize_agent": fake_optimize,
            "build_report": fake_report,
        }
    )

    final_state = compiled.invoke(
        {
            "brief_input": BriefInput(
                brief_text="Build an operations documentation assistant.",
                documents_dir=tmp_path,
            ),
            "target_pass_rate": 0.9,
            "max_iterations": 3,
        }
    )

    assert len(final_state["documents"]) == 1
    assert final_state["iteration"] == 2
    assert len(final_state["scorecards"]) == 3  # initial (0) + opt 1 + opt 2
    assert final_state["stop_reason"] == "completed_at_iteration_2"

