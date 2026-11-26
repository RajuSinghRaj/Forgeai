"""Tests for ForgeAI Pydantic data contracts and assignment constraints."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from forgeai.schemas import (
    BriefInput,
    ChunkingConfig,
    ClarifyingAnswer,
    DeclarativeAgentDefinition,
    EvalCase,
    EvalCaseResult,
    ExpectedBehavior,
    FactoryRunReport,
    FailureClassification,
    GroundabilityReport,
    GroundabilityVerdict,
    IterationScorecard,
    OptimizationIterationLog,
    OutputContract,
    RetrievalConfig,
    StructuredRequirements,
    ToolSelection,
)


def _sample_requirements() -> StructuredRequirements:
    return StructuredRequirements(
        domain_name="Generic Operations",
        scope="Answer factual questions strictly from supplied documentation.",
        audience="Operations analysts",
        tone="Concise and objective",
        must_answer_topics=["Eligibility rules", "Approval timelines"],
        must_refuse_topics=["Legal advice"],
        constraints=["Always cite source document"],
        clarifying_qa=[
            ClarifyingAnswer(
                question_id="q1",
                question="Who uses this agent?",
                answer="Operations analysts",
            )
        ],
    )


def _sample_agent() -> DeclarativeAgentDefinition:
    return DeclarativeAgentDefinition(
        agent_id="agent-v1",
        name="Ops Reference Agent",
        version=1,
        domain_summary="Answers grounded operational policy questions.",
        system_prompt="You are an evidence-grounded assistant. Cite sources for every claim.",
        chunking=ChunkingConfig(chunk_size=600, chunk_overlap=100),
        retrieval=RetrievalConfig(collection_name="ops_docs", top_k=4),
        tools=[ToolSelection(name="retrieve_documents", purpose="Retrieve grounded passages")],
        output_contract=OutputContract(require_citations=True),
        must_answer_topics=["Eligibility rules"],
        must_refuse_topics=["Legal advice"],
    )


def test_brief_input_rejects_short_text(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        BriefInput(brief_text="too short", documents_dir=tmp_path)


def test_structured_requirements_enforces_max_five_clarifying_questions() -> None:
    too_many = [
        ClarifyingAnswer(question_id=f"q{i}", question=f"Question {i}?", answer="Yes")
        for i in range(6)
    ]
    with pytest.raises(ValidationError):
        StructuredRequirements(
            domain_name="Test",
            scope="Answer questions from corpus.",
            audience="Staff",
            tone="Neutral",
            must_answer_topics=["Topic A"],
            clarifying_qa=too_many,
        )


def test_chunking_config_rejects_overlap_ge_chunk_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=300, chunk_overlap=300)


def test_declarative_agent_json_roundtrip() -> None:
    agent = _sample_agent()
    payload = agent.model_dump_json()
    reloaded = DeclarativeAgentDefinition.model_validate_json(payload)
    assert reloaded == agent


def test_eval_case_and_result_validation() -> None:
    with pytest.raises(ValidationError):
        EvalCase(
            case_id="c1",
            question="What is the approval window?",
            expected_behavior=ExpectedBehavior.ANSWER,
            expected_facts=[],
            source_documents=["policy.pdf"],
            topic="Approval timelines",
        )

    refuse_case = EvalCase(
        case_id="c2",
        question="Can you draft a binding legal contract?",
        expected_behavior=ExpectedBehavior.REFUSE,
        topic="Legal advice",
    )
    assert refuse_case.expected_behavior == ExpectedBehavior.REFUSE

    with pytest.raises(ValidationError):
        EvalCaseResult(
            case_id="c1",
            question="What is the approval window?",
            expected_behavior=ExpectedBehavior.ANSWER,
            agent_answer="Uncited guess.",
            refused=False,
            grounding_score=0.2,
            correctness_score=0.0,
            refusal_correct=True,
            passed=False,
            failure_classification=FailureClassification.NONE,
            judge_reasoning="Failed without classification.",
        )


def test_factory_run_report_enforces_max_three_optimization_logs() -> None:
    scorecard = IterationScorecard(
        iteration=0,
        total_cases=1,
        passed_cases=0,
        pass_rate=0.0,
        avg_grounding_score=0.3,
        avg_correctness_score=0.2,
        refusal_accuracy=1.0,
    )
    with pytest.raises(ValidationError):
        OptimizationIterationLog(
            iteration=4,
            diagnosis_summary="Fourth iteration must be rejected.",
            before_scorecard=scorecard,
            agent_version_before=4,
            agent_version_after=5,
        )

    logs = [
        OptimizationIterationLog(
            iteration=min(i, 3),
            diagnosis_summary="Adjusted retrieval top_k.",
            before_scorecard=scorecard,
            agent_version_before=i,
            agent_version_after=i + 1,
        )
        for i in range(1, 5)
    ]
    with pytest.raises(ValidationError):
        FactoryRunReport(
            run_id="run-1",
            gemini_model="gemini-2.5-flash",
            embedding_model="all-MiniLM-L6-v2",
            requirements=_sample_requirements(),
            groundability=GroundabilityReport(
                overall_verdict=GroundabilityVerdict.SUPPORTED,
                can_proceed=True,
            ),
            final_agent=_sample_agent(),
            optimization_logs=logs,
            stop_reason="max_iterations",
        )

