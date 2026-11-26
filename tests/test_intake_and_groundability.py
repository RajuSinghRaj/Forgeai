"""Deterministic Phase 2 unit tests for Intake Interviewer, Groundability Guard, and LangGraph."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from forgeai.config import ForgeSettings
from forgeai.intake import GroundabilityGuard, IntakeInterviewer
from forgeai.retrieval import ChromaVectorStore, load_document_folder
from forgeai.schemas import (
    BriefInput,
    ClarifyingAnswer,
    ClarifyingQuestion,
    ClarifyingQuestionBatch,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    StructuredRequirements,
)
from forgeai.workflow import build_factory_graph


class _DeterministicEmbedder:
    """Offline character-trigram hash embedder for deterministic ChromaDB tests."""

    def __init__(self, dim: int = 48) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "deterministic-test-embedder"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        lowered = text.lower()
        for i in range(max(1, len(lowered) - 2)):
            tok = lowered[i : i + 3]
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _StubGeminiClient:
    """Mock GeminiClient returning scripted schema instances and recording prompts."""

    def __init__(self, responses: list[Any] | None = None, error: Exception | None = None) -> None:
        self._responses = list(responses or [])
        self._error = error
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, response_schema: type[Any], **_kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError(f"No stubbed response remaining for {response_schema.__name__}.")
        result = self._responses.pop(0)
        assert isinstance(result, response_schema)
        return result


def _create_sample_corpus(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "leave_policy.md").write_text(
        "Annual leave entitlement is 20 paid days per calendar year. "
        "Sick leave requires medical certification after 3 consecutive days.",
        encoding="utf-8",
    )
    (docs_dir / "attendance_rules.csv").write_text(
        "rule_id,topic,policy\n"
        "R1,Core hours,Core working hours are 10:00 to 16:00 local time.\n"
        "R2,Remote work,Employees may work remotely up to 2 days per week with manager sign-off.\n",
        encoding="utf-8",
    )
    return docs_dir


def test_intake_interviewer_generates_bounded_questions_and_handles_partial_answers(
    tmp_path: Path,
) -> None:
    docs_dir = _create_sample_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    brief = BriefInput(
        brief_text="Build an internal policy assistant for employees using our leave and attendance files.",
        documents_dir=docs_dir,
    )

    five_questions = [
        ClarifyingQuestion(
            question_id=f"q{i}",
            question=f"Clarifying question {i} about scope or tone?",
            rationale="Needed to remove ambiguity.",
            target_field="scope",
        )
        for i in range(1, 6)
    ]
    synthesized = StructuredRequirements(
        domain_name="Workplace Policy",
        purpose="Answer internal employee policy questions strictly from supplied documents.",
        scope="Cover annual leave, sick leave, core hours, and remote work rules.",
        audience="Full-time employees",
        tone="Clear, neutral, and helpful",
        must_answer_topics=["Annual leave entitlement", "Remote work rules"],
        must_refuse_topics=["Legal representation"],
        expected_response_behavior="Always cite source_file and unit_label; refuse out-of-scope requests.",
        constraints=["Cite exact policy documents"],
        assumptions=["Assumed standard full-time employee eligibility where skipped."],
    )
    stub = _StubGeminiClient(
        responses=[
            ClarifyingQuestionBatch(questions=five_questions),
            synthesized,
        ]
    )

    interviewer = IntakeInterviewer(gemini_client=stub, max_questions=5)
    questions = interviewer.generate_questions(brief, docs, units)
    assert len(questions) == 5
    assert "leave_policy.md" in stub.prompts[0]
    assert "attendance_rules.csv" in stub.prompts[0]

    # Partial answers: q1 answered, q2 blank/skipped, q3 partially answered
    requirements = interviewer.synthesize_requirements(
        brief,
        docs,
        units,
        clarifying_questions=questions,
        clarifying_answers={
            "q1": "Only full-time staff, neutral tone.",
            "q2": "   ",
            "q3": "Not sure yet, use document defaults.",
        },
    )
    assert requirements.purpose
    assert requirements.expected_response_behavior
    assert len(requirements.clarifying_qa) == 2  # skipped blank q2 excluded
    assert "skipped by requester" in stub.prompts[1]


def test_intake_interviewer_enforces_max_five_questions_and_validates_limit() -> None:
    with pytest.raises(ValueError):
        IntakeInterviewer(gemini_client=_StubGeminiClient(), max_questions=6)

    raw_questions = [
        ClarifyingQuestion(
            question_id=f"q{i}",
            question=f"Distinct clarifying question number {i}?",
            rationale="Clarify boundary.",
            target_field="tone",
        )
        for i in range(1, 6)
    ]
    stub = _StubGeminiClient(responses=[ClarifyingQuestionBatch(questions=raw_questions)])
    interviewer = IntakeInterviewer(gemini_client=stub, max_questions=3)
    brief = BriefInput(
        brief_text="Build a general operational assistant grounded in local reference files.",
        documents_dir=Path("."),
    )
    capped = interviewer.generate_questions(brief, [], [])
    assert len(capped) == 3


def test_groundability_guard_fully_supported_and_source_attribution(tmp_path: Path) -> None:
    docs_dir = _create_sample_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    requirements = StructuredRequirements(
        domain_name="Workplace Policy",
        purpose="Answer leave and remote work questions.",
        scope="Leave entitlement and remote work guidelines.",
        audience="Employees",
        tone="Neutral",
        must_answer_topics=["Annual leave entitlement", "Remote work rules"],
        must_refuse_topics=["Medical diagnosis"],
    )

    llm_report = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Annual leave entitlement",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["leave_policy.md"],
                evidence_summary="leave_policy.md specifies 20 paid annual leave days.",
            ),
            GroundabilityFinding(
                topic="Remote work rules",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["attendance_rules.csv"],
                evidence_summary="attendance_rules.csv allows up to 2 remote days per week.",
            ),
            GroundabilityFinding(
                topic="Medical diagnosis",
                verdict=GroundabilityVerdict.UNSUPPORTED,
                is_out_of_scope=True,
                supporting_sources=[],
                evidence_summary="Out-of-scope refusal topic.",
                gap_description="Intentionally refused request.",
            ),
        ],
    )
    stub = _StubGeminiClient(responses=[llm_report])
    store = ChromaVectorStore(
        collection_name="test_supported",
        persist_dir=tmp_path / "chroma",
        embedding_provider=_DeterministicEmbedder(),
    )
    guard = GroundabilityGuard(gemini_client=stub, vector_store=store)
    report = guard.assess(requirements, docs, units)

    assert report.overall_verdict == GroundabilityVerdict.SUPPORTED
    assert report.can_proceed is True
    assert report.approval_required is False
    assert report.ungrounded_topics == []
    assert "Medical diagnosis" in report.out_of_scope_topics
    assert any(f.supporting_sources == ["leave_policy.md"] for f in report.findings)


def test_groundability_guard_partially_supported_and_blocks_unsupported(tmp_path: Path) -> None:
    docs_dir = _create_sample_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    requirements = StructuredRequirements(
        domain_name="Workplace Policy",
        scope="Leave, remote work, and salary revisions.",
        audience="Employees",
        tone="Neutral",
        must_answer_topics=[
            "Annual leave entitlement",
            "Remote work approval workflow",
            "Employee salary revision policy",
        ],
    )

    # Partial + unsupported findings
    llm_report = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.PARTIALLY_SUPPORTED,
        # Intentionally wrong True from LLM to verify Python guardrails override!
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Annual leave entitlement",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["leave_policy.md"],
                evidence_summary="20 paid days per year.",
            ),
            GroundabilityFinding(
                topic="Remote work approval workflow",
                verdict=GroundabilityVerdict.PARTIALLY_SUPPORTED,
                supporting_sources=["attendance_rules.csv"],
                evidence_summary="Mentions manager sign-off but not portal steps.",
                gap_description="Missing step-by-step approval workflow.",
            ),
            GroundabilityFinding(
                topic="Employee salary revision policy",
                verdict=GroundabilityVerdict.UNSUPPORTED,
                supporting_sources=[],
                evidence_summary="No salary revision information exists in corpus.",
                gap_description="No compensation or salary revision document provided.",
            ),
        ],
    )
    stub = _StubGeminiClient(responses=[llm_report])
    guard = GroundabilityGuard(
        gemini_client=stub,
        embedding_provider=_DeterministicEmbedder(),
    )

    report = guard.assess(requirements, docs, units, approved_with_gaps=False)
    assert report.overall_verdict == GroundabilityVerdict.UNSUPPORTED
    assert report.can_proceed is False
    assert report.approval_required is True
    assert report.ungrounded_topics == ["Employee salary revision policy"]

    # Explicit requester approval allows proceeding while preserving ungrounded_topics
    stub_approved = _StubGeminiClient(responses=[llm_report])
    guard_approved = GroundabilityGuard(
        gemini_client=stub_approved,
        embedding_provider=_DeterministicEmbedder(),
    )
    approved_report = guard_approved.assess(requirements, docs, units, approved_with_gaps=True)
    assert approved_report.can_proceed is True
    assert approved_report.approved_with_gaps is True
    assert approved_report.ungrounded_topics == ["Employee salary revision policy"]


def test_groundability_guard_prevents_silent_pass_on_omitted_or_hallucinated_sources(
    tmp_path: Path,
) -> None:
    docs_dir = _create_sample_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    requirements = StructuredRequirements(
        domain_name="Workplace Policy",
        scope="Leave and compensation.",
        audience="Employees",
        tone="Neutral",
        must_answer_topics=["Annual leave entitlement", "Bonus payout formula"],
    )

    # LLM omits "Bonus payout formula" and cites a non-existent filename for "Annual leave entitlement".
    deceptive_report = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Annual leave entitlement",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["nonexistent_doc.pdf"],
                evidence_summary="Claimed support from non-existent file.",
            ),
        ],
    )
    stub = _StubGeminiClient(responses=[deceptive_report])
    guard = GroundabilityGuard(
        gemini_client=stub,
        embedding_provider=_DeterministicEmbedder(),
    )
    report = guard.assess(requirements, docs, units)

    assert report.overall_verdict == GroundabilityVerdict.UNSUPPORTED
    assert report.can_proceed is False
    assert set(report.ungrounded_topics) == {
        "Annual leave entitlement",
        "Bonus payout formula",
    }


def test_langgraph_phase2_transitions_and_error_handling(tmp_path: Path) -> None:
    docs_dir = _create_sample_corpus(tmp_path)
    q1 = ClarifyingQuestion(
        question_id="q1",
        question="Should temporary contractors be included?",
        rationale="Audience clarification.",
        target_field="audience",
    )
    reqs = StructuredRequirements(
        domain_name="Workplace Policy",
        purpose="Answer internal leave questions.",
        scope="Annual leave and attendance rules.",
        audience="Employees",
        tone="Concise",
        must_answer_topics=["Annual leave entitlement"],
        must_refuse_topics=["Salary changes"],
    )
    ground_ok = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Annual leave entitlement",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["leave_policy.md"],
                evidence_summary="Supported by leave_policy.md.",
            )
        ],
    )

    stub = _StubGeminiClient(
        responses=[
            ClarifyingQuestionBatch(questions=[q1]),
            reqs,
            ground_ok,
        ]
    )
    settings = ForgeSettings(_env_file=None, gemini_api_key="test-key", gemini_model="gemini-3.8-flash")
    graph = build_factory_graph(
        settings=settings,
        gemini_client=stub,
        embedding_provider=_DeterministicEmbedder(),
    )

    brief = BriefInput(
        brief_text="Build an internal policy assistant grounded in our leave and attendance documents.",
        documents_dir=docs_dir,
    )

    # Pass 1: Ingest documents and pause with clarifying questions
    state1 = graph.invoke({"brief_input": brief, "stop_after_phase2": True})
    assert len(state1["documents"]) == 2
    assert state1["awaiting_user_answers"] is True
    assert len(state1["clarifying_questions"]) == 1
    assert "requirements" not in state1

    # Pass 2: Provide answer, synthesize requirements, and run Groundability Guard
    state2 = graph.invoke(
        {
            **state1,
            "clarifying_answers": [
                ClarifyingAnswer(question_id="q1", question=q1.question, answer="Full-time only.")
            ],
            "interview_completed": True,
            "stop_after_phase2": True,
        }
    )
    assert state2["awaiting_user_answers"] is False
    assert state2["requirements"].clarifying_qa[0].answer == "Full-time only."
    assert state2["groundability"].can_proceed is True
    assert state2["phase2_approved"] is True

    # Unsupported groundability must block generate_agent even when stop_after_phase2=False
    visited: list[str] = []
    blocked_report = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.UNSUPPORTED,
        can_proceed=False,
        approval_required=True,
        findings=[
            GroundabilityFinding(
                topic="Annual leave entitlement",
                verdict=GroundabilityVerdict.UNSUPPORTED,
                supporting_sources=[],
                evidence_summary="No evidence.",
                gap_description="Missing policy file.",
            )
        ],
        ungrounded_topics=["Annual leave entitlement"],
    )
    blocked_graph = build_factory_graph(
        settings=settings,
        gemini_client=_StubGeminiClient(responses=[reqs, blocked_report]),
        embedding_provider=_DeterministicEmbedder(),
        node_handlers={
            "generate_agent": lambda _s: visited.append("generate_agent") or {},
            "build_report": lambda _s: visited.append("build_report") or {"stop_reason": "blocked_by_groundability"},
        },
    )
    blocked_state = blocked_graph.invoke(
        {
            "brief_input": brief,
            "interview_completed": True,
            "clarifying_answers": [],
            "stop_after_phase2": False,
        }
    )
    assert visited == ["build_report"]
    assert blocked_state["phase2_approved"] is False
    assert blocked_state["stop_reason"] == "blocked_by_groundability"

    # Missing folder raises clear FileNotFoundError
    with pytest.raises(FileNotFoundError):
        graph.invoke(
            {
                "brief_input": BriefInput(
                    brief_text="Build an assistant for a missing directory.",
                    documents_dir=tmp_path / "missing_dir",
                ),
                "stop_after_phase2": True,
            }
        )

    # LLM failure surfaces clean RuntimeError without leaking keys
    failing_graph = build_factory_graph(
        settings=settings,
        gemini_client=_StubGeminiClient(error=RuntimeError("503 UNAVAILABLE")),
        embedding_provider=_DeterministicEmbedder(),
    )
    with pytest.raises(RuntimeError, match="Intake question generation failed"):
        failing_graph.invoke({"brief_input": brief, "stop_after_phase2": True})
