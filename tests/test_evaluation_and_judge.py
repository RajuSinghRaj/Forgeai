"""Deterministic Phase 4 unit tests for EvalGenerator, HybridJudge, persistence, and LangGraph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from forgeai.config import ForgeSettings
from forgeai.evaluation import (
    EvalGenerator,
    HybridJudge,
    InsufficientEvalCasesError,
    compute_scorecard,
    save_evaluation_artifacts,
)
from forgeai.retrieval import load_document_folder
from forgeai.runner.agent_runner import AgentResponse
from forgeai.schemas import (
    BriefInput,
    ChunkingConfig,
    DeclarativeAgentDefinition,
    EvalCase,
    EvalCaseCategory,
    EvalCaseResult,
    EvalSuiteDraft,
    ExpectedBehavior,
    FailureClassification,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    JudgeLLMVerdict,
    OutputContract,
    RetrievalConfig,
    RetrievedEvidence,
    StructuredRequirements,
    ToolSelection,
)
from forgeai.workflow import build_factory_graph


class _DeterministicEmbedder:
    def __init__(self, dim: int = 32) -> None:
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


class _ScriptedGeminiClient:
    """Mock GeminiClient supporting both `generate_structured` and `generate_text`."""

    def __init__(
        self,
        structured_responses: list[Any] | None = None,
        text_responses: list[str] | None = None,
    ) -> None:
        self._structured = list(structured_responses or [])
        self._texts = list(text_responses or [])
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, response_schema: type[Any], **_kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if not self._structured:
            raise AssertionError(f"No stubbed structured response left for {response_schema.__name__}.")
        item = self._structured.pop(0)
        assert isinstance(item, response_schema)
        return item

    def generate_text(self, prompt: str, **_kwargs: Any) -> str:
        self.prompts.append(prompt)
        if not self._texts:
            raise AssertionError("No stubbed text response left.")
        return self._texts.pop(0)


def _create_corpus(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text(
        "Inspection occurs every 30 days. Escalations go to duty_lead within 24 hours.",
        encoding="utf-8",
    )
    (docs_dir / "catalog.csv").write_text(
        "item_id,name,lead_days,stock\n"
        "A-10,Hydraulic valve,14,12\n"
        "B-20,Pressure gauge,7,5\n"
        "C-30,Thermal seal,21,0\n",
        encoding="utf-8",
    )
    return docs_dir


def _sample_requirements() -> StructuredRequirements:
    return StructuredRequirements(
        domain_name="Operations Support",
        purpose="Answer grounded procedure and catalog questions.",
        scope="Inspection intervals, escalations, and catalog item availability.",
        audience="Operations staff",
        tone="Concise and factual",
        must_answer_topics=["Inspection intervals", "Catalog lead times"],
        must_refuse_topics=["Pricing negotiations"],
    )


def _sample_agent() -> DeclarativeAgentDefinition:
    return DeclarativeAgentDefinition(
        agent_id="ops_agent_01",
        name="Ops Agent",
        version=1,
        domain_summary="Grounded operations assistant.",
        system_prompt="Answer strictly from supplied documents and cite source_file.",
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=80),
        retrieval=RetrievalConfig(collection_name="col_ops_01", top_k=4),
        tools=[ToolSelection(name="retrieve_documents", purpose="Search corpus")],
        output_contract=OutputContract(
            response_format="markdown",
            require_citations=True,
            citation_format="[source_file | unit]",
        ),
        source_documents=["policy.md", "catalog.csv"],
        must_answer_topics=["Inspection intervals", "Catalog lead times"],
        must_refuse_topics=["Pricing negotiations"],
    )


def _build_twenty_two_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    answerable_topics = [
        ("Inspection cadence", "How often does standard equipment inspection occur?", ["30 days"], ["policy.md"]),
        ("Escalation SLA", "Within how many hours must escalations reach the duty lead?", ["24 hours"], ["policy.md"]),
        ("Escalation contact", "Who receives operational escalations?", ["duty_lead"], ["policy.md"]),
        ("Item A-10 name", "What is the catalog description for item A-10?", ["Hydraulic valve"], ["catalog.csv"]),
        ("Item A-10 lead", "What is the lead time in days for item A-10?", ["14"], ["catalog.csv"]),
        ("Item A-10 stock", "How many units of A-10 are in stock?", ["12"], ["catalog.csv"]),
        ("Item B-20 name", "What product corresponds to item ID B-20?", ["Pressure gauge"], ["catalog.csv"]),
        ("Item B-20 lead", "How many lead days are listed for B-20?", ["7"], ["catalog.csv"]),
        ("Item B-20 stock", "What is the stock quantity for Pressure gauge B-20?", ["5"], ["catalog.csv"]),
        ("Item C-30 name", "Which part has item ID C-30?", ["Thermal seal"], ["catalog.csv"]),
        ("Item C-30 lead", "What is the replenishment lead time for Thermal seal C-30?", ["21"], ["catalog.csv"]),
        ("Item C-30 stock", "Is Thermal seal C-30 currently in stock?", ["0"], ["catalog.csv"]),
    ]
    for idx, (topic, q, facts, srcs) in enumerate(answerable_topics, start=1):
        cases.append(
            EvalCase(
                case_id=f"ans_{idx:02d}",
                question=q,
                category=EvalCaseCategory.ANSWERABLE,
                expected_behavior=ExpectedBehavior.ANSWER,
                expected_facts=facts,
                source_documents=srcs,
                evidence_references=[f"{srcs[0]} | unit"],
                topic=topic,
                evaluation_criteria=["Factually accurate", "Cites source"],
            )
        )

    oos_questions = [
        ("Pricing negotiations", "Can you negotiate a 15% bulk discount on hydraulic valves?"),
        ("Vendor bank details", "What are the supplier wire transfer bank account numbers?"),
        ("Personal HR leave", "How many vacation days do operations managers receive?"),
        ("Warranty litigation", "Draft a legal demand letter for delayed valve shipments."),
    ]
    for idx, (topic, q) in enumerate(oos_questions, start=1):
        cases.append(
            EvalCase(
                case_id=f"oos_{idx:02d}",
                question=q,
                category=EvalCaseCategory.OUT_OF_SCOPE,
                expected_behavior=ExpectedBehavior.REFUSE,
                topic=topic,
                expected_refusal_behavior="Refuse and explain limitation.",
                evaluation_criteria=["Explicit refusal without fabrication"],
            )
        )

    edge_cases = [
        (
            "Cross-document escalation & out-of-stock",
            "If C-30 has 0 stock and needs escalation, who do we notify and what is its lead time?",
            ["duty_lead", "24 hours", "21"],
            ["policy.md", "catalog.csv"],
            ExpectedBehavior.ANSWER,
        ),
        (
            "Cross-document inspection & gauge",
            "Does the 30-day inspection schedule cover Pressure gauge B-20 with 7 lead days?",
            ["30 days", "7"],
            ["policy.md", "catalog.csv"],
            ExpectedBehavior.ANSWER,
        ),
        (
            "Near-miss item ID",
            "What is the stock count for item A-20 in the catalog?",
            [],
            [],
            ExpectedBehavior.REFUSE,
        ),
        (
            "Ambiguous assumption",
            "Since all seals ship in 3 days, when will C-30 arrive?",
            ["21"],
            ["catalog.csv"],
            ExpectedBehavior.ANSWER,
        ),
    ]
    for idx, (topic, q, facts, srcs, behavior) in enumerate(edge_cases, start=1):
        cases.append(
            EvalCase(
                case_id=f"edge_{idx:02d}",
                question=q,
                category=EvalCaseCategory.EDGE_CASE,
                expected_behavior=behavior,
                expected_facts=facts,
                source_documents=srcs,
                topic=topic,
                evaluation_criteria=["Handles ambiguity or cross-doc synthesis"],
            )
        )
    return cases


def test_eval_generator_produces_20_plus_cases_and_rejects_duplicates_and_fabricated_sources(
    tmp_path: Path,
) -> None:
    docs_dir = _create_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    valid_20 = _build_twenty_two_cases()

    # Add 1 exact duplicate, 1 near-duplicate, and 1 fabricated-source case
    dup = valid_20[0].model_copy(update={"case_id": "dup_1"})
    near_dup = valid_20[1].model_copy(
        update={
            "case_id": "dup_2",
            "question": "Within how many hours must escalations reach the duty lead please?",
        }
    )
    fabricated = EvalCase(
        case_id="fab_1",
        question="What does the secret annex say about turbine cooling?",
        category=EvalCaseCategory.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_facts=["45 PSI"],
        source_documents=["fabricated_annex.pdf"],
        topic="Cooling",
    )

    stub = _ScriptedGeminiClient(
        structured_responses=[EvalSuiteDraft(cases=valid_20 + [dup, near_dup, fabricated])]
    )
    generator = EvalGenerator(gemini_client=stub, min_cases=20)
    cases, meta = generator.generate(_sample_requirements(), _sample_agent(), docs, units)

    assert len(cases) == 20
    assert meta.accepted_cases == 20
    assert meta.rejected_duplicates == 2
    assert meta.rejected_fabricated_sources == 1
    assert meta.category_counts == {"answerable": 12, "out_of_scope": 4, "edge_case": 4}
    assert any(c.is_cross_document for c in cases)

    # Incomplete set (< 20 valid cases) must fail honestly instead of padding
    short_stub = _ScriptedGeminiClient(structured_responses=[EvalSuiteDraft(cases=valid_20[:10])])
    short_gen = EvalGenerator(gemini_client=short_stub, min_cases=20)
    with pytest.raises(InsufficientEvalCasesError) as exc_info:
        short_gen.generate(_sample_requirements(), _sample_agent(), docs, units)
    assert exc_info.value.metadata.shortfall == 10


def test_hybrid_judge_grounding_citations_contract_refusals_and_overrides() -> None:
    known = {"policy.md", "catalog.csv"}
    contract = OutputContract(
        response_format="json",
        require_citations=True,
        required_fields=["answer", "source"],
        max_response_words=80,
    )
    ans_case = _build_twenty_two_cases()[0]
    oos_case = _build_twenty_two_cases()[12]

    ev_policy = RetrievedEvidence(
        chunk_id="c1",
        doc_id="d1",
        source_file="policy.md",
        unit_label="document",
        text_excerpt="Inspection occurs every 30 days.",
    )

    # 1. Passing JSON + valid citation
    judge_ok = HybridJudge(
        gemini_client=_ScriptedGeminiClient(
            structured_responses=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=1.0,
                    correctness_score=1.0,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Accurate and grounded in policy.md.",
                )
            ]
        ),
        known_sources=known,
    )
    res_ok = judge_ok.evaluate_case(
        ans_case,
        AgentResponse(
            answer=json.dumps({"answer": "Every 30 days.", "source": "policy.md"}),
            refused=False,
            evidence=[ev_policy],
        ),
        contract,
    )
    assert res_ok.passed is True
    assert res_ok.failure_classification == FailureClassification.NONE
    assert res_ok.contract_compliant is True
    assert res_ok.citation_valid is True

    # 2. Fabricated citation must override LLM grounded=True and record disagreement
    judge_fab = HybridJudge(
        gemini_client=_ScriptedGeminiClient(
            structured_responses=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=0.95,
                    correctness_score=0.9,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="LLM missed fabricated filename.",
                )
            ]
        ),
        known_sources=known,
    )
    res_fab = judge_fab.evaluate_case(
        ans_case,
        AgentResponse(
            answer=json.dumps({"answer": "Every 30 days.", "source": "invented_manual.pdf"}),
            refused=False,
            evidence=[ev_policy],
        ),
        contract,
    )
    assert res_fab.passed is False
    assert res_fab.citation_valid is False
    assert res_fab.failure_classification == FailureClassification.HALLUCINATION
    assert res_fab.disagreements

    # 3. Output-contract violation (missing required JSON field) -> PROMPT_DEFECT
    judge_contract = HybridJudge(
        gemini_client=_ScriptedGeminiClient(
            structured_responses=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=0.9,
                    correctness_score=0.9,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Factually correct.",
                )
            ]
        ),
        known_sources=known,
    )
    res_contract = judge_contract.evaluate_case(
        ans_case,
        AgentResponse(
            answer="Inspection happens every 30 days [policy.md].",
            refused=False,
            evidence=[ev_policy],
        ),
        contract,
    )
    assert res_contract.passed is False
    assert res_contract.contract_compliant is False
    assert res_contract.failure_classification == FailureClassification.PROMPT_DEFECT

    # 4. Incorrect refusal on answerable case (with evidence retrieved) -> WRONG_REFUSAL
    judge_wrong_ref = HybridJudge(
        gemini_client=_ScriptedGeminiClient(
            structured_responses=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=0.8,
                    correctness_score=0.0,
                    addresses_question=False,
                    refusal_appropriate=False,
                    reasoning="Agent refused despite relevant evidence.",
                )
            ]
        ),
        known_sources=known,
    )
    res_wrong_ref = judge_wrong_ref.evaluate_case(
        ans_case,
        AgentResponse(
            answer="I cannot answer this request based on the provided documents and scope.",
            refused=True,
            evidence=[ev_policy],
        ),
        contract,
    )
    assert res_wrong_ref.passed is False
    assert res_wrong_ref.refusal_correct is False
    assert res_wrong_ref.failure_classification == FailureClassification.WRONG_REFUSAL

    # 5. Correct refusal on out-of-scope case -> PASS
    judge_ref = HybridJudge(
        gemini_client=_ScriptedGeminiClient(
            structured_responses=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=1.0,
                    correctness_score=1.0,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Appropriately refused out-of-scope pricing request.",
                )
            ]
        ),
        known_sources=known,
    )
    res_oos = judge_ref.evaluate_case(
        oos_case,
        AgentResponse(
            answer="I cannot answer this request based on the provided documents and scope.",
            refused=True,
            evidence=[],
        ),
        contract,
    )
    assert res_oos.passed is True
    assert res_oos.refusal_correct is True


def test_scorecard_metrics_and_artifact_persistence_collision_protection(tmp_path: Path) -> None:
    cases = _build_twenty_two_cases()[:4]
    results = [
        EvalCaseResult(
            case_id="c1",
            question="Q1?",
            category=EvalCaseCategory.ANSWERABLE,
            expected_behavior=ExpectedBehavior.ANSWER,
            agent_answer="Ans [policy.md]",
            refused=False,
            cited_sources=["policy.md"],
            grounding_score=1.0,
            correctness_score=1.0,
            refusal_correct=True,
            citation_valid=True,
            contract_compliant=True,
            passed=True,
            failure_classification=FailureClassification.NONE,
            judge_reasoning="Good.",
        ),
        EvalCaseResult(
            case_id="c2",
            question="Q2?",
            category=EvalCaseCategory.OUT_OF_SCOPE,
            expected_behavior=ExpectedBehavior.REFUSE,
            agent_answer="I cannot answer.",
            refused=True,
            grounding_score=1.0,
            correctness_score=1.0,
            refusal_correct=True,
            citation_valid=True,
            contract_compliant=True,
            passed=True,
            failure_classification=FailureClassification.NONE,
            judge_reasoning="Good refusal.",
        ),
        EvalCaseResult(
            case_id="c3",
            question="Q3?",
            category=EvalCaseCategory.EDGE_CASE,
            expected_behavior=ExpectedBehavior.ANSWER,
            agent_answer="Bad guess.",
            refused=False,
            grounding_score=0.2,
            correctness_score=0.1,
            refusal_correct=True,
            citation_valid=False,
            contract_compliant=False,
            hallucinated_claims=["Fake claim"],
            passed=False,
            failure_classification=FailureClassification.HALLUCINATION,
            judge_reasoning="Hallucinated.",
        ),
    ]
    sc = compute_scorecard(results, iteration=0)
    assert sc.total_cases == 3
    assert sc.passed_cases == 2
    assert sc.failed_cases == 1
    assert sc.answerable_accuracy == 1.0
    assert sc.out_of_scope_refusal_accuracy == 1.0
    assert sc.edge_case_accuracy == 0.0
    assert sc.citation_accuracy == round(2 / 3, 4)
    assert sc.contract_compliance_rate == round(2 / 3, 4)

    paths1 = save_evaluation_artifacts(
        run_label="ops_run",
        eval_cases=cases,
        eval_results=results,
        scorecard=sc,
        generation_metadata=None,
        artifacts_dir=tmp_path,
    )
    paths2 = save_evaluation_artifacts(
        run_label="ops_run",
        eval_cases=cases,
        eval_results=results,
        scorecard=sc,
        generation_metadata=None,
        artifacts_dir=tmp_path,
    )
    assert paths1.cases_path.exists()
    assert paths1.results_path.exists()
    assert paths1.scorecard_path.exists()
    assert paths1.metadata_path.exists()
    assert paths2.cases_path != paths1.cases_path
    assert "_1_" in paths2.cases_path.name


def test_langgraph_phase4_execution_and_incomplete_suite_error(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path)
    brief = BriefInput(
        brief_text="Build an operations support agent grounded in policy.md and catalog.csv.",
        documents_dir=docs_dir,
    )
    reqs = _sample_requirements()
    agent = _sample_agent()
    ground_ok = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Inspection intervals",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["policy.md"],
                evidence_summary="Supported.",
            ),
            GroundabilityFinding(
                topic="Catalog lead times",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["catalog.csv"],
                evidence_summary="Supported.",
            ),
        ],
    )

    cases_20 = _build_twenty_two_cases()
    text_answers = []
    verdicts = []
    for case in cases_20:
        if case.expected_behavior == ExpectedBehavior.REFUSE:
            text_answers.append("I cannot answer this request based on the provided documents and scope.")
            verdicts.append(
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=1.0,
                    correctness_score=1.0,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Refused appropriately.",
                )
            )
        else:
            src = case.source_documents[0]
            text_answers.append(f"Answer: {' '.join(case.expected_facts)} [{src} | unit].")
            verdicts.append(
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=1.0,
                    correctness_score=1.0,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Grounded and accurate.",
                )
            )

    stub = _ScriptedGeminiClient(
        structured_responses=[EvalSuiteDraft(cases=cases_20)] + verdicts,
        text_responses=text_answers,
    )
    settings = ForgeSettings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="gemini-3.8-flash",
        artifacts_dir=tmp_path / "artifacts",
    )
    graph = build_factory_graph(
        settings=settings,
        gemini_client=stub,
        embedding_provider=_DeterministicEmbedder(),
    )

    state = graph.invoke(
        {
            "brief_input": brief,
            "requirements": reqs,
            "groundability": ground_ok,
            "agent_definition": agent,
            "interview_completed": True,
            "stop_after_phase4": True,
        }
    )

    assert state["phase4_status"] == "completed"
    assert len(state["eval_cases"]) == 20
    assert len(state["eval_results"]) == 20
    assert state["scorecards"][-1].total_cases == 20
    assert state["scorecards"][-1].pass_rate == 1.0
    for path_str in state["eval_artifact_paths"].values():
        assert Path(path_str).exists()

    # Incomplete evaluation generation (< 20 valid cases) raises InsufficientEvalCasesError
    bad_graph = build_factory_graph(
        settings=settings,
        gemini_client=_ScriptedGeminiClient(
            structured_responses=[EvalSuiteDraft(cases=cases_20[:5])]
        ),
        embedding_provider=_DeterministicEmbedder(),
    )
    with pytest.raises(InsufficientEvalCasesError):
        bad_graph.invoke(
            {
                "brief_input": brief,
                "requirements": reqs,
                "groundability": ground_ok,
                "agent_definition": agent,
                "interview_completed": True,
                "stop_after_phase4": True,
            }
        )

