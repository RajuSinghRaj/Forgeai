"""Zero-cost tests for ForgeAI Phase B: Batched Hybrid-Judge Grading."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from forgeai.config import ForgeSettings
from forgeai.evaluation import HybridJudge, compute_scorecard, run_evaluation_suite
from forgeai.optimization.optimizer import CriticOptimizer
from forgeai.retrieval import load_document_folder
from forgeai.runner.agent_runner import AgentResponse
from forgeai.schemas import (
    ChunkingConfig,
    DeclarativeAgentDefinition,
    EvalCase,
    EvalCaseCategory,
    ExpectedBehavior,
    FailureClassification,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    JudgeBatchItemVerdict,
    JudgeLLMBatchVerdict,
    OptimizationProposalDraft,
    OutputContract,
    RetrievalConfig,
    RetrievedEvidence,
    StructuredRequirements,
    TargetedProposalItem,
    ToolSelection,
)


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


class _MockBatchJudgeGemini:
    """Mock GeminiClient tracking individual vs batch judge calls without network access."""

    def __init__(self, batch_fn, text_fn=None) -> None:
        self._batch_fn = batch_fn
        self._text_fn = text_fn or (lambda *_a, **_kw: "Inspection is every 30 days [policy.md | document].")
        self.batch_prompts: list[str] = []
        self.batch_case_id_lists: list[list[str]] = []
        self.single_judge_calls = 0

    def generate_text(self, prompt: str, **_kwargs: Any) -> str:
        return self._text_fn(prompt)

    def generate_structured(self, prompt: str, response_schema: type[Any], **_kwargs: Any) -> Any:
        if response_schema.__name__ == "JudgeLLMVerdict":
            self.single_judge_calls += 1
            raise AssertionError("Expected batched judging, not single JudgeLLMVerdict.")
        if response_schema is JudgeLLMBatchVerdict:
            self.batch_prompts.append(prompt)
            match = re.search(r"expected case_ids=(\[[^\n]+\])", prompt)
            assert match is not None
            case_ids = json.loads(match.group(1))
            self.batch_case_id_lists.append(case_ids)
            return self._batch_fn(case_ids, len(self.batch_prompts))
        return self._batch_fn([], len(self.batch_prompts), schema=response_schema)


def _make_cases(count: int) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for i in range(1, count + 1):
        if i % 4 == 0:
            cases.append(
                EvalCase(
                    case_id=f"case_{i:02d}",
                    topic="Negotiation",
                    category=EvalCaseCategory.OUT_OF_SCOPE,
                    expected_behavior=ExpectedBehavior.REFUSE,
                    question=f"Can you discount price by {i}%?",
                )
            )
        else:
            cases.append(
                EvalCase(
                    case_id=f"case_{i:02d}",
                    topic="Inspection",
                    category=EvalCaseCategory.ANSWERABLE,
                    expected_behavior=ExpectedBehavior.ANSWER,
                    expected_facts=["30 days"],
                    source_documents=["policy.md"],
                    evaluation_criteria=["Mention 30 days", "Cite policy.md"],
                    question=f"What is inspection rule #{i}?",
                )
            )
    return cases


def _make_response(case: EvalCase, *, fabricate_source: bool = False) -> AgentResponse:
    ev = [
        RetrievedEvidence(
            chunk_id=f"chk_{case.case_id}",
            doc_id="doc1",
            source_file="policy.md",
            unit_label="section_1",
            text_excerpt="Inspection occurs every 30 days.",
        )
    ]
    if case.expected_behavior == ExpectedBehavior.REFUSE:
        return AgentResponse(
            answer="I cannot answer this request based on the provided documents and scope.",
            refused=True,
            evidence=ev,
            citations=[],
            tools_used=["retrieve_documents"],
        )
    src = "fake_invented.pdf" if fabricate_source else "policy.md"
    return AgentResponse(
        answer=f"Inspection occurs every 30 days [{src} | section_1].",
        refused=False,
        evidence=ev,
        citations=[f"{src} (section_1)"],
        tools_used=["retrieve_documents"],
    )


def _valid_item(case_id: str, **overrides: Any) -> JudgeBatchItemVerdict:
    defaults = {
        "case_id": case_id,
        "grounded": True,
        "grounding_score": 1.0,
        "correctness_score": 1.0,
        "addresses_question": True,
        "refusal_appropriate": True,
        "omitted_facts": [],
        "hallucinated_claims": [],
        "unsupported_verbosity": False,
        "suggested_failure_classification": FailureClassification.NONE,
        "reasoning": f"Evaluated {case_id} accurately.",
    }
    defaults.update(overrides)
    return JudgeBatchItemVerdict(**defaults)


def test_20_cases_make_4_requests_and_22_cases_make_5_requests() -> None:
    contract = OutputContract(response_format="markdown", require_citations=True)

    def ok_batch(case_ids: list[str], _idx: int) -> JudgeLLMBatchVerdict:
        return JudgeLLMBatchVerdict(verdicts=[_valid_item(cid) for cid in case_ids])

    # 20 cases -> exactly 4 batch requests of 5 cases each
    client_20 = _MockBatchJudgeGemini(ok_batch)
    judge_20 = HybridJudge(gemini_client=client_20, known_sources={"policy.md"})
    cases_20 = _make_cases(20)
    res_20 = judge_20.evaluate_batch([(c, _make_response(c)) for c in cases_20], contract)
    assert len(res_20) == 20
    assert len(client_20.batch_case_id_lists) == 4
    assert [len(b) for b in client_20.batch_case_id_lists] == [5, 5, 5, 5]
    assert all(r.passed for r in res_20)

    # 22 cases -> exactly 5 batch requests (4 batches of 5 + last batch of 2)
    client_22 = _MockBatchJudgeGemini(ok_batch)
    judge_22 = HybridJudge(gemini_client=client_22, known_sources={"policy.md"})
    cases_22 = _make_cases(22)
    res_22 = judge_22.evaluate_batch([(c, _make_response(c)) for c in cases_22], contract)
    assert len(res_22) == 22
    assert len(client_22.batch_case_id_lists) == 5
    assert [len(b) for b in client_22.batch_case_id_lists] == [5, 5, 5, 5, 2]
    assert [r.case_id for r in res_22] == [c.case_id for c in cases_22]


def test_out_of_order_verdicts_matched_by_case_id_and_independent_scoring() -> None:
    contract = OutputContract(response_format="markdown", require_citations=True)
    cases = _make_cases(5)

    def reversed_batch(case_ids: list[str], _idx: int) -> JudgeLLMBatchVerdict:
        # Return verdicts in reverse order, with case_02 failing hallucination & others passing
        verdicts = []
        for cid in reversed(case_ids):
            if cid == "case_02":
                verdicts.append(
                    _valid_item(
                        cid,
                        grounded=False,
                        grounding_score=0.25,
                        correctness_score=0.4,
                        hallucinated_claims=["Invented figure"],
                        suggested_failure_classification=FailureClassification.HALLUCINATION,
                        reasoning="Hallucinated in case_02.",
                    )
                )
            else:
                verdicts.append(_valid_item(cid, correctness_score=0.95, reasoning=f"Clean {cid}"))
        return JudgeLLMBatchVerdict(verdicts=verdicts)

    client = _MockBatchJudgeGemini(reversed_batch)
    judge = HybridJudge(gemini_client=client, known_sources={"policy.md"})
    # Make case_03 cite a fabricated source to confirm deterministic checks still run per case
    items = [
        (c, _make_response(c, fabricate_source=(c.case_id == "case_03")))
        for c in cases
    ]
    results = judge.evaluate_batch(items, contract)
    by_id = {r.case_id: r for r in results}

    assert [r.case_id for r in results] == ["case_01", "case_02", "case_03", "case_04", "case_05"]
    assert by_id["case_01"].passed is True
    assert by_id["case_02"].passed is False
    assert by_id["case_02"].failure_classification == FailureClassification.HALLUCINATION
    assert by_id["case_03"].passed is False
    assert by_id["case_03"].citation_valid is False
    assert by_id["case_03"].disagreements  # LLM marked grounded, deterministic override caught invented source
    assert by_id["case_04"].passed is True
    assert by_id["case_05"].passed is True


@pytest.mark.parametrize(
    "bad_mode",
    ["missing_id", "duplicate_id", "unknown_id", "malformed_whitespace", "malformed_object"],
)
def test_invalid_batch_responses_rejected_without_fabricated_pass_or_extra_calls(
    bad_mode: str,
) -> None:
    contract = OutputContract(response_format="markdown", require_citations=True)
    cases = _make_cases(5)

    def bad_batch(case_ids: list[str], _idx: int) -> Any:
        if bad_mode == "missing_id":
            return JudgeLLMBatchVerdict(verdicts=[_valid_item(cid) for cid in case_ids[:-1]])
        if bad_mode == "duplicate_id":
            dup = case_ids[:-1] + [case_ids[0]]
            return JudgeLLMBatchVerdict(verdicts=[_valid_item(cid) for cid in dup])
        if bad_mode == "unknown_id":
            unk = case_ids[:-1] + ["case_999"]
            return JudgeLLMBatchVerdict(verdicts=[_valid_item(cid) for cid in unk])
        if bad_mode == "malformed_whitespace":
            ws = case_ids[:-1] + [f" {case_ids[-1]} "]
            return JudgeLLMBatchVerdict(verdicts=[_valid_item(cid) for cid in ws])
        # malformed_object
        return {"not_a_schema": True}

    client = _MockBatchJudgeGemini(bad_batch)
    judge = HybridJudge(gemini_client=client, known_sources={"policy.md"})
    results = judge.evaluate_batch([(c, _make_response(c)) for c in cases], contract)

    assert len(client.batch_case_id_lists) == 1  # No extra/unbounded retries on invalid batch
    assert len(results) == 5
    for r in results:
        assert r.passed is False
        assert r.grounding_score == 0.0
        assert r.correctness_score == 0.0
        assert r.failure_classification != FailureClassification.NONE
        assert any("rejected" in d.lower() for d in r.disagreements)


def test_configurable_batch_size_and_optimizer_scorecard_compatibility(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text("Inspection occurs every 30 days.", encoding="utf-8")
    docs, units = load_document_folder(docs_dir)

    settings = ForgeSettings(_env_file=None, judge_batch_size=4)
    assert settings.judge_batch_size == 4

    req = StructuredRequirements(
        domain_name="Ops",
        purpose="Check inspection policy.",
        scope="Policy",
        audience="Staff",
        tone="Concise",
        must_answer_topics=["Inspection"],
        must_refuse_topics=["Negotiation"],
    )
    gb = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        approval_required=False,
        findings=[
            GroundabilityFinding(
                topic="Inspection",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["policy.md"],
                evidence_summary="Supported in policy.md.",
            )
        ],
    )
    agent = DeclarativeAgentDefinition(
        agent_id="ops_batch_agent",
        name="Ops Batch Agent",
        version=1,
        domain_summary="Grounded ops assistant.",
        system_prompt="Answer strictly from policy.md and cite it.",
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50),
        retrieval=RetrievalConfig(collection_name="col_batch", top_k=3),
        tools=[ToolSelection(name="retrieve_documents", purpose="Retrieve policy passages")],
        output_contract=OutputContract(response_format="markdown", require_citations=True),
        must_answer_topics=req.must_answer_topics,
        must_refuse_topics=req.must_refuse_topics,
        source_documents=["policy.md"],
        approved_requirements=req,
    )

    cases_6 = _make_cases(6)  # configurable batch_size=4 -> 2 batch judge requests ([4, 2])
    baseline_calls: list[list[str]] = []

    def batch_handler(case_ids: list[str], _idx: int, schema=None) -> Any:
        if schema is OptimizationProposalDraft:
            return OptimizationProposalDraft(
                actionable=True,
                diagnosis_summary="Increase top_k to improve context coverage.",
                proposed_changes=[
                    TargetedProposalItem(
                        target_component="system_prompt",
                        failure_driver=FailureClassification.PROMPT_DEFECT,
                        rationale="Add explicit citation instruction.",
                        summary="Add explicit citation instruction.",
                        prompt_instruction_addition="Always cite [policy.md | document].",
                    )
                ],
            )
        baseline_calls.append(case_ids)
        return JudgeLLMBatchVerdict(verdicts=[_valid_item(cid) for cid in case_ids])

    client = _MockBatchJudgeGemini(
        batch_handler,
        text_fn=lambda p, **_kw: (
            "I cannot answer this request based on the provided documents and scope."
            if "discount" in p.lower()
            else "Inspection is every 30 days [policy.md | document]."
        ),
    )
    baseline_results, baseline_sc = run_evaluation_suite(
        agent,
        cases_6,
        units,
        gemini_client=client,
        embedding_provider=_DeterministicEmbedder(),
        judge_batch_size=settings.judge_batch_size,
    )
    assert len(baseline_results) == 6
    assert baseline_sc.pass_rate == 1.0
    assert [len(b) for b in baseline_calls] == [4, 2]

    # Force 1 baseline failure so CriticOptimizer runs 1 optimization pass using batch judging
    baseline_results_with_fail = list(baseline_results)
    baseline_results_with_fail[0] = baseline_results_with_fail[0].model_copy(
        update={
            "passed": False,
            "failure_classification": FailureClassification.PROMPT_DEFECT,
            "correctness_score": 0.4,
        }
    )
    baseline_sc_fail = compute_scorecard(baseline_results_with_fail, iteration=0)

    baseline_calls.clear()
    optimizer = CriticOptimizer(
        gemini_client=client,
        max_iterations=3,
        target_pass_rate=0.90,
        judge_batch_size=settings.judge_batch_size,
    )
    opt_res = optimizer.optimize(
        run_id="batch_opt_check",
        requirements=req,
        groundability=gb,
        baseline_agent=agent,
        eval_cases=cases_6,
        baseline_results=baseline_results_with_fail,
        baseline_scorecard=baseline_sc_fail,
        documents=docs,
        units=units,
        artifacts_dir=tmp_path / "artifacts",
        embedding_provider=_DeterministicEmbedder(),
    )
    assert opt_res.final_report.final_pass_rate == 1.0
    assert len(opt_res.optimization_logs) == 1
    assert [len(b) for b in baseline_calls] == [4, 2]
