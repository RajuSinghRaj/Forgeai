"""Domain-agnostic tests for non-JSON contracts, refusal detection, CSV chunk consistency, judge taxonomy, and UI reporting."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from forgeai.evaluation import HybridJudge, compute_scorecard, run_evaluation_suite
from forgeai.optimization.optimizer import CriticOptimizer
from forgeai.retrieval import ChromaVectorStore, load_document_folder, rebatch_csv_units
from forgeai.runner.agent_runner import AgentResponse, DeclarativeAgentRunner, _detect_refusal
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
    JudgeBatchItemVerdict,
    JudgeLLMBatchVerdict,
    JudgeLLMVerdict,
    OptimizationProposalDraft,
    OutputContract,
    RetrievalConfig,
    RetrievedEvidence,
    StructuredRequirements,
    TargetedProposalItem,
    ToolSelection,
)
from forgeai.tools.registry import build_default_registry


class _HashEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "hash-audit-embedder"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2] or ["empty"]:
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _ScriptedGemini:
    def __init__(self, structured: list[Any] | None = None, texts: list[str] | None = None) -> None:
        self._structured = list(structured or [])
        self._texts = list(texts or [])
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, schema: type[Any], **_kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if not self._structured:
            raise AssertionError(f"Unexpected structured call for {schema.__name__}.")
        item = self._structured.pop(0)
        assert isinstance(item, schema)
        return item

    def generate_text(self, prompt: str, **_kwargs: Any) -> str:
        self.prompts.append(prompt)
        if self._texts:
            return self._texts.pop(0)
        return "Grounded response [policy.md | document]."


def _build_agent(
    *,
    agent_id: str = "synth_agent_01",
    domain: str = "Maritime Port",
    collection: str = "col_synth_01",
    sources: list[str] | None = None,
    csv_rows_per_unit: int = 1,
    top_k: int = 2,
    response_format: str = "Markdown bulleted or structured text.",
    required_fields: list[str] | None = None,
    response_sections: list[str] | None = None,
) -> DeclarativeAgentDefinition:
    return DeclarativeAgentDefinition(
        agent_id=agent_id,
        name=f"{domain} Assistant",
        version=1,
        domain_summary=f"Assistant for {domain}.",
        system_prompt="Answer strictly from supplied documents and cite sources.",
        chunking=ChunkingConfig(
            chunk_size=600,
            chunk_overlap=100,
            csv_rows_per_unit=csv_rows_per_unit,
        ),
        retrieval=RetrievalConfig(collection_name=collection, top_k=top_k),
        tools=[
            ToolSelection(name="retrieve_documents", purpose="Retrieve documents."),
            ToolSelection(name="inspect_csv_records", purpose="Inspect CSV tables."),
        ],
        output_contract=OutputContract(
            response_format=response_format,
            require_citations=True,
            citation_format="[Source: <filename>]",
            refusal_message_template=(
                "I am sorry, but I cannot assist with that request as it falls outside documented scope."
            ),
            response_sections=response_sections or ["Direct Answer", "Details & Rules", "Source References"],
            required_fields=required_fields if required_fields is not None else ["answer_summary", "citations"],
        ),
        source_documents=sources or ["protocol.md"],
        must_answer_topics=[domain],
        must_refuse_topics=["Personal records"],
    )


def test_markdown_and_strict_json_output_contract_validation() -> None:
    """Markdown contracts recognize normalized headings & citations; JSON remains strict."""
    ev = RetrievedEvidence(
        chunk_id="d1:document:c0",
        doc_id="d1",
        source_file="radiology_sop.pdf",
        unit_label="page_1",
        text_excerpt="Contrast screening requires eGFR check within 48 hours of MRI.",
    )
    case = EvalCase(
        case_id="rad_01",
        question="When must eGFR screening occur before contrast MRI?",
        category=EvalCaseCategory.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_facts=["eGFR check within 48 hours"],
        source_documents=["radiology_sop.pdf"],
        topic="Radiology Screening",
    )

    # 1. Markdown contract with snake_case required_fields (`answer_summary`, `citations`) passes
    #    when answer uses human headings (`### Direct Answer`, `### Source References`) & citations.
    md_contract = OutputContract(
        response_format="Markdown bulleted or structured text.",
        require_citations=True,
        citation_format="[Source: <filename>]",
        response_sections=["Direct Answer", "Details & Rules", "Source References"],
        required_fields=["answer_summary", "citations"],
    )
    md_judge = HybridJudge(
        gemini_client=_ScriptedGemini(
            structured=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=1.0,
                    correctness_score=1.0,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Accurate radiology protocol answer.",
                )
            ]
        ),
        known_sources={"radiology_sop.pdf"},
    )
    md_answer = (
        "### Direct Answer\n"
        "Contrast screening requires an eGFR check within 48 hours [Source: radiology_sop.pdf].\n\n"
        "### Source References\n"
        "* [Source: radiology_sop.pdf]"
    )
    res_md = md_judge.evaluate_case(
        case,
        AgentResponse(answer=md_answer, refused=False, evidence=[ev]),
        md_contract,
    )
    assert res_md.passed is True
    assert res_md.contract_compliant is True
    assert res_md.failure_classification == FailureClassification.NONE

    # 2. Markdown contract STILL rejects missing genuinely mandatory domain field (e.g. `escalation_contact`)
    strict_md_contract = OutputContract(
        response_format="Markdown structured text",
        require_citations=True,
        required_fields=["escalation_contact", "citations"],
    )
    det_missing = md_judge.run_deterministic_checks(
        case,
        AgentResponse(answer=md_answer, refused=False, evidence=[ev]),
        strict_md_contract,
        known_sources={"radiology_sop.pdf"},
    )
    assert det_missing.contract_compliant is False
    assert "escalation_contact" in det_missing.issues[0]

    # 3. Strict JSON contract rejects non-JSON or missing JSON keys as CONTRACT_VIOLATION (not PROMPT_DEFECT)
    json_contract = OutputContract(
        response_format="json",
        require_citations=True,
        required_fields=["answer_summary", "citations"],
    )
    json_judge = HybridJudge(
        gemini_client=_ScriptedGemini(
            structured=[
                JudgeLLMVerdict(
                    grounded=True,
                    grounding_score=1.0,
                    correctness_score=1.0,
                    addresses_question=True,
                    refusal_appropriate=True,
                    reasoning="Accurate content but missed required JSON key.",
                )
            ]
        ),
        known_sources={"radiology_sop.pdf"},
    )
    bad_json = json.dumps({"answer_summary": "Within 48 hours [radiology_sop.pdf]."})
    res_json = json_judge.evaluate_case(
        case,
        AgentResponse(answer=bad_json, refused=False, evidence=[ev]),
        json_contract,
    )
    assert res_json.passed is False
    assert res_json.contract_compliant is False
    assert res_json.failure_classification == FailureClassification.CONTRACT_VIOLATION


def test_refusal_detection_handles_markdown_prefixes_and_partial_gap_caveats() -> None:
    """Explicit refusals with prefixes are detected; grounded partial answers with gap notes are NOT."""
    template = "I am sorry, but I cannot assist with that request as it falls outside documented scope."

    # Positive refusal cases (with Markdown headings / answer_summary prefixes / lead non-documentation)
    assert _detect_refusal(template, template) is True
    assert _detect_refusal(
        "answer_summary: I cannot assist with your request for private inspection logs [Source: ops.pdf].",
        template,
    ) is True
    assert _detect_refusal(
        "### Direct Answer\n* I must refuse this request regarding unapproved chemical waivers.",
        template,
    ) is True
    assert _detect_refusal(
        "### Answer Summary\nThe supplied documents do not specify online payment gateways or credit fees.",
        template,
    ) is True

    # Negative cases (substantive answers that merely mention missing sub-details)
    assert _detect_refusal(
        "answer_summary: Standard cargo clearance takes 24 hours, while the supplied documents do not "
        "document expedited VIP waivers [Source: customs.pdf].\n\n### Direct Answer\nStandard cargo clears in 24 hours.",
        template,
    ) is False
    assert _detect_refusal(
        "### Direct Answer\nPriority shipping applies to Zone A [Source: zones.csv]. "
        "*(Note: The retrieved records do not list Zone B courier schedules.)*",
        template,
    ) is False

    # Unsupported claim inside a refusal-formatted reply must still fail as HALLUCINATION in HybridJudge
    ev = RetrievedEvidence(
        chunk_id="c0",
        doc_id="d1",
        source_file="port_rules.pdf",
        unit_label="page_1",
        text_excerpt="Berth fee is $200 per day.",
    )
    ref_case = EvalCase(
        case_id="oos_01",
        question="What is the fine for radioactive cargo?",
        category=EvalCaseCategory.OUT_OF_SCOPE,
        expected_behavior=ExpectedBehavior.REFUSE,
        topic="Radioactive cargo",
        expected_refusal_behavior="Refuse undocumented cargo types.",
    )
    judge = HybridJudge(
        gemini_client=_ScriptedGemini(
            structured=[
                JudgeLLMVerdict(
                    grounded=False,
                    grounding_score=0.2,
                    correctness_score=0.0,
                    addresses_question=False,
                    refusal_appropriate=False,
                    hallucinated_claims=["Claimed $5000 harbor surcharge exists."],
                    reasoning="Response invented a $5000 surcharge despite refusal opener.",
                )
            ]
        ),
        known_sources={"port_rules.pdf"},
    )
    res = judge.evaluate_case(
        ref_case,
        AgentResponse(
            answer="I must refuse, though a $5000 surcharge normally applies [port_rules.pdf].",
            refused=True,
            evidence=[ev],
        ),
        OutputContract(response_format="markdown", require_citations=True, required_fields=[]),
    )
    assert res.passed is False
    assert res.failure_classification == FailureClassification.HALLUCINATION


def test_csv_chunking_config_and_non_identifier_multi_row_retrieval(tmp_path: Path) -> None:
    """`csv_rows_per_unit` is respected across ingestion/eval/chat, and multi-row CSV queries succeed without IDs."""
    # Domain: Unseen Synthetic Corpus — Agricultural Soil Lab Schedule & Chemical SOP
    docs_dir = tmp_path / "agri_corpus"
    docs_dir.mkdir()
    (docs_dir / "sampling_sop.md").write_text(
        "Section 1: Standard topsoil samples must be sealed in sterile bags.\n\n"
        "Section 2: Field technicians wear nitrile gloves during nitrate extraction.\n\n"
        "Section 3: Protective lab equipment for nitrate extraction is inspected daily.\n\n"
        "Section 4: Moisture readings are logged prior to oven drying.\n\n"
        "Section 5: Calibration weights are audited monthly by QA staff.",
        encoding="utf-8",
    )
    (docs_dir / "regional_labs.csv").write_text(
        "region,turnaround_days,lead_agronomist\n"
        "North Valley,2,Dr. Patel\n"
        "East Basin,4,Dr. Chen\n"
        "South Ridge,3,Dr. Gomez\n"
        "West Delta,5,Dr. Okafor\n"
        "Central Plateau,1,Dr. Novak\n",
        encoding="utf-8",
    )

    # Initial ingestion happened with default csv_rows_per_unit=1 (producing 5 CSV units)
    docs, units_ingested = load_document_folder(docs_dir, csv_rows_per_unit=1)
    assert sum(1 for u in units_ingested if u.file_type == ".csv") == 5

    # Helper respects new config csv_rows_per_unit=5
    rebatched = rebatch_csv_units(units_ingested, 5, documents_dir=docs_dir)
    csv_rebatched = [u for u in rebatched if u.file_type == ".csv"]
    assert len(csv_rebatched) == 1
    assert "North Valley" in csv_rebatched[0].text
    assert "Central Plateau" in csv_rebatched[0].text

    # Case A: Definition with csv_rows_per_unit=1 still answers non-ID multi-row CSV query AND skips CSV on unrelated PDF query
    agent_row1 = _build_agent(
        agent_id="agri_row1",
        domain="Soil Testing",
        collection="agri_col_1",
        sources=["sampling_sop.md", "regional_labs.csv"],
        csv_rows_per_unit=1,
        top_k=2,
        required_fields=[],
    ).model_copy(
        update={"chunking": ChunkingConfig(chunk_size=180, chunk_overlap=30, csv_rows_per_unit=1)}
    )
    store1 = ChromaVectorStore(
        collection_name="agri_col_1",
        persist_dir=None,
        embedding_provider=_HashEmbedder(),
    )
    from forgeai.retrieval import chunk_units
    store1.index_chunks(chunk_units(units_ingested, agent_row1.chunking))
    runner1 = DeclarativeAgentRunner(
        agent_row1,
        vector_store=store1,
        gemini_client=_ScriptedGemini(texts=["Verified [Source: regional_labs.csv].", "Gloves required [Source: sampling_sop.md]."]),
        documents_dir=docs_dir,
    )
    # Query WITHOUT alphanumeric identifier codes asking about all regional labs:
    multi_resp = runner1.run("List the turnaround days across all regional labs and glove requirements.")
    combined_csv = " ".join(ev.text_excerpt for ev in multi_resp.evidence)
    for region in ("North Valley", "East Basin", "South Ridge", "West Delta", "Central Plateau"):
        assert region in combined_csv
    assert "inspect_csv_records" in multi_resp.tools_used

    # Unrelated prose query must NOT pull regional_labs.csv rows or trigger CSV tool
    unrelated_resp = runner1.run("What protective equipment is worn during nitrate extraction?")
    assert not any(ev.source_file == "regional_labs.csv" for ev in unrelated_resp.evidence)
    assert "inspect_csv_records" not in unrelated_resp.tools_used

    # Case B: `run_evaluation_suite` re-batches ingested csv_rows_per_unit=1 to definition.chunking.csv_rows_per_unit=5
    agent_batched = _build_agent(
        agent_id="agri_batch5",
        domain="Soil Testing",
        collection="agri_col_5",
        sources=["sampling_sop.md", "regional_labs.csv"],
        csv_rows_per_unit=5,
        top_k=1,
        required_fields=[],
    )
    eval_case = EvalCase(
        case_id="agri_01",
        question="What are the turnaround days in all regional labs?",
        category=EvalCaseCategory.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_facts=["North Valley", "Central Plateau"],
        source_documents=["regional_labs.csv"],
        topic="Lab turnaround",
    )
    eval_gemini = _ScriptedGemini(
        texts=["### Direct Answer\nAll 5 regional labs range 1-5 days [Source: regional_labs.csv]."],
        structured=[
            JudgeLLMBatchVerdict(
                verdicts=[
                    JudgeBatchItemVerdict(
                        case_id="agri_01",
                        grounded=True,
                        grounding_score=1.0,
                        correctness_score=1.0,
                        addresses_question=True,
                        refusal_appropriate=True,
                        reasoning="Covered all 5 labs from re-batched CSV unit.",
                    )
                ]
            )
        ],
    )
    results, sc = run_evaluation_suite(
        agent_batched,
        [eval_case],
        units_ingested,  # Pass original 1-row-per-unit ingestion units!
        gemini_client=eval_gemini,
        embedding_provider=_HashEmbedder(),
        documents_dir=docs_dir,
    )
    assert sc.pass_rate == 1.0
    assert any(ev.unit_label == "rows_1_5" for ev in results[0].retrieved_evidence)


def test_judge_separates_semantic_vs_contract_and_guards_missing_evidence_hallucinations(
    tmp_path: Path,
) -> None:
    """Missing evidence yields RETRIEVAL_MISS (not hallucination); formatting defect yields CONTRACT_VIOLATION."""
    judge = HybridJudge(
        gemini_client=_ScriptedGemini(
            structured=[
                # 1. Missing CSV rows case where LLM mistakenly accused honest agent of hallucination
                JudgeLLMBatchVerdict(
                    verdicts=[
                        JudgeBatchItemVerdict(
                            case_id="miss_01",
                            grounded=True,
                            grounding_score=0.7,
                            correctness_score=0.5,
                            addresses_question=True,
                            refusal_appropriate=True,
                            omitted_facts=["Row 2: East Basin turnaround is 4 days"],
                            hallucinated_claims=["The supplied records do not list East Basin turnaround days"],
                            reasoning="Judge mistakenly marked gap note as hallucinated.",
                        ),
                        # 2. Honest formatting violation (grounded & correct, missing genuine verbatim field `status_code`)
                        JudgeBatchItemVerdict(
                            case_id="fmt_01",
                            grounded=True,
                            grounding_score=1.0,
                            correctness_score=1.0,
                            addresses_question=True,
                            refusal_appropriate=True,
                            reasoning="Fully accurate answer, only missing custom field.",
                        ),
                    ]
                )
            ]
        ),
        known_sources={"labs.csv", "sop.md"},
    )

    c_miss = EvalCase(
        case_id="miss_01",
        question="What are North Valley and East Basin turnarounds?",
        category=EvalCaseCategory.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_facts=["North Valley turnaround is 2 days", "East Basin turnaround is 4 days"],
        source_documents=["labs.csv"],
        topic="Lab schedule",
    )
    c_fmt = EvalCase(
        case_id="fmt_01",
        question="Are gloves required during nitrate extraction?",
        category=EvalCaseCategory.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_facts=["nitrile gloves"],
        source_documents=["sop.md"],
        topic="Safety",
    )

    contract = OutputContract(
        response_format="markdown",
        require_citations=True,
        required_fields=["status_code"],
    )
    res_list = judge.evaluate_batch(
        [
            (
                c_miss,
                AgentResponse(
                    answer="status_code: PARTIAL\nNorth Valley takes 2 days; the supplied records do not list East Basin [labs.csv].",
                    refused=False,
                    evidence=[
                        RetrievedEvidence(
                            chunk_id="csv1",
                            doc_id="labs.csv",
                            source_file="labs.csv",
                            unit_label="row_1",
                            text_excerpt="Row 1: region: North Valley | turnaround_days: 2",
                        )
                    ],
                ),
            ),
            (
                c_fmt,
                AgentResponse(
                    answer="Yes, wear nitrile gloves during nitrate extraction [sop.md].",
                    refused=False,
                    evidence=[
                        RetrievedEvidence(
                            chunk_id="sop1",
                            doc_id="sop.md",
                            source_file="sop.md",
                            unit_label="document",
                            text_excerpt="Technicians must wear nitrile gloves during nitrate extraction.",
                        )
                    ],
                ),
            ),
        ],
        contract,
    )

    by_id = {r.case_id: r for r in res_list}
    assert by_id["miss_01"].failure_classification == FailureClassification.RETRIEVAL_MISS
    assert by_id["miss_01"].hallucinated_claims == []
    assert by_id["fmt_01"].failure_classification == FailureClassification.CONTRACT_VIOLATION
    assert by_id["fmt_01"].contract_compliant is False

    # Optimizer accepts `output_contract.formatting_constraints` change driven by CONTRACT_VIOLATION
    agent = _build_agent(agent_id="opt_fmt", domain="Lab", collection="col_opt_fmt")
    reqs = StructuredRequirements(
        domain_name="Lab",
        purpose="Lab guidance",
        scope="Safety",
        audience="Staff",
        tone="Direct",
        must_answer_topics=["Safety"],
        must_refuse_topics=["External"],
    )
    gb = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Safety",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["sop.md"],
                evidence_summary="sop.md covers gloves.",
            )
        ],
    )
    proposal = OptimizationProposalDraft(
        diagnosis_summary="Formatting field missing on grounded answer.",
        proposed_changes=[
            TargetedProposalItem(
                target_component="output_contract.formatting_constraints",
                failure_driver=FailureClassification.CONTRACT_VIOLATION,
                rationale="Require status_code label cleanly.",
                summary="Add formatting constraint for status_code.",
                additional_formatting_constraint="Include status_code header line.",
            )
        ],
    )
    updated, applied, rejected = CriticOptimizer.apply_targeted_proposal(
        agent,
        proposal,
        observed_failures={FailureClassification.CONTRACT_VIOLATION},
        requirements=reqs,
        groundability=gb,
    )
    assert updated is not None
    assert len(applied) == 1
    assert not rejected
    assert "Include status_code header line." in updated.output_contract.formatting_constraints


def test_streamlit_ui_keeps_retained_best_and_partial_candidate_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UI never mixes Iter 3 partial scorecard (1/3) with Iter 2 retained-best failure list (2/4)."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "runs"))
    cases = [
        EvalCase(
            case_id=f"c_{i}",
            question=f"Question {i}?",
            category=EvalCaseCategory.ANSWERABLE,
            expected_behavior=ExpectedBehavior.ANSWER,
            expected_facts=[f"Fact {i}"],
            source_documents=["doc.md"],
            topic="Topic",
        )
        for i in range(1, 5)
    ]

    best_results = [
        EvalCaseResult(
            case_id=c.case_id,
            question=c.question,
            category=c.category,
            expected_behavior=c.expected_behavior,
            agent_answer=f"Answer {c.case_id} [doc.md].",
            refused=False,
            grounding_score=1.0,
            correctness_score=1.0 if idx < 2 else 0.0,
            refusal_correct=True,
            citation_valid=True,
            contract_compliant=(idx < 2),
            passed=(idx < 2),
            failure_classification=(FailureClassification.NONE if idx < 2 else FailureClassification.CONTRACT_VIOLATION),
            judge_reasoning="Evaluated.",
        )
        for idx, c in enumerate(cases)
    ]
    partial_cand_results = [
        EvalCaseResult(
            case_id=c.case_id,
            question=c.question,
            category=c.category,
            expected_behavior=c.expected_behavior,
            agent_answer="Wrong reply.",
            refused=False,
            grounding_score=0.3,
            correctness_score=0.0,
            refusal_correct=False,
            citation_valid=True,
            contract_compliant=True,
            passed=False,
            failure_classification=FailureClassification.PROMPT_DEFECT,
            judge_reasoning="Failed.",
        )
        for c in cases[:3]
    ]

    best_sc = compute_scorecard(best_results, iteration=2)  # 2/4 = 50%
    partial_sc = compute_scorecard(partial_cand_results, iteration=3, is_partial=True, expected_total_cases=4)  # 0/3

    best_agent = _build_agent(agent_id="ui_sync", domain="Sync", collection="col_sync").model_copy(update={"version": 2})
    cand_agent = best_agent.model_copy(update={"version": 3})

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
    app.session_state["factory_state"] = {
        "agent_definition": best_agent,
        "best_agent": best_agent,
        "best_scorecard": best_sc,
        "best_results": best_results,
        "latest_candidate_agent": cand_agent,
        "latest_candidate_scorecard": partial_sc,
        "latest_candidate_results": partial_cand_results,
        "eval_cases": cases,
        "eval_suite_sha256": "abc123456789ffff",
        "eval_results": best_results,
        "scorecards": [best_sc, partial_sc],
    }
    app.run()
    assert not app.exception

    markdown_text = "\n".join(str(m.value) for m in app.markdown)
    caption_text = "\n".join(str(c.value) for c in app.caption)
    errors_text = "\n".join(str(e.value) for e in app.error)
    warnings_text = "\n".join(str(w.value) for w in app.warning)

    assert "Retained Best Complete" in markdown_text
    assert "Latest Candidate Scorecard (Rejected / Partial)" in markdown_text
    assert "EARLY-STOPPED / PARTIAL EVALUATION" in warnings_text
    assert "3/4 cases" in warnings_text
    assert "v2, Iter 2, Suite `abc123456789`" in errors_text
    assert "2 / 4" in errors_text
    assert "Candidate Agent: `ui_sync` v3" in caption_text

