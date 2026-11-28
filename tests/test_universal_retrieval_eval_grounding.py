"""Domain-agnostic regression tests for universal retrieval, eval integrity, grounding, and judge classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forgeai.evaluation import EvalGenerator, HybridJudge, compute_scorecard
from forgeai.generator.agent_generator import _ensure_prompt_guardrails
from forgeai.optimization.optimizer import CriticOptimizer
from forgeai.retrieval import ChromaVectorStore, chunk_units, load_document_folder
from forgeai.runner.agent_runner import AgentResponse, DeclarativeAgentRunner
from forgeai.schemas import (
    ChunkingConfig,
    DeclarativeAgentDefinition,
    EvalCase,
    EvalCaseCategory,
    EvalSuiteDraft,
    ExpectedBehavior,
    FailureClassification,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    JudgeBatchItemVerdict,
    JudgeLLMBatchVerdict,
    JudgeLLMVerdict,
    OutputContract,
    RetrievalConfig,
    RetrievedEvidence,
    StructuredRequirements,
    ToolSelection,
)
from forgeai.tools.registry import build_default_registry


class _HashEmbedder:
    """Deterministic local embedder separating lexical topics cleanly without network calls."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "hash-test-embedder"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        words = [w for w in text.lower().replace("\n", " ").split() if len(w) > 2]
        for tok in words or ["empty"]:
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _ScriptedGemini:
    """Mock Gemini client tracking structured and text calls across prompts."""

    def __init__(self, structured: list[Any] | None = None, texts: list[str] | None = None) -> None:
        self._structured = list(structured or [])
        self._texts = list(texts or [])
        self.structured_prompts: list[str] = []
        self.text_prompts: list[str] = []
        self.system_instructions: list[str | None] = []

    def generate_structured(self, prompt: str, schema: type[Any], **_kwargs: Any) -> Any:
        self.structured_prompts.append(prompt)
        if not self._structured:
            raise AssertionError(f"Unexpected structured call for {schema.__name__}.")
        obj = self._structured.pop(0)
        assert isinstance(obj, schema)
        return obj

    def generate_text(
        self, prompt: str, *, system_instruction: str | None = None, **_kwargs: Any
    ) -> str:
        self.text_prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        if self._texts:
            return self._texts.pop(0)
        return "Documented rule cited [policy.md | document]."


def _build_domain_agent(
    *,
    agent_id: str,
    domain: str,
    collection_name: str,
    source_documents: list[str],
    top_k: int = 3,
) -> DeclarativeAgentDefinition:
    return DeclarativeAgentDefinition(
        agent_id=agent_id,
        name=f"{domain} Assistant",
        version=1,
        domain_summary=f"Grounded assistant for {domain}.",
        system_prompt="Answer strictly from supplied documents with inline citations.",
        chunking=ChunkingConfig(chunk_size=260, chunk_overlap=70),
        retrieval=RetrievalConfig(collection_name=collection_name, top_k=top_k),
        tools=[
            ToolSelection(name="retrieve_documents", purpose="Retrieve corpus chunks."),
            ToolSelection(name="inspect_csv_records", purpose="Lookup CSV rows."),
        ],
        output_contract=OutputContract(
            response_format="markdown",
            require_citations=True,
            citation_format="[source_file | unit]",
        ),
        source_documents=source_documents,
        must_answer_topics=[domain],
        must_refuse_topics=["Unrelated requests"],
    )


def _assert_no_midword_overlap_starts(texts: list[str], full_text: str) -> None:
    for idx, chunk in enumerate(texts):
        if idx == 0:
            continue
        pos = full_text.find(chunk)
        assert pos != -1, f"Chunk {idx} does not cleanly match source substring."
        if pos > 0:
            prev_char = full_text[pos - 1]
            assert prev_char.isspace(), (
                f"Chunk {idx} started mid-word after {prev_char!r}: {chunk[:40]!r}"
            )


def test_chunking_preserves_sentences_sections_and_word_boundaries_across_domains(
    tmp_path: Path,
) -> None:
    """Chunking must preserve complete words, sentences, headings, and metadata across 3 domains."""
    domain_texts = {
        "library_membership.md": (
            "1. Visitor and Alumni Eligibility\n"
            "Alumni and community visitors hold reading-room privileges only and do not qualify "
            "for physical off-site borrowing.\n\n"
            "2. Faculty Privileges\n"
            "Full-time faculty may borrow up to 40 volumes per semester upon presenting valid ID."
        ),
        "hr_leave_policy.md": (
            "# Bereavement and Parental Leave\n"
            "Full-time staff receive 16 weeks of paid parental leave within 12 months of birth.\n\n"
            "# Notice Requirements\n"
            "Employees must submit leave requests 30 calendar days in advance unless emergency."
        ),
        "finance_vendor_terms.md": (
            "Payment Terms:\n"
            "Approved supplier invoices are settled on a Net 45 calendar-day schedule after inspection.\n\n"
            "Dispute Window:\n"
            "Invoice discrepancies must be flagged in writing within 10 business days of receipt."
        ),
    }

    for idx, (fname, body) in enumerate(domain_texts.items()):
        dom_dir = tmp_path / f"dom_{idx}"
        dom_dir.mkdir()
        (dom_dir / fname).write_text(body, encoding="utf-8")
        _, units = load_document_folder(dom_dir)
        chunks = chunk_units(units, ChunkingConfig(chunk_size=140, chunk_overlap=45))
        assert len(chunks) >= 2
        _assert_no_midword_overlap_starts([c.text for c in chunks], body.strip())
        assert all(c.source_file == fname and c.doc_id and c.unit_label for c in chunks)
        assert any("section" in c.metadata for c in chunks)


def test_runner_preserves_multiple_unit_chunks_and_pulls_adjacent_chunks_multi_domain(
    tmp_path: Path,
) -> None:
    """Runner must not deduplicate away sibling chunks from the same page and must fetch neighbors."""
    # Domain 1: University Library (c0 has fine cap, c1 has grace rule; query matches c1 strongly)
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    lib_body = (
        "1. Overdue Accrual Cap.\n"
        "Daily overdue fines stop accruing once the cumulative balance reaches the maximum cap of $25.00 per item.\n\n"
        "2. Standard Account Hold.\n"
        "Accounts exceeding $10.00 cannot check out equipment until settled at the desk. "
        "Replacement fees apply separately for lost media items and damaged reference bindings."
    )
    (lib_dir / "fines_policy.md").write_text(lib_body, encoding="utf-8")
    (lib_dir / "library_hours.csv").write_text(
        "day,status,hours\nSunday,CLOSED,Closed all day\nMonday,OPEN,08:00-20:00\n",
        encoding="utf-8",
    )

    _, lib_units = load_document_folder(lib_dir)
    lib_agent = _build_domain_agent(
        agent_id="lib_test",
        domain="Library Policy",
        collection_name="lib_adj_col",
        source_documents=["fines_policy.md", "library_hours.csv"],
        top_k=1,
    )
    lib_store = ChromaVectorStore(
        collection_name="lib_adj_col",
        persist_dir=None,
        embedding_provider=_HashEmbedder(),
    )
    lib_chunks = chunk_units(lib_units, lib_agent.chunking)
    # Ensure fines_policy.md split into >= 2 chunks under the same unit_label ("document")
    fines_chunks = [c for c in lib_chunks if c.source_file == "fines_policy.md"]
    assert len(fines_chunks) >= 2
    lib_store.index_chunks(lib_chunks)

    # Confirm adjacent-chunk helper directly retrieves sibling c0 when given c1
    adj = lib_store.get_adjacent_chunks([
        RetrievedEvidence(
            chunk_id=fines_chunks[1].chunk_id,
            doc_id=fines_chunks[1].doc_id,
            source_file=fines_chunks[1].source_file,
            unit_label=fines_chunks[1].unit_label,
            text_excerpt=fines_chunks[1].text,
        )
    ])
    assert any(ev.chunk_id == fines_chunks[0].chunk_id for ev in adj)

    # Runner with top_k=1 querying replacement fees / hold still includes neighbor c0 ($25.00 cap)
    lib_gemini = _ScriptedGemini(texts=["Fines are capped at $25.00 [fines_policy.md | document]."])
    runner = DeclarativeAgentRunner(
        lib_agent,
        vector_store=lib_store,
        gemini_client=lib_gemini,
        tool_registry=build_default_registry(vector_store=lib_store, documents_dir=lib_dir),
        documents_dir=lib_dir,
    )
    res_lib = runner.run("accounts hold replacement fees lost media")
    retrieved_ids = {ev.chunk_id for ev in res_lib.evidence}
    assert fines_chunks[0].chunk_id in retrieved_ids
    assert fines_chunks[1].chunk_id in retrieved_ids
    # Anti-extrapolation rule is enforced in runtime prompt
    assert "extrapolate" in lib_gemini.text_prompts[0].lower()

    # Domain 2 & Domain 3: HR & Finance isolated runners with CSV lookup + prose neighbors
    for dom_name, md_name, md_text, csv_name, csv_text, query, expected_token in [
        (
            "HR Travel",
            "travel_policy.md",
            "1. Per Diem Ceiling.\nMeals are reimbursed up to $75 per travel day with receipts.\n\n"
            "2. Rail vs Air.\nEconomy rail is mandatory for trips under 250 miles in distance.",
            "grades.csv",
            "grade_id,approval_level\nG-12,VP Finance\n",
            "G-12 trips distance economy rail",
            "VP Finance",
        ),
        (
            "Procurement",
            "procurement_sla.md",
            "1. Emergency Waiver.\nEmergency purchases over $5000 require dual CFO and CPO sign-off.\n\n"
            "2. Standard Audit.\nAll purchase orders undergo quarterly compliance sampling review.",
            "vendors.csv",
            "vendor_id,tier\nV-88,Preferred\n",
            "V-88 compliance sampling quarterly audit",
            "Preferred",
        ),
    ]:
        dom_dir = tmp_path / dom_name.lower().replace(" ", "_")
        dom_dir.mkdir()
        (dom_dir / md_name).write_text(md_text, encoding="utf-8")
        (dom_dir / csv_name).write_text(csv_text, encoding="utf-8")
        _, d_units = load_document_folder(dom_dir)
        d_agent = _build_domain_agent(
            agent_id=dom_name.lower(),
            domain=dom_name,
            collection_name=f"col_{dom_name.lower().replace(' ', '_')}",
            source_documents=[md_name, csv_name],
            top_k=2,
        )
        d_store = ChromaVectorStore(
            collection_name=d_agent.retrieval.collection_name,
            persist_dir=None,
            embedding_provider=_HashEmbedder(),
        )
        d_chunks = chunk_units(d_units, ChunkingConfig(chunk_size=120, chunk_overlap=35))
        d_store.index_chunks(d_chunks)
        d_runner = DeclarativeAgentRunner(
            d_agent,
            vector_store=d_store,
            gemini_client=_ScriptedGemini(texts=[f"Verified [{csv_name} | row_1]."]),
            documents_dir=dom_dir,
        )
        resp = d_runner.run(query)
        combined = " ".join(ev.text_excerpt for ev in resp.evidence)
        assert expected_token in combined
        assert "inspect_csv_records" in resp.tools_used


def test_eval_generator_repairs_contradictory_refusal_without_breaking_valid_refusals(
    tmp_path: Path,
) -> None:
    """EvalGenerator repairs `refuse` edge cases whose cited corpus units clearly state the answer."""
    docs_dir = tmp_path / "lib"
    docs_dir.mkdir()
    (docs_dir / "borrowing_rules.md").write_text(
        "Reservation holds remain available at the desk. "
        "The pickup window after notification is 2 calendar days before the hold expires.",
        encoding="utf-8",
    )
    (docs_dir / "catalog.csv").write_text("item_id,title\nB-101,Algorithms\n", encoding="utf-8")
    docs, units = load_document_folder(docs_dir)

    req = StructuredRequirements(
        domain_name="Library",
        purpose="Answer borrowing policy queries.",
        scope="Reservations and catalog.",
        audience="Students",
        tone="Clear",
        must_answer_topics=["Reservation pickup window"],
        must_refuse_topics=["Personal student GPA lookup"],
    )
    agent = _build_domain_agent(
        agent_id="lib_eval",
        domain="Library",
        collection_name="lib_eval_col",
        source_documents=["borrowing_rules.md", "catalog.csv"],
    )

    contradictory = EvalCase(
        case_id="eval_013",
        question="After receiving notification that a book reservation is ready, what is the pickup window before it expires?",
        category=EvalCaseCategory.EDGE_CASE,
        expected_behavior=ExpectedBehavior.REFUSE,
        source_documents=["borrowing_rules.md"],
        evidence_references=["borrowing_rules.md | document"],
        topic="Reservation pickup",
        expected_refusal_behavior="Refuse if unspecified.",
    )
    legit_near_miss = EvalCase(
        case_id="eval_019",
        question="Is item B-999 listed in catalog.csv?",
        category=EvalCaseCategory.EDGE_CASE,
        expected_behavior=ExpectedBehavior.REFUSE,
        source_documents=["catalog.csv"],
        topic="Near-miss catalog ID",
        expected_refusal_behavior="Refuse because B-999 is not in catalog.",
    )
    repaired_13 = contradictory.model_copy(
        update={
            "expected_behavior": ExpectedBehavior.ANSWER,
            "expected_facts": ["2 calendar days"],
            "expected_refusal_behavior": "",
        }
    )

    gemini = _ScriptedGemini(
        structured=[
            EvalSuiteDraft(cases=[contradictory, legit_near_miss]),
            EvalSuiteDraft(cases=[repaired_13]),
        ]
    )
    gen = EvalGenerator(gemini_client=gemini, min_cases=2)
    cases, meta = gen.generate(req, agent, docs, units)

    assert len(gemini.structured_prompts) == 2
    # Only the genuinely contradictory refusal was sent for repair (not near-miss B-999)
    assert "eval_013" in gemini.structured_prompts[1]
    assert "eval_019" not in gemini.structured_prompts[1]
    by_id = {c.case_id: c for c in cases}
    assert by_id["eval_013"].expected_behavior == ExpectedBehavior.ANSWER
    assert by_id["eval_013"].expected_facts == ["2 calendar days"]
    assert by_id["eval_019"].expected_behavior == ExpectedBehavior.REFUSE
    assert meta.accepted_cases == 2


def test_judge_and_optimizer_distinguish_retrieval_miss_eval_case_defect_and_extrapolation(
    tmp_path: Path,
) -> None:
    """Judge detects retrieval_miss within same file, eval_case_defect, and extrapolation hallucination."""
    docs_dir = tmp_path / "corpus"
    docs_dir.mkdir()
    (docs_dir / "rules.md").write_text(
        "Section 1: All regular members may renew twice. Section 2: Pickup window is 2 calendar days.",
        encoding="utf-8",
    )
    docs, units = load_document_folder(docs_dir)
    agent = _build_domain_agent(
        agent_id="judge_test",
        domain="Policy",
        collection_name="judge_col",
        source_documents=["rules.md"],
    )
    contract = agent.output_contract

    # Case A: Same source_file chunk retrieved (`c1` without `c0` containing expected_facts) + refused -> RETRIEVAL_MISS
    case_ret_miss = EvalCase(
        case_id="c_ret",
        question="How many times may regular members renew?",
        category=EvalCaseCategory.ANSWERABLE,
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_facts=["renew twice"],
        source_documents=["rules.md"],
        topic="Renewals",
    )
    # Case B: eval_case_defect (marked refuse even though retrieved chunk directly answers question)
    case_defect = EvalCase(
        case_id="c_def",
        question="What is the exact reservation pickup window in rules.md?",
        category=EvalCaseCategory.EDGE_CASE,
        expected_behavior=ExpectedBehavior.REFUSE,
        source_documents=["rules.md"],
        evidence_references=["rules.md | document"],
        topic="Pickup",
        expected_refusal_behavior="Refuse ambiguous edge inquiry.",
    )
    # Case C: Extrapolation across docs (`expected_behavior=refuse`, agent invented causal rule) -> HALLUCINATION
    case_extrap = EvalCase(
        case_id="c_extrap",
        question="Since Sunday is closed, are overdue fines paused on Sundays?",
        category=EvalCaseCategory.EDGE_CASE,
        expected_behavior=ExpectedBehavior.REFUSE,
        source_documents=["rules.md"],
        topic="Closed-day rule",
        expected_refusal_behavior="Refuse undocumented interaction.",
    )

    judge_gemini = _ScriptedGemini(
        structured=[
            JudgeLLMBatchVerdict(
                verdicts=[
                    JudgeBatchItemVerdict(
                        case_id="c_ret",
                        grounded=True,
                        grounding_score=1.0,
                        correctness_score=0.0,
                        addresses_question=False,
                        refusal_appropriate=False,
                        suggested_failure_classification=FailureClassification.RETRIEVAL_MISS,
                        reasoning="Chunk omitted Section 1 renewal rule.",
                    ),
                    JudgeBatchItemVerdict(
                        case_id="c_def",
                        grounded=True,
                        grounding_score=1.0,
                        correctness_score=1.0,
                        addresses_question=True,
                        refusal_appropriate=False,
                        eval_case_defect=True,
                        suggested_failure_classification=FailureClassification.EVAL_CASE_DEFECT,
                        reasoning="rules.md directly states pickup window is 2 calendar days; eval case refused erroneously.",
                    ),
                    JudgeBatchItemVerdict(
                        case_id="c_extrap",
                        grounded=False,
                        grounding_score=0.25,
                        correctness_score=0.0,
                        addresses_question=True,
                        refusal_appropriate=False,
                        hallucinated_claims=["Assumed closed days never accrue fines without text support."],
                        suggested_failure_classification=FailureClassification.HALLUCINATION,
                        reasoning="Agent extrapolated unstated closed-day exception.",
                    ),
                ]
            )
        ]
    )
    judge = HybridJudge(gemini_client=judge_gemini, known_sources={"rules.md"})
    results = judge.evaluate_batch(
        [
            (
                case_ret_miss,
                AgentResponse(
                    answer="I cannot answer this request based on the provided documents and scope.",
                    refused=True,
                    evidence=[
                        RetrievedEvidence(
                            chunk_id="d1:document:c1",
                            doc_id="d1",
                            source_file="rules.md",
                            unit_label="document",
                            text_excerpt="Section 2: Pickup window is 2 calendar days.",
                        )
                    ],
                ),
            ),
            (
                case_defect,
                AgentResponse(
                    answer="The pickup window is 2 calendar days [rules.md | document].",
                    refused=False,
                    evidence=[
                        RetrievedEvidence(
                            chunk_id="d1:document:c1",
                            doc_id="d1",
                            source_file="rules.md",
                            unit_label="document",
                            text_excerpt="Section 2: Pickup window is 2 calendar days.",
                        )
                    ],
                ),
            ),
            (
                case_extrap,
                AgentResponse(
                    answer="Fines pause on Sundays because the library is closed [rules.md | document].",
                    refused=False,
                    evidence=[
                        RetrievedEvidence(
                            chunk_id="d1:document:c0",
                            doc_id="d1",
                            source_file="rules.md",
                            unit_label="document",
                            text_excerpt="Sunday status CLOSED.",
                        )
                    ],
                ),
            ),
        ],
        contract,
    )

    by_id = {r.case_id: r for r in results}
    assert by_id["c_ret"].failure_classification == FailureClassification.RETRIEVAL_MISS
    assert by_id["c_def"].eval_case_defect is True
    assert by_id["c_def"].failure_classification == FailureClassification.EVAL_CASE_DEFECT
    assert by_id["c_extrap"].failure_classification == FailureClassification.HALLUCINATION

    # If only EVAL_CASE_DEFECT fails, CriticOptimizer must NOT call Gemini or mutate the agent
    req = StructuredRequirements(
        domain_name="Policy",
        purpose="Answer rules.",
        scope="Rules.",
        audience="Users",
        tone="Clear",
        must_answer_topics=["Pickup"],
        must_refuse_topics=["External"],
    )
    gb = GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Pickup",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["rules.md"],
                evidence_summary="rules.md documents the 2 calendar day pickup window.",
            )
        ],
    )
    no_call_gemini = _ScriptedGemini()
    optimizer = CriticOptimizer(gemini_client=no_call_gemini, max_iterations=2)
    defect_only = [by_id["c_def"]]
    opt_res = optimizer.optimize(
        run_id="defect_guard_run",
        requirements=req,
        groundability=gb,
        baseline_agent=agent,
        eval_cases=[case_defect],
        baseline_results=defect_only,
        baseline_scorecard=compute_scorecard(defect_only, iteration=0),
        documents=docs,
        units=units,
        artifacts_dir=tmp_path / "opt_artifacts",
    )
    assert opt_res.stop_reason == "only_eval_case_defects_remaining"
    assert opt_res.final_agent.version == 1
    assert len(no_call_gemini.structured_prompts) == 0


def test_prompt_guardrails_include_anti_extrapolation_and_partial_coverage() -> None:
    req = StructuredRequirements(
        domain_name="Operations",
        purpose="Answer manuals.",
        scope="Procedures",
        audience="Staff",
        tone="Direct",
        must_answer_topics=["Procedures"],
        must_refuse_topics=["Legal advice"],
    )
    gb = GroundabilityReport(overall_verdict=GroundabilityVerdict.SUPPORTED, can_proceed=True)
    hardened = _ensure_prompt_guardrails("Answer questions accurately.", req, gb)
    assert "extrapolate" in hardened.lower() or "separate documents" in hardened.lower()
    assert "untrusted data" in hardened.lower()
