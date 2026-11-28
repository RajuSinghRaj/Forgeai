"""Zero-cost tests for ForgeAI Phase C: Persistent Checkpointing & Resume."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from forgeai.cli import main as cli_main
from forgeai.cli import resume_factory_cli
from forgeai.config import ForgeSettings
from forgeai.evaluation import compute_scorecard, run_evaluation_suite
from forgeai.optimization.integrity import EvalIntegrityError, compute_eval_suite_hash
from forgeai.retrieval import load_document_folder
from forgeai.schemas import (
    AgentGenerationDraft,
    BriefInput,
    ChunkingConfig,
    ClarifyingQuestion,
    ClarifyingQuestionBatch,
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
    OptimizationProposalDraft,
    OutputContract,
    RetrievalConfig,
    RetrievalStrategyDraft,
    StructuredRequirements,
    SystemPromptDraft,
    TargetedProposalItem,
    ToolSelection,
    ToolSelectionDraft,
    ToolSelectionDraftItem,
)
from forgeai.workflow import (
    CheckpointCorruptionError,
    RunCheckpointStore,
    build_factory_graph,
    list_saved_runs,
    load_run_checkpoint,
)
from forgeai.workflow.checkpoints import progress_file_for


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


class _TrackingGeminiClient:
    """Mock GeminiClient tracking all calls, prompts, and schema names with zero network IO."""

    def __init__(self, structured_fn, text_fn=None) -> None:
        self._structured_fn = structured_fn
        self._text_fn = text_fn or (
            lambda p, **_kw: (
                "I cannot answer this request based on the provided documents and scope."
                if "discount fee" in p.lower()
                else "Inspection occurs every 30 days [policy.md | document]."
            )
        )
        self.structured_calls: list[str] = []
        self.text_prompts: list[str] = []
        self.judge_case_id_batches: list[list[str]] = []

    def generate_structured(self, prompt: str, response_schema: type[Any], **_kwargs: Any) -> Any:
        self.structured_calls.append(response_schema.__name__)
        if response_schema is JudgeLLMBatchVerdict:
            m = re.search(r"expected case_ids=(\[[^\n]+\])", prompt)
            cids = json.loads(m.group(1)) if m else []
            self.judge_case_id_batches.append(cids)
            return self._structured_fn(prompt, response_schema, case_ids=cids)
        return self._structured_fn(prompt, response_schema, case_ids=[])

    def generate_text(self, prompt: str, **_kwargs: Any) -> str:
        self.text_prompts.append(prompt)
        return self._text_fn(prompt)


def _create_corpus(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "policy.md").write_text(
        "Inspection occurs every 30 days. Escalations go to duty_lead within 24 hours.",
        encoding="utf-8",
    )
    return docs_dir


def _make_cases(count: int = 20) -> list[EvalCase]:
    distinct_nouns = [
        "boiler", "valve", "gasket", "pump", "sensor",
        "turbine", "compressor", "manifold", "chiller", "filter",
        "actuator", "gearbox", "bearing", "coupler", "seal",
        "thermostat", "strainer", "exhaust", "nozzle", "regulator",
        "relay", "switch", "gauge", "flange",
    ]
    cases: list[EvalCase] = []
    for i in range(1, count + 1):
        noun = distinct_nouns[(i - 1) % len(distinct_nouns)]
        if i % 5 == 0:
            cases.append(
                EvalCase(
                    case_id=f"case_{i:02d}",
                    topic="Negotiation",
                    category=EvalCaseCategory.OUT_OF_SCOPE,
                    expected_behavior=ExpectedBehavior.REFUSE,
                    question=f"Can you grant a custom discount fee on the {noun} invoice?",
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
                    question=f"How frequently must the marine {noun} assembly undergo scheduled inspection?",
                )
            )
    return cases


def _make_verdict_item(case_id: str, **overrides: Any) -> JudgeBatchItemVerdict:
    base = {
        "case_id": case_id,
        "grounded": True,
        "grounding_score": 1.0,
        "correctness_score": 1.0,
        "addresses_question": True,
        "refusal_appropriate": True,
        "reasoning": f"Verified {case_id}.",
    }
    base.update(overrides)
    return JudgeBatchItemVerdict(**base)


def _sample_requirements() -> StructuredRequirements:
    return StructuredRequirements(
        domain_name="Policy Ops",
        purpose="Answer grounded policy questions.",
        scope="Inspection intervals and escalation SLAs.",
        audience="Staff",
        tone="Concise",
        must_answer_topics=["Inspection intervals"],
        must_refuse_topics=["Pricing discounts"],
    )


def _sample_groundability() -> GroundabilityReport:
    return GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=True,
        findings=[
            GroundabilityFinding(
                topic="Inspection intervals",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["policy.md"],
                evidence_summary="Policy states 30-day inspections.",
            )
        ],
    )


def _sample_agent(version: int = 1, run_id: str = "run_test") -> DeclarativeAgentDefinition:
    req = _sample_requirements()
    return DeclarativeAgentDefinition(
        agent_id="policy_agent_01",
        name="Policy Agent",
        version=version,
        domain_summary="Grounded policy assistant.",
        system_prompt="Answer strictly from policy.md and cite [policy.md | document].",
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50),
        retrieval=RetrievalConfig(collection_name="col_policy", top_k=4),
        tools=[ToolSelection(name="retrieve_documents", purpose="Search corpus")],
        output_contract=OutputContract(response_format="markdown", require_citations=True),
        must_answer_topics=req.must_answer_topics,
        must_refuse_topics=req.must_refuse_topics,
        source_documents=["policy.md"],
        approved_requirements=req,
        metadata={"run_id": run_id},
    )


def _settings(tmp_path: Path, **overrides: Any) -> ForgeSettings:
    defaults: dict[str, Any] = {
        "_env_file": None,
        "gemini_api_key": "test-key",
        "gemini_model": "gemini-3.5-flash-lite",
        "artifacts_dir": tmp_path / "artifacts" / "runs",
        "max_optimization_iterations": 3,
        "judge_batch_size": 5,
    }
    defaults.update(overrides)
    return ForgeSettings(**defaults)


def test_resume_after_intake_and_after_agent_generation(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path)
    settings = _settings(tmp_path)
    brief = BriefInput(
        brief_text="Build an operations policy assistant grounded in policy.md.",
        documents_dir=docs_dir,
        run_id="resume_intake_run",
    )

    # Phase A of test: stop after intake interview questions
    def intake_fn(_prompt: str, schema: type[Any], **_kw: Any) -> Any:
        if schema is ClarifyingQuestionBatch:
            return ClarifyingQuestionBatch(
                questions=[
                    ClarifyingQuestion(
                        question_id="q1",
                        question="Include SLA hours?",
                        rationale="Clarify detail.",
                        target_field="scope",
                    )
                ]
            )
        raise AssertionError(f"Unexpected call during intake: {schema}")

    client1 = _TrackingGeminiClient(intake_fn)
    g1 = build_factory_graph(
        settings=settings,
        gemini_client=client1,
        embedding_provider=_DeterministicEmbedder(),
    )
    paused = g1.invoke({"brief_input": brief, "stop_after_phase2": True})
    assert paused["awaiting_user_answers"] is True
    assert client1.structured_calls == ["ClarifyingQuestionBatch"]

    run_dir = settings.artifacts_dir / "resume_intake_run"
    assert (run_dir / "run_state.json").exists()
    assert (run_dir / "intake_questions.json").exists()

    # Load from disk checkpoint and complete Phase 2 + Phase 3 without repeating intake questions
    loaded = load_run_checkpoint(run_dir)
    loaded["clarifying_answers"] = {"q1": "Yes"}
    loaded["interview_completed"] = True
    loaded["stop_after_phase2"] = False
    loaded["stop_after_phase3"] = True

    def phase2_3_fn(_prompt: str, schema: type[Any], **_kw: Any) -> Any:
        if schema is ClarifyingQuestionBatch:
            raise AssertionError("Intake questions must NOT be regenerated after checkpoint resume.")
        if schema is StructuredRequirements:
            return _sample_requirements()
        if schema is GroundabilityReport:
            return _sample_groundability()
        if schema is AgentGenerationDraft:
            return AgentGenerationDraft(
                system_prompt_draft=SystemPromptDraft(
                    agent_name="Policy Agent",
                    description="Grounded assistant.",
                    domain_summary="Policy",
                    system_prompt=(
                        "Answer strictly from policy.md, include verifiable citations, "
                        "and refuse unsupported requests cleanly."
                    ),
                ),
                retrieval_strategy_draft=RetrievalStrategyDraft(
                    chunk_size=500, chunk_overlap=50, top_k=4
                ),
                tool_selection_draft=ToolSelectionDraft(
                    selected_tools=[
                        ToolSelectionDraftItem(
                            name="retrieve_documents",
                            purpose="Retrieve policy text",
                            reason="Corpus lookup",
                        )
                    ]
                ),
                output_contract=OutputContract(response_format="markdown", require_citations=True),
            )
        raise AssertionError(schema)

    client2 = _TrackingGeminiClient(phase2_3_fn)
    g2 = build_factory_graph(
        settings=settings,
        gemini_client=client2,
        embedding_provider=_DeterministicEmbedder(),
    )
    after_p3 = g2.invoke(loaded)
    assert "ClarifyingQuestionBatch" not in client2.structured_calls
    assert (run_dir / "requirements.json").exists()
    assert (run_dir / "groundability.json").exists()
    assert (run_dir / "agent_v1.json").exists()
    saved_agent_id = after_p3["agent_definition"].agent_id

    # Interrupt after agent generation -> reload -> finish Phase 4 eval gen only without repeating Phases 1-3
    reloaded_p3 = load_run_checkpoint(run_dir)
    reloaded_p3["stop_after_phase3"] = False
    reloaded_p3["stop_after_eval_generation"] = True
    reloaded_p3["stop_after_phase4"] = True
    cases_20 = _make_cases(20)

    def eval_gen_only_fn(_prompt: str, schema: type[Any], **_kw: Any) -> Any:
        if schema is EvalSuiteDraft:
            return EvalSuiteDraft(cases=cases_20)
        raise AssertionError(f"Should NOT regenerate intake/groundability/agent; got {schema}")

    client3 = _TrackingGeminiClient(eval_gen_only_fn)
    g3 = build_factory_graph(
        settings=settings,
        gemini_client=client3,
        embedding_provider=_DeterministicEmbedder(),
    )
    after_eval_gen = g3.invoke(reloaded_p3)
    assert client3.structured_calls == ["EvalSuiteDraft"]
    assert after_eval_gen["agent_definition"].agent_id == saved_agent_id
    assert (run_dir / "eval_cases.json").exists()
    assert (run_dir / "eval_suite_sha256").exists()


def test_20_case_interrupted_at_12_and_partial_5_case_batch_resume(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path)
    _, units = load_document_folder(docs_dir)
    cases_20 = _make_cases(20)
    suite_hash = compute_eval_suite_hash(cases_20)
    agent = _sample_agent(version=1, run_id="eval_resume")
    cp_dir = tmp_path / "artifacts" / "runs" / "eval_resume"

    # Simulated run 1: batches 1 & 2 complete (10 cases), batch 3 has 2 cases then batch fails,
    # OR stop via fault injection after 12 cases (write partial 12/20 into progress file).
    # First test a full 10-case checkpoint via 2 complete 5-case batches then crash on batch 3:
    batch_num = {"n": 0}

    def crash_on_batch_3(_prompt: str, _schema: type[Any], case_ids: list[str]) -> JudgeLLMBatchVerdict:
        batch_num["n"] += 1
        if batch_num["n"] == 3:
            raise RuntimeError("503 UNAVAILABLE simulated crash")
        return JudgeLLMBatchVerdict(verdicts=[_make_verdict_item(cid) for cid in case_ids])

    client_fail = _TrackingGeminiClient(crash_on_batch_3)
    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        run_evaluation_suite(
            agent,
            cases_20,
            units,
            gemini_client=client_fail,
            embedding_provider=_DeterministicEmbedder(),
            checkpoint_dir=cp_dir,
            run_id="eval_resume",
            eval_suite_sha256=suite_hash,
            judge_batch_size=5,
        )

    prog_path = progress_file_for(cp_dir, iteration=0, agent_version=1)
    lines_after_crash = [json.loads(line) for line in prog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines_after_crash) == 10  # Two 5-case batches persisted immediately!

    # Write cases 11 & 12 cleanly via append_progress_record (simulating crash midway through batch 3
    # at 12/20 completed with the exact immutable 20-case suite SHA256), then resume full 20-case suite!
    from forgeai.evaluation.judge import HybridJudge as _HJ
    from forgeai.runner.agent_runner import AgentResponse as _AR
    from forgeai.schemas.evaluation import RetrievedEvidence as _RE
    from forgeai.workflow.checkpoints import EvalProgressRecord, append_progress_record

    sim_judge = _HJ(
        gemini_client=_TrackingGeminiClient(
            lambda _p, _s, case_ids: JudgeLLMBatchVerdict(verdicts=[_make_verdict_item(cid) for cid in case_ids])
        ),
        known_sources={"policy.md"},
    )
    sim_items = [
        (
            c,
            _AR(
                answer=(
                    "I cannot answer this request based on the provided documents and scope."
                    if c.expected_behavior == ExpectedBehavior.REFUSE
                    else "Inspection occurs every 30 days [policy.md | document]."
                ),
                refused=(c.expected_behavior == ExpectedBehavior.REFUSE),
                evidence=[
                    _RE(
                        chunk_id=f"c_{c.case_id}",
                        doc_id="d1",
                        source_file="policy.md",
                        unit_label="document",
                        text_excerpt="Inspection occurs every 30 days.",
                    )
                ],
                citations=([] if c.expected_behavior == ExpectedBehavior.REFUSE else ["policy.md (document)"]),
                tools_used=["retrieve_documents"],
            ),
        )
        for c in cases_20[10:12]
    ]
    partial_verdicts = sim_judge.evaluate_batch(
        sim_items, agent.output_contract, known_sources={"policy.md"}, batch_size=5
    )
    for pv in partial_verdicts:
        append_progress_record(
            prog_path,
            EvalProgressRecord(
                run_id="eval_resume",
                iteration=0,
                agent_version=1,
                eval_suite_sha256=suite_hash,
                case_id=pv.case_id,
                result=pv,
            ),
        )

    # Resume full 20-case run -> ONLY remaining 8 cases (case_13..case_20) execute (batches [5, 3])
    client_resume = _TrackingGeminiClient(
        lambda _p, _s, case_ids: JudgeLLMBatchVerdict(verdicts=[_make_verdict_item(cid) for cid in case_ids])
    )
    full_results, full_sc = run_evaluation_suite(
        agent,
        cases_20,
        units,
        gemini_client=client_resume,
        embedding_provider=_DeterministicEmbedder(),
        checkpoint_dir=cp_dir,
        run_id="eval_resume",
        eval_suite_sha256=suite_hash,
        judge_batch_size=5,
    )
    assert len(full_results) == 20
    assert full_sc.total_cases == 20
    assert full_sc.pass_rate == 1.0
    assert len(client_resume.text_prompts) == 8
    assert client_resume.judge_case_id_batches == [
        ["case_13", "case_14", "case_15", "case_16", "case_17"],
        ["case_18", "case_19", "case_20"],
    ]
    # Re-running again with all 20 checkpointed makes 0 runner and 0 judge calls
    client_zero = _TrackingGeminiClient(lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("No calls expected")))
    cached_res, _ = run_evaluation_suite(
        agent,
        cases_20,
        units,
        gemini_client=client_zero,
        embedding_provider=_DeterministicEmbedder(),
        checkpoint_dir=cp_dir,
        run_id="eval_resume",
        eval_suite_sha256=suite_hash,
        judge_batch_size=5,
    )
    assert len(cached_res) == 20
    assert len(client_zero.text_prompts) == 0
    assert len(client_zero.judge_case_id_batches) == 0


def test_sha256_mismatch_agent_version_isolation_and_corrupt_jsonl(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path)
    _, units = load_document_folder(docs_dir)
    cases = _make_cases(5)
    suite_hash = compute_eval_suite_hash(cases)
    cp_dir = tmp_path / "cp_security"

    agent_v1 = _sample_agent(version=1, run_id="sec_run")
    client_v1 = _TrackingGeminiClient(
        lambda _p, _s, case_ids: JudgeLLMBatchVerdict(verdicts=[_make_verdict_item(cid) for cid in case_ids])
    )
    res_v1, _ = run_evaluation_suite(
        agent_v1,
        cases,
        units,
        gemini_client=client_v1,
        embedding_provider=_DeterministicEmbedder(),
        checkpoint_dir=cp_dir,
        run_id="sec_run",
        eval_suite_sha256=suite_hash,
    )
    assert len(res_v1) == 5

    # Different agent version (v2) MUST NOT reuse v1 results
    agent_v2 = _sample_agent(version=2, run_id="sec_run")
    client_v2 = _TrackingGeminiClient(
        lambda _p, _s, case_ids: JudgeLLMBatchVerdict(verdicts=[_make_verdict_item(cid) for cid in case_ids])
    )
    res_v2, _ = run_evaluation_suite(
        agent_v2,
        cases,
        units,
        gemini_client=client_v2,
        embedding_provider=_DeterministicEmbedder(),
        checkpoint_dir=cp_dir,
        run_id="sec_run",
        eval_suite_sha256=suite_hash,
    )
    assert len(res_v2) == 5
    assert len(client_v2.text_prompts) == 5
    assert client_v2.judge_case_id_batches == [["case_01", "case_02", "case_03", "case_04", "case_05"]]

    # SHA256 mismatch rejects unsafe reuse
    with pytest.raises(EvalIntegrityError):
        run_evaluation_suite(
            agent_v1,
            cases,
            units,
            gemini_client=client_v1,
            embedding_provider=_DeterministicEmbedder(),
            checkpoint_dir=cp_dir,
            run_id="sec_run",
            eval_suite_sha256="0" * 64,
        )

    # Corrupt JSONL entry raises CheckpointCorruptionError cleanly
    prog_v1 = progress_file_for(cp_dir, iteration=0, agent_version=1)
    with prog_v1.open("a", encoding="utf-8") as handle:
        handle.write("{corrupt_incomplete_json\n")

    with pytest.raises(CheckpointCorruptionError):
        run_evaluation_suite(
            agent_v1,
            cases,
            units,
            gemini_client=client_v1,
            embedding_provider=_DeterministicEmbedder(),
            checkpoint_dir=cp_dir,
            run_id="sec_run",
            eval_suite_sha256=suite_hash,
        )


def test_interrupted_optimization_resumes_respecting_max_3_iterations(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    settings = _settings(tmp_path, min_eval_cases=5, max_optimization_iterations=3, target_pass_rate=0.95)
    brief = BriefInput(brief_text="Assist with inspection policy.", documents_dir=docs_dir, run_id="opt_resume")
    cases_5 = _make_cases(5)
    suite_hash = compute_eval_suite_hash(cases_5)
    agent_v1 = _sample_agent(version=1, run_id="opt_resume")

    # Baseline failed on 1 case; Iteration 1 proposed and failed mid-eval (only 2/5 done on iter1)
    store = RunCheckpointStore(settings.artifacts_dir, "opt_resume")
    store.save_requirements(_sample_requirements())
    store.save_groundability(_sample_groundability())
    store.save_agent(agent_v1)
    store.save_eval_cases(cases_5, suite_hash)

    baseline_client = _TrackingGeminiClient(
        lambda _p, _s, case_ids: JudgeLLMBatchVerdict(
            verdicts=[
                _make_verdict_item(
                    cid,
                    grounded=(cid != "case_01"),
                    grounding_score=0.2 if cid == "case_01" else 1.0,
                    correctness_score=0.2 if cid == "case_01" else 1.0,
                )
                for cid in case_ids
            ]
        )
    )
    base_res, base_sc = run_evaluation_suite(
        agent_v1,
        cases_5,
        units,
        gemini_client=baseline_client,
        embedding_provider=_DeterministicEmbedder(),
        checkpoint_dir=store.run_dir,
        run_id="opt_resume",
        eval_suite_sha256=suite_hash,
    )
    store.save_evaluation_outputs(
        iteration=0,
        agent=agent_v1,
        results=base_res,
        scorecard=base_sc,
        all_scorecards=[base_sc],
    )
    store.sync_from_state(
        {
            "brief_input": brief,
            "documents": docs,
            "agent_definition": agent_v1,
            "baseline_agent": agent_v1,
            "best_agent": agent_v1,
            "eval_suite_sha256": suite_hash,
            "min_eval_cases": 5,
            "max_iterations": 3,
            "target_pass_rate": 0.95,
        },
        status="evaluated_iter_0",
    )

    # First run: iteration 1 creates v2 candidate, then fails mid-iteration-1 eval;
    # run 1st two cases into eval_progress_iter1_v2.jsonl so only 3 cases remain on resume!
    prop_counter = {"n": 0}

    def opt_fn_interrupt(_prompt: str, schema: type[Any], case_ids: list[str]) -> Any:
        if schema is OptimizationProposalDraft:
            prop_counter["n"] += 1
            return OptimizationProposalDraft(
                actionable=True,
                diagnosis_summary=f"Refine prompt #{prop_counter['n']}.",
                proposed_changes=[
                    TargetedProposalItem(
                        target_component="system_prompt",
                        failure_driver=FailureClassification.HALLUCINATION,
                        rationale="Fix case_01 grounding",
                        summary=f"Prompt rule #{prop_counter['n']}.",
                        prompt_instruction_addition=f"Rule {prop_counter['n']}: state 30 days clearly.",
                    )
                ],
            )
        if schema is JudgeLLMBatchVerdict:
            raise RuntimeError("503 UNAVAILABLE during iter 1")
        raise AssertionError(schema)

    client_int = _TrackingGeminiClient(opt_fn_interrupt)
    graph_int = build_factory_graph(
        settings=settings,
        gemini_client=client_int,
        embedding_provider=_DeterministicEmbedder(),
    )
    loaded_start = load_run_checkpoint(store.run_dir)
    loaded_start["document_units"] = units
    loaded_start["stop_after_phase4"] = False
    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        graph_int.invoke(loaded_start)

    # Candidate v2 and pending_proposal are now checkpointed at iteration=1!
    loaded_mid = load_run_checkpoint(store.run_dir)
    assert loaded_mid["iteration"] == 1
    assert loaded_mid["agent_definition"].version == 2
    assert loaded_mid["pending_proposal"] is not None

    # Seed 2 completed cases for iteration 1 v2 before resuming so iter 1 finishes with only 3 missing cases,
    # then iterations 2 and 3 fail to reach 0.95 (so hard limit stops cleanly at iteration 3)!
    agent_v2 = loaded_mid["agent_definition"]
    from forgeai.workflow.checkpoints import EvalProgressRecord as _EPR2, append_progress_record as _apr2

    for res_item in base_res[1:3]:
        _apr2(
            progress_file_for(store.run_dir, iteration=1, agent_version=agent_v2.version),
            _EPR2(
                run_id="opt_resume",
                iteration=1,
                agent_version=agent_v2.version,
                eval_suite_sha256=suite_hash,
                case_id=res_item.case_id,
                result=res_item,
            ),
        )

    # Resume end-to-end via CLI resume_factory_cli -> must NOT re-propose iter 1, must run ONLY 3 missing
    # cases on iter 1, then run iter 2 & iter 3 and stop strictly at 3 total optimization iterations!
    resume_prop = {"n": 1}

    def opt_fn_finish(_prompt: str, schema: type[Any], case_ids: list[str]) -> Any:
        if schema is OptimizationProposalDraft:
            resume_prop["n"] += 1
            return OptimizationProposalDraft(
                actionable=True,
                diagnosis_summary=f"Refine prompt #{resume_prop['n']}.",
                proposed_changes=[
                    TargetedProposalItem(
                        target_component="system_prompt",
                        failure_driver=FailureClassification.HALLUCINATION,
                        rationale="Refine instruction for grounding",
                        summary=f"Prompt rule #{resume_prop['n']}.",
                        prompt_instruction_addition=f"Rule {resume_prop['n']}.",
                    )
                ],
            )
        if schema is JudgeLLMBatchVerdict:
            # Keep case_01 failing so all 3 iterations run cleanly and stop at max 3
            return JudgeLLMBatchVerdict(
                verdicts=[
                    _make_verdict_item(
                        cid,
                        grounded=(cid != "case_01"),
                        grounding_score=0.3 if cid == "case_01" else 1.0,
                        correctness_score=0.3 if cid == "case_01" else 1.0,
                    )
                    for cid in case_ids
                ]
            )
        raise AssertionError(schema)

    client_resume = _TrackingGeminiClient(opt_fn_finish)
    final_state = resume_factory_cli(
        resume_dir=store.run_dir,
        settings=settings,
        gemini_client=client_resume,
        embedding_provider=_DeterministicEmbedder(),
        execution_metadata={"execution_mode": "mocked_resume_test", "live_gemini_verified": False},
    )
    assert final_state["stop_reason"] == "max_iterations_exhausted"
    assert len(final_state["optimization_logs"]) == 3
    assert [log.iteration for log in final_state["optimization_logs"]] == [1, 2, 3]
    # Iteration 1 judge batch contained ONLY missing cases [case_01 (prior fail), case_05 (OUT_OF_SCOPE), case_04]
    assert client_resume.judge_case_id_batches[0] == ["case_01", "case_05", "case_04"]
    assert client_resume.structured_calls.count("OptimizationProposalDraft") == 2  # iter 2 & 3 only!
    assert (store.run_dir / "final_report.json").exists()

    # Re-resuming a completed run via CLI/graph is a 0-call no-op!
    client_noop = _TrackingGeminiClient(lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("No calls")))
    noop_state = resume_factory_cli(
        resume_dir=store.run_dir,
        settings=settings,
        gemini_client=client_noop,
        embedding_provider=_DeterministicEmbedder(),
    )
    assert noop_state["final_report"].stop_reason == "max_iterations_exhausted"
    assert client_noop.structured_calls == []
    assert client_noop.text_prompts == []


def test_streamlit_resume_ui_under_apptest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docs_dir = _create_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    settings = _settings(tmp_path, min_eval_cases=5)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("ARTIFACTS_DIR", str(settings.artifacts_dir))

    brief = BriefInput(brief_text="Policy Q&A", documents_dir=docs_dir, run_id="ui_resume_demo")
    cases_20 = _make_cases(20)
    suite_hash = compute_eval_suite_hash(cases_20)
    agent = _sample_agent(version=1, run_id="ui_resume_demo")

    store = RunCheckpointStore(settings.artifacts_dir, "ui_resume_demo")
    store.save_requirements(_sample_requirements())
    store.save_groundability(_sample_groundability())
    store.save_agent(agent)
    store.save_eval_cases(cases_20, suite_hash)
    store.sync_from_state(
        {
            "brief_input": brief,
            "documents": docs,
            "agent_definition": agent,
            "eval_suite_sha256": suite_hash,
            "min_eval_cases": 20,
        },
        status="evaluating_iter_0",
    )

    # Checkpoint 12 out of 20 cases cleanly -> 8 remaining
    from forgeai.evaluation.judge import HybridJudge as _HJ3
    from forgeai.runner.agent_runner import AgentResponse as _AR3
    from forgeai.schemas.evaluation import RetrievedEvidence as _RE3
    from forgeai.workflow.checkpoints import EvalProgressRecord as _EPR3, append_progress_record as _apr3

    ui_judge = _HJ3(
        gemini_client=_TrackingGeminiClient(
            lambda _p, _s, cids: JudgeLLMBatchVerdict(verdicts=[_make_verdict_item(cid) for cid in cids])
        ),
        known_sources={"policy.md"},
    )
    ui_verdicts = ui_judge.evaluate_batch(
        [
            (
                c,
                _AR3(
                    answer=(
                        "I cannot answer this request based on the provided documents and scope."
                        if c.expected_behavior == ExpectedBehavior.REFUSE
                        else "Inspection occurs every 30 days [policy.md | document]."
                    ),
                    refused=(c.expected_behavior == ExpectedBehavior.REFUSE),
                    evidence=[
                        _RE3(
                            chunk_id=f"u_{c.case_id}",
                            doc_id="d1",
                            source_file="policy.md",
                            unit_label="document",
                            text_excerpt="Inspection occurs every 30 days.",
                        )
                    ],
                    citations=([] if c.expected_behavior == ExpectedBehavior.REFUSE else ["policy.md (document)"]),
                    tools_used=["retrieve_documents"],
                ),
            )
            for c in cases_20[:12]
        ],
        agent.output_contract,
        known_sources={"policy.md"},
        batch_size=5,
    )
    prog_ui = progress_file_for(store.run_dir, iteration=0, agent_version=1)
    for uv in ui_verdicts:
        _apr3(
            prog_ui,
            _EPR3(
                run_id="ui_resume_demo",
                iteration=0,
                agent_version=1,
                eval_suite_sha256=suite_hash,
                case_id=uv.case_id,
                result=uv,
            ),
        )

    runs = list_saved_runs(settings.artifacts_dir)
    assert len(runs) == 1
    assert runs[0]["completed_cases"] == 12
    assert runs[0]["remaining_cases"] == 8

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception

    assert any(sb.label == "Saved Run ID" for sb in at.selectbox)
    load_btn = next(b for b in at.button if b.label == "Load Selected Run")
    assert load_btn.disabled is False
    load_btn.click().run(timeout=30)
    assert not at.exception
    assert at.session_state["factory_state"]["run_id"] == "ui_resume_demo"
    assert any("12/20" in info.value and "8" in info.value for info in at.info)

    # Simultaneous resume blocked when active_operation is set
    at_busy = AppTest.from_file(app_path)
    at_busy.session_state["active_operation"] = "run_evaluations"
    at_busy.run(timeout=30)
    assert next(b for b in at_busy.button if b.label == "Load Selected Run").disabled is True


def test_cli_resume_subcommand_missing_dir_returns_error(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    rc = cli_main(["run", "--resume", str(missing_dir), "--out", str(tmp_path / "out")])
    assert rc == 1
    assert (tmp_path / "out" / "run_error.json").exists()


def test_exact_12_of_20_checkpoint_resume_verification(tmp_path: Path) -> None:
    """Verify exact interruption after 12/20 completed cases resumes only cases 13-20 (8 remaining)."""
    docs_dir = _create_corpus(tmp_path)
    docs, units = load_document_folder(docs_dir)
    settings = _settings(tmp_path, min_eval_cases=20, max_optimization_iterations=0, judge_batch_size=5)
    brief = BriefInput(
        brief_text="Build a grounded policy assistant using policy.md.",
        documents_dir=docs_dir,
        run_id="exact_12_of_20",
    )
    cases_20 = _make_cases(20)
    suite_hash = compute_eval_suite_hash(cases_20)
    agent_v1 = _sample_agent(version=1, run_id="exact_12_of_20")

    # Persist completed Phases 1-3 + eval_cases (20) + SHA256
    store = RunCheckpointStore(settings.artifacts_dir, "exact_12_of_20")
    store.save_requirements(_sample_requirements())
    store.save_groundability(_sample_groundability())
    store.save_agent(agent_v1)
    store.save_eval_cases(cases_20, suite_hash)
    store.sync_from_state(
        {
            "brief_input": brief,
            "documents": docs,
            "agent_definition": agent_v1,
            "eval_suite_sha256": suite_hash,
            "min_eval_cases": 20,
            "max_iterations": 0,
            "interview_completed": True,
        },
        status="evaluating_iter_0",
    )

    # Simulate exact interruption after 12 completed cases (case_01..case_12 persisted in JSONL)
    from forgeai.evaluation.judge import HybridJudge
    from forgeai.runner.agent_runner import AgentResponse
    from forgeai.schemas.evaluation import RetrievedEvidence
    from forgeai.workflow.checkpoints import (
        EvalProgressRecord,
        append_progress_record,
        load_progress_records,
    )

    seed_judge = HybridJudge(
        gemini_client=_TrackingGeminiClient(
            lambda _p, _s, case_ids: JudgeLLMBatchVerdict(
                verdicts=[_make_verdict_item(cid) for cid in case_ids]
            )
        ),
        known_sources={"policy.md"},
    )
    seed_items = [
        (
            c,
            AgentResponse(
                answer=(
                    "I cannot answer this request based on the provided documents and scope."
                    if c.expected_behavior == ExpectedBehavior.REFUSE
                    else "Inspection occurs every 30 days [policy.md | document]."
                ),
                refused=(c.expected_behavior == ExpectedBehavior.REFUSE),
                evidence=[
                    RetrievedEvidence(
                        chunk_id=f"seed_{c.case_id}",
                        doc_id="doc1",
                        source_file="policy.md",
                        unit_label="document",
                        text_excerpt="Inspection occurs every 30 days.",
                    )
                ],
                citations=(
                    [] if c.expected_behavior == ExpectedBehavior.REFUSE else ["policy.md (document)"]
                ),
                tools_used=["retrieve_documents"],
            ),
        )
        for c in cases_20[:12]
    ]
    seeded_verdicts = seed_judge.evaluate_batch(
        seed_items,
        agent_v1.output_contract,
        known_sources={"policy.md"},
        batch_size=5,
    )
    prog_path = progress_file_for(store.run_dir, iteration=0, agent_version=1)
    for res in seeded_verdicts:
        append_progress_record(
            prog_path,
            EvalProgressRecord(
                run_id="exact_12_of_20",
                iteration=0,
                agent_version=1,
                eval_suite_sha256=suite_hash,
                case_id=res.case_id,
                result=res,
            ),
        )

    # 1. Verify first 12 cases are loaded cleanly from checkpoint
    loaded_map = load_progress_records(
        prog_path,
        expected_run_id="exact_12_of_20",
        expected_iteration=0,
        expected_agent_version=1,
        expected_eval_suite_sha256=suite_hash,
        valid_case_ids={c.case_id for c in cases_20},
    )
    assert list(loaded_map.keys()) == [c.case_id for c in cases_20[:12]]
    checkpoint_state = load_run_checkpoint(store.run_dir)
    assert checkpoint_state["checkpoint_completed_cases"] == 12
    assert checkpoint_state["checkpoint_total_cases"] == 20

    # Resume end-to-end via CLI/workflow
    resume_client = _TrackingGeminiClient(
        lambda _p, _s, case_ids: JudgeLLMBatchVerdict(
            verdicts=[_make_verdict_item(cid) for cid in case_ids]
        )
    )
    final_state = resume_factory_cli(
        resume_dir=store.run_dir,
        settings=settings,
        gemini_client=resume_client,
        embedding_provider=_DeterministicEmbedder(),
        execution_metadata={"execution_mode": "mocked_12_of_20_resume", "live_gemini_verified": False},
    )

    # 2 & 3. None of the first 12 cases hit runner or judge; only cases 13-20 (8 calls) ran
    assert len(resume_client.text_prompts) == 8
    executed_questions = [c.question for c in cases_20[12:]]
    skipped_questions = [c.question for c in cases_20[:12]]
    for q in executed_questions:
        assert any(q in p for p in resume_client.text_prompts)
    for q in skipped_questions:
        assert not any(q in p for p in resume_client.text_prompts)

    # 4. Only missing cases (case_13..case_20) sent to judge in batches [5, 3]
    assert resume_client.structured_calls == ["JudgeLLMBatchVerdict", "JudgeLLMBatchVerdict"]
    assert resume_client.judge_case_id_batches == [
        ["case_13", "case_14", "case_15", "case_16", "case_17"],
        ["case_18", "case_19", "case_20"],
    ]

    # 5. Final scorecard contains all 20 cases in order
    assert len(final_state["eval_results"]) == 20
    assert [r.case_id for r in final_state["eval_results"]] == [c.case_id for c in cases_20]
    assert final_state["baseline_scorecard"].total_cases == 20
    assert final_state["final_report"].final_scorecard.total_cases == 20
    assert final_state["final_report"].final_pass_rate == 1.0

    # 6. SHA256 mismatch and different agent_version protections remain active
    with pytest.raises(EvalIntegrityError):
        run_evaluation_suite(
            agent_v1,
            cases_20,
            units,
            gemini_client=resume_client,
            embedding_provider=_DeterministicEmbedder(),
            checkpoint_dir=store.run_dir,
            run_id="exact_12_of_20",
            eval_suite_sha256="f" * 64,
        )

    agent_v2 = _sample_agent(version=2, run_id="exact_12_of_20")
    v2_client = _TrackingGeminiClient(
        lambda _p, _s, case_ids: JudgeLLMBatchVerdict(
            verdicts=[_make_verdict_item(cid) for cid in case_ids]
        )
    )
    run_evaluation_suite(
        agent_v2,
        cases_20,
        units,
        gemini_client=v2_client,
        embedding_provider=_DeterministicEmbedder(),
        checkpoint_dir=store.run_dir,
        run_id="exact_12_of_20",
        eval_suite_sha256=suite_hash,
        judge_batch_size=5,
    )
    # v2 cannot reuse v1's checkpointed results -> executes all 20 cases across 4 batches
    assert len(v2_client.text_prompts) == 20
    assert [len(b) for b in v2_client.judge_case_id_batches] == [5, 5, 5, 5]
