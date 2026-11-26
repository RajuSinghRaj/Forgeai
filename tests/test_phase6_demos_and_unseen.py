"""Phase 6 validation tests: both demonstrations, artifacts, generic runner, and unseen-brief domain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forgeai.cli import run_factory_cli
from forgeai.config import ForgeSettings
from forgeai.optimization import verify_eval_suite_integrity
from forgeai.retrieval import ChromaVectorStore, chunk_units, load_document_folder
from forgeai.runner import DeclarativeAgentRunner
from forgeai.schemas import (
    BriefInput,
    ClarifyingQuestion,
    ClarifyingQuestionBatch,
    DeclarativeAgentDefinition,
    EvalCase,
    EvalCaseCategory,
    EvalCaseResult,
    EvalSuiteDraft,
    ExpectedBehavior,
    FactoryRunReport,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    IterationScorecard,
    JudgeLLMVerdict,
    OutputContract,
    RetrievalStrategyDraft,
    StructuredRequirements,
    SystemPromptDraft,
    ToolSelectionDraft,
    ToolSelectionDraftItem,
)
from forgeai.tools import build_default_registry
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


class _ScriptedGemini:
    def __init__(self, structured_fn=None, text_fn=None) -> None:
        self._structured_fn = structured_fn
        self._text_fn = text_fn

    def generate_structured(self, prompt: str, response_schema: type[Any], **_kwargs: Any) -> Any:
        result = self._structured_fn(prompt, response_schema)
        assert isinstance(result, response_schema)
        return result

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        return self._text_fn(prompt, **kwargs)


def test_demo_corpora_and_synthetic_transparency_exist() -> None:
    root = Path(__file__).resolve().parent.parent
    assert (root / "demos" / "SYNTHETIC_DATA_NOTICE.md").exists()
    assert (root / "demos" / "hr_policy" / "synthesis_prompt.md").exists()
    assert (root / "demos" / "ship_spares" / "synthesis_prompt.md").exists()

    hr_docs, hr_units = load_document_folder(root / "demos" / "hr_policy" / "documents")
    assert len(hr_docs) == 5
    assert any(d.file_type == ".pdf" for d in hr_docs)
    assert len(hr_units) >= 7

    ship_docs, ship_units = load_document_folder(root / "demos" / "ship_spares" / "documents")
    assert len(ship_docs) == 3
    assert {d.file_type for d in ship_docs} == {".csv", ".pdf"}
    assert len(ship_units) >= 22


def test_committed_demo_artifacts_are_valid_distinct_and_runnable(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    hr_art = root / "artifacts" / "demos" / "hr_policy"
    ship_art = root / "artifacts" / "demos" / "ship_spares"

    required_files = [
        "brief.txt",
        "intake_questions.json",
        "unattended_assumptions.json",
        "requirements.json",
        "groundability.json",
        "baseline_agent.json",
        "eval_cases.json",
        "baseline_results.json",
        "baseline_scorecard.json",
        "optimization_history.json",
        "final_agent.json",
        "final_report.json",
        "run_metadata.json",
    ]
    for folder in (hr_art, ship_art):
        for fname in required_files:
            assert (folder / fname).exists(), f"Missing {folder / fname}"

    hr_agent = DeclarativeAgentDefinition.model_validate_json((hr_art / "final_agent.json").read_text(encoding="utf-8"))
    ship_agent = DeclarativeAgentDefinition.model_validate_json((ship_art / "final_agent.json").read_text(encoding="utf-8"))
    assert hr_agent.agent_id != ship_agent.agent_id

    # Tool-selection proof
    hr_tools = {t.name for t in hr_agent.tools}
    assert "retrieve_documents" in hr_tools
    assert "inspect_csv_records" not in hr_tools
    assert "calculate_math" not in hr_tools

    ship_tools = {t.name for t in ship_agent.tools}
    assert {"retrieve_documents", "inspect_csv_records", "calculate_math"}.issubset(ship_tools)

    # Validate >= 20 eval cases, scorecards, and SHA-256 suite integrity
    for folder, docs_subdir in (
        (hr_art, root / "demos" / "hr_policy" / "documents"),
        (ship_art, root / "demos" / "ship_spares" / "documents"),
    ):
        cases = [EvalCase.model_validate(item) for item in json.loads((folder / "eval_cases.json").read_text(encoding="utf-8"))]
        assert len(cases) >= 20
        assert {c.category for c in cases} == {
            EvalCaseCategory.ANSWERABLE,
            EvalCaseCategory.OUT_OF_SCOPE,
            EvalCaseCategory.EDGE_CASE,
        }
        report = FactoryRunReport.model_validate_json((folder / "final_report.json").read_text(encoding="utf-8"))
        verify_eval_suite_integrity(cases, report.eval_suite_sha256)
        IterationScorecard.model_validate_json((folder / "baseline_scorecard.json").read_text(encoding="utf-8"))
        results = [EvalCaseResult.model_validate(r) for r in json.loads((folder / "baseline_results.json").read_text(encoding="utf-8"))]
        assert len(results) == len(cases)

        # Verify generic DeclarativeAgentRunner loads final_agent.json and executes against indexed corpus
        _docs, units = load_document_folder(docs_subdir)
        store = ChromaVectorStore(
            collection_name=f"verify_{folder.name}",
            persist_dir=tmp_path / f"chroma_{folder.name}",
            embedding_provider=_DeterministicEmbedder(),
        )
        agent_def = DeclarativeAgentDefinition.model_validate_json((folder / "final_agent.json").read_text(encoding="utf-8"))
        store.index_chunks(chunk_units(units, agent_def.chunking))
        runner = DeclarativeAgentRunner.from_json_file(
            folder / "final_agent.json",
            vector_store=store,
            gemini_client=_ScriptedGemini(text_fn=lambda p, **_k: "Grounded response [source | unit]."),
            tool_registry=build_default_registry(vector_store=store, documents_dir=docs_subdir),
        )
        resp = runner.run(cases[0].question)
        assert resp.evidence
        assert resp.answer


def test_unseen_brief_unrelated_domain_and_unattended_policies(tmp_path: Path) -> None:
    """Validate the generic factory on a 3rd unseen domain (Lab Equipment Calibration)."""
    docs_dir = tmp_path / "lab_docs"
    docs_dir.mkdir()
    (docs_dir / "calibration_sop.md").write_text(
        "Pipettes must be calibrated every 90 days. Spectrophotometers require monthly zeroing.",
        encoding="utf-8",
    )
    (docs_dir / "lab_instruments.csv").write_text(
        "asset_id,instrument,interval_days,status\n"
        "LAB-01,Micropipette 100uL,90,calibrated\n"
        "LAB-02,UV Spectrophotometer,30,due_soon\n",
        encoding="utf-8",
    )
    unseen_brief = (
        "Build a laboratory calibration assistant for lab technicians. Answer calibration intervals "
        "and instrument status using the SOP and asset CSV, cite sources, and refuse equipment repair advice."
    )

    def structured_fn(_prompt: str, schema: type[Any]) -> Any:
        if schema is ClarifyingQuestionBatch:
            return ClarifyingQuestionBatch(
                questions=[
                    ClarifyingQuestion(
                        question_id="q1",
                        question="Should technicians be warned when an instrument status is due_soon?",
                        rationale="Clarify alert formatting.",
                        target_field="expected_response_behavior",
                    )
                ]
            )
        if schema is StructuredRequirements:
            return StructuredRequirements(
                domain_name="Lab Calibration",
                purpose="Answer equipment calibration schedules and asset status.",
                scope="Calibration intervals and instrument status records.",
                audience="Lab technicians",
                tone="Concise and accurate",
                must_answer_topics=["Calibration intervals", "Instrument status"],
                must_refuse_topics=["Hardware repair instructions"],
            )
        if schema is GroundabilityReport:
            return GroundabilityReport(
                overall_verdict=GroundabilityVerdict.SUPPORTED,
                can_proceed=True,
                findings=[
                    GroundabilityFinding(
                        topic="Calibration intervals",
                        verdict=GroundabilityVerdict.SUPPORTED,
                        supporting_sources=["calibration_sop.md"],
                        evidence_summary="SOP specifies 90-day pipette and monthly spectrophotometer intervals.",
                    ),
                    GroundabilityFinding(
                        topic="Instrument status",
                        verdict=GroundabilityVerdict.SUPPORTED,
                        supporting_sources=["lab_instruments.csv"],
                        evidence_summary="CSV tracks asset_id and status.",
                    ),
                ],
            )
        if schema is SystemPromptDraft:
            return SystemPromptDraft(
                agent_name="Lab Calibration Assistant",
                description="Answers calibration schedule and instrument status questions for lab technicians.",
                domain_summary="Laboratory equipment calibration reference.",
                system_prompt="Answer strictly from calibration SOP and CSV records with citations.",
            )
        if schema is RetrievalStrategyDraft:
            return RetrievalStrategyDraft(chunk_size=500, chunk_overlap=80, top_k=4)
        if schema is ToolSelectionDraft:
            return ToolSelectionDraft(
                selected_tools=[
                    ToolSelectionDraftItem(name="retrieve_documents", purpose="Search SOP.", reason="Grounded SOP lookup."),
                    ToolSelectionDraftItem(name="inspect_csv_records", purpose="Lookup asset rows.", reason="Corpus has CSV asset log."),
                ]
            )
        if schema is OutputContract:
            return OutputContract(response_format="markdown", require_citations=True)
        if schema is EvalSuiteDraft:
            distinct_questions = [
                ("lab_1", "How often must micropipettes be calibrated according to the SOP?", ["90"], ["calibration_sop.md"]),
                ("lab_2", "What maintenance interval applies to UV spectrophotometers?", ["monthly"], ["calibration_sop.md"]),
                ("lab_3", "What is the current calibration status and interval of asset LAB-01?", ["calibrated", "90"], ["lab_instruments.csv"]),
                ("lab_4", "Which instrument in the asset log is marked due_soon and requires 30-day checks?", ["LAB-02", "due_soon", "30"], ["lab_instruments.csv", "calibration_sop.md"]),
            ]
            cases = [
                EvalCase(
                    case_id=cid,
                    question=q,
                    category=EvalCaseCategory.ANSWERABLE,
                    expected_behavior=ExpectedBehavior.ANSWER,
                    expected_facts=facts,
                    source_documents=srcs,
                    topic="Calibration",
                )
                for cid, q, facts, srcs in distinct_questions
            ] + [
                EvalCase(
                    case_id="lab_oos",
                    question="How do I disassemble and solder the UV spectrophotometer lamp board?",
                    category=EvalCaseCategory.OUT_OF_SCOPE,
                    expected_behavior=ExpectedBehavior.REFUSE,
                    topic="Hardware repair instructions",
                )
            ]
            return EvalSuiteDraft(cases=cases)
        if schema is JudgeLLMVerdict:
            return JudgeLLMVerdict(
                grounded=True,
                grounding_score=1.0,
                correctness_score=1.0,
                addresses_question=True,
                refusal_appropriate=True,
                reasoning="Accurate and grounded.",
            )
        raise AssertionError(schema)

    def text_fn(prompt: str, **_kwargs: Any) -> str:
        if "solder" in prompt.lower():
            return "I cannot answer this request based on the provided documents and scope."
        return (
            "Micropipettes are calibrated every 90 days (LAB-01 calibrated), and UV "
            "spectrophotometers (LAB-02 due_soon, 30 days) require monthly zeroing "
            "[calibration_sop.md | document] [lab_instruments.csv | row_1]."
        )

    settings = ForgeSettings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="gemini-3.8-flash",
        artifacts_dir=tmp_path / "runs",
        min_eval_cases=5,
        max_optimization_iterations=1,
    )
    stub = _ScriptedGemini(structured_fn=structured_fn, text_fn=text_fn)
    embedder = _DeterministicEmbedder()

    # 1. Verify stop_for_clarification pauses without inventing answers
    graph = build_factory_graph(settings=settings, gemini_client=stub, embedding_provider=embedder)
    paused = graph.invoke(
        {
            "brief_input": BriefInput(brief_text=unseen_brief, documents_dir=docs_dir),
            "unattended_policy": "stop_for_clarification",
        }
    )
    assert paused["awaiting_user_answers"] is True
    assert len(paused["clarifying_questions"]) == 1
    assert "requirements" not in paused

    # 2. Verify proceed_with_assumptions runs end-to-end on the unseen domain
    out_dir = tmp_path / "unseen_bundle"
    state = run_factory_cli(
        brief_text=unseen_brief,
        docs_dir=docs_dir,
        out_dir=out_dir,
        run_id="unseen_lab",
        unattended_policy="proceed_with_assumptions",
        max_iterations=1,
        min_eval_cases=5,
        settings=settings,
        gemini_client=stub,
        embedding_provider=embedder,
        execution_metadata={"execution_mode": "mocked_unit_test", "live_gemini_verified": False},
    )
    assert state["unattended_assumptions"]
    assert state["final_report"].final_pass_rate == 1.0
    assert (out_dir / "final_agent.json").exists()
    assert (out_dir / "final_report.json").exists()

    # 3. Load generated unseen agent via generic DeclarativeAgentRunner and run a query
    _docs, units = load_document_folder(docs_dir)
    store = ChromaVectorStore(
        collection_name="unseen_verify",
        persist_dir=tmp_path / "chroma_unseen",
        embedding_provider=embedder,
    )
    store.index_chunks(chunk_units(units, state["agent_definition"].chunking))
    runner = DeclarativeAgentRunner.from_json_file(
        out_dir / "final_agent.json",
        vector_store=store,
        gemini_client=stub,
    )
    ans = runner.run("How often should pipettes be calibrated?")
    assert "90" in ans.answer
    assert ans.evidence

    # 4. Verify generic exact identifier lookup invokes inspect_csv_records when selected
    csv_runner = DeclarativeAgentRunner.from_json_file(
        out_dir / "final_agent.json",
        vector_store=store,
        gemini_client=stub,
        tool_registry=build_default_registry(vector_store=store, documents_dir=docs_dir),
    )
    id_ans = csv_runner.run("What is the calibration status of LAB-02?")
    assert "inspect_csv_records" in id_ans.tools_used
    assert any(ev.source_file == "lab_instruments.csv" and ev.unit_label == "row_2" for ev in id_ans.evidence)
