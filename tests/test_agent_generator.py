"""Deterministic Phase 3 unit tests for DeclarativeAgentGenerator, tools, persistence, and LangGraph."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgeai.config import ForgeSettings
from forgeai.generator import (
    DeclarativeAgentGenerator,
    GroundabilityApprovalError,
    ToolValidationError,
    load_agent_definition,
    save_agent_definition,
)
from forgeai.retrieval import load_document_folder
from forgeai.schemas import (
    BriefInput,
    DeclarativeAgentDefinition,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    MetadataFilterEntry,
    OutputContract,
    RetrievalStrategyDraft,
    StructuredRequirements,
    SystemPromptDraft,
    ToolSelectionDraft,
    ToolSelectionDraftItem,
)
from forgeai.tools import build_default_registry
from forgeai.workflow import build_factory_graph


class _StubGeminiClient:
    """Mock GeminiClient returning scripted structured responses."""

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


def _sample_requirements() -> StructuredRequirements:
    return StructuredRequirements(
        domain_name="Technical Support",
        purpose="Help field engineers verify equipment procedures and part tables.",
        scope="Maintenance schedules, spare-part lookups, and safety checks.",
        audience="Field technicians",
        tone="Direct, precise, and safety-conscious",
        must_answer_topics=["Inspection intervals", "Part lead times"],
        must_refuse_topics=["Unauthorized firmware modification"],
        expected_response_behavior="Respond in structured sections with source citations.",
        constraints=["Cite source_file and unit_label for every fact"],
    )


def _sample_groundability(*, can_proceed: bool = True, with_gaps: bool = False) -> GroundabilityReport:
    if with_gaps:
        return GroundabilityReport(
            overall_verdict=GroundabilityVerdict.UNSUPPORTED,
            can_proceed=can_proceed,
            approval_required=True,
            approved_with_gaps=can_proceed,
            findings=[
                GroundabilityFinding(
                    topic="Inspection intervals",
                    verdict=GroundabilityVerdict.SUPPORTED,
                    supporting_sources=["manual.md"],
                    evidence_summary="manual.md documents 30-day checks.",
                ),
                GroundabilityFinding(
                    topic="Part lead times",
                    verdict=GroundabilityVerdict.UNSUPPORTED,
                    supporting_sources=[],
                    evidence_summary="No lead-time table supplied.",
                    gap_description="Missing vendor lead-time data.",
                ),
            ],
            ungrounded_topics=["Part lead times"],
        )

    return GroundabilityReport(
        overall_verdict=GroundabilityVerdict.SUPPORTED,
        can_proceed=can_proceed,
        findings=[
            GroundabilityFinding(
                topic="Inspection intervals",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["manual.md"],
                evidence_summary="manual.md documents 30-day checks.",
            ),
            GroundabilityFinding(
                topic="Part lead times",
                verdict=GroundabilityVerdict.SUPPORTED,
                supporting_sources=["parts.csv"],
                evidence_summary="parts.csv lists lead_days per part.",
            ),
        ],
    )


def _create_corpus(tmp_path: Path, *, include_csv: bool = True) -> Path:
    docs_dir = tmp_path / "corpus"
    docs_dir.mkdir()
    (docs_dir / "manual.md").write_text(
        "Standard inspection interval is every 30 operating days.",
        encoding="utf-8",
    )
    if include_csv:
        (docs_dir / "parts.csv").write_text(
            "part_id,description,lead_days\nP-10,Seal kit,14\nP-20,Sensor,7\n",
            encoding="utf-8",
        )
    return docs_dir


def test_tool_registry_includes_four_safe_tools(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path, include_csv=True)
    registry = build_default_registry(documents_dir=docs_dir)
    specs = {s["name"] for s in registry.list_specs()}
    assert specs == {
        "retrieve_documents",
        "inspect_csv_records",
        "calculate_math",
        "list_corpus_documents",
    }

    math_out = registry.execute("calculate_math", expression="(14 + 7) * 2")
    assert math_out["result"] == 42.0

    with pytest.raises(ValueError):
        registry.execute("calculate_math", expression="__import__('os').system('echo bad')")

    files = registry.execute("list_corpus_documents")
    assert {f["file_name"] for f in files} == {"manual.md", "parts.csv"}


def test_chunking_validation_and_retrieval_strips_invented_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RetrievalStrategyDraft(chunk_size=400, chunk_overlap=400)

    docs_dir = _create_corpus(tmp_path, include_csv=True)
    docs, units = load_document_folder(docs_dir)
    stub = _StubGeminiClient(
        responses=[
            RetrievalStrategyDraft(
                chunk_size=700,
                chunk_overlap=100,
                csv_rows_per_unit=1,
                top_k=6,
                score_threshold=0.25,
                metadata_filters=[
                    MetadataFilterEntry(field="file_type", value=".csv"),
                    MetadataFilterEntry(field="source_file", value="hallucinated.pdf"),
                    MetadataFilterEntry(field="invented_key", value="anything"),
                ],
                require_source_citations=True,
                rationale="Balanced chunking for short manuals and CSV rows.",
            )
        ]
    )
    generator = DeclarativeAgentGenerator(
        gemini_client=stub,
        embedding_model="all-MiniLM-L6-v2",
    )
    chunking, retrieval, _rationale = generator.generate_retrieval_strategy(
        _sample_requirements(),
        docs,
        units,
        collection_name="col_test",
    )

    assert chunking.chunk_size == 700
    assert chunking.chunk_overlap == 100
    assert chunking.csv_rows_per_unit == 1
    assert retrieval.top_k == 6
    assert retrieval.embedding_model == "all-MiniLM-L6-v2"
    # Invented filename and invented field must be stripped
    assert retrieval.metadata_filters == {"file_type": ".csv"}


def test_tool_selection_rejects_nonexistent_and_enforces_least_privilege(tmp_path: Path) -> None:
    # 1. Nonexistent tool rejection
    docs_dir = _create_corpus(tmp_path, include_csv=False)
    docs, units = load_document_folder(docs_dir)
    registry = build_default_registry(documents_dir=docs_dir)

    bad_stub = _StubGeminiClient(
        responses=[
            ToolSelectionDraft(
                selected_tools=[
                    ToolSelectionDraftItem(
                        name="retrieve_documents",
                        purpose="Retrieve policy passages.",
                        reason="Required for grounded Q&A.",
                    ),
                    ToolSelectionDraftItem(
                        name="nonexistent_weather_api",
                        purpose="Fetch forecast.",
                        reason="Invented tool.",
                    ),
                ]
            )
        ]
    )
    generator = DeclarativeAgentGenerator(gemini_client=bad_stub, tool_registry=registry)
    with pytest.raises(ToolValidationError, match="nonexistent_weather_api"):
        generator.select_tools(_sample_requirements(), docs, units)

    # 2. Least-privilege: inspect_csv_records omitted when corpus has no CSVs; calculate_math not attached
    lp_stub = _StubGeminiClient(
        responses=[
            ToolSelectionDraft(
                selected_tools=[
                    ToolSelectionDraftItem(
                        name="retrieve_documents",
                        purpose="Retrieve manual passages.",
                        reason="Core document search.",
                    ),
                    ToolSelectionDraftItem(
                        name="inspect_csv_records",
                        purpose="Inspect CSV tables.",
                        reason="Model mistakenly suggested CSV tool on markdown-only corpus.",
                    ),
                ]
            )
        ]
    )
    lp_generator = DeclarativeAgentGenerator(gemini_client=lp_stub, tool_registry=registry)
    selected, _reasons = lp_generator.select_tools(_sample_requirements(), docs, units)
    assert [t.name for t in selected] == ["retrieve_documents"]


def test_complete_agent_generation_guardrails_serialization_and_persistence(
    tmp_path: Path,
) -> None:
    docs_dir = _create_corpus(tmp_path, include_csv=True)
    docs, units = load_document_folder(docs_dir)
    registry = build_default_registry(documents_dir=docs_dir)
    reqs = _sample_requirements()

    # Groundability approval enforcement
    blocked_gen = DeclarativeAgentGenerator(gemini_client=_StubGeminiClient(), tool_registry=registry)
    with pytest.raises(GroundabilityApprovalError):
        blocked_gen.generate(reqs, _sample_groundability(can_proceed=False), docs, units)

    # Full generation with approved gaps
    approved_gap_report = _sample_groundability(can_proceed=True, with_gaps=True)
    stub = _StubGeminiClient(
        responses=[
            SystemPromptDraft(
                agent_name="Field Operations Assistant",
                description="Grounded assistant for field technicians inspecting maintenance manuals and part records.",
                domain_summary="Equipment maintenance and spare-part operations.",
                system_prompt=(
                    "You are a precise technical assistant for field technicians. "
                    "Answer strictly from retrieved manual and CSV records with citations."
                ),
            ),
            RetrievalStrategyDraft(
                chunk_size=800,
                chunk_overlap=120,
                csv_rows_per_unit=1,
                top_k=5,
                score_threshold=0.2,
                metadata_filters=[],
            ),
            ToolSelectionDraft(
                selected_tools=[
                    ToolSelectionDraftItem(
                        name="retrieve_documents",
                        purpose="Search maintenance manuals and indexed records.",
                        reason="Needed for grounded retrieval.",
                    ),
                    ToolSelectionDraftItem(
                        name="inspect_csv_records",
                        purpose="Filter structured rows in parts.csv.",
                        reason="Corpus includes CSV part tables.",
                    ),
                ]
            ),
            OutputContract(
                response_format="structured_sections",
                require_citations=True,
                citation_format="[source_file | unit]",
                response_sections=["Answer", "Evidence", "Limitations"],
                required_fields=["summary", "source_citation"],
                optional_fields=["part_id"],
                formatting_constraints=["Max 300 words"],
                validation_rules=["Every factual claim must cite a source_file."],
            ),
        ]
    )
    generator = DeclarativeAgentGenerator(gemini_client=stub, tool_registry=registry)
    agent = generator.generate(reqs, approved_gap_report, docs, units, run_id="run-p3")

    assert agent.agent_id.startswith("field_operations_assistant_")
    assert "untrusted data" in agent.system_prompt.lower()
    assert "did not execute" in agent.system_prompt.lower()
    assert "Part lead times" in agent.system_prompt
    assert agent.approved_gaps == ["Part lead times"]
    assert len(agent.groundability_limitations) == 1
    assert agent.source_documents == ["manual.md", "parts.csv"]
    assert agent.approved_requirements == reqs
    assert agent.output_contract.required_fields == ["summary", "source_citation"]
    assert [t.name for t in agent.tools] == ["retrieve_documents", "inspect_csv_records"]

    # Persistence + collision protection + JSON reload round-trip
    artifacts_dir = tmp_path / "artifacts"
    path1 = save_agent_definition(agent, artifacts_dir)
    path2 = save_agent_definition(agent, artifacts_dir)

    assert path1.exists()
    assert path2.exists()
    assert path1 != path2
    assert path2.stem.endswith("_1")

    reloaded: DeclarativeAgentDefinition = load_agent_definition(path1)
    assert reloaded == agent


def test_langgraph_phase3_integration_and_failure_handling(tmp_path: Path) -> None:
    docs_dir = _create_corpus(tmp_path, include_csv=True)
    brief = BriefInput(
        brief_text="Build a field operations assistant grounded in our maintenance manual and parts CSV.",
        documents_dir=docs_dir,
    )
    reqs = _sample_requirements()
    ground_ok = _sample_groundability(can_proceed=True)

    stub = _StubGeminiClient(
        responses=[
            ground_ok,
            SystemPromptDraft(
                agent_name="Maintenance Agent",
                description="Answers grounded maintenance and part queries for technicians.",
                domain_summary="Equipment maintenance operations.",
                system_prompt="Answer strictly from supplied documents and tools using citations.",
            ),
            RetrievalStrategyDraft(chunk_size=600, chunk_overlap=100, top_k=4),
            ToolSelectionDraft(
                selected_tools=[
                    ToolSelectionDraftItem(
                        name="retrieve_documents",
                        purpose="Retrieve maintenance passages.",
                        reason="Core grounded search.",
                    )
                ]
            ),
            OutputContract(
                response_format="markdown",
                required_fields=["answer", "citations"],
            ),
        ]
    )
    settings = ForgeSettings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="gemini-3.8-flash",
        artifacts_dir=tmp_path / "runs",
    )

    class _LocalTestEmbedder:
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

    embedder = _LocalTestEmbedder()

    graph = build_factory_graph(
        settings=settings,
        gemini_client=stub,
        embedding_provider=embedder,
    )

    result = graph.invoke(
        {
            "brief_input": brief,
            "requirements": reqs,
            "interview_completed": True,
            "stop_after_phase3": True,
        }
    )

    assert result["phase3_status"] == "completed"
    assert result["agent_definition"].name == "Maintenance Agent"
    artifact_path = Path(result["agent_artifact_path"])
    assert artifact_path.exists()
    assert load_agent_definition(artifact_path) == result["agent_definition"]

    # Blocked groundability must prevent generate_agent from running
    blocked_stub = _StubGeminiClient(responses=[_sample_groundability(can_proceed=False, with_gaps=True)])
    blocked_graph = build_factory_graph(
        settings=settings,
        gemini_client=blocked_stub,
        embedding_provider=embedder,
    )
    blocked_state = blocked_graph.invoke(
        {
            "brief_input": brief,
            "requirements": reqs,
            "interview_completed": True,
            "stop_after_phase3": True,
        }
    )
    assert blocked_state["phase3_status"] == "blocked_by_groundability"
    assert "agent_definition" not in blocked_state

    # Gemini failure during agent generation raises descriptive RuntimeError
    failing_graph = build_factory_graph(
        settings=settings,
        gemini_client=_StubGeminiClient(error=RuntimeError("503 UNAVAILABLE")),
        embedding_provider=embedder,
        node_handlers={
            "groundability_guard": lambda _s: {"groundability": ground_ok, "phase2_approved": True},
        },
    )
    with pytest.raises(RuntimeError, match="Declarative agent generation failed"):
        failing_graph.invoke(
            {
                "brief_input": brief,
                "requirements": reqs,
                "groundability": ground_ok,
                "interview_completed": True,
                "stop_after_phase3": True,
            }
        )
