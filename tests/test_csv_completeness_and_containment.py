"""Regression tests for universal CSV completeness, truncation disclosure, and strict path containment."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pytest

from forgeai.retrieval import ChromaVectorStore, chunk_units, load_document_folder
from forgeai.runner.agent_runner import DeclarativeAgentRunner
from forgeai.schemas import (
    ChunkingConfig,
    DeclarativeAgentDefinition,
    OutputContract,
    RetrievalConfig,
    ToolSelection,
)
from forgeai.tools.registry import build_default_registry, count_csv_rows_contained, resolve_contained_csv_path


class _TokenEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "token-test-embedder"

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


class _PromptCapturingGemini:
    def __init__(self, reply: str = "Verified [fleet_nodes.csv].") -> None:
        self.prompts: list[str] = []
        self._reply = reply

    def generate_text(self, prompt: str, **_kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self._reply


def _write_csv(path: Path, row_count: int) -> None:
    lines = ["node_code,zone,status,capacity"]
    for i in range(1, row_count + 1):
        zone = "north" if i % 2 == 0 else "south"
        status = "active" if i <= max(1, row_count // 2) else "standby"
        lines.append(f"N-{i:03d},{zone},{status},{100 + i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_csv_runner(
    tmp_path: Path,
    row_count: int,
    *,
    gemini: _PromptCapturingGemini | None = None,
) -> tuple[DeclarativeAgentRunner, Path, _PromptCapturingGemini]:
    docs_dir = tmp_path / f"run_csv_{row_count}"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(docs_dir / "fleet_nodes.csv", row_count)
    (docs_dir / "notes.md").write_text(
        "Fleet inspection rules require logging sensor health every cycle.",
        encoding="utf-8",
    )

    _, units = load_document_folder(docs_dir, csv_rows_per_unit=1)
    agent = DeclarativeAgentDefinition(
        agent_id=f"fleet_{row_count}",
        name="Fleet Inspector",
        version=1,
        domain_summary="Grounded fleet node and inspection assistant.",
        system_prompt="Answer strictly from supplied documents and CSV records.",
        chunking=ChunkingConfig(chunk_size=600, chunk_overlap=100, csv_rows_per_unit=1),
        retrieval=RetrievalConfig(collection_name=f"col_fleet_{row_count}", top_k=3),
        tools=[
            ToolSelection(name="retrieve_documents", purpose="Retrieve documents."),
            ToolSelection(name="inspect_csv_records", purpose="Inspect CSV tables."),
        ],
        output_contract=OutputContract(
            response_format="markdown",
            require_citations=True,
            citation_format="[source_file | unit]",
            required_fields=[],
        ),
        source_documents=["fleet_nodes.csv", "notes.md"],
        must_answer_topics=["Fleet nodes"],
        must_refuse_topics=["External budgets"],
    )
    store = ChromaVectorStore(
        collection_name=agent.retrieval.collection_name,
        persist_dir=None,
        embedding_provider=_TokenEmbedder(),
    )
    store.index_chunks(chunk_units(units, agent.chunking))
    client = gemini or _PromptCapturingGemini()
    runner = DeclarativeAgentRunner(
        agent,
        vector_store=store,
        gemini_client=client,
        documents_dir=docs_dir,
    )
    return runner, docs_dir, client


@pytest.mark.parametrize("row_count", [10, 15, 25])
def test_csv_10_15_25_rows_exhaustive_retrieval_is_complete_without_false_truncation(
    tmp_path: Path,
    row_count: int,
) -> None:
    """10-, 15-, and 25-row CSVs retrieve all rows via safe pagination with zero false truncation."""
    runner, _docs_dir, client = _build_csv_runner(tmp_path, row_count)
    resp = runner.run("List all nodes across the entire fleet_nodes table.")
    row_evs = [ev for ev in resp.evidence if ev.source_file == "fleet_nodes.csv" and ev.unit_label.startswith("row_")]
    notices = [ev for ev in resp.evidence if ev.unit_label == "truncation_notice"]

    assert len(row_evs) == row_count
    assert not notices
    assert f"N-{row_count:03d}" in client.prompts[0]


def test_csv_100_rows_discloses_truncation_honestly_without_loading_full_table(
    tmp_path: Path,
) -> None:
    """100-row CSV stays bounded cleanly, attaches explicit total-row disclosure, and preserves targeted queries."""
    gemini = _PromptCapturingGemini(
        reply="Retrieved 30 of 100 total rows from fleet_nodes.csv; remaining 70 rows were not loaded [fleet_nodes.csv | truncation_notice]."
    )
    runner, _docs_dir, _ = _build_csv_runner(tmp_path, 100, gemini=gemini)

    # 1. Exhaustive request loads bounded 30-row slice + explicit truncation indicator with total=100
    resp_all = runner.run("Show all entries in the complete fleet_nodes table.")
    row_evs = [ev for ev in resp_all.evidence if ev.source_file == "fleet_nodes.csv" and ev.unit_label.startswith("row_")]
    notices = [ev for ev in resp_all.evidence if ev.unit_label == "truncation_notice"]

    assert len(row_evs) == 30
    assert len(notices) == 1
    assert "30 of 100 total rows" in notices[0].text_excerpt
    assert "70 additional rows" in notices[0].text_excerpt
    assert "DO NOT present this subset as the complete table" in notices[0].text_excerpt
    assert "DO NOT" in gemini.prompts[0] and "truncation_notice" in gemini.prompts[0]
    assert "100 total rows" in resp_all.answer

    # 2. Targeted identifier request retrieves exact single row efficiently with NO truncation warning
    resp_one = runner.run("What is the status and capacity for node N-087?")
    one_rows = [ev for ev in resp_one.evidence if ev.source_file == "fleet_nodes.csv" and ev.unit_label == "row_87"]
    assert len(one_rows) == 1
    assert "N-087" in one_rows[0].text_excerpt
    assert not any(ev.unit_label == "truncation_notice" for ev in resp_one.evidence)


def test_csv_25_rows_targeted_filter_truncation_discloses_match_and_total_counts(
    tmp_path: Path,
) -> None:
    """Targeted keyword lookup hitting >8 matching rows in a 25-row CSV discloses truncation cleanly."""
    runner, _docs_dir, client = _build_csv_runner(tmp_path, 25)
    resp = runner.run("Which fleet_nodes rows have north zone status?")
    notices = [ev for ev in resp.evidence if ev.unit_label == "truncation_notice"]
    assert len(notices) == 1
    assert "25 total rows" in notices[0].text_excerpt
    assert "partial scope" in notices[0].text_excerpt
    assert "TRUNCATED CSV TABLE" in client.prompts[0]


def test_csv_path_containment_blocks_traversal_sibling_prefix_and_symlinks(
    tmp_path: Path,
) -> None:
    """Path containment permits nested valid CSVs while blocking traversal, prefix tricks, and escaping symlinks."""
    active_dir = tmp_path / "run_1"
    sibling_prefix_dir = tmp_path / "run_1_secret"
    outside_dir = tmp_path / "outside"
    active_dir.mkdir()
    sibling_prefix_dir.mkdir()
    outside_dir.mkdir()

    (active_dir / "valid.csv").write_text("k,v\nA,1\n", encoding="utf-8")
    (active_dir / "nested").mkdir()
    (active_dir / "nested" / "sub.csv").write_text("k,v\nB,2\n", encoding="utf-8")
    (sibling_prefix_dir / "leak.csv").write_text("k,v\nSECRET,999\n", encoding="utf-8")
    (outside_dir / "ext.csv").write_text("k,v\nEXT,888\n", encoding="utf-8")

    registry = build_default_registry(documents_dir=active_dir)

    # Valid root and nested paths inside active_dir work
    assert registry.execute("inspect_csv_records", file_name="valid.csv")[0]["k"] == "A"
    assert registry.execute("inspect_csv_records", file_name="nested/sub.csv")[0]["k"] == "B"

    # Parent traversal & sibling-directory prefix attacks MUST raise ValueError
    for bad_path in (
        "../outside/ext.csv",
        "nested/../../outside/ext.csv",
        "../run_1_secret/leak.csv",
        "../run_1_secret/nonexistent.csv",
        str(outside_dir / "ext.csv"),
    ):
        with pytest.raises(ValueError, match="within documents_dir"):
            resolve_contained_csv_path(active_dir, bad_path)
        with pytest.raises(ValueError, match="within documents_dir"):
            count_csv_rows_contained(active_dir, bad_path)
        with pytest.raises(ValueError, match="within documents_dir"):
            registry.execute("inspect_csv_records", file_name=bad_path)

    # External symlink inside active_dir must also be rejected (Windows requires Dev Mode/Admin for symlinks)
    symlink_path = active_dir / "escape_link.csv"
    try:
        os.symlink(outside_dir / "ext.csv", symlink_path)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match="within documents_dir"):
            resolve_contained_csv_path(active_dir, "escape_link.csv")
        with pytest.raises(ValueError, match="within documents_dir"):
            registry.execute("inspect_csv_records", file_name="escape_link.csv")

