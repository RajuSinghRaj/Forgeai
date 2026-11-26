"""Tests for PDF/CSV/Markdown ingestion, chunking, and persistent ChromaDB retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path

from forgeai.retrieval import ChromaVectorStore, chunk_units, load_document_folder
from forgeai.schemas import ChunkingConfig


def _write_minimal_pdf(path: Path, text: str) -> None:
    """Write a minimal single-page PDF containing readable ASCII text."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        ),
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        (
            f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1")
            + stream
            + b"\nendstream endobj\n"
        ),
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    path.write_bytes(bytes(pdf))


class _DeterministicTestEmbedder:
    """Fast, offline character-trigram hash embedder for unit tests."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "deterministic-test-embedder"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        lowered = text.lower()
        tokens = [lowered[i : i + 3] for i in range(max(1, len(lowered) - 2))]
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def test_load_chunk_index_and_query_pdf_csv_md(tmp_path: Path) -> None:
    docs_dir = tmp_path / "corpus"
    docs_dir.mkdir()

    _write_minimal_pdf(docs_dir / "manual.pdf", "Standard inspection interval is 30 days.")
    (docs_dir / "catalog.csv").write_text(
        "part_id,description,lead_days\nP-100,Hydraulic seal kit,14\nP-200,Thermal sensor,7\n",
        encoding="utf-8",
    )
    (docs_dir / "notes.md").write_text(
        "Escalation policy: contact duty supervisor within 24 hours for urgent delays.",
        encoding="utf-8",
    )

    docs, units = load_document_folder(docs_dir)
    assert len(docs) == 3
    assert {d.file_type for d in docs} == {".pdf", ".csv", ".md"}
    assert len(units) == 4  # 1 pdf page + 2 csv rows + 1 md document

    chunks = chunk_units(units, ChunkingConfig(chunk_size=200, chunk_overlap=30))
    assert len(chunks) >= 4
    assert all(c.doc_id and c.source_file and c.unit_label for c in chunks)

    store = ChromaVectorStore(
        collection_name="test_corpus",
        persist_dir=tmp_path / "chroma",
        embedding_provider=_DeterministicTestEmbedder(),
    )
    inserted = store.index_chunks(chunks)
    assert inserted == len(chunks)
    assert store.count() == len(chunks)

    results = store.query("Hydraulic seal kit lead days", top_k=2)
    assert len(results) == 2
    assert all(ev.source_file and ev.unit_label and ev.doc_id for ev in results)
    assert any(ev.source_file == "catalog.csv" for ev in results)

