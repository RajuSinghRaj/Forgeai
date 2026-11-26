"""Text chunking utilities that preserve document and source-unit provenance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forgeai.retrieval.loaders import DocumentUnit
from forgeai.schemas.agent import ChunkingConfig


class DocumentChunk(BaseModel):
    """A single indexable text chunk with full source traceability metadata."""

    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    file_type: str = Field(..., min_length=1)
    unit_label: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _split_text(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_pos = -1
            for sep in separators:
                candidate = cleaned.rfind(sep, start, end)
                if candidate > start:
                    split_pos = candidate + len(sep)
                    break
            if split_pos > start:
                end = split_pos

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def chunk_units(units: list[DocumentUnit], config: ChunkingConfig | None = None) -> list[DocumentChunk]:
    """Split extracted `DocumentUnit` items into `DocumentChunk` records."""
    cfg = config or ChunkingConfig()
    output: list[DocumentChunk] = []

    for unit in units:
        pieces = _split_text(
            unit.text,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.separators,
        )
        for chunk_idx, piece in enumerate(pieces):
            chunk_id = f"{unit.doc_id}:{unit.unit_label}:c{chunk_idx}"
            meta = {
                "doc_id": unit.doc_id,
                "source_file": unit.source_file,
                "file_type": unit.file_type,
                "unit_label": unit.unit_label,
                "chunk_index": chunk_idx,
            }
            for key, value in unit.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[key] = value

            output.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=unit.doc_id,
                    source_file=unit.source_file,
                    file_type=unit.file_type,
                    unit_label=unit.unit_label,
                    chunk_index=chunk_idx,
                    text=piece,
                    metadata=meta,
                )
            )
    return output

