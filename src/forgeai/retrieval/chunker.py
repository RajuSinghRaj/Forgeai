"""Text chunking utilities that preserve document and source-unit provenance."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from forgeai.retrieval.loaders import DocumentUnit
from forgeai.schemas.agent import ChunkingConfig

_HEADING_SPLIT_RE = re.compile(
    r"\n(?=(?:#{1,6}\s+\S|\d+(?:\.\d+)*[.)]\s+[A-Z]|[A-Z][A-Za-z0-9 &/-]{2,55}:\s*(?:\n|$)))"
)
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|\n)")
_HEADING_LINE_RE = re.compile(
    r"^(?:#{1,6}\s+(.+)|(\d+(?:\.\d+)*[.)]\s+[A-Z][^\n]{1,70})|([A-Z][A-Za-z0-9 &/-]{2,55}:))$"
)


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


def _detect_section_heading(text: str) -> str | None:
    """Return the most prominent leading section/numbered heading found near the top of `text`."""
    for raw_line in text.splitlines()[:5]:
        line = raw_line.strip()
        if not line:
            continue
        match = _HEADING_LINE_RE.match(line)
        if match:
            heading = next((g for g in match.groups() if g), "").strip().rstrip(":")
            if heading:
                return heading[:100]
    return None


def _choose_split_end(
    cleaned: str,
    start: int,
    max_end: int,
    separators: list[str],
) -> int:
    """Select the cleanest split index <= `max_end`, preferring section and sentence boundaries."""
    min_split = start + max(1, (max_end - start) // 3)

    # 1. Section / paragraph boundaries ("\n\n" or heading start)
    para_pos = cleaned.rfind("\n\n", min_split, max_end)
    if para_pos > start:
        return para_pos + 2
    heading_positions = [m.start() + 1 for m in _HEADING_SPLIT_RE.finditer(cleaned, min_split, max_end)]
    if heading_positions:
        return heading_positions[-1]

    # 2. Complete sentence boundaries inside the outer window
    sentence_positions = [m.end() for m in _SENTENCE_END_RE.finditer(cleaned, start, max_end)]
    valid_sentences = [pos for pos in sentence_positions if pos >= min_split]
    if valid_sentences:
        return valid_sentences[-1]

    # 3. Caller-supplied separators / line / word boundaries
    for sep in separators:
        if sep in ("\n\n", ". "):
            continue
        candidate = cleaned.rfind(sep, start, max_end)
        if candidate > start:
            return candidate + len(sep)

    # 4. Last resort: any whitespace before max_end so we never split mid-word if avoidable
    ws = max(cleaned.rfind(" ", start, max_end), cleaned.rfind("\n", start, max_end))
    if ws > start:
        return ws + 1
    return max_end


def _snap_overlap_start(cleaned: str, start: int, end: int, chunk_overlap: int) -> int:
    """Advance `end - chunk_overlap` to a sentence start or word boundary so chunks never begin mid-word."""
    if chunk_overlap <= 0:
        return end
    raw_next = max(end - chunk_overlap, start + 1)
    if raw_next >= end:
        return end

    # Prefer starting overlap cleanly at a sentence start inside [raw_next, end)
    for m in _SENTENCE_END_RE.finditer(cleaned, raw_next, end):
        if m.end() > start and m.end() < end:
            return m.end()

    # Otherwise snap to the start of the next word if raw_next landed inside a token
    pos = raw_next
    if pos > 0 and not cleaned[pos - 1].isspace():
        while pos < end and not cleaned[pos].isspace():
            pos += 1
        while pos < end and cleaned[pos].isspace():
            pos += 1
    elif pos < end and cleaned[pos].isspace():
        while pos < end and cleaned[pos].isspace():
            pos += 1

    if pos <= start or pos >= end:
        return end
    return pos


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
        max_end = min(start + chunk_size, length)
        end = max_end if max_end >= length else _choose_split_end(cleaned, start, max_end, separators)

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = _snap_overlap_start(cleaned, start, end, chunk_overlap)

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
        current_section = _detect_section_heading(unit.text)
        for chunk_idx, piece in enumerate(pieces):
            chunk_section = _detect_section_heading(piece) or current_section
            if chunk_section:
                current_section = chunk_section
            chunk_id = f"{unit.doc_id}:{unit.unit_label}:c{chunk_idx}"
            meta = {
                "doc_id": unit.doc_id,
                "source_file": unit.source_file,
                "file_type": unit.file_type,
                "unit_label": unit.unit_label,
                "chunk_index": chunk_idx,
            }
            if chunk_section:
                meta["section"] = chunk_section
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

