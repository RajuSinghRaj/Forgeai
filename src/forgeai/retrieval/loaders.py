"""Domain-independent document loaders for PDF, CSV, and plain-text/Markdown files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forgeai.schemas.intake import DocumentInput

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md"}


class DocumentUnit(BaseModel):
    """Logical source unit extracted from a document (e.g., one PDF page or CSV row)."""

    doc_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    file_path: Path
    file_type: str = Field(..., min_length=1)
    unit_label: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _make_doc_id(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() else "_" for ch in relative_path.stem).strip("_")[:24] or "doc"
    return f"{stem}_{digest}"


def _load_pdf(path: Path, rel_path: Path, doc_id: str) -> list[DocumentUnit]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    units: list[DocumentUnit] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        units.append(
            DocumentUnit(
                doc_id=doc_id,
                source_file=rel_path.as_posix(),
                file_path=path,
                file_type=".pdf",
                unit_label=f"page_{idx}",
                text=text,
                metadata={"page_number": idx, "total_pages": len(reader.pages)},
            )
        )
    return units


def _load_csv(path: Path, rel_path: Path, doc_id: str, rows_per_unit: int = 1) -> list[DocumentUnit]:
    units: list[DocumentUnit] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        batch: list[tuple[int, str]] = []

        for row_num, row in enumerate(reader, start=1):
            parts = [f"{k}: {(v or '').strip()}" for k, v in row.items() if k is not None]
            row_text = " | ".join(parts).strip()
            if not row_text:
                continue
            batch.append((row_num, row_text))

            if len(batch) >= rows_per_unit:
                units.append(_build_csv_unit(doc_id, rel_path, path, columns, batch))
                batch = []

        if batch:
            units.append(_build_csv_unit(doc_id, rel_path, path, columns, batch))
    return units


def _build_csv_unit(
    doc_id: str,
    rel_path: Path,
    path: Path,
    columns: list[str],
    batch: list[tuple[int, str]],
) -> DocumentUnit:
    start_row = batch[0][0]
    end_row = batch[-1][0]
    unit_label = f"row_{start_row}" if start_row == end_row else f"rows_{start_row}_{end_row}"
    body = "\n".join(f"Row {num}: {text}" for num, text in batch)
    return DocumentUnit(
        doc_id=doc_id,
        source_file=rel_path.as_posix(),
        file_path=path,
        file_type=".csv",
        unit_label=unit_label,
        text=body,
        metadata={"row_start": start_row, "row_end": end_row, "columns": columns},
    )


def _load_text(path: Path, rel_path: Path, doc_id: str, ext: str) -> list[DocumentUnit]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    return [
        DocumentUnit(
            doc_id=doc_id,
            source_file=rel_path.as_posix(),
            file_path=path,
            file_type=ext,
            unit_label="document",
            text=text,
            metadata={},
        )
    ]


def load_document_folder(
    documents_dir: Path,
    *,
    csv_rows_per_unit: int = 1,
) -> tuple[list[DocumentInput], list[DocumentUnit]]:
    """Scan `documents_dir` recursively and extract traceable `DocumentUnit` objects."""
    root = Path(documents_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Document folder does not exist or is not a directory: {root}")

    doc_inputs: list[DocumentInput] = []
    all_units: list[DocumentUnit] = []

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        rel_path = path.relative_to(root)
        doc_id = _make_doc_id(rel_path)

        if ext == ".pdf":
            units = _load_pdf(path, rel_path, doc_id)
        elif ext == ".csv":
            units = _load_csv(path, rel_path, doc_id, rows_per_unit=csv_rows_per_unit)
        else:
            units = _load_text(path, rel_path, doc_id, ext)

        doc_inputs.append(
            DocumentInput(
                doc_id=doc_id,
                file_name=rel_path.as_posix(),
                file_path=path,
                file_type=ext,
                size_bytes=path.stat().st_size,
                unit_count=len(units),
            )
        )
        all_units.extend(units)

    return doc_inputs, all_units

