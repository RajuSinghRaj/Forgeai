"""Validated universal document upload handling and per-run corpus isolation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from forgeai.retrieval.loaders import DocumentUnit, load_document_folder
from forgeai.schemas.intake import DocumentInput

ALLOWED_UPLOAD_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv")
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_FILES = 50

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadValidationError(ValueError):
    """Raised when uploaded files fail type, emptiness, size, or parsing checks."""


class UploadedFilePayload(BaseModel):
    """Normalized in-memory representation of a browser-uploaded document."""

    filename: str = Field(..., min_length=1)
    content: bytes


class UploadedCorpusResult(BaseModel):
    """Validated, isolated uploaded document corpus associated with a single ForgeAI run."""

    run_id: str = Field(..., min_length=1)
    documents_dir: Path
    corpus_sha256: str = Field(..., min_length=16)
    documents: list[DocumentInput] = Field(default_factory=list)
    units: list[DocumentUnit] = Field(default_factory=list)
    saved_files: list[str] = Field(default_factory=list)
    renamed_files: dict[str, str] = Field(default_factory=dict)
    total_bytes: int = Field(default=0, ge=0)


def coerce_uploaded_files(files: Iterable[Any]) -> list[UploadedFilePayload]:
    """Convert Streamlit `UploadedFile` objects, pairs, or payloads cleanly into normalized payloads."""
    normalized: list[UploadedFilePayload] = []
    for item in files:
        if isinstance(item, UploadedFilePayload):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(
                UploadedFilePayload(filename=str(item["filename"]), content=bytes(item["content"]))
            )
        elif isinstance(item, tuple) and len(item) == 2:
            normalized.append(UploadedFilePayload(filename=str(item[0]), content=bytes(item[1])))
        elif hasattr(item, "name") and hasattr(item, "getvalue"):
            normalized.append(
                UploadedFilePayload(filename=str(item.name), content=bytes(item.getvalue()))
            )
        else:
            raise UploadValidationError(f"Unsupported upload object type: {type(item).__name__}")
    return normalized


def sanitize_and_deduplicate_filename(raw_name: str, used_lower_names: set[str]) -> str:
    """Return a traversal-safe filename preserving extension and resolving collisions cleanly."""
    base_name = Path(raw_name.replace("\\", "/")).name.strip()
    if not base_name:
        raise UploadValidationError("Uploaded file must have a non-empty filename.")

    suffix = Path(base_name).suffix.lower()
    stem = Path(base_name).stem
    safe_stem = _SAFE_STEM_RE.sub("_", stem).strip("._-") or "document"
    candidate = f"{safe_stem}{suffix}"

    counter = 1
    while candidate.lower() in used_lower_names:
        candidate = f"{safe_stem}_{counter}{suffix}"
        counter += 1

    used_lower_names.add(candidate.lower())
    return candidate


def prepare_uploaded_corpus(
    files: Iterable[Any],
    *,
    artifacts_dir: Path | str = Path("artifacts/runs"),
    run_id: str | None = None,
    csv_rows_per_unit: int = 1,
    max_files: int = MAX_UPLOAD_FILES,
    max_file_bytes: int = MAX_FILE_SIZE_BYTES,
    max_total_bytes: int = MAX_TOTAL_UPLOAD_BYTES,
) -> UploadedCorpusResult:
    """Validate uploaded documents and persist them into an isolated per-run corpus directory."""
    payloads = coerce_uploaded_files(files)
    if not payloads:
        raise UploadValidationError("Upload at least one PDF, DOCX, TXT, MD, or CSV document.")
    if len(payloads) > max_files:
        raise UploadValidationError(
            f"Too many uploaded files ({len(payloads)}). Maximum allowed per run is {max_files}."
        )

    used_lower_names: set[str] = set()
    staged_entries: list[tuple[str, str, bytes]] = []
    renamed_files: dict[str, str] = {}
    total_bytes = 0

    for idx, item in enumerate(payloads, start=1):
        raw_name = item.filename
        ext = Path(raw_name).suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported file type for '{raw_name}'. Allowed types: "
                f"{', '.join(ALLOWED_UPLOAD_EXTENSIONS)}."
            )
        size = len(item.content)
        if size == 0:
            raise UploadValidationError(f"Uploaded file '{raw_name}' is empty (0 bytes).")
        if size > max_file_bytes:
            limit_mb = max_file_bytes / (1024 * 1024)
            raise UploadValidationError(
                f"Uploaded file '{raw_name}' exceeds per-file limit ({limit_mb:.1f} MB)."
            )
        total_bytes += size
        if total_bytes > max_total_bytes:
            total_mb = max_total_bytes / (1024 * 1024)
            raise UploadValidationError(
                f"Total upload size exceeds limit ({total_mb:.1f} MB)."
            )

        saved_name = sanitize_and_deduplicate_filename(raw_name, used_lower_names)
        if saved_name != Path(raw_name.replace("\\", "/")).name:
            renamed_files[f"{idx}:{raw_name}"] = saved_name
        staged_entries.append((raw_name, saved_name, item.content))

    effective_run_id = (
        run_id.strip()
        if run_id and run_id.strip()
        else f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    run_dir = Path(artifacts_dir) / effective_run_id
    created_run_dir = not run_dir.exists()
    upload_dir = run_dir / "uploaded_documents"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    try:
        for _orig, saved_name, data in staged_entries:
            dest = upload_dir / saved_name
            dest.write_bytes(data)
            hasher.update(saved_name.encode("utf-8"))
            hasher.update(hashlib.sha256(data).digest())

        documents, units = load_document_folder(upload_dir, csv_rows_per_unit=csv_rows_per_unit)
        empty_parsed = [doc.file_name for doc in documents if doc.unit_count == 0]
        if empty_parsed:
            raise UploadValidationError(
                "Uploaded file(s) contain no extractable text or rows: "
                + ", ".join(empty_parsed)
            )
        if not documents or not units:
            raise UploadValidationError("No readable content extracted from uploaded documents.")
    except UploadValidationError:
        shutil.rmtree(upload_dir, ignore_errors=True)
        if created_run_dir and run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        if created_run_dir and run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise UploadValidationError(f"Failed to read uploaded documents: {exc}") from exc

    corpus_sha256 = hasher.hexdigest()
    manifest = {
        "run_id": effective_run_id,
        "corpus_sha256": corpus_sha256,
        "file_count": len(documents),
        "total_bytes": total_bytes,
        "saved_files": [doc.file_name for doc in documents],
        "renamed_files": renamed_files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return UploadedCorpusResult(
        run_id=effective_run_id,
        documents_dir=upload_dir,
        corpus_sha256=corpus_sha256,
        documents=documents,
        units=units,
        saved_files=[doc.file_name for doc in documents],
        renamed_files=renamed_files,
        total_bytes=total_bytes,
    )

