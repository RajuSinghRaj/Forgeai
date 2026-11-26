"""Collision-safe persistence for evaluation sets, results, scorecards, and metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from pydantic import BaseModel

from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    EvalGenerationMetadata,
    IterationScorecard,
)


class EvaluationArtifactPaths(BaseModel):
    """Paths written for a single evaluation run."""

    cases_path: Path
    results_path: Path
    scorecard_path: Path
    metadata_path: Path


def _slugify(value: str, fallback: str = "eval_run") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned[:48] or fallback


def _next_unique_stem(target_dir: Path, base_stem: str) -> str:
    suffixes = ("_cases.json", "_results.json", "_scorecard.json", "_metadata.json")
    candidate = base_stem
    counter = 1
    while any((target_dir / f"{candidate}{suf}").exists() for suf in suffixes):
        candidate = f"{base_stem}_{counter}"
        counter += 1
    return candidate


def _write_exclusive_json(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def save_evaluation_artifacts(
    *,
    run_label: str,
    eval_cases: list[EvalCase],
    eval_results: list[EvalCaseResult],
    scorecard: IterationScorecard,
    generation_metadata: EvalGenerationMetadata | None,
    artifacts_dir: Path | str,
) -> EvaluationArtifactPaths:
    """Persist evaluation cases, results, scorecard, and metadata without overwriting."""
    target_dir = Path(artifacts_dir) / "evaluations"
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = _next_unique_stem(target_dir, f"{_slugify(run_label)}_iter{scorecard.iteration}")
    cases_path = target_dir / f"{stem}_cases.json"
    results_path = target_dir / f"{stem}_results.json"
    scorecard_path = target_dir / f"{stem}_scorecard.json"
    metadata_path = target_dir / f"{stem}_metadata.json"

    _write_exclusive_json(
        cases_path,
        json.dumps([c.model_dump(mode="json") for c in eval_cases], indent=2),
    )
    _write_exclusive_json(
        results_path,
        json.dumps([r.model_dump(mode="json") for r in eval_results], indent=2),
    )
    _write_exclusive_json(scorecard_path, scorecard.model_dump_json(indent=2))
    meta_payload = (
        generation_metadata.model_dump_json(indent=2)
        if generation_metadata is not None
        else json.dumps({"note": "No generation metadata supplied."}, indent=2)
    )
    _write_exclusive_json(metadata_path, meta_payload)

    return EvaluationArtifactPaths(
        cases_path=cases_path,
        results_path=results_path,
        scorecard_path=scorecard_path,
        metadata_path=metadata_path,
    )

