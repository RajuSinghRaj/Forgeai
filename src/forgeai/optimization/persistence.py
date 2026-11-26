"""Collision-safe artifact persistence for optimization runs and final reports."""

from __future__ import annotations

import json
import re
from pathlib import Path

from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    FactoryRunReport,
    IterationScorecard,
    OptimizationIterationLog,
    OptimizationProposalDraft,
)


def _slugify(value: str, fallback: str = "opt_run") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned[:48] or fallback


def _write_exclusive(path: Path, payload: str) -> Path:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path


class OptimizationArtifactStore:
    """Creates a unique run directory and persists baseline, per-iteration, and final artifacts."""

    def __init__(self, artifacts_dir: Path | str, run_id: str) -> None:
        base_dir = Path(artifacts_dir) / "optimization"
        base_dir.mkdir(parents=True, exist_ok=True)

        slug = _slugify(run_id)
        candidate = base_dir / slug
        counter = 1
        while candidate.exists():
            candidate = base_dir / f"{slug}_{counter}"
            counter += 1
        candidate.mkdir(parents=True, exist_ok=False)
        self.run_dir = candidate
        self.paths: dict[str, str] = {}

    def write_baseline(
        self,
        *,
        agent: DeclarativeAgentDefinition,
        scorecard: IterationScorecard,
        eval_cases: list[EvalCase],
    ) -> dict[str, str]:
        p_agent = _write_exclusive(
            self.run_dir / "baseline_agent.json",
            agent.model_dump_json(indent=2),
        )
        p_scorecard = _write_exclusive(
            self.run_dir / "baseline_scorecard.json",
            scorecard.model_dump_json(indent=2),
        )
        p_cases = _write_exclusive(
            self.run_dir / "original_eval_suite.json",
            json.dumps([c.model_dump(mode="json") for c in eval_cases], indent=2),
        )
        written = {
            "baseline_agent": str(p_agent),
            "baseline_scorecard": str(p_scorecard),
            "original_eval_suite": str(p_cases),
        }
        self.paths.update(written)
        return written

    def write_iteration(
        self,
        *,
        iteration: int,
        proposal: OptimizationProposalDraft,
        agent: DeclarativeAgentDefinition,
        results: list[EvalCaseResult],
        scorecard: IterationScorecard,
    ) -> dict[str, str]:
        prefix = f"iteration_{iteration}"
        p_prop = _write_exclusive(
            self.run_dir / f"{prefix}_proposal.json",
            proposal.model_dump_json(indent=2),
        )
        p_agent = _write_exclusive(
            self.run_dir / f"{prefix}_agent.json",
            agent.model_dump_json(indent=2),
        )
        p_res = _write_exclusive(
            self.run_dir / f"{prefix}_results.json",
            json.dumps([r.model_dump(mode="json") for r in results], indent=2),
        )
        p_sc = _write_exclusive(
            self.run_dir / f"{prefix}_scorecard.json",
            scorecard.model_dump_json(indent=2),
        )
        written = {
            f"{prefix}_proposal": str(p_prop),
            f"{prefix}_agent": str(p_agent),
            f"{prefix}_results": str(p_res),
            f"{prefix}_scorecard": str(p_sc),
        }
        self.paths.update(written)
        return written

    def write_final(
        self,
        *,
        history: list[OptimizationIterationLog],
        final_agent: DeclarativeAgentDefinition,
        final_report: FactoryRunReport,
    ) -> dict[str, str]:
        p_hist = _write_exclusive(
            self.run_dir / "optimization_history.json",
            json.dumps([h.model_dump(mode="json") for h in history], indent=2),
        )
        p_agent = _write_exclusive(
            self.run_dir / "final_agent.json",
            final_agent.model_dump_json(indent=2),
        )
        # Include the paths of final artifacts inside final_report.json as well
        all_paths = {
            **self.paths,
            "optimization_history": str(p_hist),
            "final_agent": str(p_agent),
            "final_report": str(self.run_dir / "final_report.json"),
        }
        report_with_paths = final_report.model_copy(update={"artifact_paths": all_paths})
        p_report = _write_exclusive(
            self.run_dir / "final_report.json",
            report_with_paths.model_dump_json(indent=2),
        )
        written = {
            "optimization_history": str(p_hist),
            "final_agent": str(p_agent),
            "final_report": str(p_report),
        }
        self.paths.update(written)
        return written

