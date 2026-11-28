"""Atomic checkpointing, manifest management, and per-case progress resume for ForgeAI."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    EvalGenerationMetadata,
    FactoryRunReport,
    FailureClassification,
    IterationScorecard,
    OptimizationChange,
    OptimizationIterationLog,
    OptimizationProposalDraft,
    RejectedChangeRecord,
)
from forgeai.schemas.intake import (
    BriefInput,
    ClarifyingAnswer,
    ClarifyingQuestion,
    DocumentInput,
    GroundabilityReport,
    StructuredRequirements,
)

_RESERVED_SUBDIRS = {"agents", "evaluations", "optimization", "telemetry", "demos", "runs"}


class CheckpointCorruptionError(ValueError):
    """Raised when a persisted checkpoint manifest, JSON file, or JSONL entry is invalid."""


class EvalProgressRecord(BaseModel):
    """Append-only single-case evaluation checkpoint line."""

    run_id: str = Field(..., min_length=1)
    iteration: int = Field(..., ge=0, le=3)
    agent_version: int = Field(..., ge=1)
    eval_suite_sha256: str = Field(..., min_length=16)
    case_id: str = Field(..., min_length=1)
    result: EvalCaseResult


class RunManifest(BaseModel):
    """Durable checkpoint manifest stored at `artifacts/runs/<run_id>/run_state.json`."""

    run_id: str = Field(..., min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = Field(default="initialized")
    brief_text: str = Field(default="")
    documents_dir: str = Field(default="")
    csv_rows_per_unit: int = Field(default=1, ge=1)
    min_eval_cases: int = Field(default=20, ge=1)
    max_iterations: int = Field(default=3, ge=0, le=3)
    target_pass_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    judge_batch_size: int = Field(default=5, ge=1, le=20)
    unattended_policy: str = Field(default="stop_for_clarification")
    unattended_assumptions: list[str] = Field(default_factory=list)
    clarifying_answers: dict[str, str] | list[ClarifyingAnswer] | None = None
    approved_with_gaps: bool = False
    interview_completed: bool = False
    awaiting_user_answers: bool = False
    documents: list[DocumentInput] = Field(default_factory=list)
    eval_suite_sha256: str = Field(default="")
    eval_generation_metadata: EvalGenerationMetadata | None = None
    active_agent_version: int | None = None
    baseline_agent_version: int | None = None
    best_agent_version: int | None = None
    iteration: int = Field(default=0, ge=0, le=3)
    optimization_halted: bool = False
    stop_reason: str = Field(default="")
    evaluated_signature: tuple[str, int, str] | None = None
    optimized_signature: tuple[str, int, str] | None = None
    pending_proposal: OptimizationProposalDraft | None = None
    pending_applied_changes: list[OptimizationChange] = Field(default_factory=list)
    pending_rejected_changes: list[RejectedChangeRecord] = Field(default_factory=list)
    pending_observed_categories: list[FailureClassification] = Field(default_factory=list)
    pending_failed_ids: list[str] = Field(default_factory=list)
    artifact_files: dict[str, str] = Field(default_factory=dict)


def _slugify_run_id(value: str, fallback: str = "forgeai_run") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:64] or fallback


def atomic_write_text(path: Path, payload: str) -> Path:
    """Write `payload` to a same-directory temp file, fsync, and atomically replace `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def append_progress_record(progress_file: Path, record: EvalProgressRecord) -> None:
    """Append one completed case verdict line to `progress_file` and flush immediately."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json() + "\n"
    with progress_file.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_progress_records(
    progress_file: Path,
    *,
    expected_run_id: str | None = None,
    expected_iteration: int,
    expected_agent_version: int,
    expected_eval_suite_sha256: str,
    valid_case_ids: set[str] | None = None,
) -> dict[str, EvalCaseResult]:
    """Load and validate completed case results from an append-only JSONL progress file.

    Raises `CheckpointCorruptionError` if any line is malformed JSON/schema or contains an
    eval suite SHA-256 mismatch. Safely ignores records belonging to a different agent
    version, iteration, or run_id and deduplicates repeated valid records.
    """
    if not progress_file.exists():
        return {}

    raw_text = progress_file.read_text(encoding="utf-8")
    completed: dict[str, EvalCaseResult] = {}
    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
            record = EvalProgressRecord.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise CheckpointCorruptionError(
                f"Corrupt evaluation checkpoint at {progress_file}:{line_no}: {exc}"
            ) from exc

        if record.case_id != record.result.case_id:
            raise CheckpointCorruptionError(
                f"Mismatched case_id inside checkpoint at {progress_file}:{line_no}: "
                f"{record.case_id!r} != {record.result.case_id!r}."
            )
        if record.agent_version != expected_agent_version or record.iteration != expected_iteration:
            continue
        if expected_run_id is not None and record.run_id != expected_run_id:
            continue
        if record.eval_suite_sha256 != expected_eval_suite_sha256:
            raise CheckpointCorruptionError(
                f"Evaluation suite SHA-256 mismatch in checkpoint at {progress_file}:{line_no}: "
                f"expected {expected_eval_suite_sha256[:12]}..., found {record.eval_suite_sha256[:12]}...."
            )
        if valid_case_ids is not None and record.case_id not in valid_case_ids:
            continue

        completed[record.case_id] = record.result

    return completed


def progress_file_for(run_dir: Path, iteration: int, agent_version: int) -> Path:
    """Return the canonical per-iteration, per-agent-version progress JSONL path."""
    return run_dir / f"eval_progress_iter{iteration}_v{agent_version}.jsonl"


class RunCheckpointStore:
    """Manages persistent atomic checkpoints inside `artifacts/runs/<run_id>/`."""

    def __init__(self, artifacts_dir: Path | str, run_id: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.run_id = _slugify_run_id(run_id)
        self.run_dir = self.artifacts_dir / self.run_id
        self.manifest_path = self.run_dir / "run_state.json"

    def ensure_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def load_manifest(self) -> RunManifest | None:
        if not self.manifest_path.exists():
            return None
        try:
            return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        except (ValidationError, OSError, ValueError) as exc:
            raise CheckpointCorruptionError(
                f"Corrupt run manifest at {self.manifest_path}: {exc}"
            ) from exc

    def save_manifest(self, manifest: RunManifest) -> Path:
        self.ensure_dir()
        updated = manifest.model_copy(
            update={"updated_at": datetime.now(timezone.utc).isoformat()}
        )
        return atomic_write_text(self.manifest_path, updated.model_dump_json(indent=2))

    def save_questions(self, questions: list[ClarifyingQuestion]) -> Path:
        self.ensure_dir()
        path = self.run_dir / "intake_questions.json"
        return atomic_write_text(
            path,
            json.dumps([q.model_dump(mode="json") for q in questions], indent=2),
        )

    def save_requirements(self, requirements: StructuredRequirements) -> Path:
        self.ensure_dir()
        return atomic_write_text(
            self.run_dir / "requirements.json",
            requirements.model_dump_json(indent=2),
        )

    def save_groundability(self, report: GroundabilityReport) -> Path:
        self.ensure_dir()
        return atomic_write_text(
            self.run_dir / "groundability.json",
            report.model_dump_json(indent=2),
        )

    def save_agent(self, agent: DeclarativeAgentDefinition, *, is_final: bool = False) -> Path:
        self.ensure_dir()
        path = self.run_dir / f"agent_v{agent.version}.json"
        atomic_write_text(path, agent.model_dump_json(indent=2))
        if is_final:
            atomic_write_text(self.run_dir / "final_agent.json", agent.model_dump_json(indent=2))
        return path

    def save_eval_cases(self, eval_cases: list[EvalCase], eval_suite_sha256: str) -> tuple[Path, Path]:
        self.ensure_dir()
        cases_path = self.run_dir / "eval_cases.json"
        hash_path = self.run_dir / "eval_suite_sha256"
        atomic_write_text(
            cases_path,
            json.dumps([c.model_dump(mode="json") for c in eval_cases], indent=2),
        )
        atomic_write_text(hash_path, eval_suite_sha256 + "\n")
        return cases_path, hash_path

    def save_evaluation_outputs(
        self,
        *,
        iteration: int,
        agent: DeclarativeAgentDefinition,
        results: list[EvalCaseResult],
        scorecard: IterationScorecard,
        all_scorecards: list[IterationScorecard],
        proposal: OptimizationProposalDraft | None = None,
    ) -> dict[str, str]:
        self.ensure_dir()
        written: dict[str, str] = {}
        if iteration == 0:
            sc_path = self.run_dir / "baseline_scorecard.json"
            res_path = self.run_dir / "baseline_results.json"
            ag_path = self.run_dir / "baseline_agent.json"
            atomic_write_text(ag_path, agent.model_dump_json(indent=2))
            written["baseline_agent"] = str(ag_path)
        else:
            prefix = f"iteration_{iteration}"
            if proposal is not None:
                prop_path = self.run_dir / f"{prefix}_proposal.json"
                atomic_write_text(prop_path, proposal.model_dump_json(indent=2))
                written[f"{prefix}_proposal"] = str(prop_path)
            ag_path = self.run_dir / f"{prefix}_agent.json"
            atomic_write_text(ag_path, agent.model_dump_json(indent=2))
            written[f"{prefix}_agent"] = str(ag_path)
            sc_path = self.run_dir / f"{prefix}_scorecard.json"
            res_path = self.run_dir / f"{prefix}_results.json"
            written[f"{prefix}_scorecard"] = str(sc_path)
            written[f"{prefix}_results"] = str(res_path)

        atomic_write_text(sc_path, scorecard.model_dump_json(indent=2))
        atomic_write_text(
            res_path,
            json.dumps([r.model_dump(mode="json") for r in results], indent=2),
        )
        sc_list_path = self.run_dir / "scorecards.json"
        atomic_write_text(
            sc_list_path,
            json.dumps([s.model_dump(mode="json") for s in all_scorecards], indent=2),
        )
        written["scorecards"] = str(sc_list_path)
        return written

    def save_optimization_history(self, logs: list[OptimizationIterationLog]) -> Path:
        self.ensure_dir()
        path = self.run_dir / "optimization_history.json"
        return atomic_write_text(
            path,
            json.dumps([h.model_dump(mode="json") for h in logs], indent=2),
        )

    def save_final_outputs(
        self,
        *,
        final_agent: DeclarativeAgentDefinition,
        final_report: FactoryRunReport,
        history: list[OptimizationIterationLog],
    ) -> dict[str, str]:
        self.ensure_dir()
        hist_path = self.save_optimization_history(history)
        agent_path = self.save_agent(final_agent, is_final=True)
        final_agent_path = self.run_dir / "final_agent.json"
        report_path = self.run_dir / "final_report.json"
        atomic_write_text(report_path, final_report.model_dump_json(indent=2))
        return {
            "optimization_history": str(hist_path),
            f"agent_v{final_agent.version}": str(agent_path),
            "final_agent": str(final_agent_path),
            "final_report": str(report_path),
        }

    def sync_from_state(self, state: dict[str, Any], *, status: str) -> RunManifest:
        """Persist current manifest fields cleanly from `FactoryState` without secrets."""
        existing = self.load_manifest()
        brief = state.get("brief_input")
        manifest = existing or RunManifest(run_id=self.run_id)

        updates: dict[str, Any] = {
            "status": status,
            "csv_rows_per_unit": state.get("csv_rows_per_unit", manifest.csv_rows_per_unit),
            "min_eval_cases": state.get("min_eval_cases", manifest.min_eval_cases),
            "max_iterations": state.get("max_iterations", manifest.max_iterations),
            "target_pass_rate": state.get("target_pass_rate", manifest.target_pass_rate),
            "unattended_policy": state.get("unattended_policy", manifest.unattended_policy),
            "unattended_assumptions": list(
                state.get("unattended_assumptions") or manifest.unattended_assumptions
            ),
            "clarifying_answers": (
                state["clarifying_answers"]
                if state.get("clarifying_answers") is not None
                else manifest.clarifying_answers
            ),
            "approved_with_gaps": bool(
                state.get("approved_with_gaps", manifest.approved_with_gaps)
            ),
            "interview_completed": bool(
                state.get("interview_completed", manifest.interview_completed)
            ),
            "awaiting_user_answers": bool(
                state.get("awaiting_user_answers", manifest.awaiting_user_answers)
            ),
            "iteration": state.get("iteration", manifest.iteration),
            "optimization_halted": bool(
                state.get("optimization_halted", manifest.optimization_halted)
            ),
            "stop_reason": state.get("stop_reason", manifest.stop_reason) or "",
        }
        if brief is not None:
            updates["brief_text"] = brief.brief_text
            updates["documents_dir"] = str(brief.documents_dir)
        if state.get("documents") is not None:
            updates["documents"] = list(state["documents"])
        if state.get("eval_suite_sha256"):
            updates["eval_suite_sha256"] = state["eval_suite_sha256"]
        if state.get("eval_generation_metadata") is not None:
            updates["eval_generation_metadata"] = state["eval_generation_metadata"]

        agent_def = state.get("agent_definition")
        if agent_def is not None:
            updates["active_agent_version"] = agent_def.version
        baseline_agent = state.get("baseline_agent")
        if baseline_agent is not None:
            updates["baseline_agent_version"] = baseline_agent.version
        best_agent = state.get("best_agent")
        if best_agent is not None:
            updates["best_agent_version"] = best_agent.version

        if state.get("evaluated_signature") is not None:
            updates["evaluated_signature"] = tuple(state["evaluated_signature"])
        if state.get("optimized_signature") is not None:
            updates["optimized_signature"] = tuple(state["optimized_signature"])

        updates["pending_proposal"] = state.get("pending_proposal")
        updates["pending_applied_changes"] = list(state.get("pending_applied_changes") or [])
        updates["pending_rejected_changes"] = list(state.get("pending_rejected_changes") or [])
        updates["pending_observed_categories"] = list(
            state.get("pending_observed_categories") or []
        )
        updates["pending_failed_ids"] = list(state.get("pending_failed_ids") or [])

        merged_files = dict(manifest.artifact_files)
        merged_files["run_state"] = str(self.manifest_path)
        updates["artifact_files"] = merged_files

        updated = manifest.model_copy(update=updates)
        self.save_manifest(updated)
        return updated


def _read_validated_json(path: Path, loader) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
        return loader(raw)
    except Exception as exc:
        raise CheckpointCorruptionError(f"Corrupt checkpoint file {path}: {exc}") from exc


def load_run_checkpoint(run_dir: Path | str) -> dict[str, Any]:
    """Load and validate a saved run checkpoint directory into a resumable `FactoryState`."""
    target_dir = Path(run_dir)
    manifest_path = target_dir / "run_state.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run_state.json manifest found in `{target_dir}`.")

    manifest = _read_validated_json(manifest_path, RunManifest.model_validate_json)
    state: dict[str, Any] = {
        "run_id": manifest.run_id,
        "checkpoint_dir": str(target_dir),
        "brief_input": BriefInput(
            brief_text=manifest.brief_text,
            documents_dir=Path(manifest.documents_dir),
            run_id=manifest.run_id,
        ),
        "documents": list(manifest.documents),
        "csv_rows_per_unit": manifest.csv_rows_per_unit,
        "min_eval_cases": manifest.min_eval_cases,
        "max_iterations": manifest.max_iterations,
        "target_pass_rate": manifest.target_pass_rate,
        "unattended_policy": manifest.unattended_policy,
        "unattended_assumptions": list(manifest.unattended_assumptions),
        "clarifying_answers": manifest.clarifying_answers,
        "approved_with_gaps": manifest.approved_with_gaps,
        "interview_completed": manifest.interview_completed,
        "awaiting_user_answers": manifest.awaiting_user_answers,
        "iteration": manifest.iteration,
        "optimization_halted": manifest.optimization_halted,
        "stop_reason": manifest.stop_reason,
    }
    if manifest.evaluated_signature is not None:
        state["evaluated_signature"] = tuple(manifest.evaluated_signature)
    if manifest.optimized_signature is not None:
        state["optimized_signature"] = tuple(manifest.optimized_signature)
    if manifest.eval_generation_metadata is not None:
        state["eval_generation_metadata"] = manifest.eval_generation_metadata
    if manifest.pending_proposal is not None:
        state["pending_proposal"] = manifest.pending_proposal
        state["pending_applied_changes"] = list(manifest.pending_applied_changes)
        state["pending_rejected_changes"] = list(manifest.pending_rejected_changes)
        state["pending_observed_categories"] = list(manifest.pending_observed_categories)
        state["pending_failed_ids"] = list(manifest.pending_failed_ids)

    q_path = target_dir / "intake_questions.json"
    if q_path.exists():
        state["clarifying_questions"] = _read_validated_json(
            q_path,
            lambda raw: [ClarifyingQuestion.model_validate(x) for x in json.loads(raw)],
        )

    req_path = target_dir / "requirements.json"
    if req_path.exists():
        state["requirements"] = _read_validated_json(
            req_path, StructuredRequirements.model_validate_json
        )

    gb_path = target_dir / "groundability.json"
    if gb_path.exists():
        state["groundability"] = _read_validated_json(
            gb_path, GroundabilityReport.model_validate_json
        )
        state["phase2_approved"] = state["groundability"].can_proceed

    def _load_agent_ver(ver: int | None) -> DeclarativeAgentDefinition | None:
        if ver is None:
            return None
        cand = target_dir / f"agent_v{ver}.json"
        if not cand.exists():
            raise CheckpointCorruptionError(f"Expected agent checkpoint file missing: {cand}")
        return _read_validated_json(cand, DeclarativeAgentDefinition.model_validate_json)

    active_agent = _load_agent_ver(manifest.active_agent_version)
    if active_agent is not None:
        state["agent_definition"] = active_agent
        state["agent_artifact_path"] = str(target_dir / f"agent_v{active_agent.version}.json")
        state["phase3_status"] = "completed"

    baseline_agent = _load_agent_ver(manifest.baseline_agent_version)
    if baseline_agent is not None:
        state["baseline_agent"] = baseline_agent

    best_agent = _load_agent_ver(manifest.best_agent_version)
    if best_agent is not None:
        state["best_agent"] = best_agent

    cases_path = target_dir / "eval_cases.json"
    hash_path = target_dir / "eval_suite_sha256"
    if cases_path.exists():
        from forgeai.optimization.integrity import compute_eval_suite_hash, verify_eval_suite_integrity

        eval_cases = _read_validated_json(
            cases_path,
            lambda raw: [EvalCase.model_validate(x) for x in json.loads(raw)],
        )
        expected_hash = manifest.eval_suite_sha256
        if hash_path.exists():
            file_hash = hash_path.read_text(encoding="utf-8").strip()
            if expected_hash and file_hash != expected_hash:
                raise CheckpointCorruptionError(
                    "Evaluation suite hash mismatch between run_state.json and eval_suite_sha256 file."
                )
            expected_hash = file_hash
        if not expected_hash:
            expected_hash = compute_eval_suite_hash(eval_cases)
        verify_eval_suite_integrity(eval_cases, expected_hash)
        state["eval_cases"] = eval_cases
        state["eval_suite_sha256"] = expected_hash

    sc_list_path = target_dir / "scorecards.json"
    if sc_list_path.exists():
        state["scorecards"] = _read_validated_json(
            sc_list_path,
            lambda raw: [IterationScorecard.model_validate(x) for x in json.loads(raw)],
        )

    base_sc_path = target_dir / "baseline_scorecard.json"
    base_res_path = target_dir / "baseline_results.json"
    if base_sc_path.exists() and base_res_path.exists():
        state["baseline_scorecard"] = _read_validated_json(
            base_sc_path, IterationScorecard.model_validate_json
        )
        state["baseline_results"] = _read_validated_json(
            base_res_path,
            lambda raw: [EvalCaseResult.model_validate(x) for x in json.loads(raw)],
        )
        if "scorecards" not in state:
            state["scorecards"] = [state["baseline_scorecard"]]

    hist_path = target_dir / "optimization_history.json"
    if hist_path.exists():
        state["optimization_logs"] = _read_validated_json(
            hist_path,
            lambda raw: [OptimizationIterationLog.model_validate(x) for x in json.loads(raw)],
        )

    # Restore best_results/best_scorecard and current eval_results from latest completed iteration
    scorecards_loaded = state.get("scorecards") or []
    best_iter = len(scorecards_loaded) - 1 if scorecards_loaded else 0
    if best_agent is not None and scorecards_loaded:
        best_sc = state.get("baseline_scorecard")
        best_res = state.get("baseline_results")
        logs_loaded = state.get("optimization_logs") or []
        accepted_iters = [log.iteration for log in logs_loaded if log.accepted]
        if accepted_iters:
            last_acc = accepted_iters[-1]
            acc_sc = target_dir / f"iteration_{last_acc}_scorecard.json"
            acc_res = target_dir / f"iteration_{last_acc}_results.json"
            if acc_sc.exists() and acc_res.exists():
                best_sc = _read_validated_json(acc_sc, IterationScorecard.model_validate_json)
                best_res = _read_validated_json(
                    acc_res, lambda raw: [EvalCaseResult.model_validate(x) for x in json.loads(raw)]
                )
        if best_sc is not None and best_res is not None:
            state["best_scorecard"] = best_sc
            state["best_results"] = best_res
            state["eval_results"] = best_res
    elif "baseline_results" in state:
        state["eval_results"] = state["baseline_results"]

    # Validate any partial in-progress evaluation file so corrupt checkpoints fail early on load
    if "eval_cases" in state and active_agent is not None:
        prog_path = progress_file_for(target_dir, manifest.iteration, active_agent.version)
        if prog_path.exists():
            completed_map = load_progress_records(
                prog_path,
                expected_run_id=manifest.run_id,
                expected_iteration=manifest.iteration,
                expected_agent_version=active_agent.version,
                expected_eval_suite_sha256=state["eval_suite_sha256"],
                valid_case_ids={c.case_id for c in state["eval_cases"]},
            )
            state["checkpoint_completed_cases"] = len(completed_map)
            state["checkpoint_total_cases"] = len(state["eval_cases"])

    report_path = target_dir / "final_report.json"
    if report_path.exists():
        state["final_report"] = _read_validated_json(
            report_path, FactoryRunReport.model_validate_json
        )
        state["phase5_status"] = "completed"

    return state


def inspect_run_progress_summary(run_dir: Path | str) -> dict[str, Any]:
    """Return a non-throwing status summary for a saved run directory in the UI."""
    target_dir = Path(run_dir)
    manifest_path = target_dir / "run_state.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    total_cases = 0
    completed_cases = 0
    eval_cases_path = target_dir / "eval_cases.json"
    if eval_cases_path.exists() and manifest.active_agent_version is not None:
        cases = [EvalCase.model_validate(x) for x in json.loads(eval_cases_path.read_text(encoding="utf-8"))]
        total_cases = len(cases)
        prog_path = progress_file_for(target_dir, manifest.iteration, manifest.active_agent_version)
        if prog_path.exists() and manifest.eval_suite_sha256:
            done = load_progress_records(
                prog_path,
                expected_run_id=manifest.run_id,
                expected_iteration=manifest.iteration,
                expected_agent_version=manifest.active_agent_version,
                expected_eval_suite_sha256=manifest.eval_suite_sha256,
                valid_case_ids={c.case_id for c in cases},
            )
            completed_cases = len(done)
        elif (
            (target_dir / "baseline_results.json").exists()
            and manifest.iteration == 0
            and not manifest.pending_proposal
        ) or (
            manifest.iteration > 0
            and (target_dir / f"iteration_{manifest.iteration}_results.json").exists()
        ):
            completed_cases = total_cases

    return {
        "run_id": manifest.run_id,
        "run_dir": str(target_dir),
        "status": manifest.status,
        "updated_at": manifest.updated_at,
        "iteration": manifest.iteration,
        "active_agent_version": manifest.active_agent_version,
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "remaining_cases": max(0, total_cases - completed_cases),
        "completed": (target_dir / "final_report.json").exists(),
    }


def list_saved_runs(artifacts_dir: Path | str = Path("artifacts/runs")) -> list[dict[str, Any]]:
    """List resumable/saved run directories containing `run_state.json`, newest first."""
    base = Path(artifacts_dir)
    if not base.exists() or not base.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for entry in base.iterdir():
        if not entry.is_dir() or entry.name in _RESERVED_SUBDIRS:
            continue
        if not (entry / "run_state.json").exists():
            continue
        try:
            summaries.append(inspect_run_progress_summary(entry))
        except Exception as exc:
            summaries.append(
                {
                    "run_id": entry.name,
                    "run_dir": str(entry),
                    "status": "corrupt_checkpoint",
                    "updated_at": "",
                    "error": str(exc),
                    "total_cases": 0,
                    "completed_cases": 0,
                    "remaining_cases": 0,
                    "completed": False,
                }
            )

    summaries.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
    return summaries
