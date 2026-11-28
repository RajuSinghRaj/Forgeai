"""Execution harness running a generated agent across evaluation cases and judging outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from forgeai.evaluation.early_stop import (
    can_candidate_reach_acceptance,
    prioritize_cases_for_candidate,
)
from forgeai.evaluation.judge import HybridJudge
from forgeai.evaluation.metrics import compute_scorecard
from forgeai.llm.gemini_client import GeminiClient
from forgeai.llm.telemetry import ApiTelemetryTracker, telemetry_context
from forgeai.retrieval.chunker import chunk_units
from forgeai.retrieval.embeddings import EmbeddingProvider, LocalSentenceTransformerEmbeddings
from forgeai.retrieval.loaders import DocumentUnit, rebatch_csv_units
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.runner.agent_runner import DeclarativeAgentRunner
from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import EvalCase, EvalCaseResult, IterationScorecard
from forgeai.tools.registry import ToolRegistry, build_default_registry


def _infer_documents_dir(units: list[DocumentUnit]) -> Path | None:
    """Safely infer a single common corpus directory from `DocumentUnit.file_path` when consistent."""
    roots: set[Path] = set()
    for unit in units:
        if unit.file_path is None:
            continue
        file_path = Path(unit.file_path)
        rel_parts = Path(unit.source_file).parts
        if rel_parts and len(file_path.parts) > len(rel_parts):
            candidate = Path(*file_path.parts[: -len(rel_parts)])
            if candidate.exists() and candidate.is_dir():
                roots.add(candidate.resolve())
                continue
        parent = file_path.parent
        if parent.exists() and parent.is_dir():
            roots.add(parent.resolve())
    return next(iter(roots)) if len(roots) == 1 else None


def run_evaluation_suite(
    definition: DeclarativeAgentDefinition,
    eval_cases: list[EvalCase],
    units: list[DocumentUnit],
    *,
    gemini_client: GeminiClient | None = None,
    vector_store: ChromaVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    judge: HybridJudge | None = None,
    iteration: int = 0,
    judge_batch_size: int = 5,
    checkpoint_dir: Path | str | None = None,
    progress_file: Path | str | None = None,
    run_id: str | None = None,
    eval_suite_sha256: str | None = None,
    early_stop_baseline: IterationScorecard | None = None,
    prior_results: list[EvalCaseResult] | None = None,
    early_stop_out: dict[str, Any] | None = None,
    telemetry: ApiTelemetryTracker | None = None,
    documents_dir: Path | str | None = None,
    tool_registry: ToolRegistry | None = None,
) -> tuple[list[EvalCaseResult], IterationScorecard]:
    """Execute `DeclarativeAgentRunner` for missing cases and judge in batches with checkpoints."""
    from forgeai.optimization.integrity import compute_eval_suite_hash, verify_eval_suite_integrity
    from forgeai.workflow.checkpoints import (
        EvalProgressRecord,
        append_progress_record,
        load_progress_records,
        progress_file_for,
    )

    client = gemini_client or GeminiClient()
    suite_hash = compute_eval_suite_hash(eval_cases)
    if eval_suite_sha256:
        verify_eval_suite_integrity(eval_cases, eval_suite_sha256)
        suite_hash = eval_suite_sha256

    effective_run_id = run_id or str(definition.metadata.get("run_id") or definition.agent_id)
    target_progress_file: Path | None = None
    if progress_file is not None:
        target_progress_file = Path(progress_file)
    elif checkpoint_dir is not None:
        target_progress_file = progress_file_for(Path(checkpoint_dir), iteration, definition.version)

    completed_by_id: dict[str, EvalCaseResult] = {}
    if target_progress_file is not None:
        completed_by_id = load_progress_records(
            target_progress_file,
            expected_run_id=effective_run_id if run_id is not None else None,
            expected_iteration=iteration,
            expected_agent_version=definition.version,
            expected_eval_suite_sha256=suite_hash,
            valid_case_ids={c.case_id for c in eval_cases},
        )

    missing_cases = [case for case in eval_cases if case.case_id not in completed_by_id]
    if early_stop_baseline is not None and missing_cases:
        missing_cases = prioritize_cases_for_candidate(missing_cases, prior_results)

    early_stopped = False
    early_stop_reason = ""
    early_stopped_cases = 0
    requests_saved = 0
    batch_step = max(1, judge_batch_size)
    active_telem = telemetry or getattr(client, "telemetry", None)

    # Recalculate upper bounds on already-checkpointed partial results before any new API calls.
    if early_stop_baseline is not None and completed_by_id and missing_cases:
        can_reach, reason = can_candidate_reach_acceptance(
            early_stop_baseline,
            list(completed_by_id.values()),
            eval_cases,
        )
        if not can_reach:
            early_stopped = True
            early_stop_reason = reason
            early_stopped_cases = len(missing_cases)
            skipped_batches = (len(missing_cases) + batch_step - 1) // batch_step
            requests_saved = early_stopped_cases + skipped_batches
            if active_telem is not None and hasattr(active_telem, "record_efficiency"):
                active_telem.record_efficiency(
                    stage="optimization_early_stopping",
                    requests_saved=requests_saved,
                    early_stopped_cases=early_stopped_cases,
                    early_stops=1,
                )
            missing_cases = []

    resolved_docs_dir = Path(documents_dir) if documents_dir is not None else _infer_documents_dir(units)
    effective_units = rebatch_csv_units(
        units,
        definition.chunking.csv_rows_per_unit,
        documents_dir=resolved_docs_dir,
    )

    if missing_cases:
        if vector_store is not None:
            store = vector_store
        else:
            embedder = embedding_provider or LocalSentenceTransformerEmbeddings(
                definition.retrieval.embedding_model
            )
            store = ChromaVectorStore(
                collection_name=f"eval_{uuid4().hex[:8]}",
                persist_dir=None,
                embedding_provider=embedder,
            )
            chunks = chunk_units(effective_units, definition.chunking)
            if chunks:
                store.index_chunks(chunks)

        active_registry = tool_registry or build_default_registry(
            vector_store=store,
            documents_dir=resolved_docs_dir,
        )
        runner = DeclarativeAgentRunner(
            definition,
            vector_store=store,
            gemini_client=client,
            tool_registry=active_registry,
            documents_dir=resolved_docs_dir,
        )
        known_sources = set(definition.source_documents) | {u.source_file for u in effective_units}
        active_judge = judge or HybridJudge(
            gemini_client=client,
            known_sources=known_sources,
            batch_size=judge_batch_size,
        )

        is_candidate_eval = early_stop_baseline is not None or iteration >= 1
        phase_ctx = "optimization_candidate_evaluation" if is_candidate_eval else ""

        with telemetry_context(phase_ctx):
            for start in range(0, len(missing_cases), batch_step):
                chunk_cases = missing_cases[start : start + batch_step]
                chunk_executed: list[tuple[EvalCase, Any]] = [
                    (case, runner.run(case.question)) for case in chunk_cases
                ]
                chunk_results = active_judge.evaluate_batch(
                    chunk_executed,
                    definition.output_contract,
                    known_sources=known_sources,
                    batch_size=len(chunk_executed),
                )
                for verdict in chunk_results:
                    completed_by_id[verdict.case_id] = verdict
                    if target_progress_file is not None:
                        append_progress_record(
                            target_progress_file,
                            EvalProgressRecord(
                                run_id=effective_run_id,
                                iteration=iteration,
                                agent_version=definition.version,
                                eval_suite_sha256=suite_hash,
                                case_id=verdict.case_id,
                                result=verdict,
                            ),
                        )

                if early_stop_baseline is not None and (start + batch_step) < len(missing_cases):
                    can_reach, reason = can_candidate_reach_acceptance(
                        early_stop_baseline,
                        list(completed_by_id.values()),
                        eval_cases,
                    )
                    if not can_reach:
                        remaining_after = missing_cases[start + batch_step :]
                        early_stopped = True
                        early_stop_reason = reason
                        early_stopped_cases = len(remaining_after)
                        skipped_batches = (len(remaining_after) + batch_step - 1) // batch_step
                        # Each skipped case avoids 1 runner request; each remaining batch avoids 1 judge request.
                        requests_saved = early_stopped_cases + skipped_batches
                        if active_telem is not None and hasattr(active_telem, "record_efficiency"):
                            active_telem.record_efficiency(
                                stage="optimization_early_stopping",
                                requests_saved=requests_saved,
                                early_stopped_cases=early_stopped_cases,
                                early_stops=1,
                            )
                        break

    if early_stop_out is not None:
        early_stop_out.clear()
        early_stop_out.update(
            {
                "early_stopped": early_stopped,
                "early_stop_reason": early_stop_reason,
                "evaluated_cases_count": len(completed_by_id),
                "early_stopped_cases": early_stopped_cases,
                "requests_saved": requests_saved,
            }
        )

    ordered_results = [
        completed_by_id[case.case_id] for case in eval_cases if case.case_id in completed_by_id
    ]
    is_partial = early_stopped and len(ordered_results) < len(eval_cases)
    scorecard = compute_scorecard(
        ordered_results,
        iteration=iteration,
        is_partial=is_partial,
        expected_total_cases=len(eval_cases) if is_partial else None,
    )
    return ordered_results, scorecard
