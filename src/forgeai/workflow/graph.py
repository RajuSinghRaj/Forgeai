"""LangGraph workflow orchestration for the ForgeAI agent factory.

Phase 5 wires `optimize_agent` and `build_report` using `CriticOptimizer`,
immutable evaluation-suite hashing, regression-aware acceptance checks,
per-iteration artifact persistence, and honest final reporting.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from forgeai.config import ForgeSettings, load_settings
from forgeai.evaluation import (
    EvalGenerator,
    InsufficientEvalCasesError,
    run_evaluation_suite,
    save_evaluation_artifacts,
)
from forgeai.generator import (
    DeclarativeAgentGenerator,
    GroundabilityApprovalError,
    save_agent_definition,
)
from forgeai.intake import GroundabilityGuard, IntakeInterviewer
from forgeai.llm.gemini_client import GeminiClient
from forgeai.optimization import (
    CriticOptimizer,
    OptimizationArtifactStore,
    build_final_report,
    snapshot_eval_suite,
    verify_eval_suite_integrity,
)
from forgeai.retrieval.embeddings import EmbeddingProvider
from forgeai.retrieval.loaders import load_document_folder
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.schemas.evaluation import (
    OptimizationChange,
    OptimizationIterationLog,
    RejectedChangeRecord,
)
from forgeai.tools.registry import ToolRegistry, build_default_registry
from forgeai.workflow.state import FactoryState

NodeHandler = Callable[[FactoryState], FactoryState]


def should_continue_after_intake(state: FactoryState) -> Literal["await_answers", "groundability_guard"]:
    """Pause after question generation until clarifying answers are submitted."""
    if state.get("awaiting_user_answers"):
        return "await_answers"
    return "groundability_guard"


def should_continue_optimization(
    state: FactoryState,
) -> Literal["optimize", "report", "phase_complete"]:
    """Decide whether to run Critic–Optimizer, emit report, or stop after Phase 4."""
    if state.get("stop_after_phase4"):
        return "phase_complete"

    scorecards = state.get("scorecards") or []
    if not scorecards:
        return "report"

    latest = state.get("best_scorecard") or scorecards[-1]
    target = state.get("target_pass_rate", 0.9)
    iteration = state.get("iteration", 0)
    max_iterations = min(max(state.get("max_iterations", 3), 0), 3)

    if latest.pass_rate >= target:
        return "report"
    if iteration >= max_iterations:
        return "report"
    return "optimize"


def should_reevaluate_after_optimization(
    state: FactoryState,
) -> Literal["run_and_judge", "report"]:
    """Route to re-evaluation unless Critic–Optimizer halted with no actionable proposal."""
    if state.get("optimization_halted"):
        return "report"
    return "run_and_judge"


def should_proceed_after_groundability(
    state: FactoryState,
) -> Literal["generate_agent", "report", "phase_complete"]:
    """Route to agent generation only when groundability allows proceeding."""
    if state.get("stop_after_phase2"):
        return "phase_complete"
    report = state.get("groundability")
    if report is None or not report.can_proceed:
        if state.get("stop_after_phase3") or state.get("stop_after_phase4"):
            return "phase_complete"
        return "report"
    return "generate_agent"


def should_continue_after_agent_generation(
    state: FactoryState,
) -> Literal["generate_evals", "phase_complete"]:
    """Stop cleanly after agent generation when running up to Phase 3."""
    if state.get("stop_after_phase3"):
        return "phase_complete"
    return "generate_evals"


def should_continue_after_eval_generation(
    state: FactoryState,
) -> Literal["run_and_judge", "phase_complete"]:
    """Allow pausing right after evaluation-set generation when requested."""
    if state.get("stop_after_eval_generation"):
        return "phase_complete"
    return "run_and_judge"


def _write_json_x(path: Path, payload: str) -> None:
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)


def build_factory_graph(
    node_handlers: dict[str, NodeHandler] | None = None,
    *,
    settings: ForgeSettings | None = None,
    gemini_client: GeminiClient | None = None,
    vector_store: ChromaVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    tool_registry: ToolRegistry | None = None,
):
    """Build and compile the ForgeAI LangGraph workflow."""
    cfg = settings or load_settings()
    client = gemini_client or GeminiClient(cfg)
    interviewer = IntakeInterviewer(
        gemini_client=client,
        max_questions=cfg.max_clarifying_questions,
    )
    guard = GroundabilityGuard(
        gemini_client=client,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        embedding_model=cfg.embedding_model,
    )
    generator = DeclarativeAgentGenerator(
        gemini_client=client,
        tool_registry=tool_registry,
        embedding_model=cfg.embedding_model,
        default_csv_rows_per_unit=cfg.csv_rows_per_unit,
    )

    def _ingest_documents_node(state: FactoryState) -> FactoryState:
        if state.get("documents") is not None and state.get("document_units") is not None:
            return {}
        brief = state["brief_input"]
        rows_per_unit = state.get("csv_rows_per_unit", cfg.csv_rows_per_unit)
        docs, units = load_document_folder(brief.documents_dir, csv_rows_per_unit=rows_per_unit)
        return {
            "documents": docs,
            "document_units": units,
            "iteration": state.get("iteration", 0),
            "max_iterations": min(max(state.get("max_iterations", cfg.max_optimization_iterations), 0), 3),
            "target_pass_rate": state.get("target_pass_rate", cfg.target_pass_rate),
            "min_eval_cases": state.get("min_eval_cases", cfg.min_eval_cases),
        }

    def _intake_interview_node(state: FactoryState) -> FactoryState:
        if state.get("requirements") is not None:
            return {"awaiting_user_answers": False}

        brief = state["brief_input"]
        docs = state.get("documents") or []
        units = state.get("document_units") or []

        if state.get("interview_completed") or state.get("clarifying_answers") is not None:
            try:
                requirements = interviewer.synthesize_requirements(
                    brief,
                    docs,
                    units,
                    clarifying_questions=state.get("clarifying_questions"),
                    clarifying_answers=state.get("clarifying_answers"),
                )
            except Exception as exc:
                raise RuntimeError(f"Requirements synthesis failed: {exc}") from exc
            return {
                "requirements": requirements,
                "awaiting_user_answers": False,
                "interview_completed": True,
            }

        try:
            questions = interviewer.generate_questions(brief, docs, units)
        except Exception as exc:
            raise RuntimeError(f"Intake question generation failed: {exc}") from exc

        policy = state.get("unattended_policy", "stop_for_clarification")
        if policy == "proceed_with_assumptions":
            assumptions = [
                f"Unattended intake assumption for {q.question_id} ({q.target_field}): {q.question}"
                for q in questions
            ]
            try:
                requirements = interviewer.synthesize_requirements(
                    brief,
                    docs,
                    units,
                    clarifying_questions=questions,
                    clarifying_answers=None,
                )
            except Exception as exc:
                raise RuntimeError(f"Unattended requirements synthesis failed: {exc}") from exc
            merged_assumptions = list(dict.fromkeys(list(requirements.assumptions) + assumptions))
            requirements = requirements.model_copy(update={"assumptions": merged_assumptions})
            return {
                "clarifying_questions": questions,
                "unattended_assumptions": assumptions,
                "requirements": requirements,
                "awaiting_user_answers": False,
                "interview_completed": True,
            }

        return {
            "clarifying_questions": questions,
            "awaiting_user_answers": True,
        }

    def _groundability_guard_node(state: FactoryState) -> FactoryState:
        if state.get("groundability") is not None and state.get("agent_definition") is not None:
            return {"phase2_approved": state["groundability"].can_proceed}
        requirements = state.get("requirements")
        if requirements is None:
            raise ValueError("Groundability Guard requires `requirements` in state.")
        docs = state.get("documents") or []
        units = state.get("document_units") or []
        approved_with_gaps = bool(state.get("approved_with_gaps", False))
        try:
            report = guard.assess(
                requirements,
                docs,
                units,
                approved_with_gaps=approved_with_gaps,
            )
        except Exception as exc:
            raise RuntimeError(f"Groundability check failed: {exc}") from exc
        return {
            "groundability": report,
            "phase2_approved": report.can_proceed,
            "phase3_status": "ready" if report.can_proceed else "blocked_by_groundability",
        }

    def _generate_agent_node(state: FactoryState) -> FactoryState:
        if state.get("agent_definition") is not None:
            return {"phase3_status": state.get("phase3_status", "completed")}
        requirements = state.get("requirements")
        if requirements is None:
            raise ValueError("Declarative Agent Generator requires `requirements` in state.")
        groundability = state.get("groundability")
        if groundability is None or not groundability.can_proceed:
            raise GroundabilityApprovalError(
                "Cannot generate agent: groundability approval is missing or blocked."
            )

        docs = state.get("documents") or []
        units = state.get("document_units") or []
        brief = state.get("brief_input")
        active_registry = tool_registry or build_default_registry(
            vector_store=vector_store,
            documents_dir=brief.documents_dir if brief else None,
        )
        try:
            definition = generator.generate(
                requirements,
                groundability,
                docs,
                units,
                tool_registry=active_registry,
                run_id=state.get("run_id"),
            )
            artifact_path = save_agent_definition(definition, cfg.artifacts_dir)
        except GroundabilityApprovalError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Declarative agent generation failed: {exc}") from exc

        return {
            "agent_definition": definition,
            "agent_artifact_path": str(artifact_path),
            "phase3_status": "completed",
        }

    def _generate_evals_node(state: FactoryState) -> FactoryState:
        min_cases = state.get("min_eval_cases", cfg.min_eval_cases)
        existing_cases = state.get("eval_cases")
        if existing_cases is not None and len(existing_cases) >= min_cases:
            updates: FactoryState = {}
            if not state.get("eval_suite_sha256"):
                frozen, digest = snapshot_eval_suite(existing_cases)
                updates["eval_cases"] = frozen
                updates["eval_suite_sha256"] = digest
            return updates

        requirements = state.get("requirements")
        agent_def = state.get("agent_definition")
        if requirements is None or agent_def is None:
            raise ValueError("Evaluation Generator requires `requirements` and `agent_definition` in state.")

        docs = state.get("documents") or []
        units = state.get("document_units") or []
        eval_gen = EvalGenerator(gemini_client=client, min_cases=min_cases)
        try:
            cases, meta = eval_gen.generate(
                requirements,
                agent_def,
                docs,
                units,
                groundability=state.get("groundability"),
            )
        except InsufficientEvalCasesError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Evaluation generation failed: {exc}") from exc

        frozen, digest = snapshot_eval_suite(cases)
        return {
            "eval_cases": frozen,
            "eval_suite_sha256": digest,
            "eval_generation_metadata": meta,
        }

    def _ensure_opt_store(state: FactoryState, agent_id: str) -> tuple[Path, dict[str, str]]:
        existing_dir = state.get("optimization_run_dir")
        if existing_dir:
            return Path(existing_dir), dict(state.get("optimization_artifact_paths") or {})
        run_label = state.get("run_id") or f"{agent_id}_{uuid4().hex[:6]}"
        store = OptimizationArtifactStore(cfg.artifacts_dir, run_label)
        return store.run_dir, dict(store.paths)

    def _run_and_judge_node(state: FactoryState) -> FactoryState:
        agent_def = state.get("agent_definition")
        eval_cases = state.get("eval_cases") or []
        min_cases = state.get("min_eval_cases", cfg.min_eval_cases)
        if agent_def is None:
            raise ValueError("Evaluation execution requires `agent_definition` in state.")
        if len(eval_cases) < min_cases:
            raise ValueError(
                f"Cannot run evaluation: {len(eval_cases)} cases provided, minimum {min_cases} required."
            )

        iteration = state.get("iteration", 0)
        existing_hash = state.get("eval_suite_sha256")
        if existing_hash:
            verify_eval_suite_integrity(eval_cases, existing_hash)
            frozen_cases = eval_cases
            eval_hash = existing_hash
        else:
            frozen_cases, eval_hash = snapshot_eval_suite(eval_cases)

        # Avoid re-running baseline when resuming from Phase 4 into Phase 5 optimization
        if (
            iteration == 0
            and state.get("baseline_scorecard") is not None
            and state.get("eval_results")
            and state.get("pending_proposal") is None
            and not state.get("stop_after_phase4")
        ):
            run_dir, opt_paths = _ensure_opt_store(state, agent_def.agent_id)
            _write_json_x(run_dir / "baseline_agent.json", agent_def.model_dump_json(indent=2))
            _write_json_x(
                run_dir / "baseline_scorecard.json",
                state["baseline_scorecard"].model_dump_json(indent=2),
            )
            _write_json_x(
                run_dir / "original_eval_suite.json",
                json.dumps([c.model_dump(mode="json") for c in frozen_cases], indent=2),
            )
            opt_paths.update(
                {
                    "baseline_agent": str(run_dir / "baseline_agent.json"),
                    "baseline_scorecard": str(run_dir / "baseline_scorecard.json"),
                    "original_eval_suite": str(run_dir / "original_eval_suite.json"),
                }
            )
            return {
                "eval_suite_sha256": eval_hash,
                "optimization_run_dir": str(run_dir),
                "optimization_artifact_paths": opt_paths,
            }

        units = state.get("document_units") or []
        best_agent = state.get("best_agent") or agent_def
        active_store = (
            vector_store
            if (vector_store is not None and agent_def.chunking == best_agent.chunking)
            else None
        )

        try:
            results, scorecard = run_evaluation_suite(
                agent_def,
                frozen_cases,
                units,
                gemini_client=client,
                vector_store=active_store,
                embedding_provider=embedding_provider,
                iteration=iteration,
            )
            artifact_paths = save_evaluation_artifacts(
                run_label=state.get("run_id") or agent_def.agent_id,
                eval_cases=frozen_cases,
                eval_results=results,
                scorecard=scorecard,
                generation_metadata=state.get("eval_generation_metadata"),
                artifacts_dir=cfg.artifacts_dir,
            )
        except Exception as exc:
            raise RuntimeError(f"Evaluation execution and judging failed: {exc}") from exc

        scorecards = list(state.get("scorecards") or [])
        scorecards.append(scorecard)
        updates: FactoryState = {
            "eval_cases": frozen_cases,
            "eval_suite_sha256": eval_hash,
            "eval_results": results,
            "scorecards": scorecards,
            "eval_artifact_paths": {
                "cases_path": str(artifact_paths.cases_path),
                "results_path": str(artifact_paths.results_path),
                "scorecard_path": str(artifact_paths.scorecard_path),
                "metadata_path": str(artifact_paths.metadata_path),
            },
            "phase4_status": "completed",
        }

        if iteration == 0:
            updates["baseline_agent"] = agent_def
            updates["baseline_scorecard"] = scorecard
            updates["baseline_results"] = results
            updates["best_agent"] = agent_def
            updates["best_scorecard"] = scorecard
            updates["best_results"] = results
            if not state.get("stop_after_phase4"):
                run_dir, opt_paths = _ensure_opt_store(state, agent_def.agent_id)
                _write_json_x(run_dir / "baseline_agent.json", agent_def.model_dump_json(indent=2))
                _write_json_x(run_dir / "baseline_scorecard.json", scorecard.model_dump_json(indent=2))
                _write_json_x(
                    run_dir / "original_eval_suite.json",
                    json.dumps([c.model_dump(mode="json") for c in frozen_cases], indent=2),
                )
                opt_paths.update(
                    {
                        "baseline_agent": str(run_dir / "baseline_agent.json"),
                        "baseline_scorecard": str(run_dir / "baseline_scorecard.json"),
                        "original_eval_suite": str(run_dir / "original_eval_suite.json"),
                    }
                )
                updates["optimization_run_dir"] = str(run_dir)
                updates["optimization_artifact_paths"] = opt_paths
            return updates

        # Iteration >= 1: compare candidate against retained best agent
        prev_best_agent = state.get("best_agent") or state.get("baseline_agent") or agent_def
        prev_best_scorecard = state.get("best_scorecard") or scorecards[0]
        prev_best_results = state.get("best_results") or results
        proposal = state.get("pending_proposal")
        applied_changes = list(state.get("pending_applied_changes") or [])
        rejected_changes = list(state.get("pending_rejected_changes") or [])
        observed_cats = list(state.get("pending_observed_categories") or [])
        failed_ids = list(state.get("pending_failed_ids") or [])

        accepted, acceptance_reason, deltas, _regressions = CriticOptimizer.evaluate_acceptance(
            prev_best_scorecard,
            scorecard,
        )
        if accepted:
            new_best_agent = agent_def
            new_best_scorecard = scorecard
            new_best_results = results
        else:
            new_best_agent = prev_best_agent
            new_best_scorecard = prev_best_scorecard
            new_best_results = prev_best_results
            if observed_cats:
                rejected_changes.append(
                    RejectedChangeRecord(
                        target_component="candidate_agent",
                        failure_driver=observed_cats[0],
                        summary=f"Candidate v{agent_def.version}",
                        reason=acceptance_reason,
                    )
                )

        run_dir, opt_paths = _ensure_opt_store(state, prev_best_agent.agent_id)
        prefix = f"iteration_{iteration}"
        if proposal is not None:
            _write_json_x(run_dir / f"{prefix}_proposal.json", proposal.model_dump_json(indent=2))
            opt_paths[f"{prefix}_proposal"] = str(run_dir / f"{prefix}_proposal.json")
        _write_json_x(run_dir / f"{prefix}_agent.json", agent_def.model_dump_json(indent=2))
        _write_json_x(
            run_dir / f"{prefix}_results.json",
            json.dumps([r.model_dump(mode="json") for r in results], indent=2),
        )
        _write_json_x(run_dir / f"{prefix}_scorecard.json", scorecard.model_dump_json(indent=2))
        opt_paths[f"{prefix}_agent"] = str(run_dir / f"{prefix}_agent.json")
        opt_paths[f"{prefix}_results"] = str(run_dir / f"{prefix}_results.json")
        opt_paths[f"{prefix}_scorecard"] = str(run_dir / f"{prefix}_scorecard.json")

        target_rate = state.get("target_pass_rate", cfg.target_pass_rate)
        max_iters = min(max(state.get("max_iterations", cfg.max_optimization_iterations), 0), 3)
        iter_stop: str | None = None
        if new_best_scorecard.pass_rate >= target_rate:
            iter_stop = (
                "all_cases_passed"
                if new_best_scorecard.pass_rate == 1.0
                else "target_pass_rate_achieved"
            )
        elif iteration >= max_iters:
            iter_stop = "max_iterations_exhausted"

        logs = list(state.get("optimization_logs") or [])
        logs.append(
            OptimizationIterationLog(
                iteration=iteration,
                diagnosis_summary=(
                    proposal.diagnosis_summary if proposal is not None else "Targeted candidate evaluation."
                ),
                observed_failure_categories=observed_cats,
                failed_case_ids=failed_ids,
                proposed_changes=(
                    [
                        OptimizationChange(
                            target_component=p.target_component,
                            failure_driver=p.failure_driver,
                            rationale=p.rationale,
                            summary=p.summary,
                        )
                        for p in proposal.proposed_changes
                    ]
                    if proposal is not None
                    else applied_changes
                ),
                targeted_changes=applied_changes,
                rejected_changes=rejected_changes,
                before_scorecard=prev_best_scorecard,
                after_scorecard=scorecard,
                metric_deltas=deltas,
                accepted=accepted,
                acceptance_reason=acceptance_reason,
                remaining_failed_case_ids=[r.case_id for r in new_best_results if not r.passed],
                agent_version_before=prev_best_agent.version,
                agent_version_after=new_best_agent.version,
                stop_reason=iter_stop,
            )
        )

        updates.update(
            {
                "agent_definition": new_best_agent,
                "best_agent": new_best_agent,
                "best_scorecard": new_best_scorecard,
                "best_results": new_best_results,
                "eval_results": new_best_results,
                "optimization_logs": logs,
                "optimization_run_dir": str(run_dir),
                "optimization_artifact_paths": opt_paths,
                "stop_reason": iter_stop or state.get("stop_reason", ""),
            }
        )
        return updates

    def _optimize_agent_node(state: FactoryState) -> FactoryState:
        max_iters = min(max(state.get("max_iterations", cfg.max_optimization_iterations), 0), 3)
        next_iter = state.get("iteration", 0) + 1
        if next_iter > max_iters:
            return {
                "optimization_halted": True,
                "stop_reason": "max_iterations_exhausted",
            }

        eval_cases = state.get("eval_cases") or []
        if state.get("eval_suite_sha256"):
            verify_eval_suite_integrity(eval_cases, state["eval_suite_sha256"])

        best_agent = state.get("best_agent") or state["agent_definition"]
        best_scorecard = state.get("best_scorecard") or state["scorecards"][-1]
        best_results = state.get("best_results") or state["eval_results"]
        requirements = state["requirements"]
        groundability = state["groundability"]
        docs = state.get("documents") or []
        units = state.get("document_units") or []

        optimizer = CriticOptimizer(
            gemini_client=client,
            max_iterations=max_iters,
            target_pass_rate=state.get("target_pass_rate", cfg.target_pass_rate),
            gemini_model=(cfg.gemini_model or "gemini-3.8-flash"),
            embedding_model=cfg.embedding_model,
        )
        grouped = optimizer.group_failures(best_results)
        if not grouped:
            return {
                "optimization_halted": True,
                "stop_reason": "all_cases_passed",
            }

        observed_cats = list(grouped.keys())
        failed_ids = [r.case_id for items in grouped.values() for r in items]

        try:
            proposal = optimizer.propose_changes(
                iteration=next_iter,
                requirements=requirements,
                agent=best_agent,
                scorecard=best_scorecard,
                eval_results=best_results,
                documents=docs,
                units=units,
            )
        except Exception as exc:
            raise RuntimeError(f"Critic–Optimizer proposal failed: {exc}") from exc

        candidate_agent, applied, rejected = optimizer.apply_targeted_proposal(
            best_agent,
            proposal,
            observed_failures=set(observed_cats),
            requirements=requirements,
            groundability=groundability,
        )
        if candidate_agent is None or not applied:
            logs = list(state.get("optimization_logs") or [])
            logs.append(
                OptimizationIterationLog(
                    iteration=next_iter,
                    diagnosis_summary=proposal.diagnosis_summary,
                    observed_failure_categories=observed_cats,
                    failed_case_ids=failed_ids,
                    proposed_changes=[
                        OptimizationChange(
                            target_component=p.target_component,
                            failure_driver=p.failure_driver,
                            rationale=p.rationale,
                            summary=p.summary,
                        )
                        for p in proposal.proposed_changes
                    ],
                    targeted_changes=[],
                    rejected_changes=rejected,
                    before_scorecard=best_scorecard,
                    after_scorecard=None,
                    metric_deltas={},
                    accepted=False,
                    acceptance_reason=proposal.no_action_reason
                    or "No valid failure-relevant changes could be applied.",
                    remaining_failed_case_ids=failed_ids,
                    agent_version_before=best_agent.version,
                    agent_version_after=best_agent.version,
                    stop_reason="no_actionable_proposal",
                )
            )
            return {
                "iteration": next_iter,
                "agent_definition": best_agent,
                "optimization_logs": logs,
                "optimization_halted": True,
                "stop_reason": "no_actionable_proposal",
            }

        return {
            "iteration": next_iter,
            "agent_definition": candidate_agent,
            "pending_proposal": proposal,
            "pending_applied_changes": applied,
            "pending_rejected_changes": rejected,
            "pending_observed_categories": observed_cats,
            "pending_failed_ids": failed_ids,
            "optimization_halted": False,
        }

    def _build_report_node(state: FactoryState) -> FactoryState:
        eval_cases = state.get("eval_cases") or []
        if state.get("eval_suite_sha256") and eval_cases:
            verify_eval_suite_integrity(eval_cases, state["eval_suite_sha256"])

        best_agent = state.get("best_agent") or state.get("agent_definition")
        if best_agent is None:
            return {
                "stop_reason": state.get("stop_reason") or "blocked_by_groundability",
                "phase5_status": "blocked_by_groundability",
            }

        scorecards = list(state.get("scorecards") or [])
        baseline_sc = state.get("baseline_scorecard") or scorecards[0]
        final_sc = state.get("best_scorecard") or scorecards[-1]
        final_results = state.get("best_results") or state.get("eval_results") or []
        logs = list(state.get("optimization_logs") or [])
        target = state.get("target_pass_rate", cfg.target_pass_rate)
        max_iters = min(max(state.get("max_iterations", cfg.max_optimization_iterations), 0), 3)

        stop_reason = state.get("stop_reason") or ""
        if not stop_reason:
            if final_sc.pass_rate >= target:
                if not logs:
                    stop_reason = (
                        "all_cases_passed" if final_sc.pass_rate == 1.0 else "baseline_passed_target"
                    )
                else:
                    stop_reason = (
                        "all_cases_passed"
                        if final_sc.pass_rate == 1.0
                        else "target_pass_rate_achieved"
                    )
            elif max_iters == 0:
                stop_reason = "optimization_disabled_budget_zero"
            else:
                stop_reason = "max_iterations_exhausted"

        run_dir, opt_paths = _ensure_opt_store(state, best_agent.agent_id)
        _write_json_x(run_dir / "baseline_agent.json", (state.get("baseline_agent") or best_agent).model_dump_json(indent=2))
        _write_json_x(run_dir / "baseline_scorecard.json", baseline_sc.model_dump_json(indent=2))
        _write_json_x(
            run_dir / "original_eval_suite.json",
            json.dumps([c.model_dump(mode="json") for c in eval_cases], indent=2),
        )
        _write_json_x(
            run_dir / "optimization_history.json",
            json.dumps([h.model_dump(mode="json") for h in logs], indent=2),
        )
        _write_json_x(run_dir / "final_agent.json", best_agent.model_dump_json(indent=2))

        opt_paths.update(
            {
                "baseline_agent": str(run_dir / "baseline_agent.json"),
                "baseline_scorecard": str(run_dir / "baseline_scorecard.json"),
                "original_eval_suite": str(run_dir / "original_eval_suite.json"),
                "optimization_history": str(run_dir / "optimization_history.json"),
                "final_agent": str(run_dir / "final_agent.json"),
                "final_report": str(run_dir / "final_report.json"),
            }
        )

        report = build_final_report(
            run_id=state.get("run_id") or run_dir.name,
            gemini_model=(cfg.gemini_model or "gemini-3.8-flash"),
            embedding_model=cfg.embedding_model,
            requirements=state["requirements"],
            groundability=state["groundability"],
            final_agent=best_agent,
            eval_cases=eval_cases,
            final_results=final_results,
            baseline_scorecard=baseline_sc,
            final_scorecard=final_sc,
            scorecards=scorecards,
            optimization_logs=logs,
            target_pass_rate=target,
            stop_reason=stop_reason,
            eval_suite_sha256=state.get("eval_suite_sha256", ""),
            artifact_paths=opt_paths,
        )
        _write_json_x(run_dir / "final_report.json", report.model_dump_json(indent=2))

        return {
            "agent_definition": best_agent,
            "eval_results": final_results,
            "final_report": report,
            "stop_reason": stop_reason,
            "optimization_run_dir": str(run_dir),
            "optimization_artifact_paths": opt_paths,
            "phase5_status": "completed",
        }

    handlers = node_handlers or {}
    graph = StateGraph(FactoryState)

    graph.add_node("ingest_documents", handlers.get("ingest_documents", _ingest_documents_node))
    graph.add_node("intake_interview", handlers.get("intake_interview", _intake_interview_node))
    graph.add_node(
        "groundability_guard",
        handlers.get("groundability_guard", _groundability_guard_node),
    )
    graph.add_node("generate_agent", handlers.get("generate_agent", _generate_agent_node))
    graph.add_node("generate_evals", handlers.get("generate_evals", _generate_evals_node))
    graph.add_node("run_and_judge", handlers.get("run_and_judge", _run_and_judge_node))
    graph.add_node("optimize_agent", handlers.get("optimize_agent", _optimize_agent_node))
    graph.add_node("build_report", handlers.get("build_report", _build_report_node))

    graph.add_edge(START, "ingest_documents")
    graph.add_edge("ingest_documents", "intake_interview")
    graph.add_conditional_edges(
        "intake_interview",
        should_continue_after_intake,
        {
            "await_answers": END,
            "groundability_guard": "groundability_guard",
        },
    )
    graph.add_conditional_edges(
        "groundability_guard",
        should_proceed_after_groundability,
        {
            "generate_agent": "generate_agent",
            "report": "build_report",
            "phase_complete": END,
        },
    )
    graph.add_conditional_edges(
        "generate_agent",
        should_continue_after_agent_generation,
        {
            "generate_evals": "generate_evals",
            "phase_complete": END,
        },
    )
    graph.add_conditional_edges(
        "generate_evals",
        should_continue_after_eval_generation,
        {
            "run_and_judge": "run_and_judge",
            "phase_complete": END,
        },
    )
    graph.add_conditional_edges(
        "run_and_judge",
        should_continue_optimization,
        {
            "optimize": "optimize_agent",
            "report": "build_report",
            "phase_complete": END,
        },
    )
    graph.add_conditional_edges(
        "optimize_agent",
        should_reevaluate_after_optimization,
        {
            "run_and_judge": "run_and_judge",
            "report": "build_report",
        },
    )
    graph.add_edge("build_report", END)

    return graph.compile()
