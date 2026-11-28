"""Targeted Critic–Optimizer loop with evaluation integrity and regression guards."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from forgeai.evaluation.early_stop import (
    can_candidate_reach_acceptance,
    prioritize_cases_for_candidate,
)
from forgeai.evaluation.judge import HybridJudge
from forgeai.evaluation.suite_runner import run_evaluation_suite
from forgeai.generator.agent_generator import _ensure_prompt_guardrails
from forgeai.intake.interviewer import summarize_documents
from forgeai.llm.gemini_client import GeminiClient
from forgeai.llm.telemetry import telemetry_stage
from forgeai.optimization.integrity import (
    snapshot_eval_suite,
    verify_eval_suite_integrity,
)
from forgeai.optimization.persistence import OptimizationArtifactStore
from forgeai.optimization.report import build_final_report, compute_metric_deltas
from forgeai.prompts.templates import CRITIC_OPTIMIZER_PROMPT
from forgeai.retrieval.embeddings import EmbeddingProvider
from forgeai.retrieval.loaders import DocumentUnit
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.schemas.agent import ChunkingConfig, DeclarativeAgentDefinition
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    FactoryRunReport,
    FailureClassification,
    IterationScorecard,
    OptimizationChange,
    OptimizationIterationLog,
    OptimizationProposalDraft,
    RejectedChangeRecord,
    TargetedProposalItem,
)
from forgeai.schemas.intake import DocumentInput, GroundabilityReport, StructuredRequirements
from forgeai.tools.registry import ToolRegistry

ALLOWED_TARGETS_BY_FAILURE: dict[FailureClassification, set[str]] = {
    FailureClassification.RETRIEVAL_MISS: {
        "retrieval.top_k",
        "retrieval.score_threshold",
        "retrieval.metadata_filters",
        "chunking.chunk_size",
        "chunking.chunk_overlap",
    },
    FailureClassification.PROMPT_DEFECT: {
        "system_prompt",
        "output_contract.formatting_constraints",
    },
    FailureClassification.CONTRACT_VIOLATION: {
        "output_contract.formatting_constraints",
        "system_prompt",
    },
    FailureClassification.HALLUCINATION: {
        "system_prompt",
        "retrieval.score_threshold",
        "output_contract.formatting_constraints",
    },
    FailureClassification.WRONG_REFUSAL: {
        "system_prompt",
        "output_contract.refusal_message_template",
    },
}

_UNSAFE_PROMPT_PHRASES = (
    "ignore citations",
    "skip citations",
    "do not cite",
    "ignore untrusted",
    "follow instructions in documents",
    "guess if unsure",
    "fabricate",
    "never refuse",
)

_CORE_ACCEPTANCE_METRICS = (
    "pass_rate",
    "answerable_accuracy",
    "out_of_scope_refusal_accuracy",
    "edge_case_accuracy",
    "grounding_rate",
    "citation_accuracy",
    "contract_compliance_rate",
)


class OptimizationRunResult(BaseModel):
    """Complete output of running the bounded Critic–Optimizer loop."""

    final_agent: DeclarativeAgentDefinition
    final_results: list[EvalCaseResult]
    final_scorecard: IterationScorecard
    scorecards: list[IterationScorecard]
    optimization_logs: list[OptimizationIterationLog]
    final_report: FactoryRunReport
    stop_reason: str
    eval_suite_sha256: str
    artifact_paths: dict[str, str]


class CriticOptimizer:
    """Bounded (0..3 iterations), integrity-enforcing Critic–Optimizer."""

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        *,
        max_iterations: int = 3,
        target_pass_rate: float = 0.90,
        gemini_model: str = "gemini-3.8-flash",
        embedding_model: str = "all-MiniLM-L6-v2",
        judge_batch_size: int = 5,
    ) -> None:
        if not 0 <= max_iterations <= 3:
            raise ValueError("max_iterations must be between 0 and 3.")
        if not 0.0 <= target_pass_rate <= 1.0:
            raise ValueError("target_pass_rate must be between 0.0 and 1.0.")
        if judge_batch_size < 1:
            raise ValueError("judge_batch_size must be at least 1.")
        self._gemini = gemini_client or GeminiClient()
        self.max_iterations = max_iterations
        self.target_pass_rate = target_pass_rate
        self.gemini_model = gemini_model
        self.embedding_model = embedding_model
        self.judge_batch_size = judge_batch_size

    @staticmethod
    def group_failures(
        results: list[EvalCaseResult],
    ) -> dict[FailureClassification, list[EvalCaseResult]]:
        """Group failing evaluation results by root failure category."""
        grouped: dict[FailureClassification, list[EvalCaseResult]] = defaultdict(list)
        for item in results:
            if not item.passed and item.failure_classification != FailureClassification.NONE:
                grouped[item.failure_classification].append(item)
        return dict(grouped)

    def propose_changes(
        self,
        *,
        iteration: int,
        requirements: StructuredRequirements,
        agent: DeclarativeAgentDefinition,
        scorecard: IterationScorecard,
        eval_results: list[EvalCaseResult],
        documents: list[DocumentInput],
        units: list[DocumentUnit],
    ) -> OptimizationProposalDraft:
        """Ask Gemini Critic–Optimizer for targeted, minimal fixes to observed failures."""
        grouped = self.group_failures(eval_results)
        failure_groups_payload = {
            cat.value: [r.case_id for r in items] for cat, items in grouped.items()
        }
        failing_detail = [
            {
                "case_id": r.case_id,
                "category": r.category.value,
                "question": r.question,
                "expected_behavior": r.expected_behavior.value,
                "refused": r.refused,
                "failure_classification": r.failure_classification.value,
                "grounding_score": r.grounding_score,
                "correctness_score": r.correctness_score,
                "citation_valid": r.citation_valid,
                "contract_compliant": r.contract_compliant,
                "deterministic_issues": r.deterministic_issues,
                "omitted_facts": r.omitted_facts,
                "hallucinated_claims": r.hallucinated_claims,
                "retrieved_sources": [ev.source_file for ev in r.retrieved_evidence],
                "judge_reasoning": r.judge_reasoning,
            }
            for items in grouped.values()
            for r in items
        ]

        prompt = CRITIC_OPTIMIZER_PROMPT.format(
            iteration=iteration,
            max_iterations=self.max_iterations,
            requirements_json=requirements.model_dump_json(indent=2),
            agent_json=agent.model_dump_json(indent=2),
            scorecard_json=scorecard.model_dump_json(indent=2),
            failure_groups_json=json.dumps(failure_groups_payload, indent=2),
            failing_cases_json=json.dumps(failing_detail, indent=2),
            document_summary=summarize_documents(documents, units),
        )
        with telemetry_stage("critic_optimizer"):
            return self._gemini.generate_structured(
                prompt,
                OptimizationProposalDraft,
                temperature=0.1,
            )

    @staticmethod
    def apply_targeted_proposal(
        agent: DeclarativeAgentDefinition,
        proposal: OptimizationProposalDraft,
        *,
        observed_failures: set[FailureClassification],
        requirements: StructuredRequirements,
        groundability: GroundabilityReport,
    ) -> tuple[
        DeclarativeAgentDefinition | None,
        list[OptimizationChange],
        list[RejectedChangeRecord],
    ]:
        """Validate and apply ONLY safe, failure-relevant patches to `agent`."""
        proposed_changes: list[OptimizationChange] = []
        rejected: list[RejectedChangeRecord] = []

        if not proposal.actionable or not proposal.proposed_changes:
            return None, [], []

        updated = agent.model_copy(deep=True)
        applied: list[OptimizationChange] = []

        for item in proposal.proposed_changes:
            change_summary = OptimizationChange(
                target_component=item.target_component,
                failure_driver=item.failure_driver,
                rationale=item.rationale,
                summary=item.summary,
            )
            proposed_changes.append(change_summary)

            rejection_reason = CriticOptimizer._validate_patch_item(
                item,
                observed_failures=observed_failures,
            )
            if rejection_reason is not None:
                rejected.append(
                    RejectedChangeRecord(
                        target_component=item.target_component,
                        failure_driver=item.failure_driver,
                        summary=item.summary,
                        reason=rejection_reason,
                    )
                )
                continue

            did_modify, err = CriticOptimizer._apply_single_patch(
                updated,
                item,
                requirements=requirements,
                groundability=groundability,
            )
            if err is not None:
                rejected.append(
                    RejectedChangeRecord(
                        target_component=item.target_component,
                        failure_driver=item.failure_driver,
                        summary=item.summary,
                        reason=err,
                    )
                )
                continue
            if did_modify:
                applied.append(change_summary)

        if not applied:
            return None, [], rejected

        updated = updated.model_copy(update={"version": agent.version + 1})
        return updated, applied, rejected

    @staticmethod
    def _validate_patch_item(
        item: TargetedProposalItem,
        *,
        observed_failures: set[FailureClassification],
    ) -> str | None:
        if item.failure_driver == FailureClassification.NONE:
            return "Cannot apply change with failure_driver='none'."
        if item.failure_driver not in observed_failures:
            return (
                f"Failure driver '{item.failure_driver.value}' was not observed in failed cases."
            )
        allowed_targets = ALLOWED_TARGETS_BY_FAILURE.get(item.failure_driver, set())
        if item.target_component not in allowed_targets:
            return (
                f"Target '{item.target_component}' is unrelated to observed failure driver "
                f"'{item.failure_driver.value}'."
            )
        if item.prompt_instruction_addition:
            lowered = item.prompt_instruction_addition.lower()
            for phrase in _UNSAFE_PROMPT_PHRASES:
                if phrase in lowered:
                    return f"Rejected unsafe prompt modification containing '{phrase}'."
        if item.new_refusal_message_template is not None and not item.new_refusal_message_template.strip():
            return "Refusal message template cannot be empty."
        return None

    @staticmethod
    def _apply_single_patch(
        agent: DeclarativeAgentDefinition,
        item: TargetedProposalItem,
        *,
        requirements: StructuredRequirements,
        groundability: GroundabilityReport,
    ) -> tuple[bool, str | None]:
        target = item.target_component

        if target == "system_prompt":
            addition = (item.prompt_instruction_addition or "").strip()
            if not addition:
                return False, "Missing `prompt_instruction_addition` for system_prompt patch."
            if addition in agent.system_prompt:
                return False, "Proposed system_prompt instruction is already present."
            combined = f"{agent.system_prompt.strip()}\n\nTargeted Refinement ({item.failure_driver.value}):\n- {addition}"
            agent.system_prompt = _ensure_prompt_guardrails(combined, requirements, groundability)
            return True, None

        if target == "retrieval.top_k":
            if item.new_top_k is None or item.new_top_k == agent.retrieval.top_k:
                return False, "No effective `new_top_k` change specified."
            agent.retrieval = agent.retrieval.model_copy(update={"top_k": item.new_top_k})
            return True, None

        if target == "retrieval.score_threshold":
            if item.new_score_threshold == agent.retrieval.score_threshold:
                return False, "No effective `new_score_threshold` change specified."
            agent.retrieval = agent.retrieval.model_copy(
                update={"score_threshold": item.new_score_threshold}
            )
            return True, None

        if target == "retrieval.metadata_filters":
            if not item.clear_metadata_filters or not agent.retrieval.metadata_filters:
                return False, "No effective metadata_filters refinement specified."
            agent.retrieval = agent.retrieval.model_copy(update={"metadata_filters": {}})
            return True, None

        if target in {"chunking.chunk_size", "chunking.chunk_overlap"}:
            new_size = item.new_chunk_size if item.new_chunk_size is not None else agent.chunking.chunk_size
            new_overlap = (
                item.new_chunk_overlap
                if item.new_chunk_overlap is not None
                else agent.chunking.chunk_overlap
            )
            if new_size == agent.chunking.chunk_size and new_overlap == agent.chunking.chunk_overlap:
                return False, "No effective chunking parameter change specified."
            try:
                agent.chunking = ChunkingConfig(
                    chunk_size=new_size,
                    chunk_overlap=new_overlap,
                    csv_rows_per_unit=agent.chunking.csv_rows_per_unit,
                    separators=agent.chunking.separators,
                )
            except Exception as exc:
                return False, f"Invalid chunking configuration: {exc}"
            return True, None

        if target == "output_contract.refusal_message_template":
            msg = (item.new_refusal_message_template or "").strip()
            if not msg or msg == agent.output_contract.refusal_message_template:
                return False, "No effective refusal_message_template change specified."
            agent.output_contract = agent.output_contract.model_copy(
                update={"refusal_message_template": msg}
            )
            return True, None

        if target == "output_contract.formatting_constraints":
            constraint = (item.additional_formatting_constraint or "").strip()
            if not constraint or constraint in agent.output_contract.formatting_constraints:
                return False, "No effective formatting constraint addition specified."
            updated_constraints = list(agent.output_contract.formatting_constraints) + [constraint]
            agent.output_contract = agent.output_contract.model_copy(
                update={"formatting_constraints": updated_constraints}
            )
            return True, None

        return False, f"Unsupported target component: {target}"

    @staticmethod
    def evaluate_acceptance(
        before: IterationScorecard,
        after: IterationScorecard,
    ) -> tuple[bool, str, dict[str, float], list[str]]:
        """Decide whether candidate improves overall quality without damaging core metrics."""
        deltas = compute_metric_deltas(before, after)
        regressions = [
            f"{metric} ({deltas[metric]:+.4f})"
            for metric in _CORE_ACCEPTANCE_METRICS
            if deltas.get(metric, 0.0) < -0.05
        ]
        safety_drop = any(
            deltas.get(m, 0.0) < -0.10
            for m in (
                "out_of_scope_refusal_accuracy",
                "grounding_rate",
                "citation_accuracy",
                "contract_compliance_rate",
            )
        )

        if deltas["pass_rate"] > 0.0 and not safety_drop:
            note = f"Accepted: pass_rate improved by {deltas['pass_rate']:+.4f}."
            if regressions:
                note += f" Minor trade-offs observed: {', '.join(regressions)}."
            return True, note, deltas, regressions

        if deltas["pass_rate"] == 0.0:
            any_regression = any(deltas.get(m, 0.0) < 0.0 for m in _CORE_ACCEPTANCE_METRICS)
            quality_gain = (
                deltas.get("grounding_rate", 0.0) > 0.0
                or deltas.get("citation_accuracy", 0.0) > 0.0
                or deltas.get("contract_compliance_rate", 0.0) > 0.0
                or deltas.get("avg_correctness_score", 0.0) > 0.02
            )
            if quality_gain and not any_regression:
                return (
                    True,
                    "Accepted: pass_rate held steady with zero regressions and improved grounding/correctness.",
                    deltas,
                    regressions,
                )

        reason = (
            f"Rejected: pass_rate delta {deltas['pass_rate']:+.4f}; "
            + (f"regressions in {', '.join(regressions)}." if regressions else "no net improvement.")
        )
        return False, reason, deltas, regressions

    def optimize(
        self,
        *,
        run_id: str | None = None,
        requirements: StructuredRequirements,
        groundability: GroundabilityReport,
        baseline_agent: DeclarativeAgentDefinition,
        eval_cases: list[EvalCase],
        baseline_results: list[EvalCaseResult],
        baseline_scorecard: IterationScorecard,
        documents: list[DocumentInput],
        units: list[DocumentUnit],
        artifacts_dir: Path | str,
        vector_store: ChromaVectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        judge: HybridJudge | None = None,
        checkpoint_dir: Path | str | None = None,
        documents_dir: Path | str | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> OptimizationRunResult:
        """Run the complete bounded (0..3) optimization loop with full artifact persistence."""
        effective_run_id = run_id or f"{baseline_agent.agent_id}_{uuid4().hex[:6]}"
        frozen_cases, eval_hash = snapshot_eval_suite(eval_cases)
        store = OptimizationArtifactStore(artifacts_dir, effective_run_id)
        store.write_baseline(
            agent=baseline_agent,
            scorecard=baseline_scorecard,
            eval_cases=frozen_cases,
        )

        best_agent = baseline_agent.model_copy(deep=True)
        best_results = list(baseline_results)
        best_scorecard = baseline_scorecard
        scorecards: list[IterationScorecard] = [baseline_scorecard]
        logs: list[OptimizationIterationLog] = []

        stop_reason = ""
        if baseline_scorecard.total_cases > 0 and baseline_scorecard.pass_rate >= self.target_pass_rate:
            stop_reason = (
                "all_cases_passed"
                if baseline_scorecard.pass_rate == 1.0
                else "baseline_passed_target"
            )
        elif self.max_iterations == 0:
            stop_reason = "optimization_disabled_budget_zero"

        if not stop_reason:
            for iteration in range(1, self.max_iterations + 1):
                verify_eval_suite_integrity(frozen_cases, eval_hash)
                grouped = self.group_failures(best_results)
                if not grouped:
                    stop_reason = "all_cases_passed"
                    break

                observed_categories = list(grouped.keys())
                actionable_categories = [
                    cat for cat in observed_categories if cat in ALLOWED_TARGETS_BY_FAILURE
                ]
                failed_ids = [r.case_id for items in grouped.values() for r in items]

                if not actionable_categories:
                    stop_reason = "only_eval_case_defects_remaining"
                    logs.append(
                        OptimizationIterationLog(
                            iteration=iteration,
                            diagnosis_summary=(
                                "Remaining failures are evaluation-case defects rather than "
                                "agent defects; skipping prompt/retrieval mutation."
                            ),
                            observed_failure_categories=observed_categories,
                            failed_case_ids=failed_ids,
                            proposed_changes=[],
                            targeted_changes=[],
                            rejected_changes=[],
                            before_scorecard=best_scorecard,
                            after_scorecard=None,
                            metric_deltas={},
                            accepted=False,
                            acceptance_reason=(
                                "Evaluation-suite integrity prevents modifying frozen eval cases "
                                "or mutating the agent for non-agent eval defects."
                            ),
                            remaining_failed_case_ids=failed_ids,
                            agent_version_before=best_agent.version,
                            agent_version_after=best_agent.version,
                            stop_reason=stop_reason,
                        )
                    )
                    break

                proposal = self.propose_changes(
                    iteration=iteration,
                    requirements=requirements,
                    agent=best_agent,
                    scorecard=best_scorecard,
                    eval_results=best_results,
                    documents=documents,
                    units=units,
                )

                proposed_records = [
                    OptimizationChange(
                        target_component=item.target_component,
                        failure_driver=item.failure_driver,
                        rationale=item.rationale,
                        summary=item.summary,
                    )
                    for item in proposal.proposed_changes
                ]

                candidate_agent, applied_changes, rejected_changes = self.apply_targeted_proposal(
                    best_agent,
                    proposal,
                    observed_failures=set(observed_categories),
                    requirements=requirements,
                    groundability=groundability,
                )

                if candidate_agent is None or not applied_changes:
                    stop_reason = "no_actionable_proposal"
                    logs.append(
                        OptimizationIterationLog(
                            iteration=iteration,
                            diagnosis_summary=proposal.diagnosis_summary,
                            observed_failure_categories=observed_categories,
                            failed_case_ids=failed_ids,
                            proposed_changes=proposed_records,
                            targeted_changes=[],
                            rejected_changes=rejected_changes,
                            before_scorecard=best_scorecard,
                            after_scorecard=None,
                            metric_deltas={},
                            accepted=False,
                            acceptance_reason=proposal.no_action_reason
                            or "No valid failure-relevant changes could be applied.",
                            remaining_failed_case_ids=failed_ids,
                            agent_version_before=best_agent.version,
                            agent_version_after=best_agent.version,
                            stop_reason=stop_reason,
                        )
                    )
                    break

                # Re-verify immutable suite hash before executing candidate
                verify_eval_suite_integrity(frozen_cases, eval_hash)

                # If chunking changed, force fresh chunk indexing even when an external store exists
                active_store = (
                    vector_store
                    if (vector_store is not None and candidate_agent.chunking == best_agent.chunking)
                    else None
                )
                early_stop_info: dict[str, object] = {}
                candidate_results, candidate_scorecard = run_evaluation_suite(
                    candidate_agent,
                    frozen_cases,
                    units,
                    gemini_client=self._gemini,
                    vector_store=active_store,
                    embedding_provider=embedding_provider,
                    judge=judge,
                    iteration=iteration,
                    judge_batch_size=self.judge_batch_size,
                    checkpoint_dir=checkpoint_dir or store.run_dir,
                    run_id=effective_run_id,
                    eval_suite_sha256=eval_hash,
                    early_stop_baseline=best_scorecard,
                    prior_results=best_results,
                    early_stop_out=early_stop_info,
                    documents_dir=documents_dir,
                    tool_registry=tool_registry,
                )
                scorecards.append(candidate_scorecard)

                store.write_iteration(
                    iteration=iteration,
                    proposal=proposal,
                    agent=candidate_agent,
                    results=candidate_results,
                    scorecard=candidate_scorecard,
                )

                early_stopped = bool(early_stop_info.get("early_stopped", False))
                early_stop_reason = str(early_stop_info.get("early_stop_reason") or "")
                evaluated_cases_count = int(
                    early_stop_info.get("evaluated_cases_count") or len(candidate_results)
                )
                early_stopped_cases = int(early_stop_info.get("early_stopped_cases") or 0)
                requests_saved = int(early_stop_info.get("requests_saved") or 0)

                if early_stopped or candidate_scorecard.is_partial:
                    accepted = False
                    acceptance_reason = early_stop_reason or "Early stopped on partial evaluation."
                    deltas = {}
                else:
                    accepted, acceptance_reason, deltas, _regressions = self.evaluate_acceptance(
                        best_scorecard,
                        candidate_scorecard,
                    )

                version_before = best_agent.version
                prev_best_scorecard = best_scorecard
                if accepted:
                    best_agent = candidate_agent
                    best_results = candidate_results
                    best_scorecard = candidate_scorecard
                else:
                    rejected_changes.append(
                        RejectedChangeRecord(
                            target_component="candidate_agent",
                            failure_driver=actionable_categories[0],
                            summary=f"Candidate v{candidate_agent.version}",
                            reason=acceptance_reason,
                        )
                    )

                remaining_ids = [r.case_id for r in best_results if not r.passed]

                iter_stop: str | None = None
                if best_scorecard.pass_rate >= self.target_pass_rate:
                    iter_stop = (
                        "all_cases_passed"
                        if best_scorecard.pass_rate == 1.0
                        else "target_pass_rate_achieved"
                    )
                    stop_reason = iter_stop
                elif iteration == self.max_iterations:
                    iter_stop = "max_iterations_exhausted"
                    stop_reason = iter_stop

                logs.append(
                    OptimizationIterationLog(
                        iteration=iteration,
                        diagnosis_summary=proposal.diagnosis_summary,
                        observed_failure_categories=observed_categories,
                        failed_case_ids=failed_ids,
                        proposed_changes=proposed_records,
                        targeted_changes=applied_changes,
                        rejected_changes=rejected_changes,
                        before_scorecard=prev_best_scorecard,
                        after_scorecard=candidate_scorecard,
                        metric_deltas=deltas,
                        accepted=accepted,
                        acceptance_reason=acceptance_reason,
                        remaining_failed_case_ids=remaining_ids,
                        agent_version_before=version_before,
                        agent_version_after=best_agent.version,
                        early_stopped=early_stopped,
                        early_stop_reason=early_stop_reason,
                        evaluated_cases_count=evaluated_cases_count,
                        early_stopped_cases=early_stopped_cases,
                        requests_saved=requests_saved,
                        stop_reason=iter_stop,
                    )
                )

                if stop_reason:
                    break

        if not stop_reason:
            stop_reason = "max_iterations_exhausted"

        verify_eval_suite_integrity(frozen_cases, eval_hash)
        report = build_final_report(
            run_id=effective_run_id,
            gemini_model=self.gemini_model,
            embedding_model=self.embedding_model,
            requirements=requirements,
            groundability=groundability,
            final_agent=best_agent,
            eval_cases=frozen_cases,
            final_results=best_results,
            baseline_scorecard=baseline_scorecard,
            final_scorecard=best_scorecard,
            scorecards=scorecards,
            optimization_logs=logs,
            target_pass_rate=self.target_pass_rate,
            stop_reason=stop_reason,
            eval_suite_sha256=eval_hash,
            artifact_paths=store.paths,
            agent_generation_requests_saved=int(
                (baseline_agent.metadata or {}).get("agent_generation_requests_saved", 0) or 0
            ),
        )
        store.write_final(
            history=logs,
            final_agent=best_agent,
            final_report=report,
        )
        report = report.model_copy(update={"artifact_paths": dict(store.paths)})

        return OptimizationRunResult(
            final_agent=best_agent,
            final_results=best_results,
            final_scorecard=best_scorecard,
            scorecards=scorecards,
            optimization_logs=logs,
            final_report=report,
            stop_reason=stop_reason,
            eval_suite_sha256=eval_hash,
            artifact_paths=dict(store.paths),
        )
