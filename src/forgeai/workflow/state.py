"""Shared state schema for the LangGraph-orchestrated ForgeAI factory."""

from __future__ import annotations

from typing import Literal, TypedDict

from forgeai.retrieval.loaders import DocumentUnit
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


class FactoryState(TypedDict, total=False):
    """Explicit state passed across LangGraph nodes in the ForgeAI workflow."""

    run_id: str
    checkpoint_dir: str
    checkpoint_completed_cases: int
    checkpoint_total_cases: int
    brief_input: BriefInput
    csv_rows_per_unit: int
    documents: list[DocumentInput]
    document_units: list[DocumentUnit]
    clarifying_questions: list[ClarifyingQuestion]
    clarifying_answers: list[ClarifyingAnswer] | dict[str, str]
    unattended_policy: Literal["stop_for_clarification", "proceed_with_assumptions"]
    unattended_assumptions: list[str]
    awaiting_user_answers: bool
    interview_completed: bool
    requirements: StructuredRequirements
    groundability: GroundabilityReport
    approved_with_gaps: bool
    phase2_approved: bool
    stop_after_phase2: bool
    stop_after_phase3: bool
    stop_after_eval_generation: bool
    stop_after_phase4: bool
    phase3_status: str
    phase4_status: str
    phase5_status: str
    agent_definition: DeclarativeAgentDefinition
    agent_artifact_path: str
    min_eval_cases: int
    eval_cases: list[EvalCase]
    eval_suite_sha256: str
    eval_generation_metadata: EvalGenerationMetadata
    eval_results: list[EvalCaseResult]
    eval_artifact_paths: dict[str, str]
    evaluated_signature: tuple[str, int, str]
    optimized_signature: tuple[str, int, str]
    baseline_agent: DeclarativeAgentDefinition
    baseline_scorecard: IterationScorecard
    baseline_results: list[EvalCaseResult]
    best_agent: DeclarativeAgentDefinition
    best_scorecard: IterationScorecard
    best_results: list[EvalCaseResult]
    latest_candidate_agent: DeclarativeAgentDefinition
    latest_candidate_scorecard: IterationScorecard
    latest_candidate_results: list[EvalCaseResult]
    pending_proposal: OptimizationProposalDraft
    pending_applied_changes: list[OptimizationChange]
    pending_rejected_changes: list[RejectedChangeRecord]
    pending_observed_categories: list[FailureClassification]
    pending_failed_ids: list[str]
    optimization_halted: bool
    optimization_run_dir: str
    optimization_artifact_paths: dict[str, str]
    scorecards: list[IterationScorecard]
    optimization_logs: list[OptimizationIterationLog]
    iteration: int
    max_iterations: int
    target_pass_rate: float
    stop_reason: str
    final_report: FactoryRunReport

