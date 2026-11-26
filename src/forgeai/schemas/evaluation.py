"""Schemas for generated evaluation cases, judge verdicts, optimization logs, and final reports."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.intake import GroundabilityReport, StructuredRequirements


class EvalCaseCategory(str, Enum):
    """Canonical category of an automatically generated evaluation case."""

    ANSWERABLE = "answerable"
    OUT_OF_SCOPE = "out_of_scope"
    EDGE_CASE = "edge_case"


class ExpectedBehavior(str, Enum):
    """Expected high-level action for an evaluation test case."""

    ANSWER = "answer"
    REFUSE = "refuse"


class FailureClassification(str, Enum):
    """Canonical failure taxonomy used by the Judge and Critic–Optimizer."""

    NONE = "none"
    RETRIEVAL_MISS = "retrieval_miss"
    PROMPT_DEFECT = "prompt_defect"
    HALLUCINATION = "hallucination"
    WRONG_REFUSAL = "wrong_refusal"


class EvalCase(BaseModel):
    """A single automatically generated evaluation test case grounded in source documents."""

    case_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=5)
    category: EvalCaseCategory = EvalCaseCategory.ANSWERABLE
    expected_behavior: ExpectedBehavior
    expected_facts: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    topic: str = Field(..., min_length=1)
    expected_refusal_behavior: str = Field(default="")
    evaluation_criteria: list[str] = Field(default_factory=list)
    is_cross_document: bool = False
    rationale: str = Field(default="")

    @model_validator(mode="after")
    def _validate_expectations(self) -> "EvalCase":
        if self.expected_behavior == ExpectedBehavior.ANSWER:
            if not self.expected_facts:
                raise ValueError("Answerable EvalCase must include at least one expected_fact.")
            if not self.source_documents:
                raise ValueError("Answerable EvalCase must cite at least one source_document.")
        if len(set(self.source_documents)) >= 2:
            self.is_cross_document = True
        return self


class EvalSuiteDraft(BaseModel):
    """Structured Gemini output containing automatically generated evaluation cases."""

    cases: list[EvalCase] = Field(default_factory=list)
    generation_notes: str = Field(default="")


class EvalGenerationMetadata(BaseModel):
    """Audit summary for an automatic evaluation-set generation pass."""

    total_raw_cases: int = Field(..., ge=0)
    accepted_cases: int = Field(..., ge=0)
    rejected_fabricated_sources: int = Field(default=0, ge=0)
    rejected_ungrounded_answerable: int = Field(default=0, ge=0)
    rejected_duplicates: int = Field(default=0, ge=0)
    shortfall: int = Field(default=0, ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    source_documents_referenced: list[str] = Field(default_factory=list)


class RetrievedEvidence(BaseModel):
    """Traceable document chunk retrieved during agent execution."""

    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    unit_label: str = Field(..., description="Page number, CSV row range, or section identifier.")
    text_excerpt: str = Field(..., min_length=1)
    distance: float | None = None


class JudgeLLMVerdict(BaseModel):
    """Structured semantic evaluation output returned by Gemini Judge."""

    grounded: bool
    grounding_score: float = Field(..., ge=0.0, le=1.0)
    correctness_score: float = Field(..., ge=0.0, le=1.0)
    addresses_question: bool
    refusal_appropriate: bool
    omitted_facts: list[str] = Field(default_factory=list)
    hallucinated_claims: list[str] = Field(default_factory=list)
    unsupported_verbosity: bool = False
    suggested_failure_classification: FailureClassification = FailureClassification.NONE
    reasoning: str = Field(..., min_length=5)


class EvalCaseResult(BaseModel):
    """Per-case execution output and hybrid Judge evaluation verdict."""

    case_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    category: EvalCaseCategory = EvalCaseCategory.ANSWERABLE
    expected_behavior: ExpectedBehavior
    agent_answer: str
    refused: bool
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    grounding_score: float = Field(..., ge=0.0, le=1.0)
    correctness_score: float = Field(..., ge=0.0, le=1.0)
    refusal_correct: bool
    citation_valid: bool = True
    contract_compliant: bool = True
    omitted_facts: list[str] = Field(default_factory=list)
    hallucinated_claims: list[str] = Field(default_factory=list)
    deterministic_issues: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    passed: bool
    failure_classification: FailureClassification = FailureClassification.NONE
    judge_reasoning: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_failure_classification(self) -> "EvalCaseResult":
        if self.passed and self.failure_classification != FailureClassification.NONE:
            raise ValueError("Passing EvalCaseResult must have failure_classification='none'.")
        if not self.passed and self.failure_classification == FailureClassification.NONE:
            raise ValueError("Failing EvalCaseResult must classify the failure cause.")
        return self


class IterationScorecard(BaseModel):
    """Aggregated quantitative metrics for one evaluation/optimization iteration."""

    iteration: int = Field(..., ge=0, le=3)
    total_cases: int = Field(..., ge=0)
    passed_cases: int = Field(..., ge=0)
    failed_cases: int = Field(default=0, ge=0)
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    answerable_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    out_of_scope_refusal_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    edge_case_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    contract_compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_grounding_score: float = Field(..., ge=0.0, le=1.0)
    avg_correctness_score: float = Field(..., ge=0.0, le=1.0)
    refusal_accuracy: float = Field(..., ge=0.0, le=1.0)
    failure_counts: dict[FailureClassification, int] = Field(default_factory=dict)


class OptimizationChange(BaseModel):
    """Concrete targeted change proposed or applied by the Critic–Optimizer."""

    target_component: str = Field(
        ...,
        description="Affected component (e.g. system_prompt, retrieval.top_k, chunking.chunk_size, output_contract).",
    )
    failure_driver: FailureClassification
    rationale: str = Field(..., min_length=5)
    summary: str = Field(..., min_length=5)


class RootCauseGroup(BaseModel):
    """Root-cause summary for one observed failure category."""

    failure_category: FailureClassification
    failed_case_ids: list[str] = Field(default_factory=list)
    root_cause: str = Field(..., min_length=5)


AllowedOptimizationTarget = Literal[
    "system_prompt",
    "retrieval.top_k",
    "retrieval.score_threshold",
    "retrieval.metadata_filters",
    "chunking.chunk_size",
    "chunking.chunk_overlap",
    "output_contract.refusal_message_template",
    "output_contract.formatting_constraints",
]


class TargetedProposalItem(BaseModel):
    """Structured, typed patch proposed by the Critic–Optimizer."""

    target_component: AllowedOptimizationTarget
    failure_driver: FailureClassification
    rationale: str = Field(..., min_length=5)
    summary: str = Field(..., min_length=5)
    prompt_instruction_addition: str | None = None
    new_top_k: int | None = Field(default=None, ge=1, le=20)
    new_score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    clear_metadata_filters: bool = False
    new_chunk_size: int | None = Field(default=None, ge=100, le=4000)
    new_chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    new_refusal_message_template: str | None = None
    additional_formatting_constraint: str | None = None


class OptimizationProposalDraft(BaseModel):
    """Structured Gemini Critic–Optimizer output for one iteration."""

    actionable: bool = True
    diagnosis_summary: str = Field(..., min_length=5)
    root_causes: list[RootCauseGroup] = Field(default_factory=list)
    proposed_changes: list[TargetedProposalItem] = Field(default_factory=list)
    no_action_reason: str | None = None


class RejectedChangeRecord(BaseModel):
    """Audit record for a proposed change that was rejected before or after evaluation."""

    target_component: str
    failure_driver: FailureClassification
    summary: str
    reason: str


class OptimizationIterationLog(BaseModel):
    """Audit record for a single Critic–Optimizer iteration."""

    iteration: int = Field(..., ge=1, le=3)
    diagnosis_summary: str = Field(..., min_length=5)
    observed_failure_categories: list[FailureClassification] = Field(default_factory=list)
    failed_case_ids: list[str] = Field(default_factory=list)
    proposed_changes: list[OptimizationChange] = Field(default_factory=list)
    targeted_changes: list[OptimizationChange] = Field(default_factory=list)
    rejected_changes: list[RejectedChangeRecord] = Field(default_factory=list)
    before_scorecard: IterationScorecard
    after_scorecard: IterationScorecard | None = None
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    accepted: bool = False
    acceptance_reason: str = ""
    remaining_failed_case_ids: list[str] = Field(default_factory=list)
    agent_version_before: int = Field(..., ge=1)
    agent_version_after: int = Field(..., ge=1)
    stop_reason: str | None = None


class FactoryRunReport(BaseModel):
    """Complete, evidence-backed final report emitted by a ForgeAI factory run."""

    run_id: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gemini_model: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    requirements: StructuredRequirements
    groundability: GroundabilityReport
    final_agent: DeclarativeAgentDefinition
    baseline_scorecard: IterationScorecard | None = None
    final_scorecard: IterationScorecard | None = None
    initial_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    final_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    eval_cases: list[EvalCase] = Field(default_factory=list)
    final_results: list[EvalCaseResult] = Field(default_factory=list)
    remaining_failed_cases: list[EvalCaseResult] = Field(default_factory=list)
    scorecards: list[IterationScorecard] = Field(default_factory=list)
    optimization_logs: list[OptimizationIterationLog] = Field(default_factory=list, max_length=3)
    iterations_used: int = Field(default=0, ge=0, le=3)
    target_pass_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    passing_criteria_met: bool = False
    eval_suite_sha256: str = ""
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    honest_limitations: list[str] = Field(default_factory=list)
    stop_reason: str = Field(..., min_length=1)
