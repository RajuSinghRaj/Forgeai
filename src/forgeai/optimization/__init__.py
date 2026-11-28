"""Phase 5 Critic–Optimizer, evaluation integrity guards, persistence, and final reports."""

from forgeai.evaluation.early_stop import (
    can_candidate_reach_acceptance,
    prioritize_cases_for_candidate,
)
from forgeai.optimization.integrity import (
    EvalIntegrityError,
    compute_eval_suite_hash,
    snapshot_eval_suite,
    verify_eval_suite_integrity,
)
from forgeai.optimization.optimizer import CriticOptimizer, OptimizationRunResult
from forgeai.optimization.persistence import OptimizationArtifactStore
from forgeai.optimization.report import build_final_report, compute_metric_deltas

__all__ = [
    "CriticOptimizer",
    "EvalIntegrityError",
    "OptimizationArtifactStore",
    "OptimizationRunResult",
    "build_final_report",
    "can_candidate_reach_acceptance",
    "compute_eval_suite_hash",
    "compute_metric_deltas",
    "prioritize_cases_for_candidate",
    "snapshot_eval_suite",
    "verify_eval_suite_integrity",
]

