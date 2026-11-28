"""Evaluation generation, hybrid judging, suite execution, and artifact persistence."""

from forgeai.evaluation.generator import EvalGenerator, InsufficientEvalCasesError
from forgeai.evaluation.judge import BatchJudgeValidationError, DeterministicCheckResult, HybridJudge
from forgeai.evaluation.metrics import compute_scorecard
from forgeai.evaluation.pdf_report import (
    build_evaluation_report_pdf,
    build_evaluation_report_pdf_from_state,
)
from forgeai.evaluation.persistence import EvaluationArtifactPaths, save_evaluation_artifacts
from forgeai.evaluation.suite_runner import run_evaluation_suite

__all__ = [
    "BatchJudgeValidationError",
    "DeterministicCheckResult",
    "EvalGenerator",
    "EvaluationArtifactPaths",
    "HybridJudge",
    "InsufficientEvalCasesError",
    "build_evaluation_report_pdf",
    "build_evaluation_report_pdf_from_state",
    "compute_scorecard",
    "run_evaluation_suite",
    "save_evaluation_artifacts",
]
