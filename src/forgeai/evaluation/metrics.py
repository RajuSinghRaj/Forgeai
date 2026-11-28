"""Deterministic scorecard aggregation over per-case evaluation verdicts."""

from __future__ import annotations

from forgeai.schemas.evaluation import (
    EvalCaseCategory,
    EvalCaseResult,
    FailureClassification,
    IterationScorecard,
)


def _category_pass_rate(results: list[EvalCaseResult], category: EvalCaseCategory) -> float:
    subset = [r for r in results if r.category == category]
    if not subset:
        return 0.0
    return round(sum(1 for r in subset if r.passed) / len(subset), 4)


def compute_scorecard(
    results: list[EvalCaseResult],
    iteration: int,
    *,
    is_partial: bool = False,
    expected_total_cases: int | None = None,
) -> IterationScorecard:
    """Compute an `IterationScorecard` strictly from observed `EvalCaseResult` items."""
    total = len(results)
    failure_counts: dict[FailureClassification, int] = {
        FailureClassification.RETRIEVAL_MISS: 0,
        FailureClassification.PROMPT_DEFECT: 0,
        FailureClassification.CONTRACT_VIOLATION: 0,
        FailureClassification.HALLUCINATION: 0,
        FailureClassification.WRONG_REFUSAL: 0,
    }
    if total == 0:
        return IterationScorecard(
            iteration=iteration,
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            pass_rate=0.0,
            answerable_accuracy=0.0,
            out_of_scope_refusal_accuracy=0.0,
            edge_case_accuracy=0.0,
            grounding_rate=0.0,
            citation_accuracy=0.0,
            contract_compliance_rate=0.0,
            avg_grounding_score=0.0,
            avg_correctness_score=0.0,
            refusal_accuracy=0.0,
            failure_counts=failure_counts,
            is_partial=is_partial,
            expected_total_cases=expected_total_cases,
        )

    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    avg_grounding = sum(r.grounding_score for r in results) / total
    avg_correctness = sum(r.correctness_score for r in results) / total
    refusal_acc = sum(1 for r in results if r.refusal_correct) / total
    grounding_rate = (
        sum(1 for r in results if r.grounding_score >= 0.7 and not r.hallucinated_claims) / total
    )
    citation_acc = sum(1 for r in results if r.citation_valid) / total
    contract_rate = sum(1 for r in results if r.contract_compliant) / total

    oos_cases = [r for r in results if r.category == EvalCaseCategory.OUT_OF_SCOPE]
    oos_refusal_acc = (
        round(sum(1 for r in oos_cases if r.refusal_correct) / len(oos_cases), 4)
        if oos_cases
        else 0.0
    )

    for item in results:
        if item.failure_classification != FailureClassification.NONE:
            failure_counts[item.failure_classification] = (
                failure_counts.get(item.failure_classification, 0) + 1
            )

    return IterationScorecard(
        iteration=iteration,
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        pass_rate=round(passed / total, 4),
        answerable_accuracy=_category_pass_rate(results, EvalCaseCategory.ANSWERABLE),
        out_of_scope_refusal_accuracy=oos_refusal_acc,
        edge_case_accuracy=_category_pass_rate(results, EvalCaseCategory.EDGE_CASE),
        grounding_rate=round(grounding_rate, 4),
        citation_accuracy=round(citation_acc, 4),
        contract_compliance_rate=round(contract_rate, 4),
        avg_grounding_score=round(avg_grounding, 4),
        avg_correctness_score=round(avg_correctness, 4),
        refusal_accuracy=round(refusal_acc, 4),
        failure_counts=failure_counts,
        is_partial=is_partial,
        expected_total_cases=expected_total_cases,
    )
