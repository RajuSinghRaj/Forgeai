"""Mathematical regression-aware early stopping and candidate case prioritization."""

from __future__ import annotations

from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseCategory,
    EvalCaseResult,
    IterationScorecard,
)

CORE_ACCEPTANCE_METRICS = (
    "pass_rate",
    "answerable_accuracy",
    "out_of_scope_refusal_accuracy",
    "edge_case_accuracy",
    "grounding_rate",
    "citation_accuracy",
    "contract_compliance_rate",
)

SAFETY_ACCEPTANCE_METRICS = (
    "out_of_scope_refusal_accuracy",
    "grounding_rate",
    "citation_accuracy",
    "contract_compliance_rate",
)


def prioritize_cases_for_candidate(
    eval_cases: list[EvalCase],
    prior_results: list[EvalCaseResult] | None = None,
) -> list[EvalCase]:
    """Order candidate evaluation cases: previously failed first, then high-risk/edge, then rest."""
    prior_by_id = {r.case_id: r for r in (prior_results or [])}
    failed_prior: list[EvalCase] = []
    high_risk_oos: list[EvalCase] = []
    high_risk_edge: list[EvalCase] = []
    remaining_answerable: list[EvalCase] = []

    for case in eval_cases:
        prev = prior_by_id.get(case.case_id)
        if prev is not None and not prev.passed:
            failed_prior.append(case)
        elif case.category == EvalCaseCategory.OUT_OF_SCOPE:
            high_risk_oos.append(case)
        elif case.category == EvalCaseCategory.EDGE_CASE:
            high_risk_edge.append(case)
        else:
            remaining_answerable.append(case)

    return failed_prior + high_risk_oos + high_risk_edge + remaining_answerable


def compute_upper_bounds(
    completed_results: list[EvalCaseResult],
    all_cases: list[EvalCase],
) -> dict[str, float]:
    """Compute exact achievable upper bounds assuming every unevaluated case scores 100%."""
    total = len(all_cases)
    if total == 0:
        return {m: 0.0 for m in (*CORE_ACCEPTANCE_METRICS, "avg_correctness_score")}

    done_by_id = {r.case_id: r for r in completed_results}
    done = list(done_by_id.values())
    remaining = [c for c in all_cases if c.case_id not in done_by_id]
    rem_count = len(remaining)

    max_passed = sum(1 for r in done if r.passed) + rem_count
    max_grounded = (
        sum(1 for r in done if r.grounding_score >= 0.7 and not r.hallucinated_claims)
        + rem_count
    )
    max_citations = sum(1 for r in done if r.citation_valid) + rem_count
    max_compliant = sum(1 for r in done if r.contract_compliant) + rem_count
    max_correctness_sum = sum(r.correctness_score for r in done) + rem_count

    total_ans = sum(1 for c in all_cases if c.category == EvalCaseCategory.ANSWERABLE)
    total_oos = sum(1 for c in all_cases if c.category == EvalCaseCategory.OUT_OF_SCOPE)
    total_edge = sum(1 for c in all_cases if c.category == EvalCaseCategory.EDGE_CASE)

    rem_ans = sum(1 for c in remaining if c.category == EvalCaseCategory.ANSWERABLE)
    rem_oos = sum(1 for c in remaining if c.category == EvalCaseCategory.OUT_OF_SCOPE)
    rem_edge = sum(1 for c in remaining if c.category == EvalCaseCategory.EDGE_CASE)

    done_ans_pass = sum(
        1 for r in done if r.category == EvalCaseCategory.ANSWERABLE and r.passed
    )
    done_oos_refusal = sum(
        1 for r in done if r.category == EvalCaseCategory.OUT_OF_SCOPE and r.refusal_correct
    )
    done_edge_pass = sum(
        1 for r in done if r.category == EvalCaseCategory.EDGE_CASE and r.passed
    )

    return {
        "pass_rate": round(max_passed / total, 4),
        "answerable_accuracy": round((done_ans_pass + rem_ans) / total_ans, 4) if total_ans else 0.0,
        "out_of_scope_refusal_accuracy": (
            round((done_oos_refusal + rem_oos) / total_oos, 4) if total_oos else 0.0
        ),
        "edge_case_accuracy": round((done_edge_pass + rem_edge) / total_edge, 4) if total_edge else 0.0,
        "grounding_rate": round(max_grounded / total, 4),
        "citation_accuracy": round(max_citations / total, 4),
        "contract_compliance_rate": round(max_compliant / total, 4),
        "avg_correctness_score": round(max_correctness_sum / total, 4),
    }


def can_candidate_reach_acceptance(
    before: IterationScorecard,
    completed_results: list[EvalCaseResult],
    all_cases: list[EvalCase],
) -> tuple[bool, str]:
    """Return False with reason ONLY when candidate cannot mathematically meet acceptance rules."""
    total = len(all_cases)
    if total == 0:
        return True, ""

    done_ids = {r.case_id for r in completed_results}
    remaining_count = sum(1 for c in all_cases if c.case_id not in done_ids)
    if remaining_count <= 0:
        return True, ""

    done_count = total - remaining_count
    bounds = compute_upper_bounds(completed_results, all_cases)

    # Path A requires pass_rate > before.pass_rate; Path B requires pass_rate == before.pass_rate.
    if bounds["pass_rate"] < before.pass_rate:
        return (
            False,
            f"Early stopped after {done_count}/{total} cases: maximum achievable pass_rate "
            f"({bounds['pass_rate']:.2%}) cannot reach retained best ({before.pass_rate:.2%}).",
        )

    # Safety thresholds block Path A on > -0.10 drop and Path B on any negative delta.
    for metric in SAFETY_ACCEPTANCE_METRICS:
        max_delta = round(bounds[metric] - float(getattr(before, metric)), 4)
        if max_delta < -0.10:
            return (
                False,
                f"Early stopped after {done_count}/{total} cases: maximum achievable {metric} "
                f"delta ({max_delta:+.4f}) violates -0.10 safety tolerance.",
            )

    # If pass_rate can at best tie before.pass_rate, only Path B acceptance is reachable.
    if bounds["pass_rate"] == before.pass_rate:
        for metric in CORE_ACCEPTANCE_METRICS:
            max_delta = round(bounds[metric] - float(getattr(before, metric)), 4)
            if max_delta < 0.0:
                return (
                    False,
                    f"Early stopped after {done_count}/{total} cases: pass_rate can at best tie "
                    f"({before.pass_rate:.2%}) and {metric} cannot avoid regression ({max_delta:+.4f}).",
                )

        max_quality_gain = (
            round(bounds["grounding_rate"] - before.grounding_rate, 4) > 0.0
            or round(bounds["citation_accuracy"] - before.citation_accuracy, 4) > 0.0
            or round(bounds["contract_compliance_rate"] - before.contract_compliance_rate, 4) > 0.0
            or round(bounds["avg_correctness_score"] - before.avg_correctness_score, 4) > 0.02
        )
        if not max_quality_gain:
            return (
                False,
                f"Early stopped after {done_count}/{total} cases: pass_rate can at best tie "
                f"({before.pass_rate:.2%}) with no achievable grounding, citation, contract, "
                f"or correctness improvement.",
            )

    return True, ""

