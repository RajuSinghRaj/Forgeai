"""Honest final report builder and metric comparison utilities."""

from __future__ import annotations

from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    FactoryRunReport,
    IterationScorecard,
    OptimizationIterationLog,
)
from forgeai.schemas.intake import GroundabilityReport, StructuredRequirements

COMPARED_METRICS = (
    "pass_rate",
    "answerable_accuracy",
    "out_of_scope_refusal_accuracy",
    "edge_case_accuracy",
    "grounding_rate",
    "citation_accuracy",
    "contract_compliance_rate",
    "avg_grounding_score",
    "avg_correctness_score",
    "refusal_accuracy",
)


def compute_metric_deltas(
    before: IterationScorecard,
    after: IterationScorecard,
) -> dict[str, float]:
    """Compute signed metric deltas (`after - before`) across scorecard metrics."""
    return {
        metric: round(float(getattr(after, metric)) - float(getattr(before, metric)), 4)
        for metric in COMPARED_METRICS
    }


def build_final_report(
    *,
    run_id: str,
    gemini_model: str,
    embedding_model: str,
    requirements: StructuredRequirements,
    groundability: GroundabilityReport,
    final_agent: DeclarativeAgentDefinition,
    eval_cases: list[EvalCase],
    final_results: list[EvalCaseResult],
    baseline_scorecard: IterationScorecard,
    final_scorecard: IterationScorecard,
    scorecards: list[IterationScorecard],
    optimization_logs: list[OptimizationIterationLog],
    target_pass_rate: float,
    stop_reason: str,
    eval_suite_sha256: str = "",
    artifact_paths: dict[str, str] | None = None,
    agent_generation_requests_saved: int | None = None,
) -> FactoryRunReport:
    """Construct an evidence-backed `FactoryRunReport` without inflating scores."""
    remaining_failed = [r for r in final_results if not r.passed]
    metric_deltas = compute_metric_deltas(baseline_scorecard, final_scorecard)
    passing_met = (
        final_scorecard.total_cases > 0 and final_scorecard.pass_rate >= target_pass_rate
    )

    limitations = list(final_agent.groundability_limitations)
    if groundability.approved_with_gaps and groundability.ungrounded_topics:
        limitations.append(
            "Proceeded with requester-approved corpus gaps: "
            + ", ".join(groundability.ungrounded_topics)
        )
    if not passing_met:
        limitations.append(
            f"Target pass rate ({target_pass_rate:.0%}) was not achieved; final pass rate is "
            f"{final_scorecard.pass_rate:.0%} ({final_scorecard.passed_cases}/"
            f"{final_scorecard.total_cases} passed) after {len(optimization_logs)} iteration(s). "
            f"Stop reason: {stop_reason}."
        )
    if optimization_logs and metric_deltas.get("pass_rate", 0.0) <= 0.0 and remaining_failed:
        limitations.append(
            "Optimization iterations produced no net pass-rate improvement over the baseline agent."
        )
    for failed in remaining_failed:
        limitations.append(
            f"Remaining failure {failed.case_id} [{failed.failure_classification.value}]: "
            f"{failed.question}"
        )
    for log in optimization_logs:
        if not log.accepted and log.acceptance_reason:
            limitations.append(
                f"Iteration {log.iteration} candidate rejected: {log.acceptance_reason}"
            )
        if log.early_stopped and log.early_stop_reason:
            limitations.append(
                f"Iteration {log.iteration} candidate early-stopped: {log.early_stop_reason}"
            )

    deduped_limitations = list(dict.fromkeys(limitations))
    early_stops = sum(1 for log in optimization_logs if log.early_stopped)
    stopped_cases_total = sum(int(log.early_stopped_cases or 0) for log in optimization_logs)
    requests_saved_es = sum(int(log.requests_saved or 0) for log in optimization_logs)
    gen_saved = (
        int(agent_generation_requests_saved)
        if agent_generation_requests_saved is not None
        else int((final_agent.metadata or {}).get("agent_generation_requests_saved", 0) or 0)
    )

    return FactoryRunReport(
        run_id=run_id,
        gemini_model=gemini_model,
        embedding_model=embedding_model,
        requirements=requirements,
        groundability=groundability,
        final_agent=final_agent,
        baseline_scorecard=baseline_scorecard,
        final_scorecard=final_scorecard,
        initial_pass_rate=baseline_scorecard.pass_rate,
        final_pass_rate=final_scorecard.pass_rate,
        metric_deltas=metric_deltas,
        eval_cases=eval_cases,
        final_results=final_results,
        remaining_failed_cases=remaining_failed,
        scorecards=scorecards,
        optimization_logs=optimization_logs,
        iterations_used=len(optimization_logs),
        target_pass_rate=target_pass_rate,
        passing_criteria_met=passing_met,
        eval_suite_sha256=eval_suite_sha256,
        artifact_paths=dict(artifact_paths or {}),
        honest_limitations=deduped_limitations,
        optimization_early_stops=early_stops,
        early_stopped_cases_total=stopped_cases_total,
        requests_saved_early_stopping=requests_saved_es,
        agent_generation_requests_saved=gen_saved,
        stop_reason=stop_reason,
    )

