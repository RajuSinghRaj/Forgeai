"""Hybrid Automated Judge combining deterministic integrity checks with Gemini assessment."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from forgeai.llm.gemini_client import GeminiClient
from forgeai.prompts.templates import JUDGE_PROMPT
from forgeai.runner.agent_runner import AgentResponse
from forgeai.schemas.agent import OutputContract
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    ExpectedBehavior,
    FailureClassification,
    JudgeLLMVerdict,
)

_FILENAME_RE = re.compile(r"\b[\w./-]+\.(?:pdf|csv|md|txt)\b", re.IGNORECASE)


class DeterministicCheckResult(BaseModel):
    """Objective programmatic checks executed independently of the LLM judge."""

    citation_valid: bool
    cited_sources: list[str] = Field(default_factory=list)
    invented_citations: list[str] = Field(default_factory=list)
    contract_compliant: bool
    refusal_correct: bool
    missing_expected_tokens: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def _extract_cited_filenames(answer: str) -> list[str]:
    matches = _FILENAME_RE.findall(answer)
    # Preserve order while deduplicating
    return list(dict.fromkeys(matches))


def _extract_salient_tokens(expected_facts: list[str]) -> list[str]:
    """Extract numeric literals or uppercase/hyphenated codes worth checking deterministically."""
    tokens: list[str] = []
    for fact in expected_facts:
        for tok in re.findall(r"\b(?:[A-Z]+-\d+|\d+(?:\.\d+)?)\b", fact):
            if tok not in tokens:
                tokens.append(tok)
    return tokens


class HybridJudge:
    """Evaluates candidate agent responses with deterministic safeguards + Gemini semantic grading."""

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        *,
        known_sources: set[str] | None = None,
        enable_consistency_recheck: bool = False,
    ) -> None:
        self._gemini = gemini_client or GeminiClient()
        self._known_sources = set(known_sources or ())
        self._enable_consistency_recheck = enable_consistency_recheck

    def run_deterministic_checks(
        self,
        case: EvalCase,
        response: AgentResponse,
        contract: OutputContract,
        *,
        known_sources: set[str] | None = None,
    ) -> DeterministicCheckResult:
        """Verify citations, source existence, contract rules, and refusal invariants."""
        issues: list[str] = []
        answer = response.answer.strip()
        evidence_sources = {ev.source_file for ev in response.evidence}
        allowed_sources = set(self._known_sources) | set(known_sources or ()) | evidence_sources

        answer_cited = _extract_cited_filenames(answer)
        invented_citations = [
            src
            for src in answer_cited
            if src not in allowed_sources
            or (evidence_sources and src not in evidence_sources)
        ]

        citation_valid = True
        if invented_citations:
            citation_valid = False
            issues.append(f"Unbacked or invented source citations: {invented_citations}.")

        if (
            case.expected_behavior == ExpectedBehavior.ANSWER
            and not response.refused
            and contract.require_citations
        ):
            if not answer_cited:
                citation_valid = False
                issues.append("Answer lacks required inline source citation.")

        # Contract compliance checks
        contract_compliant = True
        if not answer:
            contract_compliant = False
            issues.append("Agent returned an empty response.")
        elif not response.refused:
            if contract.max_response_words and len(answer.split()) > contract.max_response_words:
                contract_compliant = False
                issues.append(
                    f"Response exceeds max_response_words ({len(answer.split())} > {contract.max_response_words})."
                )
            if contract.response_format.lower() == "json":
                try:
                    parsed = json.loads(answer)
                    if isinstance(parsed, dict):
                        missing_keys = [
                            field for field in contract.required_fields if field not in parsed
                        ]
                        if missing_keys:
                            contract_compliant = False
                            issues.append(f"JSON response missing required fields: {missing_keys}.")
                    else:
                        contract_compliant = False
                        issues.append("JSON response must be a JSON object.")
                except json.JSONDecodeError:
                    contract_compliant = False
                    issues.append("Response failed JSON contract parsing.")
            else:
                lowered = answer.lower()
                missing_fields = [
                    field for field in contract.required_fields if field.lower() not in lowered
                ]
                if missing_fields:
                    contract_compliant = False
                    issues.append(f"Response missing required contract fields: {missing_fields}.")
                if contract.response_format.lower() == "structured_sections":
                    missing_sections = [
                        sec for sec in contract.response_sections if sec.lower() not in lowered
                    ]
                    if missing_sections:
                        contract_compliant = False
                        issues.append(
                            f"Response missing required sections: {missing_sections}."
                        )

        # Refusal invariant checks
        refusal_correct = True
        if case.expected_behavior == ExpectedBehavior.REFUSE and not response.refused:
            refusal_correct = False
            issues.append("Out-of-scope or unsupported request was answered instead of refused.")
        elif case.expected_behavior == ExpectedBehavior.ANSWER and response.refused:
            refusal_correct = False
            issues.append("Answerable request was incorrectly refused.")

        missing_tokens: list[str] = []
        if case.expected_behavior == ExpectedBehavior.ANSWER and not response.refused:
            lowered_ans = answer.lower()
            for tok in _extract_salient_tokens(case.expected_facts):
                if tok.lower() not in lowered_ans:
                    missing_tokens.append(tok)

        return DeterministicCheckResult(
            citation_valid=citation_valid,
            cited_sources=answer_cited,
            invented_citations=invented_citations,
            contract_compliant=contract_compliant,
            refusal_correct=refusal_correct,
            missing_expected_tokens=missing_tokens,
            issues=issues,
        )

    def evaluate_case(
        self,
        case: EvalCase,
        response: AgentResponse,
        contract: OutputContract,
        *,
        known_sources: set[str] | None = None,
    ) -> EvalCaseResult:
        """Assess a single case using deterministic checks plus blinded Gemini judgment."""
        det = self.run_deterministic_checks(
            case,
            response,
            contract,
            known_sources=known_sources,
        )

        # Blinded payload: never reveal generator/agent model identity to the judge.
        eval_case_payload = case.model_dump(mode="json")
        agent_output_payload = {
            "agent_answer": response.answer,
            "refused": response.refused,
            "retrieved_evidence": [
                {
                    "source_file": ev.source_file,
                    "unit_label": ev.unit_label,
                    "text_excerpt": ev.text_excerpt,
                }
                for ev in response.evidence
            ],
        }
        prompt = JUDGE_PROMPT.format(
            eval_case_json=json.dumps(eval_case_payload, indent=2),
            agent_output_json=json.dumps(agent_output_payload, indent=2),
            deterministic_notes=det.model_dump_json(indent=2),
        )
        llm_verdict = self._gemini.generate_structured(prompt, JudgeLLMVerdict, temperature=0.0)

        # Optional order-swapped consistency recheck on borderline answerable cases
        disagreements: list[str] = []
        if (
            self._enable_consistency_recheck
            and case.expected_behavior == ExpectedBehavior.ANSWER
            and not det.issues
            and 0.45 <= llm_verdict.correctness_score <= 0.75
            and len(response.evidence) > 1
        ):
            swapped_payload = dict(agent_output_payload)
            swapped_payload["retrieved_evidence"] = list(
                reversed(agent_output_payload["retrieved_evidence"])
            )
            swapped_prompt = JUDGE_PROMPT.format(
                eval_case_json=json.dumps(eval_case_payload, indent=2),
                agent_output_json=json.dumps(swapped_payload, indent=2),
                deterministic_notes=det.model_dump_json(indent=2),
            )
            recheck = self._gemini.generate_structured(
                swapped_prompt, JudgeLLMVerdict, temperature=0.0
            )
            if abs(recheck.correctness_score - llm_verdict.correctness_score) >= 0.2:
                disagreements.append(
                    "Order-swapped LLM judge recheck diverged; averaging grounding/correctness scores."
                )
            llm_verdict = llm_verdict.model_copy(
                update={
                    "grounding_score": round(
                        (llm_verdict.grounding_score + recheck.grounding_score) / 2.0, 4
                    ),
                    "correctness_score": round(
                        (llm_verdict.correctness_score + recheck.correctness_score) / 2.0, 4
                    ),
                    "grounded": llm_verdict.grounded and recheck.grounded,
                }
            )

        return self._fuse_verdicts(case, response, det, llm_verdict, disagreements)

    @staticmethod
    def _fuse_verdicts(
        case: EvalCase,
        response: AgentResponse,
        det: DeterministicCheckResult,
        llm: JudgeLLMVerdict,
        disagreements: list[str],
    ) -> EvalCaseResult:
        grounding_score = llm.grounding_score
        correctness_score = llm.correctness_score

        if det.invented_citations and llm.grounded:
            disagreements.append(
                "LLM judged answer grounded, overridden by deterministic invented-citation failure."
            )
            grounding_score = min(grounding_score, 0.3)

        if not det.citation_valid and not det.invented_citations and llm.grounded:
            disagreements.append(
                "LLM judged answer grounded despite missing required inline source citations."
            )

        if case.expected_behavior == ExpectedBehavior.REFUSE and not response.refused and llm.grounded:
            disagreements.append(
                "LLM marked out-of-scope response grounded without refusal; overridden by refusal guard."
            )
            grounding_score = min(grounding_score, 0.4)

        if case.expected_behavior == ExpectedBehavior.ANSWER and response.refused and llm.refusal_appropriate:
            disagreements.append(
                "LLM accepted refusal on an answerable case; overridden by deterministic check."
            )

        if llm.unsupported_verbosity:
            grounding_score = min(grounding_score, 0.6)

        evidence_sources = {ev.source_file for ev in response.evidence}
        expected_sources_retrieved = bool(set(case.source_documents) & evidence_sources)

        # Determine final pass/fail and canonical failure classification
        failure = FailureClassification.NONE
        if case.expected_behavior == ExpectedBehavior.ANSWER and response.refused:
            if not response.evidence or (case.source_documents and not expected_sources_retrieved):
                failure = FailureClassification.RETRIEVAL_MISS
            else:
                failure = FailureClassification.WRONG_REFUSAL
        elif case.expected_behavior == ExpectedBehavior.REFUSE and not response.refused:
            if llm.hallucinated_claims or not llm.grounded or det.invented_citations:
                failure = FailureClassification.HALLUCINATION
            else:
                failure = FailureClassification.PROMPT_DEFECT
        elif case.expected_behavior == ExpectedBehavior.REFUSE and response.refused:
            if not det.contract_compliant or not llm.refusal_appropriate:
                failure = FailureClassification.PROMPT_DEFECT
        else:
            # Answerable and answered
            if det.invented_citations or llm.hallucinated_claims or not llm.grounded or llm.unsupported_verbosity or grounding_score < 0.7:
                failure = FailureClassification.HALLUCINATION
            elif (
                (case.source_documents and not expected_sources_retrieved)
                or llm.suggested_failure_classification == FailureClassification.RETRIEVAL_MISS
            ) and correctness_score < 0.7:
                failure = FailureClassification.RETRIEVAL_MISS
            elif not det.citation_valid or not det.contract_compliant or not llm.addresses_question or correctness_score < 0.7:
                failure = FailureClassification.PROMPT_DEFECT

        passed = (
            failure == FailureClassification.NONE
            and det.refusal_correct
            and det.citation_valid
            and det.contract_compliant
            and not det.issues
        )
        if not passed and failure == FailureClassification.NONE:
            failure = FailureClassification.PROMPT_DEFECT

        reasoning_parts = [llm.reasoning.strip()]
        if det.issues:
            reasoning_parts.append("Deterministic issues: " + "; ".join(det.issues))
        if disagreements:
            reasoning_parts.append("Disagreements: " + "; ".join(disagreements))

        return EvalCaseResult(
            case_id=case.case_id,
            question=case.question,
            category=case.category,
            expected_behavior=case.expected_behavior,
            agent_answer=response.answer,
            refused=response.refused,
            retrieved_evidence=response.evidence,
            cited_sources=det.cited_sources,
            grounding_score=round(grounding_score, 4),
            correctness_score=round(correctness_score, 4),
            refusal_correct=det.refusal_correct,
            citation_valid=det.citation_valid,
            contract_compliant=det.contract_compliant,
            omitted_facts=list(llm.omitted_facts),
            hallucinated_claims=list(llm.hallucinated_claims),
            deterministic_issues=det.issues,
            disagreements=disagreements,
            passed=passed,
            failure_classification=failure,
            judge_reasoning=" | ".join(reasoning_parts),
        )

