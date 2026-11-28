"""Hybrid Automated Judge combining deterministic integrity checks with Gemini assessment."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from forgeai.llm.gemini_client import GeminiClient
from forgeai.llm.telemetry import telemetry_stage
from forgeai.prompts.templates import BATCH_JUDGE_PROMPT, JUDGE_PROMPT
from forgeai.runner.agent_runner import AgentResponse
from forgeai.schemas.agent import OutputContract
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalCaseResult,
    ExpectedBehavior,
    FailureClassification,
    JudgeLLMBatchVerdict,
    JudgeLLMVerdict,
)

_FILENAME_RE = re.compile(r"\b[\w./-]+\.(?:pdf|csv|md|txt)\b", re.IGNORECASE)


class BatchJudgeValidationError(ValueError):
    """Raised when a batched judge payload has missing, duplicate, unknown, or malformed IDs."""


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


def _salient_fact_words(fact: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", fact.lower()) if len(w) >= 4]


def _evidence_contains_expected_facts(case: EvalCase, response: AgentResponse) -> bool:
    """Return True when retrieved evidence contains expected tokens/fact phrases."""
    if not response.evidence:
        return False
    combined = " ".join(ev.text_excerpt.lower() for ev in response.evidence)
    salient = _extract_salient_tokens(case.expected_facts)
    if salient and any(tok.lower() in combined for tok in salient):
        return True
    for fact in case.expected_facts:
        words = _salient_fact_words(fact)
        if words and (sum(1 for w in words if w in combined) / len(words)) >= 0.6:
            return True
    return False


_DIRECT_MATCH_STOPWORDS = {
    "about", "after", "been", "being", "could", "does", "from", "have", "into",
    "just", "more", "most", "much", "must", "only", "other", "same", "should",
    "some", "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "very", "what", "when", "where", "which", "while", "whom",
    "whose", "will", "with", "would", "your",
}


def _evidence_directly_answers_question(case: EvalCase, response: AgentResponse) -> bool:
    """Conservative check confirming retrieved evidence directly matches question wording."""
    if not response.evidence or (not case.source_documents and not case.evidence_references):
        return False
    q_words = {
        w
        for w in re.findall(r"[a-z0-9]+", case.question.lower())
        if len(w) >= 4 and w not in _DIRECT_MATCH_STOPWORDS
    }
    if len(q_words) < 3:
        return False
    for ev in response.evidence:
        ev_words = set(re.findall(r"[a-z0-9]+", ev.text_excerpt.lower()))
        if len(q_words & ev_words) / len(q_words) >= 0.65:
            return True
    return False


_MISSING_INFO_CLAIM_RE = re.compile(
    r"\b(?:do not list|does not list|do not specify|does not specify|do not document|"
    r"does not document|not documented|not specified|not listed|not covered|"
    r"not available|unavailable|missing from|does not contain|do not contain)\b",
    re.IGNORECASE,
)


def _normalize_contract_label(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-&/*#`:]+", " ", value.lower())).strip()


def _has_markdown_field_equivalent(
    field: str,
    *,
    raw_lowered: str,
    norm_answer: str,
    answer_cited: list[str],
    answer: str,
) -> bool:
    """Format-aware Markdown field check recognizing human-readable headings and citations."""
    if field.lower() in raw_lowered:
        return True
    norm_field = _normalize_contract_label(field)
    if not norm_field:
        return True
    if norm_field in norm_answer:
        return True

    citation_aliases = {"citations", "citation", "source citations", "source references", "references", "sources"}
    if norm_field in citation_aliases:
        return bool(answer_cited) or any(
            alias in norm_answer for alias in ("source references", "references", "source:", "citations")
        )

    answer_aliases = {"answer summary", "summary", "direct answer", "answer", "response"}
    if norm_field in answer_aliases:
        return any(
            alias in norm_answer for alias in ("direct answer", "answer summary", "summary", "answer")
        ) or len(answer.strip()) > 0
    return False


def _evidence_contains_all_expected_facts(case: EvalCase, response: AgentResponse) -> bool:
    if not case.expected_facts or not response.evidence:
        return False
    combined = " ".join(ev.text_excerpt.lower() for ev in response.evidence)
    for fact in case.expected_facts:
        salient = _extract_salient_tokens([fact])
        if salient:
            if not any(tok.lower() in combined for tok in salient):
                return False
            continue
        words = _salient_fact_words(fact)
        if words and (sum(1 for w in words if w in combined) / len(words)) < 0.6:
            return False
    return True


def _filter_spurious_missing_info_hallucinations(
    case: EvalCase,
    response: AgentResponse,
    hallucinated_claims: list[str],
) -> tuple[list[str], list[str]]:
    """Drop LLM hallucination flags that only restate genuinely missing corpus evidence."""
    if not hallucinated_claims or case.expected_behavior != ExpectedBehavior.ANSWER:
        return list(hallucinated_claims), []
    if _evidence_contains_all_expected_facts(case, response):
        return list(hallucinated_claims), []

    kept: list[str] = []
    dropped: list[str] = []
    for claim in hallucinated_claims:
        if _MISSING_INFO_CLAIM_RE.search(claim):
            dropped.append(claim)
        else:
            kept.append(claim)
    return kept, dropped
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
        batch_size: int = 5,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self._gemini = gemini_client or GeminiClient()
        self._known_sources = set(known_sources or ())
        self._enable_consistency_recheck = enable_consistency_recheck
        self.batch_size = batch_size

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
                norm_answer = _normalize_contract_label(answer)
                missing_fields = [
                    field
                    for field in contract.required_fields
                    if not _has_markdown_field_equivalent(
                        field,
                        raw_lowered=lowered,
                        norm_answer=norm_answer,
                        answer_cited=answer_cited,
                        answer=answer,
                    )
                ]
                if missing_fields:
                    contract_compliant = False
                    issues.append(f"Response missing required contract fields: {missing_fields}.")
                if contract.response_format.lower() == "structured_sections":
                    missing_sections = [
                        sec
                        for sec in contract.response_sections
                        if _normalize_contract_label(sec) not in norm_answer
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

    @staticmethod
    def _build_blinded_payloads(
        case: EvalCase,
        response: AgentResponse,
    ) -> tuple[dict, dict]:
        """Build blinded case and agent output payloads without leaking model identity."""
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
        return eval_case_payload, agent_output_payload

    @staticmethod
    def validate_batch_verdicts(
        batch_verdict: JudgeLLMBatchVerdict,
        expected_ids: list[str],
    ) -> dict[str, JudgeLLMVerdict]:
        """Strictly validate keyed batch verdicts by `case_id` with zero positional matching."""
        if batch_verdict is None or not isinstance(getattr(batch_verdict, "verdicts", None), list):
            raise BatchJudgeValidationError("Malformed batch response: missing verdicts list.")

        expected_set = set(expected_ids)
        matched: dict[str, JudgeLLMVerdict] = {}
        for entry in batch_verdict.verdicts:
            raw_id = getattr(entry, "case_id", None)
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise BatchJudgeValidationError("Malformed batch verdict: empty or non-string case_id.")
            cid = raw_id.strip()
            if cid != raw_id:
                raise BatchJudgeValidationError(f"Malformed case_id formatting: {raw_id!r}.")
            if cid not in expected_set:
                raise BatchJudgeValidationError(f"Unknown case_id '{cid}' returned in batch.")
            if cid in matched:
                raise BatchJudgeValidationError(f"Duplicate case_id '{cid}' returned in batch.")
            matched[cid] = entry.to_llm_verdict()

        missing = [cid for cid in expected_ids if cid not in matched]
        if missing:
            raise BatchJudgeValidationError(f"Missing verdict(s) for case_id(s): {missing}.")
        return matched

    @staticmethod
    def _build_rejected_batch_result(
        case: EvalCase,
        response: AgentResponse,
        det: DeterministicCheckResult,
        reason: str,
    ) -> EvalCaseResult:
        """Mark an invalid-batch case failed honestly without fabricating passing scores."""
        evidence_sources = {ev.source_file for ev in response.evidence}
        expected_sources_retrieved = bool(set(case.source_documents) & evidence_sources)
        if case.expected_behavior == ExpectedBehavior.ANSWER and response.refused:
            has_expected_content = _evidence_contains_expected_facts(case, response)
            if (
                not response.evidence
                or (case.source_documents and not expected_sources_retrieved)
                or (case.expected_facts and not has_expected_content)
            ):
                failure = FailureClassification.RETRIEVAL_MISS
            else:
                failure = FailureClassification.WRONG_REFUSAL
        elif det.invented_citations:
            failure = FailureClassification.HALLUCINATION
        elif not det.contract_compliant and det.refusal_correct and det.citation_valid:
            failure = FailureClassification.CONTRACT_VIOLATION
        else:
            failure = FailureClassification.PROMPT_DEFECT

        reasoning_parts = [f"Batch judge response rejected: {reason}"]
        if det.issues:
            reasoning_parts.append("Deterministic issues: " + "; ".join(det.issues))
        return EvalCaseResult(
            case_id=case.case_id,
            question=case.question,
            category=case.category,
            expected_behavior=case.expected_behavior,
            agent_answer=response.answer,
            refused=response.refused,
            retrieved_evidence=response.evidence,
            cited_sources=det.cited_sources,
            grounding_score=0.0,
            correctness_score=0.0,
            refusal_correct=det.refusal_correct,
            citation_valid=det.citation_valid,
            contract_compliant=det.contract_compliant,
            omitted_facts=[],
            hallucinated_claims=[],
            deterministic_issues=det.issues,
            disagreements=[f"Batch judge response rejected: {reason}"],
            passed=False,
            failure_classification=failure,
            judge_reasoning=" | ".join(reasoning_parts),
        )

    def _finalize_case_judgment(
        self,
        case: EvalCase,
        response: AgentResponse,
        det: DeterministicCheckResult,
        llm_verdict: JudgeLLMVerdict,
        *,
        eval_case_payload: dict,
        agent_output_payload: dict,
    ) -> EvalCaseResult:
        """Apply optional consistency recheck and deterministic-LLM verdict fusion."""
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
            with telemetry_stage("hybrid_judge"):
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
        eval_case_payload, agent_output_payload = self._build_blinded_payloads(case, response)
        prompt = JUDGE_PROMPT.format(
            eval_case_json=json.dumps(eval_case_payload, indent=2),
            agent_output_json=json.dumps(agent_output_payload, indent=2),
            deterministic_notes=det.model_dump_json(indent=2),
        )
        with telemetry_stage("hybrid_judge"):
            llm_verdict = self._gemini.generate_structured(prompt, JudgeLLMVerdict, temperature=0.0)
        return self._finalize_case_judgment(
            case,
            response,
            det,
            llm_verdict,
            eval_case_payload=eval_case_payload,
            agent_output_payload=agent_output_payload,
        )

    def evaluate_batch(
        self,
        items: list[tuple[EvalCase, AgentResponse]],
        contract: OutputContract,
        *,
        known_sources: set[str] | None = None,
        batch_size: int | None = None,
    ) -> list[EvalCaseResult]:
        """Grade all cases via independent deterministic checks + batched blinded Gemini calls."""
        if not items:
            return []
        eff_batch_size = batch_size if batch_size is not None else self.batch_size
        if eff_batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        prepared: list[tuple[EvalCase, AgentResponse, DeterministicCheckResult, dict, dict]] = []
        for case, response in items:
            det = self.run_deterministic_checks(
                case,
                response,
                contract,
                known_sources=known_sources,
            )
            case_payload, output_payload = self._build_blinded_payloads(case, response)
            prepared.append((case, response, det, case_payload, output_payload))

        results: list[EvalCaseResult] = []
        for start in range(0, len(prepared), eff_batch_size):
            chunk = prepared[start : start + eff_batch_size]
            expected_ids = [c.case_id for c, _r, _d, _cp, _op in chunk]
            if len(set(expected_ids)) != len(expected_ids):
                raise BatchJudgeValidationError("Duplicate case_id found inside evaluation batch input.")

            batch_payload = [
                {
                    "case_id": case.case_id,
                    "eval_case": case_payload,
                    "agent_output": output_payload,
                    "deterministic_notes": det.model_dump(mode="json"),
                }
                for case, _resp, det, case_payload, output_payload in chunk
            ]
            prompt = BATCH_JUDGE_PROMPT.format(
                batch_count=len(chunk),
                expected_case_ids_json=json.dumps(expected_ids),
                batch_items_json=json.dumps(batch_payload, indent=2),
            )
            try:
                with telemetry_stage("hybrid_judge"):
                    raw_batch = self._gemini.generate_structured(
                        prompt, JudgeLLMBatchVerdict, temperature=0.0
                    )
                verdicts_by_id = self.validate_batch_verdicts(raw_batch, expected_ids)
            except (
                BatchJudgeValidationError,
                ValidationError,
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:
                reason = str(exc) or type(exc).__name__
                for case, response, det, _cp, _op in chunk:
                    results.append(self._build_rejected_batch_result(case, response, det, reason))
                continue

            for case, response, det, case_payload, output_payload in chunk:
                llm_verdict = verdicts_by_id[case.case_id]
                results.append(
                    self._finalize_case_judgment(
                        case,
                        response,
                        det,
                        llm_verdict,
                        eval_case_payload=case_payload,
                        agent_output_payload=output_payload,
                    )
                )

        return results

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

        is_eval_case_defect = False
        if (
            case.expected_behavior == ExpectedBehavior.REFUSE
            and not response.refused
            and (
                llm.eval_case_defect
                or llm.suggested_failure_classification == FailureClassification.EVAL_CASE_DEFECT
                or (
                    llm.grounded
                    and grounding_score >= 0.85
                    and llm.correctness_score >= 0.85
                    and llm.addresses_question
                    and det.citation_valid
                    and not det.invented_citations
                    and not llm.hallucinated_claims
                    and _evidence_directly_answers_question(case, response)
                )
            )
            and det.citation_valid
            and llm.grounded
            and grounding_score >= 0.7
            and not det.invented_citations
            and not llm.hallucinated_claims
        ):
            is_eval_case_defect = True
            disagreements.append(
                "Evaluation-case defect detected: retrieved corpus evidence directly answers the "
                "question despite expected_behavior='refuse'."
            )
        elif case.expected_behavior == ExpectedBehavior.REFUSE and not response.refused and llm.grounded:
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
        has_expected_content = _evidence_contains_expected_facts(case, response)
        effective_hallucinations, dropped_gap_hallucinations = (
            _filter_spurious_missing_info_hallucinations(case, response, llm.hallucinated_claims)
        )
        if dropped_gap_hallucinations:
            disagreements.append(
                "LLM flagged honest missing-evidence statement as hallucination despite absent retrieved facts."
            )

        # Determine final pass/fail and canonical failure classification
        failure = FailureClassification.NONE
        if is_eval_case_defect:
            failure = FailureClassification.EVAL_CASE_DEFECT
        elif case.expected_behavior == ExpectedBehavior.ANSWER and response.refused:
            if (
                not response.evidence
                or (case.source_documents and not expected_sources_retrieved)
                or llm.suggested_failure_classification == FailureClassification.RETRIEVAL_MISS
                or (case.expected_facts and not has_expected_content)
            ):
                failure = FailureClassification.RETRIEVAL_MISS
            else:
                failure = FailureClassification.WRONG_REFUSAL
        elif case.expected_behavior == ExpectedBehavior.REFUSE and not response.refused:
            if effective_hallucinations or not llm.grounded or det.invented_citations:
                failure = FailureClassification.HALLUCINATION
            else:
                failure = FailureClassification.PROMPT_DEFECT
        elif case.expected_behavior == ExpectedBehavior.REFUSE and response.refused:
            if effective_hallucinations or not llm.grounded or det.invented_citations:
                failure = FailureClassification.HALLUCINATION
            elif not llm.refusal_appropriate:
                failure = FailureClassification.PROMPT_DEFECT
            elif not det.contract_compliant:
                failure = FailureClassification.CONTRACT_VIOLATION
        else:
            # Answerable and answered
            if det.invented_citations or effective_hallucinations:
                failure = FailureClassification.HALLUCINATION
            elif (
                llm.eval_case_defect
                or llm.suggested_failure_classification == FailureClassification.EVAL_CASE_DEFECT
            ):
                is_eval_case_defect = True
                failure = FailureClassification.EVAL_CASE_DEFECT
            elif (
                (case.source_documents and not expected_sources_retrieved)
                or llm.suggested_failure_classification == FailureClassification.RETRIEVAL_MISS
                or (dropped_gap_hallucinations and not _evidence_contains_all_expected_facts(case, response))
            ) and correctness_score < 0.7:
                failure = FailureClassification.RETRIEVAL_MISS
            elif dropped_gap_hallucinations and not _evidence_contains_all_expected_facts(case, response):
                failure = FailureClassification.RETRIEVAL_MISS
            elif not llm.grounded or llm.unsupported_verbosity or grounding_score < 0.7:
                failure = FailureClassification.HALLUCINATION
            elif not det.citation_valid or not llm.addresses_question or correctness_score < 0.7:
                failure = FailureClassification.PROMPT_DEFECT
            elif not det.contract_compliant:
                failure = FailureClassification.CONTRACT_VIOLATION

        passed = (
            failure == FailureClassification.NONE
            and det.refusal_correct
            and det.citation_valid
            and det.contract_compliant
            and not det.issues
        )
        if not passed and failure == FailureClassification.NONE:
            failure = (
                FailureClassification.CONTRACT_VIOLATION
                if not det.contract_compliant
                else FailureClassification.PROMPT_DEFECT
            )

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
            hallucinated_claims=effective_hallucinations,
            deterministic_issues=det.issues,
            disagreements=disagreements,
            passed=passed,
            failure_classification=failure,
            eval_case_defect=is_eval_case_defect,
            judge_reasoning=" | ".join(reasoning_parts),
        )

