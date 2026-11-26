"""Automatic, evidence-grounded evaluation-set generator for ForgeAI Phase 4."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from forgeai.intake.interviewer import summarize_documents
from forgeai.llm.gemini_client import GeminiClient
from forgeai.prompts.templates import EVAL_GENERATOR_PROMPT
from forgeai.retrieval.loaders import DocumentUnit
from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import (
    EvalCase,
    EvalGenerationMetadata,
    EvalSuiteDraft,
    ExpectedBehavior,
)
from forgeai.schemas.intake import DocumentInput, GroundabilityReport, StructuredRequirements


class InsufficientEvalCasesError(ValueError):
    """Raised honestly when valid non-duplicate cases fall short of the minimum."""

    def __init__(self, metadata: EvalGenerationMetadata, message: str | None = None) -> None:
        self.metadata = metadata
        detail = message or (
            f"Generated {metadata.accepted_cases} valid evaluation cases, "
            f"short of required minimum ({metadata.accepted_cases + metadata.shortfall}). "
            f"Rejected: {metadata.rejected_fabricated_sources} fabricated sources, "
            f"{metadata.rejected_ungrounded_answerable} ungrounded answerable, "
            f"{metadata.rejected_duplicates} duplicates."
        )
        super().__init__(detail)


def _normalize_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1}


def _is_near_duplicate(question: str, seen_norm: list[str], seen_tokens: list[set[str]]) -> bool:
    norm = re.sub(r"\W+", " ", question.lower()).strip()
    if norm in seen_norm:
        return True
    tokens = _normalize_tokens(question)
    if not tokens:
        return False
    for existing in seen_tokens:
        union = tokens | existing
        if union and (len(tokens & existing) / len(union)) >= 0.85:
            return True
    return False


def _sample_corpus_passages(
    units: list[DocumentUnit],
    *,
    max_units: int = 30,
    max_chars: int = 450,
) -> str:
    """Sample passages round-robin across documents with exact provenance labels."""
    if not units:
        return "(No document units available in corpus.)"

    by_doc: dict[str, list[DocumentUnit]] = defaultdict(list)
    for unit in units:
        by_doc[unit.source_file].append(unit)

    sampled: list[DocumentUnit] = []
    doc_keys = sorted(by_doc)
    idx = 0
    while len(sampled) < max_units:
        added = False
        for key in doc_keys:
            bucket = by_doc[key]
            if idx < len(bucket):
                sampled.append(bucket[idx])
                added = True
                if len(sampled) >= max_units:
                    break
        if not added:
            break
        idx += 1

    lines = []
    for unit in sampled:
        excerpt = " ".join(unit.text.split())[:max_chars]
        lines.append(f"- [{unit.source_file} | {unit.unit_label}] {excerpt}")
    return "\n".join(lines)


class EvalGenerator:
    """Generates and validates a 20+ case evaluation suite grounded in corpus evidence."""

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        *,
        min_cases: int = 20,
    ) -> None:
        if min_cases < 1:
            raise ValueError("min_cases must be at least 1.")
        self._gemini = gemini_client or GeminiClient()
        self.min_cases = min_cases

    def generate(
        self,
        requirements: StructuredRequirements,
        agent_definition: DeclarativeAgentDefinition,
        documents: list[DocumentInput],
        units: list[DocumentUnit],
        *,
        groundability: GroundabilityReport | None = None,
    ) -> tuple[list[EvalCase], EvalGenerationMetadata]:
        """Generate and strictly validate an evaluation suite against corpus provenance."""
        agent_summary = {
            "agent_id": agent_definition.agent_id,
            "name": agent_definition.name,
            "domain_summary": agent_definition.domain_summary,
            "must_answer_topics": agent_definition.must_answer_topics,
            "must_refuse_topics": agent_definition.must_refuse_topics,
            "approved_gaps": agent_definition.approved_gaps,
            "groundability_limitations": agent_definition.groundability_limitations,
            "output_contract": agent_definition.output_contract.model_dump(),
        }
        if groundability is not None:
            agent_summary["groundability_verdict"] = groundability.overall_verdict.value

        prompt = EVAL_GENERATOR_PROMPT.format(
            min_cases=self.min_cases,
            requirements_json=requirements.model_dump_json(indent=2),
            agent_summary_json=json.dumps(agent_summary, indent=2),
            document_summary=summarize_documents(documents, units),
            corpus_samples=_sample_corpus_passages(units),
        )
        draft = self._gemini.generate_structured(prompt, EvalSuiteDraft, temperature=0.2)

        accepted, metadata = self._validate_and_filter(draft.cases, documents, units)
        if len(accepted) < self.min_cases:
            raise InsufficientEvalCasesError(metadata)
        return accepted, metadata

    def _validate_and_filter(
        self,
        raw_cases: list[EvalCase],
        documents: list[DocumentInput],
        units: list[DocumentUnit],
    ) -> tuple[list[EvalCase], EvalGenerationMetadata]:
        valid_sources = {d.file_name for d in documents} | {u.source_file for u in units}
        sources_with_units = {u.source_file for u in units}

        accepted: list[EvalCase] = []
        seen_norm: list[str] = []
        seen_tokens: list[set[str]] = []
        seen_ids: set[str] = set()

        rejected_fabricated = 0
        rejected_ungrounded = 0
        rejected_duplicates = 0

        for idx, case in enumerate(raw_cases, start=1):
            if _is_near_duplicate(case.question, seen_norm, seen_tokens):
                rejected_duplicates += 1
                continue

            # Reject any case citing source filenames not present in the actual corpus.
            if any(src not in valid_sources for src in case.source_documents):
                rejected_fabricated += 1
                continue

            # Also check evidence_references for fabricated source prefixes (e.g. "fake.pdf | page_1").
            fabricated_ref = False
            for ref in case.evidence_references:
                prefix = ref.split("|", 1)[0].strip()
                if prefix and "." in prefix and prefix not in valid_sources:
                    fabricated_ref = True
                    break
            if fabricated_ref:
                rejected_fabricated += 1
                continue

            if case.expected_behavior == ExpectedBehavior.ANSWER:
                if not case.expected_facts or not case.source_documents:
                    rejected_ungrounded += 1
                    continue
                if not any(src in sources_with_units for src in case.source_documents):
                    rejected_ungrounded += 1
                    continue

            case_id = case.case_id.strip() or f"eval_{idx:03d}"
            while case_id in seen_ids:
                case_id = f"{case_id}_{len(seen_ids) + 1}"
            seen_ids.add(case_id)

            seen_norm.append(re.sub(r"\W+", " ", case.question.lower()).strip())
            seen_tokens.append(_normalize_tokens(case.question))

            cleaned = case.model_copy(
                update={
                    "case_id": case_id,
                    "is_cross_document": len(set(case.source_documents)) >= 2,
                }
            )
            accepted.append(cleaned)

        category_counts = dict(Counter(c.category.value for c in accepted))
        referenced_sources = sorted(
            {src for c in accepted for src in c.source_documents}
        )
        shortfall = max(0, self.min_cases - len(accepted))
        metadata = EvalGenerationMetadata(
            total_raw_cases=len(raw_cases),
            accepted_cases=len(accepted),
            rejected_fabricated_sources=rejected_fabricated,
            rejected_ungrounded_answerable=rejected_ungrounded,
            rejected_duplicates=rejected_duplicates,
            shortfall=shortfall,
            category_counts=category_counts,
            source_documents_referenced=referenced_sources,
        )
        return accepted, metadata

