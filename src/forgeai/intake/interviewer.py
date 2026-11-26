"""LLM-powered Intake Interviewer producing up to five clarifying questions
and synthesizing validated `StructuredRequirements`.
"""

from __future__ import annotations

from collections import defaultdict

from forgeai.llm.gemini_client import GeminiClient
from forgeai.prompts.templates import INTAKE_INTERVIEWER_PROMPT, REQUIREMENTS_SYNTHESIS_PROMPT
from forgeai.retrieval.loaders import DocumentUnit
from forgeai.schemas.intake import (
    BriefInput,
    ClarifyingAnswer,
    ClarifyingQuestion,
    ClarifyingQuestionBatch,
    DocumentInput,
    StructuredRequirements,
)


def summarize_documents(
    documents: list[DocumentInput],
    units: list[DocumentUnit],
    *,
    max_units_per_doc: int = 2,
    max_preview_chars: int = 240,
) -> str:
    """Create a concise, domain-independent summary of ingested corpus documents."""
    if not documents:
        return "(No supported documents found in folder.)"

    units_by_doc: dict[str, list[DocumentUnit]] = defaultdict(list)
    for unit in units:
        units_by_doc[unit.doc_id].append(unit)

    lines: list[str] = []
    for doc in documents:
        lines.append(
            f"- {doc.file_name} (type={doc.file_type}, units={doc.unit_count}, bytes={doc.size_bytes})"
        )
        for sample in units_by_doc.get(doc.doc_id, [])[:max_units_per_doc]:
            excerpt = " ".join(sample.text.split())[:max_preview_chars]
            lines.append(f"    [{sample.unit_label}] {excerpt}")
    return "\n".join(lines)


class IntakeInterviewer:
    """Generates bounded clarifying questions and distills structured requirements."""

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        *,
        max_questions: int = 5,
    ) -> None:
        if not 1 <= max_questions <= 5:
            raise ValueError("max_questions must be between 1 and 5.")
        self._gemini = gemini_client or GeminiClient()
        self.max_questions = max_questions

    def generate_questions(
        self,
        brief: BriefInput,
        documents: list[DocumentInput],
        units: list[DocumentUnit],
    ) -> list[ClarifyingQuestion]:
        """Generate up to `max_questions` (max 5) high-leverage clarifying questions."""
        doc_summary = summarize_documents(documents, units)
        prompt = INTAKE_INTERVIEWER_PROMPT.format(
            max_questions=self.max_questions,
            brief_text=brief.brief_text,
            document_summary=doc_summary,
        )
        batch = self._gemini.generate_structured(
            prompt,
            ClarifyingQuestionBatch,
            temperature=0.2,
        )
        seen_questions: set[str] = set()
        unique: list[ClarifyingQuestion] = []
        for idx, item in enumerate(batch.questions, start=1):
            norm = item.question.strip().lower()
            if not norm or norm in seen_questions:
                continue
            seen_questions.add(norm)
            qid = item.question_id.strip() or f"q{idx}"
            unique.append(
                ClarifyingQuestion(
                    question_id=qid,
                    question=item.question.strip(),
                    rationale=item.rationale.strip(),
                    target_field=item.target_field.strip(),
                )
            )
            if len(unique) >= self.max_questions:
                break
        return unique

    def synthesize_requirements(
        self,
        brief: BriefInput,
        documents: list[DocumentInput],
        units: list[DocumentUnit],
        *,
        clarifying_questions: list[ClarifyingQuestion] | None = None,
        clarifying_answers: list[ClarifyingAnswer] | dict[str, str] | None = None,
    ) -> StructuredRequirements:
        """Synthesize `StructuredRequirements` from the brief, docs, and full/partial answers."""
        normalized_answers, qa_block = self._normalize_qa(
            clarifying_questions or [],
            clarifying_answers,
        )
        doc_summary = summarize_documents(documents, units)
        prompt = REQUIREMENTS_SYNTHESIS_PROMPT.format(
            brief_text=brief.brief_text,
            clarifying_qa=qa_block,
            document_summary=doc_summary,
        )
        requirements = self._gemini.generate_structured(
            prompt,
            StructuredRequirements,
            temperature=0.1,
        )
        # Preserve the exact user-supplied answers (up to 5) deterministically.
        return requirements.model_copy(update={"clarifying_qa": normalized_answers[:5]})

    @staticmethod
    def _normalize_qa(
        questions: list[ClarifyingQuestion],
        answers: list[ClarifyingAnswer] | dict[str, str] | None,
    ) -> tuple[list[ClarifyingAnswer], str]:
        if answers is None:
            if not questions:
                return [], "(No clarifying questions asked.)"
            lines = [
                f"- [{q.question_id}] Q: {q.question}\n  A: (unanswered — treat cautiously and note assumption)"
                for q in questions[:5]
            ]
            return [], "\n".join(lines)

        if isinstance(answers, list):
            kept: list[ClarifyingAnswer] = []
            lines = []
            for item in answers[:5]:
                ans_text = item.answer.strip()
                if ans_text:
                    kept.append(
                        ClarifyingAnswer(
                            question_id=item.question_id,
                            question=item.question,
                            answer=ans_text,
                        )
                    )
                    lines.append(f"- [{item.question_id}] Q: {item.question}\n  A: {ans_text}")
                else:
                    lines.append(
                        f"- [{item.question_id}] Q: {item.question}\n  A: (skipped / partial — note assumption)"
                    )
            return kept, "\n".join(lines) if lines else "(No clarifying answers provided.)"

        # Mapping of question_id -> raw text answer (supports partial or blank skips).
        kept = []
        lines = []
        question_map = {q.question_id: q.question for q in questions}
        ordered_ids = [q.question_id for q in questions] + [
            qid for qid in answers if qid not in question_map
        ]
        for qid in ordered_ids[:5]:
            q_text = question_map.get(qid, qid)
            raw_ans = (answers.get(qid) or "").strip()
            if raw_ans:
                kept.append(ClarifyingAnswer(question_id=qid, question=q_text, answer=raw_ans))
                lines.append(f"- [{qid}] Q: {q_text}\n  A: {raw_ans}")
            else:
                lines.append(
                    f"- [{qid}] Q: {q_text}\n  A: (skipped by requester — infer cautiously and note assumption)"
                )
        return kept, "\n".join(lines) if lines else "(No clarifying answers provided.)"

