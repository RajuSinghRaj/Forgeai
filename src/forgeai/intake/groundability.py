"""Evidence-aware Groundability Guard checking requirements against corpus chunks."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from forgeai.intake.interviewer import summarize_documents
from forgeai.llm.gemini_client import GeminiClient
from forgeai.prompts.templates import GROUNDABILITY_GUARD_PROMPT
from forgeai.retrieval.chunker import chunk_units
from forgeai.retrieval.embeddings import EmbeddingProvider, LocalSentenceTransformerEmbeddings
from forgeai.retrieval.loaders import DocumentUnit
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.schemas.agent import ChunkingConfig
from forgeai.schemas.evaluation import RetrievedEvidence
from forgeai.schemas.intake import (
    DocumentInput,
    GroundabilityFinding,
    GroundabilityReport,
    GroundabilityVerdict,
    StructuredRequirements,
)


class GroundabilityGuard:
    """Compares structured requirements against retrieved document evidence."""

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        *,
        vector_store: ChromaVectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        chunk_config: ChunkingConfig | None = None,
    ) -> None:
        self._gemini = gemini_client or GeminiClient()
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_config = chunk_config or ChunkingConfig()

    def assess(
        self,
        requirements: StructuredRequirements,
        documents: list[DocumentInput],
        units: list[DocumentUnit],
        *,
        approved_with_gaps: bool = False,
        persist_dir: Path | None = None,
    ) -> GroundabilityReport:
        """Verify corpus coverage for `requirements` and return a validated `GroundabilityReport`."""
        topics_to_check = list(requirements.must_answer_topics) + [
            t for t in requirements.must_refuse_topics if t not in requirements.must_answer_topics
        ]
        evidence_by_topic = self._gather_evidence(
            topics_to_check,
            units,
            persist_dir=persist_dir,
        )

        formatted_evidence = self._format_topic_evidence(topics_to_check, evidence_by_topic)
        doc_summary = summarize_documents(documents, units)
        prompt = GROUNDABILITY_GUARD_PROMPT.format(
            requirements_json=requirements.model_dump_json(indent=2),
            document_summary=doc_summary,
            topic_evidence=formatted_evidence,
        )

        raw_report = self._gemini.generate_structured(
            prompt,
            GroundabilityReport,
            temperature=0.0,
        )
        return self._enforce_invariants(
            raw_report,
            requirements=requirements,
            documents=documents,
            units=units,
            evidence_by_topic=evidence_by_topic,
            approved_with_gaps=approved_with_gaps,
        )

    def _gather_evidence(
        self,
        topics: list[str],
        units: list[DocumentUnit],
        *,
        persist_dir: Path | None = None,
    ) -> dict[str, list[RetrievedEvidence]]:
        if not units:
            return {topic: [] for topic in topics}

        chunks = chunk_units(units, self.chunk_config)
        if not chunks:
            return {topic: [] for topic in topics}

        if self._vector_store is not None:
            if self._vector_store.count() == 0:
                self._vector_store.index_chunks(chunks)
            return {
                topic: self._vector_store.query(topic, top_k=self.top_k)
                for topic in topics
            }

        embedder = self._embedding_provider or LocalSentenceTransformerEmbeddings(
            self._embedding_model
        )
        collection_name = "groundability_check" if persist_dir is not None else f"ground_{uuid4().hex[:8]}"
        store = ChromaVectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_provider=embedder,
        )
        if persist_dir is not None:
            store.reset()
        store.index_chunks(chunks)
        return {topic: store.query(topic, top_k=self.top_k) for topic in topics}

    @staticmethod
    def _format_topic_evidence(
        topics: list[str],
        evidence_by_topic: dict[str, list[RetrievedEvidence]],
    ) -> str:
        sections: list[str] = []
        for topic in topics:
            ev_list = evidence_by_topic.get(topic, [])
            if not ev_list:
                sections.append(f"Topic: {topic}\n  (No retrieved evidence chunks.)")
                continue
            lines = [f"Topic: {topic}"]
            for idx, ev in enumerate(ev_list, start=1):
                dist_str = f"{ev.distance:.4f}" if ev.distance is not None else "n/a"
                excerpt = " ".join(ev.text_excerpt.split())[:350]
                lines.append(
                    f"  [{idx}] source_file={ev.source_file} | unit={ev.unit_label} "
                    f"| distance={dist_str} | excerpt={excerpt}"
                )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    @staticmethod
    def _enforce_invariants(
        report: GroundabilityReport,
        *,
        requirements: StructuredRequirements,
        documents: list[DocumentInput],
        units: list[DocumentUnit],
        evidence_by_topic: dict[str, list[RetrievedEvidence]],
        approved_with_gaps: bool,
    ) -> GroundabilityReport:
        """Deterministically guarantee no topic is dropped or hallucinated as supported."""
        known_sources = {d.file_name for d in documents} | {u.source_file for u in units}
        refuse_norm = {t.strip().lower() for t in requirements.must_refuse_topics}

        findings_by_norm: dict[str, GroundabilityFinding] = {}
        ordered_findings: list[GroundabilityFinding] = []

        for finding in report.findings:
            norm = finding.topic.strip().lower()
            if not norm or norm in findings_by_norm:
                continue

            retrieved = evidence_by_topic.get(finding.topic, [])
            if not retrieved:
                for k, v in evidence_by_topic.items():
                    if k.strip().lower() == norm:
                        retrieved = v
                        break
            retrieved_sources = {ev.source_file for ev in retrieved}

            # Strip any hallucinated source filenames that are not in the corpus.
            valid_sources = [
                src for src in finding.supporting_sources if src in known_sources
            ]

            is_out_of_scope = finding.is_out_of_scope or (norm in refuse_norm)
            verdict = finding.verdict
            gap_desc = finding.gap_description
            evidence_summary = finding.evidence_summary

            # Prevent unsupported in-scope topics from silently passing without real evidence.
            if not is_out_of_scope and verdict != GroundabilityVerdict.UNSUPPORTED:
                if not retrieved_sources or not valid_sources:
                    verdict = GroundabilityVerdict.UNSUPPORTED
                    valid_sources = []
                    gap_desc = (
                        gap_desc
                        or "No verifiable supporting sources from the supplied documents matched this topic."
                    )
                    evidence_summary = (
                        "Flagged as unsupported by Groundability Guard: no verified corpus source "
                        "supports this requirement."
                    )

            cleaned = GroundabilityFinding(
                topic=finding.topic.strip(),
                verdict=verdict,
                is_out_of_scope=is_out_of_scope,
                supporting_sources=list(dict.fromkeys(valid_sources)),
                evidence_summary=evidence_summary,
                gap_description=gap_desc,
            )
            findings_by_norm[norm] = cleaned
            ordered_findings.append(cleaned)

        # Ensure every must_answer_topic is accounted for (never silently dropped).
        for topic in requirements.must_answer_topics:
            norm = topic.strip().lower()
            if norm not in findings_by_norm:
                injected = GroundabilityFinding(
                    topic=topic,
                    verdict=GroundabilityVerdict.UNSUPPORTED,
                    is_out_of_scope=False,
                    supporting_sources=[],
                    evidence_summary="Topic was missing from model groundability findings.",
                    gap_description="No verified evidence was produced for this required topic.",
                )
                findings_by_norm[norm] = injected
                ordered_findings.append(injected)

        ungrounded_topics = [
            f.topic
            for f in ordered_findings
            if not f.is_out_of_scope and f.verdict == GroundabilityVerdict.UNSUPPORTED
        ]
        out_of_scope_topics = [f.topic for f in ordered_findings if f.is_out_of_scope]
        for topic in requirements.must_refuse_topics:
            if topic not in out_of_scope_topics:
                out_of_scope_topics.append(topic)

        in_scope_findings = [f for f in ordered_findings if not f.is_out_of_scope]
        if any(f.verdict == GroundabilityVerdict.UNSUPPORTED for f in in_scope_findings):
            overall_verdict = GroundabilityVerdict.UNSUPPORTED
        elif any(f.verdict == GroundabilityVerdict.PARTIALLY_SUPPORTED for f in in_scope_findings):
            overall_verdict = GroundabilityVerdict.PARTIALLY_SUPPORTED
        else:
            overall_verdict = GroundabilityVerdict.SUPPORTED

        has_unresolved_gaps = len(ungrounded_topics) > 0
        can_proceed = (not has_unresolved_gaps) or approved_with_gaps

        recommendations = list(report.recommendations)
        if has_unresolved_gaps and not approved_with_gaps:
            recommendations.append(
                "Supply documents covering the unsupported topics, narrow the agent's scope, "
                "or explicitly approve proceeding with gaps before Phase 3."
            )

        return GroundabilityReport(
            overall_verdict=overall_verdict,
            can_proceed=can_proceed,
            approval_required=has_unresolved_gaps,
            approved_with_gaps=approved_with_gaps and has_unresolved_gaps,
            findings=ordered_findings,
            ungrounded_topics=ungrounded_topics,
            out_of_scope_topics=out_of_scope_topics,
            recommendations=list(dict.fromkeys(recommendations)),
        )
