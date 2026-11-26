"""Domain-independent runtime that executes any `DeclarativeAgentDefinition`."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.llm.gemini_client import GeminiClient
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import RetrievedEvidence
from forgeai.tools.registry import ToolRegistry, build_default_registry

_IDENTIFIER_TOKEN_RE = re.compile(r"\b[A-Za-z]{1,8}[-_]\d{1,6}[A-Za-z0-9]*\b")


def _extract_identifier_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(_IDENTIFIER_TOKEN_RE.findall(text)))


class AgentResponse(BaseModel):
    """Structured response produced by executing a declarative agent."""

    answer: str
    refused: bool = False
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class DeclarativeAgentRunner:
    """Generic runner that loads a declarative agent spec and answers queries."""

    def __init__(
        self,
        definition: DeclarativeAgentDefinition,
        *,
        vector_store: ChromaVectorStore,
        gemini_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.definition = definition
        self.vector_store = vector_store
        self.gemini_client = gemini_client or GeminiClient()
        self.tool_registry = tool_registry or build_default_registry(vector_store=vector_store)

    @classmethod
    def from_json_file(
        cls,
        path: Path | str,
        *,
        vector_store: ChromaVectorStore,
        gemini_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> "DeclarativeAgentRunner":
        """Load and validate a serialized `DeclarativeAgentDefinition` JSON file."""
        raw = Path(path).read_text(encoding="utf-8")
        definition = DeclarativeAgentDefinition.model_validate_json(raw)
        return cls(
            definition,
            vector_store=vector_store,
            gemini_client=gemini_client,
            tool_registry=tool_registry,
        )

    def _gather_exact_identifier_evidence(
        self,
        tokens: list[str],
        tools_used: list[str],
    ) -> list[RetrievedEvidence]:
        """Collect exact identifier matches via selected CSV tool and vector store tokens."""
        if not tokens:
            return []

        exact: list[RetrievedEvidence] = []
        seen_keys: set[tuple[str, str]] = set()
        selected_tool_names = {t.name for t in self.definition.tools}

        if (
            "inspect_csv_records" in selected_tool_names
            and self.tool_registry.has("inspect_csv_records")
        ):
            csv_files = [src for src in self.definition.source_documents if src.lower().endswith(".csv")]
            for csv_file in csv_files:
                for tok in tokens:
                    try:
                        rows = self.tool_registry.execute(
                            "inspect_csv_records",
                            file_name=csv_file,
                            contains=tok,
                            limit=5,
                        )
                    except (RuntimeError, FileNotFoundError, ValueError):
                        rows = []
                    if rows and "inspect_csv_records" not in tools_used:
                        tools_used.append("inspect_csv_records")
                    for row in rows:
                        row_num = row.get("_row", "?")
                        unit_label = f"row_{row_num}"
                        key = (csv_file, unit_label)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        parts = [f"{k}: {v}" for k, v in row.items() if k != "_row"]
                        exact.append(
                            RetrievedEvidence(
                                chunk_id=f"csv_exact:{csv_file}:{unit_label}",
                                doc_id=csv_file,
                                source_file=csv_file,
                                unit_label=unit_label,
                                text_excerpt=f"Row {row_num}: " + " | ".join(parts),
                                distance=0.0,
                            )
                        )

        if hasattr(self.vector_store, "query_by_tokens"):
            for ev in self.vector_store.query_by_tokens(tokens, limit_per_token=3):
                key = (ev.source_file, ev.unit_label)
                if key not in seen_keys:
                    seen_keys.add(key)
                    exact.append(ev)

        return exact

    def run(self, question: str) -> AgentResponse:
        """Retrieve relevant evidence and answer (or refuse) per the output contract."""
        tools_used = ["retrieve_documents"]
        tokens = _extract_identifier_tokens(question)
        exact_evidence = self._gather_exact_identifier_evidence(tokens, tools_used)

        where = self.definition.retrieval.metadata_filters or None
        semantic_evidence = self.vector_store.query(
            question,
            top_k=self.definition.retrieval.top_k,
            where=where,
        )

        merged: list[RetrievedEvidence] = []
        seen_keys: set[tuple[str, str]] = set()
        for ev in exact_evidence + semantic_evidence:
            key = (ev.source_file, ev.unit_label)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(ev)
        evidence = merged[: max(self.definition.retrieval.top_k, len(exact_evidence)) + 2]

        context_lines = [
            f"[{idx}] {ev.source_file} ({ev.unit_label}): {ev.text_excerpt}"
            for idx, ev in enumerate(evidence, start=1)
        ]
        context_block = "\n\n".join(context_lines) if context_lines else "(No matching evidence retrieved.)"

        contract = self.definition.output_contract
        prompt = (
            f"User Question: {question}\n\n"
            f"Retrieved Evidence:\n{context_block}\n\n"
            f"Output Rules:\n"
            f"- Response format: {contract.response_format}\n"
            f"- Require citations: {contract.require_citations} ({contract.citation_format})\n"
            f"- Response sections: {', '.join(contract.response_sections) or 'None'}\n"
            f"- Required fields: {', '.join(contract.required_fields) or 'None'}\n"
            f"- Max response words: {contract.max_response_words or 'None'}\n"
            f"- Formatting constraints: {'; '.join(contract.formatting_constraints) or 'None'}\n"
            f"- Must-refuse topics: {', '.join(self.definition.must_refuse_topics) or 'None'}\n"
            f"- If out of scope or unsupported by the retrieved evidence, respond with: "
            f"{contract.refusal_message_template}\n"
        )
        answer = self.gemini_client.generate_text(
            prompt,
            system_instruction=self.definition.system_prompt,
            temperature=0.1,
        )
        citations = [f"{ev.source_file} ({ev.unit_label})" for ev in evidence]
        refusal_marker = contract.refusal_message_template.lower()[:32]
        lowered = answer.lower()
        refused = (
            refusal_marker in lowered
            or lowered.startswith("i cannot answer")
            or lowered.startswith("i am unable to answer")
            or "outside my scope" in lowered
            or "not covered in the provided documents" in lowered
        )
        return AgentResponse(
            answer=answer,
            refused=refused,
            evidence=evidence,
            citations=citations,
            tools_used=tools_used,
        )

