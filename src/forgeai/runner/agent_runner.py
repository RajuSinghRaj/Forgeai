"""Domain-independent runtime that executes any `DeclarativeAgentDefinition`."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.llm.gemini_client import GeminiClient
from forgeai.llm.telemetry import telemetry_stage
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import RetrievedEvidence
from forgeai.tools.registry import ToolRegistry, build_default_registry, count_csv_rows_contained

_LOGGER = logging.getLogger(__name__)
_IDENTIFIER_TOKEN_RE = re.compile(r"\b[A-Za-z]{1,8}[-_]\d{1,6}[A-Za-z0-9]*\b")
_QUERY_WORD_STOPWORDS = {
    "about", "across", "after", "all", "and", "are", "can", "complete", "could",
    "details", "does", "entire", "every", "for", "from", "full", "have", "how",
    "list", "many", "more", "most", "much", "show", "some", "tell", "than",
    "that", "the", "their", "them", "these", "they", "this", "versus", "what",
    "when", "where", "which", "who", "will", "with", "would", "your",
}
_AGGREGATE_TABLE_CUES = {
    "across", "all", "complete", "compare", "entire", "every", "full", "list",
    "summary", "table", "versus", "vs",
}
_EXPLICIT_REFUSAL_START_RE = re.compile(
    r"^(?:i cannot (?:answer|assist|provide|fulfill|disclose|comply)|"
    r"i am unable to (?:answer|assist|provide|fulfill|disclose)|"
    r"i (?:must|have to) refuse|"
    r"cannot (?:assist|answer|provide|fulfill) (?:this|your|that) request|"
    r"this request (?:is outside|cannot be answered|falls outside))",
    re.IGNORECASE,
)
_UNDOCUMENTED_LEAD_RE = re.compile(
    r"^(?:the\s+)?(?:supplied|provided|available|uploaded|documented)\s+"
    r"(?:documents?|files?|policies?|records?|corpus|materials?|sources?)\s+"
    r"(?:do|does)\s+not\s+(?:specify|document|state|contain|cover|mention|include)|"
    r"^(?:this|that|the requested)\s+(?:information|topic|request|question|policy|service|charge|fee|rule)\s+"
    r"is\s+(?:not\s+(?:documented|specified|covered|provided|available|included)|outside\s+(?:my|the documented)\s+scope)|"
    r"^no\s+(?:documented\s+)?(?:policy|information|rule|details?|records?)\s+"
    r"(?:is|are)\s+(?:provided|documented|specified|available)\b",
    re.IGNORECASE,
)


def _extract_identifier_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(_IDENTIFIER_TOKEN_RE.findall(text)))


def _extract_query_terms(text: str) -> list[str]:
    terms = [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _QUERY_WORD_STOPWORDS
    ]
    return list(dict.fromkeys(terms))


def _strip_leading_markdown_label(line: str) -> str:
    cleaned = re.sub(r"^[\s#>*\-•]+", "", line).strip()
    cleaned = re.sub(r"^(?:\*\*)?[a-z][a-z0-9 _-]{1,32}(?:\*\*)?\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"^[\s*`]+", "", cleaned).strip()


def _detect_refusal(answer: str, contract_refusal_template: str) -> bool:
    """Detect genuine refusals regardless of Markdown headings or `answer_summary:` prefixes."""
    text = answer.strip()
    if not text:
        return False
    lowered = text.lower()
    refusal_marker = contract_refusal_template.strip().lower()[:32]
    if refusal_marker and refusal_marker in lowered:
        return True
    if "outside my scope" in lowered or "falls outside documented" in lowered:
        return True

    # Inspect leading substantive lines/clauses cleanly without triggering on trailing caveats.
    candidate_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:8]
    for raw_line in candidate_lines:
        stripped_heading = re.sub(r"^[\s#>*\-•]+", "", raw_line).strip()
        if re.match(r"^(?:\*\*)?(?:details\s*&\s*rules|source\s+references|citations)(?:\*\*)?$", stripped_heading, re.IGNORECASE):
            break
        cleaned = _strip_leading_markdown_label(raw_line)
        if not cleaned:
            continue
        if _EXPLICIT_REFUSAL_START_RE.search(cleaned):
            return True
        if _UNDOCUMENTED_LEAD_RE.search(cleaned):
            return True
        # Also catch an explicit refusal phrase anywhere inside a dedicated Direct Answer / answer_summary line
        if re.search(
            r"\b(?:i must refuse|i cannot assist with|i cannot provide (?:individual|private|another)|"
            r"must be refused|is outside (?:my|the documented) scope|not covered in the provided documents)\b",
            cleaned,
            re.IGNORECASE,
        ):
            return True
    return False


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
        documents_dir: Path | str | None = None,
    ) -> None:
        self.definition = definition
        self.vector_store = vector_store
        self.gemini_client = gemini_client or GeminiClient()
        docs_root = Path(documents_dir) if documents_dir is not None else None
        self.tool_registry = tool_registry or build_default_registry(
            vector_store=vector_store,
            documents_dir=docs_root,
        )
        self.documents_dir: Path | None = docs_root or getattr(self.tool_registry, "documents_dir", None)

    @classmethod
    def from_json_file(
        cls,
        path: Path | str,
        *,
        vector_store: ChromaVectorStore,
        gemini_client: GeminiClient | None = None,
        tool_registry: ToolRegistry | None = None,
        documents_dir: Path | str | None = None,
    ) -> "DeclarativeAgentRunner":
        """Load and validate a serialized `DeclarativeAgentDefinition` JSON file."""
        raw = Path(path).read_text(encoding="utf-8")
        definition = DeclarativeAgentDefinition.model_validate_json(raw)
        return cls(
            definition,
            vector_store=vector_store,
            gemini_client=gemini_client,
            tool_registry=tool_registry,
            documents_dir=documents_dir,
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
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"CSV tool configuration error while querying '{csv_file}': {exc}"
                        ) from exc
                    except (FileNotFoundError, ValueError) as exc:
                        _LOGGER.warning("CSV inspection skipped for '%s': %s", csv_file, exc)
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

    def _execute_csv_page(
        self,
        *,
        file_name: str,
        contains: str | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, str]]:
        """Invoke `inspect_csv_records` safely with optional `offset` pagination support."""
        kwargs: dict[str, object] = {"file_name": file_name, "limit": limit}
        if contains is not None:
            kwargs["contains"] = contains
        tool_props = (
            self.tool_registry.get("inspect_csv_records").parameters_schema.get("properties", {})
            if self.tool_registry.has("inspect_csv_records")
            else {}
        )
        if offset > 0 and "offset" in tool_props:
            kwargs["offset"] = offset
        return self.tool_registry.execute("inspect_csv_records", **kwargs)

    def _gather_relevant_csv_rows(
        self,
        question: str,
        semantic_evidence: list[RetrievedEvidence],
        tools_used: list[str],
        *,
        exact_matched_csvs: set[str] | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve relevant CSV rows with bounded pagination and explicit truncation indicators."""
        selected_tool_names = {t.name for t in self.definition.tools}
        if (
            "inspect_csv_records" not in selected_tool_names
            or not self.tool_registry.has("inspect_csv_records")
        ):
            return []

        csv_files = [src for src in self.definition.source_documents if src.lower().endswith(".csv")]
        if not csv_files:
            return []

        query_terms = _extract_query_terms(question)
        q_words_all = set(re.findall(r"[a-z0-9]+", question.lower()))
        is_aggregate_query = bool(q_words_all & _AGGREGATE_TABLE_CUES)

        rows_by_file: dict[str, int] = {}
        for ev in semantic_evidence:
            if ev.source_file.lower().endswith(".csv"):
                rows_by_file[ev.source_file] = rows_by_file.get(ev.source_file, 0) + 1

        tool_props = self.tool_registry.get("inspect_csv_records").parameters_schema.get("properties", {})
        supports_offset = "offset" in tool_props
        page_size = 15
        max_exhaustive_rows = 30
        targeted_limit = 8

        extra: list[RetrievedEvidence] = []
        seen_keys: set[tuple[str, str]] = set()
        for csv_file in csv_files:
            if exact_matched_csvs and csv_file in exact_matched_csvs and not is_aggregate_query:
                continue
            stem_terms = set(re.findall(r"[a-z0-9]+", Path(csv_file).stem.lower()))
            stem_match = bool(stem_terms & set(query_terms))
            hit_count = rows_by_file.get(csv_file, 0)
            if not hit_count and not stem_match:
                continue

            matched_rows: list[dict[str, str]] = []
            truncated = False
            truncated_terms: list[str] = []

            # 1. Probe content keywords against CSV cells cleanly
            for term in query_terms[:6]:
                try:
                    found = self._execute_csv_page(
                        file_name=csv_file,
                        contains=term,
                        limit=targeted_limit + 1,
                        offset=0,
                    )
                except (RuntimeError, FileNotFoundError, ValueError):
                    found = []
                if found:
                    if "inspect_csv_records" not in tools_used:
                        tools_used.append("inspect_csv_records")
                    if len(found) > targeted_limit:
                        found = found[:targeted_limit]
                        if not is_aggregate_query:
                            truncated = True
                            truncated_terms.append(term)
                    matched_rows.extend(found)

            # 2. For multi-row or exhaustive queries on a relevant CSV table, paginate within safe bounds
            table_rows: list[dict[str, str]] = []
            if is_aggregate_query:
                offset = 0
                budget = max_exhaustive_rows
                while budget > 0:
                    step = min(page_size, budget)
                    try:
                        batch = self._execute_csv_page(
                            file_name=csv_file,
                            contains=None,
                            limit=step + 1,
                            offset=offset,
                        )
                    except (RuntimeError, FileNotFoundError, ValueError):
                        break
                    if not batch:
                        break
                    if "inspect_csv_records" not in tools_used:
                        tools_used.append("inspect_csv_records")
                    has_more = len(batch) > step
                    kept = batch[:step]
                    table_rows.extend(kept)
                    budget -= len(kept)
                    offset += len(kept)
                    if not has_more:
                        break
                    if budget <= 0 or not supports_offset:
                        truncated = True
                        break
            elif hit_count >= 2:
                try:
                    batch = self._execute_csv_page(
                        file_name=csv_file,
                        contains=None,
                        limit=page_size + 1,
                        offset=0,
                    )
                except (RuntimeError, FileNotFoundError, ValueError):
                    batch = []
                if batch:
                    if "inspect_csv_records" not in tools_used:
                        tools_used.append("inspect_csv_records")
                    if len(batch) > page_size:
                        truncated = True
                        table_rows.extend(batch[:page_size])
                    else:
                        table_rows.extend(batch)

            matched_rows.extend(table_rows)

            file_rows_added = 0
            for row in matched_rows:
                row_num = row.get("_row", "?")
                unit_label = f"row_{row_num}"
                key = (csv_file, unit_label)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                file_rows_added += 1
                parts = [f"{k}: {v}" for k, v in row.items() if k != "_row"]
                extra.append(
                    RetrievedEvidence(
                        chunk_id=f"csv_exact:{csv_file}:{unit_label}",
                        doc_id=csv_file,
                        source_file=csv_file,
                        unit_label=unit_label,
                        text_excerpt=f"Row {row_num}: " + " | ".join(parts),
                        distance=0.0,
                    )
                )

            if truncated and file_rows_added > 0:
                total_rows: int | None = None
                matched_count: int | None = None
                if self.documents_dir is not None:
                    try:
                        total_rows, matched_count = count_csv_rows_contained(
                            self.documents_dir,
                            csv_file,
                            contains=(truncated_terms[0] if (len(truncated_terms) == 1 and not table_rows) else None),
                        )
                    except (OSError, ValueError):
                        total_rows, matched_count = None, None

                if total_rows is not None and matched_count is not None and matched_count != total_rows:
                    count_detail = (
                        f"Retrieved {file_rows_added} matching row(s) out of {matched_count} matches "
                        f"({total_rows} total rows in '{csv_file}')."
                    )
                elif total_rows is not None:
                    remaining = max(0, total_rows - file_rows_added)
                    count_detail = (
                        f"Retrieved {file_rows_added} of {total_rows} total rows in '{csv_file}' "
                        f"({remaining} additional rows were NOT loaded)."
                    )
                else:
                    count_detail = (
                        f"Retrieved {file_rows_added} row(s) from '{csv_file}', which reached the "
                        f"retrieval cap while additional rows exist."
                    )

                extra.append(
                    RetrievedEvidence(
                        chunk_id=f"csv_notice:{csv_file}:truncated",
                        doc_id=csv_file,
                        source_file=csv_file,
                        unit_label="truncation_notice",
                        text_excerpt=(
                            f"[TRUNCATED CSV TABLE: {count_detail} "
                            f"Explicitly disclose this partial scope and DO NOT present this subset as the complete table.]"
                        ),
                        distance=0.0,
                    )
                )
        return extra

    @staticmethod
    def _parse_chunk_order(ev: RetrievedEvidence) -> tuple[str, str, int]:
        marker = f"{ev.doc_id}:{ev.unit_label}:c"
        if ev.chunk_id.startswith(marker):
            suffix = ev.chunk_id[len(marker) :]
            if suffix.isdigit():
                return (ev.doc_id, ev.unit_label, int(suffix))
        return (ev.doc_id, ev.unit_label, 10**9)

    @classmethod
    def _order_with_neighbors(
        cls,
        base_evidence: list[RetrievedEvidence],
        neighbors: list[RetrievedEvidence],
        *,
        max_items: int,
    ) -> list[RetrievedEvidence]:
        """Insert adjacent sibling chunks next to matching units without exceeding `max_items`."""
        if not neighbors:
            return base_evidence[:max_items]
        by_unit: dict[tuple[str, str], list[RetrievedEvidence]] = {}
        for ev in neighbors:
            by_unit.setdefault((ev.doc_id, ev.unit_label), []).append(ev)

        ordered: list[RetrievedEvidence] = []
        seen_ids: set[str] = set()
        for hit in base_evidence:
            unit_key = (hit.doc_id, hit.unit_label)
            unit_items = [hit] + by_unit.pop(unit_key, [])
            unit_items.sort(key=cls._parse_chunk_order)
            for item in unit_items:
                if item.chunk_id in seen_ids:
                    continue
                seen_ids.add(item.chunk_id)
                ordered.append(item)
                if len(ordered) >= max_items:
                    return ordered
        return ordered[:max_items]

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
        exact_csvs = {
            ev.source_file for ev in exact_evidence if ev.source_file.lower().endswith(".csv")
        }
        csv_supplement = self._gather_relevant_csv_rows(
            question,
            semantic_evidence,
            tools_used,
            exact_matched_csvs=exact_csvs,
        )

        truncated_csv_files = {
            ev.source_file for ev in csv_supplement if ev.unit_label == "truncation_notice"
        }

        merged: list[RetrievedEvidence] = []
        seen_chunk_ids: set[str] = set()
        seen_csv_rows: set[tuple[str, str]] = set()
        for ev in exact_evidence + csv_supplement:
            if ev.chunk_id in seen_chunk_ids:
                continue
            csv_key = (ev.source_file, ev.unit_label)
            if ev.unit_label.startswith("row_") and csv_key in seen_csv_rows:
                continue
            seen_chunk_ids.add(ev.chunk_id)
            if ev.unit_label.startswith("row_"):
                seen_csv_rows.add(csv_key)
            merged.append(ev)
        for ev in semantic_evidence:
            if ev.source_file in truncated_csv_files and ev.unit_label.startswith("row_"):
                continue
            if ev.chunk_id in seen_chunk_ids:
                continue
            csv_key = (ev.source_file, ev.unit_label)
            if ev.unit_label.startswith("row_") and csv_key in seen_csv_rows:
                continue
            seen_chunk_ids.add(ev.chunk_id)
            if ev.unit_label.startswith("row_"):
                seen_csv_rows.add(csv_key)
            merged.append(ev)

        base_limit = max(self.definition.retrieval.top_k, len(exact_evidence)) + 2 + len(csv_supplement)
        base_evidence = merged[:base_limit]

        # Expand retrieved non-CSV chunks with adjacent sibling chunks (c{i-1}, c{i+1}, c0)
        # so multi-part clauses split across chunk boundaries remain intact in context.
        neighbors: list[RetrievedEvidence] = []
        if hasattr(self.vector_store, "get_adjacent_chunks") and base_evidence:
            neighbors = [
                ev
                for ev in self.vector_store.get_adjacent_chunks(base_evidence, window=1, max_extra=4)
                if ev.chunk_id not in seen_chunk_ids
            ]

        evidence = self._order_with_neighbors(base_evidence, neighbors, max_items=base_limit + 4)

        context_lines = [
            f"[{idx}] {ev.source_file} ({ev.unit_label}): {ev.text_excerpt}"
            for idx, ev in enumerate(evidence, start=1)
        ]
        context_block = "\n\n".join(context_lines) if context_lines else "(No matching evidence retrieved.)"

        contract = self.definition.output_contract
        is_json_contract = contract.response_format.strip().lower() == "json"
        required_fields_line = (
            f"- Required JSON keys: {', '.join(contract.required_fields) or 'None'}\n"
            if is_json_contract
            else (
                f"- Required elements (use matching headings or labeled Markdown sections): "
                f"{', '.join(contract.required_fields) or 'None'}\n"
            )
        )
        prompt = (
            f"User Question: {question}\n\n"
            f"Retrieved Evidence:\n{context_block}\n\n"
            f"Output Rules:\n"
            f"- Response format: {contract.response_format}\n"
            f"- Require citations: {contract.require_citations} ({contract.citation_format})\n"
            f"- Response sections: {', '.join(contract.response_sections) or 'None'}\n"
            f"{required_fields_line}"
            f"- Max response words: {contract.max_response_words or 'None'}\n"
            f"- Formatting constraints: {'; '.join(contract.formatting_constraints) or 'None'}\n"
            f"- Must-refuse topics: {', '.join(self.definition.must_refuse_topics) or 'None'}\n"
            f"- Grounding rule: Cite ONLY directly supported facts from Retrieved Evidence. "
            f"Do NOT extrapolate or infer unstated interactions between separate documents; "
            f"if a question depends on an undocumented rule or cross-document assumption, state "
            f"the gap or refuse instead of guessing.\n"
            f"- Completeness rule: If Retrieved Evidence includes a CSV `truncation_notice`, "
            f"explicitly state how many rows were retrieved out of the total rows available and "
            f"NEVER claim the partial subset is the complete table.\n"
            f"- Partial-answer rule: Answer directly supported parts accurately with citations; "
            f"if a request is fully out of scope or unsupported by the retrieved evidence, refuse clearly using: "
            f"{contract.refusal_message_template}\n"
        )
        with telemetry_stage("agent_runner"):
            answer = self.gemini_client.generate_text(
                prompt,
                system_instruction=self.definition.system_prompt,
                temperature=0.1,
            )
        citations = [f"{ev.source_file} ({ev.unit_label})" for ev in evidence]
        refused = _detect_refusal(answer, contract.refusal_message_template)
        return AgentResponse(
            answer=answer,
            refused=refused,
            evidence=evidence,
            citations=citations,
            tools_used=tools_used,
        )

