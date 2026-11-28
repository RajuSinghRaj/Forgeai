"""Schemas for the declarative generated agent definition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from forgeai.schemas.intake import StructuredRequirements


class ChunkingConfig(BaseModel):
    """Chunking strategy used when indexing documents for the generated agent."""

    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    csv_rows_per_unit: int = Field(default=1, ge=1, le=500)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
        return self


class RetrievalConfig(BaseModel):
    """Vector retrieval parameters used by the generic agent runner."""

    collection_name: str = Field(..., min_length=1)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    require_source_citations: bool = True


class ToolSelection(BaseModel):
    """Reference to a registered domain-independent tool configured for the agent."""

    name: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=3)
    config: dict[str, Any] = Field(default_factory=dict)


class OutputContract(BaseModel):
    """Rules governing answer structure, citations, formatting, and refusal behavior."""

    response_format: str = Field(default="markdown", min_length=2)
    require_citations: bool = True
    citation_format: str = Field(default="[source_file | unit]")
    refusal_message_template: str = Field(
        default="I cannot answer this request based on the provided documents and scope."
    )
    max_response_words: int | None = Field(default=350, ge=30, le=2000)
    response_sections: list[str] = Field(default_factory=lambda: ["Answer", "Evidence"])
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    formatting_constraints: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)


class DeclarativeAgentDefinition(BaseModel):
    """Portable, domain-independent specification executed by `DeclarativeAgentRunner`."""

    agent_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=2)
    description: str = Field(default="")
    version: int = Field(default=1, ge=1)
    domain_summary: str = Field(..., min_length=5)
    system_prompt: str = Field(..., min_length=20)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig
    tools: list[ToolSelection] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)
    must_answer_topics: list[str] = Field(default_factory=list)
    must_refuse_topics: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    approved_requirements: StructuredRequirements | None = Field(default=None)
    groundability_limitations: list[str] = Field(default_factory=list)
    approved_gaps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Structured Gemini draft schemas (strictly typed, avoiding open-ended dict properties)


class SystemPromptDraft(BaseModel):
    """Structured LLM draft for the agent identity and system prompt."""

    agent_name: str = Field(..., min_length=2)
    description: str = Field(..., min_length=10)
    domain_summary: str = Field(..., min_length=5)
    system_prompt: str = Field(..., min_length=40)


class MetadataFilterEntry(BaseModel):
    """Single key/value metadata filter proposed for retrieval."""

    field: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class RetrievalStrategyDraft(BaseModel):
    """Structured LLM draft for chunking and retrieval configuration."""

    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    csv_rows_per_unit: int = Field(default=1, ge=1, le=500)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata_filters: list[MetadataFilterEntry] = Field(default_factory=list)
    require_source_citations: bool = True
    rationale: str = Field(default="")

    @model_validator(mode="after")
    def _validate_overlap(self) -> "RetrievalStrategyDraft":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
        return self


class ToolSelectionDraftItem(BaseModel):
    """Single tool chosen from the registered ToolRegistry."""

    name: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=5)
    reason: str = Field(..., min_length=5)


class ToolSelectionDraft(BaseModel):
    """Structured LLM draft for least-privilege tool selection."""

    selected_tools: list[ToolSelectionDraftItem] = Field(default_factory=list)
    least_privilege_notes: str = Field(default="")


class AgentGenerationDraft(BaseModel):
    """Composite structured Gemini output combining all Phase 3 agent components."""

    system_prompt_draft: SystemPromptDraft
    retrieval_strategy_draft: RetrievalStrategyDraft
    tool_selection_draft: ToolSelectionDraft
    output_contract: OutputContract
