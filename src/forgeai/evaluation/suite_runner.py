"""Execution harness running a generated agent across evaluation cases and judging outputs."""

from __future__ import annotations

from uuid import uuid4

from forgeai.evaluation.judge import HybridJudge
from forgeai.evaluation.metrics import compute_scorecard
from forgeai.llm.gemini_client import GeminiClient
from forgeai.retrieval.chunker import chunk_units
from forgeai.retrieval.embeddings import EmbeddingProvider, LocalSentenceTransformerEmbeddings
from forgeai.retrieval.loaders import DocumentUnit
from forgeai.retrieval.vector_store import ChromaVectorStore
from forgeai.runner.agent_runner import DeclarativeAgentRunner
from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import EvalCase, EvalCaseResult, IterationScorecard


def run_evaluation_suite(
    definition: DeclarativeAgentDefinition,
    eval_cases: list[EvalCase],
    units: list[DocumentUnit],
    *,
    gemini_client: GeminiClient | None = None,
    vector_store: ChromaVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    judge: HybridJudge | None = None,
    iteration: int = 0,
) -> tuple[list[EvalCaseResult], IterationScorecard]:
    """Execute `DeclarativeAgentRunner` for each case and judge all responses."""
    client = gemini_client or GeminiClient()

    if vector_store is not None:
        store = vector_store
    else:
        embedder = embedding_provider or LocalSentenceTransformerEmbeddings(
            definition.retrieval.embedding_model
        )
        store = ChromaVectorStore(
            collection_name=f"eval_{uuid4().hex[:8]}",
            persist_dir=None,
            embedding_provider=embedder,
        )
        chunks = chunk_units(units, definition.chunking)
        if chunks:
            store.index_chunks(chunks)

    runner = DeclarativeAgentRunner(
        definition,
        vector_store=store,
        gemini_client=client,
    )
    known_sources = set(definition.source_documents) | {u.source_file for u in units}
    active_judge = judge or HybridJudge(gemini_client=client, known_sources=known_sources)

    results: list[EvalCaseResult] = []
    for case in eval_cases:
        response = runner.run(case.question)
        verdict = active_judge.evaluate_case(
            case,
            response,
            definition.output_contract,
            known_sources=known_sources,
        )
        results.append(verdict)

    scorecard = compute_scorecard(results, iteration=iteration)
    return results, scorecard

