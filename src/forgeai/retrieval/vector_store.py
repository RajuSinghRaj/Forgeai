"""Persistent ChromaDB vector store with source-level provenance."""

from __future__ import annotations

from pathlib import Path

from forgeai.retrieval.chunker import DocumentChunk, chunk_units
from forgeai.retrieval.embeddings import EmbeddingProvider, LocalSentenceTransformerEmbeddings
from forgeai.retrieval.loaders import DocumentUnit, rebatch_csv_units
from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import RetrievedEvidence


class ChromaVectorStore:
    """ChromaDB collection wrapper storing document chunks and traceability metadata."""

    def __init__(
        self,
        collection_name: str,
        *,
        persist_dir: Path | str | None = ".forgeai/chroma",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        reset: bool = False,
    ) -> None:
        import chromadb

        self.collection_name = collection_name
        self._embedder = embedding_provider or LocalSentenceTransformerEmbeddings(embedding_model)
        if persist_dir is None:
            self.persist_dir = None
            self._client = chromadb.EphemeralClient()
        else:
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        if reset:
            try:
                self._client.delete_collection(name=collection_name)
            except Exception:
                pass
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[DocumentChunk], *, batch_size: int = 64) -> int:
        """Embed and upsert document chunks into the persistent Chroma collection."""
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [chunk.chunk_id for chunk in batch]
            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            embeddings = self._embedder.embed_documents(texts)
            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total += len(batch)
        return total

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedEvidence]:
        """Query ChromaDB and return `RetrievedEvidence` items with full source metadata."""
        if self.count() == 0:
            return []

        query_vec = self._embedder.embed_query(query_text)
        n_results = min(top_k, self.count())
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        evidences: list[RetrievedEvidence] = []
        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            doc_text = docs[idx] if idx < len(docs) else ""
            dist = float(distances[idx]) if idx < len(distances) and distances[idx] is not None else None
            evidences.append(
                RetrievedEvidence(
                    chunk_id=str(chunk_id),
                    doc_id=str(meta.get("doc_id", "unknown")),
                    source_file=str(meta.get("source_file", "unknown")),
                    unit_label=str(meta.get("unit_label", "unknown")),
                    text_excerpt=doc_text,
                    distance=dist,
                )
            )
        return evidences

    def query_by_tokens(
        self,
        tokens: list[str],
        *,
        limit_per_token: int = 3,
    ) -> list[RetrievedEvidence]:
        """Find chunks containing exact identifier substrings (domain-independent)."""
        if not tokens or self.count() == 0:
            return []

        seen_ids: set[str] = set()
        evidences: list[RetrievedEvidence] = []
        for token in tokens:
            cleaned = token.strip()
            if not cleaned:
                continue
            result = self._collection.get(
                where_document={"$contains": cleaned},
                include=["documents", "metadatas"],
                limit=limit_per_token,
            )
            ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            for idx, chunk_id in enumerate(ids):
                cid = str(chunk_id)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                meta = metas[idx] if idx < len(metas) and metas[idx] else {}
                doc_text = docs[idx] if idx < len(docs) else ""
                evidences.append(
                    RetrievedEvidence(
                        chunk_id=cid,
                        doc_id=str(meta.get("doc_id", "unknown")),
                        source_file=str(meta.get("source_file", "unknown")),
                        unit_label=str(meta.get("unit_label", "unknown")),
                        text_excerpt=doc_text,
                        distance=0.0,
                    )
                )
        return evidences

    def get_adjacent_chunks(
        self,
        evidences: list[RetrievedEvidence],
        *,
        window: int = 1,
        max_extra: int = 4,
    ) -> list[RetrievedEvidence]:
        """Fetch immediate neighbor chunks (and opening chunk `c0`) from the same document units."""
        if not evidences or window < 1 or max_extra < 1 or self.count() == 0:
            return []

        existing_ids = {ev.chunk_id for ev in evidences}
        candidate_ids: list[str] = []
        seen_candidates: set[str] = set(existing_ids)

        for ev in evidences:
            if ev.chunk_id.startswith("csv_exact:") or ev.unit_label.startswith("row_"):
                continue
            marker = f"{ev.doc_id}:{ev.unit_label}:c"
            if not ev.chunk_id.startswith(marker):
                continue
            raw_idx = ev.chunk_id[len(marker) :]
            if not raw_idx.isdigit():
                continue
            base_idx = int(raw_idx)

            neighbor_indices: list[int] = []
            for offset in range(1, window + 1):
                if base_idx - offset >= 0:
                    neighbor_indices.append(base_idx - offset)
                neighbor_indices.append(base_idx + offset)
            if 0 not in neighbor_indices:
                neighbor_indices.append(0)

            for idx in neighbor_indices:
                cid = f"{ev.doc_id}:{ev.unit_label}:c{idx}"
                if cid not in seen_candidates:
                    seen_candidates.add(cid)
                    candidate_ids.append(cid)

        if not candidate_ids:
            return []

        result = self._collection.get(
            ids=candidate_ids,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        by_id: dict[str, RetrievedEvidence] = {}
        for idx, chunk_id in enumerate(ids):
            cid = str(chunk_id)
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            doc_text = docs[idx] if idx < len(docs) else ""
            if not doc_text:
                continue
            by_id[cid] = RetrievedEvidence(
                chunk_id=cid,
                doc_id=str(meta.get("doc_id", "unknown")),
                source_file=str(meta.get("source_file", "unknown")),
                unit_label=str(meta.get("unit_label", "unknown")),
                text_excerpt=doc_text,
                distance=None,
            )

        ordered: list[RetrievedEvidence] = []
        for cid in candidate_ids:
            if cid in by_id:
                ordered.append(by_id[cid])
                if len(ordered) >= max_extra:
                    break
        return ordered

    def count(self) -> int:
        """Return number of indexed chunks in the collection."""
        return int(self._collection.count())

    def reset(self) -> None:
        """Delete and recreate the collection cleanly."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def build_isolated_vector_store(
    definition: DeclarativeAgentDefinition,
    units: list[DocumentUnit],
    *,
    persist_dir: Path | str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    collection_name: str | None = None,
    reset: bool = True,
) -> ChromaVectorStore:
    """Create a fresh, run-isolated Chroma collection containing ONLY `units` for `definition`."""
    store = ChromaVectorStore(
        collection_name=collection_name or definition.retrieval.collection_name,
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=definition.retrieval.embedding_model,
        reset=reset,
    )
    effective_units = rebatch_csv_units(units, definition.chunking.csv_rows_per_unit)
    chunks = chunk_units(effective_units, definition.chunking)
    if chunks:
        store.index_chunks(chunks)
    return store

