"""Persistent ChromaDB vector store with source-level provenance."""

from __future__ import annotations

from pathlib import Path

from forgeai.retrieval.chunker import DocumentChunk
from forgeai.retrieval.embeddings import EmbeddingProvider, LocalSentenceTransformerEmbeddings
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

