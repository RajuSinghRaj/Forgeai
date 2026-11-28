"""Local Sentence-Transformers embedding provider (no paid embedding APIs)."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    """Cache loaded SentenceTransformer models in memory across phases."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingProvider(Protocol):
    """Protocol for embedding document chunks and queries locally."""

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalSentenceTransformerEmbeddings:
    """Lazy-loading local SentenceTransformer embedder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            self._model = _load_sentence_transformer(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        vector = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vector.tolist()

