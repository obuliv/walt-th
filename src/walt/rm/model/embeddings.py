"""Pluggable text-embedding providers for reward-model feature extraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Embeds a batch of strings into fixed-size vectors."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (len(texts), self.dim) float32 array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> dict:
        """JSON-serializable identity of this provider, used for model persistence —
        enough info to reconstruct an equivalent provider without saving the model itself."""
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingProvider):
    DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_seq_length: int = 1024,
        trust_remote_code: bool = True,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # deferred: heavy (torch) import

        self.model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code, device=device)
        self._model.max_seq_length = max_seq_length
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # so cosine similarity reduces to a dot product downstream
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    @property
    def config(self) -> dict:
        return {
            "type": "sentence_transformer",
            "model_name": self.model_name,
            "max_seq_length": self._model.max_seq_length,
            "trust_remote_code": self._trust_remote_code,
            "dim": self.dim,
        }


def build_provider_from_config(config: dict) -> EmbeddingProvider:
    if config["type"] == "sentence_transformer":
        return SentenceTransformerEmbedding(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            trust_remote_code=config["trust_remote_code"],
        )
    raise ValueError(f"Unknown embedding provider config type: {config['type']!r}")
