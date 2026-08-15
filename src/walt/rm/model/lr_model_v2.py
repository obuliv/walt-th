"""V2 of the pairwise LR reward model.

LRRewardModel (V1)'s phi uses cosine_sim(embed(q), embed(sql)) as the only
question-SQL interaction feature. Because V1's embeddings are L2-normalized,
that cosine similarity *is already* a dot product — just a dot product of
unit-length vectors, which discards magnitude information.

V2 can add a second, genuinely different interaction feature: the raw (unnormalized)
dot product of the two embeddings, which preserves magnitude — e.g. a question or
SQL string whose embedding norm reflects something the model finds "unusual" or
"complex" (norms vary across sentence-transformer outputs despite superficially
similar text length) shows up here but is invisible to cosine similarity.

Which interaction feature(s) are used is configurable (use_cosine_sim/use_dot_product),
so this one class covers three variants: dot-product-only, cosine-only (~=V1, but via
the raw-embedding code path), or both together. The raw dot product's scale is very
different from cosine similarity's (bounded [-1,1]) or the unit-norm SQL embedding
dims, which can destabilize an L2-penalized fit — standardize_dot_product (on by
default) rescales it to zero-mean/unit-variance using TRAINING-set statistics only
(computed in fit(), before test data is ever touched, and persisted via save()/load()
so score() applies the exact same rescaling at inference).

phi(question, sql) = concat(embed(sql)_normalized,
                             [cosine_sim(q, sql)] if use_cosine_sim,
                             [standardized_raw_dot(q, sql)] if use_dot_product)

Only _embed_unique (fetches raw, unnormalized vectors instead of normalized ones),
_phi, fit() (adds a stats pre-pass before deferring to the parent), and save()/load()
(persist the extra config + standardization stats) are overridden; score()/warm_cache()
are inherited unchanged from LRRewardModel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from walt.rm.data.base import Example
from walt.rm.model.base import all_candidates
from walt.rm.model.embeddings import EmbeddingProvider, build_provider_from_config
from walt.rm.model.lr_model import LRRewardModel


class LRRewardModelV2(LRRewardModel):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        seed: int = 42,
        C: float = 1.0,
        penalty: str = "l2",
        l1_ratio: Optional[float] = None,
        use_cosine_sim: bool = True,
        use_dot_product: bool = True,
        standardize_dot_product: bool = True,
    ):
        if not use_cosine_sim and not use_dot_product:
            raise ValueError("LRRewardModelV2 needs at least one of use_cosine_sim/use_dot_product")
        super().__init__(embedding_provider, seed, C=C, penalty=penalty, l1_ratio=l1_ratio)
        self.use_cosine_sim = use_cosine_sim
        self.use_dot_product = use_dot_product
        self.standardize_dot_product = standardize_dot_product
        self._dot_mean = 0.0
        self._dot_std = 1.0

    def _embed_unique(self, texts: list[str], cache: dict[str, np.ndarray]) -> None:
        missing = sorted({t for t in texts if t not in cache})
        if not missing:
            return
        vectors = self.embedding_provider.embed(missing, normalize=False)
        for text, vec in zip(missing, vectors):
            cache[text] = vec

    def _raw_dot(self, question: str, sql: str) -> float:
        return float(np.dot(self._question_cache[question], self._sql_cache[sql]))

    def _phi(self, question: str, sql: str) -> np.ndarray:
        q_raw = self._question_cache[question]
        sql_raw = self._sql_cache[sql]
        sql_norm = sql_raw / np.linalg.norm(sql_raw)

        parts = [sql_norm]
        if self.use_cosine_sim:
            q_norm = q_raw / np.linalg.norm(q_raw)
            parts.append([float(np.dot(q_norm, sql_norm))])
        if self.use_dot_product:
            raw_dot = self._raw_dot(question, sql)
            if self.standardize_dot_product:
                raw_dot = (raw_dot - self._dot_mean) / self._dot_std
            parts.append([raw_dot])
        return np.concatenate(parts)

    def fit(self, train_examples: list[Example]) -> None:
        self.warm_cache(train_examples)  # populate raw-embedding cache before computing stats
        if self.use_dot_product and self.standardize_dot_product:
            dots = [
                self._raw_dot(ex.question, sql)
                for ex in train_examples
                for sql in all_candidates(ex)
            ]
            self._dot_mean = float(np.mean(dots))
            self._dot_std = float(np.std(dots)) or 1.0  # guard against a degenerate zero-variance case
        super().fit(train_examples)  # warm_cache() no-ops (already warm); _phi now uses the stats above

    def save(self, path: str | Path) -> None:
        payload = {
            "coef": self.coef_,
            "seed": self.seed,
            "C": self.C,
            "penalty": self.penalty,
            "l1_ratio": self.l1_ratio,
            "embedding_config": self.embedding_provider.config,
            "use_cosine_sim": self.use_cosine_sim,
            "use_dot_product": self.use_dot_product,
            "standardize_dot_product": self.standardize_dot_product,
            "dot_mean": self._dot_mean,
            "dot_std": self._dot_std,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, embedding_provider: EmbeddingProvider | None = None) -> "LRRewardModelV2":
        payload = joblib.load(path)
        provider = embedding_provider or build_provider_from_config(payload["embedding_config"])
        model = cls(
            embedding_provider=provider,
            seed=payload["seed"],
            C=payload.get("C", 1.0),
            penalty=payload.get("penalty", "l2"),
            l1_ratio=payload.get("l1_ratio"),
            use_cosine_sim=payload["use_cosine_sim"],
            use_dot_product=payload["use_dot_product"],
            standardize_dot_product=payload["standardize_dot_product"],
        )
        model.coef_ = payload["coef"]
        model._dot_mean = payload["dot_mean"]
        model._dot_std = payload["dot_std"]
        return model
