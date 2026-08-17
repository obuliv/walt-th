"""Embedding-scaling ablation on top of LRRewardModelV3: does scaling the raw
embed(sql) vector's dimensions before concatenation change ranking quality? Isolated
from every other change (same C, same L2 penalty, same cosine_sim/is_sql_valid
features) versus the lr_v3 baseline — only how the ~768-dim embed(sql) block is scaled
before concatenation varies.

Important baseline fact: embed(sql) already comes back L2-normalized, since
SentenceTransformerEmbedding.embed()'s default is `normalize=True` (embeddings.py) and
LRRewardModel never overrides that default. That makes `scaling="l2_normalize"` a
no-op by construction (renormalizing an already-unit-norm vector returns the same
vector) — and it means `scaling="l2_normalize_standardize"` collapses to plain
`scaling="standardize"` (normalize-then-standardize == standardize, since the normalize
step doesn't change anything here). Both modes are still implemented and run rather
than assumed away, precisely so the CV numbers can confirm that equivalence empirically
instead of taking the algebra on faith.

`scaling="standardize"` fits an sklearn StandardScaler on the *training* fold's
embed(sql) vectors only (mirroring LRRewardModelV2's train-only dot-product
standardization) — never on eval data, so no leakage across CV folds. cosine_sim and
is_sql_valid are left untouched in every mode; only the embed(sql) block is scaled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from walt.rm.data.base import Example
from walt.rm.model.base import all_candidates
from walt.rm.model.embeddings import EmbeddingProvider, build_provider_from_config
from walt.rm.model.lr.lr_model_v3 import LRRewardModelV3
from walt.rm.model.sql_features import is_sql_valid

SCALING_CHOICES = ("none", "standardize", "l2_normalize", "l2_normalize_standardize")


class LRRewardModelV3Scaled(LRRewardModelV3):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        seed: int = 42,
        C: float = 1.0,
        penalty: str = "l2",
        l1_ratio: Optional[float] = None,
        scaling: str = "none",
    ):
        if scaling not in SCALING_CHOICES:
            raise ValueError(f"scaling must be one of {SCALING_CHOICES}, got {scaling!r}")
        super().__init__(embedding_provider, seed, C=C, penalty=penalty, l1_ratio=l1_ratio)
        self.scaling = scaling
        self._scaler: StandardScaler | None = None

    def _scale_sql_vec(self, sql_vec: np.ndarray) -> np.ndarray:
        if self.scaling in ("l2_normalize", "l2_normalize_standardize"):
            norm = np.linalg.norm(sql_vec)
            if norm > 0:
                sql_vec = sql_vec / norm
        if self.scaling in ("standardize", "l2_normalize_standardize"):
            sql_vec = self._scaler.transform(sql_vec.reshape(1, -1))[0]
        return sql_vec

    def _phi(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> np.ndarray:
        q_vec = self._question_cache[question]
        sql_vec = self._sql_cache[sql]
        cos_sim = float(np.dot(q_vec, sql_vec))  # always on the original, unscaled embeddings
        scaled_sql_vec = self._scale_sql_vec(sql_vec)
        valid = 1.0 if is_sql_valid(sql) else 0.0
        return np.concatenate([scaled_sql_vec, [cos_sim], [valid]])

    def fit(self, train_examples: list[Example]) -> None:
        self.warm_cache(train_examples)
        if self.scaling in ("standardize", "l2_normalize_standardize"):
            vecs = []
            for ex in train_examples:
                for sql in all_candidates(ex):
                    v = self._sql_cache[sql]
                    if self.scaling == "l2_normalize_standardize":
                        norm = np.linalg.norm(v)
                        if norm > 0:
                            v = v / norm
                    vecs.append(v)
            self._scaler = StandardScaler().fit(np.stack(vecs))
        super().fit(train_examples)  # warm_cache no-ops; builds pairs via self._phi above

    def save(self, path: str | Path) -> None:
        payload = {
            "coef": self.coef_,
            "seed": self.seed,
            "C": self.C,
            "penalty": self.penalty,
            "l1_ratio": self.l1_ratio,
            "embedding_config": self.embedding_provider.config,
            "scaling": self.scaling,
            "scaler_mean": self._scaler.mean_ if self._scaler is not None else None,
            "scaler_scale": self._scaler.scale_ if self._scaler is not None else None,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, embedding_provider: EmbeddingProvider | None = None) -> "LRRewardModelV3Scaled":
        payload = joblib.load(path)
        provider = embedding_provider or build_provider_from_config(payload["embedding_config"])
        model = cls(
            embedding_provider=provider,
            seed=payload["seed"],
            C=payload.get("C", 1.0),
            penalty=payload.get("penalty", "l2"),
            l1_ratio=payload.get("l1_ratio"),
            scaling=payload["scaling"],
        )
        model.coef_ = payload["coef"]
        if payload.get("scaler_mean") is not None:
            scaler = StandardScaler()
            scaler.mean_ = payload["scaler_mean"]
            scaler.scale_ = payload["scaler_scale"]
            scaler.var_ = payload["scaler_scale"] ** 2
            scaler.n_features_in_ = len(payload["scaler_mean"])
            model._scaler = scaler
        return model
