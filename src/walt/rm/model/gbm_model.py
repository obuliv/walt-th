"""Gradient-boosted-trees reward model: pointwise-trained, embedding-based SQL ranker.

LRRewardModel is trained *pairwise* (on phi(q,A) - phi(q,B)) specifically because that
lets a LINEAR score decompose additively — sorting w.phi(q,sql) over any-size candidate
list is guaranteed transitive. A nonlinear model trained the same way (on feature
differences) loses that guarantee: f(phi(A) - phi(B)) is a pairwise comparator, not a
per-candidate score, and ranking N candidates from pairwise comparisons alone needs an
O(N^2) tournament that can produce cycles (A beats B, B beats C, C beats A).

So GBMRewardModel is trained *pointwise* instead: one row per candidate (label = 1 for
sql_good, 0 for each sql_bad), features = phi(question, sql) directly (no differencing),
and score(question, sql) = predict_proba(phi(question, sql))[1]. This keeps the
per-candidate score / clean total order property while allowing GradientBoosting's
nonlinear feature interactions (which the earlier lr_model_v3.py's is_sql_valid feature
can't exploit under a linear model beyond a fixed additive weight).

Reuses LRRewardModelV3's phi (V1's cosine-sim features + is_sql_valid) via inheritance —
only __init__/fit/score/save/load are overridden; _phi/_embed_unique/warm_cache come
from the parent chain unchanged. `self.coef_`/`self.C` from LRRewardModel are inherited
but unused here (this class never calls LogisticRegression); left alone rather than
un-inheriting them, since the shared embedding/caching machinery is what's actually
being reused.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from walt.rm.data.base import Example
from walt.rm.model.embeddings import EmbeddingProvider, build_provider_from_config
from walt.rm.model.lr_model_v3 import LRRewardModelV3


class GBMRewardModel(LRRewardModelV3):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        seed: int = 42,
        max_iter: int = 200,
        max_depth: Optional[int] = None,
        learning_rate: float = 0.1,
    ):
        super().__init__(embedding_provider, seed)
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.clf: Optional[HistGradientBoostingClassifier] = None

    def fit(self, train_examples: list[Example]) -> None:
        embed_start = time.perf_counter()
        self.warm_cache(train_examples)
        embed_seconds = time.perf_counter() - embed_start

        X_rows, y_rows = [], []
        for ex in train_examples:
            X_rows.append(self._phi(ex.question, ex.sql_good))
            y_rows.append(1)
            for bad in ex.sql_bad:
                X_rows.append(self._phi(ex.question, bad.sql))
                y_rows.append(0)
        X = np.stack(X_rows)
        y = np.array(y_rows)

        fit_start = time.perf_counter()
        self.clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            class_weight="balanced",  # ~1:5 positive:negative imbalance (1 good, ~5 bad per question)
            random_state=self.seed,
        )
        self.clf.fit(X, y)
        fit_seconds = time.perf_counter() - fit_start

        n_pos = int(y.sum())
        self.fit_info = {
            "n_train_examples": len(train_examples),
            "n_train_rows": len(y_rows),
            "embedding_dim": self.embedding_provider.dim,
            "feature_dim": X.shape[1],
            "label_balance": {"positive": n_pos, "negative": len(y_rows) - n_pos},
            "gbm_max_iter": self.max_iter,
            "gbm_n_iter": int(self.clf.n_iter_),
            "gbm_max_depth": self.max_depth,
            "gbm_learning_rate": self.learning_rate,
            "embed_seconds": round(embed_seconds, 3),
            "fit_seconds": round(fit_seconds, 3),
        }

    def score(self, question: str, sql: str) -> float:
        if self.clf is None:
            raise RuntimeError("GBMRewardModel.score() called before fit()/load()")
        self._embed_unique([question], self._question_cache)
        self._embed_unique([sql], self._sql_cache)
        phi = self._phi(question, sql).reshape(1, -1)
        return float(self.clf.predict_proba(phi)[0, 1])

    def save(self, path: str | Path) -> None:
        payload = {
            "clf": self.clf,
            "seed": self.seed,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "embedding_config": self.embedding_provider.config,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, embedding_provider: EmbeddingProvider | None = None) -> "GBMRewardModel":
        payload = joblib.load(path)
        provider = embedding_provider or build_provider_from_config(payload["embedding_config"])
        model = cls(
            embedding_provider=provider,
            seed=payload["seed"],
            max_iter=payload["max_iter"],
            max_depth=payload["max_depth"],
            learning_rate=payload["learning_rate"],
        )
        model.clf = payload["clf"]
        return model
