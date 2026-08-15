"""Logistic-regression reward model: pairwise-trained, embedding-based SQL ranker.

Scoring function: score(question, sql) = w . phi(question, sql), where
phi(question, sql) = concat(embed(sql), [cosine_sim(embed(question), embed(sql))]).
Embeddings are pre-normalized, so cosine similarity reduces to a dot product.

Trained by fitting sklearn LogisticRegression on phi(q, A) - phi(q, B) for
(sql_good, sql_bad) pairs, with A/B randomly assigned per pair so the intercept
doesn't pick up positional bias. Only the resulting weight vector is kept (no
intercept) — score() only needs relative order across an arbitrary-size candidate
list, which an additive constant doesn't change.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from walt.rm.data.base import Example
from walt.rm.model.base import BaseRewardModel, all_candidates
from walt.rm.model.embeddings import EmbeddingProvider, build_provider_from_config


SOLVER_BY_PENALTY = {"l2": "lbfgs", "l1": "liblinear", "elasticnet": "saga"}


class LRRewardModel(BaseRewardModel):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        seed: int = 42,
        C: float = 1.0,
        penalty: str = "l2",
        l1_ratio: Optional[float] = None,
    ):
        if penalty not in SOLVER_BY_PENALTY:
            raise ValueError(f"penalty must be one of {sorted(SOLVER_BY_PENALTY)}, got {penalty!r}")
        if penalty == "elasticnet" and l1_ratio is None:
            raise ValueError("penalty='elasticnet' requires l1_ratio")
        self.embedding_provider = embedding_provider
        self.seed = seed
        self.C = C  # inverse regularization strength (sklearn convention: smaller = more regularization)
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.coef_: Optional[np.ndarray] = None  # shape (dim + 1,)
        self._question_cache: dict[str, np.ndarray] = {}
        self._sql_cache: dict[str, np.ndarray] = {}
        # populated by fit() — training diagnostics worth keeping on record even though
        # they aren't part of the headline evaluate() metrics (convergence, timing,
        # dataset shape at fit time). Not set by load(), since a loaded model wasn't
        # just fit in this process.
        self.fit_info: dict = {}

    def _embed_unique(self, texts: list[str], cache: dict[str, np.ndarray]) -> None:
        missing = sorted({t for t in texts if t not in cache})
        if not missing:
            return
        vectors = self.embedding_provider.embed(missing)
        for text, vec in zip(missing, vectors):
            cache[text] = vec

    def warm_cache(self, examples: list[Example]) -> None:
        """Pre-embeds every unique question and unique SQL string (good + bad) across
        `examples` that isn't already cached, in one batched call each. Callers should
        warm the cache for both train and eval examples up front so score()/evaluate()
        never fall back to slow one-string-at-a-time embedding calls."""
        questions = [ex.question for ex in examples]
        sqls = [sql for ex in examples for sql in all_candidates(ex)]
        self._embed_unique(questions, self._question_cache)
        self._embed_unique(sqls, self._sql_cache)

    def _phi(self, question: str, sql: str) -> np.ndarray:
        q_vec = self._question_cache[question]
        sql_vec = self._sql_cache[sql]
        cos_sim = float(np.dot(q_vec, sql_vec))
        return np.concatenate([sql_vec, [cos_sim]])

    def fit(self, train_examples: list[Example]) -> None:
        embed_start = time.perf_counter()
        self.warm_cache(train_examples)
        embed_seconds = time.perf_counter() - embed_start

        rng = random.Random(self.seed)
        X_rows, y_rows = [], []
        for ex in train_examples:
            for bad in ex.sql_bad:
                if rng.random() < 0.5:
                    a_sql, b_sql, label = ex.sql_good, bad.sql, 1
                else:
                    a_sql, b_sql, label = bad.sql, ex.sql_good, 0
                phi_a = self._phi(ex.question, a_sql)
                phi_b = self._phi(ex.question, b_sql)
                X_rows.append(phi_a - phi_b)
                y_rows.append(label)

        X = np.stack(X_rows)
        y = np.array(y_rows)

        max_iter = 1000
        solver = SOLVER_BY_PENALTY[self.penalty]
        fit_start = time.perf_counter()
        clf = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=solver,
            l1_ratio=self.l1_ratio if self.penalty == "elasticnet" else None,
            max_iter=max_iter,
            random_state=self.seed,
        )
        clf.fit(X, y)
        fit_seconds = time.perf_counter() - fit_start

        self.coef_ = clf.coef_[0]
        n_iter = int(clf.n_iter_[0])
        n_pos = int(y.sum())
        n_zero_coefs = int(np.sum(self.coef_ == 0))
        self.fit_info = {
            "n_train_examples": len(train_examples),
            "n_train_pairs": len(y_rows),
            "embedding_dim": self.embedding_provider.dim,
            "feature_dim": int(self.coef_.shape[0]),
            "label_balance": {"a_is_good": n_pos, "b_is_good": len(y_rows) - n_pos},
            "lr_C": self.C,
            "lr_penalty": self.penalty,
            "lr_l1_ratio": self.l1_ratio,
            "lr_solver": solver,
            "lr_max_iter": max_iter,
            "lr_n_iter": n_iter,
            "lr_converged": n_iter < max_iter,
            "lr_n_zero_coefs": n_zero_coefs,  # implicit feature selection under L1/elasticnet; always 0 under L2
            "embed_seconds": round(embed_seconds, 3),
            "fit_seconds": round(fit_seconds, 3),
        }

    def score(self, question: str, sql: str) -> float:
        if self.coef_ is None:
            raise RuntimeError("LRRewardModel.score() called before fit()/load()")
        self._embed_unique([question], self._question_cache)
        self._embed_unique([sql], self._sql_cache)
        phi = self._phi(question, sql)
        return float(np.dot(self.coef_, phi))

    def save(self, path: str | Path) -> None:
        payload = {
            "coef": self.coef_,
            "seed": self.seed,
            "C": self.C,
            "penalty": self.penalty,
            "l1_ratio": self.l1_ratio,
            "embedding_config": self.embedding_provider.config,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, embedding_provider: EmbeddingProvider | None = None) -> "LRRewardModel":
        payload = joblib.load(path)
        provider = embedding_provider or build_provider_from_config(payload["embedding_config"])
        model = cls(
            embedding_provider=provider,
            seed=payload["seed"],
            C=payload.get("C", 1.0),
            penalty=payload.get("penalty", "l2"),
            l1_ratio=payload.get("l1_ratio"),
        )
        model.coef_ = payload["coef"]
        return model
