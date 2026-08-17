"""V7 of the pairwise LR reward model: V3's phi (embed(sql) + cosine_sim +
is_sql_valid) plus a length-ratio feature, six sqlglot-derived structural
counts/flags, a question/args lexical-overlap feature, and a commands-vs-args
embedding-similarity block -- deliberately built on V3, not V6, since this variant
originally did NOT include is_schema_valid (a standing constraint from earlier RM
work: no execution-against-schema feature here, only symbolic/embedding signal).
`include_schema_valid=True` (default off, for backward compat with existing saved V7
models) lifts that constraint and appends V6's own feature, for testing whether V7's
extra symbolic/lexical features add anything on top of it rather than instead of it.

phi_v7(question, sql) = concat(
    phi_v3(question, sql),                                   # 771 dims
    [sql_length_ratio(question, sql)],                        # 1
    sql_structural_counts(sql),                                # 6
    [question_arg_overlap(question, sql)],                     # 1
    <commands/args embedding-similarity block>,                # 2 (cosine) or 1536 (raw)
    [is_schema_valid(sql, sql_context)],                       # 1, only if include_schema_valid
)

embedding_diff_mode picks how the commands/args-vs-question embedding comparison is
folded in:
  - "cosine" (default): cosine_sim(embed(commands_text), embed(question)) and
    cosine_sim(embed(args_text), embed(question)) -- 2 scalars, matching V4's
    already-validated "scalar interaction feature" pattern.
  - "raw": embed(commands_text) - embed(question) and embed(args_text) -
    embed(question) -- 2 x 768 raw dims, matching what was literally requested.
    Unlike V5's context concat (a proven, coefficient-verified dead end under
    pairwise-difference training -- see CLAUDE.md), commands_text/args_text vary
    per candidate, so this does not structurally cancel; it may still just be a
    high-dimensional, overfitting-prone restatement of signal already in embed(sql)
    and cosine_sim -- an open empirical question this mode exists to test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from walt.rm.data.base import Example
from walt.rm.model.embeddings import EmbeddingProvider, build_provider_from_config
from walt.rm.model.lr.lr_model_v3 import LRRewardModelV3
from walt.rm.model.sql_features import (
    is_schema_valid,
    question_arg_overlap,
    split_sql_commands_and_args,
    sql_length_ratio,
    sql_structural_counts,
)
from walt.utils.sql_exec import normalize_sql

EMBEDDING_DIFF_MODES = ("cosine", "raw")


class LRRewardModelV7(LRRewardModelV3):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        seed: int = 42,
        C: float = 1.0,
        penalty: str = "l2",
        l1_ratio: Optional[float] = None,
        severity_zero_as_positive: bool = False,
        ignore_sql_good: bool = False,
        embedding_diff_mode: str = "cosine",
        include_schema_valid: bool = False,
    ):
        if embedding_diff_mode not in EMBEDDING_DIFF_MODES:
            raise ValueError(f"embedding_diff_mode must be one of {EMBEDDING_DIFF_MODES}, got {embedding_diff_mode!r}")
        super().__init__(
            embedding_provider,
            seed,
            C=C,
            penalty=penalty,
            l1_ratio=l1_ratio,
            severity_zero_as_positive=severity_zero_as_positive,
            ignore_sql_good=ignore_sql_good,
        )
        self.embedding_diff_mode = embedding_diff_mode
        self.include_schema_valid = include_schema_valid
        # Keyed by the *derived* commands/args text (not the raw sql) -- naturally
        # dedupes structurally-identical candidates, mirrors _sql_cache's
        # keyed-by-text convention.
        self._commands_cache: dict[str, np.ndarray] = {}
        self._args_cache: dict[str, np.ndarray] = {}

    def warm_cache(self, examples: list[Example]) -> None:
        super().warm_cache(examples)
        sqls = [normalize_sql(ex.sql_good) for ex in examples] + [
            normalize_sql(b.sql) for ex in examples for b in ex.sql_bad
        ]
        commands_texts, args_texts = [], []
        for sql in sqls:
            c, a = split_sql_commands_and_args(sql)
            commands_texts.append(c)
            args_texts.append(a)
        self._embed_unique(commands_texts, self._commands_cache)
        self._embed_unique(args_texts, self._args_cache)

    def _phi(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> np.ndarray:
        base_phi = super()._phi(question, sql, sql_context)

        length_ratio = sql_length_ratio(question, sql)
        structural = sql_structural_counts(sql)
        overlap = question_arg_overlap(question, sql)

        commands_text, args_text = split_sql_commands_and_args(sql)
        q_vec = self._question_cache[question]
        commands_vec = self._commands_cache[commands_text]
        args_vec = self._args_cache[args_text]

        if self.embedding_diff_mode == "cosine":
            diff_block = [float(np.dot(q_vec, commands_vec)), float(np.dot(q_vec, args_vec))]
        else:
            diff_block = np.concatenate([commands_vec - q_vec, args_vec - q_vec])

        parts = [base_phi, [length_ratio], structural, [overlap], diff_block]
        if self.include_schema_valid:
            parts.append([1.0 if is_schema_valid(sql, sql_context) else 0.0])
        return np.concatenate(parts)

    def score(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> float:
        if self.coef_ is None:
            raise RuntimeError("LRRewardModelV7.score() called before fit()/load()")
        sql = normalize_sql(sql)  # match warm_cache()/fit()'s cache keys — see LRRewardModel.score()
        self._embed_unique([question], self._question_cache)
        self._embed_unique([sql], self._sql_cache)
        commands_text, args_text = split_sql_commands_and_args(sql)
        self._embed_unique([commands_text], self._commands_cache)
        self._embed_unique([args_text], self._args_cache)
        phi = self._phi(question, sql, sql_context)
        return float(np.dot(self.coef_, phi))

    def save(self, path: str | Path) -> None:
        payload = {
            "coef": self.coef_,
            "seed": self.seed,
            "C": self.C,
            "penalty": self.penalty,
            "l1_ratio": self.l1_ratio,
            "severity_zero_as_positive": self.severity_zero_as_positive,
            "ignore_sql_good": self.ignore_sql_good,
            "embedding_diff_mode": self.embedding_diff_mode,
            "include_schema_valid": self.include_schema_valid,
            "embedding_config": self.embedding_provider.config,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, embedding_provider: EmbeddingProvider | None = None) -> "LRRewardModelV7":
        payload = joblib.load(path)
        provider = embedding_provider or build_provider_from_config(payload["embedding_config"])
        model = cls(
            embedding_provider=provider,
            seed=payload["seed"],
            C=payload.get("C", 1.0),
            penalty=payload.get("penalty", "l2"),
            l1_ratio=payload.get("l1_ratio"),
            severity_zero_as_positive=payload.get("severity_zero_as_positive", False),
            ignore_sql_good=payload.get("ignore_sql_good", False),
            embedding_diff_mode=payload.get("embedding_diff_mode", "cosine"),
            include_schema_valid=payload.get("include_schema_valid", False),
        )
        model.coef_ = payload["coef"]
        return model
