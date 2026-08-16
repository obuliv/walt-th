"""Shared machinery for reward model variants that also consume `sql_context` — the
schema (SQLite CREATE TABLE/INSERT INTO statements) `sql` executes against, carried on
`Example.sql_context` (see rm/data/base.py). Builds on `LRRewardModelV3` (current best:
cosine_sim + is_sql_valid), so both context variants below are strict ablations on top
of it — only `_phi` differs between lr_model_v4.py and lr_model_v5.py.

Embeds each example's whole context as one vector, keyed by the joined statement text
(`_context_key`) — the same "\\n".join(...) convention sql_agent.py already uses to
pass schema_context to the LLM, so cache hits are consistent across the codebase. A row
with no context (e.g. sql_good is itself DDL) gets an all-zero context vector rather
than an embedding call, handled per-subclass in `_phi` since V4 (a scalar cosine_sim)
and V5 (a concatenated vector) need different empty-context fallbacks.
"""
from __future__ import annotations

import numpy as np

from walt.rm.data.base import Example
from walt.rm.model.lr.lr_model_v3 import LRRewardModelV3


class ContextAwareLRRewardModel(LRRewardModelV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _context_key(sql_context: tuple[str, ...]) -> str:
        return "\n".join(sql_context)

    def _embed_context_unique(self, keys: list[str]) -> None:
        missing = sorted({k for k in keys if k and k not in self._context_cache})
        if not missing:
            return
        vectors = self.embedding_provider.embed(missing)
        for key, vec in zip(missing, vectors):
            self._context_cache[key] = vec

    def warm_cache(self, examples: list[Example]) -> None:
        super().warm_cache(examples)
        self._embed_context_unique([self._context_key(ex.sql_context) for ex in examples])

    def score(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> float:
        # Lazily embeds sql_context on an ad hoc score() call (e.g. from the agent) the
        # same way LRRewardModel.score() lazily embeds question/sql — warm_cache() is
        # the fast path, this is the fallback so score() never assumes a pre-warmed cache.
        self._embed_context_unique([self._context_key(sql_context)])
        return super().score(question, sql, sql_context)
