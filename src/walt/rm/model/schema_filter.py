"""Wraps any BaseRewardModel with a hard schema-validity pre-filter — the same signal
lr_v6 learns as a soft, weighted feature (is_schema_valid, see sql_features.py), turned
into a hard gate instead. Motivated directly by DistilBertRewardModel's agent-eval
result: its SQL-execution pass rate with RM reranking (85.7%) came in *below* its own
no-rerank baseline (86.4%), meaning it was sometimes actively preferring a
schema-invalid candidate over a valid one — something is_schema_valid catches for
free (no forward pass, no retraining), regardless of what the wrapped model itself
learned.

Implementation: score() adds a large negative offset to any candidate that fails
is_schema_valid, before delegating to the wrapped model's own score(). This guarantees
every schema-valid candidate outranks every schema-invalid one (transitively — it's
still just real-number comparison), while preserving the wrapped model's relative
ordering *within* each tier. rank() is inherited from BaseRewardModel unchanged and
gets the filtering behavior for free this way: rank 1 is always schema-valid if any
candidate is, and the ranking degrades gracefully to the wrapped model's own
preference when none are (nothing to gain from filtering in that case).

Purely an inference-time wrapper — fit()/warm_cache() just delegate to the wrapped
model, this adds no trainable state of its own.
"""
from __future__ import annotations

from typing import Sequence

from walt.rm.data.base import Example
from walt.rm.model.base import BaseRewardModel
from walt.rm.model.sql_features import is_schema_valid

# Dominates any realistic score range from either the lr_* family (bounded log-odds
# margins) or DistilBertRewardModel (raw logits, empirically small) — big enough that
# no valid-tier score could ever be pushed below an invalid-tier one by this offset.
INVALID_PENALTY = 1000.0


class SchemaFilteredRewardModel(BaseRewardModel):
    def __init__(self, inner: BaseRewardModel):
        self.inner = inner

    def fit(self, train_examples: list[Example]) -> None:
        self.inner.fit(train_examples)

    def warm_cache(self, examples: list[Example]) -> None:
        if hasattr(self.inner, "warm_cache"):
            self.inner.warm_cache(examples)

    def score(self, question: str, sql: str, sql_context: Sequence[str] = ()) -> float:
        base = self.inner.score(question, sql, sql_context)
        return base if is_schema_valid(sql, sql_context) else base - INVALID_PENALTY

    def predict_error_code(self, question: str, sql: str) -> str | None:
        return self.inner.predict_error_code(question, sql)
