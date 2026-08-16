"""A reward model with no learned signal at all — scores every candidate identically.
On its own it's useless (rank() would just return candidates in input order, since
Python's sorted() is stable). Its purpose is as the inner model under
SchemaFilteredRewardModel (schema_filter.py): with every score tied, that wrapper's
hard schema-validity gate becomes the *only* thing influencing the ranking, so the net
effect of ConstantRewardModel + SchemaFilteredRewardModel is "the first schema-valid
candidate, in the order the LLM generated them" — a clean baseline to isolate how much
of any reranking win comes from filtering out execution-broken candidates alone, with
zero learned preference among the survivors.
"""
from __future__ import annotations

from typing import Sequence

from walt.rm.data.base import Example
from walt.rm.model.base import BaseRewardModel


class ConstantRewardModel(BaseRewardModel):
    def fit(self, train_examples: list[Example]) -> None:
        pass  # nothing to learn

    def warm_cache(self, examples: list[Example]) -> None:
        pass  # nothing to embed/cache

    def score(self, question: str, sql: str, sql_context: Sequence[str] = ()) -> float:
        return 0.0
