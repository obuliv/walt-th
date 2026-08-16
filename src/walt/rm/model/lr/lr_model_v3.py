"""V3 of the pairwise LR reward model: V1's phi plus a handcrafted syntax-validity
feature, targeting the syntax_error mistake category specifically — the per-reason
breakdown from earlier runs showed it as one of the two weakest categories (~0.70-0.73
pairwise accuracy) across every embedding-only variant tried so far, which makes sense:
a general-purpose embedding model has no explicit notion of "does this parse."

phi_v3(question, sql) = concat(phi_v1(question, sql), [is_sql_valid(sql)])
— reuses V1's cosine-sim phi unchanged (via super()._phi) and appends one binary
feature. Only _phi is overridden; everything else (fit/score/warm_cache/save/load) is
inherited unchanged from LRRewardModel.
"""
from __future__ import annotations

import numpy as np

from walt.rm.model.lr.lr_model import LRRewardModel
from walt.rm.model.sql_features import is_sql_valid


class LRRewardModelV3(LRRewardModel):
    def _phi(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> np.ndarray:
        base_phi = super()._phi(question, sql, sql_context)
        return np.concatenate([base_phi, [1.0 if is_sql_valid(sql) else 0.0]])
