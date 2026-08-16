"""V4 of the pairwise LR reward model: V3's phi plus one scalar feature,
cosine_sim(embed(sql_context), embed(sql)) — does sql "look like" the schema it's
supposed to run against, by the same embedding space used for everything else here.

phi_v4(question, sql, sql_context) = concat(phi_v3(question, sql), [cosine_sim(context, sql)])

Rows with no sql_context (sql_good is itself DDL) get cosine_sim=0.0 (neutral —
"no signal") rather than embedding an empty string, which would give a well-defined
but semantically meaningless value. Only _phi is overridden; the context-embedding
machinery (_context_cache/warm_cache/score) is inherited from ContextAwareLRRewardModel.
"""
from __future__ import annotations

import numpy as np

from walt.rm.model.lr.lr_model_context import ContextAwareLRRewardModel


class LRRewardModelV4(ContextAwareLRRewardModel):
    def _phi(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> np.ndarray:
        base_phi = super()._phi(question, sql, sql_context)
        key = self._context_key(sql_context)
        if not key:
            cos_sim = 0.0
        else:
            context_vec = self._context_cache[key]
            sql_vec = self._sql_cache[sql]
            cos_sim = float(np.dot(context_vec, sql_vec))  # both L2-normalized -> dot product = cosine
        return np.concatenate([base_phi, [cos_sim]])
