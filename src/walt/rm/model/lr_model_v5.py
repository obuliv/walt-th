"""V5 of the pairwise LR reward model: V3's phi with embed(sql_context) concatenated
in full alongside embed(sql), rather than V4's single cosine_sim scalar — lets the
linear model weight individual schema-embedding dimensions independently instead of
compressing schema-vs-sql relatedness into one number.

phi_v5(question, sql, sql_context) = concat(phi_v3(question, sql), embed(sql_context))

Roughly doubles feature_dim vs V3/V4 (768 -> ~1536, plus the two scalars), which is a
real risk of overfitting on a fixed C tuned for the smaller phi — see CLAUDE.md for
whether that held up under CV. Rows with no sql_context get an all-zero context vector
(same dimensionality, so phi's shape stays fixed across rows) rather than embedding an
empty string. Only _phi is overridden; the rest is inherited from
ContextAwareLRRewardModel/LRRewardModelV3/LRRewardModel.
"""
from __future__ import annotations

import numpy as np

from walt.rm.model.lr_model_context import ContextAwareLRRewardModel


class LRRewardModelV5(ContextAwareLRRewardModel):
    def _phi(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> np.ndarray:
        base_phi = super()._phi(question, sql, sql_context)
        key = self._context_key(sql_context)
        if key:
            context_vec = self._context_cache[key]
        else:
            context_vec = np.zeros(self.embedding_provider.dim, dtype=np.float32)
        return np.concatenate([base_phi, context_vec])
