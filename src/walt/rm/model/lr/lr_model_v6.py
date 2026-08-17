"""V6 of the pairwise LR reward model: V3's phi plus a schema-validity feature — does
`sql` actually execute (no error) against sql_context's schema? Targets a gap noted
when wiring is_sql_valid (pure syntax, sql_features.py) into the agent: a candidate can
parse fine yet still reference a table/column that doesn't exist in the schema it's
supposed to run against, which pure syntax checking can't catch but is_schema_valid can
(via a real execution attempt — see sql_features.py). Unlike V4/V5's context-embedding
approach, this needs no embedding call at all, so it's cheap the same way V3's
is_sql_valid is.

phi_v6(question, sql, sql_context) = concat(phi_v3(question, sql), [is_schema_valid(sql, sql_context)])
"""
from __future__ import annotations

import numpy as np

from walt.rm.model.lr.lr_model_v3 import LRRewardModelV3
from walt.rm.model.sql_features import is_schema_valid


class LRRewardModelV6(LRRewardModelV3):
    def _phi(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> np.ndarray:
        base_phi = super()._phi(question, sql, sql_context)
        return np.concatenate([base_phi, [1.0 if is_schema_valid(sql, sql_context) else 0.0]])
