"""Handcrafted SQL features — cheap, interpretable signals that don't require an
embedding model, meant to complement (not replace) embedding-based features."""
from __future__ import annotations

import functools
import logging

import sqlglot
import sqlglot.expressions as exp

# sqlglot logs (not warnings.warn) when it falls back to parsing an unmodeled statement
# shape as a generic Command — is_sql_valid() treats that fallback as a signal (see
# below), so the log line is expected/handled, not something to surface to the user.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


@functools.lru_cache(maxsize=None)
def is_sql_valid(sql: str) -> bool:
    """Whether `sql` parses as syntactically valid SQL, via sqlglot (no schema/database
    needed — this is pure syntax, not "does this table/column exist").

    sqlglot falls back to a generic `Command` node (rather than raising) for statement
    shapes it doesn't fully model — mostly vendor-specific DDL/DCL (MySQL's `ALTER TABLE
    ... MODIFY COLUMN`, `GRANT`/`REVOKE`, `RENAME TABLE`, etc., which appear throughout
    this dataset's negatives). Treating that fallback as "invalid too" isn't perfectly
    accurate (some of those statements are fine, just unmodeled), but it substantially
    improves recall on real mistakes sqlglot's lenient parser would otherwise wave
    through — e.g. "GRANT SELECT ON employees FROM john" (should be "TO", not "FROM")
    parses as a Command without error under strict-exception-only checking.

    Cached since the same candidate SQL string is scored repeatedly across pairwise
    comparisons/CV folds."""
    try:
        return not isinstance(sqlglot.parse_one(sql), exp.Command)
    except Exception:
        return False
