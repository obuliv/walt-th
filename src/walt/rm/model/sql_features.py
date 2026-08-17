"""Handcrafted SQL features — cheap, interpretable signals that don't require an
embedding model, meant to complement (not replace) embedding-based features."""
from __future__ import annotations

import functools
import logging
import re
from typing import Optional, Sequence

import sqlglot
import sqlglot.expressions as exp
from sqlglot.optimizer.qualify import qualify

from walt.utils.sql_exec import run_sql

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


def is_schema_valid(sql: str, sql_context: Sequence[str]) -> bool:
    """Whether `sql` actually executes (no error) against `sql_context` — catches what
    is_sql_valid's pure syntax check can't: a candidate that parses fine but references
    a table/column that doesn't exist in the schema, wrong arity, a type SQLite rejects,
    etc. `sql_context` is expected to be the clean, data-free schema (CREATE TABLE/etc
    only, see walt.utils.sql_exec.clean_context()) — the same thing already threaded
    through score()/rank() everywhere else post clean-context — not the full context
    with sample rows, so a legitimate query can't spuriously fail on a data-only
    constraint (e.g. a UNIQUE conflict on inserted rows) that has nothing to do with
    whether the query itself is well-formed for this schema.

    No context at all (e.g. sql_good is itself DDL) is treated as valid/neutral — same
    convention as the context-less fallback for lr_model_context.py's cosine_sim
    feature, so a missing context never counts against a candidate.

    Cached on (sql, sql_context) since the same pair is scored repeatedly across
    pairwise comparisons/CV folds. Callers pass sql_context as either a list or a
    tuple (sql_agent.py's code path does both, depending on caller) — coerced to a
    tuple here, before the cache boundary, since lru_cache needs a hashable key and a
    list isn't one."""
    return _is_schema_valid_cached(sql, tuple(sql_context))


@functools.lru_cache(maxsize=None)
def _is_schema_valid_cached(sql: str, sql_context: tuple[str, ...]) -> bool:
    if not sql_context:
        return True
    return run_sql(sql_context, sql).success


def is_schema_valid_static(sql: str, sql_context: Sequence[str]) -> bool:
    """Static counterpart to is_schema_valid: same "does sql reference a real
    table/column for this schema" question, answered via sqlglot AST inspection +
    schema-aware qualification instead of actually executing sql. No SQLite engine
    involved — table/column names are checked against a {table: {column}} map built
    by parsing sql_context's CREATE TABLE statements.

    Two checks, both required: (1) every exp.Table node's name must be a table sqlglot
    parsed out of sql_context — sqlglot's own qualify() doesn't reject unknown tables
    on its own (confirmed directly: a JOIN against a nonexistent table qualifies
    without error unless a bare, unqualified reference to one of its columns forces
    resolution), so this is checked manually first; (2) sqlglot.optimizer.qualify.qualify
    (infer_schema=False, so it never guesses a schema for a table missing from the map)
    is run and any exception (unresolvable column, ambiguous reference, etc.) counts as
    invalid.

    This is intentionally weaker than is_schema_valid: it can't see wrong function
    arity, type mismatches SQLite itself would reject, or anything else that only
    surfaces by actually running the query — pure identifier resolution only.

    Same context convention as is_schema_valid: sql_context is the clean, data-free
    schema (CREATE TABLE only); no context at all is treated as valid/neutral. Cached
    on (sql, sql_context) since the same pair is scored repeatedly across pairwise
    comparisons/CV folds."""
    return _is_schema_valid_static_cached(sql, tuple(sql_context))


@functools.lru_cache(maxsize=None)
def _is_schema_valid_static_cached(sql: str, sql_context: tuple[str, ...]) -> bool:
    if not sql_context:
        return True
    schema = _build_schema_map(sql_context)
    if not schema:
        return True
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    known_tables = {name.lower() for name in schema}
    for table_node in tree.find_all(exp.Table):
        if table_node.name.lower() not in known_tables:
            return False
    try:
        qualify(tree, schema=schema, dialect="sqlite", infer_schema=False)
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=None)
def _build_schema_map(sql_context: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Parses sql_context's CREATE TABLE statements into a {table: {column: type}} map
    for qualify()'s schema argument. Statements that aren't CREATE TABLE (or that fail
    to parse) are silently skipped — sql_context can carry other DDL/comment noise."""
    schema: dict[str, dict[str, str]] = {}
    for statement in sql_context:
        try:
            parsed = sqlglot.parse_one(statement, dialect="sqlite")
        except Exception:
            continue
        table_node = parsed.find(exp.Table)
        if table_node is None:
            continue
        columns = {
            col.name: (col.args.get("kind").sql() if col.args.get("kind") else "TEXT")
            for col in parsed.find_all(exp.ColumnDef)
        }
        if columns:
            schema[table_node.name] = columns
    return schema


@functools.lru_cache(maxsize=None)
def _parse_cached(sql: str) -> Optional[exp.Expression]:
    """Shared parse used by every feature below, so a given sql string is only ever
    parsed once regardless of how many of these features are computed for it. sqlite
    dialect, matching corrupt.py's convention for real (generated) candidate SQL --
    is_sql_valid above deliberately uses no explicit dialect since it's checking
    generic syntax validity, not executing against a real sqlite schema."""
    try:
        return sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None


def sql_length_ratio(question: str, sql: str) -> float:
    """len(sql) / len(question) -- unlike len(sql) - len(question), this does not
    reduce to plain len(sql) under this project's pairwise-difference training
    objective (see lr_model.py's fit()): the ratio is implicitly scaled by
    1/len(question), a per-question factor that a plain difference cancels away.
    No caching needed, this is pure arithmetic."""
    return len(sql) / max(len(question), 1)


def _mask_leaf(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Column):
        return exp.column("COL")
    if isinstance(node, exp.Table):
        return exp.to_table("TAB")
    if isinstance(node, exp.Literal):
        return exp.Literal.string("VAL")
    return node


@functools.lru_cache(maxsize=None)
def split_sql_commands_and_args(sql: str) -> tuple[str, str]:
    """Splits `sql` into two texts: `commands_text`, the structural/keyword skeleton
    (SELECT/FROM/WHERE/JOIN/GROUP BY, aggregate function names, operators,
    DISTINCT/ORDER BY/LIMIT) with every table/column name and literal value replaced
    by a placeholder (COL/TAB/VAL) -- still valid, parseable SQL text, so an
    embedding model trained on real code sees something structurally real rather
    than a bag of bare keywords; and `args_text`, the original table/column names and
    literal values it referenced, space-joined in AST traversal order -- "what the
    query talks about" vs "what it says about it".

    Falls back to `(sql, sql)` when `sql` fails to parse -- a hallucinated/malformed
    candidate (already flagged separately by is_sql_valid) still produces something
    embeddable rather than crashing a caller's fit()/score()."""
    tree = _parse_cached(sql)
    if tree is None:
        return sql, sql
    commands_tree = tree.copy().transform(_mask_leaf)
    commands_text = commands_tree.sql(dialect="sqlite")
    args = [n.sql(dialect="sqlite") for n in tree.find_all(exp.Column, exp.Table, exp.Literal)]
    args_text = " ".join(args)
    return commands_text, args_text


def sql_structural_counts(sql: str) -> tuple[float, float, float, float, float, float]:
    """Fixed-order (num_joins, num_where_conditions, has_star, has_distinct,
    has_group_by, num_select_columns) -- cheap, execution-free counts/flags aimed
    directly at this project's sql_bad mistake taxonomy: num_joins/misjoined_tables,
    num_where_conditions/missing_filters, has_star/unsafe_patterns,
    has_group_by+num_select_columns/wrong_aggregation. Piggybacks on the same
    _parse_cached() parse split_sql_commands_and_args() already does.

    Falls back to an all-zero tuple when `sql` fails to parse -- treated as
    structurally empty (is_sql_valid already flags parse failure separately)."""
    tree = _parse_cached(sql)
    if tree is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num_joins = float(len(list(tree.find_all(exp.Join))))

    where = tree.find(exp.Where)
    if where is None:
        num_where_conditions = 0.0
    else:
        num_where_conditions = float(len(list(where.find_all(exp.And))) + 1)

    # Compound queries (UNION/INTERSECT/EXCEPT) have a non-Select top-level node --
    # fall back to the first Select branch in traversal order so these three still get
    # a meaningful reading instead of silently zeroing out.
    select_node = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projections = select_node.expressions if select_node is not None else []
    has_star = float(
        any(isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)) for e in projections)
    )
    has_distinct = float(bool(select_node.args.get("distinct"))) if select_node is not None else 0.0
    has_group_by = float(tree.find(exp.Group) is not None)
    num_select_columns = float(len(projections))

    return num_joins, num_where_conditions, has_star, has_distinct, has_group_by, num_select_columns


_WORD_RE = re.compile(r"\w+")


def question_arg_overlap(question: str, sql: str) -> float:
    """Jaccard similarity between the lowercased word-tokens of `question` and of
    `args_text` (split_sql_commands_and_args's table/column-name/literal-value half)
    -- a cheap complement to an embedding cosine similarity that catches exact name
    matches embeddings can blur (e.g. "customers" vs "customer"). 0.0 if either side
    tokenizes to nothing. No stopword removal/stemming, kept simple."""
    _, args_text = split_sql_commands_and_args(sql)
    q_tokens = set(_WORD_RE.findall(question.lower()))
    a_tokens = set(_WORD_RE.findall(args_text.lower()))
    union = q_tokens | a_tokens
    if not union:
        return 0.0
    return len(q_tokens & a_tokens) / len(union)
