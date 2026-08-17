"""Rule-based, sqlglot-AST corruption engine for generating sql_bad negatives — no LLM
involved, deterministic given a seeded random.Random. Each corrupt_* function takes a
parsed gold exp.Select and returns a structurally mutated exp.Select | None (None = this
category doesn't structurally apply to this query, e.g. no WHERE/JOIN/aggregation).
"""
from __future__ import annotations

import functools
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass

import sqlglot
import sqlglot.expressions as exp

from walt.rm.data.synth.schema import Schema
from walt.utils.sql_exec import execute_on_connection

REASON_NAMES = ("missing_filters", "wrong_aggregation", "unsafe_patterns", "misjoined_tables", "compound")

AGG_CLASSES: dict[str, type[exp.AggFunc]] = {
    "SUM": exp.Sum,
    "COUNT": exp.Count,
    "AVG": exp.Avg,
    "MAX": exp.Max,
    "MIN": exp.Min,
}


def corrupt_missing_filters(tree: exp.Select, rng: random.Random) -> exp.Select | None:
    tree = tree.copy()
    where = tree.args.get("where")
    if where is not None:
        conjuncts = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
        if len(conjuncts) > 1:
            drop = rng.choice(conjuncts)
            remaining = [c.copy() for c in conjuncts if c is not drop]
            where.set("this", functools.reduce(lambda a, b: exp.and_(a, b), remaining))
        else:
            tree.set("where", None)
        return tree
    if tree.args.get("having") is not None:
        tree.set("having", None)
        return tree
    return None


def corrupt_wrong_aggregation(tree: exp.Select, rng: random.Random) -> exp.Select | None:
    tree = tree.copy()
    aggs = list(tree.find_all(exp.AggFunc))
    if not aggs:
        return None
    agg = rng.choice(aggs)
    current = type(agg).__name__.upper()
    if rng.random() < 0.5:
        agg.replace(agg.this or exp.Star())
    else:
        choices = [name for name in AGG_CLASSES if name != current] or list(AGG_CLASSES)
        agg.replace(AGG_CLASSES[rng.choice(choices)](this=agg.this))
    return tree


def _has_star_select(tree: exp.Select) -> bool:
    return len(tree.expressions) == 1 and isinstance(tree.expressions[0], exp.Star)


def corrupt_unsafe_patterns(tree: exp.Select, rng: random.Random) -> exp.Select | None:
    tree = tree.copy()
    options = []
    if tree.expressions and not _has_star_select(tree):
        options.append("star")
    on_joins = [j for j in tree.args.get("joins", []) if j.args.get("on") is not None]
    if on_joins:
        options.append("cross_join")
    if not options:
        return None
    if rng.choice(options) == "star":
        tree.set("expressions", [exp.Star()])
    else:
        join = rng.choice(on_joins)
        join.set("on", None)
        join.set("kind", "CROSS")
    return tree


def _infer_table(tree: exp.Select, col: exp.Column) -> str | None:
    if col.table:
        return col.table
    from_ = tree.args.get("from_")
    if from_ is not None and isinstance(from_.this, exp.Table):
        return from_.this.name
    return None


def corrupt_misjoined_tables(tree: exp.Select, schema: Schema, rng: random.Random) -> exp.Select | None:
    tree = tree.copy()
    on_joins = [j for j in tree.args.get("joins", []) if j.args.get("on") is not None]
    if not on_joins:
        return None
    join = rng.choice(on_joins)
    columns = list(join.args["on"].find_all(exp.Column))
    if columns and rng.random() < 0.5:
        col = rng.choice(columns)
        table = _infer_table(tree, col)
        if table:
            alts = [c.name for c in schema.columns_of(table) if c.name != col.name]
            if alts:
                col.set("this", exp.to_identifier(rng.choice(alts)))
                return tree

    join_table = join.this if isinstance(join.this, exp.Table) else None
    if join_table is None:
        return None
    alts = [t for t in schema.table_names() if t != join_table.name]
    if not alts:
        return None
    join_table.set("this", exp.to_identifier(rng.choice(alts)))
    return tree


def applicable_categories(tree: exp.Select, schema: Schema) -> list[str]:
    applicable = []
    if tree.args.get("where") is not None or tree.args.get("having") is not None:
        applicable.append("missing_filters")
    if list(tree.find_all(exp.AggFunc)):
        applicable.append("wrong_aggregation")
    has_on_join = any(j.args.get("on") is not None for j in tree.args.get("joins", []))
    if not _has_star_select(tree) or has_on_join:
        applicable.append("unsafe_patterns")
    if has_on_join:
        applicable.append("misjoined_tables")
    return applicable


_BASE_CORRUPTORS = {
    "missing_filters": lambda tree, schema, rng: corrupt_missing_filters(tree, rng),
    "wrong_aggregation": lambda tree, schema, rng: corrupt_wrong_aggregation(tree, rng),
    "unsafe_patterns": lambda tree, schema, rng: corrupt_unsafe_patterns(tree, rng),
    "misjoined_tables": corrupt_misjoined_tables,
}


def corrupt_compound(tree: exp.Select, schema: Schema, rng: random.Random) -> exp.Select | None:
    applicable = applicable_categories(tree, schema)
    if not applicable:
        return None
    if len(applicable) >= 2:
        cat_a, cat_b = rng.sample(applicable, 2)
    else:
        cat_a = cat_b = applicable[0]
    mutated = _BASE_CORRUPTORS[cat_a](tree, schema, rng)
    if mutated is None:
        return None
    result = _BASE_CORRUPTORS[cat_b](mutated, schema, rng)
    return result if result is not None else mutated


CORRUPTORS = {**_BASE_CORRUPTORS, "compound": corrupt_compound}


def verify_candidate(conn: sqlite3.Connection, sql_good: str, candidate_sql: str) -> tuple[bool, bool]:
    """(executes, matches_gold), both checked against an already-open connection (reused
    across every candidate for a DB — see generate_bad_candidates) rather than rebuilding
    an in-memory DB from context statements per check. matches_gold uses Counter equality
    (order-insensitive, duplicate-row-aware — same rationale as
    sql_exec.capture_db_state's use of Counter over a plain set) — True means this
    candidate "accidentally" returns the same result as sql_good, a weak
    (non-distinguishing) negative that's still kept, just flagged."""
    bad = execute_on_connection(conn, candidate_sql)
    if not bad.success:
        return False, False
    good = execute_on_connection(conn, sql_good)
    if not good.success:
        return True, False
    return True, Counter(bad.rows or ()) == Counter(good.rows or ())


@dataclass(frozen=True)
class GeneratedBad:
    sql: str
    reason: str
    flagged_weak: bool


def generate_bad_candidates(
    sql_good: str, schema: Schema, conn: sqlite3.Connection, rng: random.Random
) -> list[GeneratedBad]:
    try:
        tree = sqlglot.parse_one(sql_good, dialect="sqlite")
    except Exception:
        return []
    if not isinstance(tree, exp.Select):
        return []

    results = []
    seen_sql = {sql_good.strip()}
    for reason in REASON_NAMES:
        mutated = CORRUPTORS[reason](tree, schema, rng)
        if mutated is None:
            continue
        candidate_sql = mutated.sql(dialect="sqlite")
        if candidate_sql.strip() in seen_sql:
            continue
        seen_sql.add(candidate_sql.strip())
        executes, matches_gold = verify_candidate(conn, sql_good, candidate_sql)
        results.append(GeneratedBad(sql=candidate_sql, reason=reason, flagged_weak=executes and matches_gold))
    return results
