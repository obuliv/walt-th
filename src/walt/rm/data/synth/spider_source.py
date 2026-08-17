"""Loads the official Spider release from a local directory (no network fetch — the real
per-database SQLite corpus is only distributed via a Google Drive link on
https://yale-lily.github.io/spider, not something worth automating; see build_synth_dataset.py
docstring for the manual setup step), scores/selects databases, and extracts+annotates DDL.

Expected layout under spider_dir (the official release's own layout):
    database/<db_id>/<db_id>.sqlite
    train_spider.json, train_others.json, dev.json  (each row: db_id/question/query)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sqlglot
import sqlglot.expressions as exp

from walt.rm.data.synth.schema import Schema, build_schema
from walt.utils.sql_exec import build_connection, execute_on_connection, load_context_from_sqlite, normalize_sql


@dataclass(frozen=True)
class SpiderPair:
    db_id: str
    question: str
    sql_good: str


def load_pairs(spider_dir: Path, files: Sequence[str]) -> list[SpiderPair]:
    pairs = []
    for filename in files:
        with (spider_dir / filename).open(encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            question = row["question"].strip()
            sql_good = normalize_sql(row["query"].strip())
            if not question or not sql_good:
                continue
            pairs.append(SpiderPair(db_id=row["db_id"], question=question, sql_good=sql_good))
    return pairs


def group_by_db(pairs: list[SpiderPair]) -> dict[str, list[SpiderPair]]:
    grouped: dict[str, list[SpiderPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.db_id, []).append(pair)
    return grouped


def discover_local_dbs(spider_dir: Path) -> dict[str, Path]:
    database_dir = spider_dir / "database"
    if not database_dir.is_dir():
        raise FileNotFoundError(
            f"{database_dir} not found. Download the official Spider release from "
            f"https://yale-lily.github.io/spider and place its contents (including the "
            f"database/ folder) under {spider_dir}."
        )
    dbs = {}
    for db_dir in sorted(database_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        sqlite_path = db_dir / f"{db_dir.name}.sqlite"
        if sqlite_path.exists():
            dbs[db_dir.name] = sqlite_path
    return dbs


def extract_ddl(sqlite_path: Path) -> list[str]:
    """The live DB's own DDL (guaranteed accurate/executable), not tables.json or a
    possibly-stale bundled schema.sql."""
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


MAX_SQLITE_SIZE_BYTES = 5_000_000


@dataclass(frozen=True)
class DBCandidate:
    db_id: str
    sqlite_path: Path
    table_count: int
    fk_count: int
    fk_density: float
    pair_count: int


def shortlist_candidates(
    spider_dir: Path,
    pairs_by_db: dict[str, list[SpiderPair]],
    min_tables: int = 3,
    max_tables: int = 8,
) -> list[DBCandidate]:
    local_dbs = discover_local_dbs(spider_dir)
    candidates = []
    for db_id in sorted(set(local_dbs) & set(pairs_by_db)):
        sqlite_path = local_dbs[db_id]
        if sqlite_path.stat().st_size > MAX_SQLITE_SIZE_BYTES:
            # A handful of Spider DBs (soccer_1: 317MB/196K rows, wta_1: 105MB,
            # baseball_1: 30MB — everything else is under 4MB) carry a huge fact table
            # unrelated to schema/join complexity; reconstructing + repeatedly querying
            # one in-memory (no indexes) turns a single DB into minutes of work for a
            # handful of pairs. Cheap file-size check, no need to open the DB at all.
            continue
        schema = build_schema(extract_ddl(sqlite_path))
        table_count = len(schema.tables)
        if not (min_tables <= table_count <= max_tables):
            continue
        fk_count = len(schema.foreign_keys)
        candidates.append(
            DBCandidate(
                db_id=db_id,
                sqlite_path=sqlite_path,
                table_count=table_count,
                fk_count=fk_count,
                fk_density=fk_count / table_count if table_count else 0.0,
                pair_count=len(pairs_by_db[db_id]),
            )
        )
    candidates.sort(key=lambda c: (-c.fk_density, -c.pair_count))
    return candidates


def verify_pairs(
    sqlite_path: Path, pairs: list[SpiderPair]
) -> tuple[list[SpiderPair], tuple[str, ...], sqlite3.Connection | None]:
    """Verifies each pair's sql_good actually executes (successfully, non-empty result)
    against the real DB. Loads the context and builds one in-memory connection *once*
    (via load_context_from_sqlite + build_connection), reused across every pair check —
    rebuilding a fresh in-memory DB per pair (as a naive run_sql(context, sql) loop would)
    is O(n_pairs * n_statements) and was the actual cost behind a 7-minute run before this
    fix. Returns (verified pairs, context, connection) — the caller (build_for_db) reuses
    the same open connection for sql_bad candidate verification too, and is responsible
    for closing it. Connection is None if the context itself failed to build (a genuine
    data issue in this particular DB) — verified is empty in that case, not raised, so one
    bad DB doesn't abort DB selection for the whole run."""
    context = load_context_from_sqlite(sqlite_path)
    try:
        conn = build_connection(context)
    except sqlite3.Error as exc:
        print(f"  WARNING: skipping {sqlite_path.stem} — context failed to build: {exc}")
        return [], context, None
    verified = []
    for pair in pairs:
        execution = execute_on_connection(conn, pair.sql_good)
        if execution.success and execution.rows:
            verified.append(pair)
    return verified, context, conn


def select_dbs_for_target(
    candidates: list[DBCandidate], pairs_by_db: dict[str, list[SpiderPair]], target: int
) -> list[tuple[DBCandidate, list[SpiderPair], tuple[str, ...], sqlite3.Connection]]:
    """Walks the pre-sorted shortlist in order, verifying each candidate's pairs and
    accumulating verified count, stopping as soon as the target is reached — only the DBs
    actually needed are processed (and returned), not the full shortlist. Each returned
    tuple's connection stays open (build_for_db reuses it for sql_bad verification, then
    closes it) — candidates with zero verified pairs have theirs closed here instead."""
    selected = []
    total_verified = 0
    for candidate in candidates:
        verified, context, conn = verify_pairs(candidate.sqlite_path, pairs_by_db[candidate.db_id])
        if not verified:
            if conn is not None:
                conn.close()
            continue
        selected.append((candidate, verified, context, conn))
        total_verified += len(verified)
        if total_verified >= target:
            return selected
    raise ValueError(
        f"Only covered {total_verified}/{target} pairs across all {len(candidates)} qualifying "
        f"DBs — widen --max-tables (or lower --min-tables) to admit more candidate databases."
    )


CATEGORICAL_MAX_DISTINCT = 10
CATEGORICAL_MIN_ROWS = 20
CATEGORICAL_MAX_RATIO = 0.05

_TEXT_TYPES = {"TEXT", "VARCHAR", "CHAR", "NVARCHAR", "NCHAR", "CLOB"}


def categorical_values(sqlite_path: Path, table: str, column: str) -> tuple[str, ...] | None:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        total_rows = cursor.fetchone()[0]
        if total_rows == 0:
            return None
        cursor.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL')
        values = sorted(str(row[0]) for row in cursor.fetchall())
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    if not values or len(values) > CATEGORICAL_MAX_DISTINCT:
        return None
    if total_rows > CATEGORICAL_MIN_ROWS and len(values) / total_rows > CATEGORICAL_MAX_RATIO:
        return None
    return tuple(values)


def annotate_ddl(sqlite_path: Path, create_statements: list[str]) -> list[str]:
    """Attaches a "one of: ..." comment to every TEXT-like column with a small, fixed set
    of real distinct values — via sqlglot's own AST/printer (col.comments), not text/regex
    matching, so column-boundary detection can't get confused by unusual DDL formatting."""
    annotated = []
    for statement in create_statements:
        try:
            tree = sqlglot.parse_one(statement, dialect="sqlite")
        except Exception:
            annotated.append(statement)
            continue
        if not isinstance(tree, exp.Create) or tree.kind != "TABLE":
            annotated.append(statement)
            continue
        table_expr = tree.this
        if not isinstance(table_expr, exp.Schema) or not isinstance(table_expr.this, exp.Table):
            annotated.append(statement)
            continue
        table_name = table_expr.this.this.name

        for col_def in table_expr.expressions:
            if not isinstance(col_def, exp.ColumnDef):
                continue
            kind = col_def.args.get("kind")
            type_name = kind.this.name if kind is not None and kind.this is not None else ""
            if type_name not in _TEXT_TYPES:
                continue
            values = categorical_values(sqlite_path, table_name, col_def.this.name)
            if values is not None:
                quoted = ", ".join(repr(v) for v in values)
                col_def.comments = [f"one of: {quoted}"]

        annotated.append(tree.sql(dialect="sqlite", pretty=True))
    return annotated


def schema_for_db(sqlite_path: Path) -> Schema:
    return build_schema(extract_ddl(sqlite_path))
