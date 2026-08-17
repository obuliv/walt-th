"""Builds a severity-scored RM training dataset from the official Spider release: real
per-database SQLite schemas + gold SQL (DB/pair selection reused unchanged from
build_synth_dataset.py/spider_source.py), paired with sql_bad negatives generated
entirely from llama3.2's own real generation mistakes via a local Ollama server — no
rule-based corruptor (corrupt.generate_bad_candidates is deliberately not used here) and
no reason/severity assignment yet. This is stage 1 of a 2-stage pipeline; stage 2
(enhance_severity_dataset.py) categorizes every candidate into a reason and assigns a
0-5 severity via Claude, optionally backfilling more candidates when coverage is thin.

Runs the identical flow for both the trainval pool (train_spider.json/train_others.json)
and the val pool (dev.json) — no special-casing between them, so eval/evaluate.py's
RM-discrimination metric has sql_bad on val rows exactly like training does.

Output rows are pipeline-private JSON (not yet a valid Example/SQLBadCandidate shape):
sql_bad entries carry only {"sql", "matches_gold"} — reason/severity are added by
stage 2. sql_context is left empty in favor of sql_context_path (relative to
$DATA_PATH), same convention as build_synth_dataset.py; sql_context_clean (annotated
CREATE TABLE-only DDL) is always populated.

Setup: same one-time manual Spider download as build_synth_dataset.py (see its
docstring) — this script never touches build_synth_dataset.py's own
data/output/synth/ output.

Usage:
    python -m walt.rm.data.synth.build_severity_dataset --shortlist-only
    python -m walt.rm.data.synth.build_severity_dataset --train-count 30 --val-count 10 --n-ollama-candidates 2
    python -m walt.rm.data.synth.build_severity_dataset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import sqlglot
import sqlglot.expressions as exp
from dotenv import load_dotenv

from walt.agent.llm.base import BaseLLM
from walt.agent.llm.caching_llm import CachingLLM
from walt.agent.llm.ollama_llm import OllamaLLM
from walt.rm.data.synth.corrupt import verify_candidate
from walt.rm.data.synth.spider_source import (
    DBCandidate,
    SpiderPair,
    annotate_ddl,
    extract_ddl,
    group_by_db,
    load_pairs,
    select_dbs_for_target,
    shortlist_candidates,
)
from walt.utils.jsonl_io import open_jsonl

load_dotenv()

try:
    DATA_DIR = Path(os.environ["DATA_PATH"]).expanduser().resolve()
except KeyError as exc:
    raise RuntimeError(
        "DATA_PATH is not set. Add it to a .env file (see .env.example) or export it in your environment."
    ) from exc
SPIDER_DIR_DEFAULT = DATA_DIR / "spider"
OUTPUT_DIR = DATA_DIR / "output" / "synth_severity"
DEFAULT_OUTPUT = OUTPUT_DIR / "synth_severity_data.jsonl"
DEFAULT_QC_OUTPUT = OUTPUT_DIR / "synth_severity_qc.json"
DEFAULT_LLM_CACHE = DATA_DIR / "output" / "llm_cache.json"

TRAIN_FILES = ["train_spider.json", "train_others.json"]
VAL_FILES = ["dev.json"]

SOURCE = "synth_spider_severity"


def _is_select_shaped(sql: str) -> bool:
    try:
        return isinstance(sqlglot.parse_one(sql, dialect="sqlite"), exp.Select)
    except Exception:
        return False


PROGRESS_INTERVAL = 10  # print a status line every N rows -- this stage's real work
# (one Ollama call per candidate) is slow enough (~1-2s/row) that per-DB-only progress
# (the pre-existing "db_id: N/M verified" print, once a whole DB finishes) can go
# silent for tens of minutes on a low-DB-count run. Combined with main()'s
# sys.stdout.reconfigure(line_buffering=True), these lines are visible in real time
# even when this script's stdout is redirected to a file (e.g. a backgrounded run) --
# Python fully buffers stdout by default when it isn't a TTY, so without that fix nothing
# would appear until the process exits regardless of how often we print.


class ProgressTracker:
    """Tracks rows completed against the total pairs this pool will actually attempt
    (sum of every selected DB's verified pair count) -- NOT --train-count/--val-count
    directly, since select_dbs_for_target can (and typically does) accumulate more
    verified pairs than the target before build_pool downsamples at the very end."""

    def __init__(self, pool_name: str, total: int, interval: int = PROGRESS_INTERVAL):
        self.pool_name = pool_name
        self.total = total
        self.interval = interval
        self.done = 0
        self.start = time.perf_counter()

    def step(self) -> None:
        self.done += 1
        if self.done % self.interval != 0 and self.done != self.total:
            return
        elapsed = time.perf_counter() - self.start
        rate_per_min = self.done / elapsed * 60 if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta_min = remaining / rate_per_min if rate_per_min > 0 else float("inf")
        pct = 100 * self.done / self.total if self.total else 100.0
        print(
            f"  [{self.pool_name}] {self.done}/{self.total} rows ({pct:.1f}%) | "
            f"{rate_per_min:.1f} rows/min | elapsed {elapsed / 60:.1f}min | "
            f"ETA ~{eta_min:.0f}min"
        )


def generate_llama_bad_candidates(
    sql_good: str,
    question: str,
    schema_text: str,
    conn: sqlite3.Connection,
    llm: BaseLLM,
    n: int,
) -> list[dict[str, Any]]:
    """llama3.2-generated candidates for this row, deduped by exact text against
    sql_good, verified against `conn` (the same long-lived, per-DB connection every
    other pair in this DB reuses — see build_for_db). Every non-duplicate candidate is
    kept, including ones that execute to the same result as sql_good
    (matches_gold=True): unlike add_llama_negatives.py, this pipeline never drops
    those — severity=0 ("same result as sql_good") is a wanted label stage 2 assigns,
    and dropping the raw candidate here would make that label impossible to produce.

    A candidate that isn't SELECT-shaped is recorded as a real, un-executed mistake
    (matches_gold=False) rather than run against `conn` at all: Spider's sql_good is
    always a SELECT, so a non-SELECT answer is unambiguously wrong, and running it
    would risk mutating the shared connection (e.g. a hallucinated DELETE/UPDATE)
    every later pair in this DB depends on — unlike corrupt.py's own candidates,
    which are always SELECT-shaped AST mutations of the gold query by construction,
    this generator's candidates come from an LLM with no such guarantee.
    """
    candidates = llm.generate_candidates(question, schema_text, n)
    seen = {sql_good.strip()}
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        if not _is_select_shaped(candidate):
            results.append({"sql": candidate, "matches_gold": False})
            continue
        executes, matches_gold = verify_candidate(conn, sql_good, candidate)
        results.append({"sql": candidate, "matches_gold": bool(executes and matches_gold)})
    return results


def build_for_db(
    candidate: DBCandidate,
    verified: list[SpiderPair],
    conn: sqlite3.Connection,
    split: str,
    llm: BaseLLM,
    n_ollama: int,
    progress: ProgressTracker,
) -> tuple[list[dict[str, Any]], list[str]]:
    """conn is the already-open, fully-loaded connection select_dbs_for_target built for
    this DB (reused here for every candidate's verify_candidate check instead of
    rebuilding — see generate_llama_bad_candidates) — closed by the caller once this
    returns. Every verified pair produces exactly one output row, even if llama3.2
    happened to generate zero usable sql_bad candidates for it (sql_bad=[]) — stage 2
    can still backfill via new_candidates, so nothing is dropped here."""
    raw_ddl = extract_ddl(candidate.sqlite_path)
    annotated_ddl = annotate_ddl(candidate.sqlite_path, raw_ddl)
    schema_text = "\n".join(annotated_ddl)
    sql_context_path = str(candidate.sqlite_path.relative_to(DATA_DIR))

    rows: list[dict[str, Any]] = []
    for pair in verified:
        sql_bad = generate_llama_bad_candidates(pair.sql_good, pair.question, schema_text, conn, llm, n_ollama)
        rows.append(
            {
                "question": pair.question,
                "sql_good": pair.sql_good,
                "source": SOURCE,
                "split": split,
                "sql_context_clean": list(annotated_ddl),
                "sql_context_path": sql_context_path,
                "sql_context_valid": True,
                "sql_bad": sql_bad,
            }
        )
        progress.step()
    return rows, annotated_ddl


def build_pool(
    pool_name: str,
    spider_dir: Path,
    files: list[str],
    split: str,
    target: int,
    min_tables: int,
    max_tables: int,
    seed: int,
    llm: BaseLLM,
    n_ollama: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    pairs_by_db = group_by_db(load_pairs(spider_dir, files))
    candidates = shortlist_candidates(spider_dir, pairs_by_db, min_tables=min_tables, max_tables=max_tables)

    print(f"\n{pool_name} pool shortlist ({len(candidates)} DBs with {min_tables}-{max_tables} tables):")
    print(f"  {'db_id':<28}{'tables':>7}{'fks':>5}{'density':>9}{'pairs':>7}")
    for c in candidates:
        print(f"  {c.db_id:<28}{c.table_count:>7}{c.fk_count:>5}{c.fk_density:>9.2f}{c.pair_count:>7}")

    selected = select_dbs_for_target(candidates, pairs_by_db, target)
    total_attempted = sum(len(verified) for _, verified, _, _ in selected)

    rows: list[dict[str, Any]] = []
    schemas: dict[str, list[str]] = {}
    print(f"\n{pool_name} pool selected DBs (target {target}, {total_attempted} pairs to attempt):")
    progress = ProgressTracker(pool_name, total_attempted)
    for db_candidate, verified, _context, conn in selected:
        try:
            db_rows, annotated_ddl = build_for_db(db_candidate, verified, conn, split, llm, n_ollama, progress)
        finally:
            conn.close()
        print(f"  {db_candidate.db_id}: {len(db_rows)}/{db_candidate.pair_count} verified")
        rows.extend(db_rows)
        schemas[db_candidate.db_id] = annotated_ddl

    if len(rows) > target:
        rows = random.Random(seed).sample(rows, target)

    return rows, schemas


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """output_path may end in .gz to write gzip-compressed instead of plain text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_jsonl(output_path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_schema_files(schemas: dict[str, list[str]], output_dir: Path) -> None:
    schema_dir = output_dir / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for db_id, annotated_ddl in schemas.items():
        (schema_dir / f"{db_id}_schema.sql").write_text("\n\n".join(annotated_ddl) + "\n")


def write_qc_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    n_candidates = sum(len(r["sql_bad"]) for r in rows)
    n_matches_gold = sum(1 for r in rows for b in r["sql_bad"] if b["matches_gold"])
    n_rows_no_candidates = sum(1 for r in rows if not r["sql_bad"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_rows": len(rows),
                    "rows_with_no_candidates": n_rows_no_candidates,
                    "total_candidates": n_candidates,
                    "matches_gold": n_matches_gold,
                }
            },
            indent=2,
        )
    )


def main() -> None:
    # Python fully buffers stdout by default whenever it isn't a TTY (e.g. redirected
    # to a file, as happens for a backgrounded run) -- without this, every print()
    # below (including ProgressTracker's) sits in a buffer and never reaches the log
    # until the process exits, making a multi-hour run look silently stuck.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spider-dir", type=Path, default=SPIDER_DIR_DEFAULT)
    parser.add_argument("--train-count", type=int, default=2000, help="Verified pairs to sample -> split=trainval")
    parser.add_argument("--val-count", type=int, default=300, help="Verified pairs to sample -> split=val")
    parser.add_argument("--min-tables", type=int, default=3)
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qc-output", type=Path, default=DEFAULT_QC_OUTPUT)
    parser.add_argument("--shortlist-only", action="store_true", help="Print both pools' shortlists and exit")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument(
        "--n-ollama-candidates", type=int, default=5, help="How many llama3.2 candidates to generate per row"
    )
    parser.add_argument(
        "--ollama-concurrency",
        type=int,
        default=1,
        help=(
            "Concurrent Ollama requests per row's candidate generation (default 1 = "
            "sequential, today's behavior). Raising this ONLY helps once the Ollama "
            "server itself is configured to accept concurrent requests -- by default "
            "(and commonly even after a fresh install) Ollama's backend runs with a "
            "single request slot (llama.cpp's `-np 1`), so client-side concurrency "
            "alone just queues at the server with no speedup. To actually parallelize: "
            "(1) quit Ollama completely (menu bar icon -> Quit Ollama, or `killall "
            "Ollama; killall ollama` in a terminal -- this WILL interrupt any Ollama "
            "call in flight, including a currently-running instance of this script); "
            "(2) run `launchctl setenv OLLAMA_NUM_PARALLEL 4` (or another value) so "
            "the GUI app picks up the env var on next launch -- a plain shell `export` "
            "won't reach it, since Ollama.app isn't launched from your shell; (3) "
            "reopen Ollama; (4) verify via `ps aux | grep llama-server` that the "
            "relaunched process shows `-np 4` (or your chosen value) instead of `-np "
            "1`; (5) re-run this script with e.g. --ollama-concurrency 4. Memory is "
            "the real ceiling, not this flag -- each concurrent slot needs its own "
            "KV-cache allocation, so don't set this far above OLLAMA_NUM_PARALLEL."
        ),
    )
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=DEFAULT_LLM_CACHE,
        help="Shared cache sql_agent.py/evaluate.py use, keyed by (model, question, schema_context)",
    )
    parser.add_argument("--no-llm-cache", action="store_true")
    args = parser.parse_args()

    if args.shortlist_only:
        train_pairs_by_db = group_by_db(load_pairs(args.spider_dir, TRAIN_FILES))
        val_pairs_by_db = group_by_db(load_pairs(args.spider_dir, VAL_FILES))
        for pool_name, pairs_by_db in [("trainval", train_pairs_by_db), ("val", val_pairs_by_db)]:
            candidates = shortlist_candidates(args.spider_dir, pairs_by_db, args.min_tables, args.max_tables)
            print(f"\n{pool_name} pool shortlist ({len(candidates)} DBs with {args.min_tables}-{args.max_tables} tables):")
            print(f"  {'db_id':<28}{'tables':>7}{'fks':>5}{'density':>9}{'pairs':>7}")
            for c in candidates:
                print(f"  {c.db_id:<28}{c.table_count:>7}{c.fk_count:>5}{c.fk_density:>9.2f}{c.pair_count:>7}")
        return

    llm: BaseLLM = OllamaLLM(model=args.ollama_model, max_concurrency=args.ollama_concurrency)
    if not args.no_llm_cache:
        llm = CachingLLM(llm, cache_path=args.llm_cache)

    train_rows, train_schemas = build_pool(
        "trainval", args.spider_dir, TRAIN_FILES, "trainval", args.train_count,
        args.min_tables, args.max_tables, args.seed, llm, args.n_ollama_candidates,
    )
    val_rows, val_schemas = build_pool(
        "val", args.spider_dir, VAL_FILES, "val", args.val_count,
        args.min_tables, args.max_tables, args.seed, llm, args.n_ollama_candidates,
    )

    all_rows = train_rows + val_rows
    random.Random(args.seed).shuffle(all_rows)
    write_jsonl(all_rows, args.output)
    write_schema_files({**train_schemas, **val_schemas}, args.output.parent)
    write_qc_report(all_rows, args.qc_output)

    n_candidates = sum(len(r["sql_bad"]) for r in all_rows)
    n_matches_gold = sum(1 for r in all_rows for b in r["sql_bad"] if b["matches_gold"])
    n_rows_no_candidates = sum(1 for r in all_rows if not r["sql_bad"])
    print(f"\nWrote {len(all_rows)} examples to {args.output}")
    print(f"  trainval: {len(train_rows)}, val: {len(val_rows)}")
    print(
        f"sql_bad candidates: {n_candidates} generated across {len(all_rows) - n_rows_no_candidates}/{len(all_rows)} "
        f"rows ({n_rows_no_candidates} row(s) got none from llama3.2 — stage 2 can still backfill), "
        f"{n_matches_gold} same-result-as-good (will become severity=0 once reviewed by enhance_severity_dataset.py)"
    )
    print(f"Wrote {len(train_schemas) + len(val_schemas)} annotated schema file(s) to {args.output.parent / 'schemas'}")


if __name__ == "__main__":
    main()
