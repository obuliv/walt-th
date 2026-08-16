"""Builds a pairwise RM training dataset from the official Spider release: real
per-database SQLite schemas + gold SQL, verified locally, paired with deterministic
sqlglot-AST sql_bad corruptions (missing_filters/wrong_aggregation/unsafe_patterns/
misjoined_tables/compound — no LLM involved).

Setup (one-time, manual): download the official Spider release from
https://yale-lily.github.io/spider (the SQLite database corpus is only distributed via a
Google Drive link there — not something worth automating) and place its contents,
including the database/ folder and train_spider.json/train_others.json/dev.json, under
$DATA_PATH/spider (i.e. data/spider/ by default).

trainval rows are sourced from train_spider.json + train_others.json's 146 databases;
val rows are sourced from dev.json's 20 databases — Spider's own train/dev split, which
uses fully disjoint database sets (confirmed empirically), so val rows are held out on
entirely different schemas, not just different rows.

Output rows leave sql_context empty and instead carry sql_context_path (relative to
$DATA_PATH, e.g. "spider/database/restaurants/restaurants.sqlite") pointing at the real
.sqlite file — see walt.utils.sql_exec.execute_with_context, which resolves this
transparently wherever an Example's context is executed against. sql_context_clean
(annotated CREATE TABLE-only DDL) is always populated.

Usage:
    python -m walt.rm.data.synth.build_synth_dataset --shortlist-only
    python -m walt.rm.data.synth.build_synth_dataset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from walt.rm.data.base import Example, SQLBadCandidate
from walt.rm.data.synth.corrupt import generate_bad_candidates
from walt.rm.data.synth.schema import build_schema
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

load_dotenv()

try:
    DATA_DIR = Path(os.environ["DATA_PATH"]).expanduser().resolve()
except KeyError as exc:
    raise RuntimeError(
        "DATA_PATH is not set. Add it to a .env file (see .env.example) or export it in your environment."
    ) from exc
SPIDER_DIR_DEFAULT = DATA_DIR / "spider"
OUTPUT_DIR = DATA_DIR / "output" / "synth"
DEFAULT_OUTPUT = OUTPUT_DIR / "synth_data.jsonl"
DEFAULT_QC_OUTPUT = OUTPUT_DIR / "synth_bad_candidate_qc.json"

TRAIN_FILES = ["train_spider.json", "train_others.json"]
VAL_FILES = ["dev.json"]


def build_for_db(
    db_id: str,
    candidate: DBCandidate,
    verified: list[SpiderPair],
    conn: sqlite3.Connection,
    split: str,
    seed: int,
) -> tuple[list[Example], list[dict], list[str]]:
    """conn is the already-open, fully-loaded connection select_dbs_for_target built for
    this DB (reused here for every sql_bad candidate check instead of rebuilding — see
    corrupt.generate_bad_candidates) — closed by the caller once this returns."""
    raw_ddl = extract_ddl(candidate.sqlite_path)
    annotated_ddl = annotate_ddl(candidate.sqlite_path, raw_ddl)
    schema = build_schema(raw_ddl)
    sql_context_path = str(candidate.sqlite_path.relative_to(DATA_DIR))

    rng = random.Random(f"{seed}:{db_id}")
    examples = []
    qc_rows = []
    for pair in verified:
        bads = generate_bad_candidates(pair.sql_good, schema, conn, rng)
        sql_bad = tuple(SQLBadCandidate(sql=b.sql, reason=b.reason) for b in bads)
        try:
            examples.append(
                Example(
                    question=pair.question,
                    sql_good=pair.sql_good,
                    source="synth_spider",
                    sql_bad=sql_bad,
                    sql_context=(),
                    sql_context_clean=tuple(annotated_ddl),
                    sql_context_path=sql_context_path,
                    sql_context_valid=True,
                    split=split,
                )
            )
        except ValueError:
            # sql_good happens to duplicate a generated sql_bad candidate's text (rare —
            # a corruption rendered back to exactly the same SQL) — same skip convention
            # as load_examples() elsewhere in the pipeline, just avoided before ever
            # reaching disk instead of warning on read.
            continue
        qc_rows.extend(
            {"db_id": db_id, "question": pair.question, "sql_good": pair.sql_good, "sql_bad": b.sql, "reason": b.reason}
            for b in bads
            if b.flagged_weak
        )
    return examples, qc_rows, annotated_ddl


def build_pool(
    pool_name: str,
    spider_dir: Path,
    files: list[str],
    split: str,
    target: int,
    min_tables: int,
    max_tables: int,
    seed: int,
) -> tuple[list[Example], list[dict], dict[str, list[str]]]:
    pairs_by_db = group_by_db(load_pairs(spider_dir, files))
    candidates = shortlist_candidates(spider_dir, pairs_by_db, min_tables=min_tables, max_tables=max_tables)

    print(f"\n{pool_name} pool shortlist ({len(candidates)} DBs with {min_tables}-{max_tables} tables):")
    print(f"  {'db_id':<28}{'tables':>7}{'fks':>5}{'density':>9}{'pairs':>7}")
    for c in candidates:
        print(f"  {c.db_id:<28}{c.table_count:>7}{c.fk_count:>5}{c.fk_density:>9.2f}{c.pair_count:>7}")

    selected = select_dbs_for_target(candidates, pairs_by_db, target)

    examples: list[Example] = []
    qc_rows: list[dict] = []
    schemas: dict[str, list[str]] = {}
    print(f"\n{pool_name} pool selected DBs (target {target}):")
    for candidate, verified, _context, conn in selected:
        try:
            db_examples, db_qc_rows, annotated_ddl = build_for_db(
                candidate.db_id, candidate, verified, conn, split, seed
            )
        finally:
            conn.close()
        print(f"  {candidate.db_id}: {len(db_examples)}/{candidate.pair_count} verified")
        examples.extend(db_examples)
        qc_rows.extend(db_qc_rows)
        schemas[candidate.db_id] = annotated_ddl

    if len(examples) > target:
        examples = random.Random(seed).sample(examples, target)

    return examples, qc_rows, schemas


def write_jsonl(examples: list[Example], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")


def write_schema_files(schemas: dict[str, list[str]], output_dir: Path) -> None:
    schema_dir = output_dir / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for db_id, annotated_ddl in schemas.items():
        (schema_dir / f"{db_id}_schema.sql").write_text("\n\n".join(annotated_ddl) + "\n")


def write_qc_report(qc_rows: list[dict], total_candidates: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": {"total_candidates": total_candidates, "flagged_weak": len(qc_rows)},
                "flagged": qc_rows,
            },
            indent=2,
        )
    )


def main() -> None:
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

    train_examples, train_qc, train_schemas = build_pool(
        "trainval", args.spider_dir, TRAIN_FILES, "trainval", args.train_count,
        args.min_tables, args.max_tables, args.seed,
    )
    val_examples, val_qc, val_schemas = build_pool(
        "val", args.spider_dir, VAL_FILES, "val", args.val_count,
        args.min_tables, args.max_tables, args.seed,
    )

    all_examples = train_examples + val_examples
    random.Random(args.seed).shuffle(all_examples)
    write_jsonl(all_examples, args.output)
    write_schema_files({**train_schemas, **val_schemas}, args.output.parent)

    all_qc = train_qc + val_qc
    total_candidates = sum(len(ex.sql_bad) for ex in all_examples)
    write_qc_report(all_qc, total_candidates, args.qc_output)

    print(f"\nWrote {len(all_examples)} examples to {args.output}")
    print(f"  trainval: {len(train_examples)}, val: {len(val_examples)}")
    print(
        f"sql_good execution check: {len(train_examples) + len(val_examples)}/"
        f"{len(train_examples) + len(val_examples)} passed (100.0%) — only verified pairs are ever written"
    )
    print(
        f"sql_bad candidates: {total_candidates} generated, {total_candidates - len(all_qc)} confirmed-different, "
        f"{len(all_qc)} flagged weak (same-result as sql_good) — see {args.qc_output}"
    )
    print(f"Wrote {len(train_schemas) + len(val_schemas)} annotated schema file(s) to {args.output.parent / 'schemas'}")


if __name__ == "__main__":
    main()
