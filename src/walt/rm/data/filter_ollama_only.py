"""Filters every row's sql_bad down to only Ollama (llama3.2)-sourced candidates --
drops every candidate Claude added in enhance_severity_dataset.py's new_candidates
step. For training a model that sees only real llama3.2 mistakes, never a
Claude-invented one.

Provenance isn't stored as an explicit field on the enhanced output -- stage 1
(synth_severity_data.jsonl, the raw Ollama pool before Claude ever sees it) and stage
2 (synth_severity_enhanced.jsonl) are matched here by (question, sql_good) as the row
key, and by exact SQL text within a row: a stage-2 sql_bad candidate whose text
appears in that row's stage-1 pool is Ollama-origin; anything else was appended by
Claude as a new_candidate (see enhance_severity_dataset.apply_severity_review, which
always keeps the reviewed existing pool first, in order, then appends new_candidates).

Usage:
    python -m walt.rm.data.filter_ollama_only \
        --stage1 data/output/synth_severity/synth_severity_data.jsonl \
        --input data/output/synth_severity/synth_severity_enhanced.jsonl \
        --output data/output/synth_severity/synth_severity_enhanced_ollamaonly.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from walt.rm.data.gen_training_data import load_records, write_jsonl


def _row_key(record: dict[str, Any]) -> tuple[str, str]:
    return (record.get("question", ""), record.get("sql_good", ""))


def filter_row(record: dict[str, Any], ollama_texts: set[str]) -> tuple[dict[str, Any], int, int]:
    sql_bad = record.get("sql_bad") or []
    kept = [b for b in sql_bad if b["sql"] in ollama_texts]
    merged = dict(record)
    merged["sql_bad"] = kept
    return merged, len(sql_bad), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", type=Path, required=True, help="Raw stage-1 Ollama-candidate pool (before Claude review).")
    parser.add_argument("--input", type=Path, required=True, help="Stage-2 enhanced dataset (reason+severity populated).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage1_records = load_records(args.stage1)
    stage1_by_key = {_row_key(r): {b["sql"] for b in (r.get("sql_bad") or [])} for r in stage1_records}

    records = load_records(args.input)
    output = []
    n_total_before = n_total_after = 0
    n_rows_emptied = 0
    n_rows_unmatched = 0
    for record in records:
        key = _row_key(record)
        ollama_texts = stage1_by_key.get(key)
        if ollama_texts is None:
            n_rows_unmatched += 1
            ollama_texts = set()
        merged, n_before, n_after = filter_row(record, ollama_texts)
        n_total_before += n_before
        n_total_after += n_after
        if n_before and not n_after:
            n_rows_emptied += 1
        output.append(merged)

    write_jsonl(output, args.output)
    print(f"Wrote {len(output)} row(s) to {args.output}")
    print(
        f"sql_bad candidates: {n_total_before} -> {n_total_after} "
        f"(dropped {n_total_before - n_total_after} Claude-added, "
        f"{100 * (n_total_before - n_total_after) / n_total_before:.1f}%)"
    )
    print(f"{n_rows_emptied}/{len(output)} row(s) now have zero sql_bad (every candidate was Claude-added)")
    if n_rows_unmatched:
        print(f"WARNING: {n_rows_unmatched} row(s) had no matching stage-1 row -- all their sql_bad were dropped")


if __name__ == "__main__":
    main()
