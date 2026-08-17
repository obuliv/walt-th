"""Filters every row's sql_bad down to only schema-valid candidates (drops
schema-invalid ones entirely) -- for training a model that only ever sees
schema-valid negatives.

Motivation: a hard is_schema_valid pre-filter (schema_filter.py) already handles
schema-invalid candidates for free at inference, with no training required -- see
CLAUDE.md's ablation ladder, where constant+schema-filter beats a fully-trained lr_v6
outright, and lr_v3 (no schema signal at all) is actively worse than no reranking.
That means a model's training capacity is currently split between two very different
problems: "is this candidate schema-valid" (already solved, for free, by the filter)
and "which schema-valid candidate is semantically correct" (the hard problem the
filter has ~no power over -- pairwise_accuracy 0.29, below random chance, when both
sides of a pair are schema-valid). Training on schema-valid-only pairs removes the
first problem from the training signal entirely, so a linear model's limited capacity
goes exclusively toward the second.

sql_context_clean (schema only, no sample data) is used for the check -- the same
convention is_schema_valid already follows everywhere else in the RM. Applied
uniformly to every row regardless of split, matching this project's established
"no special-casing between trainval and val" convention.

Usage:
    python -m walt.rm.data.filter_schema_valid \
        --input data/output/synth_severity/synth_severity_enhanced.jsonl \
        --output data/output/synth_severity/synth_severity_enhanced_schemavalid.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from walt.rm.data.gen_training_data import load_records, write_jsonl
from walt.rm.model.sql_features import is_schema_valid


def filter_row(record: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    sql_context_clean = record.get("sql_context_clean") or []
    sql_bad = record.get("sql_bad") or []
    kept = [b for b in sql_bad if is_schema_valid(b["sql"], sql_context_clean)]
    merged = dict(record)
    merged["sql_bad"] = kept
    return merged, len(sql_bad), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.input)
    output = []
    n_total_before = n_total_after = 0
    n_rows_emptied = 0
    for record in records:
        merged, n_before, n_after = filter_row(record)
        n_total_before += n_before
        n_total_after += n_after
        if n_before and not n_after:
            n_rows_emptied += 1
        output.append(merged)

    write_jsonl(output, args.output)
    print(f"Wrote {len(output)} row(s) to {args.output}")
    print(
        f"sql_bad candidates: {n_total_before} -> {n_total_after} "
        f"(dropped {n_total_before - n_total_after} schema-invalid, "
        f"{100 * (n_total_before - n_total_after) / n_total_before:.1f}%)"
    )
    print(f"{n_rows_emptied}/{len(output)} row(s) now have zero sql_bad (every candidate was schema-invalid)")


if __name__ == "__main__":
    main()
