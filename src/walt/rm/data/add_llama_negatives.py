"""Adds llama3.2's own generation mistakes to sql_bad, tagged reason="llama" — hard
negatives mined from the actual candidate-generating model's real error distribution,
instead of an LLM asked to write a plausible-looking mistake. Targets the RM-transfer
finding in CLAUDE.md directly: the RM is trained to discriminate sql_good from four
clean, deliberate mistake categories, a different error distribution than llama3.2's
actual mistakes at inference time — this exposes it to the real thing.

For each row with split=="trainval" (never "val" — this only ever touches RM training
data, must never leak into the held-out agent-eval split) and a verified
sql_context_valid, generates --n-candidates SQL queries from the question via
OllamaLLM, using the same clean (schema-only, no INSERT rows) context sql_agent.py
shows the LLM at real inference time — not asked to make a mistake, just asked to
answer the question like any other inference call. Each candidate is executed against
the row's *full* sql_context (with sample data — needed to actually tell results
apart) and compared to sql_good's own result the same way fix_sql_bad.py does
(row-set match for a SELECT-shaped sql_good, full database-state diff via
capture_db_state otherwise — reuses that exact comparison function, so "same data"
means the same thing everywhere in this project). A candidate that's byte-identical to
an existing sql_good/sql_bad entry, or whose result matches sql_good's, is dropped —
it isn't a distinguishing mistake, whether that's because llama3.2 got it right or the
sample data doesn't expose a real difference. A candidate that fails to execute at all
(unknown table/column, syntax error) is *not* a match and gets added — that's exactly
the kind of real mistake this step exists to capture.

split=="val" rows and rows without a verified sql_context_valid are passed through
byte-identical; the output always contains every input row, in the same order.

Usage:
    python -m walt.rm.data.add_llama_negatives --limit 5  # sanity check a few rows first
    python -m walt.rm.data.add_llama_negatives \
        --input data/output/gretel/gretel_enhanced_fixed.jsonl \
        --output data/output/gretel/gretel_enhanced_fixed_llama.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from walt.agent.llm.base import BaseLLM
from walt.agent.llm.caching_llm import CachingLLM
from walt.agent.llm.ollama_llm import OllamaLLM
from walt.rm.data.fix_sql_bad import _comparison_signal
from walt.rm.data.gen_training_data import load_records, write_jsonl
from walt.utils.sql_exec import clean_context, resolve_context_statements, run_sql

REASON = "llama"


def llama_negatives_for_row(llm: BaseLLM, record: dict[str, Any], n_candidates: int) -> list[dict[str, str]]:
    """New sql_bad entries (reason="llama") for record — [] if sql_context_valid is
    falsy, sql_good itself fails to execute, or every generated candidate is a
    byte-identical duplicate or matches sql_good's result."""
    if not record.get("sql_context_valid"):
        return []
    context = resolve_context_statements(record.get("sql_context") or (), record.get("sql_context_path"))
    good = run_sql(context, record["sql_good"])
    if not good.success:
        return []
    is_select = good.rows is not None
    good_ok, good_signal = _comparison_signal(context, record["sql_good"], is_select)
    if not good_ok:
        return []

    schema_text = "\n".join(clean_context(context))
    candidates = llm.generate_candidates(record["question"], schema_text, n_candidates)

    existing_sql = {record["sql_good"]} | {b["sql"] for b in record.get("sql_bad", [])}
    new_bad = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in existing_sql or candidate in seen:
            continue  # same query (exact text) — ignore
        seen.add(candidate)
        ok, signal = _comparison_signal(context, candidate, is_select)
        if ok and signal == good_signal:
            continue  # same data — ignore
        new_bad.append({"sql": candidate, "reason": REASON})
    return new_bad


PROGRESS_INTERVAL = 10  # a status line every N trainval rows -- each row costs a real
# Ollama call, slow enough that silent per-row progress can look stuck on a long run.


def process(records: list[dict[str, Any]], llm: BaseLLM, n_candidates: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {"trainval_rows": 0, "rows_touched": 0, "candidates_added": 0, "candidates_ignored": 0}
    n_trainval_total = sum(1 for r in records if r.get("split") == "trainval")
    start = time.perf_counter()
    output = []
    for record in records:
        if record.get("split") != "trainval":
            output.append(record)
            continue
        stats["trainval_rows"] += 1
        new_bad = llama_negatives_for_row(llm, record, n_candidates)
        if not new_bad:
            output.append(record)
        else:
            stats["rows_touched"] += 1
            stats["candidates_added"] += len(new_bad)
            merged = dict(record)
            merged["sql_bad"] = list(record.get("sql_bad", [])) + new_bad
            output.append(merged)
        done = stats["trainval_rows"]
        if done % PROGRESS_INTERVAL == 0 or done == n_trainval_total:
            elapsed = time.perf_counter() - start
            rate_per_min = done / elapsed * 60 if elapsed > 0 else 0.0
            eta_min = (n_trainval_total - done) / rate_per_min if rate_per_min > 0 else float("inf")
            print(
                f"  {done}/{n_trainval_total} trainval rows ({100 * done / n_trainval_total:.1f}%) | "
                f"{rate_per_min:.1f} rows/min | elapsed {elapsed / 60:.1f}min | ETA ~{eta_min:.0f}min"
            )
    return output, stats


def main() -> None:
    # See build_severity_dataset.py's main() for why this matters: stdout is fully
    # buffered by default when redirected to a file (e.g. a backgrounded run), so
    # process()'s progress prints wouldn't be visible until the process exits without
    # this.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/gretel/gretel_enhanced_fixed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/output/gretel/gretel_enhanced_fixed_llama.jsonl"))
    parser.add_argument("--n-candidates", type=int, default=3, help="How many llama3.2 candidates to generate per trainval row")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--ollama-concurrency", type=int, default=1, help="Concurrent Ollama requests per row's candidate generation. Only helps once the Ollama server itself accepts concurrent requests (see build_severity_dataset.py docstring) — otherwise pure overhead.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N trainval rows (sanity check before a full run)")
    parser.add_argument("--llm-cache", type=Path, default=Path("data/output/llm_cache.json"), help="Same cache sql_agent.py/evaluate.py use, keyed by (model, question, schema_context) — reuses candidates instead of re-calling Ollama")
    parser.add_argument("--no-llm-cache", action="store_true")
    args = parser.parse_args()

    llm: BaseLLM = OllamaLLM(model=args.ollama_model, max_concurrency=args.ollama_concurrency)
    if not args.no_llm_cache:
        llm = CachingLLM(llm, cache_path=args.llm_cache)

    records = load_records(args.input)
    if args.limit:
        limited = []
        n_trainval = 0
        for r in records:
            limited.append(r)
            if r.get("split") == "trainval":
                n_trainval += 1
                if n_trainval >= args.limit:
                    break
        records = limited

    output, stats = process(records, llm, args.n_candidates)

    print(f"trainval rows processed : {stats['trainval_rows']}")
    print(f"rows with a new llama negative added: {stats['rows_touched']}")
    print(f"total llama sql_bad candidates added: {stats['candidates_added']}")

    write_jsonl(output, args.output)
    print(f"\nWrote {len(output)} row(s) to {args.output} (val + untouched trainval rows preserved as-is)")


if __name__ == "__main__":
    main()
