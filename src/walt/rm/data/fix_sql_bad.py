"""2nd-pass LLM fixer for mislabeled `sql_bad` negatives in an already-enhanced JSONL
file (currently scoped to data/output/gretel/gretel_enhanced.jsonl).

An earlier ad hoc analysis found that a large fraction of `sql_bad` candidates execute
to the *same* result as `sql_good` on their own row's `sql_context` — i.e. they aren't
actually distinguishing negatives. But that execution-match signal alone isn't a safe
basis for deciding what to drop: it also flags candidates that are still genuinely
distinct mistakes that just happen to coincide on this row's (small, synthetic) sample
data — e.g. `SUM(flag)` vs `COUNT(*) WHERE flag = TRUE` land on the same number when
every `flag` value happens to be 0/1 in the sample, even though they're not the same
query in general. Dropping those would throw away a real negative just because the toy
data didn't distinguish it. So every flagged candidate gets one of two treatments,
decided by the LLM (not by the local execution-match heuristic, which only decides
*what to review*, never *what to drop*):
  1. It's a genuine mistake, but the sample data is too sparse/homogeneous to expose it
     (e.g. a `missing_filters` candidate drops a WHERE clause, but every seeded row
     already satisfies it) — fixed by adding sample rows to `sql_context` so the
     mistake actually shows up.
  2. The LLM confirms it's truly not a distinguishable mistake no matter what data you
     add (e.g. a cosmetic rewrite, renamed alias) — only then is it dropped from
     `sql_bad` entirely, rather than the module ever fabricating a replacement query.

This module finds every flagged candidate locally (pure SQL execution via run_sql, no
LLM needed for detection), asks Claude per affected row to classify/fix per the above,
and locally re-verifies the result before accepting it. Rows with nothing flagged are
left byte-identical; the output always contains every input row (never a subset), in
the same order — dropping a sql_bad entry only ever shrinks that row's own list, never
removes the row itself.

Modes (mirrors gen_training_data.py):
    test    - run a handful of flagged rows through synchronous single calls.
    submit  - submit every flagged row as an Anthropic Message Batch job.
    collect - poll a submitted batch, merge, and write the full output JSONL.

Usage:
    python -m walt.rm.data.fix_sql_bad test --input data/output/gretel/gretel_enhanced.jsonl --limit 3
    python -m walt.rm.data.fix_sql_bad submit --input data/output/gretel/gretel_enhanced.jsonl
    python -m walt.rm.data.fix_sql_bad collect --batch-id msgbatch_xxx --output data/output/gretel/gretel_enhanced_fixed.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import jsonschema

from walt.rm.data.gen_training_data import (
    ANTHROPIC_MODEL,
    check_batch_limits,
    extract_tool_input,
    load_records,
    load_state,
    make_custom_id,
    save_state,
    write_jsonl,
)
from walt.utils.sql_exec import run_sql

# ---------------------------------------------------------------------------
# Local detection: no LLM, pure SQL execution.
# ---------------------------------------------------------------------------


def find_flagged_indices(record: dict[str, Any]) -> list[int]:
    """Indices into record['sql_bad'] whose execution result matches sql_good's, on
    record['sql_context']. Empty if there's no sql_bad, sql_context_valid is falsy, or
    sql_good itself fails to execute — nothing to compare against in those cases."""
    sql_bad = record.get("sql_bad") or []
    if not sql_bad or not record.get("sql_context_valid"):
        return []
    context = record.get("sql_context") or []
    good = run_sql(context, record["sql_good"])
    if not good.success:
        return []
    good_rows = set(good.rows) if good.rows is not None else None

    flagged = []
    for i, bad in enumerate(sql_bad):
        result = run_sql(context, bad["sql"])
        if result.success and (set(result.rows) if result.rows is not None else None) == good_rows:
            flagged.append(i)
    return flagged


# ---------------------------------------------------------------------------
# Prompt.
# ---------------------------------------------------------------------------

FIX_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "question": "What is the average number of international tourists in the first quarter of each year?",
        "sql_good": "SELECT AVG(visitors) as avg_visitors FROM tourism_stats WHERE quarter = 1;",
        "sql_context": [
            "CREATE TABLE tourism_stats (country TEXT(50), visitors INTEGER, year INTEGER, quarter INTEGER)",
            "INSERT INTO tourism_stats (country, visitors, year, quarter) VALUES ('Spain', 15, 2020, 1), ('Germany', 18, 2020, 1), ('Spain', 16, 2021, 1), ('Germany', 19, 2021, 1)",
        ],
        "sql_bad": [
            {"sql": "SELECT AVG(visitors) as avg_visitors FROM tourism_stats;", "reason": "missing_filters"},
        ],
        "flagged_indices": [0],
        "response": {
            "sql_context_additions": [
                "INSERT INTO tourism_stats (country, visitors, year, quarter) VALUES ('France', 25, 2020, 2)"
            ],
            "drop": [],
        },
        "explanation": (
            "Every seeded row already has quarter = 1, so dropping `WHERE quarter = 1` "
            "is a no-op on this data even though it's a real mistake in general — this "
            "candidate should NOT be dropped. Adding a quarter = 2 row gives the omitted "
            "filter something to exclude, without touching sql_good or the existing rows."
        ),
    },
    {
        "question": "Show the top 5 cities with the highest population density",
        "sql_good": "SELECT name, population / area AS population_density FROM cities ORDER BY population_density DESC LIMIT 5;",
        "sql_context": [
            "CREATE TABLE cities (city_id INTEGER, name TEXT(255), population INTEGER, area REAL)",
            "INSERT INTO cities (city_id, name, population, area) VALUES (1, 'New York City', 8601000, 302.6)",
            "INSERT INTO cities (city_id, name, population, area) VALUES (2, 'Los Angeles', 4000000, 1214.9)",
        ],
        "sql_bad": [
            {"sql": "SELECT name, population / area AS population_density FROM cities ORDER BY population_density DESC;", "reason": "missing_filters"},
            {"sql": "SELECT name, population / area AS pd FROM cities ORDER BY pd DESC LIMIT 5;", "reason": "unsafe_patterns"},
        ],
        "flagged_indices": [0, 1],
        "response": {
            "sql_context_additions": [
                "INSERT INTO cities (city_id, name, population, area) VALUES (3, 'Chicago', 2700000, 589.6)",
                "INSERT INTO cities (city_id, name, population, area) VALUES (4, 'Houston', 2300000, 1651.1)",
                "INSERT INTO cities (city_id, name, population, area) VALUES (5, 'Phoenix', 1600000, 1340.6)",
                "INSERT INTO cities (city_id, name, population, area) VALUES (6, 'Philadelphia', 1580000, 347.6)",
            ],
            "drop": [1],
        },
        "explanation": (
            "Index 0 drops LIMIT 5, which only matters once there are more than 5 rows "
            "— so bringing the city count to 6 makes it a real mistake, this candidate "
            "should NOT be dropped. Index 1 is only a cosmetic rename of the alias "
            "(population_density -> pd); no amount of data would ever make it diverge "
            "from sql_good, so it's confirmed equivalent and dropped rather than kept as "
            "a fabricated replacement."
        ),
    },
    {
        "question": "How many orders have shipped?",
        "sql_good": "SELECT COUNT(*) FROM orders WHERE status = 'shipped';",
        "sql_context": [
            "CREATE TABLE orders (id INTEGER, status TEXT, tracking_number TEXT)",
            "INSERT INTO orders (id, status, tracking_number) VALUES (1, 'shipped', 'TRK1')",
            "INSERT INTO orders (id, status, tracking_number) VALUES (2, 'shipped', 'TRK2')",
            "INSERT INTO orders (id, status, tracking_number) VALUES (3, 'pending', NULL)",
        ],
        "sql_bad": [
            {"sql": "SELECT COUNT(tracking_number) FROM orders WHERE status = 'shipped';", "reason": "wrong_aggregation"},
        ],
        "flagged_indices": [0],
        "response": {
            "sql_context_additions": [
                "INSERT INTO orders (id, status, tracking_number) VALUES (4, 'shipped', NULL)"
            ],
            "drop": [],
        },
        "explanation": (
            "COUNT(tracking_number) currently matches COUNT(*) only because every "
            "'shipped' row in the sample happens to have a tracking_number. But "
            "tracking_number isn't part of the WHERE clause, so a 'shipped' row CAN "
            "have a NULL tracking_number and still pass the filter — adding one makes "
            "COUNT(*) = 3 while COUNT(tracking_number) stays 2. This is a real mistake "
            "exposed by the right data, NOT a drop."
        ),
    },
    {
        "question": "How many startups were founded by people from underrepresented racial or ethnic groups in the USA?",
        "sql_good": "SELECT COUNT(*) FROM startups WHERE location = 'USA' AND founder_race IN ('African American', 'Hispanic', 'Native American', 'Pacific Islander');",
        "sql_context": [
            "CREATE TABLE startups (id INTEGER, name TEXT, location TEXT, founder_race TEXT)",
            "INSERT INTO startups (id, name, location, founder_race) VALUES (1, 'Startup A', 'USA', 'African American')",
            "INSERT INTO startups (id, name, location, founder_race) VALUES (2, 'Startup B', 'Canada', 'Caucasian')",
            "INSERT INTO startups (id, name, location, founder_race) VALUES (3, 'Startup C', 'USA', 'Hispanic')",
        ],
        "sql_bad": [
            {"sql": "SELECT COUNT(founder_race) FROM startups WHERE location = 'USA' AND founder_race IN ('African American', 'Hispanic', 'Native American', 'Pacific Islander');", "reason": "wrong_aggregation"},
        ],
        "flagged_indices": [0],
        "response": {
            "sql_context_additions": [],
            "drop": [0],
        },
        "explanation": (
            "This looks like the same COUNT(col) vs COUNT(*) pattern as above, but here "
            "founder_race is BOTH the counted column AND the column the WHERE...IN "
            "filter tests. Any row with founder_race NULL fails 'founder_race IN "
            "(...)' (NULL IN (...) is never true), so it's excluded from the filtered "
            "result by both queries equally — no addable row can ever separate them. "
            "COUNT(founder_race) and COUNT(*) are provably identical under this WHERE "
            "clause, so this candidate is confirmed equivalent and dropped."
        ),
    },
]


def _format_fix_few_shot() -> str:
    blocks = []
    for i, ex in enumerate(FIX_FEW_SHOT_EXAMPLES, 1):
        bad_list_desc = "\n".join(
            f"  [{j}] ({b['reason']}) {b['sql']}" + ("  <-- SAME RESULT AS sql_good, needs review" if j in ex["flagged_indices"] else "")
            for j, b in enumerate(ex["sql_bad"])
        )
        blocks.append(
            f"Example {i}\n"
            f"Question: {ex['question']}\n"
            f"sql_good (already correct, do not change): {ex['sql_good']}\n"
            f"sql_context (existing, append-only): {json.dumps(ex['sql_context'])}\n"
            f"Current sql_bad list:\n{bad_list_desc}\n"
            f"Correct tool call input: {json.dumps(ex['response'])}\n"
            f"Why: {ex['explanation']}"
        )
    return "\n\n".join(blocks)


FIX_SYSTEM_PROMPT = f"""You are a meticulous SQL reviewer auditing training data for a \
text-to-SQL reward model.

You are given a (question, sql_good, sql_context) triple and its current `sql_bad` \
list. Some entries in `sql_bad` are flagged because, when executed against \
`sql_context`, they currently return the *same* result as `sql_good` — but that alone \
does NOT mean the candidate is actually equivalent to `sql_good` in general. It may \
simply be that this row's small sample data happens not to distinguish two genuinely \
different queries (e.g. `SUM(flag)` and `COUNT(*) WHERE flag = TRUE` land on the same \
number whenever every `flag` value in the sample happens to be 0 or 1, even though \
they're not the same query in general). Dropping a candidate like that would throw away \
a real negative just because the toy data didn't expose it — so your job is to tell \
these two cases apart for each flagged entry, not to assume a match means "drop it":

1. The candidate IS a genuine mistake in general — the sample data is just too sparse \
or uniform to expose it (e.g. a dropped filter that every row already satisfies, a LIMIT \
that never binds because there aren't enough rows, or a different aggregate function \
that only coincides because of the specific numbers in the sample). Fix this by adding \
one or more INSERT INTO statements to `sql_context_additions`, against the EXISTING \
tables only (same columns/types — never add CREATE TABLE statements, never modify or \
remove existing rows), such that the flagged query would now actually return a \
different result than `sql_good`. Do NOT include this index in `drop`.

2. The candidate is NOT actually a distinguishable mistake no matter what data you add \
— e.g. a cosmetic rewrite (different alias, reordered columns/clauses, a logically \
equivalent condition, an equivalent join restructuring) that always computes the same \
thing as `sql_good` in general, not just on this sample. Only in this case, add its \
index to `drop` so it's removed from `sql_bad` entirely — never fabricate a replacement \
query.

Before concluding a candidate belongs in bucket 2, specifically check whether one of \
these standard "expose it with the right data" moves would put it in bucket 1 instead: \
a row with a NULL in a relevant column (`COUNT(col)` vs `COUNT(*)` diverge once a row \
has `col` NULL — but ONLY if that row would still pass the query's own WHERE clause; if \
`col` is itself the column a WHERE/IN/equality filter tests, a NULL there can never \
pass that filter in the first place, so no addable row can ever separate them and the \
candidate really is equivalent), a row that satisfies one side of a boundary but not \
the other (e.g. `>` vs `>=`, `<` vs `<=`), or a row on one side of a JOIN with no match \
on the other (e.g. `JOIN` vs `LEFT JOIN` only diverge when something doesn't match). \
Work through whether such a row could exist given the query's own filters before \
choosing `drop` — most candidates that look like a "cosmetic rewrite" at a glance are \
actually bucket 1 once you consider the right edge-case row, but not always.

Every flagged index must end up in exactly one of these two buckets (fixed via added \
data, or confirmed-equivalent and dropped) — never both, never neither. Never modify \
`sql_good`. Never modify or remove an existing `sql_context` statement — only append. \
Every existing, unflagged sql_bad entry must keep working unchanged.

# Examples

{_format_fix_few_shot()}

Call the `emit_sql_bad_fixes` tool exactly once with your result. Do not include any \
text outside the tool call."""


def build_user_message_fix(record: dict[str, Any], flagged_indices: list[int]) -> str:
    sql_bad = record["sql_bad"]
    bad_list_desc = "\n".join(
        f"[{i}] ({b['reason']}) {b['sql']}" + ("  <-- SAME RESULT AS sql_good, needs review" if i in flagged_indices else "")
        for i, b in enumerate(sql_bad)
    )
    return (
        f"Question: {record['question']}\n"
        f"sql_good (already correct, do not change): {record['sql_good']}\n"
        f"sql_context (existing, append-only): {json.dumps(record.get('sql_context', []))}\n"
        f"Current sql_bad list:\n{bad_list_desc}"
    )


# ---------------------------------------------------------------------------
# Tool / JSON schema.
# ---------------------------------------------------------------------------

FIX_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql_context_additions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Zero or more additional INSERT INTO statements, against the EXISTING "
                "tables only (same columns/types, no new CREATE TABLE, don't touch "
                "existing rows), to append to sql_context so a flagged query that IS a "
                "genuine mistake in general actually shows up as wrong in the result. "
                "Empty if not needed."
            ),
        },
        "drop": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "0-based indices into the current sql_bad list that are confirmed to be "
                "genuinely equivalent to sql_good no matter what data is used (a "
                "cosmetic rewrite, not a real mistake) — these are removed from sql_bad "
                "entirely. Only include an index here if you're confident it's not a "
                "distinguishable mistake in general, not just on this sample."
            ),
        },
    },
    "required": ["sql_context_additions", "drop"],
    "additionalProperties": False,
}

TOOL_NAME = "emit_sql_bad_fixes"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Return sql_context additions for flagged candidates that are genuine mistakes, and drop indices for flagged candidates confirmed equivalent to sql_good.",
    "input_schema": FIX_RESULT_SCHEMA,
}

TOOL_CHOICE = {"type": "tool", "name": TOOL_NAME}

_VALIDATOR = jsonschema.Draft7Validator(FIX_RESULT_SCHEMA)


def validate_result(data: dict[str, Any]) -> None:
    _VALIDATOR.validate(data)


def request_params(record: dict[str, Any], flagged_indices: list[int]) -> dict[str, Any]:
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": FIX_SYSTEM_PROMPT,
        "tools": [TOOL_SCHEMA],
        "tool_choice": TOOL_CHOICE,
        "messages": [{"role": "user", "content": build_user_message_fix(record, flagged_indices)}],
    }


# ---------------------------------------------------------------------------
# Local verification / merge — no extra API call, mirrors gen_training_data's
# enhance_record: trust the LLM structurally, verify execution locally.
# ---------------------------------------------------------------------------


def apply_fix(record: dict[str, Any], flagged_indices: list[int], result: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    """Returns (merged_record, still_unresolved_indices). Dropped indices (LLM-confirmed
    equivalent to sql_good) are removed from sql_bad entirely, never replaced with a
    fabricated query — still_unresolved_indices are original (pre-drop) indices, only
    meaningful for reporting since the row's sql_bad list may have shrunk."""
    original_context = list(record.get("sql_context", []))
    additions = result.get("sql_context_additions") or []
    context = original_context + additions
    good = run_sql(context, record["sql_good"])
    if not good.success:
        # Additions broke sql_good (e.g. a PK/UNIQUE conflict) — never let a fix break
        # sql_good, fall back to the original context.
        context = original_context
        good = run_sql(context, record["sql_good"])
    good_rows = set(good.rows) if good.rows is not None else None

    # Only trust a drop for an index the LLM was actually shown as flagged — ignore any
    # hallucinated index outside that set.
    drop = set(result.get("drop") or []) & set(flagged_indices)

    sql_bad = []
    still_unresolved = []
    for idx, bad in enumerate(record["sql_bad"]):
        if idx in drop:
            continue
        sql_bad.append(dict(bad))
        if idx in flagged_indices:
            candidate = run_sql(context, bad["sql"])
            if candidate.success and (set(candidate.rows) if candidate.rows is not None else None) == good_rows:
                still_unresolved.append(idx)

    merged = dict(record)
    merged["sql_context"] = context
    merged["sql_bad"] = sql_bad
    merged["sql_context_valid"] = good.success

    return merged, still_unresolved


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def print_quality_summary(label: str, records: list[dict[str, Any]]) -> None:
    n_rows_flagged = 0
    n_candidates_flagged = 0
    for r in records:
        flagged = find_flagged_indices(r)
        if flagged:
            n_rows_flagged += 1
            n_candidates_flagged += len(flagged)
    n_candidates_total = sum(len(r.get("sql_bad") or []) for r in records)
    print(
        f"{label}: {n_rows_flagged}/{len(records)} row(s), "
        f"{n_candidates_flagged}/{n_candidates_total} sql_bad candidate(s) still match sql_good's result."
    )


# ---------------------------------------------------------------------------
# test: synchronous single calls.
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    all_records = load_records(args.input)
    to_fix = [(r, find_flagged_indices(r)) for r in all_records]
    to_fix = [(r, flagged) for r, flagged in to_fix if flagged]
    print(f"{len(to_fix)}/{len(all_records)} row(s) have at least one flagged sql_bad candidate.")
    to_fix = to_fix[: args.limit]
    if not to_fix:
        print("No rows to test.")
        return

    results = []
    for i, (record, flagged_indices) in enumerate(to_fix, 1):
        print(f"[{i}/{len(to_fix)}] flagged={flagged_indices} {record['question'][:80]}")
        params = request_params(record, flagged_indices)
        message = client.messages.create(**params)
        try:
            result = extract_tool_input(message)
            validate_result(result)
        except (ValueError, jsonschema.ValidationError) as exc:
            print(f"  FAILED: {exc}")
            continue
        merged, unresolved = apply_fix(record, flagged_indices, result)
        if unresolved:
            print(f"  still unresolved after fix: {unresolved}")
        results.append(merged)
        print(json.dumps(merged, indent=2))

    if args.output:
        write_jsonl(results, args.output)
        print(f"\nWrote {len(results)} test result(s) to {args.output}")


# ---------------------------------------------------------------------------
# submit: build and submit a Message Batch for every flagged row.
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    all_records = load_records(args.input)

    requests = []
    entries_by_custom_id: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(all_records):
        flagged = find_flagged_indices(record)
        if not flagged:
            continue
        custom_id = make_custom_id(i, len(all_records))
        requests.append({"custom_id": custom_id, "params": request_params(record, flagged)})
        entries_by_custom_id[custom_id] = {"record": record, "flagged_indices": flagged}

    print(f"{len(requests)}/{len(all_records)} row(s) have at least one flagged sql_bad candidate.")
    if args.limit:
        keep_ids = set(list(entries_by_custom_id)[: args.limit])
        requests = [r for r in requests if r["custom_id"] in keep_ids]
        entries_by_custom_id = {k: v for k, v in entries_by_custom_id.items() if k in keep_ids}
    if not requests:
        print("No rows to submit.")
        return

    check_batch_limits(requests)

    batch = client.messages.batches.create(requests=requests)
    save_state(batch.id, args.input, entries_by_custom_id)

    print(f"Submitted batch {batch.id} with {len(requests)} requests (status: {batch.processing_status}).")
    print(f"Collect results later with:\n  python -m walt.rm.data.fix_sql_bad collect --batch-id {batch.id} --output <path>")


# ---------------------------------------------------------------------------
# collect: poll a batch until it ends, splice fixes into a full copy of the input,
# write the complete output JSONL (every input row present, original order).
# ---------------------------------------------------------------------------


def _index_from_custom_id(custom_id: str) -> int:
    return int(custom_id.split("-")[1])


def cmd_collect(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    state = load_state(args.batch_id)
    entries_by_custom_id: dict[str, dict[str, Any]] = state["records"]
    all_records = load_records(Path(state["input_path"]))

    deadline = time.monotonic() + args.max_wait if args.max_wait else None
    while True:
        batch = client.messages.batches.retrieve(args.batch_id)
        print(f"Batch {batch.id}: {batch.processing_status} {dict(batch.request_counts)}")
        if batch.processing_status == "ended":
            break
        if args.no_wait:
            print("Not ended yet; re-run collect later (pass --no-wait to just check, omit it to poll).")
            return
        if deadline and time.monotonic() >= deadline:
            print(f"Gave up waiting after {args.max_wait}s; re-run collect later.")
            return
        time.sleep(args.poll_interval)

    counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, "invalid": 0}
    n_unresolved_rows = 0
    for item in client.messages.batches.results(args.batch_id):
        entry = entries_by_custom_id.get(item.custom_id)
        if entry is None:
            print(f"WARNING: unknown custom_id {item.custom_id!r}, skipping.")
            continue
        idx = _index_from_custom_id(item.custom_id)

        if item.result.type != "succeeded":
            counts[item.result.type] = counts.get(item.result.type, 0) + 1
            print(f"  {item.custom_id}: {item.result.type} (row left unfixed)")
            continue

        try:
            result = extract_tool_input(item.result.message)
            validate_result(result)
        except (ValueError, jsonschema.ValidationError) as exc:
            counts["invalid"] += 1
            print(f"  {item.custom_id}: invalid output ({exc}) (row left unfixed)")
            continue

        merged, unresolved = apply_fix(entry["record"], entry["flagged_indices"], result)
        all_records[idx] = merged
        counts["succeeded"] += 1
        if unresolved:
            n_unresolved_rows += 1
            print(f"  {item.custom_id}: still unresolved after fix: {unresolved}")

    write_jsonl(all_records, args.output)
    print(f"\nWrote {len(all_records)} row(s) to {args.output} (full input preserved, order unchanged).")
    print(f"Counts: {counts}")
    if n_unresolved_rows:
        print(f"{n_unresolved_rows} row(s) still have an unresolved flagged candidate after the fix.")
    print_quality_summary("Before this fix pass", [entries_by_custom_id[cid]["record"] for cid in entries_by_custom_id])
    print_quality_summary("After this fix pass (full output)", all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_p = subparsers.add_parser("test", help="Run a few flagged rows through synchronous single calls.")
    test_p.add_argument("--input", type=Path, required=True, help="Input JSONL (an *_enhanced.jsonl with sql_context + sql_bad)")
    test_p.add_argument("--limit", type=int, default=3, help="Number of flagged rows to test")
    test_p.add_argument("--output", type=Path, default=None, help="Optionally write test results to a JSONL file")
    test_p.set_defaults(func=cmd_test)

    submit_p = subparsers.add_parser("submit", help="Submit every flagged row as a Message Batch.")
    submit_p.add_argument("--input", type=Path, required=True, help="Input JSONL (an *_enhanced.jsonl with sql_context + sql_bad)")
    submit_p.add_argument("--limit", type=int, default=None, help="Only submit the first N flagged rows")
    submit_p.set_defaults(func=cmd_submit)

    collect_p = subparsers.add_parser("collect", help="Poll a batch and write the full, spliced-in output JSONL.")
    collect_p.add_argument("--batch-id", required=True, help="Batch id printed by `submit`")
    collect_p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    collect_p.add_argument("--no-wait", action="store_true", help="Check status once and exit instead of polling")
    collect_p.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between status checks")
    collect_p.add_argument("--max-wait", type=float, default=None, help="Give up polling after this many seconds")
    collect_p.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) or export it.")
    args.func(args)


if __name__ == "__main__":
    main()
