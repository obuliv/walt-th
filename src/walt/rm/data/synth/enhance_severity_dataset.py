"""Stage 2 of the severity-scored synth pipeline (see build_severity_dataset.py for
stage 1): categorizes every sql_bad candidate into a reason and assigns a 0-5 severity
via one combined Claude tool call per row, optionally backfilling more candidates when
severity/category coverage is thin.

Input rows (from build_severity_dataset.py) carry sql_bad entries shaped
{"sql", "matches_gold"} -- no reason/severity yet, since stage 1 has no rule-based
generator to supply one. This is the FIRST and only categorization pass, not a review
of existing tags (contrast with fix_sql_bad.py, which re-reviews already-categorized
candidates).

Reason taxonomy: the 5 base categories reused from gen_training_data.py
(missing_filters, wrong_aggregation, unsafe_patterns, misjoined_tables, compound) plus
"llama" as an explicit catch-all -- but ONLY for categorizing existing (llama3.2-
generated) candidates whose mistake doesn't cleanly fit one of the 5. A NEW candidate
Claude proposes to fill thin coverage must always target one of the 5 base categories
(enforced by a stricter schema on that array) -- deliberately inventing an
uncategorized mistake makes no sense.

Severity=0 ("executes to the same result as sql_good") is informed but NOT forced by
the local, real-execution hint computed in stage 1 (matches_gold) -- the hint only
proves the candidate matches sql_good ON THIS ROW'S SPECIFIC SAMPLE DATA, which can't
distinguish genuine equivalence (a cosmetic rewrite, true for any data) from a
coincidental match (a real mistake this particular sample just doesn't happen to
expose -- e.g. a LIKE '%X%' wildcard that only matches one category because no other
category name happens to contain that substring here). The LLM is shown the hint as
strong context and trusted to tell these apart: if it assigns a nonzero severity
despite the hint, that's kept as-is (see EXISTING_REVIEW_ITEM handling below). Only
the other direction is still hard-enforced: a severity=0 claim for a candidate the
hint proves DOES differ from sql_good is a plain factual error (not a judgment call),
so that's still clamped. Every new_candidates item is re-verified via
walt.utils.sql_exec.execute_with_context (which self-heals its connection cache on any
mutating statement) before being accepted -- one that fails to execute is dropped,
since a Claude-proposed new candidate carries no provenance guarantee the way a
llama3.2-sourced one does.

Rows whose LLM call fails/errors/returns invalid output are NOT left in their raw,
structurally-incomplete stage-1 shape (which has no "reason" key and would break
Example/SQLBadCandidate loading) -- they instead get a local-only fallback
categorization (empty LLM result: matches_gold-hinted candidates still correctly land
at severity 0, everything else defaults to reason="llama"/severity=1) so the output
file is always a fully valid, loadable dataset.

Modes (mirrors gen_training_data.py/fix_sql_bad.py):
    test    - run a handful of rows through synchronous single calls. REQUIRED before
              ever running submit -- review the printed output first.
    submit  - submit every row as an Anthropic Message Batch job.
    collect - poll a submitted batch, merge, and write the full output JSONL.

Usage:
    python -m walt.rm.data.synth.enhance_severity_dataset test --input data/output/synth_severity/synth_severity_data.jsonl --limit 5
    python -m walt.rm.data.synth.enhance_severity_dataset submit --input data/output/synth_severity/synth_severity_data.jsonl
    python -m walt.rm.data.synth.enhance_severity_dataset collect --batch-id msgbatch_xxx --output data/output/synth_severity/synth_severity_enhanced.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
from walt.utils.sql_exec import execute_with_context

# ---------------------------------------------------------------------------
# Reason taxonomy.
# ---------------------------------------------------------------------------

BASE_REASON_NAMES = ["missing_filters", "wrong_aggregation", "unsafe_patterns", "misjoined_tables", "compound"]
EXISTING_REASON_NAMES = BASE_REASON_NAMES + ["llama"]

REASON_DESCRIPTIONS = {
    "missing_filters": "Omits a WHERE/HAVING condition (or LIMIT) the question implies, returning too many or the wrong rows.",
    "wrong_aggregation": "Uses the wrong aggregate function, groups by the wrong column, or aggregates over the wrong column entirely.",
    "unsafe_patterns": "Uses a risky/imprecise pattern like SELECT * or an unqualified cross join instead of the specific columns/join condition the question calls for.",
    "misjoined_tables": "Joins the wrong pair of tables, joins on the wrong column, or uses the wrong join type (e.g. INNER instead of LEFT), producing an incorrect result set.",
    "compound": "Combines two or more of the above mistakes in the same query.",
    "llama": (
        "A real mistake that doesn't cleanly fit any of the 5 categories above (e.g. "
        "wrong sort direction, a reversed comparison, an off-topic answer). ONLY valid "
        "for candidates already in the pool -- never propose a NEW candidate tagged "
        "llama; a deliberately-invented mistake must always target one of the 5 named "
        "categories."
    ),
}


def _reasons_block() -> str:
    return "\n".join(f"- {name}: {REASON_DESCRIPTIONS[name]}" for name in EXISTING_REASON_NAMES)


SEVERITY_RUBRIC = """\
SEVERITY RUBRIC (0-5):
0 - Genuinely, generally equivalent to sql_good -- would produce the same result for \
ANY data consistent with the schema, not just this row's specific sample (a cosmetic \
rewrite: renamed alias, reordered columns/clauses, a logically identical \
restructuring). Never assign 0 to a candidate that isn't marked "SAME RESULT AS \
sql_good (verified)" below -- that annotation is a necessary but NOT sufficient \
condition. It proves the candidate matches sql_good on this row's specific sample \
data; it does NOT prove general equivalence. A candidate can be marked "SAME RESULT" \
and still be a real, general mistake this particular sample just doesn't happen to \
expose (e.g. LIKE '%Paper%' matching the same single row as = 'Paper' only because no \
other value in this sample's small category list happens to contain that substring -- \
on a table with a 'White Paper' or 'Paperback' category, they'd diverge). When you \
believe a marked candidate is this kind of coincidental match rather than true \
equivalence, do NOT assign 0 -- assign the severity that reflects how bad the mistake \
would be in general. Never propose a NEW candidate at severity 0 (new candidates must \
be genuinely distinguishing mistakes).
1 - Cosmetic/edge-case difference: an off-by-one LIMIT, an ORDER BY tie-break \
ambiguity, a boundary operator (> vs >=) that only diverges on an edge value.
2 - Noticeable but contained: wrong sort order, an extra/missing non-key SELECT \
column, a slightly too-wide/narrow filter that adds or drops a few rows.
3 - Moderate: materially changes the answer for a real subset of inputs -- wrong \
aggregate function (SUM vs AVG), a missing but semantically important filter, a \
plausible-but-wrong join column that shifts (not explodes) the result.
4 - Serious: joined to the wrong table entirely, SELECT * exposing unintended \
columns, a cross join causing row duplication/explosion.
5 - Severe: answers a fundamentally different question (wrong entity \
counted/aggregated, wrong sort direction returning the opposite answer), or a result \
set of a completely different shape/semantics."""

COVERAGE_RULE = """\
COVERAGE: after reviewing every existing candidate, check whether the pool (existing \
candidates + any new_candidates you add) has at least 3 candidates with severity 1-5, \
spanning both a "low" (1-2) and a "high" (4-5) severity value. If not, propose up to \
3 new_candidates (each a genuinely different mistake, in one of the 5 base categories \
only -- never llama) to fill the gap. If the pool already satisfies this, return an \
empty new_candidates array -- do not pad with redundant candidates. A row with zero \
existing candidates definitely needs new_candidates.

IMPORTANT, since this is the single most common mistake reviewers make: every object \
in new_candidates MUST have "reason" set to exactly one of missing_filters, \
wrong_aggregation, unsafe_patterns, misjoined_tables, or compound -- the tool call \
will be REJECTED outright (wasting the whole review) if "reason" is "llama" anywhere \
in new_candidates, no exceptions. "llama" is a label for categorizing an EXISTING \
candidate you didn't write yourself, never for one you're inventing. If the mistake \
you want to add doesn't obviously fit one of the 5, either pick whichever of the 5 is \
the closest fit (a forced-but-plausible fit is fine here -- you control exactly what \
the new candidate's mistake is, so choose one that fits) or simply don't add it and \
leave new_candidates shorter than 3."""

# ---------------------------------------------------------------------------
# Few-shot examples, embedded directly in the (cacheable) system prompt.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "question": "What is the total revenue by category last month?",
        "sql_good": (
            "SELECT category, SUM(price * units) AS revenue FROM sales "
            "WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01' GROUP BY category"
        ),
        "sql_context_clean": [
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT NOT NULL, price REAL NOT NULL, units INTEGER NOT NULL, sale_date TEXT NOT NULL)"
        ],
        "pool": [
            {"sql": "SELECT category, SUM(price * units) AS revenue FROM sales GROUP BY category", "matches_gold": False},
            {
                "sql": (
                    "SELECT category, SUM(price * units) AS total_revenue FROM sales "
                    "WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01' GROUP BY category"
                ),
                "matches_gold": True,
            },
            {"sql": "SELECT category FROM sales WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01'", "matches_gold": False},
        ],
        "response": {
            "existing": [
                {"index": 0, "reason": "missing_filters", "severity": 3},
                {"index": 1, "reason": "unsafe_patterns", "severity": 0},
                {"index": 2, "reason": "wrong_aggregation", "severity": 5},
            ],
            "new_candidates": [
                {
                    "sql": (
                        "SELECT category, SUM(price * units) AS revenue FROM sales "
                        "WHERE sale_date >= '2024-01-01' AND sale_date <= '2024-02-01' GROUP BY category"
                    ),
                    "reason": "missing_filters",
                    "severity": 1,
                }
            ],
        },
        "explanation": (
            "Index 1 is marked SAME RESULT AS sql_good (verified) AND is genuinely "
            "equivalent for any data -- it's just a renamed alias -- so it correctly "
            "gets severity 0. Index 0 drops the date filter entirely (moderate: wrong "
            "for any month with other data) and index 2 drops the aggregation "
            "entirely, answering with raw rows instead of a per-category sum (severe: "
            "completely different shape of answer). After reviewing, non-zero "
            "severities are {3, 5} -- only 2 candidates, and no 'low' (1-2) band "
            "represented -- so one new_candidates entry is added: a boundary flip "
            "(<= instead of <) that only diverges when a sale lands exactly at "
            "midnight on the 1st, a low-severity missing_filters mistake."
        ),
    },
    {
        "question": "Who is the longest-tenured employee?",
        "sql_good": "SELECT name FROM employees ORDER BY hire_date ASC LIMIT 1",
        "sql_context_clean": [
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, hire_date TEXT NOT NULL, department TEXT)"
        ],
        "pool": [
            {"sql": "SELECT name FROM employees ORDER BY hire_date DESC LIMIT 1", "matches_gold": False},
            {"sql": "SELECT name FROM employees ORDER BY hire_date ASC", "matches_gold": False},
            {"sql": "SELECT name FROM employees ORDER BY hire_date ASC LIMIT 2", "matches_gold": False},
        ],
        "response": {
            "existing": [
                {"index": 0, "reason": "llama", "severity": 5},
                {"index": 1, "reason": "missing_filters", "severity": 3},
                {"index": 2, "reason": "missing_filters", "severity": 1},
            ],
            "new_candidates": [],
        },
        "explanation": (
            "Index 0 flips ASC to DESC, answering 'most recently hired' -- the exact "
            "opposite question. That's a real, severe mistake, but it isn't a missing "
            "filter, wrong aggregate, unsafe pattern, or misjoin -- none of the 5 base "
            "categories fit an ordering-direction flip, so it's tagged llama. Index 1 "
            "drops LIMIT 1 entirely (moderate -- returns everyone, not just the "
            "longest-tenured) and index 2 only loosens it to LIMIT 2 (low -- an "
            "off-by-one that still contains the right answer, just not only it). "
            "Non-zero severities are now {5, 3, 1} -- 3 candidates, spanning both the "
            "low and high bands -- so no new_candidates are needed."
        ),
    },
    {
        "question": "Find the code of the document type \"Paper\".",
        "sql_good": "SELECT document_type_code FROM Ref_document_types WHERE document_type_name = \"Paper\"",
        "sql_context_clean": [
            "CREATE TABLE Ref_Document_Types (\n  Document_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'BK', 'CV', 'PR', 'RV' */\n  Document_Type_Name TEXT(255) NOT NULL, /* one of: 'Book', 'CV', 'Paper', 'Review' */\n  Document_Type_Description TEXT(255) NOT NULL\n)"
        ],
        "pool": [
            {"sql": "SELECT Document_Type_Code FROM Ref_Document_Types WHERE Document_Type_Name LIKE '%Paper%'", "matches_gold": True},
            {"sql": "SELECT Document_Type_Code, Document_Type_Name FROM Ref_Document_Types WHERE Document_Type_Name = 'Paper'", "matches_gold": False},
            {"sql": "SELECT Document_Type_Code FROM Ref_Document_Types WHERE Document_Type_Name = 'paper'", "matches_gold": False},
        ],
        "response": {
            "existing": [
                {"index": 0, "reason": "missing_filters", "severity": 3},
                {"index": 1, "reason": "unsafe_patterns", "severity": 2},
                {"index": 2, "reason": "missing_filters", "severity": 5},
            ],
            "new_candidates": [],
        },
        "explanation": (
            "Index 0 is marked SAME RESULT AS sql_good (verified) -- but that's a "
            "coincidence of this sample, not true equivalence: Document_Type_Name only "
            "has 4 distinct values here ('Book', 'CV', 'Paper', 'Review') and only "
            "'Paper' contains the substring 'Paper', so LIKE '%Paper%' happens to match "
            "the same single row as = 'Paper'. But LIKE with wildcards is a strictly "
            "looser filter than exact equality -- it would incorrectly also match a "
            "value like 'White Paper' or 'Paperback' if one existed in this table. "
            "Since this is a real, general mistake the hint's annotation doesn't rule "
            "out, it is NOT assigned severity 0 -- it gets missing_filters severity 3 "
            "(moderate: wrong only once the exact-match assumption the schema doesn't "
            "actually guarantee is violated). Index 1 adds an extra column (contained, "
            "severity 2) and index 2 is a case-sensitive literal mismatch that returns "
            "zero rows against real data since SQLite string equality is case-sensitive "
            "by default (severe, severity 5). Non-zero severities are {3, 2, 5} -- 3 "
            "candidates spanning both bands -- so no new_candidates are needed."
        ),
    },
]


def _format_few_shot() -> str:
    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        pool_desc = "\n".join(
            f"[{j}] {c['sql']}" + ("  <-- SAME RESULT AS sql_good (verified)" if c["matches_gold"] else "")
            for j, c in enumerate(ex["pool"])
        )
        blocks.append(
            f"Example {i}\n"
            f"Question: {ex['question']}\n"
            f"sql_good (already correct, do not change): {ex['sql_good']}\n"
            f"sql_context_clean (schema only, no sample data): {json.dumps(ex['sql_context_clean'])}\n"
            f"Current sql_bad candidate pool:\n{pool_desc}\n"
            f"Correct tool call input: {json.dumps(ex['response'])}\n"
            f"Why: {ex['explanation']}"
        )
    return "\n\n".join(blocks)


SYSTEM_PROMPT = f"""You are a meticulous SQL reviewer building training data for a \
text-to-SQL reward model. You are given a (question, sql_good, sql_context_clean) \
triple and a pool of sql_bad candidates that were generated by asking a small local \
model (llama3.2) to answer the same question -- some of them are real mistakes, at \
least one may be flagged as producing the same result as sql_good on this row's \
actual data (a fact verified by real SQL execution, given to you as ground truth, not \
a guess).

Your job, for every row:
1. Assign each EXISTING candidate a reason category and a severity (0-5).
2. Decide whether the pool needs more candidates to reach reasonable severity/category \
coverage, and if so, propose up to 3 new ones.

# Reason taxonomy

{_reasons_block()}

# {SEVERITY_RUBRIC}

# {COVERAGE_RULE}

# Examples

{_format_few_shot()}

Call the `emit_severity_review` tool exactly once with your result. Do not include \
any text outside the tool call."""


def build_candidate_pool_text(record: dict[str, Any]) -> str:
    sql_bad = record.get("sql_bad") or []
    if not sql_bad:
        return "(empty -- llama3.2 produced no usable candidates for this row; you should add new_candidates)"
    lines = []
    for i, b in enumerate(sql_bad):
        hint = "  <-- SAME RESULT AS sql_good (verified)" if b.get("matches_gold") else ""
        lines.append(f"[{i}] {b['sql']}{hint}")
    return "\n".join(lines)


def build_user_message(record: dict[str, Any]) -> str:
    return (
        f"Question: {record['question']}\n"
        f"sql_good (already correct, do not change): {record['sql_good']}\n"
        f"sql_context_clean (schema only, no sample data): {json.dumps(record.get('sql_context_clean', []))}\n"
        f"Current sql_bad candidate pool:\n{build_candidate_pool_text(record)}"
    )


# ---------------------------------------------------------------------------
# Tool / JSON schema.
# ---------------------------------------------------------------------------

EXISTING_REVIEW_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "index": {
            "type": "integer",
            "description": "0-based index into the candidate pool shown to you -- identifies which existing candidate this review applies to.",
        },
        "reason": {"type": "string", "enum": EXISTING_REASON_NAMES, "description": "Mistake category for this candidate."},
        "severity": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "0-5 -- see the severity rubric. Only use 0 if the candidate is marked 'SAME RESULT AS sql_good' above.",
        },
    },
    "required": ["index", "reason", "severity"],
    "additionalProperties": False,
}

NEW_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "description": "A new, genuinely distinguishing sql_bad candidate not already in the pool shown to you."},
        "reason": {
            "type": "string",
            "enum": BASE_REASON_NAMES,
            "description": "One of the 5 base categories only -- 'llama' is not a valid choice for a candidate you're inventing on purpose.",
        },
        "severity": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "1-5 only -- a deliberately-added candidate must be a real, distinguishing mistake.",
        },
        # Not used for anything (new_candidates are matched by content, never by
        # position) -- declared explicitly so a stray "index" (copied out of habit
        # from the "existing" array's shape) doesn't fail validation outright, a real
        # gap caught during testing. additionalProperties stays False otherwise, so
        # genuinely unexpected fields still fail loudly.
        "index": {"type": "integer"},
    },
    "required": ["sql", "reason", "severity"],
    "additionalProperties": False,
}

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "existing": {
            "type": "array",
            "items": EXISTING_REVIEW_ITEM_SCHEMA,
            "description": "One entry per candidate in the pool shown to you, covering every index exactly once, in any order.",
        },
        "new_candidates": {
            "type": "array",
            "maxItems": 3,
            "items": NEW_CANDIDATE_SCHEMA,
            "description": "0-3 additional sql_bad candidates to fill thin severity/category coverage. Empty if the pool already has enough.",
        },
    },
    "required": ["existing", "new_candidates"],
    "additionalProperties": False,
}

TOOL_NAME = "emit_severity_review"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Assign a reason and severity (0-5) to every existing sql_bad candidate, and optionally propose new ones to fill thin coverage.",
    "input_schema": RESULT_SCHEMA,
}

TOOL_CHOICE = {"type": "tool", "name": TOOL_NAME}

_VALIDATOR = jsonschema.Draft7Validator(RESULT_SCHEMA)


def validate_result(data: dict[str, Any]) -> None:
    _VALIDATOR.validate(data)


def request_params(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1536,
        "system": SYSTEM_PROMPT,
        "tools": [TOOL_SCHEMA],
        "tool_choice": TOOL_CHOICE,
        "messages": [{"role": "user", "content": build_user_message(record)}],
    }


# ---------------------------------------------------------------------------
# Local verification / merge -- "trust the LLM structurally, verify execution
# locally", mirroring gen_training_data.enhance_record / fix_sql_bad.apply_fix.
# ---------------------------------------------------------------------------

FALLBACK_RESULT: dict[str, Any] = {"existing": [], "new_candidates": []}


def _verify_new_candidate(record: dict[str, Any], sql: str) -> tuple[bool, bool]:
    """(executes, matches_gold) for a Claude-proposed new candidate, via
    execute_with_context's cached, self-healing connection for this row's
    sql_context_path (evicted and rebuilt automatically the instant a mutating
    statement runs against it -- see sql_exec.execute_with_context) -- mirrors
    corrupt.verify_candidate's Counter-equality semantics without needing to manage a
    raw sqlite3.Connection here."""
    context = record.get("sql_context") or ()
    path = record.get("sql_context_path")
    candidate_exec = execute_with_context(context, path, sql)
    if not candidate_exec.success:
        return False, False
    good_exec = execute_with_context(context, path, record["sql_good"])
    if not good_exec.success:
        return True, False
    return True, Counter(candidate_exec.rows or ()) == Counter(good_exec.rows or ())


def apply_severity_review(record: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    existing = record.get("sql_bad") or []
    n_existing = len(existing)
    reviewed_by_index = {
        item["index"]: item for item in result.get("existing", []) if 0 <= item.get("index", -1) < n_existing
    }

    final_sql_bad: list[dict[str, Any]] = []
    n_hint_confirmed = 0
    n_hint_overridden = 0
    n_clamped = 0
    for i, b in enumerate(existing):
        review = reviewed_by_index.get(i)
        matches_gold = bool(b.get("matches_gold"))
        if review:
            reason = review["reason"]
            severity = review["severity"]
            if severity == 0 and not matches_gold:
                # The hint proves this candidate's result DIFFERS from sql_good on
                # real execution -- a severity=0 claim here is a plain factual
                # error, not a judgment call about generalization. Clamp, don't
                # trust it.
                severity = 1
                n_clamped += 1
            elif matches_gold and severity == 0:
                n_hint_confirmed += 1
            elif matches_gold and severity > 0:
                # The LLM is overriding the "same result" hint -- e.g. it
                # recognized a real mistake this sample's data doesn't happen to
                # expose (see the module docstring / severity rubric). Trusted
                # as-is, not clamped.
                n_hint_overridden += 1
        else:
            # No LLM review for this index -- nothing to trust, fall back to the
            # local hint alone.
            reason = "llama"
            severity = 0 if matches_gold else 1
        final_sql_bad.append({"sql": b["sql"], "reason": reason, "severity": severity})

    n_new_accepted = 0
    n_new_rejected = 0
    existing_texts = {c["sql"] for c in final_sql_bad} | {record["sql_good"]}
    for new in result.get("new_candidates", []):
        sql = (new.get("sql") or "").strip()
        if not sql or sql in existing_texts:
            n_new_rejected += 1
            continue
        executes, matches_gold = _verify_new_candidate(record, sql)
        if not executes:
            n_new_rejected += 1
            continue
        existing_texts.add(sql)
        reason = new.get("reason") if new.get("reason") in BASE_REASON_NAMES else "llama"
        severity = 0 if matches_gold else max(1, int(new.get("severity") or 1))
        final_sql_bad.append({"sql": sql, "reason": reason, "severity": severity})
        n_new_accepted += 1

    merged = dict(record)
    merged["sql_bad"] = final_sql_bad
    stats = {
        "n_hint_confirmed": n_hint_confirmed,
        "n_hint_overridden": n_hint_overridden,
        "n_clamped": n_clamped,
        "n_new_accepted": n_new_accepted,
        "n_new_rejected": n_new_rejected,
    }
    return merged, stats


def print_severity_summary(label: str, stats_list: list[dict[str, int]], n_rows: int) -> None:
    totals = {"n_hint_confirmed": 0, "n_hint_overridden": 0, "n_clamped": 0, "n_new_accepted": 0, "n_new_rejected": 0}
    for s in stats_list:
        for k in totals:
            totals[k] += s[k]
    print(
        f"{label}: {len(stats_list)}/{n_rows} row(s) reviewed -- "
        f"{totals['n_hint_confirmed']} candidate(s) confirmed severity 0 (LLM agreed with local hint), "
        f"{totals['n_hint_overridden']} candidate(s) kept a nonzero severity DESPITE a same-result hint "
        f"(LLM judged it a coincidental match, not true equivalence), "
        f"{totals['n_clamped']} LLM-claimed-0 contradiction(s) clamped to 1, "
        f"{totals['n_new_accepted']} new candidate(s) accepted, {totals['n_new_rejected']} rejected."
    )


# ---------------------------------------------------------------------------
# test: synchronous single calls. Required before ever running submit.
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    records = load_records(args.input)[: args.limit]
    if not records:
        print("No records to test.")
        return

    results = []
    stats_list = []
    for i, record in enumerate(records, 1):
        n_pool = len(record.get("sql_bad") or [])
        print(f"[{i}/{len(records)}] ({n_pool} candidate(s)) {record['question'][:80]}")
        params = request_params(record)
        message = client.messages.create(**params)
        try:
            result = extract_tool_input(message)
            validate_result(result)
        except (ValueError, jsonschema.ValidationError) as exc:
            print(f"  FAILED: {exc}")
            continue
        merged, stats = apply_severity_review(record, result)
        results.append(merged)
        stats_list.append(stats)
        print(json.dumps(merged, indent=2))

    print_severity_summary("Test run", stats_list, len(records))

    if args.output:
        write_jsonl(results, args.output)
        print(f"\nWrote {len(results)} test result(s) to {args.output}")


# ---------------------------------------------------------------------------
# submit: build and submit a Message Batch for every row in the input.
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No records to submit.")
        return

    requests = []
    records_by_custom_id: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(records):
        custom_id = make_custom_id(i, len(records))
        requests.append({"custom_id": custom_id, "params": request_params(record)})
        records_by_custom_id[custom_id] = record

    check_batch_limits(requests)

    batch = client.messages.batches.create(requests=requests)
    save_state(batch.id, args.input, records_by_custom_id)

    print(f"Submitted batch {batch.id} with {len(requests)} requests (status: {batch.processing_status}).")
    print(
        "Collect results later with:\n"
        f"  python -m walt.rm.data.synth.enhance_severity_dataset collect --batch-id {batch.id} --output <path>"
    )


# ---------------------------------------------------------------------------
# collect: poll a batch until it ends, merge, write the full output JSONL.
# Every input row is guaranteed a valid (reason, severity)-populated sql_bad list in
# the output, even rows whose LLM call failed/errored/returned invalid output -- those
# fall back to a local-only categorization (see apply_severity_review/FALLBACK_RESULT)
# rather than being left in their structurally-incomplete stage-1 shape.
# ---------------------------------------------------------------------------


def _index_from_custom_id(custom_id: str) -> int:
    return int(custom_id.split("-")[1])


def cmd_collect(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    state = load_state(args.batch_id)
    records_by_custom_id: dict[str, dict[str, Any]] = state["records"]
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
    stats_list = []
    for item in client.messages.batches.results(args.batch_id):
        record = records_by_custom_id.get(item.custom_id)
        if record is None:
            print(f"WARNING: unknown custom_id {item.custom_id!r}, skipping.")
            continue
        idx = _index_from_custom_id(item.custom_id)

        result = None
        if item.result.type == "succeeded":
            try:
                candidate_result = extract_tool_input(item.result.message)
                validate_result(candidate_result)
                result = candidate_result
                counts["succeeded"] += 1
            except (ValueError, jsonschema.ValidationError) as exc:
                counts["invalid"] += 1
                print(f"  {item.custom_id}: invalid output ({exc}) -- applying local-only fallback")
        else:
            counts[item.result.type] = counts.get(item.result.type, 0) + 1
            print(f"  {item.custom_id}: {item.result.type} -- applying local-only fallback")

        merged, stats = apply_severity_review(record, result or FALLBACK_RESULT)
        all_records[idx] = merged
        stats_list.append(stats)

    write_jsonl(all_records, args.output)
    print(f"\nWrote {len(all_records)} row(s) to {args.output} (full input preserved, order unchanged).")
    print(f"Counts: {counts}")
    print_severity_summary("This collect run", stats_list, len(records_by_custom_id))
    n_fallback = len(records_by_custom_id) - counts["succeeded"]
    if n_fallback:
        print(f"{n_fallback} row(s) used the local-only fallback categorization (see counts above).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_p = subparsers.add_parser("test", help="Run a few rows through synchronous single calls. Required before submit.")
    test_p.add_argument("--input", type=Path, required=True, help="Input JSONL from build_severity_dataset.py")
    test_p.add_argument("--limit", type=int, default=5, help="Number of rows to test")
    test_p.add_argument("--output", type=Path, default=None, help="Optionally write test results to a JSONL file")
    test_p.set_defaults(func=cmd_test)

    submit_p = subparsers.add_parser("submit", help="Submit every row as a Message Batch.")
    submit_p.add_argument("--input", type=Path, required=True, help="Input JSONL from build_severity_dataset.py")
    submit_p.add_argument("--limit", type=int, default=None, help="Only submit the first N rows")
    submit_p.set_defaults(func=cmd_submit)

    collect_p = subparsers.add_parser("collect", help="Poll a batch and write the full output JSONL.")
    collect_p.add_argument("--batch-id", required=True, help="Batch id printed by submit")
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
