"""Uses Claude to enhance rm training data: fixes `sql_good`, generates labeled
`sql_bad` negatives, and synthesizes a SQLite `sql_context` (CREATE TABLE/INSERT INTO
statements) so sql_good can actually be executed, for reward-model training.

Input:  JSONL with {"question", "sql_good", "source", "split"} per line (see pre_process.py).
Output: JSONL with {"question", "sql_good", "sql_bad", "sql_context", "sql_context_valid",
        "source", "split"} per line, where sql_good has been corrected if needed,
        sql_bad is a list of 3-5 {"sql", "reason"} negatives, sql_context is a list of
        SQLite-compatible CREATE TABLE/INSERT INTO statements sql_good can run against,
        and sql_context_valid records whether sql_good actually executed successfully
        against that context (checked locally, no extra API call).

Modes:
    test    - run a handful of rows through synchronous single calls, for iterating
              on the prompt before spending on a full batch.
    submit  - submit the full input file as an Anthropic Message Batch job and save
              enough state locally to collect it later.
    collect - poll a submitted batch until it ends, then merge results back onto the
              original records and write the output JSONL.

Usage:
    python -m walt.rm.data.gen_training_data test --input data/output/rm_data.jsonl --limit 3
    python -m walt.rm.data.gen_training_data submit --input data/output/rm_data.jsonl
    python -m walt.rm.data.gen_training_data collect --batch-id msgbatch_xxx --output data/output/rm_enhanced.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import anthropic
import jsonschema
from dotenv import load_dotenv

from walt.utils.jsonl_io import open_jsonl
from walt.utils.sql_exec import clean_context, run_sql

load_dotenv()

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

STATE_DIR = Path(__file__).resolve().parent / ".batch_state"

# ---------------------------------------------------------------------------
# Taxonomy of mistakes a sql_bad example may be labeled with.
# ---------------------------------------------------------------------------

BAD_SQL_REASONS: list[tuple[str, str]] = [
    ("missing_filters", "Omits a WHERE/HAVING condition (or LIMIT) the question implies, returning too many or the wrong rows."),
    ("wrong_aggregation", "Uses the wrong aggregate function, groups by the wrong column, or aggregates over the wrong column entirely."),
    ("unsafe_patterns", "Uses a risky/imprecise pattern like SELECT * or an unqualified cross join instead of the specific columns/join condition the question calls for."),
    ("misjoined_tables", "Joins the wrong pair of tables, joins on the wrong column, or uses the wrong join type (e.g. INNER instead of LEFT), producing an incorrect result set."),
]
REASON_NAMES = [name for name, _ in BAD_SQL_REASONS]


def _reasons_block() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in BAD_SQL_REASONS)


# ---------------------------------------------------------------------------
# Few-shot examples, embedded directly in the (cacheable) system prompt.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "question": "What is the total revenue by category last month?",
        "sql_good": "SELECT category, SUM(price * units) FROM sales WHERE sale_date >= '2024-01-01' GROUP BY category",
        "response": {
            "sql_good": (
                "SELECT category, SUM(price * units) AS revenue FROM sales "
                "WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01' GROUP BY category"
            ),
            "sql_context": [
                "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT NOT NULL, price REAL NOT NULL, units INTEGER NOT NULL, sale_date TEXT NOT NULL)",
                "INSERT INTO sales (category, price, units, sale_date) VALUES ('Electronics', 100.0, 3, '2024-01-05')",
                "INSERT INTO sales (category, price, units, sale_date) VALUES ('Electronics', 50.0, 2, '2023-12-20')",
                "INSERT INTO sales (category, price, units, sale_date) VALUES ('Books', 20.0, 5, '2024-01-15')",
            ],
            "sql_bad": [
                {
                    "sql": "SELECT category, SUM(price * units) AS revenue FROM sales GROUP BY category",
                    "reason": "missing_filters",
                },
                {
                    "sql": (
                        "SELECT category, SUM(price) AS revenue FROM sales "
                        "WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01' GROUP BY category"
                    ),
                    "reason": "wrong_aggregation",
                },
                {
                    "sql": "SELECT * FROM sales WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01'",
                    "reason": "unsafe_patterns",
                },
            ],
        },
    },
    {
        "question": "List each department's name along with the number of employees in it, highest first.",
        "sql_good": (
            "SELECT d.name, COUNT(e.id) FROM department d JOIN employee e ON e.department_id = d.id "
            "GROUP BY d.name ORDER BY COUNT(e.id) DESC"
        ),
        "response": {
            "sql_good": (
                "SELECT d.name, COUNT(e.id) AS employee_count FROM department d "
                "LEFT JOIN employee e ON e.department_id = d.id "
                "GROUP BY d.name ORDER BY employee_count DESC"
            ),
            "sql_context": [
                "CREATE TABLE department (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
                "CREATE TABLE employee (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER)",
                "INSERT INTO department (name) VALUES ('Engineering')",
                "INSERT INTO department (name) VALUES ('Sales')",
                "INSERT INTO department (name) VALUES ('Marketing')",
                "INSERT INTO employee (name, department_id) VALUES ('Alice', 1)",
                "INSERT INTO employee (name, department_id) VALUES ('Bob', 1)",
                "INSERT INTO employee (name, department_id) VALUES ('Carol', 2)",
            ],
            "sql_bad": [
                {
                    "sql": (
                        "SELECT d.name, COUNT(e.id) AS employee_count FROM department d "
                        "JOIN employee e ON e.department_id = d.id GROUP BY d.name ORDER BY employee_count DESC"
                    ),
                    "reason": "misjoined_tables",
                },
                {
                    "sql": (
                        "SELECT d.name, COUNT(id) AS employee_count FROM department d "
                        "LEFT JOIN employee e ON e.department_id = d.id GROUP BY d.name ORDER BY employee_count DESC"
                    ),
                    "reason": "wrong_aggregation",
                },
                {
                    "sql": (
                        "SELECT * FROM department d LEFT JOIN employee e ON e.department_id = d.id"
                    ),
                    "reason": "unsafe_patterns",
                },
            ],
        },
    },
]


def _format_few_shot() -> str:
    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        blocks.append(
            f"Example {i}\n"
            f"Question: {ex['question']}\n"
            f"sql_good (candidate, may be wrong): {ex['sql_good']}\n"
            f"Correct tool call input: {json.dumps(ex['response'])}"
        )
    return "\n\n".join(blocks)


SYSTEM_PROMPT = f"""You are a meticulous SQL reviewer building training data for a text-to-SQL reward model.

For each (question, sql_good) pair you are given:
1. Check the candidate `sql_good` for correctness against the question. If it has a bug \
(wrong result, wrong columns, wouldn't execute, etc.), rewrite it so it correctly and \
idiomatically answers the question. If it is already correct, return it unchanged \
(only cosmetic cleanup, e.g. consistent aliasing, is allowed). Target SQLite as the \
dialect — these rows will be executed against an in-memory SQLite database.
2. Produce `sql_context`: SQLite-compatible CREATE TABLE and INSERT INTO statements, in \
execution order, defining a minimal schema and a small amount of sample data such that \
`sql_good` can be run directly against them and returns a meaningful, non-trivial result \
(at least one row, unless the question's own logic legitimately returns none). Every \
WHERE/HAVING condition, JOIN key, and aggregation grouping that appears anywhere in \
`sql_good` or in any `sql_bad` example below must actually matter for the sample data you \
provide — include at least one row each such condition/join/grouping would include or \
exclude *differently* than omitting or changing it would. An empty `sql_good` result is \
only acceptable when the question's own semantics has no answer; otherwise treat it as a \
signal to add more rows, not to move on. If `sql_good` is itself a DDL statement \
(CREATE/ALTER/DROP TABLE) rather than a query over existing data, return an empty \
`sql_context` array.
3. Produce between 3 and 5 "sql_bad" examples: plausible-looking SQL queries that a \
competent-but-imperfect model might write for the same question, each containing exactly \
one meaningful problem. Bad examples must be diverse (no repeated mistake type within the \
same question) and must be *close misses* that a careless reviewer could mistake for \
correct — not obviously broken SQL. Before finalizing each one, confirm mentally that \
executing it against `sql_context` would return a genuinely different result than \
`sql_good` — not just different-looking SQL. A cosmetic rewrite (renamed alias, reordered \
columns/clauses, a logically equivalent condition) is not a valid `sql_bad` even if the \
text looks different; if the sample data isn't rich/diverse enough to make a mistake's \
effect observable, add the rows needed in `sql_context` rather than keeping a \
non-distinguishing example.
4. Tag each sql_bad example with the single reason category from this list that best \
explains its mistake:

{_reasons_block()}

Not every category needs to be represented for every question — only use the ones that \
are natural and plausible for this specific query and schema. Don't force-fit a category \
just to cover the list; pick whichever 3-5 distinct mistakes a real model would plausibly make here.

# Examples

{_format_few_shot()}

Call the `emit_sql_review` tool exactly once with your result. Do not include any text \
outside the tool call."""


def build_user_message(question: str, sql_good: str) -> str:
    return f"Question: {question}\nsql_good (candidate, may be wrong): {sql_good}"


# ---------------------------------------------------------------------------
# Bad-only variant: for rows that already carry a verified sql_context (e.g. gretel,
# see gretel.py), skip sql_good correction and sql_context synthesis entirely and just
# generate sql_bad negatives against the existing schema.
# ---------------------------------------------------------------------------

BAD_ONLY_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "question": "What is the total volume of timber sold by each salesperson, sorted by salesperson?",
        "sql_good": (
            "SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales "
            "JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id "
            "GROUP BY salesperson_id, name ORDER BY total_volume DESC"
        ),
        "sql_context": [
            "CREATE TABLE salesperson (salesperson_id INTEGER, name TEXT, region TEXT)",
            "INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South')",
            "CREATE TABLE timber_sales (sales_id INTEGER, salesperson_id INTEGER, volume REAL, sale_date DATE)",
            "INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01')",
        ],
        "response": {
            "sql_bad": [
                {
                    "sql": (
                        "SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales "
                        "JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id "
                        "GROUP BY salesperson_id"
                    ),
                    "reason": "wrong_aggregation",
                },
                {
                    "sql": "SELECT salesperson_id, name, volume FROM timber_sales JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id",
                    "reason": "unsafe_patterns",
                },
                {
                    "sql": (
                        "SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales "
                        "JOIN salesperson ON timber_sales.sales_id = salesperson.salesperson_id "
                        "GROUP BY salesperson_id, name ORDER BY total_volume DESC"
                    ),
                    "reason": "misjoined_tables",
                },
            ],
        },
    },
]


def _format_bad_only_few_shot() -> str:
    blocks = []
    for i, ex in enumerate(BAD_ONLY_FEW_SHOT_EXAMPLES, 1):
        blocks.append(
            f"Example {i}\n"
            f"Question: {ex['question']}\n"
            f"sql_good (already correct, do not change): {ex['sql_good']}\n"
            f"sql_context (already correct, do not change): {json.dumps(ex['sql_context'])}\n"
            f"Correct tool call input: {json.dumps(ex['response'])}"
        )
    return "\n\n".join(blocks)


BAD_ONLY_SYSTEM_PROMPT = f"""You are a meticulous SQL reviewer building training data for a text-to-SQL reward model.

For each (question, sql_good, sql_context) triple you are given, `sql_good` is already \
verified to correctly answer the question against `sql_context` (SQLite CREATE TABLE/ \
INSERT INTO statements) — do not modify either one, and do not return them.

Produce between 3 and 5 "sql_bad" examples: plausible-looking SQL queries against the \
same `sql_context` that a competent-but-imperfect model might write for the same \
question, each containing exactly one meaningful problem. Bad examples must be diverse \
(no repeated mistake type within the same question) and must be *close misses* that a \
careless reviewer could mistake for correct — not obviously broken SQL. Every sql_bad \
query must reference only tables/columns that actually exist in the given sql_context. \
Before finalizing each one, confirm mentally that executing it against the given \
`sql_context` (as-is — you cannot add rows here) would return a genuinely different \
result than `sql_good`, not just different-looking SQL. A cosmetic rewrite (renamed \
alias, reordered columns/clauses, a logically equivalent condition) is not a valid \
sql_bad even if the text looks different; if the sample data is too sparse or uniform \
for a mistake type to actually change the result, pick a different mistake that this \
specific data *does* expose rather than keeping a non-distinguishing example.

Tag each sql_bad example with the single reason category from this list that best \
explains its mistake:

{_reasons_block()}

Not every category needs to be represented for every question — only use the ones that \
are natural and plausible for this specific query and schema. Don't force-fit a category \
just to cover the list; pick whichever 3-5 distinct mistakes a real model would plausibly make here.

# Examples

{_format_bad_only_few_shot()}

Call the `emit_sql_bad` tool exactly once with your result. Do not include any text \
outside the tool call."""


def build_user_message_bad_only(question: str, sql_good: str, sql_context: list[str]) -> str:
    return (
        f"Question: {question}\n"
        f"sql_good (already correct, do not change): {sql_good}\n"
        f"sql_context (already correct, do not change): {json.dumps(sql_context)}"
    )


# ---------------------------------------------------------------------------
# Tool / JSON schema — this is what forces the model's output into our shape.
# ---------------------------------------------------------------------------

SQL_BAD_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "description": "A plausible but incorrect SQL query."},
        "reason": {
            "type": "string",
            "enum": REASON_NAMES,
            "description": "Category of mistake this example demonstrates.",
        },
    },
    "required": ["sql", "reason"],
    "additionalProperties": False,
}

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql_good": {
            "type": "string",
            "description": "Corrected SQL query that correctly answers the question.",
        },
        "sql_context": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "SQLite-compatible CREATE TABLE and INSERT INTO statements, in execution "
                "order, defining a minimal schema and sample data sql_good can run "
                "directly against. Empty if sql_good is itself a DDL statement."
            ),
        },
        "sql_bad": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": SQL_BAD_ITEM_SCHEMA,
        },
    },
    "required": ["sql_good", "sql_context", "sql_bad"],
    "additionalProperties": False,
}

TOOL_NAME = "emit_sql_review"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Return the corrected SQL, its executable SQLite context, and a set of labeled incorrect SQL variants for this question.",
    "input_schema": RESULT_SCHEMA,
}

TOOL_CHOICE = {"type": "tool", "name": TOOL_NAME}

_VALIDATOR = jsonschema.Draft7Validator(RESULT_SCHEMA)

BAD_ONLY_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql_bad": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": SQL_BAD_ITEM_SCHEMA,
        },
    },
    "required": ["sql_bad"],
    "additionalProperties": False,
}

BAD_ONLY_TOOL_NAME = "emit_sql_bad"

BAD_ONLY_TOOL_SCHEMA = {
    "name": BAD_ONLY_TOOL_NAME,
    "description": "Return a set of labeled incorrect SQL variants for this question, against the given schema.",
    "input_schema": BAD_ONLY_RESULT_SCHEMA,
}

BAD_ONLY_TOOL_CHOICE = {"type": "tool", "name": BAD_ONLY_TOOL_NAME}

_BAD_ONLY_VALIDATOR = jsonschema.Draft7Validator(BAD_ONLY_RESULT_SCHEMA)


def validate_result(data: dict[str, Any], has_context: bool) -> None:
    """Raises jsonschema.ValidationError if data doesn't conform to the schema for this
    record's mode (has_context picks BAD_ONLY_RESULT_SCHEMA vs the full RESULT_SCHEMA —
    see request_params)."""
    validator = _BAD_ONLY_VALIDATOR if has_context else _VALIDATOR
    validator.validate(data)


# ---------------------------------------------------------------------------
# Empty-result enrichment: a follow-up call for rows where sql_good executes but
# returns zero rows (SYSTEM_PROMPT/BAD_ONLY_SYSTEM_PROMPT ask the model to avoid this
# up front, but it's not enforced by the schema, so it's also checked and actively
# repaired here rather than just hoped for).
# ---------------------------------------------------------------------------

ENRICH_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql_context_additions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "One or more additional INSERT INTO statements, against the EXISTING "
                "tables only (same columns/types, no new CREATE TABLE, don't touch "
                "existing rows), such that executing sql_good against the amended "
                "sql_context returns at least one row."
            ),
        },
    },
    "required": ["sql_context_additions"],
    "additionalProperties": False,
}

ENRICH_TOOL_NAME = "emit_context_enrichment"

ENRICH_TOOL_SCHEMA = {
    "name": ENRICH_TOOL_NAME,
    "description": "Return additional sample-data rows so sql_good returns a non-empty result.",
    "input_schema": ENRICH_RESULT_SCHEMA,
}

ENRICH_TOOL_CHOICE = {"type": "tool", "name": ENRICH_TOOL_NAME}

_ENRICH_VALIDATOR = jsonschema.Draft7Validator(ENRICH_RESULT_SCHEMA)

ENRICH_SYSTEM_PROMPT = """You are a meticulous SQL reviewer building training data for a \
text-to-SQL reward model.

You are given a (question, sql_good, sql_context) triple. sql_good is correct SQL for \
the question, but executing it against sql_context currently returns zero rows — the \
sample data doesn't actually exercise sql_good's WHERE/HAVING conditions, JOIN keys, or \
aggregation grouping. Add one or more INSERT INTO statements, against the tables already \
defined in sql_context (same columns/types — do not add CREATE TABLE statements or \
modify existing rows), containing at least one row that sql_good's conditions would \
actually match, so that running sql_good against the amended sql_context returns a \
meaningful, non-trivial result.

Call the `emit_context_enrichment` tool exactly once with your result. Do not include \
any text outside the tool call."""


def build_enrich_user_message(question: str, sql_good: str, sql_context: list[str]) -> str:
    return (
        f"Question: {question}\n"
        f"sql_good (already correct, do not change): {sql_good}\n"
        f"sql_context (currently returns zero rows for sql_good, do not remove or modify "
        f"any existing statement): {json.dumps(sql_context)}"
    )


def needs_enrichment(record: dict[str, Any]) -> bool:
    """True iff sql_good is a SELECT (or other row-returning statement) that executes
    successfully against sql_context but returns zero rows. DDL/INSERT/etc. statements
    have no result set at all (run_sql returns rows=None, not rows=()) — that's not an
    "empty result" to fix, so they're excluded here."""
    execution = run_sql(record.get("sql_context", []), record["sql_good"])
    return execution.success and execution.rows is not None and len(execution.rows) == 0


def enrich_context(client: anthropic.Anthropic, record: dict[str, Any]) -> dict[str, Any]:
    """One follow-up tool call asking for more INSERT rows so sql_good's result is
    non-empty. Appends the additions and locally re-verifies (no extra API call for
    that part); falls back to the original sql_context unchanged if the addition breaks
    sql_good or still returns zero rows — no retry loop, same trust-then-verify
    convention as enhance_record."""
    params = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": ENRICH_SYSTEM_PROMPT,
        "tools": [ENRICH_TOOL_SCHEMA],
        "tool_choice": ENRICH_TOOL_CHOICE,
        "messages": [
            {
                "role": "user",
                "content": build_enrich_user_message(record["question"], record["sql_good"], record.get("sql_context", [])),
            }
        ],
    }
    try:
        message = client.messages.create(**params)
        result = extract_tool_input(message)
        _ENRICH_VALIDATOR.validate(result)
    except (ValueError, jsonschema.ValidationError) as exc:
        print(f"  enrichment call failed, keeping original sql_context: {exc}")
        return record

    additions = result["sql_context_additions"]
    candidate_context = list(record.get("sql_context", [])) + additions
    execution = run_sql(candidate_context, record["sql_good"])
    if not execution.success or not execution.rows:
        print("  enrichment did not produce a non-empty result, keeping original sql_context")
        return record

    enriched = dict(record)
    enriched["sql_context"] = candidate_context
    enriched["sql_context_valid"] = True
    return enriched


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """input_path may end in .gz for a gzip-compressed file -- read transparently either way."""
    records = []
    with open_jsonl(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """output_path may end in .gz to write gzip-compressed instead of plain text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_jsonl(output_path, "wt") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def make_custom_id(index: int, total: int) -> str:
    width = max(6, len(str(total)))
    return f"row-{index:0{width}d}"


def has_context(record: dict[str, Any]) -> bool:
    """True for rows that already carry a (non-empty) sql_context — e.g. gretel (see
    gretel.py) — and so should skip sql_good correction / sql_context synthesis and only
    get sql_bad generated against what's already there."""
    return bool(record.get("sql_context"))


def request_params(record: dict[str, Any]) -> dict[str, Any]:
    question, sql_good = record["question"], record["sql_good"]
    if has_context(record):
        return {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "system": BAD_ONLY_SYSTEM_PROMPT,
            "tools": [BAD_ONLY_TOOL_SCHEMA],
            "tool_choice": BAD_ONLY_TOOL_CHOICE,
            "messages": [{"role": "user", "content": build_user_message_bad_only(question, sql_good, record["sql_context"])}],
        }
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "tools": [TOOL_SCHEMA],
        "tool_choice": TOOL_CHOICE,
        "messages": [{"role": "user", "content": build_user_message(question, sql_good)}],
    }


def extract_tool_input(message: anthropic.types.Message) -> dict[str, Any]:
    # tool_choice forces exactly one tool call (emit_sql_review or emit_sql_bad,
    # depending on request_params' mode for this record), so any tool_use block is it.
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError(f"No tool_use block in response (stop_reason={message.stop_reason!r})")


def enhance_record(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    merged = dict(record)
    merged["sql_bad"] = result["sql_bad"]
    if "sql_good" in result:
        merged["sql_good"] = result["sql_good"]
    if "sql_context" in result:
        merged["sql_context"] = result["sql_context"]
    # Local verification, no extra API call: does sql_good actually execute against
    # sql_context (freshly synthesized, or reused as-is when the row already had one)?
    execution = run_sql(merged.get("sql_context", []), merged["sql_good"])
    merged["sql_context_valid"] = execution.success
    # Schema-only view (CREATE TABLE etc, no INSERT rows) — what the LLM/RM should see
    # instead of the full context (see clean_context()).
    merged["sql_context_clean"] = list(clean_context(merged.get("sql_context", [])))
    return merged


def is_llm_ready(record: dict[str, Any]) -> bool:
    """False for rows whose sql_context is already known non-executable
    (sql_context_valid is False — e.g. a gretel row with a wrong-dialect or otherwise
    broken schema, see gretel.py). In bad-only mode the LLM is instructed not to touch
    sql_good/sql_context (see BAD_ONLY_SYSTEM_PROMPT), so it can't fix these — spending a
    call to generate sql_bad against a context that will never execute wastes money for
    a row evaluate.py will exclude anyway (see its sql_context_valid filter)."""
    return record.get("sql_context_valid") is not False


def filter_llm_ready(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept = [r for r in records if is_llm_ready(r)]
    return kept, len(records) - len(kept)


def print_context_summary(records: list[dict[str, Any]]) -> None:
    """Aggregate pass-rate signal, separate from the per-row sql_bad output: how many
    sql_good rows actually executed against their generated sql_context, so a data-gen
    run can be judged trustworthy (or not) before sinking a training run into it."""
    total = len(records)
    if not total:
        return
    valid = sum(1 for r in records if r.get("sql_context_valid"))
    print(f"\nsql_good execution check: {valid}/{total} passed ({100 * valid / total:.1f}%)")


# ---------------------------------------------------------------------------
# test: synchronous single calls, for iterating on the prompt cheaply.
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    records, skipped = filter_llm_ready(load_records(args.input))
    records = records[: args.limit]
    if skipped:
        print(f"Skipping {skipped} row(s) with a known-invalid sql_context (no LLM call).")
    if not records:
        print("No records to test.")
        return

    results = []
    enriched_count = 0
    for i, record in enumerate(records, 1):
        mode = "sql_bad-only (context reused)" if has_context(record) else "full"
        print(f"[{i}/{len(records)}] ({mode}) {record['question'][:80]}")
        params = request_params(record)
        message = client.messages.create(**params)
        try:
            result = extract_tool_input(message)
            validate_result(result, has_context(record))
        except (ValueError, jsonschema.ValidationError) as exc:
            print(f"  FAILED: {exc}")
            continue
        merged = enhance_record(record, result)
        if needs_enrichment(merged):
            print("  sql_good returned an empty result; asking for context enrichment...")
            before = merged
            merged = enrich_context(client, merged)
            if merged is not before:
                enriched_count += 1
        results.append(merged)
        print(json.dumps(merged, indent=2))

    print_context_summary(results)
    if enriched_count:
        print(f"Enriched {enriched_count} row(s) that initially returned an empty sql_good result.")

    if args.output:
        write_jsonl(results, args.output)
        print(f"\nWrote {len(results)} test result(s) to {args.output}")


# ---------------------------------------------------------------------------
# submit: build and submit a Message Batch for every record in the input.
# ---------------------------------------------------------------------------

# https://docs.anthropic.com/en/api/creating-message-batches — limits as of writing;
# verify against current docs if this starts failing unexpectedly.
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 256 * 1024 * 1024


def check_batch_limits(requests: list[dict[str, Any]]) -> None:
    if len(requests) > MAX_BATCH_REQUESTS:
        raise ValueError(
            f"Batch has {len(requests)} requests, which exceeds the "
            f"{MAX_BATCH_REQUESTS}-request-per-batch limit. Split the input and submit in chunks."
        )
    size = len(json.dumps(requests).encode("utf-8"))
    if size > MAX_BATCH_BYTES:
        raise ValueError(
            f"Batch payload is {size / 1024 / 1024:.1f} MB, which exceeds the "
            f"{MAX_BATCH_BYTES / 1024 / 1024:.0f} MB per-batch limit. Split the input and submit in chunks."
        )


def state_path(batch_id: str) -> Path:
    return STATE_DIR / f"{batch_id}.json"


def save_state(batch_id: str, input_path: Path, records_by_custom_id: dict[str, dict[str, Any]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {"input_path": str(input_path), "records": records_by_custom_id}
    state_path(batch_id).write_text(json.dumps(state))


def load_state(batch_id: str) -> dict[str, Any]:
    path = state_path(batch_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No local state for batch {batch_id} at {path}. "
            "Was it submitted from this machine with `submit`?"
        )
    return json.loads(path.read_text())


def cmd_submit(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    records, skipped = filter_llm_ready(load_records(args.input))
    if args.limit:
        records = records[: args.limit]
    if skipped:
        print(f"Skipping {skipped} row(s) with a known-invalid sql_context (no LLM call).")
    if not records:
        print("No records to submit.")
        return

    requests = []
    records_by_custom_id: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(records):
        custom_id = make_custom_id(i, len(records))
        requests.append(
            {
                "custom_id": custom_id,
                "params": request_params(record),
            }
        )
        records_by_custom_id[custom_id] = record

    check_batch_limits(requests)

    batch = client.messages.batches.create(requests=requests)
    save_state(batch.id, args.input, records_by_custom_id)

    reused = sum(1 for r in records if has_context(r))
    print(f"Submitted batch {batch.id} with {len(requests)} requests (status: {batch.processing_status}); {reused}/{len(requests)} reuse an existing sql_context (sql_bad-only).")
    print(f"Collect results later with:\n  python -m walt.rm.data.gen_training_data collect --batch-id {batch.id} --output <path>")


# ---------------------------------------------------------------------------
# collect: poll a batch until it ends, merge results, write output JSONL.
# ---------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    state = load_state(args.batch_id)
    records_by_custom_id: dict[str, dict[str, Any]] = state["records"]

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

    output_records = []
    counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, "invalid": 0}
    for item in client.messages.batches.results(args.batch_id):
        record = records_by_custom_id.get(item.custom_id)
        if record is None:
            print(f"WARNING: unknown custom_id {item.custom_id!r}, skipping.")
            continue

        if item.result.type != "succeeded":
            counts[item.result.type] = counts.get(item.result.type, 0) + 1
            print(f"  {item.custom_id}: {item.result.type}")
            continue

        try:
            result = extract_tool_input(item.result.message)
            validate_result(result, has_context(record))
        except (ValueError, jsonschema.ValidationError) as exc:
            counts["invalid"] += 1
            print(f"  {item.custom_id}: invalid output ({exc})")
            continue

        output_records.append(enhance_record(record, result))
        counts["succeeded"] += 1

    enriched_count = 0
    for i, record in enumerate(output_records):
        if needs_enrichment(record):
            print(f"  row {i}: sql_good returned an empty result; asking for context enrichment...")
            enriched = enrich_context(client, record)
            if enriched is not record:
                output_records[i] = enriched
                enriched_count += 1

    write_jsonl(output_records, args.output)
    reused = sum(1 for r in records_by_custom_id.values() if has_context(r))
    print(f"\nWrote {len(output_records)} enhanced examples to {args.output}")
    print(f"Counts: {counts}")
    print(f"{reused}/{len(records_by_custom_id)} rows reused an existing sql_context (sql_bad-only).")
    if enriched_count:
        print(f"Enriched {enriched_count} row(s) that initially returned an empty sql_good result.")
    print_context_summary(output_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_p = subparsers.add_parser("test", help="Run a few rows through synchronous single calls.")
    test_p.add_argument("--input", type=Path, required=True, help="Input JSONL (question, sql_good, source)")
    test_p.add_argument("--limit", type=int, default=3, help="Number of rows to test")
    test_p.add_argument("--output", type=Path, default=None, help="Optionally write test results to a JSONL file")
    test_p.set_defaults(func=cmd_test)

    submit_p = subparsers.add_parser("submit", help="Submit the full input file as a Message Batch.")
    submit_p.add_argument("--input", type=Path, required=True, help="Input JSONL (question, sql_good, source)")
    submit_p.add_argument("--limit", type=int, default=None, help="Only submit the first N rows")
    submit_p.set_defaults(func=cmd_submit)

    collect_p = subparsers.add_parser("collect", help="Poll a batch and write its merged output JSONL.")
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
