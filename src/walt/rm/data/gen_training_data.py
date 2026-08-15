"""Uses Claude to enhance rm training data: fixes `sql_good` and generates labeled
`sql_bad` negatives for reward-model training.

Input:  JSONL with {"question", "sql_good", "source"} per line (see pre_process.py).
Output: JSONL with {"question", "sql_good", "sql_bad", "source"} per line, where
        sql_good has been corrected if needed and sql_bad is a list of 3-5
        {"sql", "reason"} negatives.

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

load_dotenv()

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

STATE_DIR = Path(__file__).resolve().parent / ".batch_state"

# ---------------------------------------------------------------------------
# Taxonomy of mistakes a sql_bad example may be labeled with.
# ---------------------------------------------------------------------------

BAD_SQL_REASONS: list[tuple[str, str]] = [
    ("wrong_columns_or_tables", "Selects, filters, or joins the wrong column(s) or table(s), including ambiguous/unqualified column references."),
    ("wrong_join_or_aggregation", "Uses an incorrect join type, or the wrong aggregate function/GROUP BY."),
    ("wrong_filter_or_sort", "Omits or misstates a WHERE/HAVING condition, or gets ORDER BY/LIMIT wrong."),
    ("type_or_null_handling", "Compares/casts incompatible types, or mishandles NULLs (e.g. '= NULL' instead of 'IS NULL')."),
    ("syntax_error", "Malformed SQL that would fail to parse or execute."),
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
                "WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') "
                "AND sale_date < DATE_TRUNC('month', CURRENT_DATE) GROUP BY category"
            ),
            "sql_bad": [
                {
                    "sql": "SELECT category, price * units FROM sales WHERE sale_date >= '2024-01-01' GROUP BY category",
                    "reason": "wrong_join_or_aggregation",
                },
                {
                    "sql": (
                        "SELECT category, SUM(price * units) AS revenue FROM sales GROUP BY category"
                    ),
                    "reason": "wrong_filter_or_sort",
                },
                {
                    "sql": (
                        "SELECT category, SUM(price) AS revenue FROM sales "
                        "WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') GROUP BY category"
                    ),
                    "reason": "wrong_columns_or_tables",
                },
                {
                    "sql": (
                        "SELECT category SUM(price * units) AS revenue FROM sales "
                        "WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') GROUP BY category"
                    ),
                    "reason": "syntax_error",
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
            "sql_bad": [
                {
                    "sql": (
                        "SELECT d.name, COUNT(e.id) AS employee_count FROM department d "
                        "JOIN employee e ON e.department_id = d.id GROUP BY d.name ORDER BY employee_count DESC"
                    ),
                    "reason": "wrong_join_or_aggregation",
                },
                {
                    "sql": (
                        "SELECT d.name, COUNT(e.id) AS employee_count FROM department d "
                        "LEFT JOIN employee e ON e.department_id = d.id GROUP BY d.name ORDER BY employee_count ASC"
                    ),
                    "reason": "wrong_filter_or_sort",
                },
                {
                    "sql": (
                        "SELECT d.name, COUNT(id) AS employee_count FROM department d "
                        "LEFT JOIN employee e ON e.department_id = d.id GROUP BY d.name ORDER BY employee_count DESC"
                    ),
                    "reason": "wrong_columns_or_tables",
                },
                {
                    "sql": (
                        "SELECT d.name, COUNT(e.id) AS employee_count FROM department d "
                        "LEFT JOIN employee e ON e.department_id = d.id WHERE e.department_id = d.id "
                        "GROUP BY d.name ORDER BY employee_count DESC"
                    ),
                    "reason": "type_or_null_handling",
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
(only cosmetic cleanup, e.g. consistent aliasing, is allowed).
2. Produce between 3 and 5 "sql_bad" examples: plausible-looking SQL queries that a \
competent-but-imperfect model might write for the same question, each containing exactly \
one meaningful mistake. Bad examples must be diverse (no repeated mistake type) and must be \
*close misses* that a careless reviewer could mistake for correct — not obviously broken SQL.
3. Tag each sql_bad example with the single reason category from this list that best \
explains its mistake:

{_reasons_block()}

# Examples

{_format_few_shot()}

Call the `emit_sql_review` tool exactly once with your result. Do not include any text \
outside the tool call."""


def build_user_message(question: str, sql_good: str) -> str:
    return f"Question: {question}\nsql_good (candidate, may be wrong): {sql_good}"


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
        "sql_bad": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": SQL_BAD_ITEM_SCHEMA,
        },
    },
    "required": ["sql_good", "sql_bad"],
    "additionalProperties": False,
}

TOOL_NAME = "emit_sql_review"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Return the corrected SQL and a set of labeled incorrect SQL variants for this question.",
    "input_schema": RESULT_SCHEMA,
}

TOOL_CHOICE = {"type": "tool", "name": TOOL_NAME}

_VALIDATOR = jsonschema.Draft7Validator(RESULT_SCHEMA)


def validate_result(data: dict[str, Any]) -> None:
    """Raises jsonschema.ValidationError if data doesn't conform to RESULT_SCHEMA."""
    _VALIDATOR.validate(data)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_records(input_path: Path) -> list[dict[str, Any]]:
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def make_custom_id(index: int, total: int) -> str:
    width = max(6, len(str(total)))
    return f"row-{index:0{width}d}"


def request_params(question: str, sql_good: str) -> dict[str, Any]:
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "tools": [TOOL_SCHEMA],
        "tool_choice": TOOL_CHOICE,
        "messages": [{"role": "user", "content": build_user_message(question, sql_good)}],
    }


def extract_tool_input(message: anthropic.types.Message) -> dict[str, Any]:
    for block in message.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            return block.input
    raise ValueError(f"No {TOOL_NAME} tool_use block in response (stop_reason={message.stop_reason!r})")


def enhance_record(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    merged = dict(record)
    merged["sql_good"] = result["sql_good"]
    merged["sql_bad"] = result["sql_bad"]
    return merged


# ---------------------------------------------------------------------------
# test: synchronous single calls, for iterating on the prompt cheaply.
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    records = load_records(args.input)[: args.limit]
    if not records:
        print("No records to test.")
        return

    results = []
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {record['question'][:80]}")
        params = request_params(record["question"], record["sql_good"])
        message = client.messages.create(**params)
        try:
            result = extract_tool_input(message)
            validate_result(result)
        except (ValueError, jsonschema.ValidationError) as exc:
            print(f"  FAILED: {exc}")
            continue
        merged = enhance_record(record, result)
        results.append(merged)
        print(json.dumps(merged, indent=2))

    if args.output:
        write_jsonl(results, args.output)
        print(f"\nWrote {len(results)} test result(s) to {args.output}")


# ---------------------------------------------------------------------------
# submit: build and submit a Message Batch for every record in the input.
# ---------------------------------------------------------------------------


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
        requests.append(
            {
                "custom_id": custom_id,
                "params": request_params(record["question"], record["sql_good"]),
            }
        )
        records_by_custom_id[custom_id] = record

    batch = client.messages.batches.create(requests=requests)
    save_state(batch.id, args.input, records_by_custom_id)

    print(f"Submitted batch {batch.id} with {len(requests)} requests (status: {batch.processing_status}).")
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
            validate_result(result)
        except (ValueError, jsonschema.ValidationError) as exc:
            counts["invalid"] += 1
            print(f"  {item.custom_id}: invalid output ({exc})")
            continue

        output_records.append(enhance_record(record, result))
        counts["succeeded"] += 1

    write_jsonl(output_records, args.output)
    print(f"\nWrote {len(output_records)} enhanced examples to {args.output}")
    print(f"Counts: {counts}")


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
