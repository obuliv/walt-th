#!/usr/bin/env bash
# Agent-level evaluation of the model scripts/build_best_rm.sh trained: lr_v6,
# --ignore-sql-good --drop-bad-vs-bad-pairs on the ollama-only severity dataset,
# stacked with a hard --schema-filter on top -- the config docs/experiments.md
# reports as the best found (95.3% SQL pass / 59.0% QA accuracy on the
# ollama-only val split).
#
# Also runs the constant+schema-filter baseline (zero training, just the hard
# filter) on the same val rows, so the printed funnel shows what the trained
# model adds on top of the filter alone, not just "reranked vs not reranked".
# The second run is cheap: both share the default LLM candidate cache
# (data/output/llm_cache.json), so it re-scores already-generated candidates
# instead of calling the LLM again.
#
# Each evaluate.py run's full (verbose) output is captured to a log file under
# --output-dir instead of printed live -- only the combined funnel summary
# prints to the terminal. Tail the log in another shell to watch progress:
#   tail -f experiments/run1/eval_trained.log
#
# Also writes a 3-way sample log (default 50 rows) for spot-checking actual
# question/SQL/result pairs rather than just aggregate metrics: expected
# (sql_good) vs. the trained model's pick vs. the plain-filter (constant +
# schema-filter) pick vs. no rerank at all. Built by running --sample-log on
# both eval calls with the same --sample-seed (so they land on the identical
# row subset) and merging their --samples-json dumps by row index -- see
# walt.eval.evaluate's --sample-log/--samples-json.
#
# Usage:
#   scripts/evaluate_best.sh --output-dir experiments/run1   # same dir passed to build_best_rm.sh
#   scripts/evaluate_best.sh --output-dir experiments/run1 --sample-log 100
set -euo pipefail

OUTPUT_DIR=""
SAMPLE_LOG=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --sample-log) SAMPLE_LOG="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --output-dir experiments/run1" >&2
  exit 1
fi

MODEL_PATH="$OUTPUT_DIR/rm_model.joblib"
INPUT_PATH="$OUTPUT_DIR/synth_severity_enhanced_ollamaonly.jsonl"
[[ -f "$INPUT_PATH" ]] || INPUT_PATH="$INPUT_PATH.gz"  # build_best_rm.sh writes datasets gzipped; walt.rm.model.base reads .gz transparently

[[ -f "$MODEL_PATH" ]] || { echo "Model not found: $MODEL_PATH -- run scripts/build_best_rm.sh --output-dir $OUTPUT_DIR first." >&2; exit 1; }
[[ -f "$INPUT_PATH" ]] || { echo "Dataset not found: $OUTPUT_DIR/synth_severity_enhanced_ollamaonly.jsonl[.gz] -- run scripts/build_best_rm.sh --output-dir $OUTPUT_DIR first." >&2; exit 1; }

BASELINE_OUTPUT="$OUTPUT_DIR/eval_baseline_constant_schemafilter.json"
TRAINED_OUTPUT="$OUTPUT_DIR/eval_results.json"
BASELINE_LOG="$OUTPUT_DIR/eval_baseline.log"
TRAINED_LOG="$OUTPUT_DIR/eval_trained.log"
BASELINE_SAMPLES_JSON="$OUTPUT_DIR/.eval_samples_baseline.json"
TRAINED_SAMPLES_JSON="$OUTPUT_DIR/.eval_samples_trained.json"
SAMPLE_LOG_OUTPUT="$OUTPUT_DIR/eval_samples.md"

echo "Running constant+schema-filter baseline (log: $BASELINE_LOG)..."
uv run python -m walt.eval.evaluate \
  --input "$INPUT_PATH" \
  --rm-class constant \
  --schema-filter \
  --output "$BASELINE_OUTPUT" \
  --run-name best_eval_baseline_constant_schemafilter \
  --sample-log "$SAMPLE_LOG" \
  --samples-json "$BASELINE_SAMPLES_JSON" \
  > "$BASELINE_LOG" 2>&1

echo "Running trained model + schema-filter (log: $TRAINED_LOG)..."
uv run python -m walt.eval.evaluate \
  --input "$INPUT_PATH" \
  --rm-model "$MODEL_PATH" \
  --rm-class lr_v6 \
  --schema-filter \
  --output "$TRAINED_OUTPUT" \
  --run-name best_eval \
  --sample-log "$SAMPLE_LOG" \
  --samples-json "$TRAINED_SAMPLES_JSON" \
  > "$TRAINED_LOG" 2>&1

python3 - "$BASELINE_SAMPLES_JSON" "$TRAINED_SAMPLES_JSON" "$SAMPLE_LOG_OUTPUT" <<'PY'
import json, sys

baseline_samples = {s["index"]: s for s in json.loads(open(sys.argv[1]).read())}
trained_samples = json.loads(open(sys.argv[2]).read())
out_path = sys.argv[3]

def fmt_rows(detail, label="Rows"):
    if detail is None:
        return ["_not verified (`sql_context_valid=False`)_"]
    if not detail["success"]:
        return [f"❌ execution error: `{detail['error']}`"]
    if detail["rows"] is None:
        return ["_(non-SELECT statement — no row set)_"]
    if not detail["rows"]:
        return ["_(0 rows)_"]
    lines = [f"{label}:", "```"]
    lines += [repr(tuple(r)) for r in detail["rows"]]
    if detail["n_rows"] > len(detail["rows"]):
        lines.append(f"... and {detail['n_rows'] - len(detail['rows'])} more row(s)")
    lines.append("```")
    return lines

def fmt_section(title, sql, detail, correct):
    lines = [f"**{title}:**", "```sql", sql, "```"] + fmt_rows(detail)
    if correct is not None:
        lines.append(f"QA correct: {'✅ yes' if correct else '❌ no'}")
    lines.append("")
    return lines

lines = [
    "# Eval sample log: expected vs. trained model vs. plain filter vs. no rerank",
    "",
    f"- {len(trained_samples)} row(s) sampled",
    "",
]
for t in trained_samples:
    b = baseline_samples.get(t["index"])
    lines.append(f"## Row {t['index']}: {t['question']}")
    lines += ["", "**Schema:**", "```sql", "\n".join(t["schema"]), "```", ""]
    lines += ["**Expected (`sql_good`):**", "```sql", t["sql_good"], "```"]
    lines += fmt_rows(t["reference"], label="Expected rows") if t["reference"] is not None else ["_expected rows not verified (`sql_context_valid=False` for this row)_"]
    lines.append("")
    lines += fmt_section("Trained model + schema-filter", t["with_rm"]["sql"], t["with_rm"], t["with_rm"]["correct"])
    if b is not None:
        lines += fmt_section("Plain filter (constant + schema-filter)", b["with_rm"]["sql"], b["with_rm"], b["with_rm"]["correct"])
    lines += fmt_section("No rerank (first candidate)", t["without_rm"]["sql"], t["without_rm"], t["without_rm"]["correct"])
    lines.append(f"Oracle bucket: `{t['oracle_bucket'] or 'n/a'}`")
    lines += ["", "---", ""]

with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"Sample log ({len(trained_samples)} rows): {out_path}")
PY

python3 - "$BASELINE_OUTPUT" "$TRAINED_OUTPUT" <<'PY'
import json, sys

baseline = json.loads(open(sys.argv[1]).read())
trained = json.loads(open(sys.argv[2]).read())

def line(label, stats):
    rate = f" ({stats['rate']:.1%})" if stats["rate"] is not None else ""
    print(f"  {label:<32}: {stats['count']}/{stats['total']}{rate}")

print("\n== Funnel: no rerank -> constant+schema-filter -> trained model+schema-filter ==")
print("\nSQL execution pass rate:")
line("no rerank", trained["sql_pass_rate"]["without_rm"])
line("constant + schema-filter", baseline["sql_pass_rate"]["with_rm"])
line("trained + schema-filter", trained["sql_pass_rate"]["with_rm"])

print("\nEnd-to-end QA accuracy:")
line("no rerank", trained["qa_accuracy"]["without_rm"])
line("constant + schema-filter", baseline["qa_accuracy"]["with_rm"])
line("trained + schema-filter", trained["qa_accuracy"]["with_rm"])

t_oracle = trained["oracle"]
b_oracle = baseline["oracle"]
if t_oracle["mixed"]["count"]:
    print("\nOracle mixed-bucket achieved (rows where selection actually matters):")
    line("no rerank", t_oracle["mixed_achieved_without_rm"])
    line("constant + schema-filter", b_oracle["mixed_achieved_with_rm"])
    line("trained + schema-filter", t_oracle["mixed_achieved_with_rm"])
    exp = t_oracle["mixed_expected_random"]
    rate = f" ({exp['rate']:.1%})" if exp["rate"] is not None else ""
    print(f"  {'expected by random pick':<32}: {exp['expected_correct']:.1f}/{exp['total']}{rate}")
PY
