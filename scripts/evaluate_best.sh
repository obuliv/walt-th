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
# Usage:
#   scripts/evaluate_best.sh --output-dir experiments/run1   # same dir passed to build_best_rm.sh
set -euo pipefail

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
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

echo "Running constant+schema-filter baseline (log: $BASELINE_LOG)..."
uv run python -m walt.eval.evaluate \
  --input "$INPUT_PATH" \
  --rm-class constant \
  --schema-filter \
  --output "$BASELINE_OUTPUT" \
  --run-name best_eval_baseline_constant_schemafilter \
  > "$BASELINE_LOG" 2>&1

echo "Running trained model + schema-filter (log: $TRAINED_LOG)..."
uv run python -m walt.eval.evaluate \
  --input "$INPUT_PATH" \
  --rm-model "$MODEL_PATH" \
  --rm-class lr_v6 \
  --schema-filter \
  --output "$TRAINED_OUTPUT" \
  --run-name best_eval \
  > "$TRAINED_LOG" 2>&1

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
