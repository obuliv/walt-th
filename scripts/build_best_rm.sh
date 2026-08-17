#!/usr/bin/env bash
# Best-known route for training the RM (see docs/experiments.md, "ignore_sql_good"/
# ollama-only section, and CLAUDE.md "Current state"):
#
#   1. build_severity_dataset.py  -- extract Spider DBs, generate sql_bad from
#      llama3.2's real mistakes via a local Ollama server.
#   2. enhance_severity_dataset.py test -> submit -> collect  -- Claude assigns
#      reason + 0-5 severity to every candidate. (Pauses for confirmation before
#      submitting -- this is a real, billed Anthropic Message Batch.)
#   3. filter_ollama_only.py  -- keep only llama3.2-origin sql_bad.
#   4. cross_validate.py --model lr_v6 --ignore-sql-good --drop-bad-vs-bad-pairs,
#      swept over C.
#   5. train.py with the best C from the sweep.
#
# Requires: a local Ollama server with llama3.2 pulled, and ANTHROPIC_API_KEY set.
#
# Usage:
#   scripts/build_best_rm.sh --spider-dir path/to/spider --output-dir experiments/run1
set -euo pipefail

SPIDER_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spider-dir) SPIDER_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --spider-dir path/to/spider --output-dir experiments/run1" >&2
  echo "(--spider-dir defaults to \$DATA_PATH/spider if omitted, same as build_severity_dataset.py)" >&2
  exit 1
fi

TRAIN_COUNT="${TRAIN_COUNT:-2000}"
VAL_COUNT="${VAL_COUNT:-300}"
SEED="${SEED:-42}"
C_SWEEP="0.01 0.1 1 10 30 100 300"

RUNS_DIR="$OUTPUT_DIR/runs"
STAGE1="$OUTPUT_DIR/synth_severity_data.jsonl.gz"
ENHANCED="$OUTPUT_DIR/synth_severity_enhanced.jsonl.gz"
OLLAMAONLY="$OUTPUT_DIR/synth_severity_enhanced_ollamaonly.jsonl.gz"
MODEL_OUTPUT="$OUTPUT_DIR/rm_model.joblib"
METRICS_OUTPUT="$OUTPUT_DIR/rm_metrics.json"

mkdir -p "$OUTPUT_DIR" "$RUNS_DIR"

# ---- 1. extract from Spider + generate sql_bad via Ollama ------------------
echo "== [1/5] build_severity_dataset.py (Spider extraction + llama3.2 sql_bad) =="
uv run python -m walt.rm.data.synth.build_severity_dataset \
  ${SPIDER_DIR:+--spider-dir "$SPIDER_DIR"} \
  --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --seed "$SEED" \
  --output "$STAGE1"

# ---- 2. Claude severity pass (test -> submit -> collect) -------------------
echo "== [2/5] enhance_severity_dataset.py test (required gate before submit) =="
uv run python -m walt.rm.data.synth.enhance_severity_dataset test --input "$STAGE1" --limit 5

echo
read -r -p "Review the test output above. Submit the full (billed) batch to Claude? [y/N] " reply
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
  echo "Stopped. Re-run this script once you're ready to submit." >&2
  exit 0
fi

submit_out="$(uv run python -m walt.rm.data.synth.enhance_severity_dataset submit --input "$STAGE1")"
echo "$submit_out"
batch_id="$(grep -oE 'msgbatch_[A-Za-z0-9_]+' <<<"$submit_out" | head -1)"
[[ -z "$batch_id" ]] && { echo "Could not parse a batch id from submit output." >&2; exit 1; }

echo "-- polling batch $batch_id until it finishes --"
uv run python -m walt.rm.data.synth.enhance_severity_dataset collect --batch-id "$batch_id" --output "$ENHANCED"

# ---- 3. keep only llama3.2-origin sql_bad -----------------------------------
echo "== [3/5] filter_ollama_only.py =="
uv run python -m walt.rm.data.filter_ollama_only --stage1 "$STAGE1" --input "$ENHANCED" --output "$OLLAMAONLY"

# ---- 4. sweep C for lr_v6 --ignore-sql-good --drop-bad-vs-bad-pairs --------
echo "== [4/5] cross_validate.py C-sweep (lr_v6, --ignore-sql-good --drop-bad-vs-bad-pairs) over: $C_SWEEP =="
for c in $C_SWEEP; do
  echo "-- C=$c --"
  uv run python -m walt.rm.model.cross_validate \
    --model lr_v6 --ignore-sql-good --drop-bad-vs-bad-pairs --C "$c" \
    --input "$OLLAMAONLY" --run-name "sweep_C${c}" --runs-dir "$RUNS_DIR"
done

best_c="$(python3 - "$RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path

best = None
for p in sorted(Path(sys.argv[1]).glob("*sweep_C*.json")):
    rec = json.loads(p.read_text())
    pairwise = rec["metrics"].get("pairwise_accuracy")
    c = rec["config"].get("C")
    if pairwise is not None and (best is None or pairwise > best[1]):
        best = (c, pairwise)
print(best[0])
print(f"Best C={best[0]} (mean CV pairwise_accuracy={best[1]:.4f})", file=sys.stderr)
PY
)"
echo "Selected $best_c"

# ---- 5. final train with the swept C ----------------------------------------
echo "== [5/5] train.py (lr_v6, --ignore-sql-good --drop-bad-vs-bad-pairs, C=$best_c) =="
uv run python -m walt.rm.model.train \
  --model lr_v6 --ignore-sql-good --drop-bad-vs-bad-pairs --C "$best_c" \
  --input "$OLLAMAONLY" \
  --model-output "$MODEL_OUTPUT" \
  --metrics-output "$METRICS_OUTPUT" \
  --runs-dir "$RUNS_DIR" \
  --run-name best_train

echo
echo "Trained model: $MODEL_OUTPUT"
echo "Now run: scripts/evaluate_best.sh --output-dir $OUTPUT_DIR"
