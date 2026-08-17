# walt

Builds training data for a text-to-SQL reward model (RM), trains the RM, and uses it
to rerank an LLM's SQL candidates at inference time.

For architecture, all commands, and the full experiment log (what was tried, what
won, why), see [`CLAUDE.md`](CLAUDE.md) and [`docs/experiments.md`](docs/experiments.md).
This file only covers setup and the fastest path to seeing results.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY
```

`.env` needs:
- `DATA_PATH` — directory containing the raw source datasets (defaults to `./data`)
- `ANTHROPIC_API_KEY` — required for any Claude data-generation/enhancement step
- `ANTHROPIC_MODEL` — defaults to `claude-sonnet-5` if unset

Also required to run the agent/eval (not just train the RM):
- A local Ollama server with `llama3.2` pulled (`ollama pull llama3.2`) — the agent
  generates SQL candidates locally, no API key or network needed at inference time.
- The official Spider release at `$DATA_PATH/spider/` (download from
  https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset), extracted
  so `$DATA_PATH/spider/database/<db_id>/<db_id>.sqlite` and
  `$DATA_PATH/spider/{train_spider,train_others,dev}.json` exist. One-time manual step.
## Latest Benchmark
```commandline
$ ./scripts/evaluate_best.sh --output-dir runs/demo
Running constant+schema-filter baseline (log: runs/demo/eval_baseline.log)...
Running trained model + schema-filter (log: runs/demo/eval_trained.log)...
Sample log (50 rows): runs/demo/eval_samples.md

== Funnel: no rerank -> constant+schema-filter -> trained model+schema-filter ==

SQL execution pass rate:
  no rerank                       : 216/300 (72.0%)
  constant + schema-filter        : 286/300 (95.3%)
  trained + schema-filter         : 286/300 (95.3%)

End-to-end QA accuracy:
  no rerank                       : 139/300 (46.3%)
  constant + schema-filter        : 168/300 (56.0%)
  trained + schema-filter         : 177/300 (59.0%)

Oracle mixed-bucket achieved (rows where selection actually matters):
  no rerank                       : 91/156 (58.3%)
  constant + schema-filter        : 120/156 (76.9%)
  trained + schema-filter         : 129/156 (82.7%)
  expected by random pick         : 80.8/156 (51.8%)
```
## Fastest path: run eval on the checked-in demo model

[`runs/demo/`](runs/demo/) has a pre-trained RM (best-known config — see its README)
checked in, so you can see agent-level results without training anything first:

```bash
scripts/evaluate_best.sh --output-dir runs/demo
```

This runs the agent with and without RM reranking (plus a filter-only baseline) over
the held-out val split and prints a funnel: no-rerank vs. constant+schema-filter vs.
the trained model+schema-filter, for SQL execution pass rate and end-to-end QA
accuracy. Full per-run output is captured under `runs/demo/eval_*.log`.

## Training the RM from scratch

```bash
scripts/build_best_rm.sh --spider-dir path/to/spider --output-dir experiments/run1
scripts/evaluate_best.sh --output-dir experiments/run1
```

`build_best_rm.sh` runs the full best-known pipeline: extract Spider DBs + generate
`sql_bad` from llama3.2's real mistakes, a Claude pass to assign reason/severity to
each candidate (pauses for confirmation before submitting — it's a real, billed
Anthropic Message Batch), filter to llama3.2-only candidates, sweep `C` for
`lr_v6 --ignore-sql-good --drop-bad-vs-bad-pairs`, then train the final model. Takes
a while (Ollama generation + a Claude batch + several CV folds); see the script's
header comment for the full breakdown.
