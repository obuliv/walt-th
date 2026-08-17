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
`lr_v6 --ignore-sql-good --drop-bad-vs-bad-pairs`, then train the final model. Takes a while (Ollama
generation + a Claude batch + several CV folds); see the script's header comment for
the full breakdown.
