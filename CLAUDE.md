# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`walt` builds training data for a text-to-SQL reward model. It ingests raw
question/SQL datasets, standardizes and downsamples them, then uses the
Anthropic API to correct the SQL and synthesize labeled negative ("sql_bad")
examples for RM training.

`src/walt/agent/` and `src/walt/eval/` are currently empty placeholder
packages — no implementation yet.

## Setup

Dependency management is via `uv` (see `uv.lock`).

```bash
uv sync
```

Requires a `.env` file (see `.env.example`) with:
- `DATA_PATH` — directory containing the raw source datasets (defaults to `./data`)
- `ANTHROPIC_API_KEY` — required for `gen_training_data.py`
- `ANTHROPIC_MODEL` — defaults to `claude-sonnet-5` if unset

Adding the RM training dependencies (numpy, scikit-learn, sentence-transformers) pins
`transformers<5`: the default embedding model's `trust_remote_code=True` custom modeling
code imports `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`, which
transformers v5 removed. `uv add "transformers<5"` if this regresses.

## Commands

Run modules with `uv run python -m ...` (or activate `.venv` and drop the `uv run`).

Build the standardized/downsampled dataset from all registered sources:
```bash
uv run python -m walt.rm.data.pre_process --target-count 5000 --output data/output/rm_data.jsonl
```

Generate `sql_bad` negatives and correct `sql_good` via Claude (three modes):
```bash
# iterate on the prompt cheaply, synchronous calls on a few rows
uv run python -m walt.rm.data.gen_training_data test --input data/output/rm_data.jsonl --limit 3

# submit the full file as an Anthropic Message Batch job
uv run python -m walt.rm.data.gen_training_data submit --input data/output/rm_data.jsonl

# poll a submitted batch and write the merged output JSONL
uv run python -m walt.rm.data.gen_training_data collect --batch-id msgbatch_xxx --output data/output/rm_enhanced.jsonl
```

Train the pairwise-ranking reward model on `rm_enhanced.jsonl` and evaluate on a
held-out split:
```bash
uv run python -m walt.rm.model.train \
  --input data/output/rm_enhanced.jsonl \
  --model-output data/output/rm_model.joblib \
  --metrics-output data/output/rm_metrics.json
```

Every `train.py` run also logs a JSON record to `data/output/runs/` (override with
`--run-name`/`--runs-dir`, or skip with `--no-log-run`), so different
approaches/hyperparameters can be compared later. Each record has `config` (input,
embedding model, split params, row-skip counts), `metrics` (the headline scores plus a
`pairwise_accuracy_by_reason` breakdown by mistake category), and `training` (fit-time
diagnostics: LR convergence, embedding/fit/eval timing, feature dim, label balance) —
`training`/`by_reason` are recorded for future debugging but deliberately not part of
what `visualize.py` charts. Compare accumulated runs (prints a table, writes a chart to
`data/output/runs/comparison.png`):
```bash
uv run python -m walt.rm.model.visualize
```

There is no test suite or lint config configured yet (`tests/` is an empty
package stub, no pytest/ruff/mypy in `pyproject.toml`).

## Architecture

**Data pipeline (`src/walt/rm/data/`)** runs in two sequential stages:

1. **Adapters → standardized examples** (`base.py`, `spider.py`, `dbasql.py`):
   Each source has a `BaseAdapter` subclass that parses its own raw file
   format (Spider's CSV, DBASQL's JSON) and yields `Example(question,
   sql_good, source)` records. To add a new dataset, subclass `BaseAdapter`,
   implement `load()`, and register it in `pre_process.py`'s `SOURCES` dict.

2. **`pre_process.py`** (module name is `walt.rm.data.pre_process`, entry
   point historically called "launcher" — check the module docstring, it
   lags a prior rename) loads every registered source, proportionally
   downsamples each to hit `--target-count` total (never upsamples), shuffles
   with a fixed seed, and writes standardized JSONL.

3. **`gen_training_data.py`** takes that JSONL and, per row, calls Claude
   with a tool-forced schema (`emit_sql_review`) to: (a) correct `sql_good`
   if it has a bug, and (b) synthesize 3-5 `sql_bad` variants, each tagged
   with a mistake category from `BAD_SQL_REASONS` (wrong columns/tables,
   wrong join/aggregation, wrong filter/sort, type/null handling, syntax
   error, inefficient query). The system prompt + few-shot examples are
   defined in this file — edit `FEW_SHOT_EXAMPLES` and `BAD_SQL_REASONS`
   together when tuning quality. Supports `test` (sync, cheap iteration),
   `submit` (async Message Batch, cheaper at scale), and `collect` (poll +
   merge) modes; batch state is cached locally in
   `src/walt/rm/data/.batch_state/<batch_id>.json` so `collect` can be
   re-run independently of `submit`.

Output convention: JSONL files under `data/output/`, one JSON object per
line (`rm_data.jsonl` = stage 2 output, `rm_enhanced.jsonl` = stage 3
output).

**Reward model (`src/walt/rm/model/`)** scores/ranks SQL candidates for a question.
`BaseRewardModel` (`base.py`) is algorithm-agnostic: question-level train/test split
(`group_split` — splits by `Example`, never by pair, to avoid leaking a question's
other candidates across the split), `rank()`/`evaluate()` (top-1 accuracy, pairwise
accuracy, MRR) built generically on a subclass's `score(question, sql)`, and an
unused-so-far `predict_error_code()` hook for future subclasses. `LRRewardModel`
(`lr_model.py`) is the first implementation: scores via `w · phi(question, sql)` where
`phi = concat(embed(sql), [cosine_sim(embed(question), embed(sql))])`, fit by training
sklearn `LogisticRegression` on `phi(q,A) - phi(q,B)` differences for
(`sql_good`, `sql_bad`) pairs (Bradley-Terry style — a per-candidate scorer, not a
pairwise classifier, so it generalizes to ranking any-size candidate lists). Embeddings
come from a swappable `EmbeddingProvider` (`embeddings.py`); the default,
`SentenceTransformerEmbedding`, uses `jinaai/jina-embeddings-v2-base-code` and requires
`trust_remote_code=True` + the `einops` package. `Example` (`walt/rm/data/base.py`) now
carries an optional `sql_bad: tuple[SQLBadCandidate, ...]` field for this — defaulted to
`()` so the data-adapter pipeline is unaffected. `load_examples()` skips and warns on
rows where `sql_good` duplicates a `sql_bad` entry (~9/987 rows in the current
`rm_enhanced.jsonl`, a labeling artifact from the LLM data-generation step) rather than
aborting the load.

`tracking.py` (`log_run`/`load_runs`) is the shared run-logging mechanism any
`BaseRewardModel` subclass's training script can reuse — not tied to `LRRewardModel`.
`visualize.py` reads everything under a runs directory and renders a comparison
table + line chart (top1_accuracy/pairwise_accuracy/mrr per run, chronological).
