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

Train a pairwise-ranking reward model on `rm_enhanced.jsonl`, evaluate on a held-out
split, and check the train/test gap for overfitting (`--model` selects `lr_v1`/`lr_v2`/
`lr_v3`/`gbm`, default `lr_v3` with `--C 30`; see Architecture below for why):
```bash
uv run python -m walt.rm.model.train \
  --input data/output/rm_enhanced.jsonl \
  --model-output data/output/rm_model.joblib \
  --metrics-output data/output/rm_metrics.json
```

A single 80/20 split is noisy (`--seed 7` vs `--seed 42` alone swung top1_accuracy by
~1.5pp on `lr_v1`) — before trusting a delta between configs, cross-validate instead:
```bash
uv run python -m walt.rm.model.cross_validate --model lr_v3 --C 30
```
`cross_validate.py` shares `train.py`'s model/hyperparameter flags, runs k-fold (default
5) question-level CV, prints mean±std per metric, and logs a run record the same way
(`metrics` = per-fold means, full per-fold detail under `training.cv`).

Every `train.py` run also logs a JSON record to `data/output/runs/` (override with
`--run-name`/`--runs-dir`, or skip with `--no-log-run`), so different
approaches/hyperparameters can be compared later. Each record has `config` (input,
model choice, embedding model, split params, row-skip counts), `metrics` (the headline
scores plus a `pairwise_accuracy_by_reason` breakdown by mistake category), `training`
(fit-time diagnostics: LR convergence, embedding/fit/eval timing, feature dim, label
balance), and `train_metrics`/`overfitting_gap` (evaluate() re-run on the training set,
and the train-minus-test gap on the headline metrics) — everything except `metrics` is
recorded for future debugging but deliberately not part of what `visualize.py` charts.
Compare accumulated runs (prints a table, writes a chart to
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

`LRRewardModelV2` (`lr_model_v2.py`) subclasses `LRRewardModel`, overriding only
`_embed_unique`/`_phi`: V1's `embed()` always L2-normalizes, so its "cosine
similarity" feature already *is* a dot product (of unit vectors) — V2 fetches raw
(unnormalized) embeddings instead and adds the raw dot product as a second,
magnitude-sensitive interaction feature alongside the (now locally-computed) cosine
similarity (770-dim phi vs V1's 769-dim), with `use_cosine_sim`/`use_dot_product`/
`standardize_dot_product` constructor flags to ablate each piece. On the real dataset
none of these combinations beat V1 (confirmed via CV, not just a single split) — the
raw dot product just isn't informative here, standardized or not.

**Establishing a baseline (CV-driven)**: a naive single 80/20 split is too noisy to
trust small deltas between configs — `cross_validate.py`'s k-fold CV surfaced two real
improvements a single split had been masking:
- **Regularization**: sklearn's default `C=1.0` was substantially under-fitting for
  this data size (~4700 training pairs, 769 features). Sweeping `C` via CV
  (`cross_validate.py --model lr_v1 --C ...`) showed top1_accuracy climbing
  monotonically from 0.246 (`C=0.01`) to ~0.47 (`C=10`-`C=30`, tight ±0.005-0.013 std)
  before diminishing returns and rising variance set in past `C=100` (early overfitting
  signal). `C=30` is the sweet spot — `LRRewardModel`/`V2`/`V3` all take `C` as a
  constructor arg (persisted via `save()`/`load()`), and `train.py`/`cross_validate.py`
  both default to `--C 30`.
- **`LRRewardModelV3`** (`lr_model_v3.py`) subclasses `LRRewardModel`, appending one
  handcrafted feature to V1's phi: `is_sql_valid(sql)` (`sql_features.py`, via
  `sqlglot` — pure syntax parsing, no schema/database needed). Targeted directly at the
  `syntax_error` mistake category, which was consistently the weakest or
  second-weakest in every embedding-only variant's `pairwise_accuracy_by_reason`
  breakdown (~0.70-0.73). Under CV at `C=30`, V3 beats V1 (top1 0.494 vs 0.469); on the
  full train.py run this shows up concretely as `syntax_error` accuracy jumping from
  0.73 to 0.89. Note: sqlglot's default dialect treats a lot of unmodeled DDL/DCL
  (MySQL's `MODIFY COLUMN`, `GRANT`/`REVOKE`, `RENAME TABLE`, ...) as a lenient
  `Command` fallback instead of raising — `is_sql_valid` treats that fallback as
  "invalid too" (catches real mistakes like `GRANT ... FROM` that should be `TO`, at
  the cost of ~0.7% false positives on genuinely-valid `sql_good` DDL); see the
  docstring in `sql_features.py` for the recall/false-positive numbers this trades off.
- **`GBMRewardModel`** (`gbm_model.py`) tries a nonlinear model class instead of more
  features: `sklearn.ensemble.HistGradientBoostingClassifier`, trained *pointwise*
  (label = is this candidate `sql_good`, features = `phi(question, sql)` directly, no
  pairwise differencing) rather than pairwise, because a nonlinear model trained on
  feature *differences* loses the linear-scorer's clean decompose-into-a-per-candidate-
  score property (see the module docstring for why that matters for ranking an
  any-size candidate list). It's a strict feature superset of V3 (same phi) — but loses
  decisively under CV regardless of hyperparameters tried (default and a shallower/
  fewer-rounds config both land well below plain `lr_v1`, e.g. top1 ~0.36-0.37 vs
  0.469). Expected, not a bug: gradient-boosted trees split on individual features one
  at a time, which doesn't suit dense embedding dimensions where the signal is a
  weighted combination across all ~768 of them — exactly what a linear model is good at
  and trees are bad at.

**Current best baseline: `lr_v3` with `C=30`** — `top1_accuracy` 0.546, `pairwise_accuracy`
0.864, `mrr` 0.738 on the standard 80/20 split (vs the original untuned `lr_v1`
baseline's 0.424/0.811/0.658), with a modest, expected overfitting gap (~+0.02 to +0.03,
up from ~0 at `C=1.0` — less regularization trades a little generalization gap for a
much better fit, net positive here).

**Embedding model choice matters more than any single feature/hyperparameter change
tried so far.** CV-swept `lr_v3`/`C=30` against two general-purpose alternatives —
`BAAI/bge-base-en-v1.5` (top1 0.372) and `sentence-transformers/all-mpnet-base-v2`
(top1 0.377) — both landed well below `jina-embeddings-v2-base-code`'s 0.494, and
close to each other despite being different model families. This is decent evidence
that `jina-embeddings-v2-base-code`'s code-specific training (query/code retrieval
alignment) is doing real work here, not just a plausible-sounding default — a generic
strong text-embedding model isn't a substitute for one that's actually seen code.

`tracking.py` (`log_run`/`load_runs`) is the shared run-logging mechanism any
`BaseRewardModel` subclass's training script can reuse — not tied to `LRRewardModel`.
`visualize.py` reads everything under a runs directory and renders a comparison
table + line chart (top1_accuracy/pairwise_accuracy/mrr per run, chronological).
`base.py`'s `cross_validate()`/`k_fold_split()` are similarly algorithm-agnostic (take
a `model_factory` closure) — shares one pre-warmed embedding cache across all folds
(embedding a string doesn't depend on which fold it's in) rather than re-embedding
per fold, so k-fold CV costs about the same wall-clock time as a single split.
