# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`walt` builds training data for a text-to-SQL reward model. It ingests raw
question/SQL datasets, standardizes and downsamples them, then uses the
Anthropic API to correct the SQL and synthesize labeled negative ("sql_bad")
examples for RM training.

`src/walt/agent/` (LLM candidate generation → RM reranking → toy-SQLite execution) and
`src/walt/eval/` (agent-level evaluation on a held-out val split) are now implemented —
see Architecture below.

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

The SQL agent (`src/walt/agent/`) requires a local Ollama server with `llama3.2` pulled
(`ollama pull llama3.2`) — no API key, no network dependency at inference time.

`walt.rm.data.synth` (see Architecture) requires the official Spider release placed at
`$DATA_PATH/spider/` — download it from https://yale-lily.github.io/spider (the real
per-database SQLite corpus is only distributed via a Google Drive link there, not
automatable) and extract it so `$DATA_PATH/spider/database/<db_id>/<db_id>.sqlite` and
`$DATA_PATH/spider/{train_spider,train_others,dev}.json` exist. Gitignored, one-time
manual step — everything else in this repo's raw sources is either committed
(`spider_text_sql.csv`, `DBASQL.json`) or self-downloaded (`gretel/`).

## Commands

Run modules with `uv run python -m ...` (or activate `.venv` and drop the `uv run`).

Build the standardized/downsampled dataset from all registered sources — `--val-fraction`
(default `0.15`) stamps each row `split="val"` or `"trainval"`; `val` rows are held out
for agent-level evaluation and never seen by RM training/CV (see Architecture):
```bash
uv run python -m walt.rm.data.pre_process --target-count 5000 --val-fraction 0.15 --output data/output/rm_data.jsonl
```

Alternatively, build the gretel-only dataset (own pipeline, own output location — see
Architecture) by sampling directly from gretel's own train/test splits:
```bash
uv run python -m walt.rm.data.build_gretel_dataset --train-count 2000 --test-count 200
```

Alternatively, build the Spider-based synthetic dataset (own pipeline, deterministic
rule-based `sql_bad` — no LLM/`gen_training_data.py` step needed; see Architecture) —
`--shortlist-only` prints both pools' candidate-database tables (with table/FK/pair
counts) and exits without generating anything:
```bash
uv run python -m walt.rm.data.synth.build_synth_dataset --shortlist-only
uv run python -m walt.rm.data.synth.build_synth_dataset --train-count 2000 --val-count 300
```

Generate `sql_bad` negatives, correct `sql_good`, and synthesize an executable SQLite
`sql_context` via Claude (three modes) — both `test` and `collect` print an aggregate
`sql_good execution check: N/M passed (...%)` summary so data quality can be judged
before training on it. Rows that already carry a verified `sql_context` (e.g. from
`build_gretel_dataset.py`) automatically skip `sql_good`/`sql_context` generation and
only get `sql_bad` (see Architecture); rows with a `sql_context` already known invalid
are skipped entirely, printed as `Skipping N row(s) with a known-invalid sql_context`:
```bash
# iterate on the prompt cheaply, synchronous calls on a few rows
uv run python -m walt.rm.data.gen_training_data test --input data/output/rm_data.jsonl --limit 3

# submit the full file as an Anthropic Message Batch job
uv run python -m walt.rm.data.gen_training_data submit --input data/output/rm_data.jsonl

# poll a submitted batch and write the merged output JSONL
uv run python -m walt.rm.data.gen_training_data collect --batch-id msgbatch_xxx --output data/output/rm_enhanced.jsonl
```

Retroactively fix `sql_bad` candidates that don't actually differ from `sql_good`'s
result on their own `sql_context` (same test/submit/collect shape, but only rows with
at least one such candidate are sent to the LLM — everything else passes through
unchanged, and the full row set/order is always preserved in the output):
```bash
uv run python -m walt.rm.data.fix_sql_bad test --input data/output/gretel/gretel_enhanced.jsonl --limit 3
uv run python -m walt.rm.data.fix_sql_bad submit --input data/output/gretel/gretel_enhanced.jsonl
uv run python -m walt.rm.data.fix_sql_bad collect --batch-id msgbatch_xxx --output data/output/gretel/gretel_enhanced_fixed.jsonl
```

Train a pairwise-ranking reward model on `rm_enhanced.jsonl`, evaluate on a held-out
split, and check the train/test gap for overfitting (`--model` selects `lr_v1`/`lr_v2`/
`lr_v3`/`lr_v4`/`lr_v5`/`lr_v6`/`gbm`/`distilbert` — v4/v5 also consume `sql_context`,
see Architecture below for why neither beats v3, and `distilbert` fine-tunes
`distilbert-base-cased` end to end via `--distilbert-*` flags, also not beating v3 —
default `lr_v6` (v3 + a schema-validity feature) with `--C 30`, `--C` still tuned for
v3 not re-swept for v6; see Architecture below for why):
```bash
uv run python -m walt.rm.model.train \
  --input data/output/rm_enhanced.jsonl \
  --model-output data/output/rm_model.joblib \
  --metrics-output data/output/rm_metrics.json

# fine-tuned-transformer variant, run against the gretel-only dataset (see Architecture)
uv run python -m walt.rm.model.distilbert_preflight --input data/output/gretel/gretel_enhanced.jsonl  # stop-and-check first: token lengths + MPS sanity
uv run python -m walt.rm.model.train --input data/output/gretel/gretel_enhanced.jsonl --model distilbert --model-output data/output/rm_model_distilbert.pt
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

Run the SQL agent (LLM candidates → RM rerank → SQLite execution) on one row of a
`sql_context`-bearing JSONL, or an ad hoc question:
```bash
uv run python -m walt.agent.sql_agent --input data/output/rm_enhanced.jsonl --index 0
uv run python -m walt.agent.sql_agent --question "..." --schema-file schema.sql
```

Evaluate the agent end-to-end on the held-out `split="val"` rows (RM good-vs-bad
accuracy, SQL execution pass/fail, end-to-end QA accuracy — each of the latter two
reported both with RM reranking and without, i.e. just the first LLM candidate). Like
`train.py`, each run logs a JSON record (via the same `tracking.log_run`/`load_runs`,
just pointed at a separate `data/output/eval_runs/` — different metric shape than RM
training runs) so history can be compared later:
```bash
uv run python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl --rm-model data/output/rm_model.joblib
uv run python -m walt.eval.visualize  # table + comparison.png across logged eval runs
```
Both `sql_agent.py` and `evaluate.py` cache generated LLM candidates to
`data/output/llm_cache.json` by default (`--no-llm-cache` to disable), keyed by
`(model, question, schema_context)` — re-running `evaluate.py` against a different
`--rm-model` reuses the cache instead of re-calling Ollama, so RM iteration (and its
history log) is fast even though LLM generation isn't.

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
   if it has a bug, (b) synthesize 3-5 `sql_bad` variants, each tagged with a
   mistake category from `BAD_SQL_REASONS` (`missing_filters`,
   `wrong_aggregation`, `unsafe_patterns`, `misjoined_tables`), and (c)
   synthesize `sql_context` — SQLite `CREATE TABLE`/`INSERT INTO` statements
   `sql_good` can actually execute against, verified locally afterward (no
   extra API call, see "Data regeneration" below). The system prompt + few-shot
   examples are defined in this file — edit `FEW_SHOT_EXAMPLES` and
   `BAD_SQL_REASONS` together when tuning quality. Supports `test` (sync,
   cheap iteration), `submit` (async Message Batch, cheaper at scale), and
   `collect` (poll + merge) modes; batch state is cached locally in
   `src/walt/rm/data/.batch_state/<batch_id>.json` so `collect` can be
   re-run independently of `submit`.

Output convention: JSONL files under `data/output/`, one JSON object per
line (`rm_data.jsonl` = stage 2 output, `rm_enhanced.jsonl` = stage 3
output).

**gretel (`gretel.py`/`build_gretel_dataset.py`, 2026-08-15)** is a second, parallel
pipeline outside the `pre_process.py`/`SOURCES` flow above, for
`gretelai/synthetic_text_to_sql` — added because that dataset already ships its own
`sql_context` per row (no LLM synthesis needed) and its own train/test split (no
re-splitting needed). `GretelAdapter` downloads the HF-hosted parquet directly
(`pyarrow`, no `datasets` dependency needed), splits the dataset's single
semicolon-joined `sql_context` string into individual statements via `sqlglot` (falls
back to a naive `;`-split on a sqlglot parse error — a handful of rows use
non-SQLite-dialect syntax sqlglot's sqlite mode rejects), and verifies each locally via
`run_sql()` at extraction time rather than deferring to `gen_training_data.py`.
`build_gretel_dataset.py` samples exactly `--train-count`/`--test-count` rows from
gretel's own `train`/`test` splits (mapped to our `split="trainval"`/`"val"` — never
re-splitting their train data ourselves) and writes to the separate
`data/output/gretel/gretel_data.jsonl` (not merged into `rm_data.jsonl`). Raw parquet is
cached (gitignored, ~32MB) under `data/gretel/`.

`gen_training_data.py` now branches per-row on whether `sql_context` is already
populated: rows without one go through the original full flow (`emit_sql_review`) as
described above; rows that already have one (gretel) go through a second
`emit_sql_bad`-only tool call, told not to touch `sql_good`/`sql_context` and just
generate `sql_bad` against them (`has_context()`/`BAD_ONLY_*` in the module).
`is_llm_ready()` additionally skips, before either `test` or `submit` spends a call,
rows whose `sql_context_valid` was already computed `False` at extraction time — bad-only
mode can't fix those (~532/2200 for gretel, mostly non-SQLite-dialect source schemas).

**Reward model (`src/walt/rm/model/`)** scores/ranks SQL candidates for a question.
All `lr_model*.py` files below (the LR variants and `sql_features`-adjacent
`lr_model_context.py`) live under `rm/model/lr/`, not directly in `rm/model/` — moved
there (2026-08-15) once the flat directory grew to 7 LR variants; import via
`walt.rm.model.lr.lr_model_v3` or the re-exporting `walt.rm.model.lr` package `__init__`.
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

**Data regeneration: 4-category `sql_bad` taxonomy + held-out `val` split (2026-08-15).**
`gen_training_data.py`'s `BAD_SQL_REASONS` was replaced wholesale — the old 6 categories
(`wrong_columns_or_tables`, `wrong_join_or_aggregation`, `wrong_filter_or_sort`,
`type_or_null_handling`, `syntax_error`, `inefficient_query`) are gone, replaced by 4:
`missing_filters`, `wrong_aggregation`, `unsafe_patterns`, `misjoined_tables`. Rows also
now carry `sql_context` (LLM-synthesized SQLite `CREATE TABLE`/`INSERT INTO` statements
so `sql_good` is actually executable — empty when `sql_good` is itself DDL) and
`sql_context_valid` (verified locally via `walt/utils/sql_exec.py`'s `run_sql()`, no
extra API call). `pre_process.py` also now stamps every row with `split`
(`"trainval"`/`"val"`, via `--val-fraction`) — `val` is a held-out set for agent-level
evaluation (see below) that `train.py`/`cross_validate.py` filter out before
`group_split`/`k_fold_split` ever run, rather than changing how those existing
train/test-within-`trainval` mechanisms work (a single fixed train/test boundary was
already shown too noisy to trust — see CV discussion above — so only the val/trainval
boundary is persisted). Regenerating at the same scale as before (`--target-count 1000`,
same seed → same underlying question/SQL sample) via Anthropic Message Batch: 980/1000
rows succeeded (20 rejected — the model occasionally omits the now-required
`sql_context` field instead of returning an empty array; a minor, unfixed
prompt-robustness gap, ~2% of rows). Of the 980: 836 `trainval` / 144 `val`, and
`sql_context_valid` passed for 899/980 (91.7%) — the 4 reason categories are reasonably
balanced (830-1078 `sql_bad` instances each).

**Superseded baseline (spider/dbasql only): `lr_v3` with `C=30`, retrained on the
regenerated data above** — `top1_accuracy` 0.6587, `pairwise_accuracy` 0.8982, `mrr`
0.8179 on the standard 80/20 `trainval` split (5-fold CV: 0.6448 ± 0.0303 / 0.8955 ±
0.0097 / 0.8117 ± 0.0165), with a moderate overfitting gap (+0.078/+0.028/+0.045 — wider
than the ~+0.02-0.03 seen pre-regeneration, plausibly just a smaller effective
`trainval` pool: 836 rows vs the old 978). Not directly comparable to the
pre-regeneration numbers (`top1_accuracy` 0.546, `pairwise_accuracy` 0.864, `mrr` 0.738,
vs the original untuned `lr_v1` baseline's 0.424/0.811/0.658) — both the `sql_bad`
taxonomy and the row set changed, not just `C`. See below for the current baseline —
kept here only as a before/after reference point.

**Current best baseline: `lr_v3` with `C=1000`, spider/dbasql + gretel combined
(2026-08-15).** `data/output/rm_enhanced_with_gretel.jsonl` = the 980-row
`rm_enhanced.jsonl` above plus the 1666-row `gretel_enhanced.jsonl` (gretel data run
through `gen_training_data.py`'s bad-only mode — see gretel pipeline above), 2646 rows
total / 2642 usable (4 more hit the same `sql_good`-duplicates-`sql_bad` skip as
before) — 2355 `trainval` / 291 `val`. On this combined set, `C=30` (tuned on the old,
smaller `trainval` pool) noticeably *underperforms*: CV-sweeping `C` from 3 to 3000
(`cross_validate.py --model lr_v3 --C ...`) showed top1_accuracy still climbing well
past the old sweet spot — 0.542 (`C=3`) → 0.580 (`C=30`) → 0.602 (`C=300`) — before
plateauing at ~0.60 through `C=3000`, with `C=1000` landing on that plateau at the
tightest variance by far (top1 0.6006 ± 0.0056, vs ±0.015-0.024 for every other `C`
tried) — more data supporting much less regularization, as expected. Standard 80/20
split at `C=1000`: `top1_accuracy` 0.6128, `pairwise_accuracy` 0.8794, `mrr` 0.7877,
overfitting gap +0.016/+0.013/+0.014 (tighter than the superseded baseline's, despite
~3x the `trainval` rows — plausibly gretel's added diversity, not just row count).
**These numbers are lower than the superseded spider/dbasql-only baseline above**
(0.6587/0.8982/0.8179) despite the re-tuned `C` — confirmed by both CV and the single
split, so it isn't split noise. Likely cause: gretel spans far more diverse
domains/schemas per question than spider/dbasql, so `phi`'s embedding-similarity signal
has more surface area to get confused by; per-category breakdown on the combined set is
`unsafe_patterns` 0.95, `wrong_aggregation` 0.87, `missing_filters` 0.85,
`misjoined_tables` 0.84 (weakest, consistent with the superseded baseline). Read this as
a harder, more realistic baseline rather than a regression to fix — the smaller,
narrower old dataset was measuring an easier task.

**Making the RM consume `sql_context` (the schema `sql` executes against) — two things
tried, neither beats plain `lr_v3` (2026-08-15).** `BaseRewardModel.score()`/`rank()`
and `LRRewardModel._phi()`/`fit()` now thread an optional `sql_context: tuple[str, ...]`
through end to end (`evaluate()` passes `ex.sql_context`; `sql_agent.py` passes
`schema_context`) — plumbing any future context-aware variant needs, kept even though
neither variant below won:
- **`LRRewardModelV4`** (`lr_model_v4.py`): appends one scalar,
  `cosine_sim(embed(sql_context), embed(sql))`, to V3's phi (context embedded once per
  example, keyed by the joined statement text — see `lr_model_context.py`'s
  `ContextAwareLRRewardModel`, shared by V4/V5). The feature is real — a fitted
  coefficient of 1.15 at `C=1000`, not zeroed out — but CV-tied with plain `lr_v3` at
  every `C` tried (300/1000/3000; e.g. `C=1000` top1 0.5993 ± 0.0196 vs V3's
  0.6006 ± 0.0056), with consistently *higher* variance than V3 alone. No measurable
  benefit.
- **`LRRewardModelV5`** (`lr_model_v5.py`): concatenates the full `embed(sql_context)`
  vector (not just a scalar) alongside `embed(sql)`. This one isn't a "no signal"
  result — it's structurally inert. `LRRewardModel.fit()` trains on the *pairwise
  difference* `phi(q, sql_good) - phi(q, sql_bad)`; `sql_context` is identical across
  every candidate for a given question, so `embed(sql_context)` cancels to *exactly*
  zero in every training row regardless of what it encodes. Confirmed directly: the
  fitted coefficient block for those 768 dims is `0.0` to the last bit, and V5's
  predictions are byte-identical to V3's on the standard 80/20 split. Concatenating a
  per-question-constant feature can never contribute a gradient under this
  pairwise-difference objective — fixing it for real would need either a per-candidate
  interaction (e.g. `embed(sql_context) * embed(sql)` elementwise, generalizing V4's
  scalar to a full vector instead of a raw concat) or switching to pointwise training
  like `GBMRewardModel` (which already underperformed pairwise `lr_v1` here — see
  below). Left undone; this is documented as a dead end, not attempted further.

**Embedding model choice matters more than any single feature/hyperparameter change
tried so far.** CV-swept `lr_v3`/`C=30` against two general-purpose alternatives —
`BAAI/bge-base-en-v1.5` (top1 0.372) and `sentence-transformers/all-mpnet-base-v2`
(top1 0.377) — both landed well below `jina-embeddings-v2-base-code`'s 0.494, and
close to each other despite being different model families. This is decent evidence
that `jina-embeddings-v2-base-code`'s code-specific training (query/code retrieval
alignment) is doing real work here, not just a plausible-sounding default — a generic
strong text-embedding model isn't a substitute for one that's actually seen code.

**Scaling the `embed(sql)` block of phi doesn't help either — same null-result pattern
as the dot-product scaling test.** `LRRewardModelV3Scaled` (`lr_model_v3_scaled.py`,
`--scaling` on `cross_validate.py`) ablates how the ~768-dim `embed(sql)` portion of
phi is scaled before concatenation with `cosine_sim`/`is_sql_valid`, everything else
identical to the `lr_v3`/`C=30` baseline. `embed(sql)` is already L2-normalized by the
embedding provider's default (`normalize=True`), so `scaling="l2_normalize"` is a
no-op by construction and `scaling="l2_normalize_standardize"` collapses to plain
`scaling="standardize"` — both still implemented and CV-run rather than assumed, and
the numbers confirm the equivalence empirically (l2_normalize: top1 0.492±0.013 vs
baseline 0.494±0.014; l2_normalize_standardize: 0.473±0.016 vs standardize-alone:
0.470±0.014). The one real result: per-dimension standardizing (sklearn
`StandardScaler`, fit per CV fold on that fold's training data only) *costs* ~2.3pp
top1 (0.470 vs 0.494) — outside CV noise, not a fluke. Likely explanation: `C=30` was
CV-tuned against the raw (unnormalized-per-dimension) embedding scale, and
standardizing changes which dimensions the L2 penalty effectively favors; if a scaling
scheme were ever adopted, `C` would need re-tuning, but since nothing here beats the
unscaled baseline that's moot for now. Net: no scaling variant beats the current
unscaled baseline — keep `embed(sql)` as-is.

**L1 regularization (vs the default L2) is a promising but unconfirmed lead, not yet a
win.** `LRRewardModel` (and V2/V3/V3Scaled via inheritance) now take `penalty`/
`l1_ratio` (`SOLVER_BY_PENALTY = {l2: lbfgs, l1: liblinear, elasticnet: saga}` in
`lr_model.py`; sklearn requires a non-`lbfgs` solver for L1/elasticnet), exposed as
`--penalty`/`--l1-ratio` on `cross_validate.py`. Confirmed L1 does genuine feature
selection first (sklearn 1.8+ emits a `penalty`-is-being-deprecated-in-favor-of-
`l1_ratio` warning here, but still applies it correctly): a direct fit at `C=30` zeroed
395/770 coefficients under `penalty=l1` vs 0/770 under `l2`. CV-swept `lr_v3`/`C=30`
against the `l2` baseline at two seeds — L1 comes out numerically ahead on all 6
metric comparisons (top1/pairwise/mrr x 2 seeds), a consistent direction, but the
margin isn't clean: seed=42 shows top1 0.515±0.027 (l1) vs 0.494±0.014 (l2), a gap
inside roughly one *combined* std since L1's own variance is ~2x L2's there; seed=7
shows top1 0.509±0.046 (l1) vs 0.498±0.060 (l2), an even smaller gap with both
penalties noisier (matches the pre-existing seed-7-vs-42 fold-heterogeneity note
above). Net: not yet a confirmed win — a real, consistently-directional lead worth
revisiting, not a new baseline. If pursued further, `C` should be re-swept
specifically for `penalty=l1` (L1's optimal regularization strength isn't guaranteed
to be the same 30 tuned for L2) — not attempted here to keep this run isolated to
penalty type alone. The ~50% coefficient sparsity itself could also be a reason to
prefer L1 independent of the accuracy question (e.g. simpler/faster inference), if
that becomes a project goal.

`tracking.py` (`log_run`/`load_runs`) is the shared run-logging mechanism any
`BaseRewardModel` subclass's training script can reuse — not tied to `LRRewardModel`.
`visualize.py` reads everything under a runs directory and renders a comparison
table + line chart (top1_accuracy/pairwise_accuracy/mrr per run, chronological).
`base.py`'s `cross_validate()`/`k_fold_split()` are similarly algorithm-agnostic (take
a `model_factory` closure) — shares one pre-warmed embedding cache across all folds
(embedding a string doesn't depend on which fold it's in) rather than re-embedding
per fold, so k-fold CV costs about the same wall-clock time as a single split.

**Gretel-only scope (2026-08-15).** All work from this point on targets
`data/output/gretel/gretel_enhanced.jsonl` exclusively (1666 rows: 1519 `trainval` /
147 `val`, 4 more skipped for the usual `sql_good`-duplicates-`sql_bad` reason — 1662
usable), not the combined spider/dbasql+gretel set used above. There was no lr_v3
baseline trained purely on gretel data before this — every earlier "gretel" run was
actually the combined set — so one was established fresh: 5-fold CV sweeping `C` from
1 to 30000 found the same plateau pattern as the combined-set sweep but at a similar
`C`, peaking at **`C=1000`** (top1 0.5624 ± 0.0104, pairwise 0.8624 ± 0.0064, mrr
0.7571 ± 0.0072 — tightest variance of any `C` tried, consistent with the "more data
supports less regularization" pattern already seen). Matched single 80/20 split
(`seed=42`, same split later used for `DistilBertRewardModel` below): top1 0.5050,
pairwise 0.8479, mrr 0.7286, overfitting gap +0.174/+0.062/+0.100 (wider than the
combined-set gap — expected, gretel-only `trainval` is smaller: 1212 train rows).
Per-category on this split: `unsafe_patterns` 0.937 (strongest, as everywhere else in
this file), `misjoined_tables` 0.733 (weakest).

**`DistilBertRewardModel` (`distilbert_model.py`) — first fine-tuned-transformer RM,
does not beat `lr_v3` (2026-08-15).** Tries letting a model jointly attend over
`(sql_context, question, sql)` in one forward pass instead of embedding each piece
separately and combining via dot product, hypothesizing this would help most on
schema-reasoning mistakes (`misjoined_tables`/`missing_filters`). Architecture:
`distilbert-base-cased` (AutoModel, fully unfrozen) + a linear head on `[CLS]`,
manually encoding `[CLS] sql_context [SEP] question [SEP] sql [SEP]` (DistilBERT has
no `token_type_ids` — no segment embeddings — so segment structure comes purely from
`[SEP]` positions + `attention_mask`; DistilBertTokenizer's native API only supports
2-segment input, hence the manual construction). Trained genuinely pairwise, mirroring
`LRRewardModel.fit`'s exact anti-positional-bias RNG pattern: `score_A`/`score_B` are
two *independent* forward passes (each a single `(context, question, sql)` input), and
only `BCEWithLogitsLoss(score_A - score_B, label)` combines them — unlike
`GBMRewardModel`'s pointwise design (which exists because pairwise-differencing
*features* into one joint nonlinear classifier loses per-candidate score
decomposability), this stays a well-defined, always-transitive per-candidate scorer
despite being nonlinear, since the model never sees A and B jointly.

Two pre-flight checks (run before committing to a full fit(), see
`distilbert_preflight.py`) came back clean: token length using the real tokenizer and
format, 99.9% (8074/8079) of candidate sequences fit within 512 tokens untruncated
(median 162, p95 321, max 514) — truncation is a non-issue here, so the simple
truncate-`sql_context`-from-the-end fallback (no sqlglot-based schema filtering) was
used as-is; and an MPS sanity check (real model/tokenizer, tiny 8-pair batch, 15
forward+backward steps) showed no CPU-fallback ops, no errors, and a clean loss trend,
confirming native MPS support on this hardware with no
`PYTORCH_ENABLE_MPS_FALLBACK` needed.

Full run (defaults: `lr=2e-5`, `batch_size=8`, `grad_accum_steps=2` [effective 16],
`max_length=512`, early-stopping on validation `pairwise_accuracy` with
`patience=3`, question-level internal 90/10 split via the existing `group_split`):
stopped at epoch 7/15 (best weights from epoch 4 restored), ~30.3 min wall-clock on
an Apple Silicon Mac Mini via MPS (peak RSS 1.2GB; the MPS allocator separately
reported ~25.6GB, almost certainly inflated by caching-allocator pool growth across
~5700 variably-shaped micro-batches from dynamic padding rather than real working
set — RSS is the trustworthy figure here). On the same matched split as the gretel-
only `lr_v3` baseline above: top1 0.4983, pairwise 0.8387, mrr 0.7213 — **all three
slightly below `lr_v3`** (−0.007/−0.009/−0.007). Per-category: `unsafe_patterns` 0.967
(+3.0pp vs `lr_v3`), `missing_filters` 0.826 (+0.2pp, flat), `wrong_aggregation` 0.801
(−4.2pp), **`misjoined_tables` 0.698 (−3.5pp, the category the joint-attention
hypothesis predicted would improve most — instead it's the one that got worse)**.
Overfitting gap +0.322/+0.112/+0.184 — roughly 2x `lr_v3`'s on the identical split,
unsurprising for a fully-unfrozen 66M-param backbone fine-tuned on only 1091 internal
training examples (4724 pairs). Net: for this dataset size, `lr_v3` remains the
better reward model — far cheaper (38s vs ~30min) and no worse on any category that
matters. Not wired into `cross_validate.py` (no shared-cache path exists for it there
— each fold would fully re-fit from scratch — and a single held-out split matching
`lr_v3`'s shape was the goal here). The untried next lever, if this is revisited:
freezing most of the backbone and fine-tuning only the top layers + head, which would
cut both the overfitting gap and training cost — a different experiment from this one.

**SQL agent (`src/walt/agent/`)** wires an LLM candidate generator to the RM and a toy
SQLite executor: `agent/llm/base.py`'s `BaseLLM.generate_candidates(question,
schema_context, n) -> list[str]` is the swappable interface; `agent/llm/ollama_llm.py`'s
`OllamaLLM` is the only implementation so far, backed by a local Ollama server. Ollama's
HTTP API has no beam-search/multi-return (`num_beams`/`n`/`best_of`) parameter, so
`generate_candidates` makes `n` separate `chat()` calls with varied temperature/seed
rather than one decode pass — `BaseLLM`'s signature is deliberately backend-agnostic so
a future `transformers`-based backend with true `num_beams` beam search is a drop-in
swap for comparison later. `agent/sql_agent.py`'s `SqlAgent.run(question,
schema_context)` generates candidates, calls `rm.rank()` (the existing
`BaseRewardModel` method — no new RM code needed) to pick the top-scored one, executes
it via `walt/utils/sql_exec.py`'s `run_sql()` against a fresh in-memory SQLite DB seeded
from `schema_context`, and returns an `AgentResult` (`raw_candidates`,
`scored_candidates`, `best_sql`, `execution`, `final_answer`, and a `critique` field
that's always `None` — not implemented). `agent/llm/caching_llm.py`'s `CachingLLM`
wraps any `BaseLLM` with a disk cache (default `data/output/llm_cache.json`) keyed by
`(model, question, schema_context)` — deliberately *not* keyed by `n`, so a request for
fewer candidates than cached is served by slicing and only a request for *more*
triggers a real call — so re-running evaluation against a different RM (retrained,
different hyperparameters, even a different `BaseRewardModel` subclass) costs zero new
LLM calls. Enabled by default on both `sql_agent.py`'s CLI and `evaluate.py`
(`--no-llm-cache` to disable).

**Agent-level evaluation (`src/walt/eval/evaluate.py`)** runs the agent over the
held-out `split == "val"` rows (never seen by RM training/CV — see above) and reports
four things: (1) RM accuracy discriminating `sql_good` vs `sql_bad`, via the existing
`BaseRewardModel.evaluate()` — no new logic; (2) SQL execution pass/fail, and (3)
end-to-end QA accuracy (row-set match between the agent's executed result and
`run_sql(sql_context, sql_good)`'s reference result) — both (2) and (3) reported *with*
RM reranking (the agent's actual top pick) *and* without it (the first generated
candidate, i.e. what a single-shot call with no RM would produce), reusing the same
generated candidates for both so the comparison costs no extra LLM calls; and (4) an
**oracle ceiling** (added 2026-08-15, previously a one-off ad hoc analysis — now a
standard part of every run): of the `n_candidates` the LLM generated per question, how
many actually execute to the correct answer, bucketed per row into `all_correct` (any
pick wins — nothing to rerank), `zero_correct` (unreachable by *any* reranker — an LLM
generation-quality ceiling, not an RM problem), and `mixed` (0 < n_correct < n —
the only bucket where selection quality actually matters). `ceiling` =
`(all_correct + mixed) / n_qa_examples` is the best any reranker could ever score; within
`mixed`, the with-RM/without-RM achieved rates are compared against the random-chance
expectation (`Σ n_correct/n_candidates` over that bucket) so a low achieved number can
be told apart from "there was nothing to achieve." Reuses `rm_execution`/`base_execution`
already computed for (2)/(3) instead of re-running identical SQL, so this costs no extra
LLM calls and only a handful of extra local `run_sql()` calls (up to `n_candidates - 2`
per row). Each run logs a record to `data/output/eval_runs/` via `rm/model/tracking.py`'s
`log_run`/`load_runs` (reused as-is — already algorithm-agnostic, not tied to RM
training) with a flat `metrics` dict (`rm_top1_accuracy`, `sql_pass_rate_with_rm`/
`_without_rm`, `qa_accuracy_with_rm`/`_without_rm`, `oracle_ceiling` and friends) so
`eval/visualize.py` — the same table+chart pattern as `rm/model/visualize.py`, one color
per metric group and solid/dashed linestyle for with/without RM, plus a third dotted
line for `oracle_ceiling` (no with/without split — it's a property of the LLM's
candidates, not the reranker) — can track the RM-vs-no-RM gap *and* the ceiling it's
bounded by across runs over time.

**Finding: the RM does not transfer to `llama3.2`'s candidate distribution
(2026-08-15).** On the 144-row val set (123 executable rows, `n_candidates=5`),
RM-reranked and no-rerank-baseline are *tied* on end-to-end QA accuracy — 70/123
(56.9%) either way — and the RM's pick actually executes successfully slightly *less*
often than the naive first-candidate baseline (103/123 vs 107/123). This isn't the RM
quietly doing nothing: it disagrees with the baseline pick on 69/123 (56%) rows, but
among those disagreements it's an exact 15-15 split on which pick is actually correct
(39 ties) — when RM changes the answer, it's a coin flip, not an improvement. Likely
cause: the RM was trained to discriminate `sql_good` from Claude-*synthesized*
`sql_bad` negatives (four deliberate, clean mistake categories — see above), a
different error distribution than `llama3.2`'s actual generation mistakes at
`temperature=0.8`; the discriminative signal doesn't transfer out-of-distribution. Not
yet fixed — the natural next step is retraining with `llama3.2`'s own wrong candidates
folded in as `sql_bad`-style negatives, targeting the mismatch directly instead of
assuming synthetic negatives generalize.

**Oracle ceiling: 77.2% (95/123), vs 56.9% achieved — most of the gap isn't RM's to
close.** Among the 5 cached `llama3.2` candidates per val question: 41/123 (33%) rows
have *all 5* candidates correct (any pick wins, nothing to rerank), 28/123 (23%) have
*zero* correct candidates (unreachable by any reranker — a `llama3.2`
generation-quality ceiling, not an RM problem), leaving 54/123 (44%) as the "mixed"
bucket where selection quality actually matters. In that bucket, current achieved
(29/54) is barely above the ~26.8/54 a purely random pick would be expected to get by
chance (weighted by how many of the 5 are correct per row) — confirming the RM has
~no discriminative signal on `llama3.2`'s candidates specifically, consistent with the
disagreement finding above. Originally a one-off ad hoc analysis; re-run against the
now-integrated `evaluate.py` oracle-ceiling logic (see above) and reproduced exactly
(77.2%/41/28/54/29/54), confirming the two methodologies agree.

**Agent baseline on gretel's val split (2026-08-15): the transfer gap is worse here, not
just present.** Same setup (`n_candidates=5`, `llama3.2`), run against
`data/output/gretel/gretel_enhanced.jsonl`'s 147 executable val rows with the current
best RM (`lr_v3`/`C=1000` on the combined dataset — see baseline above). RM accuracy
(`sql_good` vs `sql_bad`) on this split: top1 0.544 — noticeably below the 0.613 the
same RM gets on the combined 80/20 split, consistent with gretel being the harder
subset throughout this file. At the agent level, RM reranking doesn't just fail to
help as on spider/dbasql (tied) — it actively *hurts* both metrics: SQL execution pass
131/147 (89.1%) with RM vs 135/147 (91.8%) without, QA accuracy 55/147 (37.4%) with RM
vs 59/147 (40.1%) without (a ~2.7pp regression each, not noise-sized given the n=147
sample). Separately, raw QA accuracy here (37-40%) is far below spider/dbasql's ~57%
regardless of reranking — `llama3.2` itself generates correct SQL far less often for
gretel's more domain-diverse questions, so this isn't purely an RM transfer problem;
generation quality is also lower on this harder distribution. Logged to
`data/output/eval_runs/20260815T235055Z_lr_v3_C1000_with_gretel_on_gretel_val.json`.

Oracle ceiling on this split confirms it's a generation-quality problem more than a
reranking problem: **51.0% (75/147)**, vs spider/dbasql's 77.2% — a much lower ceiling.
`zero_correct` is 49.0% (72/147, more than double spider/dbasql's 22.8%) — nearly half
the val set is unreachable by *any* reranker, `all_correct` is 24.5% (36/147), leaving
`mixed` at 26.5% (39/147). Within that mixed bucket the pattern from spider/dbasql
repeats and sharpens: achieved-without-RM (59.0%, 23/39) clearly beats both
achieved-with-RM (48.7%, 19/39) *and* the random-chance expectation (45.6%, 17.8/39) —
RM reranking is worse than doing nothing, and only barely better than chance, on
gretel's candidates specifically.

**`sql_bad` label quality: ~48% of candidates weren't distinguishing negatives, mostly
fixed by a 2nd-pass LLM step (2026-08-15).** An ad hoc audit of
`gretel_enhanced.jsonl` — executing every `sql_good`/`sql_bad` pair against its own
row's `sql_context` and diffing result sets — found 3,459/7,214 `sql_bad` candidates
(47.9%, across 85.4% of rows) actually returned the *same* result as `sql_good`, for
two distinct reasons: (1) sparse/homogeneous sample data (e.g. a `missing_filters`
candidate drops `WHERE quarter = 1`, but every seeded row already has `quarter = 1`, so
dropping it is a no-op on that data even though it's a real mistake in general), and (2)
genuinely non-distinguishing SQL — a cosmetic rewrite (renamed alias, reordered clauses)
that computes the same thing regardless of data. `fix_sql_bad.py` (new module,
`test`/`submit`/`collect` shape matching `gen_training_data.py`) detects every flagged
candidate locally via `run_sql` (no LLM needed for detection) and, per affected row,
asks Claude to either append `sql_context` sample rows (case 1) or replace the
candidate with a genuine different-result mistake in the same reason category (case 2),
locally re-verifying the result before accepting it — never touching `sql_good`, never
modifying/removing an existing `sql_context` statement. Run once against the full
1,666-row dataset (1,423 rows sent to the LLM; 47 batch results failed validation and
were left as their original, still-flagged content — no row is ever silently dropped,
output line count always matches input): row-level flagged rate fell from 85.4% to
39.3%, candidate-level from 47.9% to 18.1%.

Caveat that surfaced during verification: the result-set comparison this relies on is
blind to non-`SELECT` statements (`UPDATE`/`DELETE`/`INSERT`/`CREATE VIEW`, ~185/1,666
rows here) — `run_sql` returns `rows=None` for those (no `cursor.description`), so two
*different* UPDATE statements both look like a trivial "match" (`None == None`) even
though they'd have completely different effects on real data. Restricting to the
1,481 rows where `sql_good` is actually a `SELECT` (where the comparison is valid), the
real improvement is: row-level flagged rate 83.7% → 31.7%, candidate-level 43.7% →
10.5% — the non-`SELECT` rows are ~unchanged (184/185 before and after) because the fix
pass's own detection has the same blind spot, so the LLM correctly had nothing
meaningful to fix there. Properly verifying non-`SELECT` statements would need
comparing database state before/after execution rather than a result set — not
attempted here, out of scope for this pass.

**`gen_training_data.py` strengthened to reduce how often rows need the fix pass above
(2026-08-15).** Two changes: (1) `SYSTEM_PROMPT`/`BAD_ONLY_SYSTEM_PROMPT` now explicitly
require every WHERE/HAVING/JOIN/aggregation condition appearing anywhere in `sql_good`
or any `sql_bad` to actually matter for the sample data provided, and require mentally
confirming each `sql_bad` candidate would execute to a genuinely different result before
finalizing it — not just look different. (2) A new active mechanism,
`needs_enrichment()`/`enrich_context()`: after a row is generated, if `sql_good` is a
`SELECT`-shaped statement that executes successfully but returns zero rows, one
follow-up `emit_context_enrichment` tool call asks for more `INSERT INTO` rows (append
-only, existing tables only), locally re-verified before being accepted — wired into
both `cmd_test` (immediately, synchronously) and `cmd_collect` (a pass over the merged
batch results at the end); falls back to the original `sql_context` unchanged if the
addition breaks `sql_good` or still returns zero rows, no retry loop. `needs_enrichment`
deliberately checks `rows is not None and len(rows) == 0`, not just falsy `rows` — an
earlier version used `not execution.rows`, which also (incorrectly) matched
`rows=None` and fired on every DDL/`INSERT`/etc. row (~11% of gretel), asking the LLM to
"enrich" a schema that was never empty in the first place; caught via `cmd_test`
inspection before any real submit/collect run used it. Note this enrichment step runs
*after* `sql_bad` is already generated in the same call, so it doesn't guarantee
already-generated candidates stay meaningfully different post-enrichment (confirmed
directly: a `>` vs `>=` `missing_filters` candidate stayed a false negative because the
enrichment call, unaware of that candidate, added data that didn't happen to fall in the
boundary) — closing that fully would mean re-verifying `sql_bad` after enrichment too,
which `fix_sql_bad.py` above already does generically and can be re-run on any future
output if needed, so it wasn't duplicated here.

**`LRRewardModelV6` — a schema-validity feature closes most of the agent's RM-transfer
gap, now the default everywhere (2026-08-16).** The RM-doesn't-transfer finding above
traced to a distribution mismatch: `is_sql_valid` (pure syntax, `sql_features.py`)
essentially never fires on `llama3.2`'s actual mistakes, because its candidates almost
always parse fine — the real failure mode is referencing a table/column that doesn't
exist in the schema, a class of error pure-syntax checking can't see and the four
synthetic `sql_bad` mistake categories never model either.
`is_schema_valid(sql, sql_context)` (`sql_features.py`) targets this directly: executes
`sql` against `sql_context`'s schema via `run_sql`, cached on `(sql, sql_context)`.
`sql_context` here is expected to be the clean, data-free `CREATE TABLE`-only context
(`clean_context()`, see below), not the full context with sample rows — otherwise a
legitimate query could spuriously "fail" on a data-only constraint (e.g. a UNIQUE
conflict on inserted rows) that has nothing to do with whether the query itself is
well-formed for the schema. No context at all (e.g. `sql_good` is itself DDL) is
treated as valid/neutral. `LRRewardModelV6` (`lr_model_v6.py`) is `lr_v3`'s phi plus
this one feature — cheap, no embedding call, same pattern as `is_sql_valid` itself.

(Aside: `Example.sql_context_clean` — `sql_context` with `INSERT` statements stripped
via `clean_context()`, sqlite-`INSERT` statements identified via `sqlglot` — was added
earlier the same day so the LLM candidate generator and the RM both see only the
schema, not sample-data noise; `fit()`/`evaluate()` and `sql_agent.py`'s `run()` all
use it in place of the full `sql_context` for everything except actual execution, which
still needs the real data.)

On synthetic `sql_bad` pairwise accuracy (5-fold CV, `C=1000`, gretel-only) the feature
barely moves the needle — plain gretel top1 0.5731 vs `lr_v3`'s 0.5725, gretel_opus
(see below) 0.5918 vs 0.5829 — expected, since the synthetic negatives were written to
reference real schema elements, just with wrong logic. The win is entirely at the
agent level, where `llama3.2`'s actual candidates do hallucinate schema: on
gretel_opus's val split, SQL execution pass rate with RM reranking jumped from 76.6%
(`lr_v3`, *worse* than the 85.8% no-rerank baseline) to 96.5% (*better* than baseline);
QA accuracy's with/without-RM gap shrank from -5.6pp to -1.4pp; and the mixed-bucket
(selection-matters) achieved rate flipped from below random-chance expectation (37.3%
vs 45.1%) to above it (49.0%). Reproduced on plain gretel too (SQL pass 80.3%→97.3% vs
an 86.4% baseline, mixed-bucket 39.6%→50.0% vs 46.2% chance) — not a
gretel_opus-specific artifact. `sql_agent.py`'s `run_agent()`, `evaluate.py`'s
`--rm-class`, and `train.py`/`cross_validate.py`'s `--model` now all default to
`lr_v6`.

**`lr_v6` `C` re-swept on gretel-only data (2026-08-16): `C=300`, not the inherited
`lr_v3` defaults of `30`/`1000`.** Same sweep methodology as the original `lr_v3`
sweeps (5-fold CV, `C` from 1 to 30000), run separately on `gretel_opus` and plain
gretel. Both plateau over roughly the same `C=100`-`30000` range as `lr_v3` did, but
the peak sits lower this time: `gretel_opus` peaks at `C=300` (top1 0.5993±0.0266,
pairwise 0.8791±0.0077, mrr 0.7766±0.0141), plain gretel peaks at `C=100` (top1
0.5863±0.0322) with `C=300` a close second (0.5791±0.0335) — close enough between the
two datasets' peaks (within ~1pp, inside fold noise) that a single shared value beats
splitting by dataset, and `C=300` is the better shared pick since it's gretel_opus's
actual peak and only marginally off plain gretel's. Both `train.py` and
`cross_validate.py` now default to `--C 300`. This beats the old inherited default
(`C=30`, tuned for `lr_v3` on spider/dbasql) by a consistent +1.7-1.8pp top1 on both
datasets — a real gain, not noise. Not yet swept on the combined spider/dbasql+gretel
pool (`lr_v3`'s other tuned point, `C=1000`, was for that larger dataset) — if `lr_v6`
is ever trained on the combined set, `C` should be re-swept there too rather than
assuming `300` transfers.

QA accuracy with RM reranking still trails the no-rerank baseline by a small margin on
both datasets (39.7% vs 41.1% gretel_opus, 40.1% vs 41.5% plain gretel) —
`is_schema_valid` fixes "does the picked SQL even execute" almost completely but says
nothing about whether its logic is actually correct, so a schema-valid-but-wrong-answer
candidate can still outrank a correct one. Not addressed here.

**`gretel_opus` (2026-08-16)** is the same gretel pipeline above, run a second time with
`ANTHROPIC_MODEL=claude-opus-5` instead of the default `claude-sonnet-5`, output kept
separate under `data/output/gretel_opus/` (never overwrites the Sonnet-generated
`data/output/gretel/`) so the two are directly comparable. `gen_training_data.py`
bad-only mode against the same 1,668 llm-ready rows: 1,601 succeeded (67 failed tool
validation, same "occasionally omits a required field" gap seen with Sonnet, just a
different nested-shape failure mode). Pre-`fix_sql_bad.py`, Opus's negatives were
already more distinguishing than Sonnet's on the same data (68.3% rows / 26.5%
candidates flagged vs Sonnet's 80.8% / 42.7%); post-fix, 21.1% / 5.7% vs Sonnet's
27.0% / 8.7%. `lr_v3`/`C=1000` CV on `gretel_opus` beats plain gretel on all three
headline metrics (top1 0.5829±0.0219 vs 0.5725±0.0330, pairwise 0.8778±0.0082 vs
0.8625±0.0140, mrr 0.7703±0.0128 vs 0.7642±0.0205) — a real but modest gain, smaller
than `lr_v6`'s agent-level effect above.

**`walt.rm.data.synth` — Spider-based deterministic RM dataset, no LLM involved
(2026-08-16).** A third, independent pipeline (`src/walt/rm/data/synth/`, not registered
in `pre_process.py`'s `SOURCES`) built on the official Spider release (real per-database
SQLite files + gold SQL, manually downloaded — see Setup) rather than a flattened
question/SQL CSV or an LLM-synthesized schema. Two things distinguish it from every
other source in this file:

1. **`sql_bad` is generated by a rule-based sqlglot-AST corruption engine
   (`corrupt.py`), not an LLM.** Five categories — `missing_filters` (drop a WHERE/HAVING
   conjunct), `wrong_aggregation` (swap/drop an agg function), `unsafe_patterns`
   (`SELECT *` or an ON-join → `CROSS JOIN`), `misjoined_tables` (swap a join column/table
   for a wrong-but-plausible one, using a lightweight `schema.py` model parsed from real
   DDL), and `compound` (two of the above applied in sequence, or the same one twice when
   a query structurally supports only one — e.g. no JOIN/aggregation present). Reuses the
   exact 4 base reason strings from `gen_training_data.py`'s `BAD_SQL_REASONS` so this
   source's rows merge cleanly into any `pairwise_accuracy_by_reason` breakdown. Every
   candidate is verified against the real DB (never assumed): candidates that fail to
   execute are kept as strong negatives; candidates that execute but return the *same*
   result as `sql_good` are still kept (never silently dropped) but flagged in
   `synth_bad_candidate_qc.json` — 6.9% of 6,812 generated candidates on the first real
   run, a healthy rate (cf. the ~44-48% pre-`fix_sql_bad.py` rate on LLM-generated
   negatives elsewhere in this file).

2. **Rows don't embed `sql_context` — they carry `sql_context_path` instead.** `Example`
   (`rm/data/base.py`) gained a new optional field, `sql_context_path: str | None`
   (relative to `$DATA_PATH`, e.g. `"spider/database/restaurants/restaurants.sqlite"`),
   left empty for every other source. Traced every `sql_context` consumer before adding
   this: RM training/CV/`evaluate()` already only read `sql_context_clean` (DDL-only,
   still always populated, annotated with `-- one of: ...` categorical-value comments for
   low-cardinality TEXT columns via `annotate_ddl`) — untouched by this change, including
   `lr_v6`'s `is_schema_valid` feature. Only real SQL execution (the agent, and
   `eval/evaluate.py`'s baseline/reference/oracle-ceiling checks) needed updating:
   `sql_exec.py`'s new `execute_with_context(sql_context, sql_context_path, sql)` is a
   drop-in replacement for `run_sql(ex.sql_context, sql)` everywhere it was called, and
   `SqlAgent.run()` gained matching `sql_context_clean`/`sql_context_path` parameters
   (both default to the old derive-from-`schema_context` behavior, so every pre-existing
   caller is unaffected). When `sql_context_path` resolves a real `.sqlite` file (not the
   embedded-`sql_context` case), `execute_with_context` reuses one in-memory connection
   across every call sharing that path — built once via `sql_exec.load_context_from_sqlite`
   (reads `sqlite_master` + one `INSERT` per row directly, **not**
   `sqlite3.Connection.iterdump()`: Python 3.12+'s `iterdump()` runs `PRAGMA
   foreign_key_check` internally and raised on real FK/encoding issues found in Spider's
   own `yelp.sqlite`/an HR-domain DB's `last_name` column — data problems this module has
   no reason to fail on, since it only needs the CREATE TABLE/INSERT text) — and is
   invalidated (closed, evicted, rebuilt fresh next call) the instant a mutating statement
   (INSERT/UPDATE/DELETE/CREATE/DROP/ALTER, via a conservative sqlglot check) runs against
   it, so no candidate's side effect can leak into another's result. `eval/evaluate.py`
   additionally sorts its val rows by `sql_context_path` before the main loop so rows
   sharing a database run consecutively and the cache actually stays warm.

Dataset selection had to become multi-database: Spider's `train_spider.json`/
`train_others.json` (146 databases) and `dev.json` (20 databases) are **fully disjoint
database sets** (confirmed by direct computation, zero overlap) — Spider's own
cross-database-generalization design — so `trainval` rows are pooled from the first two
files' databases and `val` rows from `dev.json`'s, a stronger held-out guarantee than any
other source here (different schemas entirely, not just different rows). No single
3-6-table database reaches the target row counts (best single candidate: `restaurants`,
125 pairs) — `spider_source.shortlist_candidates` instead ranks all `min_tables<=N<=
max_tables` (default 3-8) databases by FK-density-then-pair-count, and
`select_dbs_for_target` greedily accumulates databases (verifying each against its real
data) until the target is reached, only processing as many as actually needed. One
empirically-discovered guard: `MAX_SQLITE_SIZE_BYTES` (5MB) excludes `soccer_1`
(317MB/196,823 rows), `wta_1` (105MB), and `baseball_1` (30MB) from candidacy — three
huge outliers among otherwise sub-4MB databases that turned a 2-second run into 7+
minutes (and, for `wta_1`, would have hit the val pool) for a handful of pairs each, with
no relevance to schema/join complexity. Default run (`--train-count 2000 --val-count
300`): 2,300 examples across 38 databases (33 `trainval`, 5 `val`, zero `db_id` overlap
between them) in ~2 seconds end-to-end, verified round-tripping through
`Example.from_dict`/`load_examples()` and a live `evaluate.py`/`sql_agent.py` run against
Ollama + a real `lr_v6` model.

**Synth baseline: `lr_v6`/`C=300` (2026-08-16) — RM discrimination saturates, but this is
the first dataset where reranking transfers cleanly to `llama3.2` (2026-08-16).** 5-fold
CV and the standard 80/20 split both land at **top1/pairwise/mrr ≈ 1.00**, every mistake
category included — confirmed not an `is_schema_valid`-specific artifact (`lr_v3` without
it still hits 99.6% top1 via CV). Read this as the deterministic corruptions being far
more surface-distinguishable than the subtle LLM-crafted negatives used everywhere else
in this file, not as the RM task being meaningfully "solved" — a `C` sweep was skipped as
pointless here (no headroom to tune against). The real signal is agent-level, on the
300-row val set (`llama3.2`, 5 candidates/question, held-out on entirely disjoint
databases from `trainval` — see above): **SQL execution pass 286/300 (95.3%) with RM vs
216/300 (72.0%) without; QA accuracy 166/300 (55.3%) with RM vs 139/300 (46.3%)
without.** Oracle ceiling 68.0% (204/300) — 16.0% all-correct, 32.0% zero-correct
(unreachable by any reranker), 52.0% mixed (selection matters); within that mixed bucket,
RM-achieved (75.6%) clearly beats both no-rerank (58.3%) and random-chance expectation
(51.8%). This is the **first source in this file where `lr_v6` reranking transfers
cleanly** to `llama3.2`'s actual candidates on both metrics — spider/dbasql was tied,
gretel/gretel_opus saw reranking actively hurt both metrics (see above). Plausible cause,
not confirmed: Spider's gold SQL and this module's corruptions both come from/target a
narrower, more template-like query distribution than gretel's, closer to what `llama3.2`
itself tends to generate, so the RM's learned signal (schema-validity + embedding
similarity) may simply generalize better here — not re-examined further. Artifacts:
`data/output/rm_model_synth_lr_v6_C300.joblib`,
`data/output/rm_metrics_synth_lr_v6_C300.json`, `data/output/eval_synth_lr_v6_C300.json`,
runs logged to `data/output/runs/`/`data/output/eval_runs/` as `synth_lr_v6_C300*`.
