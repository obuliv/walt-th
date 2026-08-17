# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**The full experiment log — every baseline, ablation, negative result, and the reasoning
behind the current defaults — lives in [`docs/experiments.md`](docs/experiments.md).**
This file keeps only structure, commands, and current state; read the log before
re-running an experiment or changing a tuned default.

## Project

`walt` builds training data for a text-to-SQL reward model (RM), then uses that RM to
rerank an LLM's SQL candidates. Three parts: `src/walt/rm/` (data pipelines + RM
training/eval), `src/walt/agent/` (LLM candidate generation → RM reranking →
SQLite execution), `src/walt/eval/` (agent-level evaluation on a held-out val split).

## Setup

Dependency management is via `uv` (see `uv.lock`):

```bash
uv sync
```

Requires a `.env` file (see `.env.example`) with:
- `DATA_PATH` — directory containing the raw source datasets (defaults to `./data`)
- `ANTHROPIC_API_KEY` — required for `gen_training_data.py`
- `ANTHROPIC_MODEL` — defaults to `claude-sonnet-5` if unset

Gotchas:
- RM training deps pin `transformers<5`: the default embedding model's
  `trust_remote_code=True` code imports `find_pruneable_heads_and_indices` from
  `transformers.pytorch_utils`, removed in v5. `uv add "transformers<5"` if this regresses.
- The agent needs a local Ollama server with `llama3.2` pulled (`ollama pull llama3.2`) —
  no API key, no network at inference time.
- `walt.rm.data.synth` needs the official Spider release at `$DATA_PATH/spider/`
  (download from https://yale-lily.github.io/spider — Google Drive only, not automatable)
  extracted so `$DATA_PATH/spider/database/<db_id>/<db_id>.sqlite` and
  `$DATA_PATH/spider/{train_spider,train_others,dev}.json` exist. Gitignored, one-time
  manual step; every other raw source is committed or self-downloaded.

There is no test suite or lint config (`tests/` is an empty package stub; no
pytest/ruff/mypy in `pyproject.toml`).

## Commands

Run modules with `uv run python -m ...` (or activate `.venv` and drop `uv run`).

### Build a dataset (four independent pipelines — see Architecture)

```bash
# 1. spider/dbasql via registered adapters; --val-fraction stamps split=val/trainval
uv run python -m walt.rm.data.pre_process --target-count 5000 --val-fraction 0.15 --output data/output/rm_data.jsonl

# 2. gretel-only (ships its own sql_context + train/test split)
uv run python -m walt.rm.data.build_gretel_dataset --train-count 2000 --test-count 200

# 3. Spider synth, deterministic rule-based sql_bad (no LLM step needed)
uv run python -m walt.rm.data.synth.build_synth_dataset --shortlist-only  # print candidate DBs, generate nothing
uv run python -m walt.rm.data.synth.build_synth_dataset --train-count 2000 --val-count 300

# 4. Spider synth, severity-scored: sql_bad from llama3.2's real mistakes + a Claude
#    reason/0-5-severity pass (`test` is a required gate before `submit`)
uv run python -m walt.rm.data.synth.build_severity_dataset --train-count 2000 --val-count 300
uv run python -m walt.rm.data.synth.enhance_severity_dataset test --input data/output/synth_severity/synth_severity_data.jsonl --limit 5
uv run python -m walt.rm.data.synth.enhance_severity_dataset submit --input data/output/synth_severity/synth_severity_data.jsonl
uv run python -m walt.rm.data.synth.enhance_severity_dataset collect --batch-id msgbatch_xxx --output data/output/synth_severity/synth_severity_enhanced.jsonl
```

### LLM data generation / repair (all `test` → `submit` → `collect`)

`test` = sync, cheap prompt iteration; `submit` = Anthropic Message Batch; `collect` =
poll + merge. Batch state is cached under `src/walt/rm/data/.batch_state/<batch_id>.json`
so `collect` is independent of `submit`. `test`/`collect` print a
`sql_good execution check: N/M passed` summary so data quality is visible before training.

```bash
# sql_bad + sql_good correction + synthesized sql_context (bad-only mode for rows that
# already carry a verified sql_context, e.g. gretel; known-invalid contexts are skipped)
uv run python -m walt.rm.data.gen_training_data test --input data/output/rm_data.jsonl --limit 3
uv run python -m walt.rm.data.gen_training_data submit --input data/output/rm_data.jsonl
uv run python -m walt.rm.data.gen_training_data collect --batch-id msgbatch_xxx --output data/output/rm_enhanced.jsonl

# repair sql_bad candidates whose result doesn't actually differ from sql_good's; only
# affected rows are sent to the LLM, row set/order is always preserved
uv run python -m walt.rm.data.fix_sql_bad test --input data/output/gretel/gretel_enhanced.jsonl --limit 3
uv run python -m walt.rm.data.fix_sql_bad submit --input data/output/gretel/gretel_enhanced.jsonl
uv run python -m walt.rm.data.fix_sql_bad collect --batch-id msgbatch_xxx --output data/output/gretel/gretel_enhanced_fixed.jsonl
```

### Train / cross-validate / compare

`--model` selects `lr_v1`..`lr_v6`/`gbm`/`distilbert`; defaults are `lr_v6` and `--C 300`
(CV-tuned for `lr_v6` on gretel-only data — re-sweep `C` for any new dataset or penalty).

```bash
uv run python -m walt.rm.model.train \
  --input data/output/rm_enhanced.jsonl \
  --model-output data/output/rm_model.joblib \
  --metrics-output data/output/rm_metrics.json

# distilbert: run the preflight first (token lengths + MPS sanity)
uv run python -m walt.rm.model.distilbert_preflight --input data/output/gretel/gretel_enhanced.jsonl
uv run python -m walt.rm.model.train --input data/output/gretel/gretel_enhanced.jsonl --model distilbert --model-output data/output/rm_model_distilbert.pt

# a single 80/20 split is too noisy to trust small deltas — cross-validate instead
uv run python -m walt.rm.model.cross_validate --model lr_v6 --C 300

uv run python -m walt.rm.model.visualize  # table + data/output/runs/comparison.png
```

Every `train.py`/`cross_validate.py` run logs JSON to `data/output/runs/`
(`--run-name`/`--runs-dir`, or `--no-log-run`): `config`, `metrics` (headline scores +
`pairwise_accuracy_by_reason`), `training` (fit diagnostics; per-fold CV detail under
`training.cv`), and `train_metrics`/`overfitting_gap`. Only `metrics` is charted.

### Run / evaluate the agent

```bash
uv run python -m walt.agent.sql_agent --input data/output/rm_enhanced.jsonl --index 0
uv run python -m walt.agent.sql_agent --question "..." --schema-file schema.sql

uv run python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl --rm-model data/output/rm_model.joblib
uv run python -m walt.eval.visualize  # table + comparison.png across data/output/eval_runs/
```

`evaluate.py` also takes `--rm-class` (incl. `constant` — a zero-signal scorer that
degrades to the LLM's own candidate order) and `--schema-filter` (hard `is_schema_valid`
pre-filter wrapping any model). Both `sql_agent.py` and `evaluate.py` cache LLM
candidates to `data/output/llm_cache.json` (`--no-llm-cache` to disable), keyed by
`(model, question, schema_context)` — **not** by `n` (fewer candidates than cached are
served by slicing) and **not** by prompt text, so editing `OllamaLLM.SYSTEM_PROMPT` does
not invalidate the cache (known gap). RM iteration therefore costs no new LLM calls.

## Architecture

Output convention: JSONL under `data/output/`, one JSON object per line. Each pipeline
owns its own output directory and never merges into another's.

### Data pipelines (`src/walt/rm/data/`)

Four independent pipelines produce the same `Example` shape:

1. **Adapters + `pre_process.py`** — a `BaseAdapter` subclass per source (`spider.py`,
   `dbasql.py`) parses its raw format and yields `Example(question, sql_good, source)`;
   to add a dataset, subclass `BaseAdapter`, implement `load()`, register it in
   `pre_process.py`'s `SOURCES`. `pre_process.py` proportionally downsamples to
   `--target-count` (never upsamples), shuffles with a fixed seed, stamps
   `split="trainval"/"val"`, writes `rm_data.jsonl`. Then `gen_training_data.py`
   (Claude, tool-forced `emit_sql_review`) corrects `sql_good`, synthesizes 3-5 `sql_bad`
   variants tagged with a `BAD_SQL_REASONS` category (`missing_filters`,
   `wrong_aggregation`, `unsafe_patterns`, `misjoined_tables`), and synthesizes an
   executable SQLite `sql_context`, locally verified (`sql_context_valid`). Edit
   `FEW_SHOT_EXAMPLES` and `BAD_SQL_REASONS` together when tuning quality. A
   `needs_enrichment()`/`enrich_context()` follow-up call adds `INSERT` rows when a
   `SELECT`-shaped `sql_good` executes but returns zero rows (append-only, re-verified;
   note it checks `rows is not None and len(rows) == 0`, not falsy `rows` — the latter
   also matches DDL rows). → `rm_enhanced.jsonl`.
2. **gretel** (`gretel.py`/`build_gretel_dataset.py`) — `gretelai/synthetic_text_to_sql`
   ships its own `sql_context` and train/test split, so no synthesis or re-splitting.
   `GretelAdapter` downloads the HF parquet directly (`pyarrow`), splits the joined
   `sql_context` via `sqlglot` (naive `;`-split fallback for non-SQLite-dialect rows),
   and verifies each statement at extraction time. `gen_training_data.py` branches on
   `has_context()` into a `emit_sql_bad`-only call for these rows. →
   `data/output/gretel/` (and `data/output/gretel_opus/` for the `ANTHROPIC_MODEL=claude-opus-5`
   rerun).
3. **`synth/`, deterministic** (`build_synth_dataset.py`) — real Spider SQLite DBs + gold
   SQL; `sql_bad` from a rule-based sqlglot-AST corruptor (`corrupt.py`: the 4 base
   reasons plus `compound`), every candidate verified against the real DB. Rows carry
   `sql_context_path` (relative to `$DATA_PATH`) instead of an embedded `sql_context`.
   `spider_source.py` shortlists/accumulates databases until the target row count is
   reached; `trainval` comes from `train_spider`/`train_others` DBs and `val` from
   `dev.json` DBs, which are **fully disjoint database sets** — a stronger held-out
   guarantee than any other source here. `MAX_SQLITE_SIZE_BYTES` (5MB) excludes three
   huge outlier DBs (`soccer_1`, `wta_1`, `baseball_1`).
4. **`synth/`, severity-scored** (`build_severity_dataset.py` + `enhance_severity_dataset.py`)
   — reuses `spider_source.py`'s DB/pair selection unchanged, but `sql_bad` comes from
   `llama3.2`'s own generation mistakes (verified via `corrupt.verify_candidate()`;
   same-result candidates are kept with a `matches_gold` hint, non-SELECT candidates are
   recorded un-executed so they can't corrupt the shared per-DB connection), then one
   Claude `emit_severity_review` call per row assigns a reason + 0-5 severity and
   backfills up to 3 new candidates for thin coverage. Reason taxonomy = the 5 synth
   categories plus `llama` as a catch-all — `llama` is valid only for categorizing an
   existing candidate, never for a Claude-proposed new one (enforced by a separate
   stricter JSON schema). `matches_gold` is a **hint, not a rule**: a nonzero severity is
   trusted over it, but a severity=0 claim contradicting real execution is clamped to 1.
   → `data/output/synth_severity/`.

Supporting: `filter_schema_valid.py` (drops schema-invalid `sql_bad` uniformly),
`add_llama_negatives.py` (folds `llama3.2`'s wrong candidates in as `reason="llama"`
negatives).

### Reward model (`src/walt/rm/model/`)

`BaseRewardModel` (`base.py`) is algorithm-agnostic: question-level `group_split`
(splits by `Example`, never by pair, so a question's candidates can't leak across the
split), `rank()`/`evaluate()` (top-1, pairwise accuracy, MRR) built on a subclass's
`score(question, sql, sql_context)`, and `cross_validate()`/`k_fold_split()` taking a
`model_factory` closure (one pre-warmed embedding cache shared across folds, so k-fold
costs about as much wall-clock as a single split).

`LRRewardModel` (`lr/lr_model.py`) scores `w · phi(question, sql)` where
`phi = concat(embed(sql), [cosine_sim(embed(question), embed(sql))])`, fit by training
sklearn `LogisticRegression` on `phi(q,A) - phi(q,B)` for candidate pairs (Bradley-Terry
style — a per-candidate scorer, so it generalizes to any-size candidate lists).
Embeddings come from a swappable `EmbeddingProvider` (`embeddings.py`); the default is
`jinaai/jina-embeddings-v2-base-code` (needs `trust_remote_code=True` + `einops`).
`penalty`/`l1_ratio` are supported via `SOLVER_BY_PENALTY`. Every SQL string is run
through `walt.utils.sql_exec.normalize_sql` (a sqlglot parse/print round-trip,
falls back to the original text on a parse error) immediately before it's
cached/embedded — in `score()`, `fit()`, and `warm_cache()`, so training pairs and
real inference-time candidates are normalized identically. This closes a formatting
leak: `sql_good` and `sql_bad` never otherwise shared a text-rendering convention
(hand-written/gretel-shipped gold vs. sqlglot-rendered corruptions or `llama3.2`'s own
generation style), so a model could partly learn "which style is this" instead of "is
this correct." `lr_v7` and `gbm_model.py` override `score()`/`fit()` directly and
replicate this normalization to stay consistent with the inherited `warm_cache()`;
every other LR variant inherits `score()`/`warm_cache()` unchanged.

Variants (all under `lr/`, re-exported by `walt.rm.model.lr`):

| model | phi | verdict |
|---|---|---|
| `lr_v1` | baseline above | superseded |
| `lr_v2` | + raw (unnormalized) dot product | no gain |
| `lr_v3` | + `is_sql_valid` (sqlglot syntax check) | strong on synthetic pairs |
| `lr_v4` | + `cosine_sim(embed(sql_context_clean), embed(sql))` | no gain |
| `lr_v5` | + full `embed(sql_context)` concat | **structurally inert** — cancels to exactly 0 under pairwise differencing |
| **`lr_v6`** | v3 + `is_schema_valid` | **default everywhere** |
| `gbm` | v3 phi, trained *pointwise* | loses decisively |
| `distilbert` | joint `(context, question, sql)` fine-tune, pairwise loss | never beats `lr_v6`, ~50x cost |
| `constant` | scores everything `0.0` | ablation baseline — degrades to LLM order |

`is_schema_valid(sql, sql_context)` (`sql_features.py`) executes `sql` against the
schema via `run_sql`, cached on `(sql, sql_context)`. It must be given the **clean,
data-free** context (`Example.sql_context_clean` — `INSERT`s stripped via
`clean_context()`), otherwise a data-only constraint violation would look like an
invalid query. No context at all is treated as valid/neutral. `schema_filter.py`'s
`SchemaFilteredRewardModel` applies the same check as a hard pre-filter around any model.

`SQLBadCandidate.severity` (`int | None`, 0-5, only ever set by
`enhance_severity_dataset.py`) drives an effective-rank pairing rule in
`LRRewardModel.fit()`: `sql_good` beats every bad; `severity is None` bads pair only
against `sql_good` (so all-`None` datasets produce bit-identical training pairs and RNG
consumption as before the field existed); `severity == 0` bads are excluded entirely;
`1..5` bads additionally pair against every other `1..5` bad of *different* severity.
`evaluate()` mirrors this (severity=0 excluded from ranking and pairwise accuracy).
`--severity-zero-as-positive` is a tested-and-rejected opt-in ablation — keep it off.
`--ignore-sql-good` is the opposite bet and currently looks like a win: it drops
`sql_good` from `positive_anchors` entirely and uses *only* `severity==0` bads as the
positive anchor (forcing `severity_zero_as_positive`'s effect on regardless of its own
setting), so training never sees Spider's literal gold text — only `llama3.2`'s own
correct-vs-incorrect distinction. Scores far worse on the RM's own pairwise test but
reranks real `llama3.2` candidates better than any trained model tried so far — see
`docs/experiments.md` and "Current state" below. `evaluate()` is unchanged by this
flag (still ranks against the real `sql_good`), so it isn't directly comparable
against `--severity-zero-as-positive`'s or the default's CV numbers as "the same test,
different training" — it's measuring transfer to a held-out ground truth the model
never trained on.

`tracking.py` (`log_run`/`load_runs`) is shared, algorithm-agnostic run logging, reused
unchanged by `eval/evaluate.py` against its own `data/output/eval_runs/`.

### Agent (`src/walt/agent/`) and evaluation (`src/walt/eval/`)

`BaseLLM.generate_candidates(question, schema_context, n)` is the swappable interface;
`OllamaLLM` is the only implementation. Ollama's HTTP API has no beam-search/multi-return
parameter, so it makes `n` separate `chat()` calls with varied temperature/seed —
`BaseLLM`'s signature is backend-agnostic so a `transformers` backend with real
`num_beams` is a drop-in swap. `max_concurrency` (`--ollama-concurrency`) fires those `n`
calls through a thread pool, always returning them in seed order; scoped to *within* one
row so `CachingLLM`'s non-concurrency-safe cache write never sees concurrent writers.
`CachingLLM` wraps any `BaseLLM` with the disk cache described under Commands.

`SqlAgent.run()` generates candidates, calls `rm.rank()`, executes the top pick via
`utils/sql_exec.py`, and returns an `AgentResult` (`critique` is always `None` — not
implemented). `sql_exec.execute_with_context(sql_context, sql_context_path, sql)` is the
single execution entry point: with an embedded context it builds a fresh in-memory DB;
with a `sql_context_path` it reuses one in-memory connection per path (built by
`load_context_from_sqlite`, which reads `sqlite_master` + rows directly rather than
`iterdump()`, whose internal `PRAGMA foreign_key_check` raises on real defects in
Spider's own DBs) and invalidates it the moment a mutating statement runs, so no
candidate's side effect leaks into another's result. `evaluate.py` sorts val rows by
`sql_context_path` to keep that cache warm.

`evaluate.py` runs the agent over held-out `split == "val"` rows and reports: RM
good-vs-bad accuracy; SQL execution pass rate and end-to-end QA accuracy (row-set match
against the reference result), each **with and without** RM reranking from the same
candidates; and an **oracle ceiling** bucketing each row into `all_correct` (nothing to
rerank), `zero_correct` (unreachable by any reranker — an LLM generation ceiling), and
`mixed` (the only bucket where selection matters, compared against random-chance
expectation).

## Conventions

- Long-running Ollama/Claude-loop scripts call `sys.stdout.reconfigure(line_buffering=True)`
  at the top of `main()` and print periodic progress with a rate/ETA — Python fully
  buffers stdout when it isn't a TTY, which made a 40+ minute backgrounded run look dead.
  Do this for any new script of that shape.
- Raising `OllamaLLM.max_concurrency` alone changes nothing: the `llama-server` Ollama
  spawns defaults to `-np 1`. Fix system-side with `launchctl setenv OLLAMA_NUM_PARALLEL 4`
  (a shell `export` doesn't reach Ollama.app), then quit and reopen Ollama.app; verify via
  `ps aux | grep llama-server`. Restarting kills in-flight requests, so do it between runs.
- Never silently drop a row: failed LLM validation falls back to the original content and
  output line count always matches input.
- Result-set comparison (used by `fix_sql_bad.py` and the QC checks) is blind to
  non-`SELECT` statements — `run_sql` returns `rows=None`, so two different `UPDATE`s look
  identical. Restrict any such analysis to `SELECT` rows or state the caveat.

## Current state (details in [`docs/experiments.md`](docs/experiments.md))

- **Default config**: `lr_v6` (`lr_v3` phi + `is_schema_valid`) at `C=0.1`, on the
  severity-scored synth dataset. `C` dropped from 300 once every SQL string started
  getting sqlglot-normalized before embedding (see above) — that closed a formatting
  leak, and the old high-`C` (low-regularization) default started overfitting once it
  was gone. `C` is tuned per dataset/config — re-sweep it, don't inherit.
- **Best RM numbers** (honest, post-normalization): severity-scored synth, `lr_v6`/`C=0.1`
  — top1 0.805 / pairwise 0.9322 / mrr 0.8888 on the standard 80/20 split. (Was
  0.9225/0.9693/0.9548 at the old `C=300` before normalization — that drop is the leak
  closing, not a regression; agent-level performance held up, see below.) gretel is
  still much harder (~0.55-0.60 top1); the deterministic-corruptor synth set still
  saturates, now at ~0.99 instead of ~1.00.
- **`is_schema_valid` is still most of the agent-level story, but no longer all of
  it.** On the severity-scored synth val set: no reranking 72.0% SQL pass / 46.3% QA →
  `lr_v3` (no schema signal) 70.3% / 45.0% (*worse than not reranking*) → `lr_v6`
  95.3% / 53.3% → `constant` + `--schema-filter` (zero training, just reject
  schema-invalid candidates) 95.3% / 56.0% → **`lr_v6` trained with
  `--ignore-sql-good`** (drops `sql_good` as the training anchor entirely, trains only
  on `llama3.2`'s own correct-vs-incorrect distinction) **+ `--schema-filter`**
  stacked on top: 95.3% / **58.3%, the best of every configuration tested, on both
  metrics simultaneously** — `--ignore-sql-good` alone gets the QA win but narrowly
  loses SQL pass (94.7%) to the filter baseline; stacking the hard filter on top closes
  that gap for free (same 58.3% QA, now tied at 95.3% SQL pass too).
- **The RM does not transfer to `llama3.2`'s candidate distribution without that hard
  schema check** — reranking was tied on spider/dbasql and actively harmful on
  gretel/gretel_opus. Schema awareness only pays off as a hard, execution-verified fact:
  soft embedding proximity to the schema (`lr_v4`) is a null-to-negative result, because a
  hallucinated column still sits close to the real schema in embedding space.
- **Open gap, mostly closed**: schema filtering used to have ~no power over semantic
  mistakes and no trained model beat it on QA accuracy — `--ignore-sql-good` +
  `--schema-filter` (above) now strictly dominates it. Not yet validated beyond the
  ollama-only dataset (full severity-scored dataset, gretel, spider/dbasql).
